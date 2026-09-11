"""Building blocks for two-dimensional ConvNeXt VAEs."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.base import HybridVAEBase


class DropPath(nn.Module):
    """Stochastic Depth (per sample)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * random_tensor


def _best_gn_groups(default_groups: int, channels: int) -> int:
    """Choose a GroupNorm group count that divides channels."""
    g = min(default_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


class ConvNeXtBlock2D(nn.Module):
    """ConvNeXt-style block for 2D images (channels-first).

    Design:
      depthwise conv (k=7 in classic ConvNeXt, here k=7) -> GroupNorm
      -> pointwise expand -> GELU -> pointwise project
      + residual

    Notes:
      - GroupNorm used to avoid channel-last permutations.
      - Capacity can be scaled via mlp_ratio.
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        gn_groups: int = 8,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        skip_dropout_p: float = 0.0,
        skip_alpha: float = 1.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )

        groups = _best_gn_groups(gn_groups, channels)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x


def _upsample_block2d(in_ch: int, out_ch: int, scale: int, use_transpose_conv: bool, gn_groups: int) -> nn.Sequential:
    if use_transpose_conv:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
            nn.GELU(),
        )

    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
        nn.GELU(),
    )


class ConvNeXtUNetEncoder2D(nn.Module):
    """ConvNeXt2D encoder with U-Net skip outputs.

    forward(x) returns:
      - h: deepest latent feature map (B, z_channels, h', w')
      - skips: list of feature maps at each resolution (for decoder), length n_levels

    Channel schedule:
      base=8 -> 16 -> 32 -> 64 -> ...
    """

    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        gn_groups: int = 8,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        skip_dropout_p: float = 0.0,
        skip_alpha: float = 1.0,
    ):
        super().__init__()
        self.n_levels = n_levels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, 8), num_channels=8, eps=1e-6),
            nn.GELU(),
        )

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.downs: nn.ModuleList = nn.ModuleList()

        total_blocks = n_levels * n_res_blocks
        if drop_path_rate and drop_path_rate > 0 and total_blocks > 0:
            dp_rates = torch.linspace(0.0, float(drop_path_rate), steps=total_blocks).tolist()
        else:
            dp_rates = [0.0] * total_blocks
        dp_i = 0

        for i in range(n_levels):
            ch = 2 ** (i + 3)       # 8, 16, 32, 64...
            ch_next = 2 ** (i + 4)  # 16, 32, 64, 128...

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock2D(ch, mlp_ratio=4.0, gn_groups=gn_groups, drop_path=dp_rates[dp_i], dropout=dropout))
                dp_i += 1
            self.blocks.append(nn.Sequential(*stage))

            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch_next, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch_next), num_channels=ch_next, eps=1e-6),
                    nn.GELU(),
                )
            )

        bottom_ch = 2 ** (n_levels + 3)
        self.to_z = nn.Conv2d(bottom_ch, z_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []

        for i in range(self.n_levels):
            x = self.blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        h = self.to_z(x)
        return h, skips


class ConvNeXtUNetDecoder2D(nn.Module):
    """ConvNeXt2D decoder with U-Net skips.

    For API compatibility, forward(z) only takes `z`.
    Skips must be set via set_skips(skips) before forward.
    """

    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_transpose_conv: bool = True,
        gn_groups: int = 8,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        skip_dropout_p: float = 0.0,
        skip_dropout_ps: Optional[Iterable[float]] = None,
        skip_alpha: float = 1.0,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_transpose_conv = use_transpose_conv
        self._skips: Optional[List[torch.Tensor]] = None
        self.skip_dropout_p = float(skip_dropout_p)
        self.skip_dropout_ps = HybridVAEBase._normalize_skip_dropout_ps(skip_dropout_ps, n_levels, self.skip_dropout_p)
        self.skip_alpha = float(skip_alpha)

        self.bottom_ch = 2 ** (n_levels + 3)
        self.from_z = nn.Sequential(
            nn.Conv2d(z_channels, self.bottom_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, self.bottom_ch), num_channels=self.bottom_ch, eps=1e-6),
            nn.GELU(),
        )

        self.ups: nn.ModuleList = nn.ModuleList()
        self.fuse: nn.ModuleList = nn.ModuleList()
        self.blocks: nn.ModuleList = nn.ModuleList()

        total_blocks = n_levels * n_res_blocks
        if drop_path_rate and drop_path_rate > 0 and total_blocks > 0:
            dp_rates = torch.linspace(0.0, float(drop_path_rate), steps=total_blocks).tolist()
        else:
            dp_rates = [0.0] * total_blocks
        dp_i = 0


        prev_ch = self.bottom_ch
        for i in range(n_levels):
            # mirror channel schedule of encoder
            ch = 2 ** (n_levels - i + 2)

            self.ups.append(_upsample_block2d(prev_ch, ch, scale=2, use_transpose_conv=use_transpose_conv, gn_groups=gn_groups))

            # skip channels at matching resolution
            skip_ch = 2 ** (n_levels - i + 2)
            self.fuse.append(
                nn.Sequential(
                    nn.Conv2d(ch + skip_ch, ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch), num_channels=ch, eps=1e-6),
                    nn.GELU(),
                )
            )

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock2D(ch, mlp_ratio=4.0, gn_groups=gn_groups, drop_path=dp_rates[dp_i], dropout=dropout))
                dp_i += 1
            self.blocks.append(nn.Sequential(*stage))

            prev_ch = ch

        self.out = nn.Conv2d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: Optional[List[torch.Tensor]]) -> None:
        self._skips = skips

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent feature map `z` into an image.

        If skips are not provided (self._skips is None), skip tensors are treated as zeros.
        During training, optional skip-dropout can be applied per sample to force latent usage.
        """
        x = self.from_z(z)

        skips = self._skips
        if skips is not None and len(skips) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} skips, got {len(skips)}")

        for i in range(self.n_levels):
            x = self.ups[i](x)

            if skips is None:
                # No skips provided -> treat as zeros (forces latent usage)
                skip_ch = 2 ** (self.n_levels - i + 2)
                skip = torch.zeros(
                    (x.shape[0], skip_ch, x.shape[-2], x.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                skip = skips[-1 - i]

            # Align spatial sizes (off-by-1 for odd inputs)
            if x.shape[-2:] != skip.shape[-2:]:
                target = (min(x.shape[-2], skip.shape[-2]), min(x.shape[-1], skip.shape[-1]))
                x = HybridVAEBase._crop_like(x, target)
                skip = HybridVAEBase._crop_like(skip, target)

            # Apply skip scaling (can be used to weaken or disable skips)
            if self.skip_alpha != 1.0:
                skip = skip * self.skip_alpha

            # Skip-Dropout (drop entire skip tensor per sample during training)
            p = self.skip_dropout_ps[-1 - i]
            if p > 0.0 and self.training:
                keep_prob = 1.0 - p
                mask = (torch.rand((skip.shape[0], 1, 1, 1), device=skip.device, dtype=skip.dtype) < keep_prob).to(skip.dtype)
                skip = skip * mask / max(keep_prob, 1e-6)

            x = torch.cat([x, skip], dim=1)
            x = self.fuse[i](x)
            x = self.blocks[i](x)

        return self.out(x)


