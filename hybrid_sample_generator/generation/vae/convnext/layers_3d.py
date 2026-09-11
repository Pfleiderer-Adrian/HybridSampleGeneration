"""Building blocks for three-dimensional ConvNeXt VAEs."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.base import HybridVAEBase


class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt-style block for 3D volumes (channels-first).

    Design (simplified):
      depthwise conv (k=7) -> GroupNorm -> pointwise expand -> GELU -> pointwise project
      + residual

    Notes:
      - GroupNorm is used instead of LayerNorm to avoid channel-last permutations.
      - You can scale capacity via mlp_ratio.
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        gn_groups: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Depthwise conv
        # NOTE: In 2D ConvNeXt commonly uses k=7. In 3D, k=7 means 7x7x7=343
        # kernel elements and can be prohibitively slow/heavy. For volumetric VAEs
        # (especially in CPU-bound inference/debug), a smaller kernel is a much
        # better default.
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # GroupNorm (stable for small batch sizes)
        groups = min(gn_groups, channels)
        # Ensure divisibility
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

        # Optional stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x


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


class ConvNeXtUNetEncoder3D(nn.Module):
    """ConvNeXt3D encoder with U-Net skip outputs.

    forward(x) returns:
      - h: deepest latent feature map (B, z_channels, d', h', w')
      - skips: list of feature maps at each resolution (for decoder), length n_levels

      base=8, then 16, 32, 64, ...
    """

    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.n_levels = n_levels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=min(gn_groups, 8), num_channels=8, eps=1e-6),
            nn.GELU(),
        )

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.downs: nn.ModuleList = nn.ModuleList()

        for i in range(n_levels):
            ch = 2 ** (i + 3)      # 8,16,32,64...
            ch_next = 2 ** (i + 4)  # 16,32,64,128...

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock3D(ch, mlp_ratio=4.0, gn_groups=gn_groups))
            self.blocks.append(nn.Sequential(*stage))

            # Downsample
            self.downs.append(
                nn.Sequential(
                    nn.Conv3d(ch, ch_next, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch_next), num_channels=ch_next, eps=1e-6),
                    nn.GELU(),
                )
            )

        # Bottom projection to z_channels
        bottom_ch = 2 ** (n_levels + 3)
        self.to_z = nn.Conv3d(bottom_ch, z_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []

        for i in range(self.n_levels):
            x = self.blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        h = self.to_z(x)
        return h, skips


class ConvNeXtUNetDecoder3D(nn.Module):
    """ConvNeXt3D decoder with U-Net skips.

    For API compatibility with the original decoder, forward(z) only takes `z`.
    Skips are provided via set_skips(skips) before calling forward. If skips
    are set to None, the decoder uses zero-skips for pure prior sampling.
    """

    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_transpose_conv: bool = True,
        skip_dropout_p: float = 0.0,
        skip_dropout_ps: Optional[Iterable[float]] = None,
        skip_alpha: float = 1.0,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_transpose_conv = use_transpose_conv
        self.skip_dropout_p = float(skip_dropout_p)
        self.skip_dropout_ps = HybridVAEBase._normalize_skip_dropout_ps(skip_dropout_ps, n_levels, self.skip_dropout_p)
        self.skip_alpha = float(skip_alpha)
        self._skips: Optional[List[torch.Tensor]] = None

        # Project latent channels up to bottom channels
        self.bottom_ch = 2 ** (n_levels + 3)
        self.from_z = nn.Sequential(
            nn.Conv3d(z_channels, self.bottom_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, self.bottom_ch), num_channels=self.bottom_ch, eps=1e-6),
            nn.GELU(),
        )

        self.ups: nn.ModuleList = nn.ModuleList()
        self.fuse: nn.ModuleList = nn.ModuleList()
        self.blocks: nn.ModuleList = nn.ModuleList()

        prev_ch = self.bottom_ch
        for i in range(n_levels):
            # At decoder step i we go to channel count of encoder level (reversed).
            # This matches the original schedule:
            #   n_filters = 2**(n_levels - i + 2)
            ch = 2 ** (n_levels - i + 2)

            self.ups.append(_upsample_block3d(prev_ch, ch, scale=2, use_transpose_conv=use_transpose_conv, gn_groups=gn_groups))

            # Skip concat: (ch + skip_ch) -> ch
            # skip_ch corresponds to encoder level (n_levels-1-i) channels: 2**((n_levels-1-i)+3) = 2**(n_levels-i+2)
            skip_ch = 2 ** (n_levels - i + 2)
            self.fuse.append(
                nn.Sequential(
                    nn.Conv3d(ch + skip_ch, ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch), num_channels=ch, eps=1e-6),
                    nn.GELU(),
                )
            )

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock3D(ch, mlp_ratio=4.0, gn_groups=gn_groups))
            self.blocks.append(nn.Sequential(*stage))

            prev_ch = ch

        self.out = nn.Conv3d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: Optional[List[torch.Tensor]]) -> None:
        self._skips = skips

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        x = self.from_z(z)

        # use reversed skips: deepest skip is last; None means zero-skips
        skips = self._skips
        if skips is not None and len(skips) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} skips, got {len(skips)}")

        for i in range(self.n_levels):
            x = self.ups[i](x)

            if skips is None:
                skip_ch = 2 ** (self.n_levels - i + 2)
                skip = torch.zeros(
                    (x.shape[0], skip_ch, x.shape[-3], x.shape[-2], x.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                skip = skips[-1 - i]

            # Skip dropout (training only): forces decoder to use latent z instead of bypassing via skips
            p = self.skip_dropout_ps[-1 - i]
            if p > 0.0 and self.training:
                keep_prob = 1.0 - p
                mask = (torch.rand((skip.shape[0], 1, 1, 1, 1), device=skip.device, dtype=skip.dtype) < keep_prob).to(skip.dtype)
                skip = skip * mask / max(keep_prob, 1e-6)
            if x.shape[-3:] != skip.shape[-3:]:
                # Center-crop the larger one to the smaller
                target = (
                    min(x.shape[-3], skip.shape[-3]),
                    min(x.shape[-2], skip.shape[-2]),
                    min(x.shape[-1], skip.shape[-1]),
                )
                x = HybridVAEBase._crop_like(x, target)
                skip = HybridVAEBase._crop_like(skip, target)

            # Skip gating: downscale skip strength to reduce bypass and force latent usage
            skip = skip * self.skip_alpha

            x = torch.cat([x, skip], dim=1)
            x = self.fuse[i](x)
            x = self.blocks[i](x)

        return self.out(x)


def _best_gn_groups(default_groups: int, channels: int) -> int:
    """Choose a GroupNorm group count that divides channels."""
    g = min(default_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


def _upsample_block3d(in_ch: int, out_ch: int, scale: int, use_transpose_conv: bool, gn_groups: int) -> nn.Sequential:
    if use_transpose_conv:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
            nn.GELU(),
        )

    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=False),
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
        nn.GELU(),
    )


