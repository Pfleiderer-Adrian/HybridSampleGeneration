"""SPADE layers for the two-dimensional conditional ConvNeXt VAE."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_sample_generator.generation.vae.base import HybridVAEBase

from hybrid_sample_generator.generation.vae.convnext.layers_2d import (
    ConvNeXtBlock2D,
    DropPath,
    _best_gn_groups,
    _upsample_block2d,
)


class SPADE2D(nn.Module):
    """Spatially-Adaptive Normalization (SPADE) for 2D data."""
    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int=128, kernel_size: int=3):
        super().__init__()
        # layer norm -> norm over all channels; affine=False: no params here as params should only come from SPADE
        self.no_param_instance_norm = nn.GroupNorm(num_groups=1, num_channels=norm_nc, affine=False)

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=kernel_size, padding=pw),
            nn.GELU()
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=pw)
    
    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.no_param_instance_norm(x)

        # scale mask to x's resolution, use 'nearest' if mask is one-hot encoded
        if tgt_mask.shape[-2:] != x.shape[-2:]:
            tgt_mask = F.interpolate(tgt_mask, size=x.shape[-2:], mode='nearest')

        activation = self.mlp_shared(tgt_mask)
        gamma = self.mlp_gamma(activation)
        beta = self.mlp_beta(activation)

        return normalized * (1 + gamma) + beta


class ConvNeXtSPADEBlock2D(nn.Module):
    """ConvNeXt-style block for 2D images (channels-first).
    SPADE instead of GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        num_anomaly_classes: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
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

        self.spade = SPADE2D(norm_nc=channels, label_nc=num_anomaly_classes)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.spade(x, mask) # use mask here
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x


class ConvNeXtSPADEUNetDecoder2D(nn.Module):
    """ConvNeXtSPADE2D decoder with U-Net skips."""
    
    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_spade_blocks: int,
        n_levels: int,
        z_channels: int,
        num_anomaly_classes: int,
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
        self.n_spade_blocks = n_spade_blocks
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
            ch = 2 ** (n_levels - i + 2)

            self.ups.append(_upsample_block2d(prev_ch, ch, scale=2, use_transpose_conv=use_transpose_conv, gn_groups=gn_groups))

            skip_ch = 2 ** (n_levels - i + 2)
            self.fuse.append(
                nn.Sequential(
                    nn.Conv2d(ch + skip_ch, ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch), num_channels=ch, eps=1e-6),
                    nn.GELU(),
                )
            )

            # use SPADE here
            num_spade = min(self.n_spade_blocks, n_res_blocks)
            stage = nn.ModuleList()
            for j in range(n_res_blocks):
                if j < num_spade:
                    stage.append(ConvNeXtSPADEBlock2D(
                        channels=ch, num_anomaly_classes=num_anomaly_classes, 
                        mlp_ratio=4.0, drop_path=dp_rates[dp_i], dropout=dropout))
                else:
                    stage.append(ConvNeXtBlock2D(
                        channels=ch, mlp_ratio=4.0, gn_groups=gn_groups, 
                        drop_path=dp_rates[dp_i], dropout=dropout))
                dp_i += 1
            self.blocks.append(stage)

            prev_ch = ch

        self.out = nn.Conv2d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: Optional[List[torch.Tensor]]) -> None:
        self._skips = skips

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode latent feature map `z` into an image using `mask` for SPADE."""
        x = self.from_z(z)

        skips = self._skips
        if skips is not None and len(skips) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} skips, got {len(skips)}")

        for i in range(self.n_levels):
            x = self.ups[i](x)

            if skips is None:
                # No skips provided -> treat as zeros
                skip_ch = 2 ** (self.n_levels - i + 2)
                skip = torch.zeros(
                    (x.shape[0], skip_ch, x.shape[-2], x.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                skip = skips[-1 - i]

            # Align spatial sizes
            if x.shape[-2:] != skip.shape[-2:]:
                target = (min(x.shape[-2], skip.shape[-2]), min(x.shape[-1], skip.shape[-1]))
                x = HybridVAEBase._crop_like(x, target)
                skip = HybridVAEBase._crop_like(skip, target)

            # Apply skip scaling
            if self.skip_alpha != 1.0:
                skip = skip * self.skip_alpha

            # Skip-Dropout manually implemented
            p = self.skip_dropout_ps[-1 - i]
            if p > 0.0 and self.training:
                keep_prob = 1.0 - p
                drop_mask = (torch.rand((skip.shape[0], 1, 1, 1), device=skip.device, dtype=skip.dtype) < keep_prob).to(skip.dtype)
                skip = skip * drop_mask / max(keep_prob, 1e-6)

            x = torch.cat([x, skip], dim=1)
            x = self.fuse[i](x)

            # mask in every SPADE block
            num_spade = min(self.n_spade_blocks, len(self.blocks[i]))
            for j, block in enumerate(self.blocks[i]):
                if j < num_spade:
                    x = block(x, mask)  # SPADE block
                else:
                    x = block(x)    # normal block

        return self.out(x)


