"""Skip-free SPADE decoder for paired conditional generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.conditional_convnext.spade import (
    ConvNeXtSPADEBlock,
)
from hybrid_sample_generator.generation.vae.convnext.layers import (
    ConvNeXtBlock,
    _best_gn_groups,
    _upsample_block,
)


def _conv(spatial_dims: int):
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")


class ConvNeXtSPADENoSkipDecoder(nn.Module):
    """ConvNeXt SPADE decoder whose only spatial condition is the target mask."""

    def __init__(
        self,
        *,
        out_channels: int,
        n_res_blocks: int,
        n_spade_blocks: int,
        n_levels: int,
        z_channels: int,
        num_anomaly_classes: int,
        use_transpose_conv: bool,
        drop_path_rate: float,
        dropout: float,
        spatial_dims: int,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        Conv = _conv(spatial_dims)
        self.n_levels = int(n_levels)
        self.n_spade_blocks = int(n_spade_blocks)

        bottom_channels = 2 ** (n_levels + 3)
        self.from_z = nn.Sequential(
            Conv(z_channels, bottom_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(
                _best_gn_groups(gn_groups, bottom_channels),
                bottom_channels,
                eps=1e-6,
            ),
            nn.GELU(),
        )

        total_blocks = n_levels * n_res_blocks
        rates = (
            torch.linspace(0.0, float(drop_path_rate), steps=total_blocks).tolist()
            if drop_path_rate > 0 and total_blocks > 0
            else [0.0] * total_blocks
        )
        rate_index = 0
        previous_channels = bottom_channels
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for level in range(n_levels):
            channels = 2 ** (n_levels - level + 2)
            self.ups.append(
                _upsample_block(
                    previous_channels,
                    channels,
                    scale=2,
                    use_transpose_conv=use_transpose_conv,
                    gn_groups=gn_groups,
                    spatial_dims=spatial_dims,
                )
            )
            stage = nn.ModuleList()
            for block_index in range(n_res_blocks):
                if block_index < min(n_spade_blocks, n_res_blocks):
                    block = ConvNeXtSPADEBlock(
                        channels,
                        num_anomaly_classes,
                        drop_path=rates[rate_index],
                        dropout=dropout,
                        spatial_dims=spatial_dims,
                    )
                else:
                    block = ConvNeXtBlock(
                        channels,
                        drop_path=rates[rate_index],
                        dropout=dropout,
                        gn_groups=gn_groups,
                        spatial_dims=spatial_dims,
                    )
                stage.append(block)
                rate_index += 1
            self.blocks.append(stage)
            previous_channels = channels

        self.out = Conv(previous_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        x = self.from_z(z)
        for level in range(self.n_levels):
            x = self.ups[level](x)
            for block_index, block in enumerate(self.blocks[level]):
                if block_index < min(self.n_spade_blocks, len(self.blocks[level])):
                    x = block(x, target_mask)
                else:
                    x = block(x)
        return self.out(x)


__all__ = ["ConvNeXtSPADENoSkipDecoder"]
