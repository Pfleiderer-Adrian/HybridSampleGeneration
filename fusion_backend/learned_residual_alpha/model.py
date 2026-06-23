from __future__ import annotations

import torch
from torch import nn


def _conv_nd(spatial_dims: int):
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError(f"spatial_dims must be 2 or 3. Got {spatial_dims!r}.")


def _norm_nd(spatial_dims: int):
    if spatial_dims == 2:
        return nn.BatchNorm2d
    if spatial_dims == 3:
        return nn.BatchNorm3d
    raise ValueError(f"spatial_dims must be 2 or 3. Got {spatial_dims!r}.")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int) -> None:
        super().__init__()
        conv = _conv_nd(spatial_dims)
        norm = _norm_nd(spatial_dims)
        self.block = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            nn.SiLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            nn.SiLU(inplace=True),
        )
        self.proj = None
        if in_channels != out_channels:
            self.proj = conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        return self.block(x) + residual


class ResidualAlphaRefiner(nn.Module):
    """
    Compact fully-convolutional refiner.

    Input channels are:
      control, anomaly, base_fused, target_mask, base_alpha, support_mask
      => 3 * image_channels + 3

    Output channels are:
      alpha_delta, residual_per_image_channel
      => 1 + image_channels
    """

    def __init__(
        self,
        *,
        input_channels: int,
        image_channels: int,
        spatial_dims: int,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.image_channels = int(image_channels)
        self.spatial_dims = int(spatial_dims)

        conv = _conv_nd(self.spatial_dims)
        layers = [conv(self.input_channels, base_channels, kernel_size=3, padding=1), nn.SiLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers.append(_ConvBlock(base_channels, base_channels, self.spatial_dims))
        self.body = nn.Sequential(*layers)
        self.head = conv(base_channels, 1 + self.image_channels, kernel_size=1)

        # An untrained backend should behave like its deterministic base fusion.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.body(x))
        alpha_delta = out[:, :1]
        residual = out[:, 1:]
        return alpha_delta, residual
