"""Dimension-independent building blocks for ResNet VAEs."""

from __future__ import annotations

import torch
import torch.nn as nn


def _spatial_layers(spatial_dims: int):
    if spatial_dims == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, "bilinear"
    if spatial_dims == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d, "trilinear"
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")


class ResidualBlock(nn.Module):
    """
    Basic residual block for spatial data.

    Inputs
    ------
    x:
        torch.Tensor (B, in_ch, H, W)

    Outputs
    -------
    torch.Tensor:
        (B, out_ch, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, leak: float = 0.2, *, spatial_dims: int):
        super().__init__()
        Conv, _, BatchNorm, _ = _spatial_layers(spatial_dims)
        self.conv1 = Conv(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = BatchNorm(out_ch)
        self.act1 = nn.LeakyReLU(leak, inplace=True)

        self.conv2 = Conv(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(out_ch)
        self.act2 = nn.LeakyReLU(leak, inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = Conv(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return identity + out


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder with optional multi-resolution skip aggregation.

    Inputs
    ------
    x:
        torch.Tensor (B, C, H, W)

    Outputs
    -------
    torch.Tensor:
        Latent feature map h of shape (B, z_channels, h', w')
        (spatial dims reduced by 2**n_levels).
    """
    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,
        leak: float = 0.2,
        *,
        spatial_dims: int,
    ):
        super().__init__()
        Conv, _, BatchNorm, _ = _spatial_layers(spatial_dims)
        self.spatial_dims = spatial_dims
        self.n_levels = n_levels
        self.use_multires_skips = use_multires_skips
        self.max_filters = 2 ** (n_levels + 3)

        # Initial projection to 8 channels
        self.input_conv = nn.Sequential(
            Conv(in_channels, 8, 3, 1, 1, bias=False),
            BatchNorm(8),
            nn.LeakyReLU(leak, inplace=True),
        )

        # Per-level stacks:
        # - res_stages: residual processing at current resolution
        # - down_stages: stride-2 downsampling conv
        # - skip_stages: multi-resolution skip projections to max_filters (optional)
        self.res_stages = nn.ModuleList()
        self.down_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        for i in range(n_levels):
            # Stage channels grow as powers of 2
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)

            # Residual blocks at current resolution
            self.res_stages.append(
                nn.Sequential(
                    *[ResidualBlock(n_filters_1, n_filters_1, leak=leak, spatial_dims=spatial_dims) for _ in range(n_res_blocks)]
                )
            )

            # Downsample by factor 2 in each spatial axis
            self.down_stages.append(
                nn.Sequential(
                    Conv(n_filters_1, n_filters_2, kernel_size=2, stride=2, padding=0, bias=False),
                    BatchNorm(n_filters_2),
                    nn.LeakyReLU(leak, inplace=True),
                )
            )

            # Optional multi-resolution skip: downsample current features to a common resolution and sum
            if use_multires_skips:
                ks = 2 ** (n_levels - i)
                self.skip_stages.append(
                    nn.Sequential(
                        Conv(n_filters_1, self.max_filters, kernel_size=ks, stride=ks, padding=0, bias=False),
                        BatchNorm(self.max_filters),
                        nn.LeakyReLU(leak, inplace=True),
                    )
                )

        # Final projection into z_channels
        self.output_conv = Conv(2 ** (n_levels + 3), z_channels, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            # Residual processing
            x = self.res_stages[i](x)
            if self.use_multires_skips:
                skips.append(self.skip_stages[i](x))
            # Downsample for next level
            x = self.down_stages[i](x)

        # Sum all multi-resolution skips into the deepest representation
        if self.use_multires_skips:
            x = x + torch.stack(skips, dim=0).sum(dim=0)

        return self.output_conv(x)


class ResNetDecoder(nn.Module):
    """
    ResNet-style decoder with optional multi-resolution skip injections from the top latent.

    Inputs
    ------
    z:
        torch.Tensor (B, z_channels, h', w')

    Outputs
    -------
    torch.Tensor:
        Reconstructed feature map (B, out_channels, H, W) after upsampling.
    """
    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,
        leak: float = 0.2,
        use_transpose_conv: bool = True,
        *,
        spatial_dims: int,
    ):
        super().__init__()
        Conv, ConvTranspose, BatchNorm, interpolation_mode = _spatial_layers(spatial_dims)
        self.spatial_dims = spatial_dims
        self.n_levels = n_levels
        self.use_multires_skips = use_multires_skips
        self.max_filters = 2 ** (n_levels + 3)
        self.use_transpose_conv = use_transpose_conv

        # Project latent channels to max_filters
        self.input_conv = nn.Sequential(
            Conv(z_channels, self.max_filters, 3, 1, 1, bias=False),
            BatchNorm(self.max_filters),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.up_stages = nn.ModuleList()
        self.res_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        def upsample_block(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
            if self.use_transpose_conv:
                return nn.Sequential(
                    ConvTranspose(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=False),
                    BatchNorm(out_ch),
                    nn.LeakyReLU(leak, inplace=True),
                )
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode=interpolation_mode, align_corners=False),
                Conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(out_ch),
                nn.LeakyReLU(leak, inplace=True),
            )

        # Start from max_filters and go up in resolution
        prev_ch = self.max_filters
        for i in range(n_levels):
            # Channels shrink as we go to higher spatial resolution
            n_filters = 2 ** (self.n_levels - i + 2)

            # Upsample by factor 2
            self.up_stages.append(
                upsample_block(prev_ch, n_filters, scale=2)
            )
            prev_ch = n_filters

            # Residual refinement
            self.res_stages.append(
                nn.Sequential(
                    *[ResidualBlock(n_filters, n_filters, leak=leak, spatial_dims=spatial_dims) for _ in range(n_res_blocks)]
                )
            )

            # Optional multi-res skip injection from top feature map z_top
            if use_multires_skips:
                ks = 2 ** (i + 1)
                self.skip_stages.append(
                    upsample_block(self.max_filters, n_filters, scale=ks)
                )

        # Output reconstruction conv
        self.output_conv = Conv(prev_ch, out_channels, 3, 1, 1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Keep a copy of the top feature map for skip injections
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.up_stages[i](z)
            z = self.res_stages[i](z)
            if self.use_multires_skips:
                # Add injected skip feature (same shape as z) if enabled
                z = z + self.skip_stages[i](z_top)

        return self.output_conv(z)


