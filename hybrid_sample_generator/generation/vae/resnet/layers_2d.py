"""Building blocks for the two-dimensional ResNet VAE."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    """
    Basic residual block for 2D images.

    Inputs
    ------
    x:
        torch.Tensor (B, in_ch, H, W)

    Outputs
    -------
    torch.Tensor:
        (B, out_ch, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, leak: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(leak, inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(leak, inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return identity + out


class ResNetEncoder2D(nn.Module):
    """
    ResNet-style 2D encoder with optional multi-resolution skip aggregation.

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
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_multires_skips = use_multires_skips
        self.max_filters = 2 ** (n_levels + 3)

        # Initial projection to 8 channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
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
                    *[ResidualBlock2D(n_filters_1, n_filters_1, leak=leak) for _ in range(n_res_blocks)]
                )
            )

            # Downsample by factor 2 in each spatial axis
            self.down_stages.append(
                nn.Sequential(
                    nn.Conv2d(n_filters_1, n_filters_2, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(n_filters_2),
                    nn.LeakyReLU(leak, inplace=True),
                )
            )

            # Optional multi-resolution skip: downsample current features to a common resolution and sum
            if use_multires_skips:
                ks = 2 ** (n_levels - i)
                self.skip_stages.append(
                    nn.Sequential(
                        nn.Conv2d(n_filters_1, self.max_filters, kernel_size=ks, stride=ks, padding=0, bias=False),
                        nn.BatchNorm2d(self.max_filters),
                        nn.LeakyReLU(leak, inplace=True),
                    )
                )

        # Final projection into z_channels
        self.output_conv = nn.Conv2d(2 ** (n_levels + 3), z_channels, 3, 1, 1, bias=True)

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


class ResNetDecoder2D(nn.Module):
    """
    ResNet-style 2D decoder with optional multi-resolution skip injections from the top latent.

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
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_multires_skips = use_multires_skips
        self.max_filters = 2 ** (n_levels + 3)
        self.use_transpose_conv = use_transpose_conv

        # Project latent channels to max_filters
        self.input_conv = nn.Sequential(
            nn.Conv2d(z_channels, self.max_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.max_filters),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.up_stages = nn.ModuleList()
        self.res_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        def upsample_block(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
            if self.use_transpose_conv:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(leak, inplace=True),
                )
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
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
                    *[ResidualBlock2D(n_filters, n_filters, leak=leak) for _ in range(n_res_blocks)]
                )
            )

            # Optional multi-res skip injection from top feature map z_top
            if use_multires_skips:
                ks = 2 ** (i + 1)
                self.skip_stages.append(
                    upsample_block(self.max_filters, n_filters, scale=ks)
                )

        # Output reconstruction conv
        self.output_conv = nn.Conv2d(prev_ch, out_channels, 3, 1, 1, bias=True)

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


