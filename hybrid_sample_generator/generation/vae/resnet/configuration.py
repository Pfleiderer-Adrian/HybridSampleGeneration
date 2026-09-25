"""Default parameters and search spaces for ResNet VAEs."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_sample_generator.generation.model_settings import (
    Choice,
    IntRange,
    SearchSpace,
)


@dataclass
class Config:
    """Concrete parameters for the ResNet VAE."""

    n_res_blocks: int = 8
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    use_multires_skips: bool = True
    recon_weight: float = 100.0
    beta_kl_start: float = 0.0
    beta_kl_max: float = 0.03
    beta_kl_warmup_start: int = 20
    beta_kl_warmup_epochs: int = 30
    free_bits: float = 0.0
    recon_loss: str = "smoothl1"
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    foreground_weight: float = 0.8
    background_weight: float = 0.2


def get_resnet_vae_configuration(spatial_dims: int) -> Config:
    if spatial_dims == 2:
        return Config(
            n_res_blocks=4,
            n_levels=4,
            z_channels=32,
            bottleneck_dim=64,
            use_multires_skips=False,
            recon_weight=5.0,
            beta_kl_max=0.1,
            use_transpose_conv=False,
        )
    if spatial_dims == 3:
        return Config(
            n_res_blocks=4,
            n_levels=4,
            z_channels=64,
            bottleneck_dim=128,
            use_multires_skips=True,
            recon_weight=100.0,
            beta_kl_max=0.05,
            foreground_weight=0.8,
            background_weight=0.2,
            recon_loss="mse",
            use_transpose_conv=False,
        )
    raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}.")


def get_resnet_vae_search(parameters: Config, spatial_dims: int) -> SearchSpace:
    search = SearchSpace(
        parameters,
        n_res_blocks=IntRange(4, 5),
    )
    if spatial_dims == 2:
        search.z_channels = Choice((32, 64))
        search.bottleneck_dim = Choice((64, 128))
    elif spatial_dims == 3:
        search.z_channels = Choice((64, 128))
        search.bottleneck_dim = Choice((128, 256))
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}.")
    return search
