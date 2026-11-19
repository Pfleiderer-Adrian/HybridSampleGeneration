"""Default parameters and search spaces for ResNet VAEs."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_sample_generator.generation.vae.shared_parameters import SHARED, param
from hybrid_sample_generator.generation.model_settings import (
    Choice,
    IntRange,
    SearchSpace,
)


@dataclass
class Config:
    """Concrete parameters for the ResNet VAE."""

    n_res_blocks: int = param(8, **SHARED["n_res_blocks"])
    n_levels: int = param(4, **SHARED["n_levels"])
    z_channels: int = param(250, **SHARED["z_channels"])
    bottleneck_dim: int = param(250, **SHARED["bottleneck_dim"])
    use_multires_skips: bool = param(True, **SHARED["use_multires_skips"])
    recon_weight: float = param(100.0, **SHARED["recon_weight"])
    beta_kl_start: float = param(0.0, **SHARED["beta_kl_start"])
    beta_kl_max: float = param(0.03, **SHARED["beta_kl_max"])
    beta_kl_warmup_start: int = param(20, **SHARED["beta_kl_warmup_start"])
    beta_kl_warmup_epochs: int = param(30, **SHARED["beta_kl_warmup_epochs"])
    free_bits: float = param(0.0, **SHARED["free_bits"])
    recon_loss: str = param("smoothl1", **SHARED["recon_loss"])
    recon_smoothl1_beta: float = param(1.0, **SHARED["recon_smoothl1_beta"])
    use_transpose_conv: bool = param(True, **SHARED["use_transpose_conv"])
    foreground_weight: float = param(0.8, **SHARED["foreground_weight"])
    background_weight: float = param(0.2, **SHARED["background_weight"])


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
