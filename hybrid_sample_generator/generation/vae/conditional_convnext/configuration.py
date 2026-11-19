"""Default parameters and search spaces for conditional ConvNeXt VAEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hybrid_sample_generator.generation.vae.shared_parameters import SHARED, param
from hybrid_sample_generator.generation.model_settings import Choice, IntRange, SearchSpace


@dataclass
class Config:
    """Concrete parameters for the conditional ConvNeXt VAE."""

    n_res_blocks: int = param(8, **SHARED["n_res_blocks"])
    n_spade_blocks: int = param(2, **SHARED["n_spade_blocks"])
    n_levels: int = param(4, **SHARED["n_levels"])
    z_channels: int = param(250, **SHARED["z_channels"])
    bottleneck_dim: int = param(250, **SHARED["bottleneck_dim"])
    recon_weight: float = param(100.0, **SHARED["recon_weight"])
    beta_kl_start: float = param(0.0, **SHARED["beta_kl_start"])
    beta_kl_max: float = param(0.03, **SHARED["beta_kl_max"])
    beta_kl_warmup_start: int = param(20, **SHARED["beta_kl_warmup_start"])
    beta_kl_warmup_epochs: int = param(30, **SHARED["beta_kl_warmup_epochs"])
    free_bits: float = param(0.0, **SHARED["free_bits"])
    latent_recon_weight: float = param(0.0, **SHARED["latent_recon_weight"])
    latent_recon_noise_scale: float = param(1.0, **SHARED["latent_recon_noise_scale"])
    latent_recon_image_noise_std: float = param(0.03, **SHARED["latent_recon_image_noise_std"])
    recon_loss: str = param("smoothl1", **SHARED["recon_loss"])
    recon_smoothl1_beta: float = param(1.0, **SHARED["recon_smoothl1_beta"])
    use_transpose_conv: bool = param(True, **SHARED["use_transpose_conv"])
    foreground_weight: float = param(0.8, **SHARED["foreground_weight"])
    background_weight: float = param(0.2, **SHARED["background_weight"])
    drop_path_rate: float = param(0.1, **SHARED["drop_path_rate"])
    dropout: float = param(0.05, **SHARED["dropout"])
    skip_dropout_p: float = param(0.0, **SHARED["skip_dropout_p"])
    skip_dropout_ps: Optional[List[float]] = param(None, **SHARED["skip_dropout_ps"])
    skip_alpha: float = param(1.0, **SHARED["skip_alpha"])
    skip_alphas: Optional[List[float]] = param(None, **SHARED["skip_alphas"])


def get_convnext_cvae_configuration(spatial_dims: int) -> Config:
    if spatial_dims == 2:
        return Config(
            n_res_blocks=4,
            n_spade_blocks=2,
            n_levels=4,
            z_channels=32,
            bottleneck_dim=64,
            recon_loss="smoothl1",
            recon_weight=10.0,
            drop_path_rate=0.001,
            dropout=0.001,
            skip_dropout_p=1.0,
            skip_dropout_ps=None,
            skip_alpha=0.0,
            use_transpose_conv=False,
            beta_kl_start=0.0,
            beta_kl_max=0.08,
            beta_kl_warmup_start=0,
            beta_kl_warmup_epochs=200,
            free_bits=0.001,
            foreground_weight=0.8,
            background_weight=0.2,
        )
    if spatial_dims == 3:
        return Config(
            n_spade_blocks=2,
            n_res_blocks=5,
            n_levels=5,
            z_channels=128,
            bottleneck_dim=256,
            recon_weight=1.0,
            beta_kl_start=0.0,
            beta_kl_max=0.01,
            beta_kl_warmup_start=0,
            beta_kl_warmup_epochs=100,
            foreground_weight=0.8,
            background_weight=0.2,
            recon_loss="mse",
            skip_dropout_p=0.6,
            skip_dropout_ps=None,
            skip_alpha=0.2,
            use_transpose_conv=False,
        )
    raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}.")


def get_convnext_cvae_search(
    parameters: Config,
    spatial_dims: int,
) -> SearchSpace:
    search = SearchSpace(parameters, n_res_blocks=IntRange(4, 5))
    if spatial_dims == 2:
        search.z_channels = Choice((32, 64))
        search.bottleneck_dim = Choice((64, 128))
        return search
    if spatial_dims == 3:
        search.z_channels = Choice((64, 128))
        search.bottleneck_dim = Choice((128, 256))
        return search
    raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}.")
