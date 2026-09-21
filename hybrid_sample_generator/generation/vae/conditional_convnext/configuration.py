"""Default parameters and search spaces for conditional ConvNeXt VAEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hybrid_sample_generator.generation.model_settings import IntRange, SearchSpace


@dataclass
class Config:
    """Concrete parameters for the conditional ConvNeXt VAE."""

    n_res_blocks: int = 8
    n_spade_blocks: int = 2
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    recon_weight: float = 100.0
    beta_kl_start: float = 0.0
    beta_kl_max: float = 0.03
    beta_kl_warmup_start: int = 20
    beta_kl_warmup_epochs: int = 30
    free_bits: float = 0.0
    recon_loss: str = "smoothl1"
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0
    drop_path_rate: float = 0.1
    dropout: float = 0.05
    skip_dropout_p: float = 0.0
    skip_dropout_ps: Optional[List[float]] = None
    skip_alpha: float = 1.0
    # Encoder order: highest resolution to deepest. Overrides skip_alpha.
    skip_alphas: Optional[List[float]] = None


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
            fg_weight=1.0,
            fg_threshold=0.0,
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
            fg_weight=1.0,
            fg_threshold=0.0,
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
    search = SearchSpace(parameters)
    if spatial_dims == 2:
        return search
    if spatial_dims == 3:
        search.n_res_blocks = IntRange(5, 6)
        search.n_levels = IntRange(5, 6)
        return search
    raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}.")
