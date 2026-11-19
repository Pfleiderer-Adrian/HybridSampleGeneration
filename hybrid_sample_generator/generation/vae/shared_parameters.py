"""Constraints and descriptions of parameters shared by the VAE variants.

Each model keeps its own defaults: ``param(8, **SHARED["n_res_blocks"])``.
"""

from hybrid_sample_generator.configuration.parameters import param


_PROBABILITY = dict(ge=0.0, le=1.0)

SHARED = {
    "n_res_blocks": dict(ge=1, doc="Number of residual blocks per level."),
    "n_spade_blocks": dict(ge=0, doc="Number of mask-conditioned SPADE blocks."),
    "n_levels": dict(ge=1, doc="Number of encoder and decoder levels."),
    "z_channels": dict(ge=1, doc="Channel count in the spatial bottleneck."),
    "bottleneck_dim": dict(ge=1, doc="Dimension of the latent vector."),
    "use_multires_skips": dict(
        doc="Use encoder features from multiple resolutions as skip connections.",
    ),
    "recon_weight": dict(ge=0.0, doc="Weight of the reconstruction loss."),
    "latent_recon_weight": dict(
        ge=0.0,
        doc="Weight of the latent reconstruction Smooth L1 loss; 0 disables the loss.",
    ),
    "latent_recon_noise_scale": dict(
        gt=0.0,
        doc="Scale of Gaussian noise added to the detached latent mean for the reconstruction cycle.",
    ),
    "latent_recon_image_noise_std": dict(
        ge=0.0,
        doc="Standard deviation of Gaussian noise added to the cycle image before re-encoding during training.",
    ),
    "beta_kl_start": dict(ge=0.0, doc="Initial weight of the KL loss."),
    "beta_kl_max": dict(ge=0.0, doc="Maximum weight of the KL loss."),
    "beta_kl_warmup_start": dict(
        ge=0, doc="Epoch at which the KL weight begins to increase.",
    ),
    "beta_kl_warmup_epochs": dict(
        ge=0, doc="Number of epochs needed to reach beta_kl_max.",
    ),
    "free_bits": dict(ge=0.0, doc="KL free-bits allowance for latent dimensions."),
    "recon_loss": dict(
        schema={"type": "choice", "choices": ["smoothl1", "mse"]},
        validated=False,
        doc="Reconstruction loss. The aliases 'smooth_l1', 'huber' and 'l2' are accepted.",
    ),
    "recon_smoothl1_beta": dict(ge=0.0, doc="Transition point of the Smooth L1 loss."),
    "use_transpose_conv": dict(doc="Use transposed convolutions for upsampling."),
    "foreground_weight": dict(
        ge=0.0,
        doc="Relative weight of the mask foreground mean in the reconstruction loss.",
    ),
    "background_weight": dict(
        ge=0.0,
        doc="Relative weight of the mask background mean in the reconstruction loss.",
    ),
    "drop_path_rate": dict(**_PROBABILITY, doc="Stochastic depth rate in ConvNeXt blocks."),
    "dropout": dict(**_PROBABILITY, doc="Dropout probability within the model."),
    "skip_dropout_p": dict(
        **_PROBABILITY, doc="Shared dropout probability for skip connections.",
    ),
    "skip_dropout_ps": dict(
        **_PROBABILITY,
        doc="Dropout per skip level; overrides skip_dropout_p and requires n_levels values.",
    ),
    "skip_alpha": dict(**_PROBABILITY, doc="Shared skip connection scale."),
    "skip_alphas": dict(
        **_PROBABILITY,
        doc="Scale per skip level in encoder order, highest resolution to deepest; overrides skip_alpha and requires n_levels values.",
    ),
    "identity_pair_probability": dict(
        **_PROBABILITY,
        doc="Probability of using an unchanged source/target pair during paired source-to-target training.",
    ),
}


__all__ = ["SHARED", "param"]
