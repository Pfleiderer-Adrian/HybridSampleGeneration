from dataclasses import asdict

from models.model_configuration import ModelConfiguration
from models.VAEs.VAE_ResNet import VAE_ResNet_2D, VAE_ResNet_3D


DEFAULT_VAE_INPUT_ARTEFACTS = ("img", "fname")


def _build_model_configuration(config_cls, in_channels, min_params, max_params, *, input_artefacts):
    return ModelConfiguration(
        asdict(config_cls(in_channels=in_channels, **min_params)),
        asdict(config_cls(in_channels=in_channels, **max_params)),
        input_artefacts=input_artefacts,
    )


def get_resnet_vae_3d_configuration(in_channels):
    base = {}
    return _build_model_configuration(
        VAE_ResNet_3D.Config,
        in_channels,
        {
            **base,
            "n_res_blocks": 4,
            "n_levels": 4,
            "z_channels": 64,
            "bottleneck_dim": 128,
            "use_multires_skips": True,
            "recon_weight": 100.0,
            "beta_kl": 0.05,
            "fg_weight": 1.0,
            "fg_threshold": 0.0,
            "recon_loss": "mse",
            "use_transpose_conv": False,
        },
        {
            **base,
            "n_res_blocks": 5,
            "n_levels": 5,
            "z_channels": 128,
            "bottleneck_dim": 256,
            "use_multires_skips": True,
            "recon_weight": 300.0,
            "beta_kl": 0.1,
            "fg_weight": 2.0,
            "fg_threshold": 0.0,
            "recon_loss": "mse",
            "use_transpose_conv": False,
        },
        input_artefacts=DEFAULT_VAE_INPUT_ARTEFACTS,
    )


def get_resnet_vae_2d_configuration(in_channels):
    base = {}
    return _build_model_configuration(
        VAE_ResNet_2D.Config,
        in_channels,
        {
            **base,
            "n_res_blocks": 4,
            "n_levels": 4,
            "z_channels": 32,
            "bottleneck_dim": 64,
            "use_multires_skips": False,
            "recon_weight": 5.0,
            "beta_kl": 0.1,
            "use_transpose_conv": False,
        },
        {
            **base,
            "n_res_blocks": 5,
            "n_levels": 5,
            "z_channels": 64,
            "bottleneck_dim": 128,
            "use_multires_skips": False,
            "recon_weight": 100.0,
            "beta_kl": 0.5,
            "use_transpose_conv": False,
        },
        input_artefacts=DEFAULT_VAE_INPUT_ARTEFACTS,
    )

