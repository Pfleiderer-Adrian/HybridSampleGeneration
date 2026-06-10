from dataclasses import asdict

from models.model_configuration import ModelConfiguration
from models.VAEs.cVAE_ConvNeXt import cVAE_ConvNeXt_2D, cVAE_ConvNeXt_3D


CONDITIONAL_VAE_INPUT_ARTEFACTS = ("img", "fname", "ori_mask", "tgt_mask")


def _build_model_configuration(config_cls, in_channels, min_params, max_params, *, input_artefacts):
    return ModelConfiguration(
        asdict(config_cls(in_channels=in_channels, **min_params)),
        asdict(config_cls(in_channels=in_channels, **max_params)),
        input_artefacts=input_artefacts,
    )


def get_convnext_cvae_3d_configuration(in_channels):
    base = {
        "num_anomaly_classes": None,
        "n_spade_blocks": 2,
        "z_channels": 128,
        "bottleneck_dim": 256,
        "use_multires_skips": True,
        "recon_weight": 1.0,
        "beta_kl": 4.0,
        "beta_kl_start": 0.0,
        "beta_kl_max": 7.0,
        "beta_kl_warmup_start": 0,
        "beta_kl_warmup_epochs": 100,
        "fg_threshold": 0.0,
        "recon_loss": "mse",
        "skip_dropout_p": 0.6,
        "skip_alpha": 0.2,
        "use_transpose_conv": False,
    }
    return _build_model_configuration(
        cVAE_ConvNeXt_3D.Config,
        in_channels,
        {
            **base,
            "n_res_blocks": 5,
            "n_levels": 5,
            "fg_weight": 1.0,
        },
        {
            **base,
            "n_res_blocks": 6,
            "n_levels": 6,
            "fg_weight": 2.0,
        },
        input_artefacts=CONDITIONAL_VAE_INPUT_ARTEFACTS,
    )


def get_convnext_cvae_2d_configuration(in_channels):
    base = {
        "num_anomaly_classes": None,
        "n_res_blocks": 4,
        "n_spade_blocks": 2,
        "n_levels": 4,
        "z_channels": 32,
        "bottleneck_dim": 64,
        "use_multires_skips": False,
        "recon_loss": "smoothl1",
        "recon_weight": 10.0,
        "drop_path_rate": 0.001,
        "dropout": 0.001,
        "skip_dropout_p": 1.0,
        "skip_alpha": 0.0,
        "use_transpose_conv": False,
        "beta_kl": 0.05,
        "beta_kl_start": 0.0,
        "beta_kl_max": 0.08,
        "beta_kl_warmup_start": 0,
        "beta_kl_warmup_epochs": 1000,
        "free_bits": 0.001,
        "fg_weight": 1.0,
        "fg_threshold": 0.0,
    }
    return _build_model_configuration(
        cVAE_ConvNeXt_2D.Config,
        in_channels,
        base,
        base,
        input_artefacts=CONDITIONAL_VAE_INPUT_ARTEFACTS,
    )


MODEL_CONFIGURATION_FACTORIES = {
    "cVAE_ConvNeXt_3D": get_convnext_cvae_3d_configuration,
    "cVAE_ConvNeXt_2D": get_convnext_cvae_2d_configuration,
}
