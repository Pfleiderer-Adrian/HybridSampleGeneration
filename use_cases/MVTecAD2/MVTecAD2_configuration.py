"""Category-specific MVTec AD 2 configuration presets."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from synthesizer.Configuration import Configuration


MVTECAD2_CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)

CATEGORY_ALIASES = {
    "wall_plugs": "wallplugs",
    "wall plugs": "wallplugs",
}

CATEGORY_ANOMALY_SHAPES: dict[str, tuple[int, int, int]] = {
    "can": (3, 64, 64),
    "fabric": (3, 64, 64),
    "fruit_jelly": (3, 64, 64),
    "rice": (3, 64, 64),
    "sheet_metal": (1, 64, 64),
    "vial": (1, 64, 64),
    "wallplugs": (1, 64, 64),
    "walnuts": (3, 64, 64),
}

CATEGORY_GENERATION_MODEL: dict[str, str] = {
    "can": "cVAE_ConvNeXt_2D",
    "fabric": "cVAE_ConvNeXt_2D",
    "fruit_jelly": "cVAE_ConvNeXt_2D",
    "rice": "cVAE_ConvNeXt_2D",
    "sheet_metal": "cVAE_ConvNeXt_2D",
    "vial": "cVAE_ConvNeXt_2D",
    "wallplugs": "cVAE_ConvNeXt_2D",
    "walnuts": "cVAE_ConvNeXt_2D",
}


def create_mvtecad2_configuration(
    category: str,
    *,
    save_path: Path | str | None = None,
    results_root: Path | str | None = None,
    apply_category_overrides: bool = True,
) -> Configuration:
    """
    Create one Configuration instance for a single MVTec AD 2 category.

    model_name and anomaly_size are fixed at instantiation time by the
    Configuration class. Use the configure_<category>() hooks below only for
    mutable pipeline settings and model-parameter search spaces.

    save_path is forwarded to Configuration. The resulting study folder is
    <save_path>/results/<study_name>. results_root is kept as a legacy alias.
    """

    if save_path is not None and results_root is not None:
        raise ValueError("Use either save_path or results_root, not both.")

    category = canonical_category(category)
    config_save_path = save_path if save_path is not None else results_root
    generation_model = CATEGORY_GENERATION_MODEL[category]

    config = Configuration(
        f"mvtecad2_{safe_name(f'{category}_{generation_model}')}",
        generation_model,
        CATEGORY_ANOMALY_SHAPES[category],
        save_path=config_save_path,
    )
    config = configure_mvtecad2_defaults(config)
    if apply_category_overrides:
        config = CATEGORY_CONFIGURATORS[category](config)
    return config


def create_mvtecad2_configurations(
    categories: Iterable[str] | None = None,
    **kwargs,
) -> dict[str, Configuration]:
    """
    Create independent Configuration instances for all selected categories.

    If save_path is provided, each category gets its own save path below it:
    <save_path>/<category>.
    """

    save_path = kwargs.pop("save_path", None)
    results_root = kwargs.pop("results_root", None)
    if save_path is not None and results_root is not None:
        raise ValueError("Use either save_path or results_root, not both.")
    base_save_path = save_path if save_path is not None else results_root

    configs: dict[str, Configuration] = {}
    for category in categories or MVTECAD2_CATEGORIES:
        category = canonical_category(category)

        category_save_path = None
        if base_save_path is not None:
            category_save_path = Path(base_save_path) / category
            category_save_path.mkdir(parents=True, exist_ok=True)

        configs[category] = create_mvtecad2_configuration(
            category,
            save_path=category_save_path,
            **kwargs,
        )
    return configs


def configure_mvtecad2_defaults(config: Configuration):
    """
    Shared MVTec AD 2 defaults for all categories.
    """
    image_channels = int(config.anomaly_size[0])

    # extraction settings
    config.fixed_roi_size = None
    config.random_offset = True
    config.random_offset_max_fraction = 0.8
    config.random_offset_foreground_threshold_rel = 0.01
    config.add_bg_noise = False
    config.min_anomaly_percentage = 0.01
    config.min_pad = (20, 20, 20)
    config.pad_ratio = (0.5, 0.5, 0.5)

    # generation settings
    config.clamp01_output = False
    config.normalization = "z-score"
    config.normalization_eps = 1e-6
    config.background_threshold = 0.18
    config.prior_sampling = False
    config.use_feedback = False
    config.feedback_threshold = 0.01
    config.threshold_relaxation_factor = 0.9
    config.variation_strength = 1.25

    # matching settings
    config.matching_routine = "local"
    config.anomaly_duplicates = True
    config.fusions_per_control = 2
    config.max_fusions_per_control_deviation = 1

    # Fusion settings
    config.set_fusion_backend("classical")
    config.fusion_params.set_fusion_params(
        max_alpha=1.0,
        sq=0.1,
        steepness_factor=5.0,
        upsampling_factor=2,
        sobel_threshold=0.01,
        dilation_size=1,
        shave_pixels=0,
        fusion_variation=True,
        alpha_variation=0.05,
        sq_variation=0.1,
        steepness_variation=1.0,
        selected_confidence="90%",
    )
    """
    config.set_fusion_backend("learned_residual_alpha")
    config.fusion_params.set_fusion_params(
    base_alpha=0.70,
    base_alpha_blur_sigma=2.0,
    alpha_delta_scale=0.35,
    residual_scale=0.35,
    residual_border_width=6,
    fusion_normalization_border_width=None,
    clamp_output=False,

    train_epochs=50,
    train_lr=1e-3,
    train_weight_decay=1e-5,
    train_crop_margin=32,
    train_inpaint_blur_sigma=10.0,
    foreground_loss_weight=3.0,
    support_loss_weight=4.0,
    alpha_delta_l1=1e-4,
    residual_l1=5e-5,
    grad_clip_norm=1.0,
    )
    """
    # Training settings
    config.val_ratio = 0.1
    config.batch_size = 8
    config.epochs = 1000
    config.lr = 1e-4
    config.grad_clip_norm = 1.0
    config.log_every = None
    config.early_stopping = True
    config.early_stopping_params = {
        "patience": 400,
        "delta": 0.0001,
    }
    config.lr_scheduler = True
    config.lr_scheduler_params = {
        "patience": 200,
        "factor": 0.1,
        "threshold": 1e-5,
    }

    # Model hyperparameter search space for Optuna. The min and max dicts together define the search space.
    config.model_params.set_hyperparameter_space(
        # min_config
        {
            "in_channels": image_channels,
            "n_res_blocks": 2,
            "n_levels": 3,
            "z_channels": 16,
            "bottleneck_dim": 32,
            "use_multires_skips": False,
            "recon_weight": 4.0,
            "beta_kl": 0.03,
            "beta_kl_start": 0.0,
            "beta_kl_max": 0.06,
            "beta_kl_warmup_start": 0,
            "beta_kl_warmup_epochs": 150,
            "free_bits": 0.0,
            "recon_loss": "smoothl1",
            "recon_smoothl1_beta": 0.35,
            "use_transpose_conv": False,
            "fg_weight": 0.8,
            "fg_threshold": 0.0,
            "drop_path_rate": 0.0,
            "dropout": 0.01,
            "skip_dropout_p": 0.75,
            "skip_alpha": 0.0,
        },
        # max_config
        {
            "in_channels": image_channels,
            "n_res_blocks": 4,
            "n_levels": 4,
            "z_channels": 96,
            "bottleneck_dim": 160,
            "use_multires_skips": False,
            "recon_weight": 24.0,
            "beta_kl": 0.08,
            "beta_kl_start": 0.0,
            "beta_kl_max": 0.25,
            "beta_kl_warmup_start": 0,
            "beta_kl_warmup_epochs": 900,
            "free_bits": 0.01,
            "recon_loss": "smoothl1",
            "recon_smoothl1_beta": 1.25,
            "use_transpose_conv": False,
            "fg_weight": 1.5,
            "fg_threshold": 0.0,
            "drop_path_rate": 0.08,
            "dropout": 0.20,
            "skip_dropout_p": 1.0,
            "skip_alpha": 0.20,
        },
    )
    return config


def configure_can(config: Configuration) -> Configuration:
    config.variation_strength = 1.5
    config.fusion_params.set_fusion_params(max_alpha=0.5)

    """
    # Prototype diffusion model settings for can category. These parameters are not tuned.
    # Diffusion model settings. num_anomaly_classes is filled after masks are loaded.
    config.model_params.set_model_params(
        prompt="a realistic close-up photo of a damaged can surface, industrial anomaly texture, high detail",
        negative_prompt="blur, low quality, text, watermark",
        resolution=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.85,
        prior_strength=0.999,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    """

    return config


def configure_fabric(config: Configuration) -> Configuration:
    config.min_roi_size = (128, 128)
    return config


def configure_fruit_jelly(config: Configuration) -> Configuration:
    return config


def configure_rice(config: Configuration) -> Configuration:
    return config


def configure_sheet_metal(config: Configuration) -> Configuration:
    return config


def configure_vial(config: Configuration) -> Configuration:
    return config


def configure_wallplugs(config: Configuration) -> Configuration:
    return config


def configure_walnuts(config: Configuration) -> Configuration:
    return config


CATEGORY_CONFIGURATORS: dict[str, Callable[[Configuration], Configuration]] = {
    "can": configure_can,
    "fabric": configure_fabric,
    "fruit_jelly": configure_fruit_jelly,
    "rice": configure_rice,
    "sheet_metal": configure_sheet_metal,
    "vial": configure_vial,
    "wallplugs": configure_wallplugs,
    "walnuts": configure_walnuts,
}


def canonical_category(category: str) -> str:
    category = category.strip().lower()
    return CATEGORY_ALIASES.get(category, category)


def safe_name(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"


MVTECAD2_CONFIGURATIONS = create_mvtecad2_configurations()
