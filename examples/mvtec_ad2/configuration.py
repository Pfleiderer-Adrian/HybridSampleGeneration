"""Category-specific MVTec AD 2 configuration presets."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from hybrid_sample_generator.configuration.root import Configuration


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
    apply_category_overrides: bool = True,
) -> Configuration:
    """
    Create one Configuration instance for a single MVTec AD 2 category.

    The constructor selects the model and anomaly size. Category hooks below
    customize the domain-specific configuration sections.

    save_path is forwarded to Configuration. The resulting study folder is
    <save_path>/results/<study_name>.
    """

    category = canonical_category(category)
    generation_model = CATEGORY_GENERATION_MODEL[category]

    config = Configuration(
        f"mvtecad2_{safe_name(f'{category}_{generation_model}')}",
        generation_model,
        CATEGORY_ANOMALY_SHAPES[category],
        save_path=save_path,
    )
    config = configure_mvtecad2_defaults(config)
    if apply_category_overrides and category in CATEGORY_CONFIGURATORS:
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

    base_save_path = kwargs.pop("save_path", None)

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
    image_channels = int(config.extraction.anomaly_size[0])

    # extraction settings
    config.extraction.add_background_noise = False
    config.extraction.min_coverage_ratio = 0.01
    config.extraction.roi.fixed_size = None
    config.extraction.roi.min_padding = (20, 20, 20)
    config.extraction.roi.padding_ratio = (0.5, 0.5, 0.5)

    # generation settings
    config.augmentation.random_offset_enabled = True
    config.augmentation.random_offset_max_fraction = 0.8
    config.augmentation.random_offset_foreground_threshold = 0.01
    config.generation.clamp_output = False
    config.extraction.normalization = "z-score"
    config.extraction.normalization_eps = 1e-6
    config.generation.background_threshold = 0.18
    config.evaluation.foreground_threshold = 0.18
    config.generation.sampling_mode = "posterior"
    config.generation.feedback.enabled = False
    config.generation.feedback.similarity_threshold = 0.01
    config.generation.feedback.threshold_relaxation_factor = 0.9
    config.generation.variation_strength = 1.25
    config.generation.variants_per_real_anomaly = 3

    # matching settings
    config.matching.routine = "local"
    config.matching.hybrids_per_original = 3
    config.matching.reuse_synthetic_across_hybrids = True
    config.matching.allow_sibling_variants_in_same_hybrid = False
    config.matching.anomalies_per_hybrid = 2
    config.matching.max_anomalies_per_hybrid_deviation = 1

    # Fusion settings
    config.fusion.set_backend("classical")
    config.fusion.parameters.max_alpha = 1.0
    config.fusion.parameters.sq = 0.1
    config.fusion.parameters.steepness_factor = 5.0
    config.fusion.parameters.upsampling_factor = 2
    config.fusion.parameters.sobel_threshold = 0.01
    config.fusion.parameters.dilation_size = 1
    config.fusion.parameters.shave_pixels = 0
    config.fusion.parameters.fusion_use_sobel_for_alpha_mask = False
    config.fusion.parameters.fusion_variation = True
    config.fusion.parameters.alpha_variation = 0.05
    config.fusion.parameters.sq_variation = 0.1
    config.fusion.parameters.steepness_variation = 1.0
    config.fusion.parameters.selected_confidence = "90%"
    # Training settings
    config.training.validation_ratio = 0.1
    config.training.batch_size = 8
    config.training.epochs = 1000
    config.training.learning_rate = 1e-4
    config.training.gradient_clip_norm = 1.0
    config.training.log_every = None
    config.training.early_stopping_enabled = True
    config.training.early_stopping = {
        "patience": 400,
        "delta": 0.0001,
    }
    config.training.lr_scheduler_enabled = True
    config.training.lr_scheduler = {
        "patience": 200,
        "factor": 0.1,
        "threshold": 1e-5,
    }

    # Model hyperparameter search space for Optuna. The min and max dicts together define the search space.
    config.model.parameters.set_hyperparameter_space(
        # min_config
        {
            "in_channels": image_channels,
            "n_res_blocks": 2,
            "n_levels": 3,
            "z_channels": 16,
            "bottleneck_dim": 32,
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
    config.generation.variation_strength = 1.5
    config.fusion.parameters.max_alpha = 0.9
    config.fusion.parameters.sobel_threshold = 0.05
    config.extraction.roi.min_size = (256, 256)

    return config


def configure_fabric(config: Configuration) -> Configuration:
    config.extraction.roi.min_size = (128, 128)
    return config


CATEGORY_CONFIGURATORS: dict[str, Callable[[Configuration], Configuration]] = {
    "can": configure_can,
    "fabric": configure_fabric,
}


def canonical_category(category: str) -> str:
    category = category.strip().lower()
    return CATEGORY_ALIASES.get(category, category)


def safe_name(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"
