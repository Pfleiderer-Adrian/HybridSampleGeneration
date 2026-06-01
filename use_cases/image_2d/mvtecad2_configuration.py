"""Category-specific MVTec AD 2 configuration presets."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
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

DEFAULT_MODEL_NAME = "VAE_ConvNeXt_2D"
DEFAULT_CHANNELS = 3
DEFAULT_ANOMALY_PATCH_SIZE = 64
DEFAULT_FIXED_ROI_SIZE = (64, 64)


def create_mvtecad2_configuration(
    category: str,
    *,
    channels: int = DEFAULT_CHANNELS,
    model_name: str = DEFAULT_MODEL_NAME,
    anomaly_patch_size: int = DEFAULT_ANOMALY_PATCH_SIZE,
    fixed_roi_size: tuple[int, int] | None = DEFAULT_FIXED_ROI_SIZE,
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
    config = Configuration(
        f"mvtecad2_{safe_name(category)}",
        model_name,
        (int(channels), int(anomaly_patch_size), int(anomaly_patch_size)),
        save_path=None if config_save_path is None else str(config_save_path),
    )
    config.config_name = category

    configure_mvtecad2_defaults(config, fixed_roi_size=fixed_roi_size)
    if apply_category_overrides:
        CATEGORY_CONFIGURATORS[category](config)

    return config


def create_mvtecad2_configurations(
    categories: Iterable[str] | None = None,
    *,
    channels: int = DEFAULT_CHANNELS,
    channels_by_category: Mapping[str, int] | None = None,
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
        category_channels = channels
        if channels_by_category is not None:
            category_channels = int(channels_by_category.get(category, channels))

        category_save_path = None
        if base_save_path is not None:
            category_save_path = Path(base_save_path) / category
            category_save_path.mkdir(parents=True, exist_ok=True)

        configs[category] = create_mvtecad2_configuration(
            category,
            channels=category_channels,
            save_path=category_save_path,
            **kwargs,
        )
    return configs


def configure_mvtecad2_defaults(
    config: Configuration,
    *,
    fixed_roi_size: tuple[int, int] | None,
) -> None:
    """
    Shared MVTec AD 2 defaults for all categories.
    """

    config.random_offset = True
    config.random_offset_max_fraction = 0.8
    config.random_offset_foreground_threshold_rel = 0.01
    config.add_bg_noise = True
    config.prior_sampling = True
    config.min_anomaly_percentage = 0.01
    config.min_pad = (20, 20, 20)

    config.pad_ratio = (0.5, 0.5, 0.5)
    config.clamp01_output = False
    config.normalization = "z-score"
    config.normalization_eps = 1e-6
    config.background_threshold = None

    config.use_feedback = True
    config.feedback_threshold = 0.01
    config.threshold_relaxation_factor = 0.9
    config.prior_sampling = False

    config.matching_routine = "local"
    config.anomaly_duplicates = False
    config.fusions_per_control = 1
    config.max_fusions_per_control_deviation = 0

    config.fixed_roi_size = fixed_roi_size
    config.update_fusion_params(
        max_alpha=0.8,
        sq=2,
        steepness_factor=3,
        upsampling_factor=2,
        sobel_threshold=0.05,
        dilation_size=2,
        shave_pixels=1,
    )
    config.fusion_variation = True
    config.fusion_variation_params = {
        "alpha_variation": 0.05,
        "sq_variation": 0.1,
        "steepness_variation": 0.1,
    }
    config.selected_confidence = "90%"
    config.confidence_z_score = config.confidence_levels[config.selected_confidence]

    config.val_ratio = 0.1
    config.batch_size = 8
    config.epochs = 1000
    config.lr = 1e-3
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

    config.update_model_param_ranges(
        {
            "n_res_blocks": (3, 5),
            "n_levels": (3, 5),
            "z_channels": (16, 64),
            "bottleneck_dim": (16, 128),
            "recon_weight": (5.0, 100.0),
            "beta_kl": (0.1, 0.5),
        }
    )
    config.set_model_params(
        {
            "use_multires_skips": False,
            "use_transpose_conv": False,
        }
    )


def configure_can(config: Configuration) -> None:
    pass


def configure_fabric(config: Configuration) -> None:
    pass


def configure_fruit_jelly(config: Configuration) -> None:
    pass


def configure_rice(config: Configuration) -> None:
    pass


def configure_sheet_metal(config: Configuration) -> None:
    pass


def configure_vial(config: Configuration) -> None:
    pass


def configure_wallplugs(config: Configuration) -> None:
    pass


def configure_walnuts(config: Configuration) -> None:
    pass


CATEGORY_CONFIGURATORS: dict[str, Callable[[Configuration], None]] = {
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
