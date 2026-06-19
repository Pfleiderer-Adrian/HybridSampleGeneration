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

    config = CATEGORY_CONFIGURATORS[category](config_save_path)
    config = configure_mvtecad2_defaults(config)
    if apply_category_overrides:
        config = configure_mvtecad2_defaults(config)

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
    return config


def configure_can(config_save_path: str) -> Configuration:
    model = "LatentDiffusionLoRA_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('can_'+ model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )

    # extraction settings
    config.fixed_roi_size = (256, 256)
    config.random_offset = False
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
    config.background_threshold = 0.1
    config.prior_sampling = True
    config.use_feedback = False
    config.feedback_threshold = 0.01
    config.threshold_relaxation_factor = 0.9

    # matching settings
    config.matching_routine = "local"
    config.anomaly_duplicates = True
    config.fusions_per_control = 2
    config.max_fusions_per_control_deviation = 1

    # Fusion settings
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

    return config


def configure_fabric(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('can_'+ model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config


def configure_fruit_jelly(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('fruit_jelly_' + model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config


def configure_rice(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('rice_' + model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config


def configure_sheet_metal(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('sheet_metal_' + model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config


def configure_vial(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('vial_' + model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config


def configure_wallplugs(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('wallplugs_' + model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config
    


def configure_walnuts(config_save_path: str) -> Configuration:
    model = "VAE_ConvNeXt_2D"
    config = Configuration(
        f"mvtecad2_{safe_name('walnuts_' + model)}",
        model,
        (3, 64, 64),
        save_path=config_save_path,
    )
    return config


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
