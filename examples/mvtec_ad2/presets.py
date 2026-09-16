"""Edit shared settings here, then override only category-specific differences."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

from examples.mvtec_ad2.configuration import Configuration


def apply_global_defaults(config: Configuration) -> None:
    """
    Shared MVTec AD 2 defaults for all categories.
    """
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
    config.matching.routine = "global"
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
    config.training.num_trials = 10
    config.training.trial_selection = "best"
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
    # Downstream settings share this defaults/override chain with the generator.
    config.downstream.seed = 42
    config.downstream.data.hybrid_fraction = 0.5
    config.downstream.data.normal_fraction = 0.5
    config.downstream.data.samples_per_epoch = 1000
    config.downstream.data.mode = "patch"
    config.downstream.data.patch_size = (512, 512)
    config.downstream.data.patch_overlap = 0.5
    config.downstream.data.texture_root = None
    config.downstream.data.image_scale = 255.0
    config.downstream.training.epochs = 100
    config.downstream.training.batch_size = 8
    config.downstream.training.learning_rate = 1e-4
    config.downstream.training.num_workers = 0
    config.downstream.training.device = "auto"
    config.downstream.training.reconstruction_width = 128
    config.downstream.training.segmentation_width = 64


def configure_can(config: Configuration) -> None:
    config.generation.variation_strength = 1.5
    config.fusion.parameters.max_alpha = 0.9
    config.fusion.parameters.sobel_threshold = 0.05
    config.extraction.roi.min_size = (256, 256)


def configure_fabric(config: Configuration) -> None:
    config.extraction.roi.min_size = (128, 128)


@dataclass(frozen=True)
class CategoryPreset:
    anomaly_shape: tuple[int, int, int] = (3, 64, 64)
    generation_model: str = "cVAE_ConvNeXt_2D"
    configure: Callable[[Configuration], None] | None = None


CATEGORY_PRESETS = {
    "can": CategoryPreset(configure=configure_can),
    "fabric": CategoryPreset(configure=configure_fabric),
    "fruit_jelly": CategoryPreset(),
    "rice": CategoryPreset(),
    "sheet_metal": CategoryPreset((1, 64, 64)),
    "vial": CategoryPreset((1, 64, 64)),
    "wallplugs": CategoryPreset((1, 64, 64)),
    "walnuts": CategoryPreset(),
}
MVTECAD2_CATEGORIES = tuple(CATEGORY_PRESETS)
CATEGORY_ALIASES = {"wall_plugs": "wallplugs", "wall plugs": "wallplugs"}


def canonical_category(category: str) -> str:
    category = category.strip().lower()
    return CATEGORY_ALIASES.get(category, category)


def safe_name(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"


def create_configuration(
    category: str, *, save_path: Path | str | None = None,
    apply_category_overrides: bool = True,
) -> Configuration:
    """Build an independent config: library defaults, shared defaults, category."""
    category = canonical_category(category)
    if category not in CATEGORY_PRESETS:
        raise ValueError(f"Unknown MVTec AD 2 category: {category!r}")
    preset = CATEGORY_PRESETS[category]
    config = Configuration(f"mvtecad2_{safe_name(f'{category}_{preset.generation_model}')}", save_path=save_path)
    config.extraction.anomaly_size = preset.anomaly_shape
    config.model.set_model(preset.generation_model)
    apply_global_defaults(config)
    if apply_category_overrides and preset.configure is not None:
        preset.configure(config)
    config.validate()
    return config


def apply_downstream_preset(config: Configuration, category: str) -> None:
    """Explicitly replace only downstream settings with the current category preset."""
    config.downstream = create_configuration(category).downstream
