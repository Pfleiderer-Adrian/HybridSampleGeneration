"""Generate and verify the searchable configuration reference from runtime defaults."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.fusion.registry import FUSION_BACKEND_REGISTRY
from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "configuration" / "reference"

# Explanations are reviewed prose. Field names, types and defaults come from code.
DESCRIPTIONS = {
    "study.name": "Study name supplied when creating Configuration.",
    "study.folder": "Directory for the database, models, and generated artifacts; defaults to results/<study name> in the working directory.",
    "study.seed": "Seed for reproducible random decisions.",
    "extraction.anomaly_size": "Target anomaly crop shape (C,H,W) or (C,D,H,W); must match the model dimensions.",
    "extraction.separate_components": "Extract each connected anomaly component separately.",
    "extraction.min_coverage_ratio": "Minimum mask coverage within a crop; range [0, 1].",
    "extraction.add_background_noise": "Add a small amount of noise to otherwise constant backgrounds.",
    "extraction.preserve_aspect_ratio": "Use one spatial scale for every axis and center-pad the remaining area; False stretches axes independently.",
    "extraction.normalization": "Intensity normalization method; defaults to z-score.",
    "extraction.normalization_eps": "Positive lower bound for numerically stable normalization.",
    "extraction.roi.fixed_size": "Fixed spatial ROI size; None selects dynamic sizing.",
    "extraction.roi.min_size": "Minimum dynamic ROI size, either scalar or per axis.",
    "extraction.roi.min_padding": "Minimum ROI padding around the anomaly on each axis.",
    "extraction.roi.padding_ratio": "Additional ROI padding relative to anomaly size on each axis.",
    "augmentation.mask_transforms.use_mask_transform": "Enable the default mask transform probabilities.",
    "augmentation.mask_transforms.mask_transform_probs": "Probabilities for global, local, and class-specific transforms.",
    "augmentation.mask_transforms.mask_transform_params": "Parameter ranges for individual transforms.",
    "augmentation.mask_transforms.priorities": "Class priority when transformed masks overlap.",
    "augmentation.mask_transforms.local_as_global": "Apply local transforms jointly to all anomaly classes.",
    "augmentation.mask_transforms.padding_factor": "Scale factor for the temporary transform canvas.",
    "augmentation.random_offset_enabled": "Enable random offsets during training.",
    "augmentation.random_offset_max_fraction": "Maximum offset as a fraction of available space; range [0, 1].",
    "augmentation.random_offset_foreground_threshold": "Foreground threshold used for training offsets.",
    "generation.sampling_mode": "Latent sampling source: 'posterior' or 'prior'.",
    "generation.posterior_skip_source": "Encoder skip source for models supporting transformed posterior skips: 'original' uses the input image; 'transformed' uses an image paired with the generated target mask. Ignored by other models and in prior mode.",
    "generation.variation_strength": "Strength of random variation during generation; non-negative.",
    "generation.clamp_output": "Clamp generated image values to [0, 1].",
    "generation.background_threshold": "Relative threshold for deriving a mask from generated images.",
    "generation.variants_per_real_anomaly": "Number of synthetic variants per real anomaly; positive.",
    "generation.feedback.enabled": "Enable image similarity feedback and repeated generation attempts.",
    "generation.feedback.similarity_threshold": "Minimum similarity for accepting a variant; range [0, 1].",
    "generation.feedback.threshold_relaxation_factor": "Factor for relaxing the threshold; range (0, 1].",
    "generation.feedback.max_attempts": "Maximum number of generation attempts; positive.",
    "matching.routine": "Candidate selection and placement method; see the matching guide.",
    "matching.hybrids_per_original": "Number of hybrid images per eligible original; positive.",
    "matching.anomalies_per_hybrid": "Target number of placed anomalies per hybrid; positive.",
    "matching.max_anomalies_per_hybrid_deviation": "Allowed random deviation from the target placement count; non-negative.",
    "matching.reuse_synthetic_across_hybrids": "Allow a synthetic variant to appear in multiple hybrids.",
    "matching.allow_sibling_variants_in_same_hybrid": "Allow variants of one real anomaly in the same hybrid.",
    "matching.batch_size": "Batch size for candidate scoring; positive.",
    "matching.intensity_weight": "Weight of intensity similarity during matching.",
    "matching.gradient_weight": "Weight of gradient similarity during matching.",
    "training.num_trials": "Number of Optuna trials; positive.",
    "training.trial_selection": "Trial to use: 'best', 'last', or a non-negative trial ID.",
    "training.validation_ratio": "Fraction of data reserved for validation; range [0, 1).",
    "training.batch_size": "Training batch size; positive.",
    "training.epochs": "Maximum number of training epochs; positive.",
    "training.learning_rate": "Initial learning rate; positive.",
    "training.dtype": "PyTorch dtype for training; None uses the model default.",
    "training.gradient_clip_norm": "Optional upper bound for the gradient norm.",
    "training.monitor_metric": "Metric used for model selection and early stopping.",
    "training.early_stopping_enabled": "Enable early stopping.",
    "training.early_stopping": "Early stopping settings, including patience and delta.",
    "training.lr_scheduler_enabled": "Enable the learning rate scheduler.",
    "training.lr_scheduler": "Learning rate scheduler settings, including patience and factor.",
    "evaluation.foreground_threshold": "Foreground threshold used during evaluation; None disables it.",
    "evaluation.outlier_thresholds": "Minimum and maximum limits per evaluation metric; None leaves a limit open.",
}

MODEL_DESCRIPTIONS = {
    "n_res_blocks": "Number of residual blocks per level.",
    "n_spade_blocks": "Number of mask-conditioned SPADE blocks.",
    "n_levels": "Number of encoder and decoder levels.",
    "z_channels": "Channel count in the spatial bottleneck.",
    "bottleneck_dim": "Dimension of the latent vector.",
    "use_multires_skips": "Use encoder features from multiple resolutions as skip connections.",
    "recon_weight": "Weight of the reconstruction loss.",
    "latent_recon_weight": "Weight of the latent reconstruction Smooth L1 loss; 0 disables the loss.",
    "latent_recon_noise_scale": "Scale of Gaussian noise added to the detached latent mean for the reconstruction cycle; positive.",
    "latent_recon_image_noise_std": "Standard deviation of Gaussian noise added to the cycle image before re-encoding during training; non-negative.",
    "beta_kl_start": "Initial weight of the KL loss.",
    "beta_kl_max": "Maximum weight of the KL loss.",
    "beta_kl_warmup_start": "Epoch at which the KL weight begins to increase.",
    "beta_kl_warmup_epochs": "Number of epochs needed to reach beta_kl_max.",
    "free_bits": "KL free-bits allowance for latent dimensions.",
    "recon_loss": "Reconstruction loss, such as 'mse' or 'smoothl1'.",
    "recon_smoothl1_beta": "Transition point of the Smooth L1 loss.",
    "use_transpose_conv": "Use transposed convolutions for upsampling.",
    "foreground_weight": "Relative weight of the mask foreground mean in the reconstruction loss; non-negative.",
    "background_weight": "Relative weight of the mask background mean in the reconstruction loss; non-negative.",
    "drop_path_rate": "Stochastic depth rate in ConvNeXt blocks.",
    "dropout": "Dropout probability within the model.",
    "skip_dropout_p": "Shared dropout probability for skip connections.",
    "skip_dropout_ps": "Dropout per skip level; overrides skip_dropout_p and requires n_levels values.",
    "skip_alpha": "Shared skip connection scale; range [0, 1].",
    "skip_alphas": "Scale per skip level; overrides skip_alpha and requires n_levels values.",
    "identity_pair_probability": (
        "Probability of using an unchanged source/target pair during paired "
        "source-to-target training; range [0, 1]."
    ),
}

FUSION_DESCRIPTIONS = {
    "max_alpha": "Maximum anomaly mixing weight; range [0, 1].",
    "sq": "Shape parameter for the spatial alpha mask.",
    "steepness_factor": "Steepness of the transition at the anomaly boundary.",
    "upsampling_factor": "Factor for finer alpha mask computation.",
    "sobel_threshold": "Threshold for Sobel edge detection.",
    "dilation_size": "Width of mask dilation.",
    "shave_pixels": "Number of boundary pixels removed.",
    "fusion_use_sobel_for_alpha_mask": "Use Sobel edges for the alpha mask.",
    "fusion_variation": "Enable random variation of fusion parameters.",
    "alpha_variation": "Variation strength for max_alpha.",
    "sq_variation": "Variation strength for sq.",
    "steepness_variation": "Variation strength for steepness_factor.",
    "selected_confidence": "Confidence level: 68%, 80%, 90%, 95%, or 99%.",
    "fusion_normalization_border_width": "Context border width: None disables normalization; -1 uses the whole image.",
    "fusion_restore_anomaly_bg_relation": "Restore the intensity relationship between anomaly and background.",
    "fusion_relation_mode": "Intensity relationship method: 'delta' or 'ratio'.",
    "fusion_relation_norm_classes_separately": "Normalize the intensity relationship separately for each anomaly class.",
    "fusion_relation_min_context_size": "Minimum number of context pixels for the intensity relationship.",
    "fusion_keep_bg": "Keep the synthetic anomaly background.",
    "fusion_bg_value": "Explicit background value; None uses automatic detection.",
    "fusion_relative_bg_threshold": "Relative threshold for detecting background pixels.",
    "fusion_bg_exterior_only": "Consider only background pixels connected to the exterior.",
    "guidance_mode": "Gradient guidance: 'source' preserves anomaly gradients; 'mixed' selects the stronger source or target gradient.",
    "solver_rtol": "Relative convergence tolerance for the conjugate-gradient solver.",
    "solver_atol": "Absolute convergence tolerance for the conjugate-gradient solver.",
    "solver_max_iterations": "Maximum conjugate-gradient iterations per image channel.",
    "clip_output": "Clip results to recognized control-image ranges such as [0, 1] or [0, 255].",
}


# study.name is required; study.folder has a derived default.
REQUIRED = {"study.name"}


def _display(value) -> str:
    if isinstance(value, dict) and len(value) > 5:
        return f"{len(value)} metrics with min=None and max=None"
    return repr(value).replace("|", "\\|").replace("\n", " ")


def _rows(instance, prefix: str, descriptions: dict[str, str], *, required=frozenset()):
    rows = []
    for item in fields(instance):
        path = f"{prefix}.{item.name}"
        value = getattr(instance, item.name)
        if is_dataclass(value) and not isinstance(value, type):
            rows.extend(_rows(value, path, descriptions, required=required))
            continue
        if path not in descriptions:
            raise ValueError(f"Missing configuration description: {path}")
        if path in required:
            default = "required at construction"
        elif path == "study.folder":
            default = "results/<study name>"
        else:
            default = _display(value)
        kind = item.type.__name__ if isinstance(item.type, type) else str(item.type)
        rows.append((path, kind.replace("|", "\\|"), default, descriptions[path]))
    return rows


def _table(rows, *, search=None):
    headers = ["Parameter", "Type", "Default", "Meaning / values"]
    if search is not None:
        headers.append("Default search")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for path, kind, default, description in rows:
        cells = [f"`config.{path}`", f"`{kind}`", f"`{default}`", description]
        if search is not None:
            name = path.rsplit(".", 1)[1]
            distribution = getattr(search, "_distributions", {}).get(name)
            cells.append(f"`{distribution!r}`" if distribution is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _validate_descriptions(used: set[str], descriptions: dict[str, str], label: str):
    unused = set(descriptions) - used
    if unused:
        raise ValueError(f"Unused {label} descriptions: {sorted(unused)}")


def _render() -> dict[Path, str]:
    config = Configuration("example", study_folder="results/example")
    pages = {}
    sections = {
        "study": ("Study", config.study),
        "extraction": ("Extraction", config.extraction),
        "augmentation": ("Augmentation", config.augmentation),
        "generation": ("Generation", config.generation),
        "matching": ("Matching", config.matching),
        "training": ("Training", config.training),
        "evaluation": ("Evaluation", config.evaluation),
    }
    root_sections = set(vars(config)) - {"schema_version", "model", "fusion"}
    if set(sections) != root_sections:
        raise ValueError(
            f"Undocumented root sections: {sorted(root_sections - set(sections))}; "
            f"obsolete sections: {sorted(set(sections) - root_sections)}"
        )
    used = set()
    for slug, (title, instance) in sections.items():
        rows = _rows(instance, slug, DESCRIPTIONS, required=REQUIRED)
        used.update(row[0] for row in rows)
        extra = ""
        if slug == "matching":
            extra = "\nAllowed `matching.routine` values: `local`, `global`, `batchwise`, `fixed_from_extraction_anomaly_fusion`, and `fixed_from_extraction_control_fusion`.\n"
        if slug == "augmentation":
            generator = TransformGenerator.from_config(
                config.augmentation.mask_transforms,
                anomaly_size=config.extraction.anomaly_size,
                background_threshold=config.generation.background_threshold,
                seed=0,
            )
            transform_lines = [
                "| Transform | Default probability | Effective default parameters |",
                "| --- | --- | --- |",
            ]
            for name in generator.transform_params:
                probabilities = (
                    generator.local_transform_probs if name.startswith("local_")
                    else generator.global_transform_probs
                )
                transform_lines.append(
                    f"| `{name}` | `{probabilities.get(name, 0)}` | "
                    f"`{_display(generator.transform_params[name])}` |"
                )
            extra = (
                "\nMasks use nearest-neighbor interpolation; jointly transformed images use "
                "linear interpolation. `mask_transform_probs` and `mask_transform_params` "
                "accept global or local transform names and class IDs. With "
                "`use_mask_transform=True`, the default anomaly size "
                f"`{config.extraction.anomaly_size}` gives these effective values; "
                "global elastic parameters change with `anomaly_size`.\n\n"
                "## Transform defaults\n\n"
                + "\n".join(transform_lines) + "\n"
            )
        if slug == "training":
            extra = (
                "\n## Nested training settings\n\n"
                "`config.training.early_stopping` contains `patience` (epochs without "
                "improvement) and `delta` (minimum improvement). "
                "`config.training.lr_scheduler` contains `patience`, `factor` "
                "(learning rate multiplier), and `threshold` (minimum improvement). "
                "Their defaults are shown in the table above.\n"
            )
        if slug == "evaluation":
            extra = "\n## Defaults\n\n`outlier_thresholds` contains `min: None` and `max: None` for Contrast, Homogeneity, Energy, Correlation, their four `roi_` variants, Volume, D-center, H-center, and W-center.\n"
        pages[OUTPUT / f"{slug}.md"] = f"# {title}\n\n{_table(rows)}\n{extra}\n"
    _validate_descriptions(used, DESCRIPTIONS, "general")

    used_model = set()
    for name, spec in sorted(MODEL_REGISTRY.items()):
        parameters = spec.build_configuration()
        rows = _rows(parameters, "model.parameters", {
            f"model.parameters.{key}": value for key, value in MODEL_DESCRIPTIONS.items()
        })
        used_model.update(row[0].rsplit(".", 1)[1] for row in rows)
        search = spec.build_search_space(parameters)
        page = (
            f"# {name}\n\n"
            f"Select with `config.model.set_model({name!r})`. The table shows the "
            "actual defaults for this model variant. A dash in Default search means "
            "the parameter stays fixed unless you assign a compatible distribution "
            "through `config.model.search`.\n\n"
            f"{_table(rows, search=search)}\n"
        )
        pages[OUTPUT / f"model-{name}.md"] = page
    _validate_descriptions(used_model, MODEL_DESCRIPTIONS, "model")

    used_fusion = set()
    for name, spec in sorted(FUSION_BACKEND_REGISTRY.items()):
        rows = _rows(spec.build_configuration(), "fusion.parameters", {
            f"fusion.parameters.{key}": value for key, value in FUSION_DESCRIPTIONS.items()
        })
        used_fusion.update(row[0].rsplit(".", 1)[1] for row in rows)
        pages[OUTPUT / f"fusion-{name}.md"] = (
            f"# Fusion: {name}\n\n"
            f"Select with `config.fusion.set_backend({name!r})`.\n\n"
            f"{_table(rows)}\n"
        )
    _validate_descriptions(used_fusion, FUSION_DESCRIPTIONS, "fusion")
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated pages differ")
    args = parser.parse_args()
    pages = _render()
    existing = set(OUTPUT.glob("*.md"))
    stale = existing - set(pages)
    changed = [path for path, content in pages.items() if not path.exists() or path.read_text() != content]
    if args.check:
        if changed or stale:
            for path in sorted(changed + list(stale)):
                print(f"Outdated configuration reference: {path.relative_to(ROOT)}")
            return 1
        print(f"Configuration reference current: {len(pages)} pages")
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in pages.items():
        path.write_text(content)
    for path in stale:
        path.unlink()
    print(f"Generated {len(pages)} configuration reference pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
