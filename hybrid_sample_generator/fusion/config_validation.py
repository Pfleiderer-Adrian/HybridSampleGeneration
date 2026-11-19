"""Parameter declarations and validation shared by the fusion backends."""

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


# Constraints and descriptions of parameters that every backend exposes.
# Backends pick their own default: ``param(2, **COMMON["..."])``.
COMMON = {
    "fusion_normalization_border_width": dict(
        ge=-1,
        doc="Context border width for intensity normalization: None disables normalization, -1 uses the whole image, 0 uses the fallback context directly.",
    ),
    "fusion_restore_anomaly_bg_relation": dict(
        doc="Restore the intensity relationship between anomaly and background.",
    ),
    "fusion_relation_mode": dict(
        choices=("delta", "ratio"),
        doc="Intensity relationship method; use 'ratio' only if intensities are not close to 0 and have matching signs.",
    ),
    "fusion_relation_norm_classes_separately": dict(
        doc="Normalize the intensity relationship separately for each anomaly class.",
    ),
    "fusion_relation_min_context_size": dict(
        ge=1, doc="Minimum number of context pixels for the intensity relationship.",
    ),
    "fusion_keep_bg": dict(doc="Keep the synthetic anomaly background."),
    "fusion_bg_value": dict(
        doc="Explicit background value; None uses automatic detection.",
    ),
    "fusion_relative_bg_threshold": dict(
        ge=0.0, doc="Relative threshold for detecting background pixels.",
    ),
    "fusion_bg_exterior_only": dict(
        doc="Consider only background pixels connected to the exterior.",
    ),
}


def validate_fusion_parameters(config) -> None:
    """Check declared types and ranges of a backend parameter object."""
    validate_parameters(config, "fusion.parameters")


__all__ = ["COMMON", "param", "validate_fusion_parameters"]
