"""Configuration model for the classical fusion backend."""

from dataclasses import dataclass

from hybrid_sample_generator.fusion.config_validation import (
    COMMON,
    param,
    validate_fusion_parameters,
)


CONFIDENCE_LEVELS = {
    "68%": 1.0,
    "80%": 1.28,
    "90%": 1.645,
    "95%": 1.960,
    "99%": 2.576,
}


@dataclass(slots=True)
class Config:
    max_alpha: float = param(
        0.8, ge=0.0, le=1.0, doc="Maximum anomaly mixing weight.",
    )
    segmentation_min_alpha: float = param(
        0.0,
        ge=0.0,
        le=1.0,
        doc="Minimum anomaly mixing weight required for a pixel to remain in the output segmentation mask; must not exceed max_alpha.",
    )
    image_harmonization: str | None = param(
        None,
        choices=("PCTNet", "LBM"),
        doc="Optional final 2D harmonization step. Requires a separately installed libcom dependency and one or three image channels.",
    )
    sq: float = param(2, gt=0.0, doc="Shape parameter for the spatial alpha mask.")
    steepness_factor: float = param(
        3, gt=0.0, doc="Steepness of the transition at the anomaly boundary.",
    )
    upsampling_factor: int = param(
        2, ge=1, doc="Factor for finer alpha mask computation.",
    )
    sobel_threshold: float = param(
        0.05, ge=0.0, doc="Threshold for Sobel edge detection.",
    )
    dilation_size: int = param(2, ge=0, doc="Width of mask dilation.")
    shave_pixels: int = param(1, ge=0, doc="Number of boundary pixels removed.")
    fusion_use_sobel_for_alpha_mask: bool = param(
        False, doc="Use Sobel edges for the alpha mask.",
    )
    fusion_variation: bool = param(
        True, doc="Enable random variation of fusion parameters.",
    )
    alpha_variation: float = param(
        0.05, ge=0.0, doc="Variation strength for max_alpha.",
    )
    sq_variation: float = param(0.1, ge=0.0, doc="Variation strength for sq.")
    steepness_variation: float = param(
        0.1, ge=0.0, doc="Variation strength for steepness_factor.",
    )
    selected_confidence: str = param(
        "90%", choices=tuple(CONFIDENCE_LEVELS), doc="Confidence level.",
    )
    # Border pixels used to normalize anomaly intensity to surrounding context:
    # None skips normalization, -1 uses the entire image, >0 uses a local
    # dilation ring; 0 uses the fallback context directly. If the ring is too
    # small, normalization falls back to
    # all target-mask-outside context pixels in the local insertion patch.
    fusion_normalization_border_width: int | None = param(
        2, **COMMON["fusion_normalization_border_width"],
    )
    fusion_restore_anomaly_bg_relation: bool = param(
        True, **COMMON["fusion_restore_anomaly_bg_relation"],
    )
    fusion_relation_mode: str = param("delta", **COMMON["fusion_relation_mode"])
    fusion_relation_norm_classes_separately: bool = param(
        False, **COMMON["fusion_relation_norm_classes_separately"],
    )
    fusion_relation_min_context_size: int = param(
        8, **COMMON["fusion_relation_min_context_size"],
    )
    fusion_keep_bg: bool = param(False, **COMMON["fusion_keep_bg"])
    fusion_bg_value: float | None = param(None, **COMMON["fusion_bg_value"])
    fusion_relative_bg_threshold: float | None = param(
        0.01, **COMMON["fusion_relative_bg_threshold"],
    )
    fusion_bg_exterior_only: bool = param(True, **COMMON["fusion_bg_exterior_only"])

    def validate(self) -> None:
        validate_fusion_parameters(self)
        if self.segmentation_min_alpha > self.max_alpha:
            raise ValueError(
                "segmentation_min_alpha must not exceed max_alpha."
            )
