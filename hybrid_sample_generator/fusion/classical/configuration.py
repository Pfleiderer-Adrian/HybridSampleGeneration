"""Configuration model for the classical fusion backend."""

from dataclasses import dataclass

from hybrid_sample_generator.fusion.config_validation import validate_parameters


CONFIDENCE_LEVELS = {
    "68%": 1.0,
    "80%": 1.28,
    "90%": 1.645,
    "95%": 1.960,
    "99%": 2.576,
}


@dataclass(slots=True)
class Config:
    max_alpha: float = 0.8
    sq: float = 2
    steepness_factor: float = 3
    upsampling_factor: int = 2
    sobel_threshold: float = 0.05
    dilation_size: int = 2
    shave_pixels: int = 1
    fusion_use_sobel_for_alpha_mask: bool = False
    fusion_variation: bool = True
    alpha_variation: float = 0.05
    sq_variation: float = 0.1
    steepness_variation: float = 0.1
    selected_confidence: str = "90%"
    # Border pixels used to normalize anomaly intensity to surrounding context:
    # None skips normalization, -1 uses the entire image, >0 uses a local
    # dilation ring; 0 uses the fallback context directly. If the ring is too
    # small, normalization falls back to
    # all target-mask-outside context pixels in the local insertion patch.
    fusion_normalization_border_width: int | None = 2
    fusion_restore_anomaly_bg_relation: bool = True
    fusion_relation_mode: str = "delta"  # alternatively "ratio" if intensities are not too close to 0 and have matching signs
    fusion_relation_norm_classes_separately: bool = False
    fusion_relation_min_context_size: int = 8
    fusion_keep_bg: bool = False
    fusion_bg_value: float | None = None
    fusion_relative_bg_threshold: float | None = 0.01
    fusion_bg_exterior_only: bool = True

    def validate(self) -> None:
        validate_parameters(
            self,
            positive=("sq", "steepness_factor", "upsampling_factor", "fusion_relation_min_context_size"),
            nonnegative=("sobel_threshold", "dilation_size", "shave_pixels", "alpha_variation",
                         "sq_variation", "steepness_variation", "fusion_relative_bg_threshold"),
            unit_interval=("max_alpha",),
        )
        if self.selected_confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"Unknown fusion confidence level {self.selected_confidence!r}.")
        if self.fusion_relation_mode not in {"delta", "ratio"}:
            raise ValueError("fusion.parameters.fusion_relation_mode must be 'delta' or 'ratio'.")
