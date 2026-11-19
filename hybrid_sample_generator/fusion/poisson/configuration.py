"""Configuration model for the Poisson fusion backend."""

from dataclasses import dataclass

from hybrid_sample_generator.fusion.config_validation import (
    COMMON,
    param,
    validate_fusion_parameters,
)


@dataclass(slots=True)
class Config:
    """Parameters for 2D and 3D gradient-domain Poisson blending."""

    guidance_mode: str = param(
        "source",
        choices=("source", "mixed"),
        doc="Gradient guidance: 'source' preserves anomaly gradients; 'mixed' selects the stronger source or target gradient.",
    )
    solver_rtol: float = param(
        1e-5,
        ge=0.0,
        doc="Relative convergence tolerance for the conjugate-gradient solver.",
    )
    solver_atol: float = param(
        0.0,
        ge=0.0,
        doc="Absolute convergence tolerance for the conjugate-gradient solver.",
    )
    solver_max_iterations: int = param(
        2000, ge=1, doc="Maximum conjugate-gradient iterations per image channel.",
    )
    clip_output: bool = param(
        True,
        doc="Clip results to recognized control-image ranges such as [0, 1] or [0, 255].",
    )
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
        if self.solver_rtol == 0.0 and self.solver_atol == 0.0:
            raise ValueError(
                "At least one of fusion.parameters.solver_rtol and "
                "fusion.parameters.solver_atol must be positive."
            )


__all__ = ["Config"]
