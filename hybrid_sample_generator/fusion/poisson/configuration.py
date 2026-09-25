"""Configuration model for the Poisson fusion backend."""

from dataclasses import dataclass

from hybrid_sample_generator.fusion.config_validation import validate_parameters


@dataclass(slots=True)
class Config:
    """Parameters for 2D and 3D gradient-domain Poisson blending."""

    guidance_mode: str = "source"
    solver_rtol: float = 1e-5
    solver_atol: float = 0.0
    solver_max_iterations: int = 2000
    clip_output: bool = True
    fusion_normalization_border_width: int | None = 2
    fusion_restore_anomaly_bg_relation: bool = True
    fusion_relation_mode: str = "delta"
    fusion_relation_norm_classes_separately: bool = False
    fusion_relation_min_context_size: int = 8
    fusion_keep_bg: bool = False
    fusion_bg_value: float | None = None
    fusion_relative_bg_threshold: float | None = 0.01
    fusion_bg_exterior_only: bool = True

    def validate(self) -> None:
        validate_parameters(
            self,
            positive=("solver_max_iterations", "fusion_relation_min_context_size"),
            nonnegative=(
                "solver_rtol",
                "solver_atol",
                "fusion_relative_bg_threshold",
            ),
        )
        if self.guidance_mode not in {"source", "mixed"}:
            raise ValueError(
                "fusion.parameters.guidance_mode must be 'source' or 'mixed'."
            )
        if self.solver_rtol == 0.0 and self.solver_atol == 0.0:
            raise ValueError(
                "At least one of fusion.parameters.solver_rtol and "
                "fusion.parameters.solver_atol must be positive."
            )
        if self.fusion_relation_mode not in {"delta", "ratio"}:
            raise ValueError(
                "fusion.parameters.fusion_relation_mode must be 'delta' or 'ratio'."
            )


__all__ = ["Config"]
