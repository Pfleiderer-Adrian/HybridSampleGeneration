"""Configuration models and defaults for study evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


def _default_outlier_thresholds() -> dict[str, dict[str, float | None]]:
    names = (
        "Contrast",
        "Homogeneity",
        "Energy",
        "Correlation",
        "roi_Contrast",
        "roi_Homogeneity",
        "roi_Energy",
        "roi_Correlation",
        "Volume",
        "D-center",
        "H-center",
        "W-center",
    )
    return {name: {"min": None, "max": None} for name in names}


@dataclass
class EvaluationConfiguration:
    """Thresholds and policies used by post-generation evaluation."""

    foreground_threshold: float | None = param(
        0.01,
        ge=0.0,
        doc="Foreground threshold used during evaluation; None disables it.",
    )
    outlier_thresholds: dict[str, dict[str, float | None]] = param(
        default_factory=_default_outlier_thresholds,
        doc="Minimum and maximum limits per evaluation metric; None leaves a limit open.",
    )

    def validate(self) -> None:
        validate_parameters(self, "evaluation")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EvaluationConfiguration":
        return cls(**values)
