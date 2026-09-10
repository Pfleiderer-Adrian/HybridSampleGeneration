from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    foreground_threshold: float | None = 0.01
    outlier_thresholds: dict[str, dict[str, float | None]] = field(
        default_factory=_default_outlier_thresholds
    )

    def validate(self) -> None:
        if self.foreground_threshold is not None and float(self.foreground_threshold) < 0:
            raise ValueError("evaluation.foreground_threshold must be non-negative or None.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EvaluationConfiguration":
        return cls(**values)
