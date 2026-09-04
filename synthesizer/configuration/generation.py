from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackConfiguration:
    enabled: bool = False
    similarity_threshold: float = 0.8
    threshold_relaxation_factor: float = 0.9
    max_attempts: int = 1000

    def validate(self) -> None:
        if not 0.0 <= float(self.similarity_threshold) <= 1.0:
            raise ValueError("generation.feedback.similarity_threshold must be in [0, 1].")
        if not 0.0 < float(self.threshold_relaxation_factor) <= 1.0:
            raise ValueError("generation.feedback.threshold_relaxation_factor must be in (0, 1].")
        if isinstance(self.max_attempts, bool) or int(self.max_attempts) <= 0:
            raise ValueError("generation.feedback.max_attempts must be a positive integer.")


@dataclass
class GenerationConfiguration:
    """Settings for producing synthetic anomaly variants."""

    sampling_mode: str = "posterior"
    variation_strength: float = 1.0
    clamp_output: bool = False
    background_threshold: float = 0.01
    variants_per_real_anomaly: int = 1
    feedback: FeedbackConfiguration = field(default_factory=FeedbackConfiguration)

    def validate(self) -> None:
        if self.sampling_mode not in {"prior", "posterior"}:
            raise ValueError("generation.sampling_mode must be 'prior' or 'posterior'.")
        if float(self.variation_strength) < 0:
            raise ValueError("generation.variation_strength must be non-negative.")
        if float(self.background_threshold) < 0:
            raise ValueError("generation.background_threshold must be non-negative.")
        if isinstance(self.variants_per_real_anomaly, bool) or int(self.variants_per_real_anomaly) <= 0:
            raise ValueError("generation.variants_per_real_anomaly must be a positive integer.")
        self.feedback.validate()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GenerationConfiguration":
        values = dict(values)
        values["feedback"] = FeedbackConfiguration(**values.get("feedback", {}))
        return cls(**values)
