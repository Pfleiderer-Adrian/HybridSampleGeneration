"""Configuration models for synthetic anomaly generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


@dataclass
class FeedbackConfiguration:
    enabled: bool = param(
        False, doc="Enable image similarity feedback and repeated generation attempts.",
    )
    similarity_threshold: float = param(
        0.8, ge=0.0, le=1.0, doc="Minimum similarity for accepting a variant.",
    )
    threshold_relaxation_factor: float = param(
        0.9, gt=0.0, le=1.0, doc="Factor for relaxing the threshold.",
    )
    max_attempts: int = param(100, ge=1, doc="Maximum number of generation attempts.")

    def validate(self) -> None:
        validate_parameters(self, "generation.feedback")


@dataclass
class GenerationConfiguration:
    """Settings for producing synthetic anomaly variants."""

    sampling_mode: str = param(
        "posterior", choices=("posterior", "prior"), doc="Latent sampling source.",
    )
    posterior_skip_source: str = param(
        "transformed",
        choices=("original", "transformed"),
        doc="Encoder skip source for models supporting transformed posterior skips: 'original' uses the input image; 'transformed' uses an image paired with the generated target mask. Ignored by other models and in prior mode.",
    )
    variation_strength: float = param(
        0.5, ge=0.0, doc="Strength of random variation during generation.",
    )
    clamp_output: bool = param(False, doc="Clamp generated image values to [0, 1].")
    background_threshold: float = param(
        0.01, ge=0.0, doc="Relative threshold for deriving a mask from generated images.",
    )
    variants_per_real_anomaly: int = param(
        3, ge=1, doc="Number of synthetic variants per real anomaly.",
    )
    feedback: FeedbackConfiguration = param(default_factory=FeedbackConfiguration)

    def validate(self) -> None:
        validate_parameters(self, "generation")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GenerationConfiguration":
        values = dict(values)
        values["feedback"] = FeedbackConfiguration(**values.get("feedback", {}))
        return cls(**values)
