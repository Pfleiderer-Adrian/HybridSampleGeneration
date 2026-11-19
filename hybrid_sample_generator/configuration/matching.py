"""Configuration models for anomaly matching and placement."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


MATCHING_ROUTINES = {
    "local",
    "global",
    "batchwise",
    "fixed_from_extraction_anomaly_fusion",
    "fixed_from_extraction_control_fusion",
}


@dataclass
class MatchingConfiguration:
    """Candidate selection and placement settings for matching."""

    routine: str = param(
        "batchwise",
        choices=MATCHING_ROUTINES,
        doc="Candidate selection and placement method; see the matching guide.",
    )
    hybrids_per_original: int = param(
        1, ge=1, doc="Number of hybrid images per eligible original.",
    )
    anomalies_per_hybrid: int = param(
        1, ge=1, doc="Target number of placed anomalies per hybrid.",
    )
    max_anomalies_per_hybrid_deviation: int = param(
        0, ge=0, doc="Allowed random deviation from the target placement count.",
    )
    reuse_synthetic_across_hybrids: bool = param(
        True, doc="Allow a synthetic variant to appear in multiple hybrids.",
    )
    allow_sibling_variants_in_same_hybrid: bool = param(
        False, doc="Allow variants of one real anomaly in the same hybrid.",
    )
    batch_size: int = param(32, ge=1, doc="Batch size for candidate scoring.")
    intensity_weight: float = param(
        1.0, doc="Weight of intensity similarity during matching.",
    )
    gradient_weight: float = param(
        2.0, doc="Weight of gradient similarity during matching.",
    )
    distance_weight: float = param(
        0.0,
        ge=0.0,
        doc="Weight of proximity to the original anomaly's normalized source position. Zero preserves intensity/gradient-only matching.",
    )

    def validate(self) -> None:
        validate_parameters(self, "matching")
        if self.intensity_weight <= 0 and self.gradient_weight <= 0 and self.distance_weight <= 0:
            raise ValueError("At least one matching weight must be positive.")
