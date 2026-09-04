from __future__ import annotations

from dataclasses import dataclass


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

    routine: str = "fixed_from_extraction_anomaly_fusion"
    hybrids_per_original: int = 1
    anomalies_per_hybrid: int = 1
    max_anomalies_per_hybrid_deviation: int = 0
    reuse_synthetic_across_hybrids: bool = True
    allow_sibling_variants_in_same_hybrid: bool = False
    batch_size: int = 64
    intensity_weight: float = 1.0
    gradient_weight: float = 2.0
    seed: int = 42

    def validate(self) -> None:
        if self.routine not in MATCHING_ROUTINES:
            raise ValueError(
                f"Unknown matching routine {self.routine!r}. Expected one of {sorted(MATCHING_ROUTINES)}."
            )
        if isinstance(self.hybrids_per_original, bool) or int(self.hybrids_per_original) <= 0:
            raise ValueError("matching.hybrids_per_original must be a positive integer.")
        if isinstance(self.anomalies_per_hybrid, bool) or int(self.anomalies_per_hybrid) <= 0:
            raise ValueError("matching.anomalies_per_hybrid must be a positive integer.")
        if int(self.max_anomalies_per_hybrid_deviation) < 0:
            raise ValueError("matching.max_anomalies_per_hybrid_deviation must be non-negative.")
        if isinstance(self.batch_size, bool) or int(self.batch_size) <= 0:
            raise ValueError("matching.batch_size must be a positive integer.")
        if float(self.intensity_weight) <= 0 and float(self.gradient_weight) <= 0:
            raise ValueError("At least one matching weight must be positive.")
