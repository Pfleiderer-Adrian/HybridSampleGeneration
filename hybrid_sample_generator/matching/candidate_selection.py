"""Candidate ordering, placement counts, and overlap filtering."""

import numpy as np

from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.domain.records import RealAnomaly


class LocalROISelector:
    """Continue sequential ROI assignment across controls, matching on demand."""

    def __init__(self, records: list[RealAnomaly], variants_by_real) -> None:
        self.records = records
        self.variants_by_real = variants_by_real
        self.index = 0

    def options(self, matcher, *, hybrid_index, used_synthetic_ids, reuse_synthetic):
        for _ in range(len(self.records)):
            record = self.records[self.index]
            self.index = (self.index + 1) % len(self.records)
            variants = self.variants_by_real[record.id]
            shift = hybrid_index % len(variants)
            available = [
                variant
                for variant in variants[shift:] + variants[:shift]
                if reuse_synthetic or variant.id not in used_synthetic_ids
            ]
            if not available:
                continue
            candidate = matcher.get(record)
            if candidate is not None:
                for variant in available:
                    yield candidate, variant


def variant_options(candidates, variants_by_real, *, hybrid_index: int):
    options = []
    for candidate in candidates:
        variants = variants_by_real[candidate.real_anomaly.id]
        shift = hybrid_index % len(variants)
        for synthetic in variants[shift:] + variants[:shift]:
            options.append((candidate, synthetic))
    return options


def placement_count(config: MatchingConfiguration, seed: int) -> int:
    count = int(config.anomalies_per_hybrid)
    deviation = int(config.max_anomalies_per_hybrid_deviation)
    if deviation:
        count += int(np.random.default_rng(seed).integers(-deviation, deviation + 1))
    return max(1, count)


def position_columns(position: tuple[float, ...]):
    if len(position) == 2:
        return None, float(position[0]), float(position[1])
    if len(position) == 3:
        return float(position[0]), float(position[1]), float(position[2])
    raise ValueError(f"Position must be 2D or 3D, got {position!r}")


def check_roi_overlap(center, roi_shape, used_positions) -> bool:
    for used_center, used_shape in used_positions:
        overlaps = all(
            abs(float(used_center[axis]) - float(center[axis]))
            < (float(used_shape[axis]) + float(roi_shape[axis])) / 2.0
            for axis in range(len(center))
        )
        if overlaps:
            return True
    return False


__all__ = [
    "LocalROISelector",
    "check_roi_overlap",
    "placement_count",
    "position_columns",
    "variant_options",
]
