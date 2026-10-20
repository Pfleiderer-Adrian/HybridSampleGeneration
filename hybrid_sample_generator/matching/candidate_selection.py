"""Candidate ordering, placement counts, and overlap filtering."""

import numpy as np

from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.domain.records import RealAnomaly
from hybrid_sample_generator.persistence.identifiers import stable_seed


def balanced_variants(variants, *, usage_counts, seed, original_id, real_id, hybrid_index):
    """Prefer least-used siblings, breaking ties reproducibly for each target."""
    ordered = sorted(variants, key=lambda variant: (variant.variant_index, variant.id))
    rng = np.random.default_rng(
        stable_seed(seed, original_id, real_id, hybrid_index, "variant_selection")
    )
    rng.shuffle(ordered)
    ordered.sort(key=lambda variant: usage_counts.get(variant.id, 0))
    return ordered


class LocalROISelector:
    """Continue sequential ROI assignment across controls, matching on demand."""

    def __init__(self, records: list[RealAnomaly], variants_by_real) -> None:
        self.records = records
        self.variants_by_real = variants_by_real
        self.index = 0

    def options(self, matcher, *, hybrid_index, used_synthetic_ids, reuse_synthetic,
                usage_counts, seed, original_id):
        for _ in range(len(self.records)):
            record = self.records[self.index]
            self.index = (self.index + 1) % len(self.records)
            variants = balanced_variants(
                self.variants_by_real[record.id], usage_counts=usage_counts,
                seed=seed, original_id=original_id, real_id=record.id,
                hybrid_index=hybrid_index,
            )
            available = [
                variant
                for variant in variants
                if reuse_synthetic or variant.id not in used_synthetic_ids
            ]
            if not available:
                continue
            candidate = matcher.get(record)
            if candidate is not None:
                for variant in available:
                    yield candidate, variant


def variant_options(candidates, variants_by_real, *, hybrid_index: int,
                    usage_counts, seed, original_id, source_usage=None):
    # Reconsider priority after every yield: the planner increments usage only
    # when it accepts a placement, including within multi-placement hybrids.
    pending = []
    for candidate in candidates:
        real_id = candidate.real_anomaly.id
        variants = balanced_variants(
            variants_by_real[real_id], usage_counts=usage_counts, seed=seed,
            original_id=original_id, real_id=real_id, hybrid_index=hybrid_index,
        )
        pending.extend((candidate, synthetic) for synthetic in variants)
    while pending:
        index = 0
        if source_usage is not None:
            index = min(
                range(len(pending)),
                key=lambda i: (
                    source_usage[pending[i][0].real_anomaly.id],
                    -float(pending[i][0].score),
                    pending[i][0].real_anomaly.id,
                ),
            )
        yield pending.pop(index)


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
