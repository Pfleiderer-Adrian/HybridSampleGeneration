"""On-demand ROI/control pair matching with persistent cache support."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.domain.records import MatchCandidate, OriginalSample, RealAnomaly
from hybrid_sample_generator.matching.template_matching import (
    PreparedArray,
    prepare_matching_array,
    template_matching_prepared,
)
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.identifiers import stable_id
from hybrid_sample_generator.persistence.study_repository import StudyRepository

MATCHER_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    real_anomaly: RealAnomaly
    score: float | None
    position: tuple[float, ...]
    center: tuple[float, ...]
    roi_shape: tuple[int, ...]


@dataclass
class MatchingStats:
    controls: int = 0
    computed_pairs: int = 0
    cache_hits: int = 0
    preparation_seconds: float = 0.0
    matching_seconds: float = 0.0
    persistence_seconds: float = 0.0

    def log(self, total_seconds: float) -> None:
        print(
            "Matching summary: "
            f"originals={self.controls}, computed_pairs={self.computed_pairs}, "
            f"cache_hits={self.cache_hits}, prepare={self.preparation_seconds:.3f}s, "
            f"match={self.matching_seconds:.3f}s, persist={self.persistence_seconds:.3f}s, "
            f"total={total_seconds:.3f}s"
        )


def matcher_signature(config: MatchingConfiguration) -> str:
    return stable_id(
        "matcher",
        MATCHER_ALGORITHM_VERSION,
        float(config.intensity_weight),
        float(config.gradient_weight),
    )


class PairMatcher:
    """Prepare and match requested ROI/control pairs and share their cache."""

    def __init__(self, original: OriginalSample, prepared_rois: dict[str, PreparedArray],
                 roi_shapes: dict[str, tuple[int, ...]], repository: StudyRepository,
                 artifact_store: ArtifactStore, config: MatchingConfiguration,
                 stats: MatchingStats) -> None:
        self.original, self.prepared_rois, self.roi_shapes = original, prepared_rois, roi_shapes
        self.repository, self.artifact_store = repository, artifact_store
        self.config, self.stats = config, stats
        self.signature = matcher_signature(config)
        self.cached = {item.real_anomaly_id: item for item in repository.list_match_candidates(original.id, self.signature)}
        self.control_prepared = self.spatial_shape = None
        self.new_records: list[MatchCandidate] = []

    def get(self, record: RealAnomaly) -> Candidate | None:
        cached = self.cached.get(record.id)
        if cached is None:
            started = perf_counter()
            if self.control_prepared is None:
                control = self.artifact_store.load_array(self.original.image_path)
                self.spatial_shape = np.asarray(control.shape[1:], dtype=float)
                self.control_prepared = prepare_matching_array(control, with_gradient=float(self.config.gradient_weight) > 0)
            prepared = self.prepared_rois.get(record.id)
            if prepared is None:
                prepared = prepare_matching_array(self.artifact_store.load_array(record.roi_image_path), with_gradient=float(self.config.gradient_weight) > 0)
                self.prepared_rois[record.id] = prepared
            self.roi_shapes[record.id] = tuple(int(size) for size in prepared.intensity.shape)
            self.stats.preparation_seconds += perf_counter() - started
            started = perf_counter()
            score, center = template_matching_prepared(prepared, self.control_prepared, self.config)
            self.stats.matching_seconds += perf_counter() - started
            self.stats.computed_pairs += 1
            valid = center is not None and np.isfinite(score) and score >= -1
            position = tuple(float(v) for v in np.asarray(center) / self.spatial_shape) if valid else None
            cached = MatchCandidate(self.original.id, record.id, self.signature, valid,
                                    float(score) if np.isfinite(score) else None, position,
                                    tuple(float(v) for v in center) if center is not None else None,
                                    self.roi_shapes[record.id])
            self.cached[record.id] = cached
            self.new_records.append(cached)
        else:
            self.stats.cache_hits += 1
        if not cached.is_valid or cached.position is None or cached.center is None:
            return None
        return Candidate(record, cached.score, cached.position, cached.center, cached.roi_shape)

    def persist(self) -> None:
        if self.new_records:
            started = perf_counter()
            self.repository.upsert_match_candidates(self.new_records)
            self.stats.persistence_seconds += perf_counter() - started
            self.new_records.clear()


__all__ = ["Candidate", "MatchingStats", "PairMatcher", "matcher_signature"]
