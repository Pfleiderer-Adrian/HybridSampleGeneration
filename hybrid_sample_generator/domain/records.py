from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OriginalSample:
    id: str
    source_name: str
    image_path: str
    segmentation_path: str | None
    spatial_dimensions: int
    has_anomaly: bool
    is_annotated: bool
    source_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchCandidate:
    """Cached result for one original/real-anomaly matching pair."""

    original_sample_id: str
    real_anomaly_id: str
    matcher_signature: str
    is_valid: bool
    score: float | None
    position: tuple[float, ...] | None
    center: tuple[float, ...] | None
    roi_shape: tuple[int, ...]


@dataclass(frozen=True)
class RealAnomaly:
    id: str
    original_sample_id: str
    component_index: int
    image_path: str
    segmentation_path: str
    roi_image_path: str
    roi_segmentation_path: str
    spatial_dimensions: int
    position_z: float | None
    position_y: float
    position_x: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_position(self) -> tuple[float, ...]:
        if self.spatial_dimensions == 2:
            return self.position_y, self.position_x
        return self.position_z, self.position_y, self.position_x


@dataclass(frozen=True)
class SyntheticAnomaly:
    id: str
    real_anomaly_id: str
    variant_index: int
    image_path: str
    segmentation_path: str
    seed: int


@dataclass(frozen=True)
class HybridSample:
    id: str
    original_sample_id: str
    variant_index: int
    image_path: str | None = None
    segmentation_path: str | None = None
    status: str = "planned"
    error: str | None = None


@dataclass(frozen=True)
class Placement:
    id: str
    hybrid_sample_id: str
    synthetic_anomaly_id: str
    order_index: int
    spatial_dimensions: int
    position_z: float | None
    position_y: float
    position_x: float
    coordinate_system: str = "normalized_center"
    score: float | None = None
    method: str = "global"
    roi_image_path: str | None = None
    roi_segmentation_path: str | None = None

    @property
    def position(self) -> tuple[float, ...]:
        if self.spatial_dimensions == 2:
            return self.position_y, self.position_x
        return self.position_z, self.position_y, self.position_x


@dataclass(frozen=True)
class StudyHierarchyEntry:
    original: OriginalSample
    hybrid: HybridSample
    placement: Placement
    synthetic_anomaly: SyntheticAnomaly
    real_anomaly: RealAnomaly
