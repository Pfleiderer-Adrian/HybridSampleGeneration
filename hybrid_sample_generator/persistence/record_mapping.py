"""Conversions between SQLite rows and study domain records."""

import json
import sqlite3

from hybrid_sample_generator.domain.records import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)

def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _metadata(value: str) -> dict:
    return json.loads(value) if value else {}


def _original_values(value: OriginalSample) -> tuple:
    return (
        value.id,
        value.source_name,
        value.image_path,
        value.segmentation_path,
        value.spatial_dimensions,
        int(value.has_anomaly),
        int(value.is_annotated),
        value.source_index,
        _json(value.metadata),
    )


def _hybrid_values(value: HybridSample) -> tuple:
    return (
        value.id,
        value.original_sample_id,
        value.variant_index,
        value.image_path,
        value.segmentation_path,
        value.status,
        value.error,
    )


def _placement_values(value: Placement) -> tuple:
    return (
        value.id,
        value.hybrid_sample_id,
        value.synthetic_anomaly_id,
        value.order_index,
        value.spatial_dimensions,
        value.position_z,
        value.position_y,
        value.position_x,
        value.coordinate_system,
        value.score,
        value.method,
        value.roi_image_path,
        value.roi_segmentation_path,
    )


def _match_candidate_values(value: MatchCandidate) -> tuple:
    return (
        value.original_sample_id,
        value.real_anomaly_id,
        value.matcher_signature,
        int(value.is_valid),
        value.score,
        None if value.position is None else json.dumps(value.position),
        None if value.center is None else json.dumps(value.center),
        json.dumps(value.roi_shape),
    )


def _original(row: sqlite3.Row) -> OriginalSample:
    return OriginalSample(
        id=row["id"],
        source_name=row["source_name"],
        image_path=row["image_path"],
        segmentation_path=row["segmentation_path"],
        spatial_dimensions=row["spatial_dimensions"],
        has_anomaly=bool(row["has_anomaly"]),
        is_annotated=bool(row["is_annotated"]),
        source_index=row["source_index"],
        metadata=_metadata(row["metadata_json"]),
    )


def _match_candidate(row: sqlite3.Row) -> MatchCandidate:
    return MatchCandidate(
        original_sample_id=row["original_sample_id"],
        real_anomaly_id=row["real_anomaly_id"],
        matcher_signature=row["matcher_signature"],
        is_valid=bool(row["is_valid"]),
        score=row["score"],
        position=(
            None
            if row["position_json"] is None
            else tuple(float(value) for value in json.loads(row["position_json"]))
        ),
        center=(
            None
            if row["center_json"] is None
            else tuple(float(value) for value in json.loads(row["center_json"]))
        ),
        roi_shape=tuple(int(value) for value in json.loads(row["roi_shape_json"])),
    )


def _real(row: sqlite3.Row) -> RealAnomaly:
    return RealAnomaly(
        row["id"], row["original_sample_id"], row["component_index"], row["image_path"],
        row["segmentation_path"], row["roi_image_path"], row["roi_segmentation_path"],
        row["spatial_dimensions"], row["position_z"], row["position_y"], row["position_x"],
        _metadata(row["metadata_json"]),
    )


def _synthetic(row: sqlite3.Row) -> SyntheticAnomaly:
    return SyntheticAnomaly(
        row["id"], row["real_anomaly_id"], row["variant_index"], row["image_path"],
        row["segmentation_path"], row["seed"],
    )


def _hybrid(row: sqlite3.Row) -> HybridSample:
    return HybridSample(
        row["id"], row["original_sample_id"], row["variant_index"], row["image_path"],
        row["segmentation_path"], row["status"], row["error"],
    )


def _placement(row: sqlite3.Row) -> Placement:
    return Placement(
        row["id"], row["hybrid_sample_id"], row["synthetic_anomaly_id"],
        row["order_index"], row["spatial_dimensions"], row["position_z"],
        row["position_y"], row["position_x"], row["coordinate_system"], row["score"],
        row["method"], row["roi_image_path"], row["roi_segmentation_path"],
    )
