"""Mutation operations for the SQLite study repository."""

from hybrid_sample_generator.domain.records import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)
from hybrid_sample_generator.persistence.record_mapping import (
    _hybrid_values,
    _json,
    _match_candidate_values,
    _original_values,
    _placement_values,
)


class StudyWriterMixin:
    def clear_real_anomalies_and_downstream(self) -> None:
        """Invalidate derived records while preserving the ingested originals."""
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.execute("DELETE FROM synthetic_anomalies")
            connection.execute("DELETE FROM match_candidates")
            connection.execute("DELETE FROM real_anomalies")

    def clear_synthetic_and_downstream(self) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.execute("DELETE FROM synthetic_anomalies")

    def replace_original_samples(self, values: list[OriginalSample]) -> None:
        """Replace the canonical input catalog and invalidate all derived data."""
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.execute("DELETE FROM synthetic_anomalies")
            connection.execute("DELETE FROM match_candidates")
            connection.execute("DELETE FROM real_anomalies")
            connection.execute("DELETE FROM original_samples")
            connection.executemany(
                """INSERT INTO original_samples
                   (id, source_name, image_path, segmentation_path, spatial_dimensions,
                    has_anomaly, is_annotated, source_index, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [_original_values(value) for value in values],
            )

    def upsert_real_anomaly(self, value: RealAnomaly) -> None:
        self._upsert(
            """INSERT INTO real_anomalies
               (id, original_sample_id, component_index, image_path, segmentation_path,
                roi_image_path, roi_segmentation_path, spatial_dimensions,
                position_z, position_y, position_x, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 original_sample_id=excluded.original_sample_id,
                 component_index=excluded.component_index,
                 image_path=excluded.image_path,
                 segmentation_path=excluded.segmentation_path,
                 roi_image_path=excluded.roi_image_path,
                 roi_segmentation_path=excluded.roi_segmentation_path,
                 spatial_dimensions=excluded.spatial_dimensions,
                 position_z=excluded.position_z,
                 position_y=excluded.position_y,
                 position_x=excluded.position_x,
                 metadata_json=excluded.metadata_json""",
            (
                value.id, value.original_sample_id, value.component_index, value.image_path,
                value.segmentation_path, value.roi_image_path, value.roi_segmentation_path,
                value.spatial_dimensions, value.position_z, value.position_y, value.position_x,
                _json(value.metadata),
            ),
        )

    def upsert_synthetic_anomaly(self, value: SyntheticAnomaly) -> None:
        self._upsert(
            """INSERT INTO synthetic_anomalies
               (id, real_anomaly_id, variant_index, image_path, segmentation_path, seed)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 real_anomaly_id=excluded.real_anomaly_id,
                 variant_index=excluded.variant_index,
                 image_path=excluded.image_path,
                 segmentation_path=excluded.segmentation_path,
                 seed=excluded.seed""",
            (
                value.id, value.real_anomaly_id, value.variant_index,
                value.image_path, value.segmentation_path, value.seed,
            ),
        )

    def upsert_hybrid_sample(self, value: HybridSample) -> None:
        self._upsert(
            """INSERT INTO hybrid_samples
               (id, original_sample_id, variant_index, image_path, segmentation_path, status, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 original_sample_id=excluded.original_sample_id,
                 variant_index=excluded.variant_index,
                 image_path=excluded.image_path,
                 segmentation_path=excluded.segmentation_path,
                 status=excluded.status,
                 error=excluded.error""",
            (
                value.id, value.original_sample_id, value.variant_index, value.image_path,
                value.segmentation_path, value.status, value.error,
            ),
        )

    def upsert_placement(self, value: Placement) -> None:
        self._upsert(
            """INSERT INTO placements
               (id, hybrid_sample_id, synthetic_anomaly_id, order_index, spatial_dimensions,
                position_z, position_y, position_x, coordinate_system, score, method,
                roi_image_path, roi_segmentation_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 hybrid_sample_id=excluded.hybrid_sample_id,
                 synthetic_anomaly_id=excluded.synthetic_anomaly_id,
                 order_index=excluded.order_index,
                 spatial_dimensions=excluded.spatial_dimensions,
                 position_z=excluded.position_z,
                 position_y=excluded.position_y,
                 position_x=excluded.position_x,
                 coordinate_system=excluded.coordinate_system,
                 score=excluded.score,
                 method=excluded.method,
                 roi_image_path=excluded.roi_image_path,
                 roi_segmentation_path=excluded.roi_segmentation_path""",
            (
                value.id, value.hybrid_sample_id, value.synthetic_anomaly_id,
                value.order_index, value.spatial_dimensions, value.position_z,
                value.position_y, value.position_x, value.coordinate_system, value.score,
                value.method, value.roi_image_path, value.roi_segmentation_path,
            ),
        )

    def replace_hybrid_plan(
        self,
        hybrids: list[HybridSample],
        placements: list[Placement],
    ) -> None:
        """Atomically replace all hybrid plans using one SQLite transaction."""
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.executemany(
                """INSERT INTO hybrid_samples
                   (id, original_sample_id, variant_index, image_path,
                    segmentation_path, status, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [_hybrid_values(value) for value in hybrids],
            )
            connection.executemany(
                """INSERT INTO placements
                   (id, hybrid_sample_id, synthetic_anomaly_id, order_index,
                    spatial_dimensions, position_z, position_y, position_x,
                    coordinate_system, score, method, roi_image_path,
                    roi_segmentation_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [_placement_values(value) for value in placements],
            )

    def upsert_match_candidates(self, values: list[MatchCandidate]) -> None:
        if not values:
            return
        with self.connection() as connection:
            connection.executemany(
                """INSERT INTO match_candidates
                   (original_sample_id, real_anomaly_id, matcher_signature,
                    is_valid, score, position_json, center_json, roi_shape_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(original_sample_id, real_anomaly_id, matcher_signature)
                   DO UPDATE SET
                     is_valid=excluded.is_valid,
                     score=excluded.score,
                     position_json=excluded.position_json,
                     center_json=excluded.center_json,
                     roi_shape_json=excluded.roi_shape_json""",
                [_match_candidate_values(value) for value in values],
            )

    def _upsert(self, sql: str, parameters: tuple) -> None:
        with self.connection() as connection:
            connection.execute(sql, parameters)
