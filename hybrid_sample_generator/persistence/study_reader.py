"""Read and hierarchy queries for the SQLite study repository."""

import sqlite3

from hybrid_sample_generator.domain.records import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    StudyHierarchyEntry,
    SyntheticAnomaly,
)
from hybrid_sample_generator.persistence.record_mapping import (
    _hybrid,
    _match_candidate,
    _metadata,
    _original,
    _placement,
    _real,
    _synthetic,
)


class StudyReaderMixin:
    def get_original_sample(self, record_id: str) -> OriginalSample:
        return _original(self._one("SELECT * FROM original_samples WHERE id = ?", (record_id,)))

    def get_real_anomaly(self, record_id: str) -> RealAnomaly:
        return _real(self._one("SELECT * FROM real_anomalies WHERE id = ?", (record_id,)))

    def get_synthetic_anomaly(self, record_id: str) -> SyntheticAnomaly:
        return _synthetic(self._one("SELECT * FROM synthetic_anomalies WHERE id = ?", (record_id,)))

    def get_hybrid_sample(self, record_id: str) -> HybridSample:
        return _hybrid(self._one("SELECT * FROM hybrid_samples WHERE id = ?", (record_id,)))

    def get_placement(self, record_id: str) -> Placement:
        return _placement(self._one("SELECT * FROM placements WHERE id = ?", (record_id,)))

    def list_original_samples(
        self,
        *,
        has_anomaly: bool | None = None,
        is_annotated: bool | None = None,
    ) -> list[OriginalSample]:
        clauses = []
        values = []
        if has_anomaly is not None:
            clauses.append("has_anomaly = ?")
            values.append(int(has_anomaly))
        if is_annotated is not None:
            clauses.append("is_annotated = ?")
            values.append(int(is_annotated))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._all(
            "SELECT * FROM original_samples"
            + where
            + " ORDER BY source_index, id",
            tuple(values),
        )
        return [_original(row) for row in rows]

    def list_match_candidates(
        self,
        original_sample_id: str,
        matcher_signature: str | None = None,
    ) -> list[MatchCandidate]:
        if matcher_signature is None:
            rows = self._all(
                """SELECT * FROM match_candidates
                   WHERE original_sample_id = ?
                   ORDER BY matcher_signature, real_anomaly_id""",
                (original_sample_id,),
            )
        else:
            rows = self._all(
                """SELECT * FROM match_candidates
                   WHERE original_sample_id = ? AND matcher_signature = ?
                   ORDER BY real_anomaly_id""",
                (original_sample_id, matcher_signature),
            )
        return [_match_candidate(row) for row in rows]

    def count_match_candidates(self) -> int:
        with self.connection() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM match_candidates").fetchone()[0]
            )

    def list_real_anomalies(self, original_sample_id: str | None = None) -> list[RealAnomaly]:
        if original_sample_id is None:
            rows = self._all("SELECT * FROM real_anomalies ORDER BY original_sample_id, component_index")
        else:
            rows = self._all(
                "SELECT * FROM real_anomalies WHERE original_sample_id = ? ORDER BY component_index",
                (original_sample_id,),
            )
        return [_real(row) for row in rows]

    def list_synthetic_anomalies(
        self, real_anomaly_id: str | None = None
    ) -> list[SyntheticAnomaly]:
        if real_anomaly_id is None:
            rows = self._all("SELECT * FROM synthetic_anomalies ORDER BY real_anomaly_id, variant_index")
        else:
            rows = self._all(
                "SELECT * FROM synthetic_anomalies WHERE real_anomaly_id = ? ORDER BY variant_index",
                (real_anomaly_id,),
            )
        return [_synthetic(row) for row in rows]

    def list_hybrid_samples(
        self, original_sample_id: str | None = None, status: str | None = None
    ) -> list[HybridSample]:
        clauses = []
        values = []
        if original_sample_id is not None:
            clauses.append("original_sample_id = ?")
            values.append(original_sample_id)
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._all(
            "SELECT * FROM hybrid_samples" + where + " ORDER BY original_sample_id, variant_index",
            tuple(values),
        )
        return [_hybrid(row) for row in rows]

    def list_placements(self, hybrid_sample_id: str | None = None) -> list[Placement]:
        if hybrid_sample_id is None:
            rows = self._all("SELECT * FROM placements ORDER BY hybrid_sample_id, order_index")
        else:
            rows = self._all(
                "SELECT * FROM placements WHERE hybrid_sample_id = ? ORDER BY order_index",
                (hybrid_sample_id,),
            )
        return [_placement(row) for row in rows]

    def hierarchy(self) -> list[StudyHierarchyEntry]:
        rows = self._all(
            """SELECT
                 o.id AS o_id, o.source_name AS o_source_name, o.image_path AS o_image_path,
                 o.segmentation_path AS o_segmentation_path,
                 o.spatial_dimensions AS o_spatial_dimensions,
                 o.has_anomaly AS o_has_anomaly, o.is_annotated AS o_is_annotated,
                 o.source_index AS o_source_index, o.metadata_json AS o_metadata_json,
                 h.id AS h_id, h.variant_index AS h_variant_index, h.image_path AS h_image_path,
                 h.segmentation_path AS h_segmentation_path, h.status AS h_status, h.error AS h_error,
                 p.id AS p_id, p.order_index AS p_order_index,
                 p.spatial_dimensions AS p_spatial_dimensions, p.position_z AS p_position_z,
                 p.position_y AS p_position_y, p.position_x AS p_position_x,
                 p.coordinate_system AS p_coordinate_system, p.score AS p_score,
                 p.method AS p_method, p.roi_image_path AS p_roi_image_path,
                 p.roi_segmentation_path AS p_roi_segmentation_path,
                 s.id AS s_id, s.real_anomaly_id AS s_real_anomaly_id,
                 s.variant_index AS s_variant_index, s.image_path AS s_image_path,
                 s.segmentation_path AS s_segmentation_path, s.seed AS s_seed,
                 r.id AS r_id, r.original_sample_id AS r_original_sample_id,
                 r.component_index AS r_component_index, r.image_path AS r_image_path,
                 r.segmentation_path AS r_segmentation_path, r.roi_image_path AS r_roi_image_path,
                 r.roi_segmentation_path AS r_roi_segmentation_path,
                 r.spatial_dimensions AS r_spatial_dimensions, r.position_z AS r_position_z,
                 r.position_y AS r_position_y, r.position_x AS r_position_x,
                 r.metadata_json AS r_metadata_json
               FROM placements p
               JOIN hybrid_samples h ON h.id = p.hybrid_sample_id
               JOIN original_samples o ON o.id = h.original_sample_id
               JOIN synthetic_anomalies s ON s.id = p.synthetic_anomaly_id
               JOIN real_anomalies r ON r.id = s.real_anomaly_id
               ORDER BY o.source_name, h.variant_index, p.order_index"""
        )
        result = []
        for row in rows:
            result.append(
                StudyHierarchyEntry(
                    original=OriginalSample(
                        id=row["o_id"],
                        source_name=row["o_source_name"],
                        image_path=row["o_image_path"],
                        segmentation_path=row["o_segmentation_path"],
                        spatial_dimensions=row["o_spatial_dimensions"],
                        has_anomaly=bool(row["o_has_anomaly"]),
                        is_annotated=bool(row["o_is_annotated"]),
                        source_index=row["o_source_index"],
                        metadata=_metadata(row["o_metadata_json"]),
                    ),
                    hybrid=HybridSample(
                        row["h_id"], row["o_id"], row["h_variant_index"],
                        row["h_image_path"], row["h_segmentation_path"],
                        row["h_status"], row["h_error"],
                    ),
                    placement=Placement(
                        row["p_id"], row["h_id"], row["s_id"], row["p_order_index"],
                        row["p_spatial_dimensions"], row["p_position_z"], row["p_position_y"],
                        row["p_position_x"], row["p_coordinate_system"], row["p_score"],
                        row["p_method"], row["p_roi_image_path"], row["p_roi_segmentation_path"],
                    ),
                    synthetic_anomaly=SyntheticAnomaly(
                        row["s_id"], row["s_real_anomaly_id"], row["s_variant_index"],
                        row["s_image_path"], row["s_segmentation_path"], row["s_seed"],
                    ),
                    real_anomaly=RealAnomaly(
                        row["r_id"], row["r_original_sample_id"], row["r_component_index"],
                        row["r_image_path"], row["r_segmentation_path"], row["r_roi_image_path"],
                        row["r_roi_segmentation_path"], row["r_spatial_dimensions"],
                        row["r_position_z"], row["r_position_y"], row["r_position_x"],
                        _metadata(row["r_metadata_json"]),
                    ),
                )
            )
        return result

    def counts(self) -> dict[str, int]:
        tables = (
            "original_samples", "real_anomalies", "synthetic_anomalies",
            "hybrid_samples", "placements",
        )
        with self.connection() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

    def _one(self, sql: str, parameters: tuple) -> sqlite3.Row:
        rows = self._all(sql, parameters)
        if not rows:
            raise KeyError(parameters[0] if parameters else "record")
        return rows[0]

    def _all(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        with self.connection() as connection:
            return list(connection.execute(sql, parameters).fetchall())

