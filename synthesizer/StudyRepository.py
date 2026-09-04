from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from synthesizer.StudyRecords import (
    HybridSample,
    OriginalSample,
    Placement,
    RealAnomaly,
    StudyHierarchyEntry,
    SyntheticAnomaly,
)


SCHEMA_VERSION = 1


def stable_id(kind: str, *components) -> str:
    encoded = json.dumps(
        [kind, *components], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()[:20]}"


def stable_seed(*components) -> int:
    encoded = json.dumps(components, ensure_ascii=False, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") & 0x7FFFFFFF


class StudyRepository:
    """Canonical SQLite index for all study entities and their relationships."""

    def __init__(self, database_path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS original_samples (
                    id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    segmentation_path TEXT,
                    spatial_dimensions INTEGER NOT NULL CHECK (spatial_dimensions IN (2, 3)),
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS real_anomalies (
                    id TEXT PRIMARY KEY,
                    original_sample_id TEXT NOT NULL REFERENCES original_samples(id) ON DELETE CASCADE,
                    component_index INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    segmentation_path TEXT NOT NULL,
                    roi_image_path TEXT NOT NULL,
                    roi_segmentation_path TEXT NOT NULL,
                    spatial_dimensions INTEGER NOT NULL CHECK (spatial_dimensions IN (2, 3)),
                    position_z REAL,
                    position_y REAL NOT NULL,
                    position_x REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE (original_sample_id, component_index),
                    CHECK (
                        (spatial_dimensions = 2 AND position_z IS NULL) OR
                        (spatial_dimensions = 3 AND position_z IS NOT NULL)
                    )
                );

                CREATE TABLE IF NOT EXISTS synthetic_anomalies (
                    id TEXT PRIMARY KEY,
                    real_anomaly_id TEXT NOT NULL REFERENCES real_anomalies(id) ON DELETE CASCADE,
                    variant_index INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    segmentation_path TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    UNIQUE (real_anomaly_id, variant_index)
                );

                CREATE TABLE IF NOT EXISTS hybrid_samples (
                    id TEXT PRIMARY KEY,
                    original_sample_id TEXT NOT NULL REFERENCES original_samples(id) ON DELETE CASCADE,
                    variant_index INTEGER NOT NULL,
                    image_path TEXT,
                    segmentation_path TEXT,
                    status TEXT NOT NULL CHECK (status IN ('planned', 'generated', 'failed')),
                    error TEXT,
                    UNIQUE (original_sample_id, variant_index)
                );

                CREATE TABLE IF NOT EXISTS placements (
                    id TEXT PRIMARY KEY,
                    hybrid_sample_id TEXT NOT NULL REFERENCES hybrid_samples(id) ON DELETE CASCADE,
                    synthetic_anomaly_id TEXT NOT NULL REFERENCES synthetic_anomalies(id) ON DELETE RESTRICT,
                    order_index INTEGER NOT NULL,
                    spatial_dimensions INTEGER NOT NULL CHECK (spatial_dimensions IN (2, 3)),
                    position_z REAL,
                    position_y REAL NOT NULL,
                    position_x REAL NOT NULL,
                    coordinate_system TEXT NOT NULL CHECK (coordinate_system = 'normalized_center'),
                    score REAL,
                    method TEXT NOT NULL,
                    roi_image_path TEXT,
                    roi_segmentation_path TEXT,
                    UNIQUE (hybrid_sample_id, order_index),
                    UNIQUE (hybrid_sample_id, synthetic_anomaly_id),
                    CHECK (
                        (spatial_dimensions = 2 AND position_z IS NULL) OR
                        (spatial_dimensions = 3 AND position_z IS NOT NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_real_original ON real_anomalies(original_sample_id);
                CREATE INDEX IF NOT EXISTS idx_synthetic_real ON synthetic_anomalies(real_anomaly_id);
                CREATE INDEX IF NOT EXISTS idx_hybrid_original ON hybrid_samples(original_sample_id);
                CREATE INDEX IF NOT EXISTS idx_placement_hybrid ON placements(hybrid_sample_id);
                CREATE INDEX IF NOT EXISTS idx_placement_synthetic ON placements(synthetic_anomaly_id);
                """
            )
            row = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(row["version"]) != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported artifact database schema {row['version']}; expected {SCHEMA_VERSION}."
                )

    def clear_all_records(self) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.execute("DELETE FROM synthetic_anomalies")
            connection.execute("DELETE FROM real_anomalies")
            connection.execute("DELETE FROM original_samples")

    def clear_synthetic_and_downstream(self) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.execute("DELETE FROM synthetic_anomalies")

    def clear_hybrid_plans(self) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM placements")
            connection.execute("DELETE FROM hybrid_samples")
            connection.execute(
                """DELETE FROM original_samples
                   WHERE NOT EXISTS (
                       SELECT 1 FROM real_anomalies
                       WHERE real_anomalies.original_sample_id = original_samples.id
                   )"""
            )

    def upsert_original_sample(self, value: OriginalSample) -> None:
        self._upsert(
            """INSERT INTO original_samples
               (id, source_name, image_path, segmentation_path, spatial_dimensions, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 source_name=excluded.source_name,
                 image_path=excluded.image_path,
                 segmentation_path=excluded.segmentation_path,
                 spatial_dimensions=excluded.spatial_dimensions,
                 metadata_json=excluded.metadata_json""",
            (
                value.id, value.source_name, value.image_path, value.segmentation_path,
                value.spatial_dimensions, _json(value.metadata),
            ),
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

    def _upsert(self, sql: str, parameters: tuple) -> None:
        with self.connection() as connection:
            connection.execute(sql, parameters)

    def get_original_sample(self, record_id: str) -> OriginalSample:
        return _original(self._one("SELECT * FROM original_samples WHERE id = ?", (record_id,)))

    def find_original_sample(self, record_id: str) -> OriginalSample | None:
        rows = self._all("SELECT * FROM original_samples WHERE id = ?", (record_id,))
        return _original(rows[0]) if rows else None

    def get_real_anomaly(self, record_id: str) -> RealAnomaly:
        return _real(self._one("SELECT * FROM real_anomalies WHERE id = ?", (record_id,)))

    def get_synthetic_anomaly(self, record_id: str) -> SyntheticAnomaly:
        return _synthetic(self._one("SELECT * FROM synthetic_anomalies WHERE id = ?", (record_id,)))

    def get_hybrid_sample(self, record_id: str) -> HybridSample:
        return _hybrid(self._one("SELECT * FROM hybrid_samples WHERE id = ?", (record_id,)))

    def get_placement(self, record_id: str) -> Placement:
        return _placement(self._one("SELECT * FROM placements WHERE id = ?", (record_id,)))

    def list_original_samples(self) -> list[OriginalSample]:
        return [_original(row) for row in self._all("SELECT * FROM original_samples ORDER BY id")]

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
                 o.spatial_dimensions AS o_spatial_dimensions, o.metadata_json AS o_metadata_json,
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
                        row["o_id"], row["o_source_name"], row["o_image_path"],
                        row["o_segmentation_path"], row["o_spatial_dimensions"],
                        _metadata(row["o_metadata_json"]),
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


def _original(row: sqlite3.Row) -> OriginalSample:
    return OriginalSample(
        row["id"], row["source_name"], row["image_path"], row["segmentation_path"],
        row["spatial_dimensions"], _metadata(row["metadata_json"]),
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
