"""SQLite schema creation and version validation."""

SCHEMA_VERSION = 2


class StudySchemaMixin:
    def _initialize_schema(self) -> None:
        with self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL)"
            )
            existing_schema = connection.execute(
                "SELECT version FROM schema_info LIMIT 1"
            ).fetchone()
            if (
                existing_schema is not None
                and int(existing_schema["version"]) != SCHEMA_VERSION
            ):
                raise ValueError(
                    f"Unsupported artifact database schema {existing_schema['version']}; "
                    f"expected {SCHEMA_VERSION}. Backward migration is intentionally "
                    f"unsupported. Move or delete {self.database_path} and run "
                    "ingest_dataset() again."
                )
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
                    has_anomaly INTEGER NOT NULL CHECK (has_anomaly IN (0, 1)),
                    is_annotated INTEGER NOT NULL CHECK (is_annotated IN (0, 1)),
                    source_index INTEGER NOT NULL UNIQUE,
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

                CREATE TABLE IF NOT EXISTS match_candidates (
                    original_sample_id TEXT NOT NULL REFERENCES original_samples(id) ON DELETE CASCADE,
                    real_anomaly_id TEXT NOT NULL REFERENCES real_anomalies(id) ON DELETE CASCADE,
                    matcher_signature TEXT NOT NULL,
                    is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
                    score REAL,
                    position_json TEXT,
                    center_json TEXT,
                    roi_shape_json TEXT NOT NULL,
                    PRIMARY KEY (original_sample_id, real_anomaly_id, matcher_signature)
                );

                CREATE INDEX IF NOT EXISTS idx_original_has_anomaly ON original_samples(has_anomaly);
                CREATE INDEX IF NOT EXISTS idx_real_original ON real_anomalies(original_sample_id);
                CREATE INDEX IF NOT EXISTS idx_synthetic_real ON synthetic_anomalies(real_anomaly_id);
                CREATE INDEX IF NOT EXISTS idx_hybrid_original ON hybrid_samples(original_sample_id);
                CREATE INDEX IF NOT EXISTS idx_placement_hybrid ON placements(hybrid_sample_id);
                CREATE INDEX IF NOT EXISTS idx_placement_synthetic ON placements(synthetic_anomaly_id);
                CREATE INDEX IF NOT EXISTS idx_match_original_signature
                    ON match_candidates(original_sample_id, matcher_signature);
                """
            )
            if existing_schema is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
