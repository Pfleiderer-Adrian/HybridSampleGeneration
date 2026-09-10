import sqlite3
import tempfile
import unittest
from pathlib import Path

from hybrid_sample_generator.domain.records import (
    HybridSample,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)
from hybrid_sample_generator.persistence.identifiers import stable_id, stable_seed
from hybrid_sample_generator.persistence.study_repository import StudyRepository


class StudyRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "study.sqlite"
        self.repository = StudyRepository(self.database_path)

    def tearDown(self):
        self.temporary.cleanup()

    def _populate_hierarchy(self):
        original = OriginalSample(
            "original", "sample", "original.npy", "mask.npy", 2, True, True, 0,
            {"patient": "example"},
        )
        real = RealAnomaly(
            "real", original.id, 0, "real.npy", "real-mask.npy", "roi.npy",
            "roi-mask.npy", 2, None, 0.5, 0.5, {"label": 1},
        )
        synthetic = SyntheticAnomaly(
            "synthetic", real.id, 0, "synthetic.npy", "synthetic-mask.npy", 17,
        )
        hybrid = HybridSample("hybrid", original.id, 0)
        placement = Placement(
            "placement", hybrid.id, synthetic.id, 0, 2, None, 0.25, 0.75,
        )
        self.repository.replace_original_samples([original])
        self.repository.upsert_real_anomaly(real)
        self.repository.upsert_synthetic_anomaly(synthetic)
        self.repository.replace_hybrid_plan([hybrid], [placement])
        return original, real, synthetic, hybrid, placement

    def test_records_and_hierarchy_round_trip(self):
        expected = self._populate_hierarchy()

        self.assertEqual(self.repository.get_original_sample("original"), expected[0])
        self.assertEqual(self.repository.get_real_anomaly("real"), expected[1])
        self.assertEqual(self.repository.get_synthetic_anomaly("synthetic"), expected[2])
        self.assertEqual(self.repository.get_hybrid_sample("hybrid"), expected[3])
        self.assertEqual(self.repository.get_placement("placement"), expected[4])
        hierarchy = self.repository.hierarchy()
        self.assertEqual(len(hierarchy), 1)
        self.assertEqual(hierarchy[0].original, expected[0])
        self.assertEqual(hierarchy[0].placement, expected[4])
        self.assertEqual(
            self.repository.counts(),
            {
                "original_samples": 1,
                "real_anomalies": 1,
                "synthetic_anomalies": 1,
                "hybrid_samples": 1,
                "placements": 1,
            },
        )

    def test_hybrid_plan_replacement_rolls_back_as_one_transaction(self):
        *_, hybrid, placement = self._populate_hierarchy()
        invalid_placement = Placement(
            "invalid", "replacement", "missing-synthetic", 0, 2, None, 0.5, 0.5,
        )

        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.replace_hybrid_plan(
                [HybridSample("replacement", "original", 1)],
                [invalid_placement],
            )

        self.assertEqual(self.repository.list_hybrid_samples(), [hybrid])
        self.assertEqual(self.repository.list_placements(), [placement])

    def test_identifiers_are_deterministic_and_schema_versions_are_rejected(self):
        self.assertEqual(stable_id("sample", "a", 1), stable_id("sample", "a", 1))
        self.assertNotEqual(stable_id("sample", "a", 1), stable_id("sample", "a", 2))
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))

        incompatible_path = Path(self.temporary.name) / "incompatible.sqlite"
        with sqlite3.connect(incompatible_path) as connection:
            connection.execute("CREATE TABLE schema_info (version INTEGER NOT NULL)")
            connection.execute("INSERT INTO schema_info(version) VALUES (999)")
        with self.assertRaisesRegex(ValueError, "Unsupported artifact database schema 999"):
            StudyRepository(incompatible_path)


if __name__ == "__main__":
    unittest.main()
