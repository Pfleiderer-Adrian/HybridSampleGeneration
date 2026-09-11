"""Shared fixtures and helpers for repository-backed dataset tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_sample_generator.domain.records import (
    HybridSample,
    OriginalSample,
    RealAnomaly,
    SyntheticAnomaly,
)
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository


class DatasetStudyTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = StudyRepository(self.root / "study.sqlite")
        self.store = ArtifactStore(self.root)
        self._populate_study()

    def tearDown(self):
        self.temporary.cleanup()

    def _save(self, entity_type, entity_id, role, array):
        return self.store.save_entity_array(entity_type, entity_id, role, array)

    def _populate_study(self):
        image = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        mask = np.zeros((1, 3, 4), dtype=np.uint8)
        mask[:, 1:, 1:3] = 1
        original = OriginalSample(
            "original-anomaly",
            "anomaly.npy",
            self._save("original_samples", "original-anomaly", "image", image),
            self._save(
                "original_samples", "original-anomaly", "segmentation", mask
            ),
            2,
            True,
            True,
            0,
            {"patient": "example"},
        )
        control = OriginalSample(
            "original-control",
            "control.npy",
            self._save(
                "original_samples", "original-control", "image", image + 20
            ),
            None,
            2,
            False,
            False,
            1,
        )
        self.repository.replace_original_samples([original, control])

        roi = image[:, 1:, 1:3]
        roi_mask = mask[:, 1:, 1:3]
        real = RealAnomaly(
            "real",
            original.id,
            0,
            self._save("real_anomalies", "real", "image", roi),
            self._save("real_anomalies", "real", "segmentation", roi_mask),
            self._save("real_anomalies", "real", "roi_image", roi + 1),
            self._save(
                "real_anomalies", "real", "roi_segmentation", roi_mask
            ),
            2,
            None,
            0.5,
            0.5,
            {"label": 1},
        )
        self.repository.upsert_real_anomaly(real)

        synthetic = SyntheticAnomaly(
            "synthetic",
            real.id,
            0,
            self._save("synthetic_anomalies", "synthetic", "image", roi + 2),
            self._save(
                "synthetic_anomalies", "synthetic", "segmentation", roi_mask
            ),
            17,
        )
        self.repository.upsert_synthetic_anomaly(synthetic)

        generated = HybridSample(
            "hybrid-generated",
            control.id,
            0,
            self._save("hybrid_samples", "hybrid-generated", "image", image + 30),
            self._save(
                "hybrid_samples", "hybrid-generated", "segmentation", mask
            ),
            "generated",
        )
        planned = HybridSample("hybrid-planned", control.id, 1)
        self.repository.upsert_hybrid_sample(generated)
        self.repository.upsert_hybrid_sample(planned)
