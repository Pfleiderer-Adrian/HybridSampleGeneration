import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from hybrid_sample_generator.domain.records import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.visualization.queries import StudyBrowserModel


class VisualizerStudyTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = StudyRepository(self.root / "artifacts.sqlite")
        self.store = ArtifactStore(self.root)
        self._create_study()
        self.csv_path = self.root / "evaluation_results" / "metric_diffs.csv"
        self._create_evaluation_csv()
        self.model = StudyBrowserModel(
            self.repository,
            self.store,
            metric_csv_path=str(self.csv_path),
        )

    def tearDown(self):
        self.temporary.cleanup()
    def _save(self, entity_type, entity_id, role, array):
        return self.store.save_entity_array(entity_type, entity_id, role, array)

    def _create_study(self):
        image = np.zeros((3, 8, 10), dtype=np.float32)
        image[0] = 255
        mask = np.zeros((1, 8, 10), dtype=np.uint8)
        mask[:, 2:5, 3:6] = 1
        empty_mask = np.zeros_like(mask)
        original_anomaly = OriginalSample(
            "original-anomaly",
            "anomaly.png",
            self._save("original_samples", "original-anomaly", "image", image),
            self._save(
                "original_samples", "original-anomaly", "segmentation", mask
            ),
            2,
            True,
            True,
            0,
        )
        original_control = OriginalSample(
            "original-control",
            "control.png",
            self._save("original_samples", "original-control", "image", image / 2),
            self._save(
                "original_samples", "original-control", "segmentation", empty_mask
            ),
            2,
            False,
            True,
            1,
        )
        self.repository.replace_original_samples([original_anomaly, original_control])

        roi = image[:, 1:5, 2:6]
        roi_mask = mask[:, 1:5, 2:6]
        real = RealAnomaly(
            "real-0",
            original_anomaly.id,
            0,
            self._save("real_anomalies", "real-0", "image", roi),
            self._save("real_anomalies", "real-0", "segmentation", roi_mask),
            self._save("real_anomalies", "real-0", "roi_image", roi),
            self._save("real_anomalies", "real-0", "roi_segmentation", roi_mask),
            2,
            None,
            0.4,
            0.5,
            {"roi_shape": [4, 4]},
        )
        self.repository.upsert_real_anomaly(real)

        synthetics = []
        for index in range(2):
            synthetic_id = f"synthetic-{index}"
            synthetics.append(
                SyntheticAnomaly(
                    synthetic_id,
                    real.id,
                    index,
                    self._save(
                        "synthetic_anomalies",
                        synthetic_id,
                        "image",
                        roi + index,
                    ),
                    self._save(
                        "synthetic_anomalies",
                        synthetic_id,
                        "segmentation",
                        roi_mask,
                    ),
                    100 + index,
                )
            )
        for synthetic in synthetics:
            self.repository.upsert_synthetic_anomaly(synthetic)

        hybrid_image = image / 2
        hybrid_image[:, 4:7, 5:8] += 10
        hybrids = [
            HybridSample("hybrid-planned", original_control.id, 0),
            HybridSample(
                "hybrid-generated",
                original_control.id,
                1,
                self._save(
                    "hybrid_samples", "hybrid-generated", "image", hybrid_image
                ),
                self._save(
                    "hybrid_samples",
                    "hybrid-generated",
                    "segmentation",
                    empty_mask,
                ),
                "generated",
            ),
        ]
        placements = []
        for index, (placement_id, hybrid_id, synthetic_id) in enumerate(
            (
                ("placement-0", "hybrid-planned", "synthetic-0"),
                ("placement-1", "hybrid-generated", "synthetic-0"),
                ("placement-2", "hybrid-generated", "synthetic-1"),
            )
        ):
            placements.append(
                Placement(
                    placement_id,
                    hybrid_id,
                    synthetic_id,
                    index if hybrid_id == "hybrid-generated" else 0,
                    2,
                    None,
                    0.5,
                    0.6,
                    score=0.9 - index * 0.1,
                    method="local",
                    roi_image_path=self._save(
                        "placements", placement_id, "roi_image", roi
                    ),
                    roi_segmentation_path=self._save(
                        "placements", placement_id, "roi_segmentation", roi_mask
                    ),
                )
            )
        self.repository.replace_hybrid_plan(hybrids, placements)
        self.repository.upsert_match_candidates(
            [
                MatchCandidate(
                    original_control.id,
                    real.id,
                    "matcher-test",
                    True,
                    0.9,
                    (0.5, 0.6),
                    (4.0, 6.0),
                    (4, 4),
                )
            ]
        )

    def _create_evaluation_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            (
                "synthetic-0",
                "real-0",
                "synthetic-0",
                "",
                "get_glcm_feature_diffs",
                {"Contrast": 0.2},
            ),
            (
                "synthetic-0",
                "real-0",
                "synthetic-0",
                "",
                "get_volume_feature_diffs",
                {"Volume": 2.0},
            ),
            (
                "synthetic-1",
                "real-0",
                "synthetic-1",
                "",
                "get_glcm_feature_diffs",
                {"Contrast": 0.5},
            ),
            (
                "placement-1",
                "real-0",
                "synthetic-0",
                "placement-1",
                "get_glcm_roi_feature_diffs",
                {"Contrast": 0.8, "roi_Energy": 0.4},
            ),
        )
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "pair_id",
                    "real_anomaly_id",
                    "synthetic_anomaly_id",
                    "placement_id",
                    "feature_calculator",
                    "metric_diffs",
                )
            )
            for *values, metrics in rows:
                writer.writerow((*values, json.dumps(metrics)))
