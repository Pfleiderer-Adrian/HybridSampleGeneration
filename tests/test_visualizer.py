import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_handler.visualizer.maintenance import StudyMaintenance
from data_handler.visualizer.queries import (
    StudyBrowserModel,
    filter_evaluation_groups,
)
from data_handler.visualizer.rendering import (
    display_mask_plane,
    display_plane,
    normalize_for_display,
)
from data_handler.visualizer.state import SelectionController
from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyRecords import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)
from synthesizer.StudyRepository import StudyRepository


class VisualizerRenderingTests(unittest.TestCase):
    def test_rgb_and_channel_rendering_preserve_expected_dimensions(self):
        rgb = np.zeros((3, 5, 7), dtype=np.float32)
        rgb[0] = 255
        rgb[1] = 128

        automatic = display_plane(rgb)
        self.assertEqual(automatic.image.shape, (5, 7, 3))
        self.assertTrue(np.all(automatic.image[..., 0] == 255))
        self.assertTrue(np.all(automatic.image[..., 1] == 128))

        green = display_plane(rgb, channel="1")
        self.assertEqual(green.image.shape, (5, 7))
        self.assertTrue(np.all(green.image == 128))

        normalized = normalize_for_display(automatic.image)
        self.assertEqual(normalized.shape, (5, 7, 3))
        self.assertGreater(float(normalized[..., 0].mean()), 0.99)
        self.assertGreater(float(normalized[..., 1].mean()), 0.45)
        self.assertLess(float(normalized[..., 2].mean()), 0.01)

    def test_volume_and_mask_rendering_share_slice_coordinates(self):
        volume = np.zeros((3, 4, 5, 7), dtype=np.float32)
        volume[0, 2] = 4
        mask = np.zeros((1, 4, 5, 7), dtype=np.uint8)
        mask[:, 2, 1:3, 2:5] = 1

        image_plane = display_plane(volume, slice_index=2)
        mask_plane = display_mask_plane(mask, slice_index=2)

        self.assertEqual(image_plane.image.shape, (5, 7, 3))
        self.assertEqual(image_plane.depth, 4)
        self.assertEqual(image_plane.slice_index, 2)
        self.assertEqual(mask_plane.image.shape, (5, 7))
        self.assertEqual(int(mask_plane.image.sum()), 6)


class VisualizerModelTests(unittest.TestCase):
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

    def test_model_resolves_variants_placements_and_generated_default(self):
        anomaly = self.model.anomaly_context("real-0", "synthetic-1")
        self.assertEqual(anomaly.original.id, "original-anomaly")
        self.assertEqual(len(anomaly.variants), 2)
        self.assertEqual(anomaly.synthetic.variant_index, 1)

        hybrid = self.model.first_hybrid_context()
        self.assertEqual(hybrid.hybrid.id, "hybrid-generated")
        self.assertEqual(len(hybrid.placements), 2)
        self.assertEqual(hybrid.selected_placement.placement.id, "placement-1")
        self.assertEqual(hybrid.selected_placement.real.id, "real-0")

        summary = self.model.summary()
        self.assertEqual(summary["synthetic_anomalies"], 2)
        self.assertEqual(summary["hybrids"], 2)
        self.assertEqual(summary["placements"], 3)
        self.assertEqual(summary["match_candidates"], 1)
        self.assertEqual(summary["evaluation_pairs"], 3)
        candidates = self.model.match_candidates("original-control")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].real_anomaly_id, "real-0")

    def test_evaluation_rows_are_grouped_and_filterable(self):
        cutout = next(
            group
            for group in self.model.evaluations
            if group.synthetic_anomaly_id == "synthetic-0" and not group.placement_id
        )
        self.assertEqual(set(cutout.metrics), {"Contrast", "Volume"})
        self.assertEqual(len(cutout.calculators), 2)

        filtered = filter_evaluation_groups(
            self.model.evaluations,
            metrics=("Contrast",),
            top_percent=34,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].placement_id, "placement-1")
        self.assertEqual([group.score for group in self.model.evaluations], [0.0] * 3)

    def test_selection_controller_publishes_normalized_ids(self):
        controller = SelectionController()
        seen = []
        controller.subscribe(seen.append)
        state = controller.update(
            source="test",
            original_sample_id="original-control",
            hybrid_sample_id="hybrid-generated",
        )
        self.assertEqual(state.source, "test")
        self.assertEqual(seen[-1], state)

    def test_removal_preview_cascades_to_complete_hybrids_and_archives_files(self):
        maintenance = StudyMaintenance(self.model)
        impact = maintenance.preview_removal("synthetic", "synthetic-0")

        self.assertEqual(impact.synthetic_ids, ("synthetic-0",))
        self.assertEqual(
            set(impact.hybrid_ids), {"hybrid-planned", "hybrid-generated"}
        )
        self.assertEqual(
            set(impact.placement_ids),
            {"placement-0", "placement-1", "placement-2"},
        )
        source_image = self.store.resolve(
            self.model.synthetic_by_id["synthetic-0"].image_path
        )
        self.assertTrue(source_image.is_file())

        trash = maintenance.archive_and_remove(impact)

        self.assertTrue(trash.is_dir())
        self.assertFalse(source_image.exists())
        self.assertNotIn("synthetic-0", self.model.synthetic_by_id)
        self.assertIn("synthetic-1", self.model.synthetic_by_id)
        self.assertEqual(len(self.model.hybrids), 0)
        self.assertEqual(len(self.model.placements), 0)

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


if __name__ == "__main__":
    unittest.main()
