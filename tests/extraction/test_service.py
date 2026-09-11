"""Tests for extraction workflow ownership and persisted anomaly records."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.extraction.service import (
    ExtractionService,
    _position_columns,
)
from hybrid_sample_generator.persistence.identifiers import stable_id


class ExtractionServiceTests(unittest.TestCase):
    def setUp(self):
        self.config = ExtractionConfiguration((1, 8, 8))
        self.repository = Mock()
        self.artifact_store = Mock()
        self.artifact_store.save_entity_array.side_effect = (
            lambda entity_type, entity_id, role, _array: (
                f"artifacts/{entity_type}/{entity_id}/{role}.npy"
            )
        )
        self.datasets = Mock()
        self.service = ExtractionService(
            self.config,
            self.repository,
            self.artifact_store,
            self.datasets,
        )

    def test_extract_requires_annotated_anomalous_originals(self):
        self.datasets.original_samples.return_value = []

        with self.assertRaisesRegex(ValueError, "No anomalous originals found"):
            self.service.extract()

        self.datasets.original_samples.assert_called_once_with(
            return_artifacts=("img", "ori_mask", "record"),
            has_anomaly=True,
            is_annotated=True,
            load_to_ram=False,
            numpy_mode=True,
        )
        self.repository.clear_real_anomalies_and_downstream.assert_not_called()

    def test_extract_requires_segmentation_artifact(self):
        self.datasets.original_samples.return_value = [
            {
                "record": SimpleNamespace(id="original-a"),
                "img": np.zeros((1, 8, 8)),
                "ori_mask": None,
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "has no segmentation artifact"):
            self.service.extract()

        self.repository.clear_real_anomalies_and_downstream.assert_called_once_with()

    def test_extract_rejects_unexpected_image_dimensions(self):
        self.datasets.original_samples.return_value = [
            {
                "record": SimpleNamespace(id="original-a"),
                "img": np.zeros((8, 8)),
                "ori_mask": np.zeros((8, 8)),
            }
        ]

        with self.assertRaisesRegex(ValueError, "expected .*C,H,W"):
            self.service.extract()

    def test_extract_rejects_unaligned_results(self):
        self.datasets.original_samples.return_value = [self._sample(3)]
        result = (
            [(np.zeros((1, 8, 8)), {"centroid_norm": (0.25, 0.75)})],
            [],
            [np.zeros((1, 8, 8))],
            [np.zeros((1, 8, 8))],
        )

        with (
            patch(
                "hybrid_sample_generator.extraction.service.crop_and_center_anomaly_2d",
                return_value=result,
            ),
            self.assertRaisesRegex(RuntimeError, "unaligned"),
        ):
            self.service.extract()

    def test_extract_rejects_empty_component_results(self):
        self.datasets.original_samples.return_value = [self._sample(3)]

        with (
            patch(
                "hybrid_sample_generator.extraction.service.crop_and_center_anomaly_2d",
                return_value=(None, None, None, None),
            ),
            self.assertRaisesRegex(ValueError, "No real anomalies were extracted"),
        ):
            self.service.extract()

        self.repository.upsert_real_anomaly.assert_not_called()

    def test_extract_dispatches_2d_and_persists_record_and_artifacts(self):
        sample = self._sample(3)
        self.datasets.original_samples.return_value = [sample]
        anomaly = np.ones((1, 8, 8), dtype=np.float32)
        roi = np.ones((1, 6, 7), dtype=np.float32)
        mask = np.ones((1, 8, 8), dtype=np.uint8)
        roi_mask = np.ones((1, 6, 7), dtype=np.uint8)
        metadata = {"centroid_norm": (0.25, 0.75), "label": 2}
        result = ([(anomaly, metadata)], [roi], [mask], [roi_mask])

        with (
            patch(
                "hybrid_sample_generator.extraction.service.crop_and_center_anomaly_2d",
                return_value=result,
            ) as extract_2d,
            patch(
                "hybrid_sample_generator.extraction.service.crop_and_center_anomaly_3d"
            ) as extract_3d,
        ):
            records = self.service.extract()

        extract_2d.assert_called_once_with(
            sample["img"],
            sample["ori_mask"],
            self.config,
        )
        extract_3d.assert_not_called()
        self.repository.clear_real_anomalies_and_downstream.assert_called_once_with()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.id, stable_id("real", "original-a", 0))
        self.assertEqual(record.original_sample_id, "original-a")
        self.assertEqual(record.component_index, 0)
        self.assertEqual(record.spatial_dimensions, 2)
        self.assertIsNone(record.position_z)
        self.assertEqual(record.source_position, (0.25, 0.75))
        self.assertEqual(record.metadata["label"], 2)
        self.assertEqual(record.metadata["roi_shape"], (6, 7))
        self.assertNotIn("roi_shape", metadata)
        self.assertEqual(self.artifact_store.save_entity_array.call_count, 4)
        roles = [
            call.args[2]
            for call in self.artifact_store.save_entity_array.call_args_list
        ]
        self.assertEqual(
            roles,
            ["image", "segmentation", "roi_image", "roi_segmentation"],
        )
        self.repository.upsert_real_anomaly.assert_called_once_with(record)

    def test_extract_dispatches_3d_and_maps_position_columns(self):
        sample = self._sample(4)
        self.datasets.original_samples.return_value = [sample]
        anomaly = np.ones((1, 4, 5, 6), dtype=np.float32)
        roi = np.ones((1, 3, 4, 5), dtype=np.float32)
        mask = np.ones_like(anomaly, dtype=np.uint8)
        roi_mask = np.ones_like(roi, dtype=np.uint8)
        result = (
            [(anomaly, {"centroid_norm": (0.2, 0.4, 0.6)})],
            [roi],
            [mask],
            [roi_mask],
        )

        with (
            patch(
                "hybrid_sample_generator.extraction.service.crop_and_center_anomaly_2d"
            ) as extract_2d,
            patch(
                "hybrid_sample_generator.extraction.service.crop_and_center_anomaly_3d",
                return_value=result,
            ) as extract_3d,
        ):
            record = self.service.extract()[0]

        extract_2d.assert_not_called()
        extract_3d.assert_called_once_with(
            sample["img"],
            sample["ori_mask"],
            self.config,
        )
        self.assertEqual(record.spatial_dimensions, 3)
        self.assertEqual(record.source_position, (0.2, 0.4, 0.6))
        self.assertEqual(record.metadata["roi_shape"], (3, 4, 5))

    def test_position_columns_rejects_unsupported_dimensions(self):
        self.assertEqual(_position_columns((0.2, 0.8)), (None, 0.2, 0.8))
        self.assertEqual(
            _position_columns((0.1, 0.2, 0.3)),
            (0.1, 0.2, 0.3),
        )
        with self.assertRaisesRegex(ValueError, "Expected a 2D or 3D"):
            _position_columns((0.5,))

    @staticmethod
    def _sample(image_ndim):
        shape = (1, 8, 8) if image_ndim == 3 else (1, 4, 5, 6)
        return {
            "record": SimpleNamespace(id="original-a"),
            "img": np.zeros(shape, dtype=np.float32),
            "ori_mask": np.zeros(shape, dtype=np.uint8),
        }


if __name__ == "__main__":
    unittest.main()
