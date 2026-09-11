"""Tests for fusion backend ownership and hybrid materialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.records import HybridSample, Placement
from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.fusion.service import FusionService, _mask_like_image


class _SyntheticDataset:
    def __init__(self, sample=None, *, length=1):
        self.sample = {} if sample is None else sample
        self.length = length
        self.loaded_ids = []

    def __len__(self):
        return self.length

    def load_sample_by_id(self, record_id):
        self.loaded_ids.append(record_id)
        return self.sample


class _RecordingBackend:
    def __init__(self):
        self.warmup_calls = []
        self.fuse_calls = []
        self.random_values = []

    def warmup(self, shape, **kwargs):
        self.warmup_calls.append((shape, kwargs))

    def fuse(self, sample, control_img, position, **kwargs):
        self.fuse_calls.append((sample, control_img.copy(), position, kwargs))
        self.random_values.append(float(np.random.random()))
        image = control_img + 1.0
        segmentation = np.ones((1, *image.shape[1:]), dtype=np.uint8)
        return FusionOutput(
            image=image,
            segmentation=segmentation,
            roi=np.full((1, 2, 2), 2.0, dtype=np.float32),
            roi_mask=np.ones((1, 2, 2), dtype=np.uint8),
        )


class FusionServiceTests(unittest.TestCase):
    def setUp(self):
        self.config = Configuration(
            "fusion-service-test",
            "VAE_ResNet_2D",
            (1, 8, 8),
            study_folder="/tmp/fusion-service-test",
        )
        self.repository = Mock()
        self.artifact_store = Mock()
        self.artifact_store.save_entity_array.side_effect = (
            lambda entity_type, entity_id, role, _array: (
                f"artifacts/{entity_type}/{entity_id}/{role}.npy"
            )
        )
        self.datasets = Mock()
        self.backend = _RecordingBackend()
        self.service = FusionService(
            self.config.fusion,
            self.config.extraction,
            self.config.study.seed,
            self.repository,
            self.artifact_store,
            self.datasets,
            backend=self.backend,
        )

    def test_materialize_requires_a_hybrid_plan(self):
        self.repository.list_hybrid_samples.return_value = []

        with self.assertRaisesRegex(ValueError, "No hybrid plan found"):
            self.service.materialize()

        self.datasets.synthetic_anomalies.assert_not_called()

    def test_materialize_requires_synthetic_anomalies(self):
        self.repository.list_hybrid_samples.return_value = [self._hybrid()]
        self.datasets.synthetic_anomalies.return_value = _SyntheticDataset(length=0)

        with self.assertRaisesRegex(ValueError, "No synthetic anomalies found"):
            self.service.materialize()

        self.datasets.synthetic_anomalies.assert_called_once()

    def test_materialize_persists_placements_and_generated_hybrid(self):
        hybrid = self._hybrid()
        placement = self._placement()
        dataset = _SyntheticDataset({"synthetic": "sample"})
        self.repository.list_hybrid_samples.return_value = [hybrid]
        self.repository.get_original_sample.return_value = SimpleNamespace(
            image_path="original-image.npy",
            segmentation_path=None,
        )
        self.repository.list_placements.return_value = [placement]
        self.datasets.synthetic_anomalies.return_value = dataset
        original_image = np.zeros((3, 8, 8), dtype=np.float32)
        self.artifact_store.load_array.return_value = original_image

        with patch(
            "hybrid_sample_generator.fusion.service.tqdm",
            side_effect=lambda values, **_kwargs: values,
        ):
            generated = self.service.materialize()

        self.assertEqual(len(generated), 1)
        result = generated[0]
        self.assertEqual(result.status, "generated")
        self.assertIsNone(result.error)
        self.assertEqual(dataset.loaded_ids, ["synthetic-a"])
        self.assertEqual(
            self.backend.warmup_calls,
            [
                (
                    original_image.shape,
                    {"config": self.config.fusion},
                )
            ],
        )
        self.assertEqual(self.backend.fuse_calls[0][2], (0.25, 0.75))
        self.assertIs(
            self.backend.fuse_calls[0][3]["extraction_config"],
            self.config.extraction,
        )
        stored_placement = self.repository.upsert_placement.call_args.args[0]
        self.assertIn("roi_image.npy", stored_placement.roi_image_path)
        self.assertIn("roi_segmentation.npy", stored_placement.roi_segmentation_path)
        self.repository.upsert_hybrid_sample.assert_called_once_with(result)
        self.assertEqual(
            [item.args[2] for item in self.artifact_store.save_entity_array.call_args_list],
            ["roi_image", "roi_segmentation", "image", "segmentation"],
        )

    def test_placement_randomness_is_repeatable(self):
        hybrid = self._hybrid()
        placement = self._placement()
        dataset = _SyntheticDataset()
        self.repository.get_original_sample.return_value = SimpleNamespace(
            image_path="original-image.npy",
            segmentation_path=None,
        )
        self.repository.list_placements.return_value = [placement]
        self.artifact_store.load_array.return_value = np.zeros(
            (1, 8, 8),
            dtype=np.float32,
        )

        self.service._materialize_hybrid(hybrid, dataset, self.backend)
        self.service._materialize_hybrid(hybrid, dataset, self.backend)

        self.assertEqual(self.backend.random_values[0], self.backend.random_values[1])

    def test_materialize_records_failures_and_respects_raise_on_error(self):
        successful = self._hybrid("hybrid-success")
        broken = self._hybrid("hybrid-broken")
        self.repository.list_hybrid_samples.return_value = [successful, broken]
        self.datasets.synthetic_anomalies.return_value = _SyntheticDataset()

        with (
            patch(
                "hybrid_sample_generator.fusion.service.tqdm",
                side_effect=lambda values, **_kwargs: values,
            ),
            patch.object(
                self.service,
                "_materialize_hybrid",
                side_effect=[successful, ValueError("broken placement")],
            ),
        ):
            generated = self.service.materialize(raise_on_error=False)

        self.assertEqual(generated, [successful])
        failed = self.repository.upsert_hybrid_sample.call_args.args[0]
        self.assertEqual(failed.id, "hybrid-broken")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error, "ValueError: broken placement")
        self.assertIsNone(failed.image_path)

        self.repository.reset_mock()
        self.repository.list_hybrid_samples.return_value = [broken]
        with (
            patch(
                "hybrid_sample_generator.fusion.service.tqdm",
                side_effect=lambda values, **_kwargs: values,
            ),
            patch.object(
                self.service,
                "_materialize_hybrid",
                side_effect=ValueError("broken placement"),
            ),
            self.assertRaisesRegex(RuntimeError, "hybrid-broken.*broken placement"),
        ):
            self.service.materialize()

    def test_materialize_hybrid_requires_a_placement(self):
        hybrid = self._hybrid()
        self.repository.get_original_sample.return_value = SimpleNamespace(
            image_path="original-image.npy",
            segmentation_path=None,
        )
        self.repository.list_placements.return_value = []
        self.artifact_store.load_array.return_value = np.zeros(
            (1, 8, 8),
            dtype=np.float32,
        )

        with self.assertRaisesRegex(ValueError, "at least one placement"):
            self.service._materialize_hybrid(
                hybrid,
                _SyntheticDataset(),
                self.backend,
            )

    def test_mask_like_image_supports_channel_and_spatial_masks(self):
        image = np.zeros((3, 4, 5), dtype=np.float32)
        full = np.ones_like(image, dtype=np.uint8)
        single_channel = np.ones((1, 4, 5), dtype=np.uint8)
        spatial = np.ones((4, 5), dtype=np.uint8)

        full_result = _mask_like_image(full, image)
        self.assertTrue(np.array_equal(full_result, full))
        self.assertIsNot(full_result, full)
        self.assertEqual(_mask_like_image(single_channel, image).shape, image.shape)
        self.assertEqual(_mask_like_image(spatial, image).shape, image.shape)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _mask_like_image(np.zeros((2, 4, 5)), image)

    @staticmethod
    def _hybrid(record_id="hybrid-a"):
        return HybridSample(
            id=record_id,
            original_sample_id="original-a",
            variant_index=0,
        )

    @staticmethod
    def _placement():
        return Placement(
            id="placement-a",
            hybrid_sample_id="hybrid-a",
            synthetic_anomaly_id="synthetic-a",
            order_index=0,
            spatial_dimensions=2,
            position_z=None,
            position_y=0.25,
            position_x=0.75,
        )


if __name__ == "__main__":
    unittest.main()
