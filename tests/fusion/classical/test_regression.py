"""Numerical regression tests protecting the classical fusion contract."""

import unittest

import numpy as np

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.fusion.classical import ClassicalFusionBackend, Config


class ClassicalNumericalRegressionTests(unittest.TestCase):
    def setUp(self):
        self.backend = ClassicalFusionBackend(
            Config(
                fusion_variation=False,
                fusion_normalization_border_width=None,
                upsampling_factor=1,
            )
        )

    def test_exact_2d_image_segmentation_and_roi(self):
        control = np.arange(49, dtype=np.float32).reshape(1, 7, 7) / 100
        output = self.backend.fuse(
            self._sample_2d(),
            control,
            (0.5, 0.5),
            extraction_config=self._extraction_config(2),
        )

        expected_image = control.copy()
        expected_image[0, 2:5, 2:5] = np.array(
            [
                [0.16, 0.514, 0.18],
                [0.606, 0.768, 0.69],
                [0.30, 0.462, 0.32],
            ],
            dtype=np.float32,
        )
        expected_segmentation = np.zeros_like(control, dtype=np.uint8)
        expected_segmentation[0, 2:5, 2:5] = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )

        np.testing.assert_allclose(output.image, expected_image, atol=1e-7)
        np.testing.assert_array_equal(output.segmentation, expected_segmentation)
        np.testing.assert_allclose(output.roi, expected_image[:, 2:7, 2:7], atol=1e-7)
        np.testing.assert_array_equal(
            output.roi_mask, expected_segmentation[:, 2:7, 2:7]
        )

    def test_exact_boundary_placement_and_shifted_roi(self):
        control = np.arange(49, dtype=np.float32).reshape(1, 7, 7) / 100
        output = self.backend.fuse(
            self._sample_2d(),
            control,
            (0.0, 0.0),
            extraction_config=self._extraction_config(2),
        )

        expected_image = control.copy()
        expected_image[0, :3, :3] = np.array(
            [
                [0.0, 0.482, 0.02],
                [0.574, 0.736, 0.658],
                [0.14, 0.43, 0.16],
            ],
            dtype=np.float32,
        )
        expected_segmentation = np.zeros_like(control, dtype=np.uint8)
        expected_segmentation[0, :3, :3] = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )

        np.testing.assert_allclose(output.image, expected_image, atol=1e-7)
        np.testing.assert_array_equal(output.segmentation, expected_segmentation)
        np.testing.assert_allclose(output.roi, expected_image[:, :5, :5], atol=1e-7)
        np.testing.assert_array_equal(
            output.roi_mask, expected_segmentation[:, :5, :5]
        )

    def test_exact_3d_image_segmentation_and_roi(self):
        control = np.arange(343, dtype=np.float32).reshape(1, 7, 7, 7) / 1000
        output = self.backend.fuse(
            self._sample_3d(),
            control,
            (0.5, 0.5, 0.5),
            extraction_config=self._extraction_config(3),
        )

        expected_image = control.copy()
        expected_image[0, 3, 2:5, 2:5] = np.array(
            [
                [0.29426155, 0.4328, 0.513],
                [0.474, 0.67420006, 0.55439997],
                [0.3954, 0.5156, 0.5958],
            ],
            dtype=np.float32,
        )
        expected_segmentation = np.zeros_like(control, dtype=np.uint8)
        expected_segmentation[0, 3, 2:5, 2:5] = 1

        np.testing.assert_allclose(output.image, expected_image, atol=1e-7)
        np.testing.assert_array_equal(output.segmentation, expected_segmentation)
        np.testing.assert_allclose(
            output.roi, expected_image[:, 2:7, 2:7, 2:7], atol=1e-7
        )
        np.testing.assert_array_equal(
            output.roi_mask, expected_segmentation[:, 2:7, 2:7, 2:7]
        )

    def test_exact_background_preservation(self):
        backend = ClassicalFusionBackend(
            Config(
                fusion_variation=False,
                fusion_normalization_border_width=None,
                upsampling_factor=1,
                fusion_keep_bg=True,
                fusion_bg_value=0.0,
                fusion_relative_bg_threshold=0.0,
                fusion_bg_exterior_only=True,
            )
        )
        control = np.zeros((1, 7, 7), dtype=np.float32)
        control[:, 1:6, 1:6] = 0.2
        output = backend.fuse(
            self._sample_2d(),
            control,
            (0.0, 0.0),
            extraction_config=self._extraction_config(2),
        )

        expected_image = control.copy()
        expected_image[0, 1, 1:3] = [0.76, 0.68]
        expected_image[0, 2, 1] = 0.44
        expected_segmentation = np.zeros_like(control, dtype=np.uint8)
        expected_segmentation[0, 1, 1:3] = 1
        expected_segmentation[0, 2, 1] = 1

        np.testing.assert_allclose(output.image, expected_image, atol=1e-7)
        np.testing.assert_array_equal(output.segmentation, expected_segmentation)
        np.testing.assert_allclose(output.roi, expected_image[:, :5, :5], atol=1e-7)
        np.testing.assert_array_equal(
            output.roi_mask, expected_segmentation[:, :5, :5]
        )

    @staticmethod
    def _sample_2d():
        anomaly = np.zeros((1, 5, 5), dtype=np.float32)
        mask = np.zeros((1, 5, 5), dtype=np.uint8)
        anomaly[0, 1:4, 1:4] = np.array(
            [[0.0, 0.6, 0.0], [0.7, 0.9, 0.8], [0.0, 0.5, 0.0]],
            dtype=np.float32,
        )
        mask[0, 1:4, 1:4] = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )
        return {
            "synth_anomaly": anomaly,
            "anomaly_meta": {"scale_factor": (1.0, 1.0)},
            "tgt_mask": mask,
            "anomaly_roi": anomaly.copy(),
            "anomaly_roi_mask": mask.copy(),
        }

    @staticmethod
    def _sample_3d():
        anomaly = np.zeros((1, 5, 5, 5), dtype=np.float32)
        mask = np.zeros((1, 5, 5, 5), dtype=np.uint8)
        anomaly[0, 2, 1:4, 1:4] = np.array(
            [[0.4, 0.5, 0.6], [0.55, 0.8, 0.65], [0.45, 0.6, 0.7]],
            dtype=np.float32,
        )
        mask[0, 2, 1:4, 1:4] = 1
        return {
            "synth_anomaly": anomaly,
            "anomaly_meta": {"scale_factor": (1.0, 1.0, 1.0)},
            "tgt_mask": mask,
            "anomaly_roi": anomaly.copy(),
            "anomaly_roi_mask": mask.copy(),
        }

    @staticmethod
    def _extraction_config(spatial_ndim):
        config = ExtractionConfiguration((1, *((5,) * spatial_ndim)))
        config.roi.fixed_size = (5,) * spatial_ndim
        return config


if __name__ == "__main__":
    unittest.main()
