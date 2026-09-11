"""Tests for anomaly extraction, normalization, resampling, and ROIs."""

import unittest
from unittest.mock import patch

import numpy as np

from hybrid_sample_generator.extraction.normalization import (
    add_background_noise_floor,
    normalize_anomaly,
)
from hybrid_sample_generator.imaging.resampling import (
    resize_and_pad_2d,
    resize_and_pad_3d,
    spatial_target_size,
)
from hybrid_sample_generator.imaging.roi import (
    crop_cube_clip,
    crop_square_clip,
    dynamic_roi_size,
)


class RoiTests(unittest.TestCase):
    def test_dynamic_roi_size_supports_scalar_and_per_axis_settings(self):
        self.assertEqual(dynamic_roi_size((10, 20), 2, 0.5, 8), [15, 30])
        self.assertEqual(
            dynamic_roi_size((10, 20, 30), (1, 2, 3), (0.1, 0.2, 0.3), (12, 24, 40)),
            [12, 24, 40],
        )

    def test_roi_crops_shift_inside_2d_and_3d_bounds(self):
        image = np.arange(1 * 6 * 8).reshape(1, 6, 8)
        volume = np.arange(1 * 5 * 6 * 7).reshape(1, 5, 6, 7)

        crop_2d = crop_square_clip(image, (0, 0), (4, 5), centroid_is_normalized=False)
        crop_3d = crop_cube_clip(volume, (4, 5, 6), (3, 4, 5), centroid_is_normalized=False)

        self.assertEqual(crop_2d.shape, (1, 4, 5))
        self.assertTrue(np.array_equal(crop_2d, image[:, :4, :5]))
        self.assertEqual(crop_3d.shape, (1, 3, 4, 5))
        self.assertTrue(np.array_equal(crop_3d, volume[:, -3:, -4:, -5:]))


class ResamplingTests(unittest.TestCase):
    def test_resize_and_pad_preserves_channel_axis_and_reports_scale(self):
        image = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
        volume = np.arange(2 * 6 * 4 * 2, dtype=np.float32).reshape(2, 6, 4, 2)

        resized_2d, scale_2d = resize_and_pad_2d(image, (4, 6))
        resized_3d, scale_3d = resize_and_pad_3d(volume, (3, 6, 4))

        self.assertEqual(resized_2d.shape, (2, 4, 6))
        self.assertEqual(scale_2d, (0.5, 1.0))
        self.assertEqual(resized_3d.shape, (2, 3, 6, 4))
        self.assertEqual(scale_3d, (0.5, 1.0, 1.0))

    def test_foreground_mask_must_match_spatial_shape(self):
        with self.assertRaisesRegex(ValueError, "foreground_mask shape"):
            resize_and_pad_2d(np.zeros((1, 4, 4)), (4, 4), foreground_mask=np.zeros((3, 3)))

    def test_spatial_target_size_accepts_optional_channel_dimension(self):
        self.assertEqual(spatial_target_size((8, 9), 2), (8, 9))
        self.assertEqual(spatial_target_size((1, 8, 9), 2), (8, 9))
        with self.assertRaises(ValueError):
            spatial_target_size((1, 2, 3, 4), 2)


class NormalizationTests(unittest.TestCase):
    def test_zscore_and_robust_normalization_return_inversion_metadata(self):
        values = np.array([0.0, 1.0, 2.0, 8.0], dtype=np.float32)

        zscore, zscore_meta = normalize_anomaly(values, "zscore", 1e-8)
        robust, robust_meta = normalize_anomaly(values, "zscore_median", 1e-8)

        self.assertAlmostEqual(float(zscore.mean()), 0.0, places=6)
        self.assertEqual(zscore_meta["norm_type"], "zscore")
        self.assertEqual(robust_meta["norm_type"], "zscore_median")
        self.assertIn("norm_median", robust_meta)
        self.assertTrue(np.isfinite(robust).all())

    def test_noise_floor_changes_only_background_values(self):
        image = np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
        with patch(
            "hybrid_sample_generator.extraction.normalization.np.random.normal",
            return_value=np.ones_like(image),
        ):
            noisy = add_background_noise_floor(image)

        self.assertTrue(np.array_equal(noisy[image > 0], image[image > 0]))
        self.assertTrue(np.array_equal(noisy[image == 0], np.ones(2, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
