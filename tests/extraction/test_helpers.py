"""Tests for anomaly extraction, normalization, resampling, and ROIs."""

import unittest
from unittest.mock import patch

import numpy as np

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.extraction.extraction import crop_and_center_anomalies
from hybrid_sample_generator.extraction.normalization import (
    add_background_noise_floor,
    normalize_anomaly,
)
from hybrid_sample_generator.imaging.resampling import (
    resize_and_pad,
    spatial_target_size,
)
from hybrid_sample_generator.imaging.roi import (
    crop_spatial_clip,
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

        crop_2d = crop_spatial_clip(image, (0, 0), (4, 5), centroid_is_normalized=False)
        crop_3d = crop_spatial_clip(volume, (4, 5, 6), (3, 4, 5), centroid_is_normalized=False)

        self.assertEqual(crop_2d.shape, (1, 4, 5))
        self.assertTrue(np.array_equal(crop_2d, image[:, :4, :5]))
        self.assertEqual(crop_3d.shape, (1, 3, 4, 5))
        self.assertTrue(np.array_equal(crop_3d, volume[:, -3:, -4:, -5:]))


class ResamplingTests(unittest.TestCase):
    def test_resize_and_pad_preserves_channel_axis_and_reports_scale(self):
        image = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
        volume = np.arange(2 * 6 * 4 * 2, dtype=np.float32).reshape(2, 6, 4, 2)

        resized_2d, scale_2d = resize_and_pad(image, (4, 6))
        resized_3d, scale_3d = resize_and_pad(volume, (3, 6, 4))

        self.assertEqual(resized_2d.shape, (2, 4, 6))
        self.assertEqual(scale_2d, (0.5, 0.5))
        self.assertEqual(resized_3d.shape, (2, 3, 6, 4))
        self.assertEqual(scale_3d, (0.5, 0.5, 0.5))

    def test_aspect_preserving_padding_stays_outside_the_mask(self):
        mask = np.ones((1, 8, 4), dtype=np.uint8)

        resized, scale = resize_and_pad(
            mask,
            (4, 6),
            order=0,
            padding_value=0,
        )

        self.assertEqual(scale, (0.5, 0.5))
        self.assertTrue(np.all(resized[:, :, :2] == 0))
        self.assertTrue(np.all(resized[:, :, 2:4] == 1))
        self.assertTrue(np.all(resized[:, :, 4:] == 0))
        foreground = np.any(resized > 0, axis=0)
        coordinates = np.where(foreground)
        cropped_shape = tuple(
            int(axis.max() - axis.min() + 1) for axis in coordinates
        )
        restored_shape = tuple(
            round(size / factor) for size, factor in zip(cropped_shape, scale)
        )
        self.assertEqual(restored_shape, mask.shape[1:])

    def test_resize_and_pad_can_stretch_axes_independently(self):
        image = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)

        resized, scale = resize_and_pad(
            image,
            (4, 6),
            preserve_aspect_ratio=False,
        )

        self.assertEqual(resized.shape, (2, 4, 6))
        self.assertEqual(scale, (0.5, 1.0))

    def test_foreground_mask_must_match_spatial_shape(self):
        with self.assertRaisesRegex(ValueError, "foreground_mask shape"):
            resize_and_pad(np.zeros((1, 4, 4)), (4, 4), foreground_mask=np.zeros((3, 3)))

    def test_spatial_target_size_accepts_optional_channel_dimension(self):
        self.assertEqual(spatial_target_size((8, 9), 2), (8, 9))
        self.assertEqual(spatial_target_size((1, 8, 9), 2), (8, 9))
        with self.assertRaises(ValueError):
            spatial_target_size((1, 2, 3, 4), 2)


class ExtractionTests(unittest.TestCase):
    def test_connected_component_extraction_supports_2d_and_3d(self):
        for spatial_shape in ((6, 8), (5, 6, 7)):
            with self.subTest(spatial_shape=spatial_shape):
                image = np.arange(np.prod(spatial_shape), dtype=np.float32).reshape(
                    (1, *spatial_shape)
                )
                segmentation = np.zeros((1, *spatial_shape), dtype=np.uint8)
                region = tuple(slice(1, 3) for _ in spatial_shape)
                segmentation[(0, *region)] = 1
                config = ExtractionConfiguration((1, *spatial_shape))
                config.min_coverage_ratio = 0.0
                config.add_background_noise = False
                config.normalization = None
                config.roi.fixed_size = tuple(4 for _ in spatial_shape)

                anomalies, rois, masks, roi_masks = crop_and_center_anomalies(
                    image, segmentation, config
                )

                self.assertEqual(len(anomalies), 1)
                self.assertEqual(anomalies[0][0].shape, image.shape)
                self.assertEqual(masks[0].shape, segmentation.shape)
                self.assertEqual(rois[0].shape[1:], config.roi.fixed_size)
                self.assertEqual(roi_masks[0].shape, rois[0].shape)
                self.assertEqual(
                    anomalies[0][1]["centroid_norm"],
                    tuple(2 / size for size in spatial_shape),
                )


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
