"""Tests for fusion preprocessing, masks, and intensity helpers."""

import unittest

import numpy as np

from hybrid_sample_generator.fusion.classical.alpha import get_alpha_mask_2d
from hybrid_sample_generator.fusion.classical.configuration import Config as ClassicalConfig
from hybrid_sample_generator.fusion.classical.intensity import infer_output_intensity_bounds
from hybrid_sample_generator.fusion.learned_residual_alpha.preprocessing import support_mask
from hybrid_sample_generator.fusion.preprocessing import (
    denormalize_anomaly,
    inverse_extraction_scale,
    spatial_label_mask,
    validate_position,
)


class FusionPreprocessingTests(unittest.TestCase):
    def test_inverse_extraction_scale_accepts_scalar_and_per_axis_values(self):
        self.assertEqual(inverse_extraction_scale(2.0, 2), (0.5, 0.5))
        self.assertEqual(inverse_extraction_scale((2.0, 4.0), 2), (0.5, 0.25))

    def test_inverse_extraction_scale_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            inverse_extraction_scale((1.0, 0.0), 2)
        with self.assertRaises(ValueError):
            inverse_extraction_scale((1.0, 2.0, 3.0), 2)

    def test_denormalize_and_spatial_validation(self):
        anomaly = np.array([[-1.0, 1.0]], dtype=np.float32)
        restored = denormalize_anomaly(
            anomaly,
            {"norm_type": "zscore", "norm_mean": 10.0, "norm_std": 2.0},
        )
        np.testing.assert_allclose(restored, [[8.0, 12.0]])
        np.testing.assert_array_equal(
            spatial_label_mask(np.array([[[0, 1]], [[2, 0]]]), 2),
            [[2, 1]],
        )
        self.assertEqual(validate_position((0.25, 0.75), 2), (0.25, 0.75))


class ClassicalFusionHelperTests(unittest.TestCase):
    def test_alpha_mask_is_bounded_and_zero_outside_support(self):
        config = ClassicalConfig(
            fusion_variation=False,
            fusion_use_sobel_for_alpha_mask=False,
            upsampling_factor=1,
        )
        anomaly = np.ones((5, 5), dtype=np.float32)
        support = np.zeros((5, 5), dtype=np.uint8)
        support[1:4, 1:4] = 1

        alpha = get_alpha_mask_2d(anomaly, config, support)

        self.assertEqual(alpha.dtype, np.float32)
        self.assertAlmostEqual(float(alpha.max()), config.max_alpha)
        self.assertTrue(np.all(alpha[support == 0] == 0.0))

    def test_output_bounds_are_inferred_only_for_known_ranges(self):
        self.assertEqual(infer_output_intensity_bounds([0.0, 1.0]), (0.0, 1.0))
        self.assertEqual(infer_output_intensity_bounds([0.0, 255.0]), (0.0, 255.0))
        self.assertIsNone(infer_output_intensity_bounds([-2.0, 300.0]))


class LearnedFusionHelperTests(unittest.TestCase):
    def test_support_mask_dilates_by_requested_border(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[2, 2] = 1.0

        support = support_mask(mask, border_width=1, spatial_dims=2)

        self.assertEqual(int(np.count_nonzero(support)), 9)
        self.assertEqual(support.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
