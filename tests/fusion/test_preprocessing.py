"""Tests for backend-independent fusion preprocessing."""

import unittest

import numpy as np

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

if __name__ == "__main__":
    unittest.main()
