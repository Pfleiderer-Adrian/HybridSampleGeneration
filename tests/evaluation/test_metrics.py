"""Tests for evaluation metrics and outlier detection."""

import unittest

import numpy as np

from hybrid_sample_generator.configuration.evaluation import EvaluationConfiguration
from hybrid_sample_generator.evaluation.metrics import (
    compute_glcm,
    get_volume_feature_diffs,
    relative_foreground_mask,
)
from hybrid_sample_generator.evaluation.outliers import find_outliers


class EvaluationMetricTests(unittest.TestCase):
    def test_glcm_is_normalized_when_valid_neighbours_exist(self):
        image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        mask = np.ones((1, 3, 3), dtype=np.uint8)
        self.assertAlmostEqual(float(compute_glcm(image, mask).sum()), 1.0)

    def test_identical_masks_have_zero_volume_and_center_differences(self):
        mask = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
        _, _, differences = get_volume_feature_diffs(mask, mask, mask, mask)
        self.assertTrue(all(value == 0 for value in differences.values()))

    def test_negative_relative_foreground_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            relative_foreground_mask(np.ones((1, 2, 2)), -0.1)

    def test_custom_outlier_threshold_is_applied(self):
        config = EvaluationConfiguration()
        config.outlier_thresholds["Contrast"]["max"] = 1.0
        entries = [{"value": 0.5, "sample": "a"}, {"value": 2.0, "sample": "b"}]
        self.assertEqual(
            find_outliers([0.5, 2.0], entries, config, "Contrast"),
            [{"value": 2.0, "sample": "b"}],
        )


if __name__ == "__main__":
    unittest.main()
