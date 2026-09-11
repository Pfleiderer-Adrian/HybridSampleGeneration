"""Tests for classical alpha-mask construction."""

import unittest

import numpy as np

from hybrid_sample_generator.fusion.classical.alpha import (
    get_alpha_mask_2d,
    get_alpha_mask_3d,
)
from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.randomness import seeded_random


class ClassicalAlphaTests(unittest.TestCase):
    def test_2d_alpha_is_bounded_and_zero_outside_support(self):
        config = Config(
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

    def test_3d_alpha_preserves_shape_and_empty_slices(self):
        config = Config(fusion_variation=False, upsampling_factor=1)
        anomaly = np.ones((3, 5, 5), dtype=np.float32)
        support = np.zeros((3, 5, 5), dtype=np.uint8)
        support[1, 1:4, 1:4] = 1

        alpha = get_alpha_mask_3d(anomaly, config, support)

        self.assertEqual(alpha.shape, anomaly.shape)
        self.assertTrue(np.all(alpha[0] == 0.0))
        self.assertGreater(float(alpha[1].max()), 0.0)
        self.assertTrue(np.all(alpha[2] == 0.0))

    def test_variation_is_repeatable_for_seeded_execution(self):
        config = Config(fusion_variation=True, upsampling_factor=1)
        anomaly = np.ones((5, 5), dtype=np.float32)
        support = np.ones((5, 5), dtype=np.uint8)

        with seeded_random(42):
            first = get_alpha_mask_2d(anomaly, config, support)
        with seeded_random(42):
            second = get_alpha_mask_2d(anomaly, config, support)

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
