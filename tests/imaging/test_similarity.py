"""Tests for the similarity score used by feedback generation."""

import unittest

import numpy as np

from hybrid_sample_generator.imaging.similarity import ssim_01


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(0).random((16, 16))

    def test_identical_images_score_one(self):
        self.assertAlmostEqual(ssim_01(self.image, self.image), 1.0)
        constant = np.full((4, 4), 0.3)
        self.assertEqual(ssim_01(constant, constant), 1.0)

    def test_score_decreases_with_distortion_and_stays_in_unit_interval(self):
        rng = np.random.default_rng(1)
        slightly = self.image + 0.02 * rng.standard_normal(self.image.shape)
        strongly = self.image + 0.5 * rng.standard_normal(self.image.shape)
        inverted = self.image.max() - self.image

        scores = [ssim_01(self.image, other) for other in (slightly, strongly, inverted)]
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], 0.9)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        # Negative structural similarity is clipped to zero.
        self.assertEqual(scores[2], 0.0)

    def test_different_constant_images_score_zero(self):
        self.assertEqual(ssim_01(np.zeros((4, 4)), np.zeros((4, 4))), 1.0)
        self.assertLess(ssim_01(np.zeros((4, 4)), np.ones((4, 4))), 0.01)

    def test_channel_first_arrays_average_over_the_first_axis(self):
        other = self.image[::-1].copy()
        stacked = ssim_01(np.stack([self.image, self.image]), np.stack([self.image, other]))
        self.assertAlmostEqual(stacked, (1.0 + ssim_01(self.image, other)) / 2)
        volume = np.stack([self.image, other])[None]
        self.assertAlmostEqual(ssim_01(volume, volume), 1.0)

    def test_score_is_symmetric_and_rejects_shape_mismatch(self):
        other = np.random.default_rng(2).random((16, 16))
        self.assertAlmostEqual(ssim_01(self.image, other), ssim_01(other, self.image))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            ssim_01(self.image, self.image[:8])


if __name__ == "__main__":
    unittest.main()
