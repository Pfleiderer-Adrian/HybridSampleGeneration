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

    def test_alpha_rises_from_the_border_and_saturates_by_steepness(self):
        support = np.zeros((21, 21), dtype=np.uint8)
        support[2:19, 2:19] = 1   # largest border distance: 9 px in the center
        anomaly = np.ones((21, 21), dtype=np.float32)
        profiles = {}
        for steepness in (1, 3, 12):
            config = Config(
                fusion_variation=False, upsampling_factor=1, max_alpha=1.0, sq=1.0,
                steepness_factor=steepness,
            )
            profiles[steepness] = get_alpha_mask_2d(anomaly, config, support)[10, 2:11]

        for profile in profiles.values():
            self.assertTrue(np.all(np.diff(profile) >= 0))
            self.assertAlmostEqual(float(profile[-1]), 1.0)
        # Border distance 1 of 9: alpha is 1/9, 3/9 and 1 for steepness 1, 3 and 12.
        np.testing.assert_allclose(
            [profiles[1][0], profiles[3][0], profiles[12][0]], [1 / 9, 3 / 9, 1.0], atol=1e-6
        )

    def test_thin_parts_keep_weight_only_with_a_high_steepness(self):
        # A thick block with a thin, 3 px wide arm.
        support = np.zeros((24, 40), dtype=np.uint8)
        support[2:22, 2:22] = 1
        support[11:14, 22:38] = 1
        anomaly = np.ones(support.shape, dtype=np.float32)
        arm_center = (12, 30)

        def alpha(steepness):
            config = Config(
                fusion_variation=False, upsampling_factor=1, max_alpha=1.0, sq=2.0,
                steepness_factor=steepness,
            )
            return float(get_alpha_mask_2d(anomaly, config, support)[arm_center])

        self.assertLess(alpha(3), 0.5)
        self.assertAlmostEqual(alpha(12), 1.0)
        # The package default keeps the arm at full weight.
        default = Config(fusion_variation=False, upsampling_factor=1)
        self.assertAlmostEqual(
            float(get_alpha_mask_2d(anomaly, default, support)[arm_center]), default.max_alpha
        )

    def test_sq_and_max_alpha_shape_the_transition(self):
        support = np.zeros((21, 21), dtype=np.uint8)
        support[2:19, 2:19] = 1
        anomaly = np.ones((21, 21), dtype=np.float32)

        def border_alpha(**settings):
            config = Config(fusion_variation=False, upsampling_factor=1, steepness_factor=3, **settings)
            return get_alpha_mask_2d(anomaly, config, support)

        linear = border_alpha(max_alpha=1.0, sq=1.0)
        late = border_alpha(max_alpha=1.0, sq=2.0)
        early = border_alpha(max_alpha=1.0, sq=0.5)
        capped = border_alpha(max_alpha=0.6, sq=1.0)

        self.assertLess(float(late[10, 2]), float(linear[10, 2]))
        self.assertGreater(float(early[10, 2]), float(linear[10, 2]))
        self.assertAlmostEqual(float(capped.max()), 0.6)
        np.testing.assert_allclose(capped, 0.6 * linear, atol=1e-6)

    def test_upsampling_keeps_shape_support_and_range(self):
        support = np.zeros((12, 12), dtype=np.uint8)
        support[3:9, 2:10] = 1
        anomaly = np.ones((12, 12), dtype=np.float32)
        for factor in (1, 2, 3):
            config = Config(fusion_variation=False, upsampling_factor=factor)
            alpha = get_alpha_mask_2d(anomaly, config, support)
            self.assertEqual(alpha.shape, support.shape)
            self.assertTrue(np.all(alpha[support == 0] == 0.0))
            self.assertGreater(float(alpha.max()), 0.0)
            self.assertLessEqual(float(alpha.max()), config.max_alpha + 1e-6)

    def test_sobel_path_limits_the_support_to_the_visible_object(self):
        # The target mask is larger than the bright object inside the cutout.
        anomaly = np.zeros((32, 32), dtype=np.float32)
        anomaly[10:22, 10:22] = 1.0
        support = np.zeros((32, 32), dtype=np.uint8)
        support[4:28, 4:28] = 1

        plain = get_alpha_mask_2d(
            anomaly, Config(fusion_variation=False, upsampling_factor=1), support
        )
        sobel = get_alpha_mask_2d(
            anomaly,
            Config(
                fusion_variation=False, upsampling_factor=1,
                fusion_use_sobel_for_alpha_mask=True, dilation_size=2, shave_pixels=1,
            ),
            support,
        )
        shaved = get_alpha_mask_2d(
            anomaly,
            Config(
                fusion_variation=False, upsampling_factor=1,
                fusion_use_sobel_for_alpha_mask=True, dilation_size=2, shave_pixels=3,
            ),
            support,
        )

        self.assertEqual(int(np.count_nonzero(plain)), int(support.sum()))
        self.assertGreater(float(sobel[16, 16]), 0.0)
        self.assertLess(int(np.count_nonzero(sobel)), int(np.count_nonzero(plain)))
        self.assertLess(int(np.count_nonzero(shaved)), int(np.count_nonzero(sobel)))
        # Nothing far from the object and nothing outside the target mask is blended.
        self.assertEqual(float(sobel[5, 5]), 0.0)
        self.assertTrue(np.all(sobel[support == 0] == 0.0))

    def test_sobel_path_never_leaves_the_target_mask_and_handles_flat_images(self):
        anomaly = np.zeros((16, 16), dtype=np.float32)
        anomaly[2:14, 2:14] = 1.0
        support = np.zeros((16, 16), dtype=np.uint8)
        support[6:10, 6:10] = 1
        config = Config(
            fusion_variation=False, upsampling_factor=1,
            fusion_use_sobel_for_alpha_mask=True, shave_pixels=0,
        )
        alpha = get_alpha_mask_2d(anomaly, config, support)
        self.assertTrue(np.all(alpha[support == 0] == 0.0))

        flat = get_alpha_mask_2d(np.ones((16, 16), dtype=np.float32), config, support)
        self.assertEqual(float(flat.max()), 0.0)

    def test_variation_changes_parameters_within_the_valid_range(self):
        support = np.zeros((21, 21), dtype=np.uint8)
        support[2:19, 2:19] = 1
        anomaly = np.ones((21, 21), dtype=np.float32)
        config = Config(fusion_variation=True, upsampling_factor=1, max_alpha=0.8)
        peaks = set()
        with seeded_random(7):
            for _ in range(20):
                alpha = get_alpha_mask_2d(anomaly, config, support)
                peaks.add(round(float(alpha.max()), 4))
                self.assertLessEqual(float(alpha.max()), 1.0)
                self.assertGreaterEqual(float(alpha.min()), 0.0)
        self.assertGreater(len(peaks), 5)

        config.selected_confidence = "50%"
        with self.assertRaisesRegex(ValueError, "selected_confidence"):
            get_alpha_mask_2d(anomaly, config, support)

    def test_empty_support_gives_an_empty_alpha_mask(self):
        config = Config(fusion_variation=False, upsampling_factor=1)
        empty = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(float(get_alpha_mask_2d(np.ones((8, 8), dtype=np.float32), config, empty).max()), 0.0)
        self.assertEqual(float(get_alpha_mask_3d(np.ones((2, 8, 8), dtype=np.float32), config, np.zeros((2, 8, 8))).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
