"""Tests for backend-independent fusion background preservation."""

import unittest

import numpy as np

from hybrid_sample_generator.fusion.background import (
    control_background_mask,
    keep_control_background_after_fusion,
)


class FusionBackgroundTests(unittest.TestCase):
    def test_exterior_filter_excludes_enclosed_low_region(self):
        image = np.ones((1, 7, 7), dtype=np.float32)
        image[:, 0, :] = 0.0
        image[:, -1, :] = 0.0
        image[:, :, 0] = 0.0
        image[:, :, -1] = 0.0
        image[:, 3, 3] = 0.0

        all_background = control_background_mask(
            image, bg_value=0.0, exterior_only=False
        )
        exterior_background = control_background_mask(
            image, bg_value=0.0, exterior_only=True
        )

        self.assertTrue(all_background[3, 3])
        self.assertFalse(exterior_background[3, 3])
        self.assertTrue(exterior_background[0, 0])

    def test_preservation_restores_image_and_clears_segmentation(self):
        control = np.zeros((2, 4, 4), dtype=np.float32)
        fused = np.ones_like(control)
        segmentation = np.ones_like(control, dtype=np.uint8)
        background = np.zeros((4, 4), dtype=bool)
        background[0, :] = True

        image, mask = keep_control_background_after_fusion(
            fused, segmentation, control, background
        )

        self.assertTrue(np.all(image[:, 0, :] == 0.0))
        self.assertTrue(np.all(image[:, 1:, :] == 1.0))
        self.assertTrue(np.all(mask[:, 0, :] == 0))
        self.assertTrue(np.all(mask[:, 1:, :] == 1))

    def test_background_helpers_validate_shapes_and_thresholds(self):
        image = np.zeros((1, 4, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "relative_bg_threshold"):
            control_background_mask(image, 0.0, -0.1)
        with self.assertRaisesRegex(ValueError, "fused_image shape"):
            keep_control_background_after_fusion(
                np.zeros((1, 3, 4)), image, image, np.zeros((4, 4))
            )
        with self.assertRaisesRegex(ValueError, "background_mask shape"):
            keep_control_background_after_fusion(
                image, image, image, np.zeros((3, 4))
            )


def _object_on_dark_background(size=24):
    """A bright textured object in the middle of a dark, slightly noisy background."""
    rng = np.random.default_rng(0)
    image = (0.05 + 0.005 * rng.random((1, size, size))).astype(np.float32)
    image[0, 6:18, 6:18] = 0.6 + 0.1 * rng.random((12, 12))
    return image


class AutomaticBackgroundDetectionTests(unittest.TestCase):
    def test_background_level_is_taken_from_the_image_border(self):
        image = _object_on_dark_background()
        background = control_background_mask(image, None, 0.1)

        self.assertEqual(background.shape, (24, 24))
        self.assertTrue(background[0, 0])
        self.assertTrue(background[5, 12])
        self.assertFalse(background[6:18, 6:18].any())
        self.assertEqual(int(background.sum()), 24 * 24 - 144)

    def test_a_rare_dark_outlier_does_not_define_the_background_level(self):
        rng = np.random.default_rng(1)
        image = (0.05 + 0.005 * rng.random((1, 128, 128))).astype(np.float32)
        image[0, 40:90, 40:90] = 0.6
        image[0, 0, 0] = -5.0   # one dead pixel among 508 border pixels

        background = control_background_mask(image, None, 0.1)

        # The level is a low percentile of the border, not its minimum.
        self.assertEqual(int(background.sum()), 128 * 128 - 50 * 50)
        self.assertFalse(background[40:90, 40:90].any())

    def test_a_frequent_minimum_is_used_directly(self):
        image = _object_on_dark_background()
        image[0][image[0] < 0.3] = 0.05   # a perfectly flat background
        exact = control_background_mask(image, None, 0.0)
        self.assertEqual(int(exact.sum()), 24 * 24 - 144)

    def test_threshold_widens_the_background_towards_the_brightest_values(self):
        image = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
        areas = [
            int(control_background_mask(image, 0.0, threshold, exterior_only=False).sum())
            for threshold in (0.0, 0.25, 0.5, 1.0)
        ]
        self.assertEqual(areas, sorted(areas))
        self.assertLess(areas[0], 3)
        self.assertAlmostEqual(areas[2] / 256, 0.5, delta=0.02)
        self.assertGreater(areas[3], 250)
        # Without a threshold only the given background value itself counts.
        self.assertEqual(
            int(control_background_mask(image, 0.0, None, exterior_only=False).sum()), areas[0]
        )

    def test_every_channel_must_be_background(self):
        image = np.zeros((2, 8, 8), dtype=np.float32)
        image[0, 2:6, 2:6] = 1.0   # object only in the first channel
        image[1, 4:8, 4:8] = 1.0   # and a different one in the second
        background = control_background_mask(image, 0.0, 0.1, exterior_only=False)
        expected = (image[0] == 0) & (image[1] == 0)
        np.testing.assert_array_equal(background, expected)

    def test_exterior_filter_works_per_slice_in_volumes(self):
        volume = np.ones((1, 3, 9, 9), dtype=np.float32)
        volume[0, :, 0, :] = 0.0     # background at the border of every slice
        volume[0, 1, 4, 4] = 0.0     # an enclosed dark voxel in the middle slice
        everything = control_background_mask(volume, 0.0, 0.0, exterior_only=False)
        exterior = control_background_mask(volume, 0.0, 0.0, exterior_only=True)

        self.assertEqual(exterior.shape, (3, 9, 9))
        self.assertTrue(everything[1, 4, 4])
        self.assertFalse(exterior[1, 4, 4])
        self.assertTrue(exterior[:, 0, :].all())

    def test_images_without_background_and_non_finite_values(self):
        bright = np.ones((1, 6, 6), dtype=np.float32)
        self.assertFalse(control_background_mask(bright, 0.0, 0.1).any())

        with_gaps = _object_on_dark_background()
        with_gaps[0, 0, 0] = np.nan
        background = control_background_mask(with_gaps, None, 0.1, exterior_only=False)
        self.assertFalse(background[0, 0])
        self.assertTrue(background[1, 1])

        unusable = np.full((1, 4, 4), np.nan, dtype=np.float32)
        self.assertFalse(control_background_mask(unusable, None, 0.1).any())
        with self.assertRaisesRegex(ValueError, "channel-first"):
            control_background_mask(np.zeros(4), 0.0)


class BackgroundPreservationInFusionTests(unittest.TestCase):
    def _fuse(self, backend_class, config):
        from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration

        control = np.full((1, 24, 24), 0.05, dtype=np.float32)
        control[0, :, 12:] = 0.6    # the object fills the right half
        anomaly = np.zeros((1, 8, 8), dtype=np.float32)
        mask = np.zeros((1, 8, 8), dtype=np.uint8)
        anomaly[0, 1:7, 1:7] = 0.9
        mask[0, 1:7, 1:7] = 1
        sample = {
            "synth_anomaly": anomaly,
            "anomaly_meta": {"scale_factor": (1.0, 1.0)},
            "tgt_mask": mask,
            "anomaly_roi": anomaly.copy(),
            "anomaly_roi_mask": mask.copy(),
        }
        extraction = ExtractionConfiguration((1, 8, 8))
        extraction.roi.fixed_size = (8, 8)
        # The anomaly is centered on the edge of the object: half of it lies on background.
        output = backend_class(config).fuse(sample, control, (0.5, 0.5), extraction_config=extraction)
        return control, output

    def test_anomaly_pixels_on_the_background_are_dropped(self):
        import warnings

        from hybrid_sample_generator.fusion.classical import ClassicalFusionBackend, Config as Classical
        from hybrid_sample_generator.fusion.poisson import PoissonFusionBackend, Config as Poisson

        cases = (
            (ClassicalFusionBackend, Classical(fusion_variation=False, fusion_normalization_border_width=None)),
            (PoissonFusionBackend, Poisson(fusion_normalization_border_width=None)),
        )
        for backend_class, config in cases:
            with self.subTest(backend=backend_class.__name__), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                control, plain = self._fuse(backend_class, config)
                config.fusion_keep_bg = True
                config.fusion_bg_value = None
                config.fusion_relative_bg_threshold = 0.1
                _, kept = self._fuse(backend_class, config)

                background = control[0] < 0.3
                # Without the option the anomaly covers background pixels as well.
                self.assertGreater(int(np.count_nonzero(plain.segmentation[0][background])), 0)
                # With it, the background stays untouched and unlabeled.
                np.testing.assert_array_equal(kept.image[0][background], control[0][background])
                self.assertEqual(int(np.count_nonzero(kept.segmentation[0][background])), 0)
                self.assertGreater(int(np.count_nonzero(kept.segmentation[0][~background])), 0)


if __name__ == "__main__":
    unittest.main()
