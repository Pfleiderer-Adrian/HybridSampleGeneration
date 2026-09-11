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


if __name__ == "__main__":
    unittest.main()
