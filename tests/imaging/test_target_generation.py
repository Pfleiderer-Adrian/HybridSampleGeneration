"""Tests for target-mask creation and the global mask transforms."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.imaging.masks.global_transforms import (
    random_global_rotation_transform,
    random_global_stretch_transform,
    random_global_zoom_transform,
)
from hybrid_sample_generator.imaging.masks.target_generation import (
    target_mask_from_original_mask,
    target_mask_from_synthetic_anomaly,
)
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


def _square_mask(size=32, start=12, stop=20):
    mask = np.zeros((1, size, size), dtype=np.uint8)
    mask[0, start:stop, start:stop] = 1
    return mask


class ThresholdTargetMaskTests(unittest.TestCase):
    def setUp(self):
        # Background 0, a faint halo of 0.2 and an anomaly of 1.0.
        self.image = np.zeros((1, 8, 8), dtype=np.float32)
        self.image[0, 2:6, 2:6] = 0.2
        self.image[0, 3:5, 3:5] = 1.0

    def test_threshold_is_relative_to_the_intensity_range(self):
        for backend in (np.asarray, torch.as_tensor):
            with self.subTest(backend=backend.__name__):
                image = backend(self.image)
                low = target_mask_from_synthetic_anomaly(image, background_threshold=0.01)
                high = target_mask_from_synthetic_anomaly(image, background_threshold=0.5)
                self.assertEqual(tuple(low.shape), (8, 8))
                self.assertEqual(int(low.sum()), 16)   # halo and anomaly
                self.assertEqual(int(high.sum()), 4)   # anomaly only

    def test_range_follows_shifted_and_scaled_intensities(self):
        shifted = self.image * 50.0 - 20.0
        mask = target_mask_from_synthetic_anomaly(shifted, background_threshold=0.5)
        self.assertEqual(int(mask.sum()), 4)

    def test_channels_are_combined_by_their_maximum(self):
        image = np.zeros((2, 4, 4), dtype=np.float32)
        image[0, 0, 0] = 1.0
        image[1, 3, 3] = 1.0
        mask = target_mask_from_synthetic_anomaly(image, background_threshold=0.5)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(int(mask.sum()), 2)

    def test_constant_and_non_finite_images_have_empty_masks(self):
        constant = np.full((1, 4, 4), 0.3, dtype=np.float32)
        self.assertEqual(int(target_mask_from_synthetic_anomaly(constant, background_threshold=0.1).sum()), 0)
        non_finite = torch.full((1, 4, 4), float("nan"))
        self.assertEqual(int(target_mask_from_synthetic_anomaly(non_finite, background_threshold=0.1).sum()), 0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "synth_anomaly_image"):
            target_mask_from_synthetic_anomaly(None, background_threshold=0.1)
        with self.assertRaisesRegex(ValueError, "background_threshold"):
            target_mask_from_synthetic_anomaly(self.image, background_threshold=-0.1)

    def test_transform_generator_selects_the_derivation_by_model_type(self):
        generator = TransformGenerator(use_mask_transform=False, background_threshold=0.5)
        derived = generator.create_target_mask(synth_anomaly_image=self.image)
        self.assertEqual(int(derived.sum()), 4)
        original = _square_mask(8, 1, 3)
        np.testing.assert_array_equal(
            generator.create_target_mask(original_mask=original, conditional=True), original
        )
        explicit = np.ones((1, 8, 8), dtype=np.uint8)
        self.assertIs(generator.create_target_mask(target_mask=explicit), explicit)


class OriginalMaskTargetTests(unittest.TestCase):
    def test_augmentation_is_applied_and_tensor_properties_are_kept(self):
        mask = _square_mask()
        flipped = lambda array: array[:, ::-1].copy()

        np.testing.assert_array_equal(target_mask_from_original_mask(mask, flipped), flipped(mask))
        tensor = torch.as_tensor(mask, dtype=torch.int16)
        result = target_mask_from_original_mask(tensor, flipped)
        self.assertEqual(result.dtype, torch.int16)
        np.testing.assert_array_equal(result.numpy(), flipped(mask))
        with self.assertRaisesRegex(ValueError, "original_mask"):
            target_mask_from_original_mask(None, flipped)


class GlobalTransformTests(unittest.TestCase):
    def test_zoom_shrinks_the_mask_around_its_center(self):
        mask = _square_mask()
        rng = np.random.default_rng(0)

        half = random_global_zoom_transform(mask, min_zoom=0.5, max_zoom=0.5, rng=rng)
        same = random_global_zoom_transform(mask, min_zoom=1.0, max_zoom=1.0, rng=rng)

        self.assertEqual(half.shape, mask.shape)
        self.assertEqual(half.dtype, mask.dtype)
        self.assertEqual(int(half.sum()), 16)
        np.testing.assert_array_equal(np.argwhere(half[0]).mean(axis=0), [15.5, 15.5])
        np.testing.assert_array_equal(same, mask)

    def test_zoom_factor_is_drawn_from_the_configured_range(self):
        mask = _square_mask()
        rng = np.random.default_rng(1)
        areas = {
            int(random_global_zoom_transform(mask, min_zoom=0.5, max_zoom=1.0, rng=rng).sum())
            for _ in range(30)
        }
        self.assertGreater(len(areas), 2)
        self.assertTrue(all(16 <= area <= 64 for area in areas))

    def test_zoom_rejects_invalid_ranges(self):
        mask = _square_mask()
        for low, high in ((0.0, 0.5), (0.9, 0.5), (-0.5, 0.5)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                random_global_zoom_transform(mask, min_zoom=low, max_zoom=high)
        with self.assertRaises(ValueError):
            random_global_zoom_transform(mask[0], min_zoom=0.5, max_zoom=0.5)

    def test_stretch_scales_axes_independently_in_2d_and_3d(self):
        rng = np.random.default_rng(2)
        mask = _square_mask()
        stretched = random_global_stretch_transform(mask, min_stretch=1.5, max_stretch=1.5, rng=rng)
        self.assertEqual(int(stretched.sum()), 144)

        ratios = set()
        for _ in range(20):
            result = random_global_stretch_transform(mask, min_stretch=1.0, max_stretch=1.5, rng=rng)
            rows = np.ptp(np.where(result[0].any(axis=1))[0]) + 1
            columns = np.ptp(np.where(result[0].any(axis=0))[0]) + 1
            ratios.add((int(rows), int(columns)))
        self.assertTrue(any(rows != columns for rows, columns in ratios))

        volume = np.zeros((1, 16, 16, 16), dtype=np.uint8)
        volume[0, 6:10, 6:10, 6:10] = 1
        doubled = random_global_stretch_transform(volume, min_stretch=2.0, max_stretch=2.0, rng=rng)
        self.assertEqual(int(doubled.sum()), 512)

    def test_rotation_keeps_the_area_and_respects_the_maximum_angle(self):
        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        mask[0, 14:18, 6:26] = 1  # a horizontal bar
        rng = np.random.default_rng(3)

        np.testing.assert_array_equal(
            random_global_rotation_transform(mask, max_rotation=0.0, rng=rng), mask
        )
        heights = []
        for _ in range(20):
            rotated = random_global_rotation_transform(mask, max_rotation=20.0, rng=rng)
            self.assertAlmostEqual(int(rotated.sum()), int(mask.sum()), delta=12)
            heights.append(np.ptp(np.where(rotated[0].any(axis=1))[0]) + 1)
        # A bar of 20 px tilted by at most 20 degrees is at most about 11 px high.
        self.assertGreater(max(heights), 4)
        self.assertLessEqual(max(heights), 12)


if __name__ == "__main__":
    unittest.main()
