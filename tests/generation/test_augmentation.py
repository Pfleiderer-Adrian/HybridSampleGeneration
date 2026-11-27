"""Tests for the random offset augmentation used during generator training."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.configuration.augmentation import AugmentationConfiguration
from hybrid_sample_generator.generation.training.augmentation import (
    RandomSpatialOffset,
    TrainingTransformDataset,
    apply_training_offset_augmentation,
)


def _cutout(spatial_shape=(12, 12), foreground=(slice(4, 7), slice(5, 8))):
    image = torch.zeros((1, *spatial_shape))
    mask = torch.zeros((1, *spatial_shape), dtype=torch.long)
    image[(0, *foreground)] = 1.0
    mask[(0, *foreground)] = 1
    return image, mask


class RandomSpatialOffsetTests(unittest.TestCase):
    def test_foreground_stays_inside_the_cutout_in_2d_and_3d(self):
        cases = (
            ((12, 12), (slice(4, 7), slice(5, 8))),
            ((8, 10, 12), (slice(2, 4), slice(3, 6), slice(4, 8))),
        )
        for spatial_shape, foreground in cases:
            with self.subTest(spatial_shape=spatial_shape):
                torch.manual_seed(0)
                image, _ = _cutout(spatial_shape, foreground)
                transform = RandomSpatialOffset(max_fraction=1.0)
                positions = set()
                for _ in range(40):
                    shifted = transform(image)
                    self.assertEqual(shifted.shape, image.shape)
                    # A translation keeps every foreground pixel.
                    self.assertEqual(float(shifted.sum()), float(image.sum()))
                    positions.add(tuple(torch.nonzero(shifted[0])[0].tolist()))
                self.assertGreater(len(positions), 5)

    def test_image_and_mask_move_together_in_dictionary_samples(self):
        torch.manual_seed(1)
        image, mask = _cutout()
        transform = RandomSpatialOffset(max_fraction=1.0)
        moved = False
        for _ in range(20):
            sample = transform.transform_sample({"img": image, "ori_mask": mask, "fname": "a"})
            self.assertEqual(sample["fname"], "a")
            self.assertTrue(torch.equal(sample["img"] > 0, sample["ori_mask"] > 0))
            moved = moved or not torch.equal(sample["img"], image)
        self.assertTrue(moved)
        # The source sample is not modified.
        self.assertEqual(int(image[0, 4:7, 5:8].sum()), 9)

    def test_image_and_mask_move_together_in_tuple_samples(self):
        torch.manual_seed(2)
        image, mask = _cutout()
        transform = RandomSpatialOffset(max_fraction=1.0)
        for _ in range(10):
            shifted_image, shifted_mask, name = transform.transform_sample((image, mask, "a"))
            self.assertEqual(name, "a")
            self.assertTrue(torch.equal(shifted_image > 0, shifted_mask > 0))

    def test_zero_fraction_constant_images_and_empty_samples_are_unchanged(self):
        image, mask = _cutout()
        still = RandomSpatialOffset(max_fraction=0.0)
        self.assertTrue(torch.equal(still(image), image))
        self.assertIs(still.transform_sample((image, mask))[0], image)

        constant = torch.full((1, 6, 6), 0.5)
        self.assertTrue(torch.equal(RandomSpatialOffset()(constant), constant))
        self.assertEqual(RandomSpatialOffset().transform_sample(()), ())
        sample = {"other": 1}
        self.assertEqual(RandomSpatialOffset().transform_sample(sample), sample)

    def test_fraction_limits_the_shift_range(self):
        torch.manual_seed(3)
        image, _ = _cutout()  # rows 4..6 of 12: up to 4 px upward, 5 px downward
        limited = RandomSpatialOffset(max_fraction=0.25)
        for _ in range(30):
            rows = torch.nonzero(limited(image)[0])[:, 0]
            self.assertGreaterEqual(int(rows.min()), 4 - 1)
            self.assertLessEqual(int(rows.max()), 6 + 1)

    def test_configuration_enables_and_disables_the_wrapper(self):
        image, mask = _cutout()
        dataset = [{"img": image, "ori_mask": mask}]
        config = AugmentationConfiguration()

        config.random_offset_enabled = False
        self.assertIs(apply_training_offset_augmentation(dataset, config), dataset)

        config.random_offset_enabled = True
        wrapped = apply_training_offset_augmentation(dataset, config)
        self.assertIsInstance(wrapped, TrainingTransformDataset)
        self.assertEqual(len(wrapped), 1)
        self.assertEqual(float(wrapped[0]["img"].sum()), float(image.sum()))

    def test_wrapper_supports_tuple_list_and_plain_items(self):
        image, mask = _cutout()
        transform = RandomSpatialOffset(max_fraction=1.0)
        self.assertIsInstance(TrainingTransformDataset([(image, mask)], transform)[0], tuple)
        self.assertIsInstance(TrainingTransformDataset([[image, mask]], transform)[0], list)
        plain = TrainingTransformDataset([np.asarray(image)], transform)[0]
        self.assertEqual(tuple(plain.shape), tuple(image.shape))


if __name__ == "__main__":
    unittest.main()
