"""Three-dimensional label transformation and class priority contracts."""
import unittest
import numpy as np
from hybrid_sample_generator.imaging.masks.local_transforms import (
    random_local_rotation_transform,
    random_local_stretch_transform,
    random_local_elastic_transform,
    random_local_dilate_transform,
)


class TransformRegressionTests(unittest.TestCase):
    def test_3d_rotation_stretch_and_elastic_are_repeatable_and_preserve_labels(self):
        mask = np.zeros((1, 12, 14, 16), dtype=np.uint8)
        mask[:, 3:6, 4:8, 5:9] = 1
        mask[:, 7:9, 8:11, 9:12] = 2
        cases = [(random_local_rotation_transform, {'max_rotation': 30}),
                 (random_local_stretch_transform, {'min_stretch': .8, 'max_stretch': 1.2}),
                 (random_local_elastic_transform, {'sigma': 2, 'magnitude': 1})]
        for transform, params in cases:
            with self.subTest(transform=transform.__name__):
                first = transform(mask, params=params, rng=np.random.default_rng(19))
                second = transform(mask, params=params, rng=np.random.default_rng(19))
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, mask.shape)
                self.assertEqual(first.dtype, mask.dtype)
                self.assertEqual(set(np.unique(first)), {0, 1, 2})
                self.assertFalse(np.shares_memory(first, mask))

    def test_overlapping_classes_obey_explicit_priority_in_2d_and_3d(self):
        for dims in (2, 3):
            mask = np.zeros((1, *((7,) * dims)), dtype=np.uint8)
            mask[(0, *((3,) * (dims - 1)), 2)] = 1
            mask[(0, *((3,) * (dims - 1)), 4)] = 2
            params = {'min_iterations': 1, 'max_iterations': 1}
            one_first = random_local_dilate_transform(mask, classes=[1, 2], priorities=[1, 2], params=params, rng=np.random.default_rng(1))
            two_first = random_local_dilate_transform(mask, classes=[1, 2], priorities=[2, 1], params=params, rng=np.random.default_rng(1))
            center = (0, *((3,) * dims))
            self.assertEqual(one_first[center], 1)
            self.assertEqual(two_first[center], 2)
