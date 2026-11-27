"""Local stretch must scale every class around its own center."""

import unittest

import numpy as np

from hybrid_sample_generator.imaging.masks.geometry import foreground_center, stretch_spatial_mask
from hybrid_sample_generator.imaging.masks.local_transforms import random_local_stretch_transform
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator

STRETCH = {"local_stretch": {"min_stretch": 1.5, "max_stretch": 1.5}}
ONLY_LOCAL_STRETCH = {
    "zoom": 0.0, "stretch": 0.0, "rotate": 0.0, "elastic": 0.0, "local_stretch": 1.0,
}


def _two_class_mask():
    """Class 1 on the left and class 2 on the right, both away from the mask center."""
    mask = np.zeros((1, 32, 64), dtype=np.uint8)
    mask[0, 12:20, 8:16] = 1
    mask[0, 10:18, 40:48] = 2
    return mask


def _box(mask, class_id):
    rows, columns = np.where(mask[0] == class_id)
    return rows.min(), rows.max(), columns.min(), columns.max()


def _generator(**settings):
    return TransformGenerator(
        ONLY_LOCAL_STRETCH,
        use_mask_transform=True,
        transform_params=STRETCH,
        rng=np.random.default_rng(0),
        **settings,
    )


class LocalStretchTests(unittest.TestCase):
    def _assert_classes_grew_in_place(self, result, mask):
        for class_id in (1, 2):
            before, after = _box(mask, class_id), _box(result, class_id)
            # 8 px become 12 px, two more on each side of the unchanged center.
            self.assertEqual(int((result == class_id).sum()), 144)
            self.assertEqual(after, (before[0] - 2, before[1] + 2, before[2] - 2, before[3] + 2))

    def test_function_scales_each_class_around_its_own_center(self):
        mask = _two_class_mask()
        result = random_local_stretch_transform(
            mask, params=STRETCH["local_stretch"], rng=np.random.default_rng(0)
        )
        self._assert_classes_grew_in_place(result, mask)

    def test_generated_target_mask_keeps_the_classes_in_place(self):
        mask = _two_class_mask()
        self._assert_classes_grew_in_place(_generator().augment_mask(mask), mask)

    def test_image_and_mask_grow_together_in_place(self):
        mask = _two_class_mask()
        image = np.where(mask == 1, 1.0, np.where(mask == 2, 2.0, 0.0)).astype(np.float32)

        target_mask, target_image = _generator().create_target_mask_and_transformed_image(mask, image)

        self._assert_classes_grew_in_place(target_mask, mask)
        np.testing.assert_allclose(target_image[target_mask == 1], 1.0)
        np.testing.assert_allclose(target_image[target_mask == 2], 2.0)

    def test_only_the_selected_class_changes(self):
        mask = _two_class_mask()
        result = TransformGenerator(
            {**ONLY_LOCAL_STRETCH, "local_stretch": 0.0, 2: {"local_stretch": 1.0}},
            use_mask_transform=True,
            transform_params=STRETCH,
            rng=np.random.default_rng(0),
        ).augment_mask(mask)

        np.testing.assert_array_equal(result == 1, mask == 1)
        self.assertEqual(int((result == 2).sum()), 144)
        self.assertEqual(_box(result, 2), (8, 19, 38, 49))

    def test_stretching_all_classes_together_still_uses_the_mask_center(self):
        mask = _two_class_mask()
        result = _generator(mask_transform_local_as_global=True).augment_mask(mask)

        # One transform of the whole mask moves the classes away from its center.
        self.assertLess(_box(result, 1)[2], _box(mask, 1)[2] - 2)
        self.assertGreater(_box(result, 2)[3], _box(mask, 2)[3] + 2)

    def test_volumes_are_scaled_around_the_class_center(self):
        mask = np.zeros((1, 24, 24, 24), dtype=np.uint8)
        mask[0, 2:6, 3:7, 14:18] = 1
        result = random_local_stretch_transform(
            mask, params={"min_stretch": 2.0, "max_stretch": 2.0}, rng=np.random.default_rng(0)
        )
        self.assertEqual(int(result.sum()), 512)
        np.testing.assert_allclose(foreground_center(result[0]), foreground_center(mask[0]))

    def test_stretch_helper_supports_both_centers(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        around_array = stretch_spatial_mask(mask, (2.0, 2.0))
        around_class = stretch_spatial_mask(mask, (2.0, 2.0), center_mask=mask)
        empty_center = stretch_spatial_mask(mask, (2.0, 2.0), center_mask=np.zeros_like(mask))

        np.testing.assert_allclose(foreground_center(around_class), foreground_center(mask))
        self.assertFalse(np.array_equal(around_array, around_class))
        np.testing.assert_array_equal(empty_center, around_array)
        self.assertIsNone(foreground_center(np.zeros((4, 4))))


if __name__ == "__main__":
    unittest.main()
