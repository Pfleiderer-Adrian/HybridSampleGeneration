"""Tests for class-specific mask transforms, class priorities and their configuration."""

import unittest

import numpy as np

from hybrid_sample_generator.configuration.augmentation import (
    AugmentationConfiguration,
    MaskTransformConfiguration,
)
from hybrid_sample_generator.imaging.masks.composition import combine_label_masks
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


def _two_class_mask(gap=8):
    """Class 1 on the left and class 2 on the right of a 32 x 48 mask."""
    mask = np.zeros((1, 32, 48), dtype=np.uint8)
    mask[0, 12:20, 8:16] = 1
    mask[0, 12:20, 16 + gap:24 + gap] = 2
    return mask


def _generator(probabilities, parameters=None, **settings):
    return TransformGenerator(
        probabilities,
        use_mask_transform=True,
        transform_params=parameters,
        rng=np.random.default_rng(0),
        **settings,
    )


DILATE_TWICE = {"min_iterations": 2, "max_iterations": 2}
NO_GLOBAL = {"zoom": 0.0, "stretch": 0.0, "rotate": 0.0, "elastic": 0.0}


class ClassSpecificTransformTests(unittest.TestCase):
    def _areas(self, mask):
        return int((mask == 1).sum()), int((mask == 2).sum())

    def test_a_local_transform_applies_to_every_class(self):
        mask = _two_class_mask()
        result = _generator(
            {**NO_GLOBAL, "local_dilate": 1.0}, {"local_dilate": DILATE_TWICE}
        ).augment_mask(mask)

        first, second = self._areas(result)
        self.assertEqual(result.shape, mask.shape)
        self.assertGreater(first, 64)
        self.assertEqual(first, second)

    def test_class_probabilities_select_the_transformed_class(self):
        mask = _two_class_mask()
        result = _generator(
            {**NO_GLOBAL, 2: {"local_dilate": 1.0}}, {"local_dilate": DILATE_TWICE}
        ).augment_mask(mask)

        first, second = self._areas(result)
        self.assertEqual(first, 64)
        self.assertGreater(second, 64)
        np.testing.assert_array_equal(result == 1, mask == 1)

    def test_class_parameters_override_the_shared_parameters(self):
        mask = _two_class_mask()
        result = _generator(
            {**NO_GLOBAL, "local_dilate": 1.0},
            {
                "local_dilate": {"min_iterations": 1, "max_iterations": 1},
                2: {"local_dilate": {"min_iterations": 3, "max_iterations": 3}},
            },
        ).augment_mask(mask)

        first, second = self._areas(result)
        self.assertGreater(first, 64)
        self.assertGreater(second, first)

    def test_a_class_probability_of_zero_disables_a_shared_transform(self):
        mask = _two_class_mask()
        result = _generator(
            {**NO_GLOBAL, "local_dilate": 1.0, 1: {"local_dilate": 0.0}},
            {"local_dilate": DILATE_TWICE},
        ).augment_mask(mask)

        first, second = self._areas(result)
        self.assertEqual(first, 64)
        self.assertGreater(second, 64)

    def test_local_stretch_can_act_on_all_classes_together(self):
        mask = _two_class_mask()
        result = _generator(
            {**NO_GLOBAL, "local_stretch": 1.0},
            {"local_stretch": {"min_stretch": 1.5, "max_stretch": 1.5}},
            mask_transform_local_as_global=True,
        ).augment_mask(mask)

        def gap(array):
            return int(np.where(array[0] == 2)[1].min() - np.where(array[0] == 1)[1].max())

        # One stretch of the whole mask enlarges both classes and moves them apart.
        self.assertEqual(int((result == 1).sum()), 144)
        self.assertEqual(int((result == 2).sum()), 144)
        self.assertGreater(gap(result), gap(mask))

    def test_merged_transforms_use_the_shared_probability(self):
        mask = _two_class_mask()
        never = _generator(
            {**NO_GLOBAL, "local_stretch": 0.0},
            {"local_stretch": {"min_stretch": 1.5, "max_stretch": 1.5}},
            mask_transform_local_as_global=True,
        ).augment_mask(mask)
        np.testing.assert_array_equal(never, mask)

    def test_merged_dilation_grows_both_classes(self):
        mask = _two_class_mask()
        result = _generator(
            {**NO_GLOBAL, "local_dilate": 1.0}, {"local_dilate": DILATE_TWICE},
            mask_transform_local_as_global=True,
        ).augment_mask(mask)
        first, second = self._areas(result)
        self.assertGreater(first, 64)
        self.assertGreater(second, 64)

    def test_priorities_decide_which_class_wins_an_overlap(self):
        mask = _two_class_mask(gap=0)   # the classes touch
        settings = ({**NO_GLOBAL, "local_dilate": 1.0}, {"local_dilate": DILATE_TWICE})
        first_wins = _generator(*settings, priorities=[1, 2]).augment_mask(mask)
        second_wins = _generator(*settings, priorities=[2, 1]).augment_mask(mask)

        # Both classes grow into each other; the prioritized class keeps the overlap.
        self.assertGreater(int((first_wins == 1).sum()), int((second_wins == 1).sum()))
        self.assertGreater(int((second_wins == 2).sum()), int((first_wins == 2).sum()))
        np.testing.assert_array_equal(first_wins > 0, second_wins > 0)

    def test_image_content_follows_its_class(self):
        mask = _two_class_mask()
        image = np.where(mask == 1, 1.0, np.where(mask == 2, 2.0, 0.0)).astype(np.float32)
        for merged in (False, True):
            with self.subTest(local_as_global=merged):
                target_mask, target_image = _generator(
                    {**NO_GLOBAL, "local_dilate": 1.0}, {"local_dilate": DILATE_TWICE},
                    mask_transform_local_as_global=merged,
                ).create_target_mask_and_transformed_image(mask, image)

                self.assertEqual(target_mask.shape, mask.shape)
                self.assertEqual(target_image.shape, image.shape)
                self.assertGreater(int((target_mask == 1).sum()), 64)
                # Pixels of a class carry the intensity of that class, also where it grew.
                np.testing.assert_allclose(target_image[target_mask == 1], 1.0)
                np.testing.assert_allclose(target_image[target_mask == 2], 2.0)

    def test_invalid_probability_and_parameter_keys_are_rejected(self):
        cases = (
            ({"sharpen": 1.0}, None, ValueError),
            ({1: 0.5}, None, TypeError),
            ({1: {"zoom": 0.5}}, None, KeyError),
            ({"zoom": 1.5}, None, ValueError),
            (None, {"zoom": 0.5}, TypeError),
            (None, {1: 0.5}, TypeError),
            (None, {1: {"zoom": {}}}, KeyError),
            (None, {1: {"local_dilate": 2}}, TypeError),
            (None, {("a",): {}}, ValueError),
        )
        for probabilities, parameters, error in cases:
            with self.subTest(probabilities=probabilities, parameters=parameters):
                with self.assertRaises(error):
                    TransformGenerator(
                        probabilities, use_mask_transform=True, transform_params=parameters
                    )


class MaskTransformConfigurationTests(unittest.TestCase):
    def test_setters_fill_probabilities_and_parameters(self):
        config = MaskTransformConfiguration()
        self.assertIs(config.setGlobalParam("rotate", probability=0.5, max_rotation=10.0), config)
        config.setGlobalParam("rotate", max_rotation=12.0)
        config.setAllClassParams("local_dilate", probability=0.3, max_iterations=3)
        config.setClassParam(2, "local_stretch", probability=1.0, max_stretch=1.1)
        config.setClassParam(2, "local_stretch", min_stretch=0.9)

        self.assertEqual(
            config.mask_transform_probs,
            {"rotate": 0.5, "local_dilate": 0.3, 2: {"local_stretch": 1.0}},
        )
        self.assertEqual(
            config.mask_transform_params,
            {
                "rotate": {"max_rotation": 12.0},
                "local_dilate": {"max_iterations": 3},
                2: {"local_stretch": {"max_stretch": 1.1, "min_stretch": 0.9}},
            },
        )

    def test_setters_reject_transforms_of_the_wrong_kind(self):
        config = MaskTransformConfiguration()
        with self.assertRaisesRegex(ValueError, "not a global transform"):
            config.setGlobalParam("local_dilate", probability=1.0)
        with self.assertRaisesRegex(ValueError, "not a local transform"):
            config.setAllClassParams("zoom", probability=1.0)
        with self.assertRaisesRegex(ValueError, "not a local transform"):
            config.setClassParam(1, "rotate", probability=1.0)

    def test_configured_settings_reach_the_generator(self):
        config = MaskTransformConfiguration()
        for name in ("zoom", "stretch", "elastic"):
            config.setGlobalParam(name, probability=0.0)
        config.setClassParam(2, "local_dilate", probability=1.0, **DILATE_TWICE)
        config.priorities = [2, 1]
        generator = TransformGenerator.from_config(
            config, anomaly_size=(1, 32, 48), background_threshold=0.01, seed=0
        )

        result = generator.augment_mask(_two_class_mask())

        self.assertEqual(int((result == 1).sum()), 64)
        self.assertGreater(int((result == 2).sum()), 64)
        self.assertEqual(generator.priorities, [2, 1])

    def test_disabled_transforms_return_the_mask_unchanged(self):
        config = MaskTransformConfiguration(use_mask_transform=False)
        config.setAllClassParams("local_dilate", probability=1.0)
        generator = TransformGenerator.from_config(
            config, anomaly_size=(1, 32, 48), background_threshold=0.01, seed=0
        )
        mask = _two_class_mask()
        np.testing.assert_array_equal(generator.augment_mask(mask), mask)

    def test_augmentation_configuration_round_trip_and_validation(self):
        config = AugmentationConfiguration()
        config.mask_transforms.setClassParam(2, "local_dilate", probability=0.5)
        config.random_offset_max_fraction = 0.4

        restored = AugmentationConfiguration.from_dict(config.to_dict())

        self.assertEqual(restored.random_offset_max_fraction, 0.4)
        self.assertEqual(restored.mask_transforms.mask_transform_probs, {2: {"local_dilate": 0.5}})
        for name, value in (
            ("random_offset_max_fraction", 1.5),
            ("random_offset_max_fraction", -0.1),
            ("random_offset_foreground_threshold", -1.0),
        ):
            broken = AugmentationConfiguration()
            setattr(broken, name, value)
            with self.assertRaises(ValueError):
                broken.validate()


class LabelMaskCompositionTests(unittest.TestCase):
    def test_overwrite_decides_which_mask_keeps_shared_pixels(self):
        first = np.zeros((1, 4, 4), dtype=np.uint8)
        second = np.zeros((1, 4, 4), dtype=np.uint8)
        first[0, :2] = 1
        second[0, 1:3] = 2

        replaced = combine_label_masks(first, second, overwrite=True)
        kept = combine_label_masks(first, second, overwrite=False, return_dtype=np.int16)

        np.testing.assert_array_equal(replaced[0, :, 0], [1, 2, 2, 0])
        np.testing.assert_array_equal(kept[0, :, 0], [1, 1, 2, 0])
        self.assertEqual(kept.dtype, np.int16)
        np.testing.assert_array_equal(first[0, :, 0], [1, 1, 0, 0])

    def test_invalid_masks_are_rejected(self):
        mask = np.zeros((1, 4, 4), dtype=np.uint8)
        with self.assertRaises(TypeError):
            combine_label_masks(mask, [[0]])
        with self.assertRaises(ValueError):
            combine_label_masks(mask, np.zeros((1, 4, 5), dtype=np.uint8))
        with self.assertRaises(ValueError):
            combine_label_masks(mask[0], mask[0])


if __name__ == "__main__":
    unittest.main()
