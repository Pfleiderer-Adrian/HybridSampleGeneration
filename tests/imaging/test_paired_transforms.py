"""Joint image and label transformations for transformed posterior skips."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


class PairedTransformTests(unittest.TestCase):
    def _sample(self, dims):
        shape = (12,) * dims
        mask = np.zeros((1, *shape), dtype=np.uint8)
        first = (slice(2, 5),) * dims
        second = (slice(7, 10),) * dims
        mask[(0, *first)] = 1
        mask[(0, *second)] = 2
        image = np.full((2, *shape), 0.1, dtype=np.float32)
        image[(slice(None), *first)] = np.array([0.8, 0.6]).reshape(
            (2, *((1,) * dims))
        )
        image[(slice(None), *second)] = np.array([0.4, 0.3]).reshape(
            (2, *((1,) * dims))
        )
        return mask, image

    def test_identity_preserves_multichannel_image_and_mask(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                mask, image = self._sample(dims)
                result_mask, result_image = TransformGenerator(
                    padding_factor=2,
                    rng=np.random.default_rng(1),
                ).create_target_mask_and_transformed_image(mask, image)
                np.testing.assert_array_equal(result_mask, mask)
                np.testing.assert_array_equal(result_image, image)
                self.assertEqual(result_mask.dtype, mask.dtype)
                self.assertEqual(result_image.dtype, image.dtype)

    def test_global_zoom_and_elastic_are_aligned_and_repeatable(self):
        for dims in (2, 3):
            for name, params in (
                ("zoom", {"min_zoom": 0.8, "max_zoom": 0.8}),
                ("elastic", {"sigma": 2, "magnitude": 1}),
            ):
                with self.subTest(dims=dims, transform=name):
                    mask, image = self._sample(dims)
                    def run():
                        return TransformGenerator(
                            {name: 1.0},
                            transform_params={name: params},
                            rng=np.random.default_rng(13),
                        ).create_target_mask_and_transformed_image(mask, image)
                    result_mask, result_image = run()
                    repeat_mask, repeat_image = run()
                    np.testing.assert_array_equal(result_mask, repeat_mask)
                    np.testing.assert_array_equal(result_image, repeat_image)
                    self.assertEqual(result_mask.shape, mask.shape)
                    self.assertEqual(result_image.shape, image.shape)
                    self.assertTrue(set(np.unique(result_mask)) <= {0, 1, 2})
                    np.testing.assert_allclose(
                        result_image[0][result_mask[0] == 1], 0.8, atol=1e-5
                    )
                    np.testing.assert_allclose(
                        result_image[0][result_mask[0] == 2], 0.4, atol=1e-5
                    )

    def test_local_dilation_moves_class_pixels_with_labels(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                mask, image = self._sample(dims)
                result_mask, result_image = TransformGenerator(
                    {"local_dilate": 1.0},
                    transform_params={
                        "local_dilate": {
                            "min_iterations": 1,
                            "max_iterations": 1,
                        }
                    },
                    rng=np.random.default_rng(8),
                ).create_target_mask_and_transformed_image(mask, image)
                self.assertGreater(np.count_nonzero(result_mask), np.count_nonzero(mask))
                np.testing.assert_allclose(
                    result_image[0][result_mask[0] == 1], 0.8, atol=1e-5
                )
                np.testing.assert_allclose(
                    result_image[0][result_mask[0] == 2], 0.4, atol=1e-5
                )

    def test_overlapping_classes_keep_mask_and_image_priority(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                shape = (7,) * dims
                mask = np.zeros((1, *shape), dtype=np.uint8)
                first = (3,) * (dims - 1) + (2,)
                second = (3,) * (dims - 1) + (4,)
                center = (3,) * dims
                mask[(0, *first)] = 1
                mask[(0, *second)] = 2
                image = np.zeros((1, *shape), dtype=np.float32)
                image[(0, *first)] = 0.4
                image[(0, *second)] = 0.8
                result_mask, result_image = TransformGenerator(
                    {"local_dilate": 1.0},
                    transform_params={
                        "local_dilate": {
                            "min_iterations": 1,
                            "max_iterations": 1,
                        }
                    },
                    priorities=[2, 1],
                    rng=np.random.default_rng(1),
                ).create_target_mask_and_transformed_image(mask, image)
                self.assertEqual(result_mask[(0, *center)], 2)
                self.assertAlmostEqual(float(result_image[(0, *center)]), 0.8)

    def test_default_global_and_local_geometry_in_both_dimensions(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                mask, image = self._sample(dims)
                settings = {
                    "zoom": 1.0,
                    "stretch": 1.0,
                    "rotate": 1.0,
                    "elastic": 1.0,
                    "local_stretch": 1.0,
                    "local_rotate": 1.0,
                    "local_elastic": 1.0,
                }
                parameters = {
                    "elastic": {"sigma": 2, "magnitude": 1},
                    "local_elastic": {"sigma": 2, "magnitude": 1},
                }
                def run(local_as_global):
                    return TransformGenerator(
                        settings,
                        transform_params=parameters,
                        mask_transform_local_as_global=local_as_global,
                        rng=np.random.default_rng(12),
                    ).create_target_mask_and_transformed_image(mask, image)
                for local_as_global in (False, True):
                    with self.subTest(local_as_global=local_as_global):
                        first_mask, first_image = run(local_as_global)
                        second_mask, second_image = run(local_as_global)
                        np.testing.assert_array_equal(first_mask, second_mask)
                        np.testing.assert_array_equal(first_image, second_image)
                        self.assertEqual(first_mask.shape, mask.shape)
                        self.assertEqual(first_image.shape, image.shape)
                        self.assertTrue(set(np.unique(first_mask)) <= {0, 1, 2})
                        self.assertTrue(np.isfinite(first_image).all())

    def test_fit_to_original_shape_keeps_pair_aligned(self):
        mask, image = self._sample(2)
        transformed_mask, transformed_image = TransformGenerator(
            {"stretch": 1.0},
            transform_params={
                "stretch": {"min_stretch": 2.0, "max_stretch": 2.0}
            },
            rng=np.random.default_rng(3),
        ).create_target_mask_and_transformed_image(mask, image)
        self.assertEqual(transformed_mask.shape, mask.shape)
        self.assertEqual(transformed_image.shape, image.shape)
        np.testing.assert_allclose(
            transformed_image[0][transformed_mask[0] == 1], 0.8, atol=1e-5
        )

    def test_torch_input_preserves_devices_and_dtypes(self):
        mask, image = self._sample(2)
        mask_tensor = torch.as_tensor(mask)
        image_tensor = torch.as_tensor(image)
        result_mask, result_image = TransformGenerator(
            rng=np.random.default_rng(2)
        ).create_target_mask_and_transformed_image(mask_tensor, image_tensor)
        self.assertEqual(result_mask.dtype, mask_tensor.dtype)
        self.assertEqual(result_image.dtype, image_tensor.dtype)
        self.assertEqual(result_mask.device, mask_tensor.device)
        self.assertEqual(result_image.device, image_tensor.device)

    def test_mismatched_image_shape(self):
        mask, image = self._sample(2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            TransformGenerator().create_target_mask_and_transformed_image(
                mask, image[:, :-1]
            )


if __name__ == "__main__":
    unittest.main()
