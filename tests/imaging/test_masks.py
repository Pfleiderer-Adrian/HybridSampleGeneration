"""Tests for mask encoding, interpolation, and transformations."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.imaging.masks.encoding import to_one_hot
from hybrid_sample_generator.imaging.masks.interpolation import (
    interpolate_masked_regions,
)
from hybrid_sample_generator.imaging.masks.local_transforms import (
    random_local_dilate_transform,
)
from hybrid_sample_generator.imaging.masks.target_generation import (
    target_mask_from_synthetic_anomaly,
)
from hybrid_sample_generator.imaging.masks.transform_generator import (
    TransformGenerator,
)


class MaskEncodingTests(unittest.TestCase):
    def test_2d_encoding_removes_background_channel(self):
        mask = torch.tensor([[0, 1], [2, 0]])

        encoded = to_one_hot(mask, num_anomaly_classes=2, spatial_dims=2)

        self.assertEqual(tuple(encoded.shape), (1, 2, 2, 2))
        self.assertEqual(encoded[0, 0, 0, 1].item(), 1.0)
        self.assertEqual(encoded[0, 1, 1, 0].item(), 1.0)
        self.assertEqual(encoded[:, :, 0, 0].sum().item(), 0.0)

    def test_3d_encoding_removes_background_channel(self):
        mask = torch.tensor([[[0, 1], [2, 0]], [[1, 0], [0, 2]]])

        encoded = to_one_hot(mask, num_anomaly_classes=2, spatial_dims=3)

        self.assertEqual(tuple(encoded.shape), (1, 2, 2, 2, 2))
        self.assertEqual(encoded[0, 0, 0, 0, 1].item(), 1.0)
        self.assertEqual(encoded[0, 1, 1, 1, 1].item(), 1.0)


class MaskInterpolationTests(unittest.TestCase):
    def test_identity_warp_preserves_masked_regions(self):
        image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        foreground = np.zeros((3, 3), dtype=bool)
        foreground[1, 1] = True

        result = interpolate_masked_regions(
            image,
            foreground,
            lambda value: value.copy(),
            lambda value: value.copy(),
        )

        np.testing.assert_array_equal(result, image)


class TargetMaskTests(unittest.TestCase):
    def test_threshold_target_mask_uses_relative_intensity_range(self):
        anomaly = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)

        result = target_mask_from_synthetic_anomaly(
            anomaly,
            background_threshold=0.5,
        )

        np.testing.assert_array_equal(result, np.array([[0, 0], [1, 1]], dtype=np.uint8))


class MaskTransformTests(unittest.TestCase):
    def test_generator_enables_default_transforms_by_default(self):
        generator = TransformGenerator()

        self.assertEqual(generator.global_transform_probs["zoom"], 1.0)
        self.assertEqual(generator.global_transform_probs["stretch"], 1.0)
        self.assertEqual(generator.global_transform_probs["elastic"], 1.0)

    def test_disabled_generator_ignores_explicit_transform_probabilities(self):
        mask = np.zeros((1, 12, 12), dtype=np.uint8)
        mask[0, 4:8, 4:8] = 1
        generator = TransformGenerator(
            {"stretch": 1.0},
            use_mask_transform=False,
            transform_params={
                "stretch": {"min_stretch": 2.0, "max_stretch": 2.0}
            },
            rng=np.random.default_rng(7),
        )

        np.testing.assert_array_equal(generator.augment_mask(mask), mask)

    def test_local_transform_preserves_shape_and_dtype(self):
        mask = np.zeros((1, 8, 8), dtype=np.uint8)
        mask[0, 3:5, 3:5] = 1

        result = random_local_dilate_transform(
            mask,
            params={"min_iterations": 1, "max_iterations": 1},
            rng=np.random.default_rng(7),
        )

        self.assertEqual(result.shape, mask.shape)
        self.assertEqual(result.dtype, mask.dtype)
        self.assertGreater(np.count_nonzero(result), np.count_nonzero(mask))

    def test_generator_is_deterministic_for_seeded_rng(self):
        mask = np.zeros((1, 12, 12), dtype=np.uint8)
        mask[0, 4:8, 4:8] = 1
        settings = {"stretch": 1.0, "local_dilate": 1.0}

        first = TransformGenerator(
            settings, use_mask_transform=True, rng=np.random.default_rng(23)
        ).augment_mask(mask)
        second = TransformGenerator(
            settings, use_mask_transform=True, rng=np.random.default_rng(23)
        ).augment_mask(mask)

        np.testing.assert_array_equal(first, second)



class ZoomTransformTests(unittest.TestCase):
    def test_zoom_can_shrink_and_enlarge_a_mask(self):
        from hybrid_sample_generator.imaging.masks.global_transforms import (
            random_global_zoom_transform,
        )

        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        mask[0, 12:20, 12:20] = 1
        rng = np.random.default_rng(0)

        smaller = random_global_zoom_transform(mask, min_zoom=0.5, max_zoom=0.5, rng=rng)
        larger = random_global_zoom_transform(mask, min_zoom=1.5, max_zoom=1.5, rng=rng)

        self.assertEqual(int(smaller.sum()), 16)    # 4 x 4
        self.assertEqual(int(larger.sum()), 144)    # 12 x 12
        for invalid in ((0.0, 1.0), (1.2, 1.0), (-1.0, 1.0), (1.0, float("inf"))):
            with self.assertRaisesRegex(ValueError, "min_zoom"):
                random_global_zoom_transform(mask, min_zoom=invalid[0], max_zoom=invalid[1], rng=rng)

    def test_generated_target_mask_grows_with_zoom_above_one(self):
        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        mask[0, 12:20, 12:20] = 1
        generator = TransformGenerator(
            {"zoom": 1.0},
            use_mask_transform=True,
            transform_params={"zoom": {"min_zoom": 1.5, "max_zoom": 1.5}},
            rng=np.random.default_rng(1),
        )

        target = generator.augment_mask(mask)

        self.assertEqual(target.shape, mask.shape)
        self.assertEqual(int(target.sum()), 144)

    def test_paired_zoom_accepts_factors_above_one(self):
        from hybrid_sample_generator.imaging.masks.paired_transforms import sample_warp

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 1
        warp = sample_warp(
            "zoom", mask, {"min_zoom": 1.5, "max_zoom": 1.5}, np.random.default_rng(2)
        )
        self.assertIsNotNone(warp)
        with self.assertRaisesRegex(ValueError, "min_zoom"):
            sample_warp("zoom", mask, {"min_zoom": 0.0, "max_zoom": 1.0}, np.random.default_rng(2))


class ElasticDefaultTests(unittest.TestCase):
    def test_default_magnitude_follows_the_shortest_axis(self):
        from hybrid_sample_generator.imaging.masks.elastic import (
            default_elastic_params_from_anomaly_size,
        )

        # Square cutouts: the same displacement on every axis, 20 % of the size.
        self.assertEqual(
            default_elastic_params_from_anomaly_size((3, 64, 64)),
            {"sigma": (13, 13), "magnitude": (13, 13)},
        )
        # Elongated cutouts: smoothing follows each axis, displacement the shortest one.
        self.assertEqual(
            default_elastic_params_from_anomaly_size((1, 64, 256)),
            {"sigma": (13, 51), "magnitude": (13, 13)},
        )
        self.assertEqual(
            default_elastic_params_from_anomaly_size((1, 16, 64, 128)),
            {"sigma": (3, 13, 26), "magnitude": (3, 3, 3)},
        )
        self.assertEqual(default_elastic_params_from_anomaly_size(None), {})


if __name__ == "__main__":
    unittest.main()
