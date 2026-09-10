import unittest

import numpy as np
import torch

from hybrid_sample_generator.imaging.masks.encoding import (
    to_one_hot_2D,
    to_one_hot_3D,
)
from hybrid_sample_generator.imaging.masks.interpolation import (
    interpolate_masked_regions,
)
from hybrid_sample_generator.imaging.masks.operations import (
    TransformGenerator as CompatibleTransformGenerator,
)
from hybrid_sample_generator.imaging.masks.target_generation import (
    target_mask_from_synthetic_anomaly,
)
from hybrid_sample_generator.imaging.masks.transforms import TransformGenerator


class MaskEncodingTests(unittest.TestCase):
    def test_2d_encoding_removes_background_channel(self):
        mask = torch.tensor([[0, 1], [2, 0]])

        encoded = to_one_hot_2D(mask, num_anomaly_classes=2)

        self.assertEqual(tuple(encoded.shape), (1, 2, 2, 2))
        self.assertEqual(encoded[0, 0, 0, 1].item(), 1.0)
        self.assertEqual(encoded[0, 1, 1, 0].item(), 1.0)
        self.assertEqual(encoded[:, :, 0, 0].sum().item(), 0.0)

    def test_3d_encoding_removes_background_channel(self):
        mask = torch.tensor([[[0, 1], [2, 0]], [[1, 0], [0, 2]]])

        encoded = to_one_hot_3D(mask, num_anomaly_classes=2)

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

    def test_operations_module_remains_a_compatibility_facade(self):
        self.assertIs(CompatibleTransformGenerator, TransformGenerator)


if __name__ == "__main__":
    unittest.main()
