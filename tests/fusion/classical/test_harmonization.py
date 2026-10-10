"""Tests for optional pretrained image harmonization adapters."""

import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hybrid_sample_generator.fusion.classical.harmonization import ImageHarmonizer


class ImageHarmonizerTests(unittest.TestCase):
    def test_rgb_conversion_masking_and_model_selection(self):
        model = Mock(return_value=np.full((2, 2, 3), 255, dtype=np.uint8))
        model_factory = Mock(return_value=model)
        fake_libcom = types.SimpleNamespace(ImageHarmonizationModel=model_factory)
        image = np.array(
            [
                [[0.0, 0.2], [0.4, 0.6]],
                [[0.1, 0.3], [0.5, 0.7]],
                [[0.2, 0.4], [0.6, 1.0]],
            ],
            dtype=np.float32,
        )
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[:, 0, 1] = 1

        with patch(
            "hybrid_sample_generator.fusion.classical.harmonization.import_module",
            return_value=fake_libcom,
        ):
            harmonizer = ImageHarmonizer("PCTNet", device="cpu")
            result = harmonizer(image, mask)

        model_factory.assert_called_once_with(device="cpu", model_type="PCTNet")
        composite_bgr, composite_mask = model.call_args.args
        self.assertEqual(composite_bgr.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            composite_mask,
            np.array([[0, 255], [0, 0]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(result[:, 1, 1], image[:, 1, 1])
        np.testing.assert_allclose(result[:, 0, 1], 1.0)
        np.testing.assert_array_equal(
            result[:, mask[0] == 0], image[:, mask[0] == 0]
        )
        self.assertEqual(result.dtype, image.dtype)

    def test_grayscale_is_replicated_and_converted_back(self):
        model = Mock(return_value=np.full((2, 2, 3), 128, dtype=np.uint8))
        fake_libcom = types.SimpleNamespace(
            ImageHarmonizationModel=Mock(return_value=model)
        )
        image = np.array([[[0.0, 0.25], [0.5, 1.0]]], dtype=np.float32)
        mask = np.ones_like(image, dtype=np.uint8)

        with patch(
            "hybrid_sample_generator.fusion.classical.harmonization.import_module",
            return_value=fake_libcom,
        ):
            result = ImageHarmonizer("LBM", device="cpu")(image, mask)

        self.assertEqual(model.call_args.args[0].shape, (2, 2, 3))
        np.testing.assert_allclose(result, 128.0 / 255.0, atol=1e-6)

    def test_rejects_unsupported_dimensions_channels_and_nonfinite_values(self):
        model = Mock(return_value=np.zeros((2, 2, 3), dtype=np.uint8))
        fake_libcom = types.SimpleNamespace(
            ImageHarmonizationModel=Mock(return_value=model)
        )
        with patch(
            "hybrid_sample_generator.fusion.classical.harmonization.import_module",
            return_value=fake_libcom,
        ):
            harmonizer = ImageHarmonizer("PCTNet", device="cpu")

        for image, message in (
            (np.zeros((1, 2, 2, 2), dtype=np.float32), "only 2D"),
            (np.zeros((2, 2, 2), dtype=np.float32), "one or three"),
            (np.full((1, 2, 2), np.nan, dtype=np.float32), "finite"),
        ):
            with self.subTest(shape=image.shape), self.assertRaisesRegex(
                ValueError, message
            ):
                harmonizer(image, np.ones_like(image, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
