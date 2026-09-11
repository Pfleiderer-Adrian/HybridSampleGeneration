"""Tests for image helpers shared by the bundled examples."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.common.image_io import ensure_chw, load_image_array, save_image


class ExampleImageIOTests(unittest.TestCase):
    def test_ensure_chw_converts_grayscale_and_color_images(self):
        grayscale = np.zeros((8, 9), dtype=np.uint8)
        color = np.zeros((8, 9, 3), dtype=np.uint8)

        self.assertEqual(ensure_chw(grayscale).shape, (1, 8, 9))
        self.assertEqual(ensure_chw(color).shape, (3, 8, 9))

    def test_saved_channel_first_image_can_be_loaded(self):
        image = np.linspace(0.0, 1.0, 3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            save_image(image, path)
            loaded = load_image_array(path)

        self.assertEqual(loaded.shape, (8, 9, 3))
        self.assertEqual(loaded.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
