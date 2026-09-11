"""Tests for image and segmentation rendering helpers."""

import unittest
from unittest.mock import Mock

from matplotlib.figure import Figure
import numpy as np

from hybrid_sample_generator.visualization.rendering import (
    PanelSpec,
    display_mask_plane,
    display_plane,
    normalize_for_display,
    render_panel,
)


class VisualizerRenderingTests(unittest.TestCase):
    def test_rgb_and_channel_rendering_preserve_expected_dimensions(self):
        rgb = np.zeros((3, 5, 7), dtype=np.float32)
        rgb[0] = 255
        rgb[1] = 128

        automatic = display_plane(rgb)
        self.assertEqual(automatic.image.shape, (5, 7, 3))
        self.assertTrue(np.all(automatic.image[..., 0] == 255))
        self.assertTrue(np.all(automatic.image[..., 1] == 128))

        green = display_plane(rgb, channel="1")
        self.assertEqual(green.image.shape, (5, 7))
        self.assertTrue(np.all(green.image == 128))

        normalized = normalize_for_display(automatic.image)
        self.assertEqual(normalized.shape, (5, 7, 3))
        self.assertGreater(float(normalized[..., 0].mean()), 0.99)
        self.assertGreater(float(normalized[..., 1].mean()), 0.45)
        self.assertLess(float(normalized[..., 2].mean()), 0.01)

    def test_volume_and_mask_rendering_share_slice_coordinates(self):
        volume = np.zeros((3, 4, 5, 7), dtype=np.float32)
        volume[0, 2] = 4
        mask = np.zeros((1, 4, 5, 7), dtype=np.uint8)
        mask[:, 2, 1:3, 2:5] = 1

        image_plane = display_plane(volume, slice_index=2)
        mask_plane = display_mask_plane(mask, slice_index=2)

        self.assertEqual(image_plane.image.shape, (5, 7, 3))
        self.assertEqual(image_plane.depth, 4)
        self.assertEqual(image_plane.slice_index, 2)
        self.assertEqual(mask_plane.image.shape, (5, 7))
        self.assertEqual(int(mask_plane.image.sum()), 6)

    def test_cutout_window_excludes_padding_and_preserves_shared_scale(self):
        reference = np.full((100, 100), -1000.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:50, 40:50] = 2
        reference[mask > 0] = np.linspace(1, 2, 100)
        display = normalize_for_display(reference, reference_mask=mask)
        foreground = display[mask > 0]
        self.assertLess(float(foreground.min()), 0.01)
        self.assertGreater(float(foreground.max()), 0.99)
        self.assertTrue(np.all(display[mask == 0] == 0))

        synthetic = np.full((7, 9), 1.5, dtype=np.float32)
        shared = normalize_for_display(
            synthetic, reference=reference, reference_mask=mask
        )
        np.testing.assert_allclose(shared, 0.5, atol=1e-6)

    def test_masked_window_handles_rgb_constant_and_missing_foreground(self):
        rgb = np.zeros((8, 10, 3), dtype=np.float32)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:4, 3:5] = 1
        rgb[mask > 0] = [10, 20, 30]
        display = normalize_for_display(rgb, reference_mask=mask)
        np.testing.assert_allclose(display[mask > 0], [[0, 0.5, 1]] * 4)

        flat = np.zeros((8, 10), dtype=np.float32)
        flat[mask > 0] = 5
        display = normalize_for_display(flat, reference_mask=mask)
        self.assertTrue(np.all(display[mask > 0] == 1))
        for fallback_mask in (np.zeros_like(mask), np.ones((2, 3)), np.full(mask.shape, np.nan)):
            with self.subTest(mask_shape=fallback_mask.shape):
                np.testing.assert_array_equal(
                    normalize_for_display(flat, reference_mask=fallback_mask),
                    normalize_for_display(flat),
                )
        invalid = np.full_like(flat, np.nan)
        self.assertTrue(np.all(normalize_for_display(invalid, reference_mask=mask) == 0))

    def test_volume_render_uses_reference_mask_with_overlay_disabled(self):
        reference = np.full((1, 3, 10, 10), -1000.0, dtype=np.float32)
        mask = np.zeros_like(reference)
        mask[0, 1, 3:5, 3:5] = 1
        reference[0, 1, 3:5, 3:5] = [[1, 2], [3, 4]]
        synthetic = np.full((1, 3, 6, 7), 2.5, dtype=np.float32)
        cache = Mock()
        cache.get.side_effect = lambda path: {"reference-mask": mask}.get(path)
        spec = PanelSpec(
            "Synthetic anomaly", image=synthetic, reference=reference,
            reference_mask_path="reference-mask",
        )
        axis = Figure().subplots()
        depth, status = render_panel(
            axis, spec, cache, slice_index=1, contrast=1, channel="auto",
            show_mask=False, mask_opacity=0.45,
        )
        self.assertEqual(depth, 3)
        self.assertNotIn("error", status)
        self.assertEqual(len(axis.images), 1)
        np.testing.assert_allclose(axis.images[0].get_array(), 0.5, atol=1e-6)
