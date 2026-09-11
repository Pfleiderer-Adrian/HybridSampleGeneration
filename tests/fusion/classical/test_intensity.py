"""Tests for classical intensity matching."""

import unittest

import numpy as np

from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.fusion.classical.intensity import (
    infer_output_intensity_bounds,
    match_local_intensity,
)


class ClassicalIntensityTests(unittest.TestCase):
    def test_output_bounds_are_inferred_only_for_known_ranges(self):
        self.assertEqual(infer_output_intensity_bounds([0.0, 1.0]), (0.0, 1.0))
        self.assertEqual(
            infer_output_intensity_bounds([0.0, 255.0]),
            (0.0, 255.0),
        )
        self.assertIsNone(infer_output_intensity_bounds([-2.0, 300.0]))

    def test_disabled_or_empty_normalization_returns_input(self):
        anomaly = np.ones((1, 4, 4), dtype=np.float32)
        control = np.zeros_like(anomaly)
        mask = np.zeros((4, 4), dtype=np.uint8)
        params = Config(fusion_normalization_border_width=None)

        disabled = match_local_intensity(
            anomaly,
            control,
            control,
            mask > 0,
            mask,
            None,
            None,
            np.zeros_like(mask, dtype=np.float32),
            params,
        )

        self.assertIs(disabled, anomaly)
        params.fusion_normalization_border_width = 2
        empty = match_local_intensity(
            anomaly,
            control,
            control,
            mask > 0,
            mask,
            None,
            None,
            np.zeros_like(mask, dtype=np.float32),
            params,
        )
        self.assertIs(empty, anomaly)

    def test_invalid_normalization_border_is_rejected(self):
        anomaly = np.ones((1, 4, 4), dtype=np.float32)
        control = np.zeros_like(anomaly)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        params = Config(
            fusion_normalization_border_width=-2,
            fusion_relation_min_context_size=1,
        )

        with self.assertRaisesRegex(ValueError, "must be None, -1, or >= 0"):
            match_local_intensity(
                anomaly,
                control,
                control,
                mask > 0,
                mask,
                None,
                None,
                mask.astype(np.float32),
                params,
            )

    def test_global_context_matching_is_finite_and_shape_preserving(self):
        anomaly = np.linspace(0.2, 0.8, 16, dtype=np.float32).reshape(1, 4, 4)
        control = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 4, 4)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        params = Config(
            fusion_normalization_border_width=-1,
            fusion_relation_min_context_size=1,
        )

        matched = match_local_intensity(
            anomaly,
            control,
            control,
            mask > 0,
            mask,
            None,
            None,
            mask.astype(np.float32),
            params,
        )

        self.assertEqual(matched.shape, anomaly.shape)
        self.assertTrue(np.isfinite(matched).all())


if __name__ == "__main__":
    unittest.main()
