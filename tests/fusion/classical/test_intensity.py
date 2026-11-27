"""Tests for classical intensity matching."""

import unittest

import numpy as np

from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.fusion.classical.intensity import (
    _fit_values_into_bounds,
    anomaly_context_relations,
    infer_output_intensity_bounds,
    match_local_intensity,
)


def _roi(background=0.5, anomaly=0.2, size=16, box=(slice(6, 10), slice(5, 11))):
    """A flat ROI with a flat anomaly; a small ramp gives both a non-zero spread."""
    ramp = np.linspace(-0.01, 0.01, size * size, dtype=np.float32).reshape(size, size)
    image = np.full((1, size, size), background, dtype=np.float32) + ramp
    mask = np.zeros((1, size, size), dtype=np.uint8)
    image[(0, *box)] = anomaly + ramp[box]
    mask[(0, *box)] = 1
    return image, mask


def _match(anomaly, control, mask, roi, roi_mask, alpha=1.0, **settings):
    """Match an anomaly that covers the whole control region of the same size."""
    params = Config(fusion_variation=False, **settings)
    return match_local_intensity(
        anomaly,
        control,
        control,
        mask > 0,
        mask,
        roi,
        roi_mask,
        np.where(mask > 0, alpha, 0.0).astype(np.float32),
        params,
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


class AnomalyContextRelationTests(unittest.TestCase):
    def test_delta_and_ratio_describe_the_anomaly_relative_to_its_ring(self):
        roi, mask = _roi(background=0.5, anomaly=0.2)

        delta = anomaly_context_relations(roi, mask, 2, "delta")
        ratio = anomaly_context_relations(roi, mask, 2, "ratio")

        self.assertAlmostEqual(delta["median_delta"], -0.3, places=2)
        self.assertNotIn("median_ratio", delta)
        self.assertAlmostEqual(ratio["median_ratio"], 0.4, places=2)
        self.assertGreater(delta["iqr_ratio"], 0.0)
        with self.assertRaisesRegex(ValueError, "delta.*ratio"):
            anomaly_context_relations(roi, mask, 2, "difference")

    def test_ratio_is_undefined_for_a_zero_context(self):
        roi, mask = _roi(background=0.0, anomaly=0.3)
        roi[0][mask[0] == 0] = 0.0
        self.assertIsNone(anomaly_context_relations(roi, mask, 2, "ratio")["median_ratio"])

    def test_ring_is_used_before_the_remaining_roi(self):
        # Far from the anomaly the ROI is much brighter; only the ring is 0.5.
        roi, mask = _roi(background=0.5, anomaly=0.2)
        roi[0, :2, :] = 5.0
        near = anomaly_context_relations(roi, mask, 2, "delta")
        self.assertAlmostEqual(near["median_delta"], -0.3, places=2)

        # A ring that is too small falls back to everything outside the anomaly.
        whole = anomaly_context_relations(roi, mask, 2, "delta", min_context_size=200)
        outside = np.median(roi[0][mask[0] == 0])
        self.assertAlmostEqual(whole["median_delta"], float(np.median(roi[0][mask[0] > 0]) - outside), places=5)

    def test_missing_or_unusable_inputs_give_no_relation(self):
        roi, mask = _roi()
        self.assertIsNone(anomaly_context_relations(None, mask, 2, "delta"))
        self.assertIsNone(anomaly_context_relations(roi, None, 2, "delta"))
        self.assertIsNone(anomaly_context_relations(roi, np.zeros_like(mask), 2, "delta"))
        self.assertIsNone(anomaly_context_relations(roi[0], mask[0], 2, "delta"))
        # Not enough context anywhere in the ROI.
        self.assertIsNone(anomaly_context_relations(roi, np.ones_like(mask), 2, "delta"))
        self.assertIsNone(anomaly_context_relations(roi, mask, 2, "delta", min_context_size=10_000))

    def test_classes_are_measured_separately_on_request(self):
        roi, mask = _roi(background=0.5, anomaly=0.2)
        roi[0, 6:10, 8:11] += 0.6   # the right half is bright
        mask[0, 6:10, 8:11] = 2

        joint = anomaly_context_relations(roi, mask, 2, "delta")
        separate = anomaly_context_relations(roi, mask, 2, "delta", norm_classes_separately=True)

        self.assertIn("median_delta", joint)
        self.assertEqual(set(separate), {1, 2})
        self.assertAlmostEqual(separate[1]["median_delta"], -0.3, places=2)
        self.assertAlmostEqual(separate[2]["median_delta"], 0.3, places=2)

    def test_3d_rois_are_supported(self):
        roi = np.full((1, 8, 8, 8), 0.5, dtype=np.float32)
        roi += np.linspace(-0.01, 0.01, 512, dtype=np.float32).reshape(1, 8, 8, 8)
        mask = np.zeros((1, 8, 8, 8), dtype=np.uint8)
        roi[0, 3:5, 3:5, 3:5] -= 0.3
        mask[0, 3:5, 3:5, 3:5] = 1
        relation = anomaly_context_relations(roi, mask, 1, "delta")
        self.assertAlmostEqual(relation["median_delta"], -0.3, places=2)


class IntensityMatchingTests(unittest.TestCase):
    def setUp(self):
        # The anomaly arrives in an arbitrary scale, as after denormalization.
        self.roi, self.roi_mask = _roi(background=0.5, anomaly=0.2)
        self.mask = self.roi_mask[0]
        self.anomaly = np.where(self.mask > 0, 40.0, 10.0).astype(np.float32)[None]
        self.anomaly += np.linspace(-1, 1, 256, dtype=np.float32).reshape(1, 16, 16)
        self.control = np.full((1, 16, 16), 0.7, dtype=np.float32)
        self.control += np.linspace(-0.01, 0.01, 256, dtype=np.float32).reshape(1, 16, 16)

    def _median(self, matched):
        return float(np.median(matched[0][self.mask > 0]))

    def test_delta_relation_is_restored_on_the_control(self):
        matched = _match(self.anomaly, self.control, self.mask, self.roi, self.roi_mask)
        # 0.3 darker than the surroundings in the source, so 0.4 on a 0.7 control.
        self.assertAlmostEqual(self._median(matched), 0.4, places=2)
        # Pixels outside the anomaly are not changed.
        np.testing.assert_array_equal(matched[0][self.mask == 0], self.anomaly[0][self.mask == 0])

    def test_ratio_relation_is_restored_on_the_control(self):
        matched = _match(
            self.anomaly, self.control, self.mask, self.roi, self.roi_mask,
            fusion_relation_mode="ratio",
        )
        self.assertAlmostEqual(self._median(matched), 0.7 * 0.4, places=2)

    def test_without_relation_the_anomaly_takes_the_context_level(self):
        matched = _match(
            self.anomaly, self.control, self.mask, self.roi, self.roi_mask,
            fusion_restore_anomaly_bg_relation=False,
        )
        self.assertAlmostEqual(self._median(matched), 0.7, places=2)
        # The same happens when the ROI of the real anomaly is not available.
        without_roi = _match(self.anomaly, self.control, self.mask, None, None)
        self.assertAlmostEqual(self._median(without_roi), 0.7, places=2)

    def test_partial_alpha_is_compensated_before_blending(self):
        alpha = 0.5
        matched = _match(self.anomaly, self.control, self.mask, self.roi, self.roi_mask, alpha=alpha)
        blended = alpha * matched + (1 - alpha) * self.control
        # After blending with half weight the anomaly still reaches its target level.
        self.assertAlmostEqual(self._median(blended), 0.4, places=2)

    def test_whole_image_and_fallback_context_modes(self):
        control = self.control.copy()
        control[0, :3, :] = 0.1   # a dark band far from the anomaly
        ring = _match(self.anomaly, control, self.mask, None, None, fusion_normalization_border_width=2)
        whole = _match(self.anomaly, control, self.mask, None, None, fusion_normalization_border_width=-1)
        outside = _match(self.anomaly, control, self.mask, None, None, fusion_normalization_border_width=0)

        self.assertAlmostEqual(self._median(ring), 0.7, places=2)
        self.assertAlmostEqual(self._median(whole), float(np.median(control)), places=2)
        self.assertAlmostEqual(self._median(outside), float(np.median(control[0][self.mask == 0])), places=2)

    def test_too_little_context_leaves_the_anomaly_unchanged(self):
        matched = _match(
            self.anomaly, self.control, self.mask, self.roi, self.roi_mask,
            fusion_relation_min_context_size=10_000,
        )
        self.assertIs(matched, self.anomaly)

    def test_classes_are_matched_to_their_own_relation(self):
        roi, roi_mask = self.roi.copy(), self.roi_mask.copy()
        roi[0, 6:10, 8:11] += 0.6
        roi_mask[0, 6:10, 8:11] = 2
        target_mask = roi_mask[0]

        joint = _match(self.anomaly, self.control, target_mask, roi, roi_mask)
        separate = _match(
            self.anomaly, self.control, target_mask, roi, roi_mask,
            fusion_relation_norm_classes_separately=True,
        )

        first, second = target_mask == 1, target_mask == 2
        self.assertAlmostEqual(float(np.median(separate[0][first])), 0.4, places=2)
        self.assertAlmostEqual(float(np.median(separate[0][second])), 1.0, places=2)
        # Jointly, one transform maps both classes, so they stay close together.
        joint_gap = abs(float(np.median(joint[0][first])) - float(np.median(joint[0][second])))
        self.assertLess(joint_gap, 0.1)

    def test_matched_values_stay_inside_the_range_of_the_control(self):
        # 0.5 brighter than the surroundings is impossible on a 0.9 control in [0, 1].
        roi, roi_mask = _roi(background=0.3, anomaly=0.8)
        control = np.full((1, 16, 16), 0.9, dtype=np.float32)
        control += np.linspace(-0.01, 0.01, 256, dtype=np.float32).reshape(1, 16, 16)
        matched = _match(self.anomaly, control, self.mask, roi, roi_mask)
        inside = matched[0][self.mask > 0]
        self.assertLessEqual(float(inside.max()), 1.0)
        self.assertGreaterEqual(float(inside.min()), 0.0)
        self.assertGreater(float(np.median(inside)), 0.9)


class BoundFittingTests(unittest.TestCase):
    def test_values_inside_the_bounds_and_missing_bounds_are_unchanged(self):
        values = np.array([0.2, 0.8], dtype=np.float32)
        self.assertIs(_fit_values_into_bounds(values, None), values)
        np.testing.assert_array_equal(_fit_values_into_bounds(values, (0.0, 1.0)), values)

    def test_a_fitting_spread_is_shifted_into_the_bounds(self):
        np.testing.assert_allclose(
            _fit_values_into_bounds(np.array([0.9, 1.3]), (0.0, 1.0)), [0.6, 1.0], atol=1e-6
        )
        np.testing.assert_allclose(
            _fit_values_into_bounds(np.array([-0.2, 0.3]), (0.0, 1.0)), [0.0, 0.5], atol=1e-6
        )

    def test_a_too_large_spread_is_compressed(self):
        fitted = _fit_values_into_bounds(np.array([-1.0, 0.0, 3.0]), (0.0, 1.0))
        np.testing.assert_allclose(fitted, [0.0, 0.25, 1.0], atol=1e-6)

    def test_bounds_follow_the_value_range_of_the_control(self):
        self.assertEqual(infer_output_intensity_bounds(np.array([0.2, 0.9])), (0.0, 1.0))
        self.assertEqual(infer_output_intensity_bounds(np.array([3.0, 200.0])), (0.0, 255.0))
        self.assertIsNone(infer_output_intensity_bounds(np.array([np.nan, np.inf])))
        self.assertIsNone(infer_output_intensity_bounds(np.array([-5.0, 5.0])))


if __name__ == "__main__":
    unittest.main()
