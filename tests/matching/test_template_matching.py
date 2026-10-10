"""Numerical matching and candidate boundary regressions."""
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.matching.template_matching import (
    template_matching, prepare_matching_array, _window_sum, _correlation,
)
from hybrid_sample_generator.matching.candidate_selection import placement_count, check_roi_overlap
from hybrid_sample_generator.matching.pair_matcher import matcher_signature


class TemplateMatchingTests(unittest.TestCase):
    def test_window_moments_match_direct_sums_in_2d_and_3d(self):
        for shape, window in (((7, 9), (3, 4)), ((6, 7, 8), (2, 3, 4))):
            array = np.random.default_rng(12).normal(size=shape)
            expected = np.lib.stride_tricks.sliding_window_view(array, window).sum(
                axis=tuple(range(-len(window), 0)))
            np.testing.assert_allclose(_window_sum(array, window), expected, atol=1e-12)

    def test_float32_large_bright_image_finds_true_patch_with_bounded_score(self):
        rng = np.random.default_rng(19)
        control = np.full((1, 768, 1024), 200, dtype=np.float32)
        template = (200 + rng.normal(0, 2, size=(1, 19, 23))).astype(np.float32)
        control[:, 511:530, 713:736] = template
        for intensity, gradient in ((1, 0), (0, 1), (1, 2)):
            score, center = template_matching(template, control, MatchingConfiguration(
                intensity_weight=intensity, gradient_weight=gradient))
            self.assertEqual(center, (520.5, 724.5))
            self.assertTrue(.5 < score <= 1)
        prepared = prepare_matching_array(control, with_gradient=True)
        self.assertEqual(prepared.intensity.dtype, np.float64)
        self.assertEqual(prepared.gradient.dtype, np.float64)

    def test_constant_templates_and_flat_search_windows_are_invalid(self):
        template = np.arange(16, dtype=np.float64).reshape(4, 4)
        control = np.full((12, 12), 200., dtype=np.float64)
        self.assertTrue(np.isnan(_correlation(template, control)).all())
        control[6:10, 5:9] = template
        scores = _correlation(template, control)
        self.assertTrue(np.isnan(scores[0, 0]))
        self.assertAlmostEqual(scores[6, 5], 1.)
        for dims in (2, 3):
            self.assertEqual(template_matching(
                np.ones((1, *((4,) * dims))),
                np.ones((1, *((8,) * dims))), MatchingConfiguration()), (-2., None))

    def test_nearly_constant_search_windows_are_rejected(self):
        template = np.arange(16, dtype=np.float64).reshape(4, 4)
        control = 200 + np.arange(144, dtype=np.float64).reshape(12, 12) * 1e-10
        self.assertTrue(np.isnan(_correlation(template, control)).all())

    def test_known_positions_in_both_dimensions_and_weight_modes(self):
        for dims in (2, 3):
            template = np.zeros((1, *((4,) * dims)))
            template[(slice(None), *((slice(1, 3),) * dims))] = np.random.default_rng(7).random((1, *((2,) * dims)))
            control = np.zeros((1, *((12,) * dims)))
            control[(slice(None), *((slice(5, 9),) * dims))] = template
            for intensity, gradient in ((1, 0), (0, 1), (1, 2)):
                with self.subTest(dims=dims, weights=(intensity, gradient)):
                    score, center = template_matching(template, control, MatchingConfiguration(intensity_weight=intensity, gradient_weight=gradient))
                    self.assertEqual(center, (7.0,) * dims)
                    self.assertGreater(score, .5)

    def test_constant_gradient_only_and_oversized_templates_are_rejected(self):
        for dims in (2, 3):
            config = MatchingConfiguration(intensity_weight=0, gradient_weight=1)
            self.assertEqual(template_matching(np.ones((1, *((4,) * dims))), np.ones((1, *((8,) * dims))), config), (-2., None))
            self.assertEqual(template_matching(np.ones((1, *((9,) * dims))), np.ones((1, *((8,) * dims))), MatchingConfiguration()), (-2., None))

    def test_placement_counts_are_seeded_bounded_and_at_least_one(self):
        config = MatchingConfiguration(anomalies_per_hybrid=2, max_anomalies_per_hybrid_deviation=4)
        counts = [placement_count(config, seed) for seed in range(100)]
        self.assertTrue(all(1 <= count <= 6 for count in counts))
        self.assertGreater(len(set(counts)), 1)
        self.assertEqual(counts, [placement_count(config, seed) for seed in range(100)])

    def test_touching_rois_do_not_overlap_in_both_dimensions(self):
        for dims in (2, 3):
            used = [((0.,) * dims, (4,) * dims)]
            self.assertFalse(check_roi_overlap((4.,) + (0.,) * (dims - 1), (4,) * dims, used))
            self.assertTrue(check_roi_overlap((3.9,) + (0.,) * (dims - 1), (4,) * dims, used))

    def test_cache_signature_changes_with_weights_and_algorithm_version(self):
        config = MatchingConfiguration()
        original = matcher_signature(config)
        config.gradient_weight += 1
        self.assertNotEqual(original, matcher_signature(config))
        with patch('hybrid_sample_generator.matching.pair_matcher.MATCHER_ALGORITHM_VERSION', 999):
            self.assertNotEqual(original, matcher_signature(MatchingConfiguration()))
