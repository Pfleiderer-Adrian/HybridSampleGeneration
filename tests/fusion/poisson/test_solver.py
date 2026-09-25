"""Tests for the sparse 2D and 3D Poisson solver."""

import unittest

import numpy as np

from hybrid_sample_generator.fusion.poisson.solver import solve_poisson


class PoissonSolverTests(unittest.TestCase):
    def test_constant_source_contrast_is_preserved_in_2d(self):
        target = np.full((2, 7, 7), 0.2, dtype=np.float32)
        source = target.copy()
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        source[:, mask] = np.array([[0.8], [0.6]], dtype=np.float32)

        result, metrics = solve_poisson(source, target, mask)

        np.testing.assert_allclose(result[0, mask], 0.8, atol=1e-5)
        np.testing.assert_allclose(result[1, mask], 0.6, atol=1e-5)
        np.testing.assert_array_equal(result[:, ~mask], target[:, ~mask])
        self.assertEqual(metrics.unknowns, 9)
        self.assertEqual(len(metrics.iterations), 2)

    def test_mixed_guidance_keeps_stronger_target_gradient(self):
        target = np.zeros((1, 3, 3), dtype=np.float32)
        target[0, 1, 1] = 1.0
        source = np.zeros_like(target)
        source[0, 1, 1] = 0.5
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        source_result, _ = solve_poisson(source, target, mask)
        mixed_result, _ = solve_poisson(
            source, target, mask, guidance_mode="mixed"
        )

        self.assertAlmostEqual(float(source_result[0, 1, 1]), 0.5)
        self.assertAlmostEqual(float(mixed_result[0, 1, 1]), 1.0)

    def test_true_3d_solution_uses_voxel_neighborhood(self):
        target = np.full((1, 5, 5, 5), 0.1, dtype=np.float32)
        source = target.copy()
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        source[:, mask] = 0.7
        source[0, 2, 2, 2] = 0.9

        result, metrics = solve_poisson(source, target, mask)

        self.assertEqual(metrics.unknowns, 27)
        self.assertGreater(float(result[0, 2, 2, 2]), float(result[0, 1, 2, 2]))
        np.testing.assert_array_equal(result[:, ~mask], target[:, ~mask])

    def test_full_domain_mask_is_anchored(self):
        target = np.full((1, 3, 3), 0.2, dtype=np.float32)
        source = np.full((1, 3, 3), 0.8, dtype=np.float32)
        mask = np.ones((3, 3), dtype=bool)

        result, metrics = solve_poisson(source, target, mask)

        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(metrics.anchored_components, 1)
        np.testing.assert_allclose(result, 0.2, atol=1e-4)

    def test_empty_mask_returns_target_without_iterations(self):
        target = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
        result, metrics = solve_poisson(target + 1, target, np.zeros((5, 5)))
        np.testing.assert_array_equal(result, target)
        self.assertEqual(metrics.unknowns, 0)
        self.assertEqual(metrics.iterations, tuple())

    def test_invalid_inputs_are_rejected(self):
        source = np.zeros((1, 3, 3), dtype=np.float32)
        target = source.copy()
        mask = np.ones((3, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "guidance_mode"):
            solve_poisson(source, target, mask, guidance_mode="invalid")
        with self.assertRaisesRegex(ValueError, "not both zero"):
            solve_poisson(source, target, mask, rtol=0.0, atol=0.0)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            solve_poisson(source, target, np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
