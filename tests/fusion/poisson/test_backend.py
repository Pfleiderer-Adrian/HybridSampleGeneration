"""Integration tests for the Poisson fusion backend."""

import unittest
import warnings

import numpy as np

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.fusion.poisson import Config, PoissonFusionBackend


class PoissonFusionBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = PoissonFusionBackend(
            Config(fusion_normalization_border_width=None)
        )

    def test_fuse_preserves_output_and_roi_shapes_in_2d_and_3d(self):
        for spatial_ndim in (2, 3):
            with self.subTest(spatial_ndim=spatial_ndim):
                control = np.full(
                    (2, *((8,) * spatial_ndim)), 0.2, dtype=np.float32
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    output = self.backend.fuse(
                        self._sample(spatial_ndim, channels=2),
                        control,
                        (0.5,) * spatial_ndim,
                        extraction_config=self._extraction_config(spatial_ndim),
                    )

                self.assertEqual(output.image.shape, control.shape)
                self.assertEqual(output.image.dtype, np.float32)
                self.assertEqual(output.segmentation.shape, control.shape)
                self.assertEqual(output.roi.shape, (2, *((4,) * spatial_ndim)))
                self.assertEqual(output.roi_mask.shape, output.roi.shape)
                self.assertGreater(int(output.segmentation.sum()), 0)
                self.assertEqual(output.metrics["solver"], "cg")

    def test_3d_performance_warning_is_emitted_once(self):
        control = np.full((1, 8, 8, 8), 0.2, dtype=np.float32)
        sample = self._sample(3)
        extraction = self._extraction_config(3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.backend.fuse(
                sample, control, (0.5, 0.5, 0.5), extraction_config=extraction
            )
            self.backend.fuse(
                sample, control, (0.5, 0.5, 0.5), extraction_config=extraction
            )

        performance_warnings = [
            item for item in caught if item.category is RuntimeWarning
        ]
        self.assertEqual(len(performance_warnings), 1)
        message = str(performance_warnings[0].message)
        self.assertIn("mask voxels: 8", message)
        self.assertIn("channels: 1", message)

    def test_2d_does_not_emit_performance_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.backend.fuse(
                self._sample(2),
                np.full((1, 8, 8), 0.2, dtype=np.float32),
                (0.5, 0.5),
                extraction_config=self._extraction_config(2),
            )
        self.assertFalse(any(item.category is RuntimeWarning for item in caught))

    def test_removed_background_artifacts_do_not_change_result(self):
        clean = self._sample(2)
        artifact = self._sample(2)
        artifact["synth_anomaly"][:, 0, :] = 100.0
        control = np.full((1, 8, 8), 0.2, dtype=np.float32)
        extraction = self._extraction_config(2)

        clean_output = self.backend.fuse(
            clean, control, (0.5, 0.5), extraction_config=extraction
        )
        artifact_output = self.backend.fuse(
            artifact, control, (0.5, 0.5), extraction_config=extraction
        )

        np.testing.assert_allclose(clean_output.image, artifact_output.image)

    def test_empty_mask_returns_unchanged_control(self):
        sample = self._sample(2)
        sample["tgt_mask"][:] = 0
        control = np.full((1, 8, 8), 0.2, dtype=np.float32)
        output = self.backend.fuse(
            sample,
            control,
            (0.5, 0.5),
            extraction_config=self._extraction_config(2),
        )
        np.testing.assert_array_equal(output.image, control)
        self.assertEqual(int(output.segmentation.sum()), 0)
        self.assertIsNone(output.roi)
        self.assertEqual(output.metrics["unknowns"], 0)

    @staticmethod
    def _sample(spatial_ndim, channels=1):
        spatial_shape = (4,) * spatial_ndim
        anomaly = np.zeros((channels, *spatial_shape), dtype=np.float32)
        mask = np.zeros((1, *spatial_shape), dtype=np.uint8)
        foreground = (slice(None), *((slice(1, 3),) * spatial_ndim))
        anomaly[foreground] = 0.8
        mask[foreground] = 1
        return {
            "synth_anomaly": anomaly,
            "anomaly_meta": {"scale_factor": (1.0,) * spatial_ndim},
            "tgt_mask": mask,
            "anomaly_roi": anomaly.copy(),
            "anomaly_roi_mask": np.repeat(mask, channels, axis=0),
        }

    @staticmethod
    def _extraction_config(spatial_ndim):
        config = ExtractionConfiguration((1, *((4,) * spatial_ndim)))
        config.roi.fixed_size = (4,) * spatial_ndim
        return config


if __name__ == "__main__":
    unittest.main()
