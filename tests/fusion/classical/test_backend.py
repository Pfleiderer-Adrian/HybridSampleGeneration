"""Characterization tests for the stable classical fusion backend."""

import unittest

import numpy as np

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.fusion.classical import ClassicalFusionBackend, Config


class ClassicalFusionBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = ClassicalFusionBackend(
            Config(
                fusion_variation=False,
                fusion_normalization_border_width=None,
                upsampling_factor=1,
            )
        )

    def test_fuse_preserves_output_and_roi_shapes_in_2d_and_3d(self):
        for spatial_ndim in (2, 3):
            with self.subTest(spatial_ndim=spatial_ndim):
                control_shape = (1, *((8,) * spatial_ndim))
                control = np.full(control_shape, 0.2, dtype=np.float32)
                extraction = self._extraction_config(spatial_ndim)

                output = self.backend.fuse(
                    self._sample(spatial_ndim),
                    control,
                    (0.5,) * spatial_ndim,
                    extraction_config=extraction,
                )

                self.assertEqual(output.image.shape, control.shape)
                self.assertEqual(output.image.dtype, np.float32)
                self.assertEqual(output.segmentation.shape, control.shape)
                self.assertEqual(output.roi.shape, (1, *((4,) * spatial_ndim)))
                self.assertEqual(
                    output.roi_mask.shape,
                    (1, *((4,) * spatial_ndim)),
                )
                self.assertGreater(int(output.segmentation.sum()), 0)

    def test_empty_target_mask_returns_unchanged_control_without_roi(self):
        sample = self._sample(2)
        sample["tgt_mask"] = np.zeros((1, 4, 4), dtype=np.uint8)
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
        self.assertIsNone(output.roi_mask)

    def test_boundary_positions_remain_inside_control_image(self):
        control = np.zeros((1, 8, 8), dtype=np.float32)
        for position in ((0.0, 0.0), (1.0, 1.0)):
            with self.subTest(position=position):
                output = self.backend.fuse(
                    self._sample(2),
                    control,
                    position,
                    extraction_config=self._extraction_config(2),
                )
                self.assertEqual(output.image.shape, control.shape)
                self.assertGreater(int(output.segmentation.sum()), 0)

    def test_fuse_validates_configuration_metadata_mask_and_dimensions(self):
        sample = self._sample(2)
        control = np.zeros((1, 8, 8), dtype=np.float32)
        extraction = self._extraction_config(2)

        with self.assertRaisesRegex(ValueError, "requires extraction_config"):
            self.backend.fuse(sample, control, (0.5, 0.5))

        missing_metadata = dict(sample, anomaly_meta=None)
        with self.assertRaisesRegex(ValueError, "anomaly_meta must be provided"):
            self.backend.fuse(
                missing_metadata,
                control,
                (0.5, 0.5),
                extraction_config=extraction,
            )

        missing_scale = dict(sample, anomaly_meta={})
        with self.assertRaisesRegex(ValueError, "scale_factor"):
            self.backend.fuse(
                missing_scale,
                control,
                (0.5, 0.5),
                extraction_config=extraction,
            )

        missing_mask = dict(sample, tgt_mask=None)
        with self.assertRaisesRegex(ValueError, "requires target_mask"):
            self.backend.fuse(
                missing_mask,
                control,
                (0.5, 0.5),
                extraction_config=extraction,
            )

        with self.assertRaisesRegex(ValueError, "Unexpected shape"):
            self.backend.fuse(
                sample,
                np.zeros((8, 8), dtype=np.float32),
                (0.5, 0.5),
                extraction_config=extraction,
            )

    @staticmethod
    def _sample(spatial_ndim):
        spatial_shape = (4,) * spatial_ndim
        anomaly = np.zeros((1, *spatial_shape), dtype=np.float32)
        mask = np.zeros((1, *spatial_shape), dtype=np.uint8)
        foreground = (slice(None), *((slice(1, 3),) * spatial_ndim))
        anomaly[foreground] = 0.8
        mask[foreground] = 1
        return {
            "synth_anomaly": anomaly,
            "anomaly_meta": {"scale_factor": (1.0,) * spatial_ndim},
            "tgt_mask": mask,
            "anomaly_roi": anomaly.copy(),
            "anomaly_roi_mask": mask.copy(),
        }

    @staticmethod
    def _extraction_config(spatial_ndim):
        config = ExtractionConfiguration((1, *((4,) * spatial_ndim)))
        config.roi.fixed_size = (4,) * spatial_ndim
        return config


if __name__ == "__main__":
    unittest.main()
