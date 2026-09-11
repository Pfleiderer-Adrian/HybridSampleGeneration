"""Tests for fusion configuration, backends, and checkpoints."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.fusion.learned_residual_alpha import Config as LearnedConfig, LearnedResidualAlphaFusionBackend
from experiments.fusion.learned_residual_alpha.preprocessing import support_mask
from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration, RoiConfiguration

class LearnedResidualAlphaTests(unittest.TestCase):
    def test_checkpoint_defaults_explicit_overrides_and_architecture_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'fusion.pt')
            parameters = LearnedConfig(base_channels=4, depth=1, residual_scale=0.2)
            original = LearnedResidualAlphaFusionBackend(parameters).warmup((1, 8, 8), device='cpu')
            original.save_checkpoint(path)
            restored = LearnedResidualAlphaFusionBackend()
            restored.load_checkpoint(path, device='cpu')
            self.assertEqual(restored.params, parameters)
            for key, value in original.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[key], value)
            override = LearnedResidualAlphaFusionBackend(LearnedConfig(base_channels=4, depth=1, residual_scale=0.1))
            override.load_checkpoint(path, device='cpu')
            self.assertEqual(override.params.residual_scale, 0.1)
            for parameters in (LearnedConfig(base_channels=8, depth=1), LearnedConfig(base_channels=4, depth=2),
                               LearnedConfig(base_channels=4, depth=1, spatial_dims=3)):
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    LearnedResidualAlphaFusionBackend(parameters).load_checkpoint(path, device='cpu')

    def test_learned_fusion_uses_configured_alpha_in_2d_and_3d(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                shape = (1, *((8,) * dims))
                control = np.full(shape, 0.2, dtype=np.float32)
                anomaly = np.full((1, *((2,) * dims)), 0.8, dtype=np.float32)
                backend = LearnedResidualAlphaFusionBackend(LearnedConfig(
                    base_channels=4, depth=1, base_alpha=0.5, base_alpha_blur_sigma=0,
                    residual_scale=0, alpha_delta_scale=0,
                ))
                extraction = ExtractionConfiguration(shape, roi=RoiConfiguration(fixed_size=(4,) * dims))
                # Keep the smoke test on CPU even when CUDA is available.
                with patch('torch.cuda.is_available', return_value=False):
                    output = backend.fuse(
                        {'synth_anomaly': anomaly, 'tgt_mask': np.ones_like(anomaly, dtype=np.uint8),
                         'anomaly_meta': {'scale_factor': 1.0}},
                        control, (0.5,) * dims, extraction_config=extraction,
                    )
                self.assertEqual(output.image.shape, shape)
                self.assertEqual(output.roi.shape, (1, *((4,) * dims)))
                mask = output.segmentation > 0
                self.assertEqual(int(mask.sum()), 2 ** dims)
                np.testing.assert_allclose(output.image[mask], 0.5)
                np.testing.assert_allclose(output.image[~mask], control[~mask])

    def test_learned_training_uses_typed_parameters_in_2d_and_3d(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                image = np.full((1, *((8,) * dims)), 0.2, dtype=np.float32)
                mask = np.zeros_like(image, dtype=np.uint8)
                region = (slice(None), *((slice(3, 5),) * dims))
                image[region] = 0.8
                mask[region] = 1
                backend = LearnedResidualAlphaFusionBackend(LearnedConfig(
                    base_channels=4, depth=1, spatial_dims=dims, train_epochs=1, log_every=0,
                ))
                result = backend.train_model([(image, mask, 'sample')], device='cpu')
                self.assertEqual(len(result['train_loss_history']), 1)
                self.assertTrue(np.isfinite(result['train_loss_history'][0]))

    def test_support_mask_dilates_by_requested_border(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[2, 2] = 1.0

        support = support_mask(mask, border_width=1, spatial_dims=2)

        self.assertEqual(int(np.count_nonzero(support)), 9)
        self.assertEqual(support.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
