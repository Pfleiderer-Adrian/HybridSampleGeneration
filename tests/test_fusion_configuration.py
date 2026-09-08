import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fusion_backend.classical import ClassicalFusionBackend, Config as ClassicalConfig
from fusion_backend.fusion_configuration import FusionSettings
from fusion_backend.fusion_registry import get_fusion_backend_spec
from fusion_backend.learned_residual_alpha import Config as LearnedConfig, LearnedResidualAlphaFusionBackend
from synthesizer.Configuration import Configuration, load_config_file
from synthesizer.HybridDataGenerator import HybridDataGenerator
from synthesizer.configuration.extraction import ExtractionConfiguration, RoiConfiguration


class FusionConfigurationTests(unittest.TestCase):
    def test_schema_three_round_trip_preserves_backend_dataclass(self):
        for backend, config_cls, field, value in (
            ('classical', ClassicalConfig, 'max_alpha', 0.7),
            ('learned_residual_alpha', LearnedConfig, 'residual_scale', 0.1),
        ):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as root:
                config = Configuration('fusion-test', 'VAE_ResNet_2D', (1, 16, 16), save_path=root)
                config.fusion.set_backend(backend)
                setattr(config.fusion.parameters, field, value)
                loaded = load_config_file(config.save_config_file())
                self.assertIsInstance(loaded.fusion.parameters, config_cls)
                self.assertEqual(getattr(loaded.fusion.parameters, field), value)
                self.assertEqual(loaded.to_dict(), config.to_dict())
                self.assertEqual(config.fusion.to_dict()['parameters'], asdict(config.fusion.parameters))

    def test_partial_parameters_receive_defaults_and_unknown_names_fail(self):
        values = {'backend': 'classical', 'parameters': {'max_alpha': 0.7}}
        settings = FusionSettings.from_dict(values)
        self.assertEqual(settings.parameters.max_alpha, 0.7)
        self.assertEqual(settings.parameters.sq, ClassicalConfig().sq)
        values['parameters']['max_alhpa'] = 0.5
        with self.assertRaises(TypeError):
            FusionSettings.from_dict(values)
        with self.assertRaises(AttributeError):
            settings.parameters.max_alhpa = 0.5

    def test_invalid_types_ranges_and_backend_mismatches_fail(self):
        for parameters in (
            ClassicalConfig(max_alpha=1.1), ClassicalConfig(max_alpha='0.5'),
            ClassicalConfig(upsampling_factor=True), ClassicalConfig(sq=float('nan')),
            ClassicalConfig(selected_confidence='invalid'), ClassicalConfig(fusion_relation_mode='invalid'),
            LearnedConfig(spatial_dims=4), LearnedConfig(depth=0),
            LearnedConfig(train_lr=-1), LearnedConfig(residual_l1=-1),
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                parameters.validate()
        with self.assertRaises(TypeError):
            FusionSettings('classical', LearnedConfig()).validate()
        with self.assertRaises(TypeError):
            get_fusion_backend_spec('classical').build(LearnedConfig())
        with self.assertRaises(TypeError):
            ClassicalFusionBackend(fusion_params={'max_alpha': 0.8})

    def test_backend_switch_resets_parameters_and_checkpoint(self):
        settings = FusionSettings.for_backend('classical')
        settings.checkpoint = 'previous.pt'
        settings.set_backend('learned_residual_alpha')
        self.assertIsInstance(settings.parameters, LearnedConfig)
        self.assertIsNone(settings.checkpoint)

    def test_generator_rebuilds_backend_after_settings_change(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration('cache-test', 'VAE_ResNet_2D', (1, 16, 16), save_path=root)
            generator = HybridDataGenerator(config)
            first = generator._ensure_fusion_backend()
            self.assertIs(generator._ensure_fusion_backend(), first)
            config.fusion.parameters.max_alpha = 0.6
            second = generator._ensure_fusion_backend()
            self.assertIsNot(first, second)
            self.assertEqual(first.params.max_alpha, 0.8)
            self.assertEqual(second.params.max_alpha, 0.6)
            config.fusion.set_backend('learned_residual_alpha')
            third = generator._ensure_fusion_backend()
            self.assertIsInstance(third, LearnedResidualAlphaFusionBackend)
            config.fusion.checkpoint = 'new.pt'
            with patch.object(LearnedResidualAlphaFusionBackend, 'load_checkpoint') as load:
                fourth = generator._ensure_fusion_backend()
                self.assertIsNot(third, fourth)
                load.assert_called_once_with('new.pt')
                self.assertIs(generator._ensure_fusion_backend(), fourth)
                self.assertEqual(load.call_count, 1)
            injected = HybridDataGenerator(config, fusion_backend=first)
            self.assertIs(injected._ensure_fusion_backend(), first)

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


if __name__ == '__main__':
    unittest.main()
