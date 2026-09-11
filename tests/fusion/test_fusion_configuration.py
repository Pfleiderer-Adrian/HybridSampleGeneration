"""Tests for fusion configuration and backend selection."""

import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import Mock

from hybrid_sample_generator.fusion.classical import ClassicalFusionBackend, Config as ClassicalConfig
from hybrid_sample_generator.fusion.settings import FusionSettings
from hybrid_sample_generator.fusion.registry import get_fusion_backend_spec
from hybrid_sample_generator.fusion.service import FusionService
from hybrid_sample_generator.configuration.root import Configuration, load_config_file

class FusionConfigurationTests(unittest.TestCase):
    def test_fusion_configuration_round_trip_preserves_backend_dataclass(self):
        for backend, config_cls, field, value in (
            ('classical', ClassicalConfig, 'max_alpha', 0.7),
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
        with self.assertRaises(TypeError):
            FusionSettings.from_dict({"backend": "classical", "parameters": {}, "checkpoint": "old.pt"})
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
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                parameters.validate()
        with self.assertRaises(TypeError):
            ClassicalFusionBackend(fusion_params={'max_alpha': 0.8})

    def test_experimental_backend_is_not_registered(self):
        with self.assertRaises(ValueError):
            get_fusion_backend_spec("learned_residual_alpha")
        with self.assertRaises(ValueError):
            FusionSettings.for_backend("learned_residual_alpha")

    def test_service_rebuilds_backend_after_settings_change(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration('cache-test', 'VAE_ResNet_2D', (1, 16, 16), save_path=root)
            service = FusionService(
                config.fusion,
                config.extraction,
                config.study.seed,
                Mock(),
                Mock(),
                Mock(),
            )
            first = service._ensure_backend()
            self.assertIs(service._ensure_backend(), first)
            config.fusion.parameters.max_alpha = 0.6
            second = service._ensure_backend()
            self.assertIsNot(first, second)
            self.assertEqual(first.params.max_alpha, 0.8)
            self.assertEqual(second.params.max_alpha, 0.6)
            injected = FusionService(
                config.fusion,
                config.extraction,
                config.study.seed,
                Mock(),
                Mock(),
                Mock(),
                backend=first,
            )
            self.assertIs(injected._ensure_backend(), first)

if __name__ == '__main__':
    unittest.main()
