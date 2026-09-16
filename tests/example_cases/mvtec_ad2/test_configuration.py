"""Preset layering, independent configurations and saved-study behavior."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.mvtec_ad2.configuration import Configuration, Experiment
from examples.mvtec_ad2.pipeline import run_existing_studies, run_new_experiment
from examples.mvtec_ad2.presets import (
    CATEGORY_PRESETS, apply_downstream_preset, create_configuration,
)
from examples.mvtec_ad2.studies import open_study


class ConfigurationTests(unittest.TestCase):
    def test_category_overrides_only_change_declared_fields(self):
        for category in CATEGORY_PRESETS:
            with self.subTest(category=category):
                config = create_configuration(category)
                shared = create_configuration(category, apply_category_overrides=False)
                self.assertEqual(shared.training.epochs, 1000)
                self.assertEqual(shared.downstream.training.batch_size, 8)
                self.assertEqual(shared.generation.variation_strength, 1.25)
                if category == 'can':
                    shared.generation.variation_strength = 1.5
                    shared.fusion.parameters.max_alpha = 0.9
                    shared.fusion.parameters.sobel_threshold = 0.05
                    shared.extraction.roi.min_size = (256, 256)
                elif category == 'fabric':
                    shared.extraction.roi.min_size = (128, 128)
                self.assertEqual(config.to_dict(), shared.to_dict())

    def test_mutable_configuration_sections_are_independent(self):
        first = create_configuration('can')
        second = create_configuration('can')
        before = deepcopy(second.to_dict())
        first.training.early_stopping['patience'] = 1
        first.downstream.training.epochs = 2
        first.model.parameters.min['z_channels'] = 20
        self.assertEqual(second.to_dict(), before)
        self.assertEqual(create_configuration('wall plugs').extraction.anomaly_size, (1, 64, 64))
        with self.assertRaisesRegex(ValueError, 'Unknown MVTec'):
            create_configuration('unknown')

    def test_saved_downstream_settings_survive_batch_training_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            config = create_configuration('can', save_path=folder)
            config.downstream.training.epochs = 7
            path = Path(config.save_config_file())
            before = path.read_bytes()
            with patch('examples.mvtec_ad2.pipeline.run_study') as execute, patch(
                'examples.mvtec_ad2.studies.create_configuration', side_effect=AssertionError('No presets')
            ):
                run_existing_studies([path.parent], steps=('train_downstream',))
            study = execute.call_args.args[0]
            self.assertEqual(study.config.downstream.training.epochs, 7)
            self.assertEqual(path.read_bytes(), before)
            before_config = study.config.to_dict()
            apply_downstream_preset(study.config, study.category)
            self.assertEqual(study.config.downstream.training.epochs, 100)
            after = study.config.to_dict()
            del before_config['downstream'], after['downstream']
            self.assertEqual(before_config, after)
            self.assertEqual(open_study(path.parent).config.downstream.training.epochs, 7)

    def test_saved_configuration_roundtrip_and_legacy_downstream_defaults(self):
        config = create_configuration('can')
        values = json.loads(json.dumps(config.to_dict()))
        restored = Configuration.from_dict(values)
        self.assertEqual(json.loads(json.dumps(restored.to_dict())), values)
        del values['downstream']
        self.assertEqual(Configuration.from_dict(values).downstream.training.epochs, 100)

    def test_invalid_experiment_does_not_prepare_studies(self):
        with patch('examples.mvtec_ad2.pipeline.prepare_studies') as prepare:
            with self.assertRaises(ValueError):
                run_new_experiment(Experiment(Path('unused'), Path('unused'), ()))
            prepare.assert_not_called()
