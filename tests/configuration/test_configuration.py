"""Tests for configuration validation, serialization, and compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

from hybrid_sample_generator.configuration.root import Configuration, load_config_file
from hybrid_sample_generator.configuration.augmentation import MaskTransformConfiguration
from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


class ConfigurationTests(unittest.TestCase):
    def test_default_configuration_and_model_selection(self):
        config = Configuration("defaults")
        config.validate()
        self.assertEqual(config.model.name, "cVAE_ConvNeXt_2D")
        self.assertEqual(config.extraction.anomaly_size, (3, 64, 64))
        config.model.parameters.set_model_param("z_channels", 123)
        config.model.set_model("VAE_ResNet_3D")
        config.extraction.anomaly_size = (1, 16, 16, 16)
        config.validate()
        self.assertEqual(config.model.parameters.min["in_channels"], 1)
        self.assertNotEqual(config.model.parameters.min["z_channels"], 123)
        before = config.model.to_dict()
        with self.assertRaises(ValueError):
            config.model.set_model("unknown-model")
        self.assertEqual(config.model.to_dict(), before)

    def test_channel_changes_preserve_hyperparameters_and_survive_loading(self):
        config = Configuration("channels")
        config.model.parameters.set_model_param_range("z_channels", 16, 32)
        config.extraction.anomaly_size = (1, 32, 32)
        self.assertEqual(config.model.parameters.min["in_channels"], 1)
        self.assertEqual(config.model.parameters.max["in_channels"], 1)
        self.assertEqual(config.model.parameters.min["z_channels"], 16)
        self.assertEqual(config.model.parameters.max["z_channels"], 32)
        loaded = Configuration.from_dict(config.to_dict())
        loaded.extraction.anomaly_size = (2, 32, 32)
        loaded.validate()
        self.assertEqual(loaded.model.parameters.min["in_channels"], 2)
        self.assertEqual(config.model.parameters.min["in_channels"], 1)
        self.assertEqual(loaded.model.parameters.max["z_channels"], 32)
        with self.assertRaises(AttributeError):
            loaded.model.parameters.set_model_param("in_channels", 4)

    def test_invalid_saved_channels_are_rejected(self):
        values = Configuration("invalid-channels").to_dict()
        values["model"]["parameters"]["min"]["in_channels"] = 17
        with self.assertRaisesRegex(ValueError, "in_channels"):
            Configuration.from_dict(values)

    def test_name_assignment_initializes_model_parameters(self):
        from hybrid_sample_generator.generation.registry import get_model_spec
        config = Configuration("assignment")
        config.extraction.anomaly_size = (1, 8, 8)
        config.model.name = "VAE_ResNet_2D"
        self.assertEqual(config.model.parameters.to_dict(),
                         get_model_spec("VAE_ResNet_2D").build_configuration(1).to_dict())

    def test_facade_and_serialization_reject_incomplete_dimension_change(self):
        from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("invalid", save_path=root)
            config.extraction.anomaly_size = (1, 8, 8, 8)
            with self.assertRaisesRegex(ValueError, "spatial dimensions"):
                HybridDataGenerator(config)
            with self.assertRaises(ValueError):
                config.to_dict()
            self.assertFalse(Path(config.study.folder).exists())

    def test_configuration_round_trip_uses_sectioned_plain_json(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("config-test", save_path=root)
            config.extraction.anomaly_size = (3, 32, 32)
            config.model.set_model("VAE_ResNet_2D")
            config.matching.routine = "local"
            config.matching.hybrids_per_original = 3
            config.matching.anomalies_per_hybrid = 2
            config.generation.variants_per_real_anomaly = 5
            config.training.batch_size = 8
            config.training.num_trials = 4
            config.training.trial_selection = 2

            path = Path(config.save_config_file())
            serialized = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_config_file(path)

            self.assertEqual(serialized["schema_version"], Configuration.SCHEMA_VERSION)
            self.assertNotIn("uses_masks", serialized["model"])
            self.assertNotIn("checkpoint", serialized["fusion"])
            self.assertFalse(hasattr(loaded.model, "uses_masks"))
            self.assertEqual(serialized["matching"]["anomalies_per_hybrid"], 2)
            self.assertEqual(serialized["matching"]["hybrids_per_original"], 3)
            self.assertEqual(serialized["generation"]["variants_per_real_anomaly"], 5)
            self.assertEqual(serialized["training"]["num_trials"], 4)
            self.assertEqual(serialized["training"]["trial_selection"], 2)
            self.assertEqual(loaded.to_dict(), config.to_dict())
            self.assertIsInstance(
                loaded.augmentation.mask_transforms,
                MaskTransformConfiguration,
            )

    def test_mask_transform_configuration_builds_runtime_generator(self):
        config = MaskTransformConfiguration(
            use_mask_transform=False,
            padding_factor=3,
            local_as_global=True,
        )
        config.setGlobalParam("rotate", probability=0.25, max_rotation=12.0)

        generator = TransformGenerator.from_config(
            config,
            anomaly_size=(1, 16, 16),
            background_threshold=0.01,
            seed=7,
        )

        self.assertEqual(generator.padding_factor, 3)
        self.assertTrue(generator.mask_transform_local_as_global)
        self.assertEqual(generator.global_transform_probs["rotate"], 0.25)
        self.assertEqual(generator.transform_params["rotate"]["max_rotation"], 12.0)

    def test_model_and_anomaly_size_dimensions_must_match(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            config = Configuration("dimension-test", save_path=root)
            config.model.set_model("VAE_ResNet_3D")
            config.extraction.anomaly_size = (1, 16, 16)
            config.validate()


    def test_matching_configuration_rejects_invalid_weights(self):
        config = MatchingConfiguration(intensity_weight=0, gradient_weight=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_matching_configuration_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            MatchingConfiguration(hybrids_per_original=0).validate()

    def test_training_configuration_rejects_invalid_trial_settings(self):
        config = Configuration("training-config-test", study_folder="/tmp/training-config-test")
        config.extraction.anomaly_size = (1, 8, 8)
        config.model.set_model("VAE_ResNet_2D")

        config.training.num_trials = 0
        with self.assertRaisesRegex(ValueError, "num_trials"):
            config.training.validate()

        config.training.num_trials = 1
        for selection in ("newest", -1, True):
            config.training.trial_selection = selection
            with self.assertRaisesRegex(ValueError, "trial_selection"):
                config.training.validate()


if __name__ == "__main__":
    unittest.main()
