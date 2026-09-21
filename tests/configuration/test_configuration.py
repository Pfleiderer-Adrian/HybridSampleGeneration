"""Tests for configuration validation, serialization, and compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

from hybrid_sample_generator.configuration.root import Configuration, load_config_file
from hybrid_sample_generator.configuration.augmentation import MaskTransformConfiguration
from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator
from hybrid_sample_generator.generation.model_settings import (
    Choice,
    FloatRange,
    IntRange,
    SearchSpace,
)


class ConfigurationTests(unittest.TestCase):
    def test_default_configuration_and_model_selection(self):
        config = Configuration("defaults")
        config.validate()
        self.assertEqual(config.model.name, "cVAE_ConvNeXt_2D")
        config.model.parameters.z_channels = 123
        config.model.set_model("VAE_ResNet_3D")
        config.extraction.anomaly_size = (1, 16, 16, 16)
        config.validate()
        self.assertNotEqual(config.model.parameters.z_channels, 123)
        before = config.model.to_dict()
        with self.assertRaises(ValueError):
            config.model.set_model("unknown-model")
        self.assertEqual(config.model.to_dict(), before)

    def test_parameters_are_plain_state_without_extraction_side_effects(self):
        config = Configuration("parameters")
        parameters = config.model.parameters
        config.extraction.anomaly_size = (1, 32, 32)
        self.assertIs(config.model.parameters, parameters)
        self.assertFalse(hasattr(parameters, "in_channels"))
        self.assertFalse(hasattr(parameters, "num_anomaly_classes"))

    def test_fixed_parameters_and_search_space_round_trip(self):
        config = Configuration("search")
        config.model.parameters.recon_weight = 8.0
        config.model.search.z_channels = Choice((16, 32, 64))
        config.model.search.dropout = FloatRange(0.01, 0.2)
        loaded = Configuration.from_dict(config.to_dict())
        self.assertEqual(loaded.model.parameters.recon_weight, 8.0)
        self.assertIsInstance(loaded.model.search, SearchSpace)
        self.assertEqual(loaded.model.search, config.model.search)
        self.assertEqual(loaded.model.search.dropout, FloatRange(0.01, 0.2))

    def test_search_space_attribute_api_validates_and_removes_distributions(self):
        config = Configuration("search-api")
        search = config.model.search

        search.dropout = FloatRange(0.0, 0.2)
        self.assertEqual(search.dropout, FloatRange(0.0, 0.2))
        self.assertIn("dropout", search)

        del search.dropout
        self.assertNotIn("dropout", search)
        with self.assertRaises(AttributeError):
            _ = search.dropout

        with self.assertRaisesRegex(AttributeError, "Unknown"):
            search.unknown = IntRange(1, 2)
        with self.assertRaisesRegex(TypeError, "recon_loss"):
            search.recon_loss = FloatRange(0.0, 1.0)
        with self.assertRaises(AttributeError):
            config.model.search = {}

    def test_search_space_clear_makes_all_parameters_fixed(self):
        config = Configuration("fixed")
        config.model.set_model("VAE_ResNet_2D")
        self.assertGreater(len(config.model.search), 0)
        config.model.search.clear()
        self.assertEqual(len(config.model.search), 0)
        config.validate()

    def test_runtime_parameters_are_rejected_in_serialized_model_config(self):
        values = Configuration("invalid-runtime").to_dict()
        values["model"]["parameters"]["in_channels"] = 17
        with self.assertRaises(TypeError):
            Configuration.from_dict(values)

    def test_set_model_initializes_parameters_and_search(self):
        from hybrid_sample_generator.generation.registry import get_model_spec

        config = Configuration("assignment")
        config.model.set_model("VAE_ResNet_2D")
        spec = get_model_spec("VAE_ResNet_2D")
        self.assertEqual(config.model.parameters, spec.build_configuration())
        self.assertEqual(
            config.model.search,
            spec.build_search_space(config.model.parameters),
        )

    def test_all_registered_models_build_dimension_specific_defaults(self):
        from hybrid_sample_generator.generation.registry import MODEL_REGISTRY

        for name, spec in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                parameters = spec.build_configuration()
                search = spec.build_search_space(parameters)
                self.assertIsInstance(parameters, spec.config_cls)
                self.assertIsInstance(search, SearchSpace)
                search.validate()

    def test_default_model_searches_use_three_capacity_parameters(self):
        from hybrid_sample_generator.generation.registry import MODEL_REGISTRY

        self.assertEqual(Configuration("trials").training.num_trials, 10)
        for name, spec in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                parameters = spec.build_configuration()
                search = spec.build_search_space(parameters)
                self.assertEqual(
                    set(search.names()),
                    {"n_res_blocks", "z_channels", "bottleneck_dim"},
                )
                self.assertEqual(search.n_res_blocks, IntRange(4, 5))
                if spec.spatial_dims == 2:
                    self.assertEqual(search.z_channels, Choice((32, 64)))
                    self.assertEqual(search.bottleneck_dim, Choice((64, 128)))
                else:
                    self.assertEqual(search.z_channels, Choice((64, 128)))
                    self.assertEqual(search.bottleneck_dim, Choice((128, 256)))

    def test_schema_eight_configuration_is_not_supported(self):
        values = Configuration("old-schema").to_dict()
        values["schema_version"] = 8
        with self.assertRaisesRegex(ValueError, "schema"):
            Configuration.from_dict(values)

    def test_search_distribution_values_are_strictly_typed(self):
        with self.assertRaises(TypeError):
            IntRange(1.0, 2)
        with self.assertRaises(TypeError):
            IntRange(1, 2, step=True)
        with self.assertRaises(ValueError):
            IntRange(0, 2, log=True)
        with self.assertRaises(TypeError):
            FloatRange(False, 1.0)

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
