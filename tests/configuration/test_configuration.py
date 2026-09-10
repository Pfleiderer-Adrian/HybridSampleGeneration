import json
import tempfile
import unittest
from pathlib import Path

from hybrid_sample_generator.configuration.root import Configuration, load_config_file
from hybrid_sample_generator.configuration.augmentation import MaskTransformConfiguration
from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


class ConfigurationTests(unittest.TestCase):
    def test_configuration_round_trip_uses_sectioned_plain_json(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "config-test",
                "VAE_ResNet_2D",
                (3, 32, 32),
                save_path=root,
            )
            config.matching.routine = "local"
            config.matching.hybrids_per_original = 3
            config.matching.anomalies_per_hybrid = 2
            config.generation.variants_per_real_anomaly = 5
            config.training.batch_size = 8

            path = Path(config.save_config_file())
            serialized = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_config_file(path)

            self.assertEqual(serialized["schema_version"], Configuration.SCHEMA_VERSION)
            self.assertNotIn("uses_masks", serialized["model"])
            self.assertFalse(hasattr(loaded.model, "uses_masks"))
            self.assertEqual(serialized["matching"]["anomalies_per_hybrid"], 2)
            self.assertEqual(serialized["matching"]["hybrids_per_original"], 3)
            self.assertEqual(serialized["generation"]["variants_per_real_anomaly"], 5)
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

    def test_matching_configuration_rejects_invalid_weights(self):
        config = MatchingConfiguration(intensity_weight=0, gradient_weight=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_matching_configuration_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            MatchingConfiguration(hybrids_per_original=0).validate()


if __name__ == "__main__":
    unittest.main()
