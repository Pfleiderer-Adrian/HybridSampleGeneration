import json
import tempfile
import unittest
from pathlib import Path

from synthesizer.Configuration import Configuration, load_config_file
from synthesizer.configuration.matching import MatchingConfiguration


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
            self.assertEqual(serialized["matching"]["anomalies_per_hybrid"], 2)
            self.assertEqual(serialized["matching"]["hybrids_per_original"], 3)
            self.assertEqual(serialized["generation"]["variants_per_real_anomaly"], 5)
            self.assertEqual(loaded.to_dict(), config.to_dict())

    def test_matching_configuration_rejects_invalid_weights(self):
        config = MatchingConfiguration(intensity_weight=0, gradient_weight=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_matching_configuration_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            MatchingConfiguration(hybrids_per_original=0).validate()


if __name__ == "__main__":
    unittest.main()
