"""End-to-end wiring of the MVTec AD 2 entry point, presets and grouped splits."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.mvtec_ad2 import run
from examples.mvtec_ad2.data.splits import SplitConfiguration, load_or_create_manifest
from examples.mvtec_ad2.downstream.presets import MIXTURES, create_downstream_configuration
from examples.mvtec_ad2.synthetic_data_generation.hybrid.pipeline import CATEGORIES, category_configuration

from .fixtures import create_images


class PresetTests(unittest.TestCase):
    def test_every_mixture_is_valid_and_draws_1000_samples(self):
        for mixture, quotas in MIXTURES.items():
            with self.subTest(mixture=mixture):
                manifest = "pairs.json" if "anomalydiffusion" in quotas else None
                config = create_downstream_configuration(mixture, anomalydiffusion_manifest=manifest)
                self.assertEqual(sum(config.data.source_quotas.values()), 1000)
                self.assertEqual(config.model, "dinov3_paired")
                self.assertEqual(config.training.selection_metric, "pixel_aupro_0.05")

    def test_anomalydiffusion_mixture_requires_manifest_and_unknown_names_fail(self):
        with self.assertRaisesRegex(ValueError, "AnomalyDiffusion"):
            create_downstream_configuration("real_anomalydiffusion")
        with self.assertRaisesRegex(ValueError, "Unknown mixture"):
            create_downstream_configuration("unknown")

    def test_every_category_configuration_validates(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                category_configuration(category).validate()
        with self.assertRaisesRegex(ValueError, "Unknown category"):
            category_configuration("unknown")


class RunTests(unittest.TestCase):
    def run_main(self, root, *arguments):
        with patch.object(run, "category_root", return_value=root / "can"), \
             patch.object(run, "study_folder", return_value=root / "study"), \
             patch.object(run, "generate_hybrids") as generate, \
             patch.object(run, "train_downstream", return_value=root / "run") as train, \
             patch.object(run, "evaluate_downstream", return_value={"validation": {}}) as evaluate, \
             patch("builtins.print"):
            run.main(["--category", "can", *arguments])
        return generate, train, evaluate

    def test_generates_hybrids_only_for_hybrid_mixtures_on_a_grouped_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_images(root)
            generate, train, evaluate = self.run_main(root)
            self.assertEqual(generate.call_count, 1)
            config, manifest, downstream = train.call_args.args
            self.assertIn("grouping", manifest)
            self.assertEqual(downstream.data.source_quotas, MIXTURES["real_draem_hybrid"])
            evaluate.assert_called_once_with(root / "run", manifest)
            generate, *_ = self.run_main(root, "--mixture", "real_draem")
            generate.assert_not_called()
            generate, *_ = self.run_main(root, "--skip-generation")
            generate.assert_not_called()
            saved = json.loads((root / "study/split_manifest.json").read_text())
            with self.assertRaisesRegex(ValueError, "grouping"):
                load_or_create_manifest(root / "study/split_manifest.json", root / "can",
                                        SplitConfiguration(**saved["split"]))
