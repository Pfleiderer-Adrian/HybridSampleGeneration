"""MVTec split and DRAEM integration tests."""
import json
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.data.datasets import MixedTrainingDataset
from examples.mvtec_ad2.synthetic_data_generation.hybrid.pairs import HybridPairs
from examples.mvtec_ad2.downstream.models.losses import training_loss
from examples.mvtec_ad2.downstream.models.draem import DRAEM
from examples.mvtec_ad2.downstream.evaluation import binary_metrics
from examples.mvtec_ad2.downstream.runner import evaluate_downstream, train_downstream
from examples.mvtec_ad2.downstream.data.sampling import source_plan
from examples.mvtec_ad2.data.splits import (
    SplitConfiguration,
    create_manifest,
    manifest_samples,
    verify_repository,
)


from ..fixtures import create_images, generator_config, seed_hybrid


class DownstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        config = DownstreamConfiguration()
        config.data.image_size = (64, 64)
        config.data.patch_size = (64, 64)
        config.data.samples_per_epoch = 4
        config.data.hybrid_fraction = 1
        config.training.epochs = 1
        config.training.batch_size = 2
        config.training.device = "cpu"
        config.training.reconstruction_width = 2
        config.training.segmentation_width = 2
        return config

    def test_configuration_and_source_quotas(self):
        config = self.config()
        config.data.samples_per_epoch = 100
        for fraction, expected in ((0, 0), (.5, 25), (1, 50)):
            config.data.hybrid_fraction = fraction
            plan = source_plan(config.data, 42, 0)
            self.assertEqual(Counter(plan)["hybrid"], expected)
            self.assertEqual(plan, source_plan(config.data, 42, 0))
        restored = DownstreamConfiguration.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored.data.hybrid_fraction, 1)

    def test_split_is_deterministic_disjoint_and_persistable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = create_manifest(root / "can", SplitConfiguration())
            self.assertEqual(manifest, create_manifest(root / "can", SplitConfiguration()))
            sets = [{item["image_path"] for item in manifest["partitions"][key]} for key in ("train", "validation", "test")]
            self.assertEqual(sum(map(len, sets)), 16)
            self.assertEqual(len(set.union(*sets)), 16)
            self.assertTrue(all(sets))

    def test_training_and_evaluation_use_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = create_manifest(root / "can", SplitConfiguration())
            config = generator_config(root / "study")
            repo, store = seed_hybrid(config, manifest)
            downstream = self.config()
            pairs = HybridPairs(repo, store, manifest, downstream.data)
            self.assertTrue(pairs[0][2].any())
            output = train_downstream(config, manifest, downstream)
            metrics = evaluate_downstream(output, manifest)
            self.assertIn("validation", metrics)
            self.assertIn("test", metrics)
            for split in ("validation", "test"):
                value = metrics[split]["pixel_aupro_0.05"]
                self.assertIsNotNone(value)
                self.assertGreaterEqual(value, 0.)
                self.assertLessEqual(value, 1.)
            self.assertTrue((output / "checkpoints" / "best.pt").is_file())
            bad = {**manifest, "partitions": {**manifest["partitions"], "train": []}}
            with self.assertRaisesRegex(ValueError, "held-out"):
                verify_repository(repo, bad)
            with self.assertRaisesRegex(ValueError, "modified"):
                evaluate_downstream(output, bad)

    def test_checkpoint_selection_maximizes_validation_aupro(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = create_manifest(root / "can", SplitConfiguration())
            generator = generator_config(root / "study")
            seed_hybrid(generator, manifest)
            config = self.config()
            config.training.epochs = 3
            # Loss improves every epoch, while PRO peaks at epoch 2.
            validation = [{"segmentation_loss": loss, "pixel_aupro_0.05": pro}
                          for loss, pro in ((.3, .2), (.2, .8), (.1, .8))]
            with patch("examples.mvtec_ad2.downstream.training.evaluate", side_effect=validation):
                output = train_downstream(generator, manifest, config)
            best = torch.load(output / "checkpoints/best.pt", weights_only=True)
            last = torch.load(output / "checkpoints/last.pt", weights_only=True)
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(last["epoch"], 3)
            self.assertEqual(best["selection_metric"], "pixel_aupro_0.05")
            self.assertEqual(best["selection_mode"], "max")
            self.assertEqual(best["selection_score"], .8)
            self.assertNotIn("selection_loss", best)
            with patch("examples.mvtec_ad2.downstream.training.evaluate",
                       return_value={"segmentation_loss": .1, "pixel_aupro_0.05": None}):
                with self.assertRaisesRegex(ValueError, "requires validation defect regions"):
                    train_downstream(generator, manifest, config)

    def test_validation_interval_and_final_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = create_manifest(root / "can", SplitConfiguration())
            generator = generator_config(root / "study")
            seed_hybrid(generator, manifest)
            config = self.config()
            config.training.epochs = 5
            config.training.validation_interval = 2
            values = [{"segmentation_loss": .1, "pixel_aupro_0.05": pro}
                      for pro in (.3, .8, .4)]
            with patch("examples.mvtec_ad2.downstream.training.evaluate", side_effect=values) as evaluate:
                output = train_downstream(generator, manifest, config)
            self.assertEqual(evaluate.call_count, 3)
            best = torch.load(output / "checkpoints/best.pt", weights_only=True)
            last = torch.load(output / "checkpoints/last.pt", weights_only=True)
            self.assertEqual(best["epoch"], 4)
            self.assertEqual(last["epoch"], 5)
            with (output / "training_history.csv").open() as file:
                history = list(csv.DictReader(file))
            self.assertEqual([row["epoch"] for row in history if row["selection_score"]], ["2", "4", "5"])
            for invalid in (0, -1, True, 1.5):
                config.training.validation_interval = invalid
                with self.assertRaisesRegex(ValueError, "validation_interval"):
                    config.validate()

    def test_perlin_source_is_deterministic_and_has_anomaly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            textures = root / "textures"
            textures.mkdir()
            Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(textures / "texture.png")
            config = self.config()
            config.data.hybrid_fraction = 0
            config.data.texture_root = str(textures)
            manifest = create_manifest(root / "can", SplitConfiguration())
            dataset = MixedTrainingDataset(manifest_samples(manifest, "train", True), None, config)
            index = dataset.plan.index("draem")
            item = dataset[index]
            self.assertTrue(torch.equal(item["image"], dataset[index]["image"]))
            self.assertTrue(item["mask"].any())

    def test_model_losses_and_metrics(self):
        model = DRAEM(2, 2)
        image = torch.rand(2, 3, 64, 64)
        reconstruction, logits = model(image)
        loss = training_loss(reconstruction, logits, image, torch.zeros(2, 1, 64, 64))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(binary_metrics([0, 1], [.1, .9]), {"auroc": 1., "ap": 1.})
        self.assertEqual(binary_metrics([0, 1], [.5, .5]), {"auroc": .5, "ap": .5})
