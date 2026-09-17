"""MVTec split and DRAEM integration tests."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from examples.mvtec_ad2.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.datasets import HybridPairs, MixedTrainingDataset
from examples.mvtec_ad2.downstream.draem.losses import training_loss
from examples.mvtec_ad2.downstream.draem.model import DRAEM
from examples.mvtec_ad2.downstream.evaluation import binary_metrics
from examples.mvtec_ad2.downstream.runner import evaluate_downstream, train_downstream
from examples.mvtec_ad2.downstream.sampling import source_plan
from examples.mvtec_ad2.splits import (
    SplitConfiguration,
    create_manifest,
    manifest_samples,
    verify_repository,
)
from hybrid_sample_generator import Configuration, HybridDataGenerator
from hybrid_sample_generator.domain.records import (
    HybridSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)


def create_images(root):
    for label in ("good", "bad"):
        folder = root / "can" / "test_public" / label
        folder.mkdir(parents=True)
        for index in range(8):
            image = np.full((64, 64, 3), 80 + index, dtype=np.uint8)
            if label == "bad":
                image[20:40, 20:40] = 220
            Image.fromarray(image).save(folder / f"{index}.png")
            if label == "bad":
                masks = root / "can" / "test_public" / "ground_truth" / "bad"
                masks.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[20:40, 20:40] = 255
                Image.fromarray(mask).save(masks / f"{index}.png")

def generator_config(folder):
    config = Configuration("mvtec-test", study_folder=folder)
    config.extraction.anomaly_size = (3, 64, 64)
    return config

def seed_hybrid(config, manifest):
    generator = HybridDataGenerator(config)
    generator.ingest_dataset(MVTecAD2Dataloader(manifest_samples(manifest, "train")))
    repo, store = generator.repository, generator.artifact_store
    healthy = repo.list_original_samples(has_anomaly=False)[0]
    donor = repo.list_original_samples(has_anomaly=True)[0]
    target = store.load_array(healthy.image_path)
    image = target.copy()
    image[:, 20:40, 20:40] = 220
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 20:40, 20:40] = 1
    image_path = store.save_entity_array("hybrid_samples", "hybrid", "image", image)
    mask_path = store.save_entity_array("hybrid_samples", "hybrid", "segmentation", mask)
    real = RealAnomaly("real", donor.id, 0, donor.image_path, donor.segmentation_path,
                       donor.image_path, donor.segmentation_path, 2, None, .5, .5)
    repo.upsert_real_anomaly(real)
    repo.upsert_synthetic_anomaly(SyntheticAnomaly("synth", real.id, 0, image_path, mask_path, 42))
    repo.upsert_hybrid_sample(HybridSample("hybrid", healthy.id, 0, image_path, mask_path, "generated"))
    repo.upsert_placement(Placement("placement", "hybrid", "synth", 0, 2, None, .5, .5))
    return repo, store

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
            self.assertTrue((output / "checkpoints" / "best.pt").is_file())
            bad = {**manifest, "partitions": {**manifest["partitions"], "train": []}}
            with self.assertRaisesRegex(ValueError, "held-out"):
                verify_repository(repo, bad)
            with self.assertRaisesRegex(ValueError, "modified"):
                evaluate_downstream(output, bad)

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
