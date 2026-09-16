"""MVTec CPU integration tests using actual repository records and images."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader
from examples.mvtec_ad2.downstream.datasets import HybridPairs, MixedTrainingDataset
from examples.mvtec_ad2.downstream.draem.model import DRAEM
from examples.mvtec_ad2.downstream.draem.losses import training_loss
from examples.mvtec_ad2.downstream.evaluation import binary_metrics
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream
from examples.mvtec_ad2.studies import prepare_studies
from examples.mvtec_ad2.pipeline import run_new_experiment, run_existing_studies, run_study
from examples.mvtec_ad2.configuration import Experiment
from examples.mvtec_ad2.studies import open_study
from examples.mvtec_ad2.configuration import SplitConfiguration
from examples.mvtec_ad2.workflow import preflight
from unittest.mock import patch
from examples.mvtec_ad2.downstream.sampling import source_plan
from examples.mvtec_ad2.splits import create_manifest, manifest_samples, verify_repository
from hybrid_sample_generator.domain.records import RealAnomaly, SyntheticAnomaly, HybridSample, Placement
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator


def create_images(root):
    for label in ("good", "bad"):
        folder = root / "can" / "test_public" / label
        folder.mkdir(parents=True)
        for i in range(8):
            image = np.full((64, 64, 3), 80+i, dtype=np.uint8)
            if label == "bad":
                image[20:40, 20:40] = 220
            Image.fromarray(image).save(folder / f"{i}.png")
            if label == "bad":
                mask_folder = root / "can" / "test_public" / "ground_truth" / "bad"
                mask_folder.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[20:40, 20:40] = 255
                Image.fromarray(mask).save(mask_folder / f"{i}.png")
    (root / "can" / "train" / "good").mkdir(parents=True)


def seed_hybrid(usecase):
    generator = HybridDataGenerator(usecase.config)
    generator.ingest_dataset(MVTecAD2Dataloader(manifest_samples(usecase.split_manifest, "train")))
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
            self.assertEqual(Counter(plan)["normal"], 50)
            self.assertEqual(plan, source_plan(config.data, 42, 0))
        restored = DownstreamConfiguration.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored.data.hybrid_fraction, 1)
        config.data.hybrid_fraction = 1.1
        with self.assertRaises(ValueError):
            config.validate()

    def test_example_configuration_roundtrip_leaves_library_unchanged(self):
        from examples.mvtec_ad2.configuration import Configuration
        from hybrid_sample_generator.configuration.root import Configuration as LibraryConfiguration

        base = LibraryConfiguration("base")
        base.extraction.anomaly_size = (1, 8, 8)
        base.model.set_model("VAE_ResNet_2D")
        self.assertFalse(hasattr(base, "downstream"))
        config = Configuration.from_dict(base.to_dict())
        config.downstream.training.epochs = 17
        loaded = Configuration.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(loaded.downstream.training.epochs, 17)
        loaded.downstream.data.hybrid_fraction = 2
        with self.assertRaises(ValueError):
            loaded.validate()

    def test_optional_test_split_uses_every_sample_without_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            config = SplitConfiguration()
            for enabled in (True, False):
                config.test_enabled = enabled
                manifest = create_manifest(root / "can", config)
                self.assertEqual(manifest, create_manifest(root / "can", config))
                parts = manifest["partitions"]
                sets = [{item["image_path"] for item in parts[key]} for key in ("train", "validation", "test")]
                self.assertEqual(sum(map(len, sets)), 16)
                self.assertEqual(len(set.union(*sets)), 16)
                self.assertEqual(bool(sets[2]), enabled)
                self.assertTrue(sets[0] and sets[1])

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
            i = dataset.plan.index("draem")
            item = dataset[i]
            self.assertTrue(torch.equal(item["image"], dataset[i]["image"]))
            self.assertTrue(item["mask"].any())
            outside = ~item["mask"].bool().expand_as(item["image"])
            self.assertTrue(torch.equal(item["image"][outside], item["target"][outside]))

    def test_hybrid_pair_provenance_and_training_both_split_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            config = self.config()
            for enabled in (True, False):
                usecase = prepare_studies(root, ("can",), save_path=root / str(enabled),
                    splits=SplitConfiguration(test_enabled=enabled))[0]
                usecase.config.downstream = config
                manifest = usecase.split_manifest
                repo, store = seed_hybrid(usecase)
                pairs = HybridPairs(repo, store, manifest, config.data)
                image, target, mask = pairs[0]
                self.assertEqual(image.shape, (3, 64, 64))
                self.assertTrue(torch.equal(image[:, :10], target[:, :10]))
                self.assertTrue(mask.any())
                output = train_downstream(usecase)
                evaluate_downstream(usecase, output.name)
                metrics = json.loads((output / "metrics.json").read_text())
                self.assertEqual("test" in metrics, enabled)
                self.assertTrue((output / "checkpoints" / "best.pt").exists())
                self.assertFalse((output / "predictions" / "test").exists() and not enabled)
                self.assertTrue((output / "predictions" / "validation" / "000000.png").exists())
                bad = dict(manifest)
                bad["partitions"] = {**manifest["partitions"], "train": []}
                with self.assertRaisesRegex(ValueError, "held-out"):
                    verify_repository(repo, bad)

    def test_mixed_training_uses_all_three_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            config = self.config()
            config.data.samples_per_epoch = 8
            config.data.hybrid_fraction = .5
            config.data.texture_root = str(root / "can" / "test_public" / "good")
            usecase = prepare_studies(root, ("can",), save_path=root / "results")[0]
            usecase.config.downstream = config
            manifest = usecase.split_manifest
            repo, store = seed_hybrid(usecase)
            dataset = MixedTrainingDataset(manifest_samples(manifest, "train", True), HybridPairs(repo, store, manifest, config.data), config)
            counts = Counter(dataset[i]["source"] for i in range(len(dataset)))
            self.assertEqual(counts, {"normal": 4, "draem": 2, "hybrid": 2})
            result = run_study(usecase,
                steps=("train_downstream", "evaluate_downstream"))
            output = result.downstream_run_folder
            self.assertEqual(result.study_folder, Path(usecase.config.study.folder))
            import csv
            with (output / "training_history.csv").open() as file:
                history = next(csv.DictReader(file))
            self.assertEqual([history[k] for k in ("normal", "draem", "hybrid")], ["4", "2", "2"])

    def test_reuses_manifest_without_discovery_and_rejects_changed_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            first = prepare_studies(root, ("can",), save_path=root/"results", splits=SplitConfiguration(test_enabled=False))[0]
            with patch("examples.mvtec_ad2.splits.create_manifest", side_effect=AssertionError("Must reuse split")):
                second = prepare_studies(root, ("can",), save_path=root/"results")[0]
            self.assertEqual(first.split_manifest, second.split_manifest)
            self.assertFalse(Path(first.config.study.paths.configuration_file).exists())
            with self.assertRaisesRegex(ValueError, "differ"):
                prepare_studies(root, ("can",), save_path=root/"results", splits=SplitConfiguration(test_enabled=True))

    def test_preflight_fails_before_generator_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            with patch("examples.mvtec_ad2.pipeline.HybridDataGenerator", side_effect=AssertionError("Must fail early")):
                with self.assertRaisesRegex(ValueError, "No materialized"):
                    run_new_experiment(Experiment(root, root/"results", ("can",)), steps=("train_downstream",))

    def test_config_is_saved_at_execution_not_on_open(self):
        from examples.mvtec_ad2.configuration import load_config_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            first = prepare_studies(root, ("can",), save_path=root/"results")[0]
            first.config.downstream.training.epochs = 7
            path = Path(first.config.study.paths.configuration_file)
            self.assertFalse(path.exists())
            run_study(first, steps=("ingest",))
            before = path.read_bytes()
            loaded = load_config_file(path)
            self.assertEqual(loaded.downstream.training.epochs, 7)
            reopened = open_study(path.parent)
            self.assertEqual(first.split_manifest, reopened.split_manifest)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare_studies(root, ("can",), save_path=root/"results")

    def test_evaluation_only_uses_selected_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            config = self.config()
            usecase = prepare_studies(root, ("can",), save_path=root/"results", splits=SplitConfiguration(test_enabled=False))[0]
            usecase.config.downstream = config
            seed_hybrid(usecase)
            with patch("examples.mvtec_ad2.downstream.training.torch.load", wraps=torch.load) as load_weights:
                output = train_downstream(usecase)
                load_weights.assert_not_called()
            before = Path(usecase.config.study.paths.configuration_file).read_bytes()
            with patch("examples.mvtec_ad2.downstream.runner.train", side_effect=AssertionError("No training")), patch("examples.mvtec_ad2.pipeline.HybridDataGenerator", side_effect=AssertionError("No generator")), patch("examples.mvtec_ad2.workflow.StudyRepository", side_effect=AssertionError("No repository")), patch("examples.mvtec_ad2.studies.create_configuration", side_effect=AssertionError("No current presets")):
                run_existing_studies(study_folders=[usecase.config.study.folder],
                    steps=("evaluate_downstream",), downstream_run_id=output.name)
            metrics = json.loads((output/"metrics.json").read_text())
            self.assertNotIn("test", metrics)
            self.assertEqual(Path(usecase.config.study.paths.configuration_file).read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "downstream_run_id"):
                preflight(usecase, ("evaluate_downstream",))

    def test_perlin_only_training_needs_no_repository_and_keeps_texture_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            case = prepare_studies(root, "can", save_path=root / "studies")[0]
            case.config.downstream = self.config()
            case.config.downstream.data.hybrid_fraction = 0
            texture = root / "textures"
            texture.mkdir()
            Image.new("RGB", (64, 64), "red").save(texture / "texture.png")
            def resolve_texture(folder, data):
                data.texture_root = str(texture)
                return texture
            with patch("examples.mvtec_ad2.downstream.runner.prepare_textures", side_effect=resolve_texture), patch("examples.mvtec_ad2.pipeline.HybridDataGenerator", side_effect=AssertionError("No generator")), patch("examples.mvtec_ad2.downstream.runner.StudyRepository", side_effect=AssertionError("No repository")):
                result = run_study(case, steps=("train_downstream", "evaluate_downstream"))
            self.assertIsNone(case.config.downstream.data.texture_root)
            snapshot = json.loads((result.downstream_run_folder / "configuration.json").read_text())
            self.assertEqual(snapshot["data"]["texture_root"], str(texture))
            self.assertFalse(Path(case.config.study.paths.artifact_database).exists())

    def test_model_losses_backpropagate_into_both_networks(self):
        model = DRAEM(2, 2)
        image = torch.rand(2, 3, 64, 64)
        reconstruction, logits = model(image)
        self.assertEqual(logits.shape, (2, 2, 64, 64))
        loss = training_loss(reconstruction, logits, image, torch.zeros(2, 1, 64, 64))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(next(model.reconstructor.parameters()).grad)
        self.assertIsNotNone(next(model.segmentor.parameters()).grad)

    def test_metric_ties_and_perfect_predictions(self):
        self.assertEqual(binary_metrics([0, 1], [.1, .9]), {"auroc": 1., "ap": 1.})
        self.assertEqual(binary_metrics([0, 1], [.5, .5]), {"auroc": .5, "ap": .5})
        self.assertIsNone(binary_metrics([0, 0], [.2, .8])["auroc"])
