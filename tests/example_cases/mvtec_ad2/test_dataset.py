"""Tests for the compact MVTec dataset adapter and executable recipes."""
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.mvtec_ad2.dataset import (
    MVTecAD2Dataloader,
    MVTecAD2Sample,
    discover_samples,
)


class MVTecDatasetTests(unittest.TestCase):
    def test_discovery_finds_controls_and_annotated_anomalies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good = root / "train" / "good"
            bad = root / "test_public" / "bad"
            masks = root / "test_public" / "ground_truth" / "bad"
            for path in (good, bad, masks):
                path.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(good / "good.png")
            Image.new("RGB", (8, 8)).save(bad / "bad.png")
            Image.new("L", (8, 8), 255).save(masks / "bad.png")
            samples = discover_samples(root)
            self.assertEqual({sample.label for sample in samples}, {"good", "bad"})
            self.assertEqual(sum(sample.mask_path is not None for sample in samples), 1)

    def test_dataloader_exposes_tuple_and_typed_views(self):
        sample = MVTecAD2Sample(Path("sample.png"), None, "sample", "train", "good")
        loader = MVTecAD2Dataloader([sample])
        with patch("examples.mvtec_ad2.dataset.load_image_array", return_value=np.ones((5, 7), dtype=np.uint8)):
            image, segmentation, sample_id = next(iter(loader))
            typed = next(loader.iter_input_samples())
        self.assertEqual(image.shape, (1, 5, 7))
        self.assertEqual(segmentation.shape, image.shape)
        self.assertEqual(sample_id, "sample")
        self.assertEqual(typed.metadata, {"split": "train", "label": "good"})

    def test_category_recipes_build_independent_valid_configurations(self):
        categories = {
            "can": (3, 64, 64),
            "fabric": (3, 64, 64),
            "fruit_jelly": (3, 64, 64),
            "rice": (3, 128, 128),
            "sheet_metal": (1, 64, 64),
            "vial": (1, 64, 64),
            "wallplugs": (1, 64, 64),
            "walnuts": (3, 64, 64),
        }
        for category, anomaly_size in categories.items():
            with self.subTest(category=category):
                recipe = importlib.import_module(f"examples.mvtec_ad2.categories.{category}")
                first = recipe.create_configuration()
                second = recipe.create_configuration()
                first.validate()
                self.assertEqual(first.extraction.anomaly_size, anomaly_size)
                self.assertEqual(first.generation.feedback.max_attempts, 1000)
                self.assertEqual(first.matching.batch_size, 64)
                self.assertEqual(first.training.num_trials, 10)
                self.assertEqual(
                    set(first.model.search.names()),
                    {"n_res_blocks", "n_levels", "z_channels", "bottleneck_dim", "dropout"},
                )
                first.generation.variants_per_real_anomaly = 99
                self.assertEqual(second.generation.variants_per_real_anomaly, 3)
