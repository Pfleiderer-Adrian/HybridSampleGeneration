"""Native-resolution crops, full-coverage inference and unchanged pixel geometry."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from examples.mvtec_ad2.dataset import MVTecAD2Sample
from examples.mvtec_ad2.downstream.configuration import (
    DownstreamConfiguration,
    TrainingConfiguration,
)
from examples.mvtec_ad2.downstream.datasets import (
    HybridPairs,
    MixedTrainingDataset,
    RealImageDataset,
)
from examples.mvtec_ad2.downstream.evaluation import evaluate, evaluation_loader
from examples.mvtec_ad2.downstream.patches import (
    crop_training_patch,
    tile_starts,
    tiled_logits,
)
from examples.mvtec_ad2.downstream.training import train
from examples.mvtec_ad2.splits import (
    SplitConfiguration,
    create_manifest,
    manifest_samples,
)

from .test_downstream import create_images, generator_config, seed_hybrid


class PixelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, image):
        self.shapes.append(tuple(image.shape))
        return image, torch.cat((torch.zeros_like(image[:, :1]), image[:, :1]), dim=1)


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_configuration_validation_and_snapshot_roundtrip(self):
        config = DownstreamConfiguration()
        self.assertEqual(config.data.mode, "patch")
        self.assertEqual(config.data.patch_size, (512, 512))
        restored = DownstreamConfiguration.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(DownstreamConfiguration.from_dict({"data": {"image_size": [64, 64]}}).data.mode, "patch")
        for name, value in (("mode", "unknown"), ("patch_size", (65, 64)),
                            ("patch_overlap", 1), ("patch_overlap", -0.1)):
            with self.subTest(name=name):
                bad = DownstreamConfiguration()
                setattr(bad.data, name, value)
                with self.assertRaises(ValueError):
                    bad.validate()

    def test_single_pixel_defects_survive_aligned_cropping_including_edges(self):
        image = np.arange(97*151, dtype=np.float32).reshape(1, 97, 151)
        for y, x in ((0, 0), (96, 150), (1, 1), (45, 73)):
            mask = np.zeros_like(image)
            mask[0, y, x] = 1
            for seed in range(4):
                result = crop_training_patch(image, image+1, mask, (64, 96),
                                             np.random.default_rng(seed), anomalous=True)
                crop, target, cropped_mask = result
                self.assertEqual(crop.shape, (1, 64, 96))
                self.assertEqual(cropped_mask.sum(), 1)
                np.testing.assert_array_equal(target, crop+1)
                self.assertEqual(crop[cropped_mask.astype(bool)][0], image[0, y, x])
                again = crop_training_patch(image, image+1, mask, (64, 96),
                                            np.random.default_rng(seed), anomalous=True)
                np.testing.assert_array_equal(crop, again[0])

    def test_small_images_are_padded_not_resized_and_empty_masks_rejected(self):
        image = np.arange(20*30, dtype=np.float32).reshape(1, 20, 30)
        mask = np.zeros_like(image)
        mask[0, -1, -1] = 1
        crop, target, cropped_mask = crop_training_patch(image, image, mask, (64, 64),
                                                        np.random.default_rng(0), anomalous=True)
        np.testing.assert_array_equal(crop[:, :20, :30], image)
        np.testing.assert_array_equal(crop, target)
        self.assertEqual(cropped_mask.sum(), 1)
        self.assertFalse(cropped_mask[:, 20:, :].any())
        with self.assertRaisesRegex(ValueError, "empty mask"):
            crop_training_patch(image, image, mask*0, (64, 64), np.random.default_rng(0), anomalous=True)

    def test_tiling_covers_all_pixels_without_resizing_or_large_gpu_batches(self):
        for shape in ((3, 113, 173), (3, 21, 39), (3, 64, 64)):
            image = torch.rand(shape)
            model = PixelModel().eval()
            logits = tiled_logits(model, image, "cpu", (64, 96), .5, 2)
            self.assertEqual(tuple(logits.shape), (2, *shape[-2:]))
            torch.testing.assert_close(logits[1], image[0])
            self.assertTrue(all(s[0] <= 2 and s[-2:] == (64, 96) for s in model.shapes))
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(tile_starts(173, 96, .5)[-1], 77)

    def test_evaluation_preserves_variable_original_shapes_and_all_mask_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            samples = []
            for i, shape in enumerate(((81, 107), (35, 42))):
                image = np.full(shape, 127, dtype=np.uint8)
                mask = np.zeros(shape, dtype=np.uint8)
                mask[-1, -1] = 255
                Image.fromarray(image).save(folder / f"{i}.png")
                Image.fromarray(mask).save(folder / f"mask{i}.png")
                samples.append(MVTecAD2Sample(folder/f"{i}.png", folder/f"mask{i}.png", str(i), "validation", "bad"))
            config = DownstreamConfiguration()
            config.data.patch_size = (64, 64)
            dataset = RealImageDataset(samples, config.data)
            self.assertEqual(tuple(dataset[0]["image"].shape), (3, 81, 107))
            self.assertEqual(dataset[0]["mask"].sum(), 1)
            loader = evaluation_loader(dataset, TrainingConfiguration(batch_size=2))
            self.assertEqual(loader.batch_size, 1)
            result = evaluate(PixelModel(), loader, "cpu", folder/"predictions", patch_batch_size=2)
            self.assertEqual(result["count"], 2)
            for i, shape in enumerate(((81, 107), (35, 42))):
                self.assertEqual(np.load(folder/"predictions"/f"{i:06d}.npy").shape, shape)
            config.data.mode = "image"
            config.data.image_size = (64, 64)
            self.assertEqual(tuple(dataset[0]["image"].shape), (3, 64, 64))
            self.assertEqual(evaluate(PixelModel(), evaluation_loader(dataset, TrainingConfiguration()), "cpu")["count"], 2)

    def test_dataset_samples_native_hybrid_patches_and_checks_stored_masks_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = create_manifest(root / "can", SplitConfiguration())
            config = generator_config(root / "study")
            repo, store = seed_hybrid(config, manifest)
            downstream = DownstreamConfiguration()
            data = downstream.data
            data.patch_size = (64, 64)
            data.hybrid_fraction = 1
            data.samples_per_epoch = 4
            tiny = np.zeros((1, 64, 64), dtype=np.float32)
            tiny[0, 1, 1] = 1
            store.save_entity_array("hybrid_samples", "hybrid", "segmentation", tiny)
            pairs = HybridPairs(repo, store, manifest, data)
            dataset = MixedTrainingDataset(manifest_samples(manifest, "train", True), pairs, downstream)
            index = dataset.plan.index("hybrid")
            with patch("examples.mvtec_ad2.downstream.transforms.F.interpolate", side_effect=AssertionError("No resize")):
                item = dataset[index]
            self.assertEqual(item["mask"].sum(), 1)
            self.assertTrue(torch.equal(item["image"], dataset[index]["image"]))
            store.save_entity_array("hybrid_samples", "hybrid", "segmentation", tiny*0)
            with self.assertRaisesRegex(ValueError, "empty stored"):
                HybridPairs(repo, store, manifest, data)

    def test_training_and_full_image_validation_with_larger_native_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.random.default_rng(3).integers(0, 256, (96, 128, 3), dtype=np.uint8)
            Image.fromarray(image).save(root/"healthy.png")
            anomalous = image.copy()
            anomalous[70:80, 90:100] = 255
            Image.fromarray(anomalous).save(root/"bad.png")
            mask = np.zeros((96, 128), dtype=np.uint8)
            mask[70:80, 90:100] = 255
            Image.fromarray(mask).save(root/"mask.png")
            good = MVTecAD2Sample(root/"healthy.png", None, "good", "train", "good")
            bad = MVTecAD2Sample(root/"bad.png", root/"mask.png", "bad", "validation", "bad")
            config = DownstreamConfiguration()
            config.data.patch_size = (64, 64)
            config.data.hybrid_fraction = 0
            config.data.samples_per_epoch = 4
            textures = root/"textures"
            textures.mkdir()
            Image.fromarray(image).save(textures/"texture.png")
            config.data.texture_root = str(textures)
            config.training.epochs = 1
            config.training.device = "cpu"
            config.training.reconstruction_width = config.training.segmentation_width = 2
            dataset = MixedTrainingDataset([good], None, config)
            validation = RealImageDataset([good, bad], config.data)
            self.assertEqual(tuple(dataset[0]["image"].shape), (3, 64, 64))
            self.assertEqual(tuple(validation[0]["image"].shape), (3, 96, 128))
            train(dataset, validation, config, root)
            checkpoint = torch.load(root/"checkpoints"/"best.pt", weights_only=True)
            self.assertEqual(checkpoint["validation"]["count"], 2)
            self.assertEqual(checkpoint["validation"]["spatial_mode"], "patch")
