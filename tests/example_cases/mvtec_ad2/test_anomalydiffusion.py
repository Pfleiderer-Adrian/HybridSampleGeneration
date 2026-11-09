"""Leakage, resolution and quota invariants of the added generator baseline."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from examples.mvtec_ad2.downstream.anomalydiffusion_pairs import AnomalyDiffusionPairs
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.quota_dataset import FixedQuotaDataset
from examples.mvtec_ad2.downstream.training import source_aware_loss
from examples.mvtec_ad2.splits import create_manifest, manifest_samples
from experiments.anomalydiffusion.prepare import native_patch, latent_pixels
from .test_downstream import create_images


class AnomalyDiffusionTests(unittest.TestCase):
    def cache(self, root):
        image = np.full((256, 256, 3), 64, np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        mask[4, 4] = 255
        for name, value in (("image", image), ("target", image), ("mask", mask)):
            Image.fromarray(value).save(root / f"{name}.png")
        split = {"fingerprint": "frozen", "partitions": {
            "train": [{"sample_id": "healthy", "label": "good"}, {"sample_id": "defect", "label": "bad"}],
            "validation": [{"sample_id": "heldout", "label": "bad"}], "test": []}}
        cache = {"version": 1, "source": "anomalydiffusion", "split_fingerprint": "frozen",
                 "training_sample_ids": ["healthy", "defect"], "rows": [{
                     "image_path": "image.png", "target_path": "target.png", "mask_path": "mask.png",
                     "recipient_sample_id": "healthy"}]}
        config = DownstreamConfiguration(model="dinov3_paired")
        config.data.patch_size = (256, 256)
        return split, cache, config

    def test_native_single_pixel_and_padding(self):
        mask = np.zeros((300, 513), np.uint8)
        mask[299, 512] = 255
        patch = native_patch(mask, 512, 256)
        self.assertEqual(patch.shape, (256, 256))
        self.assertEqual(np.count_nonzero(patch), 1)
        self.assertEqual(patch[43, 0], 255)
        # The audit must reveal losses, even when native preparation preserved pixels.
        tiny = np.zeros((256, 256), np.uint8)
        tiny[0, 0] = 255
        self.assertEqual(latent_pixels(tiny), 0)
        pooled = torch.nn.functional.adaptive_max_pool2d(torch.tensor(tiny.copy(), dtype=torch.float32)[None, None], (32, 32))
        self.assertEqual(int((pooled > 0).sum()), 1)

    def test_cache_preserves_binary_pixel_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, cache, config = self.cache(root)
            path = root / "manifest.json"
            path.write_text(json.dumps(cache))
            dataset = AnomalyDiffusionPairs(path, split, config.data)
            image, target, mask = dataset.sample(0, np.random.default_rng(42))
            self.assertEqual(float(mask.sum()), 1)
            torch.testing.assert_close(image, target)
            bad_cases = []
            altered = deepcopy(cache); altered["training_sample_ids"].append("heldout"); bad_cases.append(altered)
            altered = deepcopy(cache); altered["rows"][0]["recipient_sample_id"] = "defect"; bad_cases.append(altered)
            altered = deepcopy(cache); altered["split_fingerprint"] = "different"; bad_cases.append(altered)
            altered = deepcopy(cache); altered["smoke_test"] = True; bad_cases.append(altered)
            for invalid in bad_cases:
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    AnomalyDiffusionPairs(path, split, config.data)
            path.write_text(json.dumps(cache))
            Image.fromarray(np.zeros((256, 256), np.uint8)).save(root / "mask.png")
            with self.assertRaises(ValueError):
                AnomalyDiffusionPairs(path, split, config.data)

    def test_fixed_quotas_preserve_real_and_normal_draws(self):
        class Pairs:
            def __len__(self): return 2
            def sample(self, index, rng):
                return torch.ones(3, 64, 64), torch.zeros(3, 64, 64), torch.ones(1, 64, 64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_images(root)
            split = create_manifest(root / "can")
            normal = manifest_samples(split, "train", healthy_only=True)
            real = [s for s in manifest_samples(split, "train") if s.label != "good"]
            cfg = DownstreamConfiguration(model="dinov3_paired")
            cfg.data.patch_size = (64, 64)
            cfg.data.include_real_defects = True
            cfg.data.anomalydiffusion_manifest = "validated-by-runner.json"
            cfg.data.source_quotas = {"normal": 500, "real": 100, "anomalydiffusion": 400}
            first = FixedQuotaDataset(normal, real, None, cfg, anomalydiffusion=Pairs())
            second_cfg = deepcopy(cfg)
            second_cfg.data.source_quotas.update(anomalydiffusion=200, hybrid=200)
            second = FixedQuotaDataset(normal, real, Pairs(), second_cfg, anomalydiffusion=Pairs())
            for epoch in (0, 1):
                first.set_epoch(epoch); second.set_epoch(epoch)
                self.assertEqual(Counter(first.plan), cfg.data.source_quotas)
                self.assertEqual(Counter(second.plan), second_cfg.data.source_quotas)
                self.assertEqual(first.real_order, second.real_order)
                for source in ("normal", "real"):
                    i = first.entries.index((source, 0)); j = second.entries.index((source, 0))
                    torch.testing.assert_close(first[i]["image"], second[j]["image"])
                item = first[first.plan.index("anomalydiffusion")]
                self.assertEqual(item["source"], "anomalydiffusion")
                self.assertTrue(item["mask"].all())

    def test_anomalydiffusion_enters_paired_loss(self):
        class Model:
            def paired_loss(self, image, target, mask):
                self.seen = image[:, 0, 0, 0].tolist()
                return image.sum() * 0 + 2
            def __call__(self, image): return None, image
            def segmentation_loss(self, image, mask): return image.sum() * 0 + 1
        image = torch.arange(3.).view(3, 1, 1, 1)
        model = Model()
        result = source_aware_loss(model, image, image, image, ["real", "normal", "anomalydiffusion"], pair_healthy_samples=False)
        self.assertEqual(model.seen, [2.])
        self.assertAlmostEqual(float(result), 4 / 3, places=6)
