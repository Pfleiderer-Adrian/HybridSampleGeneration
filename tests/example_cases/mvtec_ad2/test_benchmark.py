"""Fixed budgets, correct CutMix labels, and synthetic-only paired losses."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.cutmix import cutmix_defect
from examples.mvtec_ad2.downstream.quota_dataset import FixedQuotaDataset
from examples.mvtec_ad2.downstream.training import source_aware_loss
from examples.mvtec_ad2.splits import create_manifest, manifest_samples
from .test_downstream import create_images


class BenchmarkTests(unittest.TestCase):
    def config(self, quotas):
        cfg = DownstreamConfiguration(model="dinov3_paired")
        cfg.data.include_real_defects = True
        cfg.data.source_quotas = quotas
        cfg.data.samples_per_epoch = sum(quotas.values())
        cfg.data.hybrid_fraction = 0
        cfg.data.patch_size = (64, 64)
        cfg.dinov3.pair_healthy_samples = False
        return cfg

    def test_cutmix_labels_are_defects_not_the_rectangle(self):
        healthy, donor = torch.zeros(3, 64, 64), torch.ones(3, 64, 64)
        mask = torch.zeros(1, 64, 64); mask[:, 20:24, 30:34] = 1
        for seed in range(20):
            image, result = cutmix_defect(healthy, donor, mask, np.random.default_rng(seed))
            rectangle = image[:1] > 0
            self.assertTrue(result.any())
            torch.testing.assert_close(result, mask * rectangle)
            self.assertTrue(torch.equal(image[:, ~rectangle[0]], healthy[:, ~rectangle[0]]))
        with self.assertRaises(ValueError):
            cutmix_defect(healthy, donor, torch.zeros_like(mask), np.random.default_rng(0))

    def test_quota_coverage_determinism_and_positive_cutmix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); create_images(root)
            manifest = create_manifest(root / "can")
            healthy = manifest_samples(manifest, "train", healthy_only=True)
            real = [s for s in manifest_samples(manifest, "train") if s.label != "good"]
            for source in ("real", "cutmix", "draem"):
                quotas = {"normal": 8, "real": 8}
                if source != "real": quotas[source] = 8
                cfg = self.config(quotas)
                with patch("examples.mvtec_ad2.downstream.quota_dataset.TextureSynthesizer"):
                    ds = FixedQuotaDataset(healthy, real, None, cfg)
                self.assertEqual(Counter(ds.plan), quotas)
                self.assertEqual(set(ds.real_order[:len(real)]), set(range(len(real))))
                self.assertEqual({s.sample_id for s in ds.real_indices.values()}, {s.sample_id for s in real})
                before = list(ds.entries); ds.set_epoch(0); self.assertEqual(ds.entries, before)
                for i, (s, _) in enumerate(ds.entries):
                    if s == "draem": continue
                    item = ds[i]
                    self.assertEqual(bool(item["mask"].any()), s != "normal")
                    torch.testing.assert_close(item["image"], ds[i]["image"])
                    if s == "cutmix":
                        background = ~item["mask"].bool().expand_as(item["image"])
                        self.assertTrue(torch.isfinite(item["image"][background]).all())

    def test_paired_loss_only_sees_synthetic_sources(self):
        class Model:
            def paired_loss(self, image, target, mask):
                self.paired_ids = image[:, 0, 0, 0].tolist()
                return image.sum() * 0 + 4
            def __call__(self, image):
                self.supervised_ids = image[:, 0, 0, 0].tolist()
                return None, image
            def segmentation_loss(self, logits, mask):
                return logits.sum() * 0 + 2
        sources = ["normal", "real", "draem", "cutmix", "hybrid"]
        images = torch.arange(5.).view(5, 1, 1, 1)
        model = Model()
        loss = source_aware_loss(model, images, images, torch.ones_like(images), sources,
                                 pair_healthy_samples=False)
        self.assertEqual(model.paired_ids, [2., 3., 4.])
        self.assertEqual(model.supervised_ids, [0., 1.])
        self.assertAlmostEqual(loss.item(), 3.2, places=5)
        model = Model()
        source_aware_loss(model, images[:2], images[:2], images[:2], sources[:2],
                          pair_healthy_samples=False)
        self.assertFalse(hasattr(model, "paired_ids"))

    def test_invalid_quotas_fail(self):
        cfg = self.config({"normal": 5, "real": 5})
        cfg.data.source_quotas["real"] = 4
        with self.assertRaises(ValueError): cfg.validate()
        cfg.data.source_quotas = {"normal": 5, "typo": 5}
        with self.assertRaises(ValueError): cfg.validate()

