"""Real training defects augment quotas without leaking held-out images."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import torch
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.datasets import MixedTrainingDataset
from examples.mvtec_ad2.splits import create_manifest, SplitConfiguration, manifest_samples
from .test_downstream import create_images

class RealTrainingTests(unittest.TestCase):
    def test_all_real_images_added_with_positive_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_images(root)
            manifest = create_manifest(root / 'can', SplitConfiguration())
            cfg = DownstreamConfiguration(model='dinov3_paired')
            cfg.data.include_real_defects = True
            cfg.data.hybrid_fraction = 0
            cfg.data.samples_per_epoch = 8
            cfg.data.patch_size = (64, 64)
            real = [s for s in manifest_samples(manifest, 'train') if s.label != 'good']
            with unittest.mock.patch('examples.mvtec_ad2.downstream.datasets.TextureSynthesizer'):
                ds = MixedTrainingDataset(manifest_samples(manifest, 'train', healthy_only=True), None, cfg, real_samples=real)
            self.assertEqual(len(ds), 8 + len(real))
            self.assertEqual(Counter(ds.plan), {'normal': 4, 'draem': 4, 'real': len(real)})
            self.assertEqual({s.sample_id for s in ds.real_indices.values()}, {s.sample_id for s in real})
            for i in ds.real_indices:
                sample = ds[i]
                self.assertTrue(sample['mask'].any())
                self.assertEqual(sample['source'], 'real')
                torch.testing.assert_close(sample['image'], sample['target'])
            ds.set_epoch(2)
            self.assertEqual(len(ds.real_indices), len(real))

    def test_real_augmentation_not_silently_ignored(self):
        cfg = DownstreamConfiguration()
        cfg.data.include_real_defects = True
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_real_defects_use_segmentation_loss_without_healthy_partner(self):
        from unittest.mock import patch
        from torch import nn
        from examples.mvtec_ad2.downstream.training import train
        from examples.mvtec_ad2.downstream.datasets import RealImageDataset
        class CheckedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(1, 2, 64, 64))
                self.backbone = type('Backbone', (), {'config': type('Config', (), {'to_dict': lambda _: {}})()})()
                self.real_seen = 0
            def forward(self, image):
                return None, self.logits.expand(len(image), -1, -1, -1)
            def paired_loss(self, image, target, mask):
                if mask.any():
                    raise AssertionError('Real defects must not get healthy-partner loss')
                return self.logits.square().mean() + self.logits.mean()
            def segmentation_loss(self, logits, mask):
                self.real_seen += len(mask)
                return torch.nn.functional.cross_entropy(logits, mask[:, 0].long())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); create_images(root)
            manifest = create_manifest(root / 'can', SplitConfiguration())
            cfg = DownstreamConfiguration(model='dinov3_paired')
            cfg.data.include_real_defects = True
            cfg.data.hybrid_fraction = 0
            cfg.data.samples_per_epoch = 8
            cfg.data.normal_fraction = .5
            cfg.data.patch_size = (64, 64)
            cfg.training.epochs = 1; cfg.training.num_workers = 0
            cfg.training.selection_metric = 'pixel_aupro_0.05'
            cfg.training.device = 'cpu'
            real = [s for s in manifest_samples(manifest, 'train') if s.label != 'good']
            with patch('examples.mvtec_ad2.downstream.datasets.TextureSynthesizer'):
                ds = MixedTrainingDataset(manifest_samples(manifest, 'train', healthy_only=True), None, cfg, real_samples=real)
            # Replace synthetic entries by healthy entries to isolate the real branch.
            original = ds.set_epoch
            def epoch(n):
                original(n)
                ds.plan = ['normal' if s == 'draem' else s for s in ds.plan]
            ds.set_epoch = epoch
            model = CheckedModel()
            with patch('examples.mvtec_ad2.downstream.training.build_model', return_value=model), patch('examples.mvtec_ad2.downstream.training.evaluate', return_value={'pixel_aupro_0.05': .5, 'segmentation_loss': .1}):
                train(ds, RealImageDataset([], cfg.data), cfg, root / 'run')
            self.assertEqual(model.real_seen, len(real))
