"""Masked bank construction, bounded search, inference and checkpoint integration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.memory_bank import SuperADDDualBank, Reservoir
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream
from examples.mvtec_ad2.splits import create_manifest, SplitConfiguration
from .test_dinov3 import TinyBackbone
from .test_downstream import create_images, generator_config, seed_hybrid


class MemoryBankTests(unittest.TestCase):
    def config(self):
        cfg = DownstreamConfiguration(model='superadd_dual_bank')
        cfg.memory_bank.layers = (1, 2, 3, 4)
        cfg.memory_bank.normal_capacity = 16
        cfg.memory_bank.defect_capacity = 8
        cfg.memory_bank.query_chunk_size = 3
        cfg.memory_bank.reference_chunk_size = 5
        cfg.memory_bank.calibration_images = 2
        cfg.training.device = 'cpu'
        cfg.training.num_workers = 0
        cfg.data.patch_size = (64, 64)
        cfg.data.samples_per_epoch = 8
        return cfg

    def test_chunked_neighbors_match_full_reference(self):
        model = SuperADDDualBank(self.config(), backbone=TinyBackbone())
        q = torch.randn(9, 8)
        bank = torch.randn(17, 8)
        torch.testing.assert_close(model.nearest(q, bank), torch.cdist(q, bank).min(1).values)
        with self.assertRaises(ValueError):
            model.nearest(q, bank[:0])

    def test_bounded_deterministic_reservoir(self):
        values = torch.arange(160.).reshape(20, 8)
        first, second = Reservoir(4, 42), Reservoir(4, 42)
        first.add(values[:10]); first.add(values[10:])
        second.add(values)
        self.assertEqual(len(first.values), 4)
        torch.testing.assert_close(first.values, second.values)

    def test_only_masked_tokens_enter_defect_bank_including_single_pixel(self):
        cfg = self.config()
        model = SuperADDDualBank(cfg, backbone=TinyBackbone())
        features = [torch.nn.functional.normalize(torch.randn(1, 16, 8), dim=-1)
                    for _ in cfg.memory_bank.layers]
        image = torch.zeros(3, 64, 64)
        mask = torch.zeros(1, 64, 64)
        mask[0, 33, 17] = 1  # Feature cell row=2, col=1, flattened index=9.
        with tempfile.TemporaryDirectory() as tmp, patch.object(model, 'features', return_value=(features, (4, 4))):
            model.fit([{'image': image}], [{'image': image, 'mask': mask, 'source': 'hybrid'}],
                      [{'image': image}], Path(tmp), 42)
        for layer, values in zip(cfg.memory_bank.layers, features):
            torch.testing.assert_close(getattr(model, f'defect_{layer}'), values[0, 9:10])

    def test_two_banks_distinguish_known_defects_and_reload(self):
        cfg = self.config()
        model = SuperADDDualBank(cfg, backbone=TinyBackbone())
        q = torch.zeros(1, 8); q[0, 0] = 1
        normal = -q
        for layer in cfg.memory_bank.layers:
            setattr(model, f'normal_{layer}', normal.clone())
            setattr(model, f'defect_{layer}', q.clone())
        model.fitted = True
        image = torch.zeros(1, 3, 32, 48)
        features = ([q[None].repeat(1, 6, 1) for _ in cfg.memory_bank.layers], (2, 3))
        with patch.object(model, 'features', return_value=features):
            _, logits = model(image)
            known = logits.softmax(1)[:, 1].mean()
            for layer in cfg.memory_bank.layers:
                setattr(model, f'defect_{layer}', normal.clone())
            unknown = model(image)[1].softmax(1)[:, 1].mean()
        self.assertGreater(known.item(), unknown.item())
        restored = SuperADDDualBank(cfg, backbone=TinyBackbone())
        restored.load_state_dict(model.state_dict())
        self.assertTrue(restored.fitted)
        self.assertEqual(len(restored.normal_1), 1)
        self.assertEqual(restored(image)[1].shape, (1, 2, 32, 48))

    def test_baseline_hybrid_and_normal_only_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);create_images(root)
            manifest = create_manifest(root/'can', SplitConfiguration())
            gen = generator_config(root/'study');seed_hybrid(gen, manifest)
            textures = root/'textures';textures.mkdir()
            Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(textures/'texture.png')
            cfg = self.config();cfg.data.texture_root = str(textures)
            def factory(config, **kwargs):
                torch.manual_seed(42)
                return SuperADDDualBank(config, backbone=TinyBackbone())
            for normal, hybrid in ((.5, 0.), (.5, .5), (1., 0.)):
                cfg.data.normal_fraction = normal;cfg.data.hybrid_fraction = hybrid
                with patch('examples.mvtec_ad2.downstream.runner.build_model', side_effect=factory):
                    run = train_downstream(gen, manifest, cfg)
                    metrics = evaluate_downstream(run, manifest)
                self.assertIsNotNone(metrics['test']['pixel_aupro_0.05'])
                checkpoint = torch.load(run/'checkpoints/best.pt', weights_only=True)
                sizes = checkpoint['bank_build']['bank_sizes']
                self.assertTrue(all(s['normal'] <= 16 and s['defect'] <= 8 for s in sizes.values()))
                counts = checkpoint['bank_build']['source_counts']
                if normal == 1:
                    self.assertEqual(counts, {})
                    self.assertTrue(all(s['defect'] == 0 for s in sizes.values()))
                else:
                    self.assertEqual(counts['normal'], 4)
                    self.assertEqual(counts.get('hybrid', 0), 0 if hybrid == 0 else 2)
                self.assertTrue((run/'bank_sources.json').is_file())

    def test_invalid_bank_configuration(self):
        cfg = self.config();cfg.memory_bank.normal_capacity = 0
        with self.assertRaises(ValueError):cfg.validate()
