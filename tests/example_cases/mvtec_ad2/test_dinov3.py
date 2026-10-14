"""Paired decoder training, frozen backbone and downstream snapshot integration."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
import numpy as np

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration, DINOv3Configuration
from examples.mvtec_ad2.downstream.dinov3 import DINOv3Paired, DINOv3Segmentation
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream
from examples.mvtec_ad2.downstream.sampling import source_plan
from examples.mvtec_ad2.downstream.datasets import FixedPairedValidation
from examples.mvtec_ad2.splits import create_manifest, SplitConfiguration, manifest_samples
from .test_downstream import create_images, generator_config, seed_hybrid


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=16, hidden_size=8, num_register_tokens=2, num_hidden_layers=4,
                                     to_dict=lambda: {"patch_size": 16})
        self.conv = nn.Conv2d(3, 8, 16, stride=16)

    def forward(self, pixel_values, output_hidden_states=False):
        tokens = self.conv(pixel_values).flatten(2).transpose(1, 2)
        prefix = tokens.new_zeros(len(tokens), 3, 8)
        state = torch.cat([prefix, tokens], dim=1)
        return SimpleNamespace(last_hidden_state=state, hidden_states=(state,) * 5)


class DINOv3Tests(unittest.TestCase):
    def test_deep_dpt_four_layers_native_output_and_gradients(self):
        settings = DINOv3Configuration(decoder='dpt', decoder_width=4, decoder_blocks=2)
        model = DINOv3Segmentation(settings, backbone=TinyBackbone())
        self.assertEqual(model.feature_layers, (1, 2, 3, 4))
        image = torch.rand(2, 3, 35, 49)
        mask = torch.zeros(2, 1, 35, 49)
        mask[:, :, 10:12, 20:22] = 1
        _, logits = model(image)
        self.assertEqual(logits.shape, (2, 2, 35, 49))
        model.segmentation_loss(logits, mask).backward()
        for projection in model.dpt.projections:
            self.assertIsNotNone(projection.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        restored = DINOv3Segmentation(settings, backbone=TinyBackbone())
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(logits, restored(image)[1])

    def test_segmentation_requires_only_image_and_mask(self):
        model = DINOv3Segmentation(DINOv3Configuration(decoder_width=4), backbone=TinyBackbone())
        image = torch.rand(2, 3, 32, 48)
        mask = torch.zeros(2, 1, 32, 48)
        mask[0, :, 5:8, 10:12] = 1
        with patch.object(model.backbone, 'forward', wraps=model.backbone.forward) as backbone:
            _, logits = model(image)
            loss = model.segmentation_loss(logits, mask)
            self.assertEqual(backbone.call_count, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(hasattr(model, 'paired_loss'))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def model(self, settings=None):
        return DINOv3Paired(settings or DINOv3Configuration(decoder_width=4), backbone=TinyBackbone())

    def test_frozen_backbone_native_shape_and_pair_gradients(self):
        model = self.model().train()
        self.assertFalse(model.backbone.training)
        image = torch.rand(2, 3, 35, 49)
        target = image.clone()
        mask = torch.zeros(2, 1, 35, 49)
        mask[:, :, 10:12, 20:22] = 1
        _, logits = model(image)
        self.assertEqual(logits.shape, (2, 2, 35, 49))
        loss = model.paired_loss(image, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))
        self.assertIsNotNone(model.decoder[-1].weight.grad)

    def test_healthy_partner_affects_loss_and_healthy_masks_are_finite(self):
        torch.manual_seed(12)
        model = self.model()
        image = torch.rand(1, 3, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        first = model.paired_loss(image, image, mask)
        second = model.paired_loss(image, torch.zeros_like(image), mask)
        self.assertTrue(torch.isfinite(first))
        self.assertNotEqual(first.item(), second.item())

    def test_configuration_roundtrip_old_snapshots_and_quotas(self):
        old = DownstreamConfiguration.from_dict({})
        self.assertEqual(old.model, "draem")
        config = DownstreamConfiguration(model="dinov3_paired")
        config.dinov3.consistency_weight = .3
        self.assertEqual(DownstreamConfiguration.from_dict(config.to_dict()).to_dict(), config.to_dict())
        config.data.normal_fraction = .75
        config.data.hybrid_fraction = .5
        plan = source_plan(config.data, 42, 0)
        self.assertEqual([plan.count(x) for x in ("normal", "draem", "hybrid")], [750, 125, 125])
        config.model = "invalid"
        with self.assertRaises(ValueError):
            config.validate()

    def test_actual_transformers_backbone_and_offline_reload(self):
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError:
            self.skipTest("Optional DINOv3 Transformers extra is not installed.")
        backbone_config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_register_tokens=2, image_size=32, patch_size=16)
        model = DINOv3Paired(DINOv3Configuration(decoder_width=4),
                            backbone=DINOv3ViTModel(backbone_config))
        image = torch.rand(1, 3, 32, 48)
        mask = torch.zeros(1, 1, 32, 48)
        mask[:, :, 4:6, 10:12] = 1
        model.paired_loss(image, image, mask).backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.eval()
        restored = DINOv3Paired(model.settings, backbone_config=backbone_config.to_dict())
        restored.load_state_dict(model.state_dict())
        restored.eval()
        torch.testing.assert_close(model(image)[1], restored(image)[1])

    def test_baseline_and_hybrid_train_reload_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = create_manifest(root / "can", SplitConfiguration())
            generator = generator_config(root / "study")
            seed_hybrid(generator, manifest)
            textures = root / "textures"
            textures.mkdir()
            Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(textures / "texture.png")
            config = DownstreamConfiguration(model="dinov3_paired")
            config.data.mode = "patch"
            config.data.patch_size = (64, 64)
            config.data.texture_root = str(textures)
            config.data.samples_per_epoch = 8
            config.training.epochs = 1
            config.training.batch_size = 2
            config.training.num_workers = 0
            config.training.device = "cpu"
            config.dinov3.decoder_width = 4
            config.dinov3.validation_samples = 4
            def factory(cfg, **kwargs):
                return (self.model(cfg.dinov3) if cfg.model == "dinov3_paired" else
                        DINOv3Segmentation(cfg.dinov3, backbone=TinyBackbone()))
            for model_name, fraction, criterion in (("dinov3_paired", 0., "auto"), ("dinov3_paired", .5, "auto"),
                                         ("dinov3_segmentation", 0., "auto"), ("dinov3_segmentation", .5, "auto"),
                                         ("dinov3_paired", .5, "pixel_aupro_0.05")):
                config.model = model_name
                config.training.selection_metric = criterion
                config.data.hybrid_fraction = fraction
                with patch('examples.mvtec_ad2.downstream.training.build_model', side_effect=factory), \
                     patch('examples.mvtec_ad2.downstream.runner.build_model', side_effect=factory) as reload:
                    run = train_downstream(generator, manifest, config)
                    metrics = evaluate_downstream(run, manifest)
                    self.assertIn('backbone_config', reload.call_args.kwargs)
                self.assertEqual(run.parent.name, model_name)
                checkpoint = torch.load(run / 'checkpoints/best.pt', weights_only=True)
                selection = criterion if criterion != 'auto' else ('paired_total_loss' if model_name == 'dinov3_paired' else 'segmentation_total_loss')
                self.assertEqual(checkpoint['selection_metric'], selection)
                self.assertEqual(checkpoint['selection_score'], checkpoint['validation'][selection])
                self.assertEqual(checkpoint['selection_mode'], 'max' if criterion != 'auto' else 'min')
                self.assertEqual((run / 'paired_validation.json').exists(), model_name == 'dinov3_paired' and criterion == 'auto')
                self.assertIsNotNone(metrics['test']['pixel_aupro_0.05'])
                history = (run / 'training_history.csv').read_text().splitlines()[1].split(',')
                self.assertEqual(list(map(int, history[-3:])), [4, 4, 0] if fraction == 0 else [4, 2, 2])
                fixed = FixedPairedValidation(
                    manifest_samples(
                        manifest, 'validation', healthy_only=True), config)
                before = fixed[0]
                fixed.set_epoch(99)
                after = fixed[0]
                for key in ('image', 'target', 'mask'):
                    self.assertTrue(torch.equal(before[key], after[key]))
                self.assertEqual(fixed.plan, ['draem'] * 4)
