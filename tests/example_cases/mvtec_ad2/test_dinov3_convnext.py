"""Native ConvNeXt feature scales, paired gradients and checkpoint reload."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.dinov3_convnext import DINOv3ConvNeXtPaired
from examples.mvtec_ad2.downstream.models import build_model
from examples.mvtec_ad2.downstream.training import source_aware_loss


class TinyConvNeXt(nn.Module):
    """Small native spatial-stage fixture; pretrained weights are a GPU smoke check."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="dinov3_convnext", hidden_sizes=[4, 8, 16, 32])
        self.stages = nn.ModuleList([nn.Conv2d(3, 4, 4, stride=4),
                                    nn.Conv2d(4, 8, 2, stride=2),
                                    nn.Conv2d(8, 16, 2, stride=2),
                                    nn.Conv2d(16, 32, 2, stride=2)])

    def forward(self, pixel_values, output_hidden_states=False):
        states = [pixel_values]
        for stage in self.stages:
            states.append(stage(states[-1]))
        return SimpleNamespace(hidden_states=tuple(states))


class ConvNeXtTests(unittest.TestCase):

    def config(self):
        config = DownstreamConfiguration(model="dinov3_paired")
        config.dinov3.backbone_type = "convnext"
        config.dinov3.backbone = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
        config.dinov3.decoder = "native_fpn"
        config.dinov3.decoder_width = 4
        config.dinov3.decoder_blocks = 1
        return config

    def backbone(self):
        return TinyConvNeXt()

    def test_native_stages_gradients_source_losses_and_reload(self):
        config = self.config()
        config.data.include_real_defects = True
        config.data.source_quotas = dict(normal=500, real=100, hybrid=400)
        config.dinov3.pair_healthy_samples = False
        config = DownstreamConfiguration.from_dict(config.to_dict())
        backbone = self.backbone()
        supplied = DINOv3ConvNeXtPaired(config.dinov3, backbone=backbone)
        with patch("examples.mvtec_ad2.downstream.dinov3_convnext.DINOv3ConvNeXtPaired",
                   return_value=supplied):
            model = build_model(config).train()
        self.assertIsInstance(model, DINOv3ConvNeXtPaired)
        self.assertFalse(model.backbone.training)
        image = torch.rand(3, 3, 35, 49)
        mask = torch.zeros(3, 1, 35, 49)
        mask[1:, :, 8:12, 10:15] = 1
        self.assertEqual(model(image)[1].shape, (3, 2, 35, 49))
        loss = source_aware_loss(model, image, image * .9, mask,
                                ["normal", "real", "hybrid"], pair_healthy_samples=False)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        for projection in model.decoder.projections:
            self.assertTrue(torch.isfinite(projection[1].weight.grad).all())
        self.assertIsNotNone(model.decoder.details[0].weight.grad)
        self.assertIsNotNone(model.decoder.head[-1].weight.grad)
        restored = DINOv3ConvNeXtPaired(config.dinov3, backbone=self.backbone()).eval()
        restored.load_state_dict(model.state_dict())
        model.eval()
        torch.testing.assert_close(model(image)[1], restored(image)[1])

    def test_legacy_defaults_and_incompatible_decoders(self):
        legacy = DownstreamConfiguration.from_dict({"model": "dinov3_paired"})
        self.assertEqual(legacy.dinov3.backbone_type, "vit")
        config = self.config()
        config.dinov3.decoder = "cnn"
        with self.assertRaisesRegex(ValueError, "native_fpn"):
            config.validate()
        config.dinov3.backbone_type = "vit"
        config.dinov3.decoder = "native_fpn"
        with self.assertRaisesRegex(ValueError, "native_fpn"):
            config.validate()
