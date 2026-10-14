"""Frozen DINOv3 ViT features with supervised paired anomaly segmentation."""

import torch
from torch import nn
from torch.nn import functional as F

from .draem.losses import focal_loss


class DINOv3Segmentation(nn.Module):
    def __init__(self, settings, *, backbone_config=None, backbone=None):
        super().__init__()
        self.settings = settings
        if backbone is None:
            try:
                from transformers import DINOv3ViTConfig, DINOv3ViTModel
            except ImportError as exc:
                raise ImportError("DINOv3 requires the dinov3 extra: pip install -e '.[dinov3]'.") from exc
            if backbone_config is None:
                backbone = DINOv3ViTModel.from_pretrained(settings.backbone, revision=settings.revision)
            else:
                # Full backbone weights live in the checkpoint: no download on reload.
                backbone = DINOv3ViTModel(DINOv3ViTConfig.from_dict(backbone_config))
        self.backbone = backbone.requires_grad_(False).eval()
        self.patch_size = int(backbone.config.patch_size)
        width = settings.decoder_width
        if settings.decoder == "dpt":
            from .dpt import DPTDecoder
            depth = backbone.config.num_hidden_layers
            if depth < 4:
                raise ValueError("DPT needs at least four DINOv3 transformer layers.")
            self.feature_layers = tuple((depth * (i + 1) + 3) // 4 for i in range(4))
            self.dpt = DPTDecoder(backbone.config.hidden_size, width, settings.decoder_blocks)
        else:
            self.projection = nn.Conv2d(backbone.config.hidden_size, width, 1)
        # Native-resolution image branch retains fine details below a ViT patch.
            self.details = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
            self.decoder = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1),
                                         nn.GELU(), nn.Conv2d(width, 2, 1))
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image):
        height, width = image.shape[-2:]
        p = self.patch_size
        padded = F.pad(image, (0, (-width) % p, 0, (-height) % p), mode="replicate")
        with torch.no_grad():
            if self.settings.decoder == "dpt":
                output = self.backbone(pixel_values=(padded - self.mean) / self.std, output_hidden_states=True)
            else:
                output = self.backbone(pixel_values=(padded - self.mean) / self.std)
            tokens = output.last_hidden_state
        gh, gw = padded.shape[-2] // p, padded.shape[-1] // p
        offset = 1 + self.backbone.config.num_register_tokens
        patches = tokens[:, offset:]
        if patches.shape[1] != gh * gw:
            raise ValueError("Unexpected DINOv3 spatial token layout.")
        if self.settings.decoder == "dpt":
            states = [output.hidden_states[i][:, offset:] for i in self.feature_layers]
            logits = self.dpt(states, (gh, gw), padded)[..., :height, :width]
            return None, logits
        features = patches.transpose(1, 2).reshape(len(image), -1, gh, gw)
        features = F.interpolate(self.projection(features), size=padded.shape[-2:],
                                 mode="bilinear", align_corners=False)[..., :height, :width]
        logits = self.decoder(torch.cat([features, self.details(image)], dim=1))
        # Shared inference interface; DINOv3 does not reconstruct an image.
        return None, logits

    def segmentation_loss(self, logits, mask):
        """Supervised focal + Dice, requiring only image logits and ground truth."""
        probability = logits.softmax(1)[:, 1:2]
        # Dice only on anomalous samples; empty masks use focal supervision.
        active = mask.flatten(1).sum(1) > 0
        dice = probability.sum() * 0
        if active.any():
            pred, truth = probability[active].flatten(1), mask[active].flatten(1)
            dice = (1 - (2 * (pred * truth).sum(1) + 1e-6) /
                    (pred.sum(1) + truth.sum(1) + 1e-6)).mean()
        return focal_loss(logits, mask) + self.settings.dice_weight * dice


class DINOv3Paired(DINOv3Segmentation):
    def paired_loss(self, image, target, mask):
        _, logits = self(image)
        _, healthy_logits = self(target)
        probability = logits.softmax(1)[:, 1:2]
        healthy_probability = healthy_logits.softmax(1)[:, 1:2]
        background = 1 - mask
        consistency = ((probability - healthy_probability).square() * background).sum() / background.sum().clamp_min(1)
        return (self.segmentation_loss(logits, mask)
                + self.settings.healthy_loss_weight * focal_loss(healthy_logits, torch.zeros_like(mask))
                + self.settings.consistency_weight * consistency)
