"""Frozen DINOv3 ConvNeXt with native stride-4/8/16/32 feature fusion."""

import torch
from torch import nn
from torch.nn import functional as F

from .dinov3 import DINOv3Paired, DINOv3Segmentation
from .dpt import ResidualBlock


class NativeFPNDecoder(nn.Module):
    """Top-down fusion at native scales, then RGB-guided full-size prediction."""

    def __init__(self, hidden_sizes, width, blocks):
        super().__init__()
        if len(hidden_sizes) != 4:
            raise ValueError("Native FPN requires four ConvNeXt feature stages.")
        self.projections = nn.ModuleList(
            nn.Sequential(nn.GroupNorm(1, channels), nn.Conv2d(channels, width, 1))
            for channels in hidden_sizes)
        self.refinement = nn.ModuleList(
            nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
            for _ in hidden_sizes)
        self.details = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(width, 2, 1))

    def forward(self, states, image):
        if len(states) != 4:
            raise ValueError("Expected four native ConvNeXt feature maps.")
        fused = None
        for level in reversed(range(4)):
            lateral = self.projections[level](states[level])
            if fused is not None:
                lateral = lateral + F.interpolate(fused, size=lateral.shape[-2:],
                                                  mode="bilinear", align_corners=False)
            fused = self.refinement[level](lateral)
        # The finest pretrained features have stride 4; keep the existing RGB
        # detail concept for sub-stride boundaries and thin defects.
        fused = F.interpolate(fused, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat((fused, self.details(image)), dim=1))


class DINOv3ConvNeXtSegmentation(DINOv3Segmentation):
    def __init__(self, settings, *, backbone_config=None, backbone=None):
        nn.Module.__init__(self)
        self.settings = settings
        if settings.backbone_type != "convnext" or settings.decoder != "native_fpn":
            raise ValueError("ConvNeXt segmentation requires backbone_type=convnext and decoder=native_fpn.")
        if backbone is None:
            try:
                from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel
            except ImportError as exc:
                raise ImportError("DINOv3 ConvNeXt requires Transformers >=4.56 (the dinov3 extra).") from exc
            if backbone_config is None:
                backbone = DINOv3ConvNextModel.from_pretrained(settings.backbone, revision=settings.revision)
            else:
                backbone = DINOv3ConvNextModel(DINOv3ConvNextConfig.from_dict(backbone_config))
        if getattr(backbone.config, "model_type", None) != "dinov3_convnext":
            raise ValueError("Expected a pretrained DINOv3 ConvNeXt backbone.")
        self.backbone = backbone.requires_grad_(False).eval()
        self.decoder = NativeFPNDecoder(backbone.config.hidden_sizes,
                                        settings.decoder_width, settings.decoder_blocks)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image):
        height, width = image.shape[-2:]
        padded = F.pad(image, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        with torch.no_grad():
            output = self.backbone(pixel_values=(padded - self.mean) / self.std,
                                   output_hidden_states=True)
        # Hugging Face returns the input followed by the four spatial stages.
        states = output.hidden_states[1:]
        expected = [(padded.shape[-2] // stride, padded.shape[-1] // stride)
                    for stride in (4, 8, 16, 32)]
        if len(states) != 4 or [tuple(state.shape[-2:]) for state in states] != expected:
            raise ValueError("Unexpected native ConvNeXt stride-4/8/16/32 layout.")
        logits = self.decoder(states, padded)[..., :height, :width]
        return None, logits


class DINOv3ConvNeXtPaired(DINOv3ConvNeXtSegmentation):
    paired_loss = DINOv3Paired.paired_loss
