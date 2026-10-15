"""Multi-layer DINO features with progressive residual segmentation refinement."""

import torch
from torch import nn
from torch.nn import functional as F

from .dpt import ResidualBlock


class ConvNeXtResidualBlock(nn.Module):
    """Depthwise spatial filtering and a channel MLP with LayerScale."""

    def __init__(self, width):
        super().__init__()
        self.spatial = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.channels = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                      nn.Linear(4 * width, width))
        self.layer_scale = nn.Parameter(torch.full((width,), 1e-6))

    def forward(self, image):
        branch = self.spatial(image).permute(0, 2, 3, 1)
        branch = self.channels(self.norm(branch)) * self.layer_scale
        return image + branch.permute(0, 3, 1, 2)


class ResidualFPNDecoder(nn.Module):
    """Fuse four ViT depths from coarse context to native image details."""

    def __init__(self, hidden_size, width, blocks, *, block_kind="residual"):
        super().__init__()
        if block_kind not in ("residual", "convnext"):
            raise ValueError("Unknown FPN block kind.")
        block = ConvNeXtResidualBlock if block_kind == "convnext" else ResidualBlock
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(4))
        self.projections = nn.ModuleList(nn.Conv2d(hidden_size, width, 1) for _ in range(4))
        self.fusion = nn.ModuleList(
            nn.Sequential(*[block(width) for _ in range(blocks)])
            for _ in range(4))
        detail_width = max(1, width // 2)
        self.details = nn.Sequential(
            nn.Conv2d(3, detail_width, 3, padding=1),
            *[block(detail_width) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(width + detail_width, width, 3, padding=1),
            *[block(width) for _ in range(blocks)],
            nn.GELU(), nn.Conv2d(width, 2, 1))

    def forward(self, tokens, grid, image):
        # ViT depths share one grid; construct a pyramid through learned fusion
        # and interpolation, rather than treating them as native CNN scales.
        features = [projection(norm(state).transpose(1, 2).reshape(len(image), -1, *grid))
                    for state, norm, projection in zip(tokens, self.norms, self.projections)]
        fused = None
        for level in reversed(range(4)):
            scale = 2 ** (3 - level)
            size = tuple(min(image.shape[-2 + axis], n * scale)
                         for axis, n in enumerate(grid))
            lateral = F.interpolate(features[level], size=size, mode="bilinear", align_corners=False)
            if fused is not None:
                lateral = lateral + F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            fused = self.fusion[level](lateral)
        fused = F.interpolate(fused, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([fused, self.details(image)], dim=1))
