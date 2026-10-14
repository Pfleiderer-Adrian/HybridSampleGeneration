"""DPT-style four-level reassembly and deep residual feature fusion."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, image):
        return image + self.layers(image)


class DPTDecoder(nn.Module):
    def __init__(self, hidden_size, width, blocks):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.projections = nn.ModuleList([nn.Conv2d(hidden_size, width, 1) for _ in range(4)])
        self.reassembly = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(4)])
        self.skip_blocks = nn.ModuleList([ResidualBlock(width) for _ in range(4)])
        self.fusion = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)]) for _ in range(4)])
        self.details = nn.Sequential(nn.Conv2d(3, width, 3, padding=1),
                                     *[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1),
                                  *[ResidualBlock(width) for _ in range(blocks)],
                                  nn.GELU(), nn.Conv2d(width, 2, 1))

    def forward(self, tokens, grid, image):
        pyramid = []
        for i, (state, scale) in enumerate(zip(tokens, (4., 2., 1., .5))):
            features = self.norms[i](state).transpose(1, 2).reshape(len(image), -1, *grid)
            size = tuple(max(1, int(n * scale)) for n in grid)
            features = F.interpolate(self.projections[i](features), size=size,
                                     mode="bilinear", align_corners=False)
            pyramid.append(self.reassembly[i](features))
        fused = None
        for i in reversed(range(4)):
            skip = self.skip_blocks[i](pyramid[i])
            if fused is not None:
                skip = skip + F.interpolate(fused, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.fusion[i](skip)
        fused = F.interpolate(fused, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([fused, self.details(image)], dim=1))
