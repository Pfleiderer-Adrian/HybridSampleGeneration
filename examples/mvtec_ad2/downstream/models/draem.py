"""DRAEM topology: reconstructive encoder/decoder and discriminative U-Net.

Independent implementation of Zavrtanik et al., ICCV 2021. Parameter names are
local; upstream checkpoints are not interchangeable.
"""

import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, middle, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, middle, 3, padding=1), nn.BatchNorm2d(middle), nn.ReLU(inplace=True),
        nn.Conv2d(middle, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels, widths):
        super().__init__()
        self.blocks = nn.ModuleList(block(a, b, b) for a, b in zip([in_channels] + widths[:-1], widths))

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.blocks):
            x = layer(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, incoming, stages, out_channels):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.skip_widths = []
        for up_width, skip_width, output_width in stages:
            self.ups.append(nn.Sequential(nn.Conv2d(incoming, up_width, 3, padding=1), nn.BatchNorm2d(up_width), nn.ReLU(inplace=True)))
            self.blocks.append(block(up_width + skip_width, output_width if skip_width else up_width, output_width))
            self.skip_widths.append(skip_width)
            incoming = output_width
        self.output = nn.Conv2d(incoming, out_channels, 3, padding=1)

    def forward(self, features):
        x = features[-1]
        for i, (up, layer, skip) in enumerate(zip(self.ups, self.blocks, self.skip_widths)):
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))
            if skip:
                x = torch.cat((x, features[-2-i]), dim=1)
            x = layer(x)
        return self.output(x)


class DRAEM(nn.Module):
    def __init__(self, reconstruction_width=128, segmentation_width=64):
        super().__init__()
        r, s = reconstruction_width, segmentation_width
        self.reconstructor = Encoder(3, [r, 2*r, 4*r, 8*r, 8*r])
        self.reconstruction_decoder = Decoder(8*r, [(8*r, 0, 4*r), (4*r, 0, 2*r), (2*r, 0, r), (r, 0, r)], 3)
        self.segmentor = Encoder(6, [s, 2*s, 4*s, 8*s, 8*s, 8*s])
        self.segmentation_decoder = Decoder(8*s, [(8*s, 8*s, 8*s), (4*s, 8*s, 4*s), (2*s, 4*s, 2*s), (s, 2*s, s), (s, s, s)], 2)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, 0, 0.02)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight, 1, 0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, image):
        reconstruction = self.reconstruction_decoder(self.reconstructor(image))
        logits = self.segmentation_decoder(self.segmentor(torch.cat((reconstruction, image), dim=1)))
        return reconstruction, logits
