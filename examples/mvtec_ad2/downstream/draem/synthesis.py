"""Gradient Perlin masks with randomly augmented external textures."""

from pathlib import Path
import numpy as np
import torch
from examples.common.image_io import IMAGE_EXTENSIONS
from ..transforms import image_tensor, read_image


def perlin_noise(size, resolution, rng):
    h, w = size
    ry, rx = resolution
    angles = rng.uniform(0, 2*np.pi, (ry+1, rx+1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    y, x = np.meshgrid(np.arange(h)*ry/h, np.arange(w)*rx/w, indexing="ij")
    iy, ix = y.astype(int), x.astype(int)
    fy, fx = y-iy, x-ix
    def dot(dy, dx):
        g = gradients[iy+dy, ix+dx]
        return g[..., 0]*(fy-dy) + g[..., 1]*(fx-dx)
    def fade(t):
        return t*t*t*(t*(t*6-15)+10)
    sy, sx = fade(fy), fade(fx)
    top = (1-sx)*dot(0, 0) + sx*dot(0, 1)
    bottom = (1-sx)*dot(1, 0) + sx*dot(1, 1)
    return np.sqrt(2)*((1-sy)*top + sy*bottom)


class TextureSynthesizer:
    def __init__(self, root):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if root else []
        if not self.paths:
            raise ValueError("DRAEM synthesis requires texture_root containing texture images (e.g. DTD).")

    def __call__(self, healthy, rng):
        texture = image_tensor(read_image(self.paths[int(rng.integers(len(self.paths)))]), healthy.shape[-2:], 255.0)
        # Native torch augmentations avoid upstream imgaug/NumPy incompatibilities.
        operations = rng.choice(5, 3, replace=False)
        for operation in operations:
            if operation == 0:
                texture = texture.pow(float(rng.uniform(0.5, 2)))
            elif operation == 1:
                texture = (texture * float(rng.uniform(.8, 1.2)) + float(rng.uniform(-.1, .1))).clamp(0, 1)
            elif operation == 2:
                texture = 1 - texture
            elif operation == 3:
                texture = texture.flip(-1)
            else:
                texture = torch.where(texture > .5, 1-texture, texture)
        for _ in range(32):
            resolution = tuple(int(2**rng.integers(0, 6)) for _ in range(2))
            mask = torch.from_numpy((perlin_noise(healthy.shape[-2:], resolution, rng) > .5).astype(np.float32))[None]
            if mask.any():
                break
        else:
            raise RuntimeError("Could not synthesize a nonempty Perlin mask.")
        beta = float(rng.uniform(0, .8))
        image = healthy*(1-mask) + ((1-beta)*texture + beta*healthy)*mask
        return image, mask
