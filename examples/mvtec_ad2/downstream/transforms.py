"""Aligned image, reconstruction target, and binary mask preparation."""

import numpy as np
import torch
from torch.nn import functional as F
from examples.common.image_io import ensure_chw, load_image_array


def read_image(path):
    return ensure_chw(load_image_array(path))


def image_tensor(array, size, scale):
    value = torch.as_tensor(np.array(array, copy=True, order="C"), dtype=torch.float32)
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError(f"Expected grayscale/RGB CHW image, got {value.shape}.")
    if not torch.isfinite(value).all():
        raise ValueError("Images must be finite.")
    value = (value / scale).clamp(0, 1)
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    return value if size is None else F.interpolate(value[None], size=size, mode="bilinear", align_corners=False)[0]


def mask_tensor(array, size):
    value = torch.as_tensor(np.array(array, copy=True, order="C"), dtype=torch.float32)
    if value.ndim != 3:
        raise ValueError("Expected CHW mask.")
    if not torch.isfinite(value).all():
        raise ValueError("Masks must be finite.")
    value = (value > 0).any(dim=0, keepdim=True).float()
    return value if size is None else F.interpolate(value[None], size=size, mode="nearest")[0]


def aligned_flips(image, target, mask, rng):
    for axis in (-1, -2):
        if rng.random() < 0.5:
            image, target, mask = (value.flip(axis) for value in (image, target, mask))
    return image, target, mask
