"""Shared loading, shape conversion, normalization, and saving for 2D images."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def is_image_file(path: str | Path) -> bool:
    return os.path.splitext(os.fspath(path))[1].lower() in IMAGE_EXTENSIONS


def load_image_array(path: str | Path) -> np.ndarray:
    """Load an image as a grayscale (H, W) or color (H, W, C) array."""
    with Image.open(path) as image:
        if image.mode == "P":
            image = image.convert("L")
        return np.asarray(image)


def ensure_chw(array: np.ndarray) -> np.ndarray:
    """Convert a common 2D image representation to channel-first (C, H, W)."""
    array = np.asarray(array)
    if array.ndim == 2:
        height, width = array.shape
        return array.reshape(1, height, width)
    if array.ndim == 3:
        if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
            return array
        return array.transpose(2, 0, 1)
    raise ValueError(f"Unsupported ndim={array.ndim}, shape={array.shape}")


def minmax01_chw(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each channel of a (C, H, W) array to [0, 1]."""
    image = array.astype(np.float32, copy=False)
    normalized = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        values = image[channel]
        minimum = np.nanmin(values)
        maximum = np.nanmax(values)
        value_range = maximum - minimum
        if not np.isfinite(value_range) or value_range < eps:
            normalized[channel] = 0.0
        else:
            normalized[channel] = (values - minimum) / (value_range + eps)
    return normalized


def save_image(
    array_chw: np.ndarray,
    filepath: str | Path,
    *,
    clamp: bool = True,
    quality: int = 95,
) -> None:
    """Save a channel-first grayscale, RGB, or RGBA image."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(array_chw)
    if array.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape={array.shape}")

    channels = array.shape[0]
    if channels not in (1, 3, 4):
        raise ValueError(
            f"Unsupported channel count C={channels}. Expected 1, 3, or 4."
        )

    image = array[0] if channels == 1 else array.transpose(1, 2, 0)
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        if maximum <= 1.0:
            image = image * 255.0
        if clamp:
            image = np.clip(image, 0.0, 255.0)
    elif clamp:
        image = np.clip(image, 0, 255)
    image_u8 = image.astype(np.uint8)

    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
    pil_image = Image.fromarray(image_u8, mode=mode)
    extension = filepath.suffix.lower()
    save_kwargs: dict[str, int] = {}
    if extension in (".jpg", ".jpeg"):
        if channels == 4:
            pil_image = pil_image.convert("RGB")
        save_kwargs = {"quality": int(quality), "subsampling": 0}

    pil_image.save(filepath, **save_kwargs)
