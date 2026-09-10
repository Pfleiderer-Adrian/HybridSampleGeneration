"""Normalization helpers used while extracting anomaly cutouts."""

from __future__ import annotations

import numpy as np


def normalize_anomaly(arr, normalization, eps):
    """Normalize a cutout and return the parameters needed for inversion."""
    if normalization is None or str(normalization).lower() in ("none", "null"):
        return arr, {"norm_type": None}

    norm = str(normalization).lower()
    if norm in ("zscore", "z-score", "z_score"):
        mean = float(np.mean(arr))
        std = max(float(np.std(arr)), eps)
        return (arr - mean) / std, {
            "norm_type": "zscore",
            "norm_mean": mean,
            "norm_std": std,
        }

    if norm in ("zscore_median", "z-score-median", "zscore-median"):
        median = float(np.median(arr))
        mad = max(float(np.median(np.abs(arr - median))), eps)
        return (arr - median) / mad, {
            "norm_type": "zscore_median",
            "norm_median": median,
            "norm_mad": mad,
        }

    raise ValueError(f"Unknown normalization: {normalization!r}")


def add_background_noise_floor(img, sigma_rel=0.003, eps=1e-8):
    """Add weak noise only to pixels equal to the image background value."""
    img = img.copy()
    background = img.min()
    background_mask = np.isclose(img, background, atol=eps)
    dynamic_range = img.max() - background
    sigma = sigma_rel * (dynamic_range + 1e-12)
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(img.dtype)
    img[background_mask] = background + noise[background_mask]
    return img
