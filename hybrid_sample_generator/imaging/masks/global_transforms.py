"""Whole-mask geometric transformations."""

import numpy as np

from hybrid_sample_generator.imaging.masks.geometry import (
    rotate_spatial_mask,
    sample_uniform,
    stretch_spatial_mask,
    validate_channel_first_mask,
)


def random_global_stretch_transform(
    mask_np: np.ndarray,
    min_stretch=1.0,
    max_stretch=1.2,
    rng=None,
):
    """Apply nearest-neighbour scaling around the mask center while preserving shape."""
    validate_channel_first_mask(mask_np)
    scales = sample_uniform(
        min_stretch, max_stretch, rng=rng, size=mask_np[0].ndim
    )
    transformed = stretch_spatial_mask(mask_np[0].copy(), scales=scales)
    return transformed.astype(mask_np.dtype)[None, ...]


def random_global_zoom_transform(
    mask_np: np.ndarray,
    min_zoom=0.9,
    max_zoom=0.9,
    rng=None,
):
    """Apply isotropic nearest-neighbour zoom around the mask center."""
    validate_channel_first_mask(mask_np)
    min_zoom = float(min_zoom)
    max_zoom = float(max_zoom)
    if not 0 < min_zoom <= max_zoom <= 1:
        raise ValueError(
            "min_zoom and max_zoom must satisfy "
            f"0 < min_zoom <= max_zoom <= 1, got ({min_zoom}, {max_zoom})."
        )
    zoom_factor = sample_uniform(min_zoom, max_zoom, rng=rng)
    scales = np.full(mask_np[0].ndim, zoom_factor, dtype=float)
    transformed = stretch_spatial_mask(mask_np[0].copy(), scales=scales)
    return transformed.astype(mask_np.dtype)[None, ...]


def random_global_rotation_transform(
    mask_np: np.ndarray,
    max_rotation=5.0,
    rng=None,
):
    """Apply a small nearest-neighbour rotation to the whole label mask."""
    validate_channel_first_mask(mask_np)
    angle = sample_uniform(max_value=max_rotation, rng=rng)
    transformed = rotate_spatial_mask(
        mask_np[0].copy(),
        angle=angle,
        center_mask=mask_np[0] != 0,
    )
    return transformed.astype(mask_np.dtype)[None, ...]
