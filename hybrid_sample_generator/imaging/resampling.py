"""Channel-first spatial resampling helpers."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from hybrid_sample_generator.imaging.masks.interpolation import interpolate_masked_regions


def spatial_target_size(target_size, spatial_dims: int) -> tuple[int, ...]:
    """Normalize a spatial or channel-plus-spatial target shape."""
    values = tuple(target_size)
    if len(values) == spatial_dims:
        return values
    if len(values) == spatial_dims + 1:
        return values[-spatial_dims:]
    raise ValueError(
        f"target_size must contain {spatial_dims} spatial values, optionally prefixed "
        f"by a channel size. Got {target_size}"
    )


def _resize_and_pad(arr, target_size, *, order: int, foreground_mask, spatial_dims: int):
    expected_ndim = spatial_dims + 1
    if arr.ndim != expected_ndim:
        raise ValueError(f"Expected a {expected_ndim}D channel-first array. Got {arr.shape}")

    target_size = spatial_target_size(target_size, spatial_dims)
    if foreground_mask is not None:
        foreground_mask = np.asarray(foreground_mask, dtype=bool)
        if foreground_mask.shape != arr.shape[1:]:
            raise ValueError(
                f"foreground_mask shape {foreground_mask.shape} does not match "
                f"array spatial shape {arr.shape[1:]}."
            )

    scale_spatial = tuple(
        min(target / source, 1.0)
        for source, target in zip(arr.shape[1:], target_size)
    )
    if any(scale < 1.0 for scale in scale_spatial):
        if order == 0 or foreground_mask is None:
            arr = zoom(arr, (1.0, *scale_spatial), order=order)
        else:
            arr = interpolate_masked_regions(
                arr,
                foreground_mask,
                warp=lambda spatial: zoom(spatial, scale_spatial, order=order),
                nearest_warp=lambda spatial: zoom(spatial, scale_spatial, order=0),
            )

    padding = []
    for current, target in zip(arr.shape[1:], target_size):
        total = max(target - current, 0)
        before = total // 2
        padding.append((before, total - before))

    padded = np.pad(
        arr,
        ((0, 0), *padding),
        mode="constant",
        constant_values=float(np.min(arr)),
    )
    crop = (slice(None), *(slice(0, target) for target in target_size))
    return padded[crop], scale_spatial


def resize_and_pad_2d(arr, target_size, order=1, foreground_mask=None):
    """Downscale and center-pad a ``(C, H, W)`` array."""
    if arr.ndim != 3:
        raise ValueError(f"resize_and_pad_2d expects 3D (C,h,w). Got {arr.shape}")
    return _resize_and_pad(
        arr,
        target_size,
        order=order,
        foreground_mask=foreground_mask,
        spatial_dims=2,
    )


def resize_and_pad_3d(arr, target_size, order=1, foreground_mask=None):
    """Downscale and center-pad a ``(C, D, H, W)`` array."""
    if arr.ndim != 4:
        raise ValueError(f"resize_and_pad_3d expects 4D (C,d,h,w). Got {arr.shape}")
    return _resize_and_pad(
        arr,
        target_size,
        order=order,
        foreground_mask=foreground_mask,
        spatial_dims=3,
    )
