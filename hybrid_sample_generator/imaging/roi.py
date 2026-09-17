"""Dimension-independent region-of-interest sizing and cropping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_axis_tuple(value, ndim: int, name: str) -> tuple:
    if np.isscalar(value):
        return (value,) * ndim

    values = tuple(value)
    if len(values) < ndim:
        raise ValueError(f"{name} must have at least len {ndim}. Got {value!r}")
    return values[:ndim]


def dynamic_roi_size(
    spatial_shape: Sequence[int],
    min_roi_padding,
    roi_padding_ratio,
    min_roi_size,
) -> list[int]:
    """Return a per-axis ROI size derived from an anomaly's spatial shape."""
    spatial_shape = tuple(int(size) for size in spatial_shape)
    min_roi_padding = _as_axis_tuple(min_roi_padding, len(spatial_shape), "min_roi_padding")
    roi_padding_ratio = _as_axis_tuple(roi_padding_ratio, len(spatial_shape), "roi_padding_ratio")
    min_roi_size = _as_axis_tuple(min_roi_size, len(spatial_shape), "min_roi_size")

    return [
        max(int(size + max(axis_padding, size * axis_ratio)), int(axis_minimum))
        for size, axis_padding, axis_ratio, axis_minimum in zip(
            spatial_shape,
            min_roi_padding,
            roi_padding_ratio,
            min_roi_size,
        )
    ]


def crop_spatial_clip(
    arr: np.ndarray,
    centroid,
    size,
    *,
    centroid_is_normalized: bool | None = None,
) -> np.ndarray:
    """Crop a channel-first 2D or 3D ROI and keep it inside array bounds."""
    spatial_dims = arr.ndim - 1
    if spatial_dims not in (2, 3):
        raise ValueError(
            "crop_spatial_clip expects shape (C,H,W) or (C,D,H,W). "
            f"Got {arr.shape}"
        )
    expected_ndim = spatial_dims + 1
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"Expected a channel-first array with {expected_ndim} dimensions. Got {arr.shape}"
        )

    centroid = tuple(centroid)
    if len(centroid) == expected_ndim:
        centroid = centroid[1:]
    elif len(centroid) != spatial_dims:
        raise ValueError(
            f"centroid must have length {spatial_dims} or {expected_ndim}. Got {centroid}"
        )

    spatial_shape = arr.shape[1:]
    crop_size = tuple(int(axis) for axis in tuple(size)[-spatial_dims:])
    if len(crop_size) != spatial_dims:
        raise ValueError(f"size must contain at least {spatial_dims} values. Got {size!r}")

    if centroid_is_normalized is None:
        centroid_is_normalized = all(0.0 <= coordinate <= 1.2 for coordinate in centroid)
    if centroid_is_normalized:
        centroid = tuple(coordinate * axis_size for coordinate, axis_size in zip(centroid, spatial_shape))

    centers = tuple(int(round(coordinate)) for coordinate in centroid)
    slices = []
    for center, requested, available in zip(centers, crop_size, spatial_shape):
        start = center - requested // 2
        stop = start + requested
        if start < 0:
            stop -= start
            start = 0
        elif stop > available:
            start -= stop - available
            stop = available
        slices.append(slice(max(start, 0), min(stop, available)))

    return arr[(slice(None), *slices)]
