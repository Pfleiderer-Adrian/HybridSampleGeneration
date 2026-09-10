"""Shared normalization and spatial validation for fusion backends."""

import numpy as np


def inverse_extraction_scale(scale_factor, ndim):
    """Return the spatial zoom that restores an extraction resize."""
    if np.isscalar(scale_factor):
        scale = np.full(ndim, float(scale_factor), dtype=np.float32)
    else:
        scale = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
        if scale.size != ndim:
            raise ValueError(f"scale_factor must have len {ndim}. Got {scale_factor!r}")
    if np.any(scale <= 0):
        raise ValueError(f"scale_factor values must be > 0. Got {scale_factor!r}")
    return tuple(float(1.0 / value) for value in scale)


def denormalize_anomaly(anomaly, normalization_meta):
    if not normalization_meta:
        return anomaly
    norm_type = normalization_meta.get("norm_type")
    if norm_type == "zscore":
        mean = normalization_meta.get("norm_mean")
        std = normalization_meta.get("norm_std")
        return anomaly if mean is None or std is None else anomaly * float(std) + float(mean)
    if norm_type == "zscore_median":
        median = normalization_meta.get("norm_median")
        mad = normalization_meta.get("norm_mad")
        return anomaly if median is None or mad is None else anomaly * float(mad) + float(median)
    return anomaly


def spatial_label_mask(mask, spatial_ndim):
    mask = np.asarray(mask)
    if mask.ndim == spatial_ndim:
        return mask
    if mask.ndim == spatial_ndim + 1:
        return np.max(mask, axis=0)
    raise ValueError(
        f"target mask must have {spatial_ndim} or {spatial_ndim + 1} dims. Got {mask.shape}"
    )


def validate_position(position, spatial_ndim):
    axes = "(H,W)" if spatial_ndim == 2 else "(D,H,W)"
    if position is None:
        raise ValueError(
            f"position must be provided (expected len {spatial_ndim}: {axes})."
        )
    position = list(position)
    if len(position) != spatial_ndim:
        raise ValueError(
            f"position must have len {spatial_ndim} {axes}. Got {position!r}"
        )
    return tuple(float(value) for value in position)


__all__ = ["denormalize_anomaly", "inverse_extraction_scale", "spatial_label_mask", "validate_position"]
