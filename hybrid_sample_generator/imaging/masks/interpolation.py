"""Foreground-aware interpolation for channel-first images and volumes."""

import numpy as np


def interpolate_masked_regions(
    image_np,
    foreground_mask,
    warp,
    nearest_warp,
    *,
    interpolate_background=True,
    background_fill=None,
    eps=None,
):
    """Warp an image without mixing foreground and background values."""
    image_np = np.asarray(image_np)
    foreground_mask = np.asarray(foreground_mask, dtype=bool)
    if image_np.ndim not in (3, 4):
        raise ValueError(
            f"Expected image shape (C,H,W) or (C,D,H,W), got {image_np.shape}."
        )
    if foreground_mask.shape != image_np.shape[1:]:
        raise ValueError(
            f"foreground_mask shape {foreground_mask.shape} does not match "
            f"image spatial shape {image_np.shape[1:]}."
        )

    eps = np.finfo(np.float32).eps if eps is None else float(eps)
    foreground_support = foreground_mask.astype(np.float32)
    foreground_weights = warp(foreground_support)
    foreground_values = np.stack(
        [warp(channel * foreground_support) for channel in image_np], axis=0
    )
    normalized_foreground = np.zeros_like(foreground_values)
    valid_foreground = foreground_weights > eps
    normalized_foreground[:, valid_foreground] = (
        foreground_values[:, valid_foreground]
        / foreground_weights[valid_foreground][None, ...]
    )
    transformed_foreground = nearest_warp(foreground_support) != 0

    if interpolate_background:
        background_support = 1.0 - foreground_support
        background_weights = warp(background_support)
        background_values = np.stack(
            [warp(channel * background_support) for channel in image_np], axis=0
        )
        result = np.zeros_like(background_values)
        valid_background = background_weights > eps
        result[:, valid_background] = (
            background_values[:, valid_background]
            / background_weights[valid_background][None, ...]
        )
    else:
        valid_background = np.zeros_like(transformed_foreground, dtype=bool)
        result = np.zeros_like(foreground_values)

    if background_fill is None:
        fill_values = []
        for channel in image_np:
            finite_values = channel[~foreground_mask]
            finite_values = finite_values[np.isfinite(finite_values)]
            fill_values.append(
                float(np.min(finite_values)) if finite_values.size else 0.0
            )
        fill_values = np.asarray(fill_values, dtype=result.dtype)
    elif np.isscalar(background_fill):
        fill_values = np.full(image_np.shape[0], background_fill, dtype=result.dtype)
    else:
        fill_values = np.asarray(background_fill, dtype=result.dtype)
        if fill_values.shape != (image_np.shape[0],):
            raise ValueError(
                f"background_fill must be scalar or have shape ({image_np.shape[0]},), "
                f"got {fill_values.shape}."
            )

    invalid_background = ~valid_background & ~transformed_foreground
    result[:, invalid_background] = fill_values[:, None]
    result[:, transformed_foreground] = normalized_foreground[:, transformed_foreground]
    return result.astype(image_np.dtype, copy=False)


__all__ = ["interpolate_masked_regions"]
