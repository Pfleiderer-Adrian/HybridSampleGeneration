"""Composition of channel-first label masks."""

import numpy as np


def combine_label_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    overwrite: bool = True,
    return_dtype=None,
) -> np.ndarray:
    if not isinstance(mask_a, np.ndarray) or not isinstance(mask_b, np.ndarray):
        raise TypeError("mask_a and mask_b must be NumPy arrays.")
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shapes must match, got {mask_a.shape} vs {mask_b.shape}.")
    if mask_a.ndim not in (3, 4):
        raise ValueError(f"Expected channel-first 2D/3D masks, got {mask_a.shape}.")
    result = mask_a.copy()
    foreground = mask_b > 0
    if not overwrite:
        foreground &= result == 0
    result[foreground] = mask_b[foreground]
    return result if return_dtype is None else result.astype(return_dtype, copy=False)


__all__ = ["combine_label_masks"]
