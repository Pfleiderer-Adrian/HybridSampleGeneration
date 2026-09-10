"""Shared geometry and sampling helpers for label-mask transforms."""

import numpy as np
import scipy.ndimage as ndi


DEFAULT_PADDING_MODE = "constant"


def sample_uniform(min_value=None, max_value=None, *, rng=None, size=None, integer=False):
    """Sample from a uniform range. With only max_value, use [-max_value, max_value]."""
    if max_value is None:
        if min_value is None:
            raise ValueError("sample_uniform requires min_value or max_value.")
        max_value = min_value
        min_value = -max_value
    elif min_value is None:
        min_value = -max_value

    rng = rng if rng is not None else np.random.default_rng()
    if integer:
        low = int(min_value)
        high = int(max_value)
        if low > high:
            raise ValueError(
                f"sample_uniform integer range must be ordered, got ({low}, {high})."
            )
        sample = rng.integers(low, high + 1, size=size)
        if size is None:
            return int(sample)
        return sample

    sample = rng.uniform(float(min_value), float(max_value), size=size)
    if size is None:
        return float(sample)
    return sample


def stretch_spatial_mask(mask, scales):
    """Scale a spatial mask around its center while preserving its shape."""
    inv_scales = 1.0 / np.array(scales)
    matrix = np.diag(inv_scales)
    center = np.array(mask.shape) / 2.0
    offset = center - np.dot(matrix, center)
    return ndi.affine_transform(
        mask,
        matrix=matrix,
        offset=offset,
        output_shape=mask.shape,
        order=0,
        mode=DEFAULT_PADDING_MODE,
        cval=0,
    )


def rotate_spatial_mask(mask, angle, center_mask=None):
    """Rotate the final two axes around the selected mask's center."""
    if mask.ndim < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got {mask.ndim}.")
    if not np.any(center_mask):
        return mask.copy()

    coords = np.where(center_mask)
    center = np.array(
        [(np.min(axis_coords) + np.max(axis_coords)) / 2.0 for axis_coords in coords],
        dtype=float,
    )
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=float,
    )
    matrix = np.eye(mask.ndim, dtype=float)
    matrix[-2:, -2:] = rotation_matrix
    offset = np.zeros(mask.ndim, dtype=float)
    offset[-2:] = center[-2:] - rotation_matrix @ center[-2:]
    return ndi.affine_transform(
        mask,
        matrix=matrix,
        offset=offset,
        output_shape=mask.shape,
        order=0,
        mode=DEFAULT_PADDING_MODE,
        cval=0,
        prefilter=False,
    )


def pad_mask_for_transforms(mask_np, padding_factor=2):
    """Center a channel-first mask on a larger zero-filled transform canvas."""
    validate_channel_first_mask(mask_np)
    spatial_shape = np.asarray(mask_np.shape[1:], dtype=int)
    padded_shape = spatial_shape * int(padding_factor)
    total_padding = padded_shape - spatial_shape
    pad_width = [(0, 0)] + [
        (int(padding // 2), int(padding - padding // 2))
        for padding in total_padding
    ]
    return np.pad(mask_np, pad_width, mode="constant", constant_values=0)


def fit_mask_to_spatial_shape(mask_np, target_shape):
    """Minimally zoom out around the canvas center, then restore target_shape."""
    target_shape = np.asarray(target_shape, dtype=int)
    spatial_shape = np.asarray(mask_np.shape[1:], dtype=int)
    if target_shape.shape != spatial_shape.shape or np.any(target_shape <= 0):
        raise ValueError(
            f"Invalid target spatial shape {tuple(target_shape)} for mask {mask_np.shape}."
        )
    if np.any(target_shape > spatial_shape):
        raise ValueError(
            f"Target shape {tuple(target_shape)} exceeds transform canvas {tuple(spatial_shape)}."
        )

    crop_start = (spatial_shape - target_shape) // 2
    crop_end = crop_start + target_shape - 1
    spatial_mask = mask_np[0]
    foreground = spatial_mask != 0
    if np.any(foreground):
        center = spatial_shape.astype(float) / 2.0
        fit_scale = 1.0
        for axis, axis_coords in enumerate(np.where(foreground)):
            min_coord = float(np.min(axis_coords))
            max_coord = float(np.max(axis_coords))
            if min_coord < crop_start[axis]:
                fit_scale = min(
                    fit_scale,
                    (center[axis] - crop_start[axis]) / (center[axis] - min_coord),
                )
            if max_coord > crop_end[axis]:
                fit_scale = min(
                    fit_scale,
                    (crop_end[axis] - center[axis]) / (max_coord - center[axis]),
                )
        if fit_scale < 1.0:
            fit_scale = max(np.nextafter(fit_scale, 0.0), np.finfo(float).eps)
            spatial_mask = stretch_spatial_mask(
                spatial_mask,
                scales=np.full(spatial_mask.ndim, fit_scale, dtype=float),
            ).astype(mask_np.dtype)

    crop_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(crop_start, target_shape)
    )
    return spatial_mask[crop_slices][None, ...]


def validate_channel_first_mask(mask_np):
    """Validate the supported single-channel 2D and 3D mask layouts."""
    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(
            "Expected mask with shape (1, H, W) or (1, D, H, W), "
            f"got {mask_np.shape}."
        )
