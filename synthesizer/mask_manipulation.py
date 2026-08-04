from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

def to_one_hot_3D(mask: torch.Tensor, num_anomaly_classes: int) -> torch.Tensor:
    """Converts 3D/4D/5D integer masks to 5D one-hot float tensors of shape (B, C, D, H, W)."""
    
    # already 5D and one hot encoded (more than one Channel)
    if mask.ndim == 5 and mask.shape[1] > 1:
        return mask.float()
        
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask.squeeze(1) # -> (B, D, H, W)

    # missing batch dim
    if mask.ndim == 3:
        mask = mask.unsqueeze(0) # -> (1, D, H, W)
        
    # mask must be (B, D, H, W) here
    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape (B, D, H, W) after cleanup, got: {mask.shape}.")
        
    mask = mask.long()
    
    num_classes = num_anomaly_classes + 1
    # (B, D, H, W) -> (B, D, H, W, num_classes)
    mask_oh = F.one_hot(mask, num_classes=num_classes)
    
    # remove class 0 channel (background channel)
    mask_oh = mask_oh[..., 1:] 
        
    # (B, D, H, W, C) -> (B, C, D, H, W)
    return mask_oh.permute(0, 4, 1, 2, 3).float()

def to_one_hot_2D(mask: torch.Tensor, num_anomaly_classes: int) -> torch.Tensor:
    """Converts 2D/3D/4D integer masks to 4D one-hot float tensors of shape (B, C, H, W)."""

    # already 4D and one hot encoded without background channel
    if mask.ndim == 4 and mask.shape[1] == num_anomaly_classes:
        return mask.float()

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)  # -> (B, H, W)

    # missing batch dim
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # -> (1, H, W)

    # mask must be (B, H, W) here
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape (B, H, W) after cleanup, got: {mask.shape}.")

    mask = mask.long()

    num_classes = num_anomaly_classes + 1
    # (B, H, W) -> (B, H, W, num_classes)
    mask_oh = F.one_hot(mask, num_classes=num_classes)

    # remove class 0 channel (background channel)
    mask_oh = mask_oh[..., 1:]

    # (B, H, W, C) -> (B, C, H, W)
    return mask_oh.permute(0, 3, 1, 2).float()


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
            raise ValueError(f"sample_uniform integer range must be ordered, got ({low}, {high}).")
        sample = rng.integers(low, high + 1, size=size)
        if size is None:
            return int(sample)
        return sample

    sample = rng.uniform(float(min_value), float(max_value), size=size)
    if size is None:
        return float(sample)
    return sample


def random_global_stretch_transform(
    mask_np: np.ndarray,
    min_stretch=1.0,
    max_stretch=1.2,
    rng=None,
    order=0,
):
    """Apply nearest-neighbour scaling around the mask center while preserving shape."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    transformed_mask = mask_np[0].copy()
    scales = sample_uniform(min_stretch, max_stretch, rng=rng, size=transformed_mask.ndim)
    transformed_mask = _stretch_spatial(
        transformed_mask,
        scales=scales,
        order=order,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


def random_global_zoom_transform(
    mask_np: np.ndarray,
    min_zoom=0.9,
    max_zoom=0.9,
    rng=None,
    order=0,
):
    """Apply isotropic nearest-neighbour zoom around the mask center while preserving shape."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    min_zoom = float(min_zoom)
    max_zoom = float(max_zoom)
    if not 0 < min_zoom <= max_zoom <= 1:
        raise ValueError(f"min_zoom and max_zoom must satisfy 0 < min_zoom <= max_zoom <= 1, got ({min_zoom}, {max_zoom}).")

    zoom_factor = sample_uniform(min_zoom, max_zoom, rng=rng)
    transformed_mask = mask_np[0].copy()
    scales = np.full(transformed_mask.ndim, zoom_factor, dtype=float)
    transformed_mask = _stretch_spatial(
        transformed_mask,
        scales=scales,
        order=order,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


def _stretch_spatial(mask, scales, order=0):
    inv_scales = 1.0 / np.array(scales)
    matrix = np.diag(inv_scales)

    center = np.array(mask.shape) / 2.0
    offset = center - np.dot(matrix, center)

    return ndi.affine_transform(
        mask,
        matrix=matrix,
        offset=offset,
        output_shape=mask.shape,
        order=order,
        mode=DEFAULT_PADDING_MODE,
        cval=0,
    )


def _rotation_center(mask_np: np.ndarray | None) -> np.ndarray | None:
    """Return the bounding-box center of a spatial mask without a channel axis."""
    if mask_np is None:
        return None

    foreground = mask_np != 0
    if not np.any(foreground):
        return None
    return np.array(
        [
            (float(axis_coords.min()) + float(axis_coords.max())) / 2.0
            for axis_coords in np.where(foreground)
        ],
        dtype=float,
    )


def _rotate_spatial(mask, angle, center_mask=None, center=None, order=0):
    if mask.ndim < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got {mask.ndim}.")

    if center is None:
        center = _rotation_center(center_mask)
        if center is None:
            return mask.copy()
    else:
        center = np.asarray(center, dtype=float)
        if center.shape != (mask.ndim,):
            raise ValueError(
                f"Expected rotation center with shape ({mask.ndim},), got {center.shape}."
            )

    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation_matrix = np.array(
        [
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle],
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
        order=order,
        mode=DEFAULT_PADDING_MODE,
        cval=0,
        prefilter=order > 1,
    )


def random_global_rotation_transform(
    mask_np: np.ndarray,
    max_rotation=5.0,
    rng=None,
    center=None,
    order=0,
):
    """Apply a small nearest-neighbour rotation to the whole label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    angle = sample_uniform(max_value=max_rotation, rng=rng)
    transformed_mask = _rotate_spatial(
        mask_np[0].copy(),
        angle=angle,
        center_mask=mask_np[0] != 0,
        center=center,
        order=order,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


def random_local_stretch_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None):
    """Stretch selected classes in a 2D or 3D channel-first label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()
    spatial_ndim = transformed_mask.ndim

    if classes is None:
        classes = [cls for cls in np.unique(transformed_mask) if cls != 0]
    if priorities is None:
        priorities = classes

    scales = sample_uniform(
        params.get("min_stretch", 0.95),
        params.get("max_stretch", 1.05),
        rng=rng,
        size=spatial_ndim,
    )
    class_masks = {}

    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask):
            binary_mask = _stretch_spatial(
                binary_mask,
                scales=scales,
            ).astype(bool)
        class_masks[cls] = binary_mask

    final_mask = transformed_mask.copy()
    for cls in classes:
        final_mask[final_mask == cls] = 0
    for cls in reversed(priorities):
        if cls in class_masks:
            final_mask[class_masks[cls]] = cls

    final_mask = final_mask.astype(original_dtype)

    return final_mask[None, ...]


def random_local_rotation_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None):
    """Rotate selected classes in a 2D or 3D channel-first label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()

    if classes is None:
        classes = [cls for cls in np.unique(transformed_mask) if cls != 0]
    if priorities is None:
        priorities = classes

    angle = sample_uniform(max_value=params.get("max_rotation", 5.0), rng=rng)
    class_masks = {}

    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask):
            binary_mask = _rotate_spatial(
                binary_mask,
                angle=angle,
                center_mask=binary_mask,
            ).astype(bool)
        class_masks[cls] = binary_mask

    final_mask = transformed_mask.copy()
    for cls in classes:
        final_mask[final_mask == cls] = 0
    for cls in reversed(priorities):
        if cls in class_masks:
            final_mask[class_masks[cls]] = cls

    final_mask = final_mask.astype(original_dtype)

    return final_mask[None, ...]


def random_local_dilate_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None):
    """Dilate selected classes in a 2D or 3D channel-first label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()

    if classes is None:
        classes = [cls for cls in np.unique(transformed_mask) if cls != 0]
    if priorities is None:
        priorities = classes

    class_masks = {}
    iterations = sample_uniform(
        params.get("min_iterations", 0),
        params.get("max_iterations", 2),
        rng=rng,
        integer=True,
    )

    for cls in classes:
        binary_mask = transformed_mask == cls

        if np.any(binary_mask) and iterations > 0:
            binary_mask = ndi.binary_dilation(binary_mask, iterations=iterations)

        class_masks[cls] = binary_mask

    final_mask = transformed_mask.copy()
    for cls in classes:
        final_mask[final_mask == cls] = 0
    for cls in reversed(priorities):
        if cls in class_masks:
            final_mask[class_masks[cls]] = cls

    final_mask = final_mask.astype(original_dtype)
    
    return final_mask[None, ...]


def _as_axis_tuple(value, ndim, name):
    if np.isscalar(value):
        return (float(value),) * ndim
    if len(value) != ndim:
        raise ValueError(f"{name} must be scalar or contain exactly {ndim} values, got {value!r}.")
    return tuple(float(v) for v in value)


def random_elastic_transform(
    mask_np: np.ndarray,
    sigma=30,
    magnitude=20,
    rng=None,
    order=0,
):
    """Apply a smooth random displacement field to a 2D or 3D channel-first label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    transformed_mask = mask_np[0].copy()
    spatial_ndim = transformed_mask.ndim
    sigma = _as_axis_tuple(sigma, spatial_ndim, "sigma")
    magnitude = _as_axis_tuple(magnitude, spatial_ndim, "magnitude")
    rng = rng if rng is not None else np.random.default_rng()

    coordinates = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in transformed_mask.shape],
        indexing="ij",
    )

    displaced_coordinates = []
    for axis in range(spatial_ndim):
        random_field = rng.uniform(-1.0, 1.0, size=transformed_mask.shape).astype(np.float32)
        smooth_field = ndi.gaussian_filter(random_field, sigma=sigma, mode="reflect")

        max_abs = np.max(np.abs(smooth_field))
        if max_abs > 0:
            smooth_field = smooth_field / max_abs

        displacement = smooth_field * magnitude[axis]
        displaced_coordinates.append(coordinates[axis] + displacement)

    transformed_mask = ndi.map_coordinates(
        transformed_mask,
        displaced_coordinates,
        order=order,
        mode=DEFAULT_PADDING_MODE,
        cval=0, # bg value for constant padding (for masks)
        prefilter=order > 1,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


def random_local_elastic_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None):
    """Apply elastic deformation to selected classes in a 2D or 3D channel-first label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()

    if classes is None:
        classes = [cls for cls in np.unique(transformed_mask) if cls != 0]
    if priorities is None:
        priorities = classes

    class_masks = {}
    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask):
            binary_mask = random_elastic_transform(
                binary_mask[None, ...],
                sigma=params.get("sigma", 30),
                magnitude=params.get("magnitude", 20),
                rng=rng,
            )[0].astype(bool)
        class_masks[cls] = binary_mask

    final_mask = transformed_mask.copy()
    for cls in classes:
        final_mask[final_mask == cls] = 0
    for cls in reversed(priorities):
        if cls in class_masks:
            final_mask[class_masks[cls]] = cls

    final_mask = final_mask.astype(original_dtype)

    return final_mask[None, ...]


DEFAULT_PADDING_MODE = "constant"

DEFAULT_TRANSFORM_PROBS = {
    "zoom": 1,
    "stretch": 1,
    "rotate": 0,
    "elastic": 1,
    "local_dilate": 0,
    "local_stretch": 0,
    "local_rotate": 0,
    "local_elastic": 0,
}

DEFAULT_TRANSFORM_PARAMS = {
    "zoom": {
        "min_zoom": 0.9,
        "max_zoom": 0.9,
    },
    "stretch": {
        "min_stretch": 1.0,
        "max_stretch": 1.2,
    },
    "rotate": {
        "max_rotation": 5.0,
    },
    "elastic": {
        "sigma": 30,
        "magnitude": 20,
    },
    "local_dilate": {
        "min_iterations": 0,
        "max_iterations": 2,
    },
    "local_stretch": {
        "min_stretch": 0.95,
        "max_stretch": 1.05,
    },
    "local_rotate": {
        "max_rotation": 5.0,
    },
    "local_elastic": {
        "sigma": 30,
        "magnitude": 20,
    },
}

# if the minimum/neutral parameter value is not 0 it needs to be added here
LOCAL_PARAM_NEUTRAL_VALUES = {
    "min_stretch": 1,
    "max_stretch": 1,
}


def default_elastic_params_from_anomaly_size(anomaly_size):
    """Derive conservative elastic defaults from a channel-first anomaly size."""
    if anomaly_size is None:
        return {}

    spatial_shape = tuple(int(size) for size in anomaly_size)
    if len(spatial_shape) in (3, 4):
        spatial_shape = spatial_shape[1:]

    if len(spatial_shape) not in (2, 3) or any(size <= 0 for size in spatial_shape):
        return {}

    sigma = tuple(max(2, int(round(size * 0.2))) for size in spatial_shape)
    magnitude = tuple(max(1, int(round(size * 0.2))) for size in spatial_shape)

    return {
        "sigma": sigma,
        "magnitude": magnitude,
    }



def _pad_for_transforms(array_np, padding_factor=2):
    """Center a channel-first array on a larger zero-filled transform canvas."""
    if array_np.ndim not in (3, 4):
        raise ValueError(
            f"Expected channel-first array with shape (C, H, W) or (C, D, H, W), "
            f"got {array_np.shape}."
        )

    spatial_shape = np.asarray(array_np.shape[1:], dtype=int)
    padded_shape = spatial_shape * int(padding_factor)
    total_padding = padded_shape - spatial_shape
    pad_width = [(0, 0)] + [
        (int(padding // 2), int(padding - padding // 2))
        for padding in total_padding
    ]
    return np.pad(array_np, pad_width, mode="constant", constant_values=0)


def _fit_mask_to_spatial_shape(mask_np, target_shape):
    """Minimally zoom out around the canvas center, then restore target_shape."""
    target_shape = np.asarray(target_shape, dtype=int)
    spatial_shape = np.asarray(mask_np.shape[1:], dtype=int)
    if target_shape.shape != spatial_shape.shape or np.any(target_shape <= 0):
        raise ValueError(f"Invalid target spatial shape {tuple(target_shape)} for mask {mask_np.shape}.")
    if np.any(target_shape > spatial_shape):
        raise ValueError(f"Target shape {tuple(target_shape)} exceeds transform canvas {tuple(spatial_shape)}.")

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
            # Stay just inside the crop despite nearest-neighbour boundary rounding.
            fit_scale = max(np.nextafter(fit_scale, 0.0), np.finfo(float).eps)
            spatial_mask = _stretch_spatial(
                spatial_mask,
                scales=np.full(spatial_mask.ndim, fit_scale, dtype=float),
            ).astype(mask_np.dtype)

    crop_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(crop_start, target_shape)
    )
    return spatial_mask[crop_slices][None, ...]

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
    """Warp a channel-first image without mixing foreground and background."""
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
            values = channel[~foreground_mask]
            finite_values = values[np.isfinite(values)]
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


def _fit_image_like_mask(image_np, transformed_mask, target_shape, image_interpolation_order):
    """Mirror _fit_mask_to_spatial_shape for a channel-first image."""
    target_shape = np.asarray(target_shape, dtype=int)
    spatial_shape = np.asarray(transformed_mask.shape[1:], dtype=int)
    crop_start = (spatial_shape - target_shape) // 2
    crop_end = crop_start + target_shape - 1
    foreground = transformed_mask[0] != 0
    fit_scale = 1.0
    if np.any(foreground):
        center = spatial_shape.astype(float) / 2.0
        for axis, coords in enumerate(np.where(foreground)):
            low, high = float(coords.min()), float(coords.max())
            if low < crop_start[axis]:
                fit_scale = min(fit_scale, (center[axis] - crop_start[axis]) / (center[axis] - low))
            if high > crop_end[axis]:
                fit_scale = min(fit_scale, (crop_end[axis] - center[axis]) / (high - center[axis]))
    if fit_scale < 1.0:
        fit_scale = max(np.nextafter(fit_scale, 0.0), np.finfo(float).eps)
        scales = np.full(len(spatial_shape), fit_scale, dtype=float)
        if image_interpolation_order == 0:
            image_np = np.stack(
                [_stretch_spatial(channel, scales) for channel in image_np],
                axis=0,
            ).astype(image_np.dtype, copy=False)
        else:
            image_np = interpolate_masked_regions(
                image_np,
                foreground,
                warp=lambda spatial: _stretch_spatial(
                    spatial, scales, order=image_interpolation_order
                ),
                nearest_warp=lambda spatial: _stretch_spatial(
                    spatial, scales, order=0
                ),
            )
    crop = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(crop_start, target_shape)
    )
    return image_np[(slice(None), *crop)]

class TransformGenerator:
    """Central orchestration object for mask augmentation."""

    @classmethod
    def from_config(cls, config):
        transform_config = getattr(config, "transform_config", None)
        return cls(
            getattr(transform_config, "mask_transform_probs", None),
            use_mask_transform=getattr(transform_config, "use_mask_transform", True),
            padding_factor=getattr(transform_config, "padding_factor", 2),
            transform_params=getattr(transform_config, "mask_transform_params", None),
            priorities=getattr(
                transform_config,
                "priorities",
                getattr(transform_config, "mask_transform_priorities", None),
            ),
            rng=getattr(transform_config, "rng", None) or getattr(config, "rng", None),
            anomaly_size=getattr(config, "anomaly_size", None),
            background_threshold=getattr(config, "background_threshold", 0.01),
            mask_transform_local_as_global=getattr(
                transform_config,
                "local_as_global",
                getattr(transform_config, "mask_transform_local_as_global", False),
            ),
            transform_image_for_posterior_generation=getattr(transform_config, "transform_image_for_posterior_generation", True),
            image_interpolation_order=getattr(transform_config, "image_interpolation_order", 1),
        )

    @dataclass
    class Config:
        use_mask_transform: bool = True
        mask_transform_probs: Dict[int | str, Any] = field(default_factory=dict)
        mask_transform_params: Dict[int | str, Dict[str, Any]] = field(default_factory=dict)
        priorities: list[int] | tuple[int, ...] | None = None
        rng: np.random.Generator | None = None
        local_as_global: bool = False
        transform_image_for_posterior_generation: bool = True
        image_interpolation_order: int = 1
        padding_factor: int = 2

        def setGlobalParam(self, transform_name: str, probability=None, **params):
            if transform_name not in TransformGenerator.GLOBAL_TRANSFORMS:
                raise ValueError(f"{transform_name!r} is not a global transform.")
            return self._set_transform_config(transform_name, probability, params)

        def setClassParam(self, class_id: int, transform_name: str, probability=None, **params):
            if transform_name not in TransformGenerator.LOCAL_TRANSFORMS:
                raise ValueError(f"{transform_name!r} is not a local transform.")
            return self._set_transform_config(transform_name, probability, params, class_id=class_id)

        def setAllClassParams(self, transform_name: str, probability=None, **params):
            if transform_name not in TransformGenerator.LOCAL_TRANSFORMS:
                raise ValueError(f"{transform_name!r} is not a local transform.")
            return self._set_transform_config(transform_name, probability, params)

        def _set_transform_config(self, transform_name: str, probability, params: dict, class_id: int | None = None):
            if probability is not None:
                if class_id is None:
                    self.mask_transform_probs[transform_name] = probability
                else:
                    self.mask_transform_probs.setdefault(class_id, {})[transform_name] = probability

            if params:
                if class_id is None:
                    self.mask_transform_params.setdefault(transform_name, {}).update(params)
                else:
                    self.mask_transform_params.setdefault(class_id, {}).setdefault(transform_name, {}).update(params)

            return self

    GLOBAL_TRANSFORMS = {
        "zoom": random_global_zoom_transform,
        "elastic": random_elastic_transform,
        "stretch": random_global_stretch_transform,
        "rotate": random_global_rotation_transform,
    }
    LOCAL_TRANSFORMS = {
        "local_dilate": random_local_dilate_transform,
        "local_stretch": random_local_stretch_transform,
        "local_rotate": random_local_rotation_transform,
        "local_elastic": random_local_elastic_transform,
    }
    LOCAL_AS_GLOBAL_TRANSFORMS = {
        "local_stretch": "stretch",
        "local_rotate": "rotate",
        "local_elastic": "elastic",
    }

    def __init__(
        self,
        transform_probs: Dict[int | str, Any] | None = None,
        *,
        use_mask_transform: bool = False,
        padding_factor: int = 2,
        transform_params: Dict[int | str, Dict[str, Any]] | None = None,
        priorities: list[int] | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
        anomaly_size: tuple[int, ...] | list[int] | None = None,
        background_threshold: float | None = 0.01,
        mask_transform_local_as_global: bool = False,
        transform_image_for_posterior_generation: bool = True,
        image_interpolation_order: int = 1,
    ) -> None:
        self.global_transform_probs = {}
        self.local_transform_probs = {}
        self.class_transform_probs = {}
        self.padding_factor = padding_factor
        if use_mask_transform:
            self.set_transform_probs(DEFAULT_TRANSFORM_PROBS)
        if transform_probs:
            self.set_transform_probs(transform_probs)
        self.transform_params = deepcopy(DEFAULT_TRANSFORM_PARAMS)
        if use_mask_transform:
            self.transform_params["elastic"].update(default_elastic_params_from_anomaly_size(anomaly_size))
        self.class_transform_params = {}
        self.priorities = priorities
        if transform_params:
            self.set_transform_params(transform_params)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.background_threshold = background_threshold
        self.mask_transform_local_as_global = mask_transform_local_as_global
        self.transform_image_for_posterior_generation = transform_image_for_posterior_generation
        self.image_interpolation_order = image_interpolation_order

    def create_target_mask(
        self,
        *,
        synth_anomaly_image=None,
        original_mask=None,
        target_mask=None,
        conditional: bool = False,
    ):
        if target_mask is not None:
            return target_mask
        if conditional:
            return self.create_target_mask_from_original_mask(original_mask)
        return self.create_target_mask_from_synth_anomaly(synth_anomaly_image)

    def create_target_mask_from_original_mask(self, original_mask):
        if original_mask is None:
            raise ValueError("original_mask is required for conditional target-mask generation.")

        if torch.is_tensor(original_mask):
            device = original_mask.device
            dtype = original_mask.dtype
            augmented = self.augment_mask(original_mask.detach().cpu().numpy())
            return torch.as_tensor(augmented, device=device, dtype=dtype)

        return self.augment_mask(np.asarray(original_mask))

    def create_target_mask_and_transformed_image(self, original_mask, image):
        """Apply exactly the same sampled generation-time transforms to mask and image."""
        mask_tensor = torch.is_tensor(original_mask)
        image_tensor = torch.is_tensor(image)
        mask_np = original_mask.detach().cpu().numpy() if mask_tensor else np.asarray(original_mask)
        image_device = image.device if image_tensor else None
        image_dtype = image.dtype if image_tensor else None
        image_np = image.detach().cpu().numpy() if image_tensor else np.asarray(image)
        if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
            raise ValueError(f"Expected mask (1,H,W) or (1,D,H,W), got {mask_np.shape}.")
        if image_np.ndim != mask_np.ndim or image_np.shape[1:] != mask_np.shape[1:]:
            raise ValueError(f"Image shape {image_np.shape} does not match mask shape {mask_np.shape}.")

        target_shape = mask_np.shape[1:]
        mask = _pad_for_transforms(mask_np, padding_factor=self.padding_factor)
        image = _pad_for_transforms(image_np, padding_factor=self.padding_factor)
        if self.image_interpolation_order > 0:
            original_background = mask_np[0] == 0
            valid_canvas = _pad_for_transforms(
                np.ones_like(mask_np, dtype=np.uint8),
                padding_factor=self.padding_factor,
            )[0].astype(bool)
            for channel_index, channel in enumerate(image_np):
                background_values = channel[original_background]
                finite_background = background_values[np.isfinite(background_values)]
                background_fill = (
                    float(np.min(finite_background))
                    if finite_background.size
                    else 0.0
                )
                image[channel_index, ~valid_canvas] = background_fill

        def transform_image_channels(image_to_transform, transform, foreground_mask):
            before = deepcopy(self.rng.bit_generator.state)
            after = [before]

            def replay_transform(spatial, order):
                self.rng.bit_generator.state = deepcopy(before)
                transformed = transform(spatial[None, ...], order)[0]
                after[0] = deepcopy(self.rng.bit_generator.state)
                return transformed

            if self.image_interpolation_order == 0:
                transformed_image = np.stack(
                    [replay_transform(channel, 0) for channel in image_to_transform],
                    axis=0,
                ).astype(image_to_transform.dtype, copy=False)
            else:
                transformed_image = interpolate_masked_regions(
                    image_to_transform,
                    foreground_mask,
                    warp=lambda spatial: replay_transform(
                        spatial, self.image_interpolation_order
                    ),
                    nearest_warp=lambda spatial: replay_transform(spatial, 0),
                )

            self.rng.bit_generator.state = after[0]
            return transformed_image

        def transform_image_global(
            image_to_transform,
            transform_name,
            params,
            center_mask,
        ):
            if transform_name == "rotate":
                rotation_center = _rotation_center(center_mask[0])
                if rotation_center is None:
                    return image_to_transform
                return transform_image_channels(
                    image_to_transform,
                    lambda channel, order, params=params, center=rotation_center:
                    random_global_rotation_transform(
                        channel, rng=self.rng, center=center,
                        order=order, **params
                    ),
                    center_mask[0] != 0,
                )

            return transform_image_channels(
                image_to_transform,
                lambda channel, order, name=transform_name, params=params:
                self.GLOBAL_TRANSFORMS[name](
                    channel, rng=self.rng, order=order, **params
                ),
                center_mask[0] != 0,
            )

        def transform_image_local(
            image_to_transform,
            transform_name,
            params,
            previous_mask,
            transformed_mask,
            class_ids,
            parameter_state,
        ):
            if transform_name == "local_dilate":
                for class_id in class_ids:
                    source_mask = previous_mask[0] == class_id
                    added = (transformed_mask[0] == class_id) & ~source_mask
                    if np.any(source_mask) and np.any(added):
                        nearest = ndi.distance_transform_edt(
                            ~source_mask,
                            return_distances=False,
                            return_indices=True,
                        )
                        source_coords = tuple(axis[added] for axis in nearest)
                        image_to_transform[:, added] = image_to_transform[
                            (slice(None), *source_coords)
                        ]
                return image_to_transform

            self.rng.bit_generator.state = parameter_state
            transformed_image = transform_image_global(
                image_to_transform,
                self.LOCAL_AS_GLOBAL_TRANSFORMS[transform_name],
                params,
                previous_mask,
            )
            return transformed_image

        for transform_name in self.GLOBAL_TRANSFORMS:
            probability = self.global_transform_probs.get(transform_name)
            if probability is None or not self._should_apply(probability):
                continue
            parameter_state = deepcopy(self.rng.bit_generator.state)
            mask_before_transform = mask
            mask = self._apply_global_transform(mask, transform_name)
            after_state = deepcopy(self.rng.bit_generator.state)
            self.rng.bit_generator.state = parameter_state
            image = transform_image_global(
                image,
                transform_name,
                dict(self.transform_params.get(transform_name, {})),
                mask_before_transform,
            )
            self.rng.bit_generator.state = after_state

        class_order = self._local_class_order(mask)
        if self.mask_transform_local_as_global:
            for transform_name in self.LOCAL_TRANSFORMS:
                probability = self._merged_local_probability(transform_name, class_order)
                if probability is None or not self._should_apply(probability):
                    continue
                local_params = self._merged_local_params(transform_name, class_order)
                parameter_state = deepcopy(self.rng.bit_generator.state)
                previous_mask = mask.copy()
                mask = self._apply_merged_local_transform(mask, transform_name, class_order)
                after_state = deepcopy(self.rng.bit_generator.state)
                image = transform_image_local(
                    image,
                    transform_name,
                    local_params,
                    previous_mask,
                    mask,
                    class_order,
                    parameter_state,
                )
                self.rng.bit_generator.state = after_state
        else:
            class_masks = {
                class_id: mask[0] == class_id
                for class_id in class_order
            }
            spatial_background = mask[0] == 0
            background_image = image.copy()
            if np.any(spatial_background):
                nearest_background = ndi.distance_transform_edt(
                    ~spatial_background,
                    return_distances=False,
                    return_indices=True,
                )
                foreground_coords = tuple(
                    axis[~spatial_background] for axis in nearest_background
                )
                background_image[:, ~spatial_background] = image[
                    (slice(None), *foreground_coords)
                ]
            class_images = {
                class_id: image * class_masks[class_id][None, ...]
                for class_id in class_order
            }
            any_local_transform_applied = False

            for class_id in class_order:
                class_transforms = dict(self.local_transform_probs)
                class_transforms.update(self.class_transform_probs.get(class_id, {}))
                for transform_name in self.LOCAL_TRANSFORMS:
                    probability = class_transforms.get(transform_name)
                    if probability is None or not self._should_apply(probability):
                        continue

                    previous_class_mask = class_masks[class_id]
                    class_canvas = np.zeros_like(mask)
                    class_canvas[0][previous_class_mask] = class_id
                    parameter_state = deepcopy(self.rng.bit_generator.state)
                    transformed_canvas = self._apply_local_transform(
                        class_canvas, class_id, transform_name
                    )
                    after_state = deepcopy(self.rng.bit_generator.state)
                    transformed_class_mask = transformed_canvas[0] == class_id
                    class_image = class_images[class_id]

                    local_params = dict(
                        self.transform_params.get(transform_name, {})
                    )
                    local_params.update(
                        self.class_transform_params.get(class_id, {}).get(
                            transform_name, {}
                        )
                    )
                    class_image = transform_image_local(
                        class_image,
                        transform_name,
                        local_params,
                        class_canvas,
                        transformed_canvas,
                        [class_id],
                        parameter_state,
                    )

                    self.rng.bit_generator.state = after_state
                    class_masks[class_id] = transformed_class_mask
                    class_images[class_id] = (
                        class_image * transformed_class_mask[None, ...]
                    )
                    any_local_transform_applied = True

            if any_local_transform_applied:
                mask = self._compose_class_masks(
                    class_masks, class_order, mask.dtype
                )
                image = background_image.copy()
                for class_id in reversed(class_order):
                    visible = class_masks[class_id]
                    image[:, visible] = class_images[class_id][:, visible]

        transformed_image = _fit_image_like_mask(
            image, mask, target_shape, self.image_interpolation_order
        )
        transformed_mask = _fit_mask_to_spatial_shape(mask, target_shape)
        if mask_tensor:
            transformed_mask = torch.as_tensor(
                transformed_mask, device=original_mask.device, dtype=original_mask.dtype
            )
        if image_tensor:
            transformed_image = torch.as_tensor(
                transformed_image, device=image_device, dtype=image_dtype
            )
        return transformed_mask, transformed_image
    def create_target_mask_from_synth_anomaly(self, synth_anomaly_image):
        if synth_anomaly_image is None:
            raise ValueError("synth_anomaly_image is required for threshold target-mask generation.")

        threshold_rel = 0.0 if self.background_threshold is None else float(self.background_threshold)
        if threshold_rel < 0.0:
            raise ValueError(f"background_threshold must be >= 0, got {self.background_threshold}.")

        if torch.is_tensor(synth_anomaly_image):
            threshold_source = synth_anomaly_image
            if not torch.is_floating_point(threshold_source):
                threshold_source = threshold_source.to(torch.float32)

            finite_values = threshold_source[torch.isfinite(threshold_source)]
            if finite_values.numel() == 0:
                return torch.zeros_like(torch.amax(threshold_source, dim=0), dtype=torch.uint8)

            min_val = torch.min(finite_values)
            max_val = torch.max(finite_values)
            threshold = min_val + threshold_rel * (max_val - min_val)
            synth_projection = torch.amax(threshold_source, dim=0)
            return (synth_projection > threshold).to(torch.uint8)

        min_val = float(np.nanmin(synth_anomaly_image))
        max_val = float(np.nanmax(synth_anomaly_image))
        threshold = min_val + threshold_rel * (max_val - min_val)
        synth_projection = np.max(synth_anomaly_image, axis=0)
        return (synth_projection > threshold).astype(np.uint8)

    def augment_mask(self, mask_np: np.ndarray) -> np.ndarray:
        original_spatial_shape = mask_np.shape[1:]
        augmented = _pad_for_transforms(mask_np, padding_factor=self.padding_factor)

        for transform_name in self.GLOBAL_TRANSFORMS:
            probability = self.global_transform_probs.get(transform_name)
            if probability is not None and self._should_apply(probability):
                augmented = self._apply_global_transform(augmented, transform_name)

        class_order = self._local_class_order(augmented)
        if self.mask_transform_local_as_global:
            for transform_name in self.LOCAL_TRANSFORMS:
                probability = self._merged_local_probability(transform_name, class_order)
                if probability is not None and self._should_apply(probability):
                    augmented = self._apply_merged_local_transform(augmented, transform_name, class_order)
            return _fit_mask_to_spatial_shape(augmented, original_spatial_shape)

        class_masks = {
            class_id: (augmented[0] == class_id)
            for class_id in class_order
        }
        any_local_transform_applied = False

        for class_id in class_order:
            class_canvas = np.zeros_like(augmented)
            class_canvas[0][class_masks[class_id]] = class_id

            class_transforms = dict(self.local_transform_probs)
            class_transforms.update(self.class_transform_probs.get(class_id, {}))
            for transform_name in self.LOCAL_TRANSFORMS:
                probability = class_transforms.get(transform_name)
                if probability is not None and self._should_apply(probability):
                    class_canvas = self._apply_local_transform(class_canvas, class_id, transform_name)
                    any_local_transform_applied = True

            class_masks[class_id] = class_canvas[0] == class_id

        if any_local_transform_applied:
            augmented = self._compose_class_masks(class_masks, class_order, mask_np.dtype)

        return _fit_mask_to_spatial_shape(augmented, original_spatial_shape)

    def set_transform_probs(self, probs: Dict[int | str, Any] | None = None) -> None:
        if probs is None:
            return
        for key, value in probs.items():
            if isinstance(key, str) and key in self.GLOBAL_TRANSFORMS:
                self.global_transform_probs[key] = self._validate_probability(value)
            elif isinstance(key, str) and key in self.LOCAL_TRANSFORMS:
                self.local_transform_probs[key] = self._validate_probability(value)
            elif isinstance(key, int):
                # class specific
                class_id = key
                if not isinstance(value, dict):
                    raise TypeError(
                        "Class-specific transform probabilities must be a dict, "
                        f"got {value!r} for class {class_id}."
                    )
                self.class_transform_probs.setdefault(class_id, {})
                for transform_name, probability in value.items():
                    if transform_name not in self.LOCAL_TRANSFORMS:
                        raise KeyError(
                            f"Class-specific transform probabilities are only supported for local transforms. "
                            f"Got {transform_name!r}. Available: {sorted(self.LOCAL_TRANSFORMS)}"
                        )
                    self.class_transform_probs[class_id][transform_name] = self._validate_probability(probability)
            else:
                available = sorted({*self.GLOBAL_TRANSFORMS, *self.LOCAL_TRANSFORMS})
                raise ValueError(
                    "mask_transform_probs keys must be transform names "
                    f"or integer class ids, got {key!r}. Available transforms: {available}."
                )

    def set_transform_params(self, params: Dict[int | str, Dict[str, Any]] | None = None) -> None:
        if params is None:
            return
        for key, value in params.items():
            if isinstance(key, str) and key in self.transform_params:
                if not isinstance(value, dict):
                    raise TypeError(f"Transform params for {key!r} must be a dict, got {value!r}.")
                self.transform_params[key].update(dict(value))
            elif isinstance(key, int):
                # class specific
                class_id = key
                if not isinstance(value, dict):
                    raise TypeError(
                        "Class-specific transform params must be a dict, "
                        f"got {value!r} for class {class_id}."
                    )
                self.class_transform_params.setdefault(class_id, {})
                for transform_name, transform_updates in value.items():
                    if transform_name not in self.LOCAL_TRANSFORMS:
                        raise KeyError(
                            f"Class-specific transform params are only supported for local transforms. "
                            f"Got {transform_name!r}. Available: {sorted(self.LOCAL_TRANSFORMS)}"
                        )
                    if not isinstance(transform_updates, dict):
                        raise TypeError(
                            f"Class-specific transform params for {transform_name!r} must be a dict, "
                            f"got {transform_updates!r}."
                        )
                    self.class_transform_params[class_id].setdefault(transform_name, {})
                    self.class_transform_params[class_id][transform_name].update(dict(transform_updates))
            else:
                raise ValueError(
                    "mask_transform_params keys must be transform names "
                    f"or integer class ids, got {key!r}."
                )

    def _apply_global_transform(self, mask_np: np.ndarray, transform_name: str) -> np.ndarray:
        transform = self.GLOBAL_TRANSFORMS[transform_name]
        params = dict(self.transform_params.get(transform_name, {}))
        return transform(mask_np, rng=self.rng, **params)

    def _apply_merged_local_transform(
        self,
        mask_np: np.ndarray,
        transform_name: str,
        class_order: list[int],
    ) -> np.ndarray:
        params = self._merged_local_params(transform_name, class_order)
        global_transform_name = self.LOCAL_AS_GLOBAL_TRANSFORMS.get(transform_name)
        if global_transform_name is not None:
            transform = self.GLOBAL_TRANSFORMS[global_transform_name]
            return transform(mask_np, rng=self.rng, **params)

        transform = self.LOCAL_TRANSFORMS[transform_name]
        return transform(
            mask_np,
            classes=class_order,
            priorities=class_order,
            params=params,
            rng=self.rng,
        )

    def _apply_local_transform(
        self,
        mask_np: np.ndarray,
        class_id: int,
        transform_name: str,
    ) -> np.ndarray:
        transform = self.LOCAL_TRANSFORMS[transform_name]
        params = dict(self.transform_params.get(transform_name, {}))
        params.update(self.class_transform_params.get(class_id, {}).get(transform_name, {}))
        return transform(
            mask_np,
            classes=[class_id],
            params=params,
            rng=self.rng,
        )

    def _merged_local_probability(self, transform_name: str, class_order: list[int]) -> float | None:
        if not class_order:
            return None

        probabilities = []
        for class_id in class_order:
            probability = self.class_transform_probs.get(class_id, {}).get(
                transform_name,
                self.local_transform_probs.get(transform_name),
            )
            if probability is None:
                return None
            probabilities.append(probability)
        return min(probabilities)

    def _merged_local_params(self, transform_name: str, class_order: list[int]) -> dict:
        class_params = []
        for class_id in class_order:
            params = dict(self.transform_params.get(transform_name, {}))
            params.update(self.class_transform_params.get(class_id, {}).get(transform_name, {}))
            class_params.append(params)

        if not class_params:
            return dict(self.transform_params.get(transform_name, {}))

        keys = set().union(*(params.keys() for params in class_params))
        merged = {}
        for key in keys:
            values = [params[key] for params in class_params if key in params]
            neutral_value = LOCAL_PARAM_NEUTRAL_VALUES.get(key)
            if neutral_value is None:
                neutral_value = 0
            merged[key] = min(values, key=lambda value: abs(value - neutral_value))

        # ensure ordered local param range (min<=max if suffix is equal)
        for min_key, min_value in list(merged.items()):
            if not min_key.startswith("min_"):
                continue
            max_key = f"max_{min_key[4:]}"
            if max_key in merged and min_value > merged[max_key]:
                raise ValueError(
                    f"Merged local transform params for {transform_name!r} have no overlap: "
                    f"{min_key}={min_value!r} > {max_key}={merged[max_key]!r}."
                )

        return merged

    def _validate_probability(self, probability) -> float:
        probability = float(probability)
        if not 0 <= probability <= 1:
            raise ValueError(f"Transform probability must be between 0 and 1, got {probability}.")
        return probability

    def _should_apply(self, probability: float) -> bool:
        return bool(self.rng.random() < probability)

    def _local_class_order(self, mask_np: np.ndarray) -> list[int]:
        present_classes = [int(class_id) for class_id in np.unique(mask_np[0]) if class_id != 0]
        if self.priorities is not None:
            configured = [class_id for class_id in self.priorities if class_id in present_classes]
            missing = [class_id for class_id in present_classes if class_id not in configured]
            return configured + sorted(missing)
        return sorted(present_classes)

    def _compose_class_masks(
        self,
        class_masks: Dict[int, np.ndarray],
        class_order: list[int],
        dtype,
    ) -> np.ndarray:
        if not class_masks:
            raise ValueError("class_masks must not be empty.")

        spatial_shape = next(iter(class_masks.values())).shape
        composed = np.zeros(spatial_shape, dtype=dtype)

        for class_id in reversed(class_order):
            composed[class_masks[class_id]] = class_id

        return composed[None, ...]
