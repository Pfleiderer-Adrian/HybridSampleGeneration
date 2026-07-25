from copy import deepcopy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F


def _validate_class_id(class_id: int) -> int:
    class_id = int(class_id)
    if class_id <= 0:
        raise ValueError(f"class_id must be greater than 0, got {class_id}.")
    return class_id


def _validate_output_count(count: int) -> int:
    if isinstance(count, bool) or int(count) != count or int(count) < 1:
        raise ValueError(f"output count must be a positive integer, got {count!r}.")
    return int(count)


def _resolve_output_count(output_count: int, class_output_counts: Dict[int, int], class_ids=None) -> int:
    if class_ids is None:
        return output_count
    if np.isscalar(class_ids):
        class_ids = [class_ids]
    counts = [class_output_counts.get(_validate_class_id(class_id), output_count) for class_id in class_ids if int(class_id) != 0]
    return max(counts, default=output_count)

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
):
    """Apply nearest-neighbour scaling around the mask center while preserving shape."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    transformed_mask = mask_np[0].copy()
    scales = sample_uniform(min_stretch, max_stretch, rng=rng, size=transformed_mask.ndim)
    transformed_mask = _stretch_spatial_mask(
        transformed_mask,
        scales=scales,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


def random_global_zoom_transform(
    mask_np: np.ndarray,
    min_zoom=0.9,
    max_zoom=0.9,
    rng=None,
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
    transformed_mask = _stretch_spatial_mask(
        transformed_mask,
        scales=scales,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


def _stretch_spatial_mask(mask, scales):
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


def _rotate_spatial_mask(mask, angle, center_mask=None):
    if mask.ndim < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got {mask.ndim}.")

    if not np.any(center_mask):
        return mask.copy()

    # Compute the rotation center from the selected mask bounding box
    coords = np.where(center_mask)
    center = np.array(
        [(np.min(axis_coords) + np.max(axis_coords)) / 2.0 for axis_coords in coords],
        dtype=float,
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
        order=0,
        mode=DEFAULT_PADDING_MODE,
        cval=0,
        prefilter=False,
    )


def random_global_rotation_transform(
    mask_np: np.ndarray,
    max_rotation=5.0,
    rng=None,
):
    """Apply a small nearest-neighbour rotation to the whole label mask."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    angle = sample_uniform(max_value=max_rotation, rng=rng)
    transformed_mask = _rotate_spatial_mask(
        mask_np[0].copy(),
        angle=angle,
        center_mask=mask_np[0] != 0,
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
            binary_mask = _stretch_spatial_mask(
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
            binary_mask = _rotate_spatial_mask(
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
        order=0,
        mode=DEFAULT_PADDING_MODE,
        cval=0, # bg value for constant padding
        prefilter=False,
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



def _pad_mask_for_transforms(mask_np, padding_factor=2):
    """Center a channel-first mask on a larger zero-filled transform canvas."""
    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    spatial_shape = np.asarray(mask_np.shape[1:], dtype=int)
    padded_shape = spatial_shape * int(padding_factor)
    total_padding = padded_shape - spatial_shape
    pad_width = [(0, 0)] + [
        (int(padding // 2), int(padding - padding // 2))
        for padding in total_padding
    ]
    return np.pad(mask_np, pad_width, mode="constant", constant_values=0)


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
            spatial_mask = _stretch_spatial_mask(
                spatial_mask,
                scales=np.full(spatial_mask.ndim, fit_scale, dtype=float),
            ).astype(mask_np.dtype)

    crop_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(crop_start, target_shape)
    )
    return spatial_mask[crop_slices][None, ...]


class TransformGenerator:
    """Central orchestration object for mask augmentation."""

    @classmethod
    def from_config(cls, config):
        transform_config = config.transform_config
        return cls(
            transform_config.mask_transform_probs,
            use_mask_transform=transform_config.use_mask_transform,
            padding_factor=transform_config.padding_factor,
            transform_params=transform_config.mask_transform_params,
            priorities=transform_config.priorities,
            rng=transform_config.rng,
            anomaly_size=config.anomaly_size,
            background_threshold=config.background_threshold,
            mask_transform_local_as_global=transform_config.local_as_global,
            output_count=transform_config.output_count,
            class_output_counts=transform_config.class_output_counts,
        )

    @dataclass
    class Config:
        use_mask_transform: bool = True
        mask_transform_probs: Dict[int | str, Any] = field(default_factory=dict)
        mask_transform_params: Dict[int | str, Dict[str, Any]] = field(default_factory=dict)
        priorities: list[int] | tuple[int, ...] | None = None
        rng: np.random.Generator | None = None
        local_as_global: bool = False
        padding_factor: int = 2
        output_count: int = 1
        class_output_counts: Dict[int, int] = field(default_factory=dict)

        def setOutputCount(self, count: int):
            self.output_count = _validate_output_count(count)
            return self

        def setClassOutputCount(self, class_id: int, count: int):
            self.class_output_counts[_validate_class_id(class_id)] = _validate_output_count(count)
            return self

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
        output_count: int = 1,
        class_output_counts: Dict[int, int] | None = None,
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
        self.output_count = _validate_output_count(output_count)
        self.class_output_counts = {
            _validate_class_id(class_id): _validate_output_count(count)
            for class_id, count in (class_output_counts or {}).items()
        }

    def get_output_count(self, class_ids=None) -> int:
        return _resolve_output_count(self.output_count, self.class_output_counts, class_ids)

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
        augmented = _pad_mask_for_transforms(mask_np, factor=self.padding_factor)

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


@dataclass(frozen=True)
class SyntheticVariant:
    basename: str
    image: np.ndarray
    target_mask: np.ndarray
    metadata: dict


def _present_classes(sample: dict, source_basename: str, mask_loader: Callable[[str], np.ndarray]) -> list[int]:
    mask = sample.get("ori_mask")
    if mask is None:
        mask = mask_loader(source_basename)
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return [int(class_id) for class_id in np.unique(mask) if int(class_id) != 0]


def _variant_basename(source_basename: str, variant_index: int) -> str:
    stem, extension = os.path.splitext(source_basename)
    if extension != ".npy":
        raise ValueError(f"Expected an .npy source basename, got {source_basename!r}.")
    return f"{stem}_variant{variant_index + 1:03d}{extension}"


def generate_variants(
    model,
    sample: dict,
    *,
    mode: str,
    config,
    target_mask_generator,
    mask_loader: Callable[[str], np.ndarray],
    generate: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
) -> Iterator[SyntheticVariant]:
    """Generate one independent model output for every configured mask variant."""
    source_basename = sample["fname"]
    class_ids = _present_classes(sample, source_basename, mask_loader)
    output_count = target_mask_generator.get_output_count(class_ids)

    for variant_index in range(output_count):
        if generate is None:
            image, target_mask = model.generate(
                sample, mode=mode, variation_strength=config.variation_strength,
                clamp_01=config.clamp01_output,
                target_mask_generator=target_mask_generator,
            )
        else:
            image, target_mask = generate()

        yield SyntheticVariant(
            basename=_variant_basename(source_basename, variant_index),
            image=image,
            target_mask=target_mask,
            metadata={"source_anomaly": source_basename},
        )
