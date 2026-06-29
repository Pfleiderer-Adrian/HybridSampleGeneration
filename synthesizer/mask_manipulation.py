from copy import deepcopy
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
    min_stretch=0.95,
    max_stretch=1.05,
    rng=None,
):
    """Apply nearest-neighbour scaling around the mask center while preserving shape."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    transformed_mask = mask_np[0].copy()
    scales = sample_uniform(min_stretch, max_stretch, rng=rng, size=transformed_mask.ndim)
    scales = _limit_scales_to_mask_bounds(transformed_mask != 0, scales)
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


def _limit_scales_to_mask_bounds(foreground_mask, scales):
    """Clamp stretch scales so foreground stays inside the current spatial bounds (anomaly_size)."""
    scales = np.asarray(scales, dtype=float).copy()

    if not np.any(foreground_mask):
        return scales

    center = np.array(foreground_mask.shape, dtype=float) / 2.0
    upper_bound = np.array(foreground_mask.shape, dtype=float) - 1.0
    foreground_coords = np.where(foreground_mask)   # list for every axis that combine to index positions of mask pixels

    for axis, axis_coords in enumerate(foreground_coords):
        min_coord = float(np.min(axis_coords))  # first index with anomaly in that axis
        max_coord = float(np.max(axis_coords))
        max_scale = np.inf

        if min_coord < center[axis]:
            max_scale = min(max_scale, (0.0 - center[axis]) / (min_coord - center[axis]))
        if max_coord > center[axis]:
            max_scale = min(max_scale, (upper_bound[axis] - center[axis]) / (max_coord - center[axis]))
        if np.isfinite(max_scale):
            scales[axis] = min(scales[axis], max_scale)

    return scales


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


def _rotate_spatial_mask(mask, angle):
    if mask.ndim < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got {mask.ndim}.")

    return ndi.rotate(
        mask,
        angle=angle,
        axes=(-2, -1),
        reshape=False,
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
    selected_foreground = np.isin(transformed_mask, classes)
    scales = _limit_scales_to_mask_bounds(selected_foreground, scales)
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
    sigma=40,
    magnitude=40,
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


DEFAULT_PADDING_MODE = "constant"

DEFAULT_TRANSFORM_PROBS = {
    "zoom": 1,
    "stretch": 1,
    "rotate": 0,
    "elastic": 1,
    "local_dilate": 0.5,
    "local_stretch": 0,
    "local_rotate": 0,
}

DEFAULT_TRANSFORM_PARAMS = {
    "zoom": {
        "min_zoom": 0.9,
        "max_zoom": 0.9,
    },
    "stretch": {
        "min_stretch": 0.9,
        "max_stretch": 1.1,
    },
    "rotate": {
        "max_rotation": 5.0,
    },
    "elastic": {
        "sigma": 40,
        "magnitude": 40,
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


class TransformGenerator:
    """Central orchestration object for mask augmentation."""

    @classmethod
    def from_config(cls, config):
        return cls(
            config.mask_transform_probs,
            use_default_mask_transforms=config.use_default_mask_transforms,
            transform_params=config.mask_transform_params,
            priorities=config.mask_transform_priorities,
            rng=config.rng,
            anomaly_size=config.anomaly_size,
            background_threshold=config.background_threshold,
        )

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
    }

    def __init__(
        self,
        transform_probs: Dict[int | str, Any] | None = None,
        *,
        use_default_mask_transforms: bool = False,
        transform_params: Dict[int | str, Dict[str, Any]] | None = None,
        priorities: list[int] | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
        anomaly_size: tuple[int, ...] | list[int] | None = None,
        background_threshold: float | None = 0.01,
    ) -> None:
        self.global_transform_probs = {}
        self.local_transform_probs = {}
        self.class_transform_probs = {}
        if use_default_mask_transforms:
            self.set_transform_probs(DEFAULT_TRANSFORM_PROBS)
        if transform_probs:
            self.set_transform_probs(transform_probs)
        self.transform_params = deepcopy(DEFAULT_TRANSFORM_PARAMS)
        if use_default_mask_transforms:
            self.transform_params["elastic"].update(default_elastic_params_from_anomaly_size(anomaly_size))
        self.class_transform_params = {}
        self.priorities = priorities
        if transform_params:
            self.set_transform_params(transform_params)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.background_threshold = background_threshold

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
        augmented = mask_np.copy()

        for transform_name in self.GLOBAL_TRANSFORMS:
            probability = self.global_transform_probs.get(transform_name)
            if probability is not None and self._should_apply(probability):
                augmented = self._apply_global_transform(augmented, transform_name)

        class_order = self._local_class_order(augmented)
        for class_id in class_order:
            class_transforms = dict(self.local_transform_probs)
            class_transforms.update(self.class_transform_probs.get(class_id, {}))
            for transform_name in self.LOCAL_TRANSFORMS:
                probability = class_transforms.get(transform_name)
                if probability is not None and self._should_apply(probability):
                    augmented = self._apply_local_transform(augmented, class_id, transform_name, class_order)

        return augmented

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

    def _apply_local_transform(
        self,
        mask_np: np.ndarray,
        class_id: int,
        transform_name: str,
        priorities: list[int],
    ) -> np.ndarray:
        transform = self.LOCAL_TRANSFORMS[transform_name]
        params = dict(self.transform_params.get(transform_name, {}))
        params.update(self.class_transform_params.get(class_id, {}).get(transform_name, {}))
        return transform(
            mask_np,
            classes=[class_id],
            priorities=priorities,
            params=params,
            rng=self.rng,
        )

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
