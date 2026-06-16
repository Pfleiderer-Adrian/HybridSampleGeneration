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


def random_global_stretch_transform(mask_np: np.ndarray, stretch_min=0.95, stretch_max=1.05, rng=None):
    """Apply nearest-neighbour scaling around the mask center while preserving shape."""
    original_dtype = mask_np.dtype

    if mask_np.ndim not in (3, 4) or mask_np.shape[0] != 1:
        raise ValueError(f"Expected mask with shape (1, H, W) or (1, D, H, W), got {mask_np.shape}.")

    transformed_mask = mask_np[0].copy()
    spatial_ndim = transformed_mask.ndim
    rng = rng if rng is not None else np.random.default_rng()
    scales = [rng.uniform(stretch_min, stretch_max) for _ in range(spatial_ndim)]

    inv_scales = 1.0 / np.array(scales)
    matrix = np.diag(inv_scales)

    center = np.array(transformed_mask.shape) / 2.0
    offset = center - np.dot(matrix, center)

    transformed_mask = ndi.affine_transform(
        transformed_mask,
        matrix=matrix,
        offset=offset,
        output_shape=transformed_mask.shape,
        order=0,
        mode="constant",
        cval=0,
    ).astype(original_dtype)

    return transformed_mask[None, ...]
    

def _sample_iterations(iterations, rng=None):
    if isinstance(iterations, (tuple, list)):
        if len(iterations) != 2:
            raise ValueError(f"iterations range must contain exactly two values, got {iterations!r}.")
        low, high = int(iterations[0]), int(iterations[1])
        if low > high:
            raise ValueError(f"iterations range must be ordered as (min, max), got {iterations!r}.")
        rng = rng if rng is not None else np.random.default_rng()
        return int(rng.integers(low, high + 1))
    return int(iterations)


def random_dilate_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None):
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
    iterations = _sample_iterations(params.get("iterations", 1), rng=rng)

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
    padding_mode="constant",
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
        mode=padding_mode,
        cval=0, # bg value for constant padding
        prefilter=False,
    ).astype(original_dtype)

    return transformed_mask[None, ...]


DEFAULT_TRANSFORM_PROBS = {
    "stretch": 1,
    "elastic": 1,
    "dilate": 0.5,
}

DEFAULT_TRANSFORM_PARAMS = {
    "elastic": {
        "sigma": 40,
        "magnitude": 40,
        "padding_mode": "constant",
    },
    "dilate": {
        "iterations": (1, 2),
    },
    "stretch": {
        "stretch_min": 0.95,
        "stretch_max": 1.05,
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
            getattr(config, "mask_transform_probs", None),
            use_default_mask_transforms=getattr(config, "use_default_mask_transforms", False),
            transform_params=getattr(config, "mask_transform_params", None),
            priorities=getattr(config, "mask_transform_priorities", None),
            rng=getattr(config, "rng", None),
            anomaly_size=getattr(config, "anomaly_size", None),
            background_threshold=getattr(config, "background_threshold", 0.01),
        )

    GLOBAL_TRANSFORMS = {
        "elastic": random_elastic_transform,
        "stretch": random_global_stretch_transform,
    }
    LOCAL_TRANSFORMS = {
        "dilate": random_dilate_transform,
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

        for transform_name, probability in self.global_transform_probs.items():
            if self._should_apply(probability):
                augmented = self._apply_global_transform(augmented, transform_name)

        class_order = self._local_class_order(augmented)
        for class_id in class_order:
            class_transforms = dict(self.local_transform_probs)
            class_transforms.update(self.class_transform_probs.get(class_id, {}))
            for transform_name, probability in class_transforms.items():
                if self._should_apply(probability):
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
