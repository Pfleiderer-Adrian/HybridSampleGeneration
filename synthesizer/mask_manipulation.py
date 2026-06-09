from copy import deepcopy
from typing import Any, Dict

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from monai.transforms import Rand2DElastic, Rand3DElastic

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


def random_global_stretch_transform(mask_np: np.ndarray, stretch_min=0.95, stretch_max=1.05):
    """Apply nearest-neighbour scaling around the mask center while preserving shape."""
    original_dtype = mask_np.dtype
    ndim = mask_np.ndim

    if ndim not in (3, 4):
        raise ValueError(f"Unterstuetzt nur 2D oder 3D Daten. Deine Maske hat {ndim} Dimensionen.")

    transformed_mask = mask_np[0].copy()
    spatial_ndim = transformed_mask.ndim
    scales = [np.random.uniform(stretch_min, stretch_max) for _ in range(spatial_ndim)]

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

    if mask_np.shape[0] == 1:
        return transformed_mask[None, ...]
    return np.repeat(transformed_mask[None, ...], mask_np.shape[0], axis=0)


def _sample_iterations(iterations):
    if isinstance(iterations, (tuple, list)):
        if len(iterations) != 2:
            raise ValueError(f"iterations range must contain exactly two values, got {iterations!r}.")
        low, high = int(iterations[0]), int(iterations[1])
        if low > high:
            raise ValueError(f"iterations range must be ordered as (min, max), got {iterations!r}.")
        return int(np.random.randint(low, high + 1))
    return int(iterations)


def random_dilate_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None):
    """Dilate selected classes in a 2D or 3D channel-first label mask."""
    original_dtype = mask_np.dtype
    ndim = mask_np.ndim

    if ndim not in (3, 4):
        raise ValueError(f"Unterstuetzt nur 2D oder 3D Daten. Deine Maske hat {ndim} Dimensionen.")

    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()

    if classes is None:
        classes = [cls for cls in np.unique(transformed_mask) if cls != 0]
    if priorities is None:
        priorities = classes

    class_masks = {}
    iterations = _sample_iterations(params.get("iterations", 1))

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
    if mask_np.shape[0] == 1:
        return final_mask[None, ...]
    return np.repeat(final_mask[None, ...], mask_np.shape[0], axis=0)


def random_elastic_transform(mask_np: np.ndarray, sigma=5, magnitude=200):
    """
    Nimmt ein 2D oder 3D NumPy-Array, wählt dynamisch die richtige 
    MONAI-Transformation (2D oder 3D) und gibt das verzerrte Array zurück.
    """
    original_dtype = mask_np.dtype
    ndim = mask_np.ndim

    # MONAI erwartet das Format [Kanal, Raumdimensionen] -> [1, (D), H, W]
    mask_tensor = torch.from_numpy(mask_np).float()

    if ndim == 3:
        # Fuer 2D (PNG) -> Steuerung der Gitterdichte ueber 'spacing'
        re_deform = Rand2DElastic(
            spacing=(sigma, sigma), # abstand der kontrollpunkte
            magnitude_range=(magnitude, magnitude),
            prob=1.0,
            mode="nearest",
            padding_mode="reflection",
        )
    elif ndim == 4:
        # Fuer 3D (NIfTI) -> Steuerung der Gitterglaettung ueber 'sigma_range'
        re_deform = Rand3DElastic(
            sigma_range=(sigma, sigma),
            magnitude_range=(magnitude, magnitude),
            prob=1.0,
            mode="nearest",
            padding_mode="reflection",
        )
    else:
        raise ValueError(f"Unterstützt nur 2D oder 3D Daten. Deine Maske hat {ndim} Dimensionen.")

    transformed_tensor = re_deform(mask_tensor)

    return transformed_tensor.numpy().astype(original_dtype)


GLOBAL_TRANSFORM_PROBS = {
    "stretch": 1,
    "elastic": 1,
}

LOCAL_TRANSFORM_PROBS = {
    "dilate": 0.5,
}


DEFAULT_TRANSFORM_PARAMS = {
    "elastic": {
        "sigma": 70,
        "magnitude": 2,
    },
    "dilate": {
        "iterations": (1, 2),
    },
    "stretch": {
        "stretch_min": 0.95,
        "stretch_max": 1.05,
    },
}


class TransformGenerator:
    """Central orchestration object for mask augmentation."""

    GLOBAL_TRANSFORMS = {
        "elastic": random_elastic_transform,
        "stretch": random_global_stretch_transform,
    }
    LOCAL_TRANSFORMS = {
        "dilate": random_dilate_transform,
    }

    def __init__(
        self,
        global_transforms: Dict[str, float] | None = None,
        local_transforms: Dict[int, Dict[str, float]] | None = None,
        *,
        use_mask_transform: bool = False,
        transform_params: Dict[str, Dict[str, Any]] | None = None,
        class_transform_params: Dict[int, Dict[str, Dict[str, Any]]] | None = None,
        priorities: list[int] | tuple[int, ...] | None = None,
        num_anomaly_classes: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.global_transforms = {}
        if use_mask_transform:
            self.global_transforms.update(GLOBAL_TRANSFORM_PROBS)
        self.global_transforms.update(global_transforms or {})

        self.default_local_transforms = dict(LOCAL_TRANSFORM_PROBS) if use_mask_transform else {}
        self.local_transforms = dict(local_transforms or {})
        self.transform_params = deepcopy(DEFAULT_TRANSFORM_PARAMS)
        self.class_transform_params = {}
        self.num_anomaly_classes = None if num_anomaly_classes is None else int(num_anomaly_classes)
        self.priorities = None if priorities is None else [int(class_id) for class_id in priorities]
        if transform_params:
            self.set_transform_params(transform_params)
        if class_transform_params:
            self.set_class_transform_params(class_transform_params)
        self.rng = rng if rng is not None else np.random.default_rng()

    def augment_mask(self, mask_np: np.ndarray) -> np.ndarray:
        augmented = mask_np.copy()

        for transform_name, probability in self.global_transforms.items():
            if self._should_apply(probability):
                augmented = self._apply_global_transform(augmented, transform_name)

        for class_id in self._local_class_order(augmented):
            class_transforms = dict(self.default_local_transforms)
            class_transforms.update(self.local_transforms.get(class_id, {}))
            for transform_name, probability in class_transforms.items():
                if self._should_apply(probability):
                    augmented = self._apply_local_transform(augmented, class_id, transform_name)

        return augmented

    def set_transform_params(self, params: Dict[str, Dict[str, Any]] | None = None, **kwargs) -> None:
        updates = {} if params is None else dict(params)
        updates.update(kwargs)
        for transform_name, transform_updates in updates.items():
            if transform_name not in self.transform_params:
                self.transform_params[transform_name] = {}
            self.transform_params[transform_name].update(dict(transform_updates))

    def set_class_transform_params(
        self,
        params: Dict[int, Dict[str, Dict[str, Any]]] | None = None,
        **kwargs,
    ) -> None:
        updates = {} if params is None else dict(params)
        updates.update(kwargs)
        for class_id, class_updates in updates.items():
            self.class_transform_params.setdefault(class_id, {})
            for transform_name, transform_updates in class_updates.items():
                self.class_transform_params[class_id].setdefault(transform_name, {})
                self.class_transform_params[class_id][transform_name].update(dict(transform_updates))

    def _apply_global_transform(self, mask_np: np.ndarray, transform_name: str) -> np.ndarray:
        transform = self.GLOBAL_TRANSFORMS[transform_name]
        params = dict(self.transform_params.get(transform_name, {}))
        return transform(mask_np, **params)

    def _apply_local_transform(self, mask_np: np.ndarray, class_id: int, transform_name: str) -> np.ndarray:
        transform = self.LOCAL_TRANSFORMS[transform_name]
        params = dict(self.transform_params.get(transform_name, {}))
        params.update(self.class_transform_params.get(class_id, {}).get(transform_name, {}))
        classes = params.pop("classes", None)
        priorities = params.pop("priorities", None)
        return transform(
            mask_np,
            classes=[class_id] if classes is None else classes,
            priorities=self._local_class_order(mask_np) if priorities is None else priorities,
            params=params,
        )

    def _should_apply(self, probability: float) -> bool:
        probability = float(probability)
        if not 0 <= probability <= 1:
            raise ValueError(f"Transform probability must be between 0 and 1, got {probability}.")
        return bool(self.rng.random() < probability)

    def _local_class_order(self, mask_np: np.ndarray) -> list[int]:
        present_classes = [int(class_id) for class_id in np.unique(mask_np[0]) if class_id != 0]
        if self.priorities is not None:
            configured = [class_id for class_id in self.priorities if class_id in present_classes]
            missing = [class_id for class_id in present_classes if class_id not in configured]
            return configured + sorted(missing)
        if self.num_anomaly_classes is not None:
            priority_order = list(range(1, self.num_anomaly_classes + 1))
        else:
            priority_order = list(range(1, int(mask_np[0].max()) + 1))
        return [class_id for class_id in priority_order if class_id in present_classes]
