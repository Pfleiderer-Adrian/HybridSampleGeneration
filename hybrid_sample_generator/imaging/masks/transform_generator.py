"""Configuration-driven orchestration of mask transformations."""

from copy import deepcopy
from typing import Any, Dict

import numpy as np
import scipy.ndimage as ndi
import torch

from hybrid_sample_generator.configuration.augmentation import MaskTransformConfiguration
from hybrid_sample_generator.imaging.masks.elastic import (
    default_elastic_params_from_anomaly_size,
    random_elastic_transform,
)
from hybrid_sample_generator.imaging.masks.geometry import (
    fit_mask_to_spatial_shape,
    pad_mask_for_transforms,
    sample_uniform,
)
from hybrid_sample_generator.imaging.masks.global_transforms import (
    random_global_rotation_transform,
    random_global_stretch_transform,
    random_global_zoom_transform,
)
from hybrid_sample_generator.imaging.masks.local_transforms import (
    random_local_dilate_transform,
    random_local_elastic_transform,
    random_local_rotation_transform,
    random_local_stretch_transform,
)
from hybrid_sample_generator.imaging.masks.paired_transforms import (
    apply_warp_pair,
    dilate_class_pair,
    fit_pair,
    pad_pair,
    sample_warp,
)
from hybrid_sample_generator.imaging.masks.target_generation import (
    target_mask_from_original_mask,
    target_mask_from_synthetic_anomaly,
)


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
    "zoom": {"min_zoom": 0.9, "max_zoom": 0.9},
    "stretch": {"min_stretch": 1.0, "max_stretch": 1.2},
    "rotate": {"max_rotation": 5.0},
    "elastic": {"sigma": 30, "magnitude": 20},
    "local_dilate": {"min_iterations": 0, "max_iterations": 2},
    "local_stretch": {"min_stretch": 0.95, "max_stretch": 1.05},
    "local_rotate": {"max_rotation": 5.0},
    "local_elastic": {"sigma": 30, "magnitude": 20},
}

LOCAL_PARAM_NEUTRAL_VALUES = {"min_stretch": 1, "max_stretch": 1}


class TransformGenerator:
    """Central orchestration object for mask augmentation."""

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

    @classmethod
    def from_config(
        cls,
        config: MaskTransformConfiguration,
        *,
        anomaly_size,
        background_threshold,
        seed: int | None = None,
    ):
        """Build from mask-transform settings and explicit shared inputs."""
        return cls(
            config.mask_transform_probs,
            use_mask_transform=config.use_mask_transform,
            padding_factor=config.padding_factor,
            transform_params=config.mask_transform_params,
            priorities=config.priorities,
            rng=np.random.default_rng(seed),
            anomaly_size=anomaly_size,
            background_threshold=background_threshold,
            mask_transform_local_as_global=config.local_as_global,
        )

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
            self.transform_params["elastic"].update(
                default_elastic_params_from_anomaly_size(anomaly_size)
            )
        self.class_transform_params = {}
        self.priorities = priorities
        if transform_params:
            self.set_transform_params(transform_params)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.background_threshold = background_threshold
        self.mask_transform_local_as_global = mask_transform_local_as_global

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
        return target_mask_from_original_mask(original_mask, self.augment_mask)

    def create_target_mask_and_transformed_image(self, original_mask, image):
        """Draw each transform once and apply it to labels and image together."""
        mask_is_tensor = torch.is_tensor(original_mask)
        image_is_tensor = torch.is_tensor(image)
        mask_np = (
            original_mask.detach().cpu().numpy()
            if mask_is_tensor else np.asarray(original_mask)
        )
        image_np = image.detach().cpu().numpy() if image_is_tensor else np.asarray(image)
        original_shape = mask_np.shape[1:]
        mask, transformed_image = pad_pair(
            mask_np, image_np, self.padding_factor
        )
        for name in self.GLOBAL_TRANSFORMS:
            probability = self.global_transform_probs.get(name)
            if probability is not None and self._should_apply(probability):
                mask, transformed_image = self._warp_pair(
                    mask, transformed_image, name, self.transform_params[name]
                )

        class_order = self._local_class_order(mask)
        if self.mask_transform_local_as_global:
            for name in self.LOCAL_TRANSFORMS:
                probability = self._merged_local_probability(name, class_order)
                if probability is None or not self._should_apply(probability):
                    continue
                params = self._merged_local_params(name, class_order)
                if name == "local_dilate":
                    mask, transformed_image = self._dilate_pair(
                        mask, transformed_image, class_order, params
                    )
                else:
                    mask, transformed_image = self._warp_pair(
                        mask, transformed_image,
                        self.LOCAL_AS_GLOBAL_TRANSFORMS[name], params,
                    )
        elif class_order:
            mask, transformed_image = self._apply_local_pair_transforms(
                mask, transformed_image, class_order
            )

        mask, transformed_image = fit_pair(
            mask, transformed_image, original_shape
        )
        if mask_is_tensor:
            mask = torch.as_tensor(
                mask, device=original_mask.device, dtype=original_mask.dtype
            )
        if image_is_tensor:
            transformed_image = torch.as_tensor(
                transformed_image, device=image.device, dtype=image.dtype
            )
        return mask, transformed_image

    def _warp_pair(self, mask, image, name, params):
        warp = sample_warp(name, mask[0], params, self.rng)
        return apply_warp_pair(mask, image, warp)

    def _dilate_pair(self, mask, image, class_order, params):
        iterations = sample_uniform(
            params.get("min_iterations", 0),
            params.get("max_iterations", 2),
            rng=self.rng, integer=True,
        )
        class_masks = {}
        class_images = {}
        for class_id in class_order:
            source = mask[0] == class_id
            class_masks[class_id], class_images[class_id] = dilate_class_pair(
                source, image * source[None, ...], iterations
            )
        composed = self._compose_class_masks(class_masks, class_order, mask.dtype)
        result = image.copy()
        for class_id in reversed(class_order):
            visible = composed[0] == class_id
            result[:, visible] = class_images[class_id][:, visible]
        return composed, result

    def _apply_local_pair_transforms(self, mask, image, class_order):
        original_background = mask[0] == 0
        background_image = image.copy()
        if np.any(original_background):
            nearest = ndi.distance_transform_edt(
                ~original_background, return_distances=False, return_indices=True
            )
            foreground = ~original_background
            background_image[:, foreground] = image[
                (slice(None), *(axis[foreground] for axis in nearest))
            ]

        class_masks = {class_id: mask[0] == class_id for class_id in class_order}
        class_images = {
            class_id: image * class_masks[class_id][None, ...]
            for class_id in class_order
        }
        applied = False
        for class_id in class_order:
            probabilities = dict(self.local_transform_probs)
            probabilities.update(self.class_transform_probs.get(class_id, {}))
            for name in self.LOCAL_TRANSFORMS:
                probability = probabilities.get(name)
                if probability is None or not self._should_apply(probability):
                    continue
                params = dict(self.transform_params[name])
                params.update(
                    self.class_transform_params.get(class_id, {}).get(name, {})
                )
                if name == "local_dilate":
                    iterations = sample_uniform(
                        params.get("min_iterations", 0),
                        params.get("max_iterations", 2),
                        rng=self.rng, integer=True,
                    )
                    class_masks[class_id], class_images[class_id] = dilate_class_pair(
                        class_masks[class_id], class_images[class_id], iterations
                    )
                else:
                    class_mask = class_masks[class_id][None, ...].astype(mask.dtype)
                    class_mask, class_image = self._warp_pair(
                        class_mask, class_images[class_id],
                        name.removeprefix("local_"), params,
                    )
                    class_masks[class_id] = class_mask[0] != 0
                    class_images[class_id] = (
                        class_image * class_masks[class_id][None, ...]
                    )
                applied = True

        if not applied:
            return mask, image
        composed = self._compose_class_masks(class_masks, class_order, mask.dtype)
        result = background_image.copy()
        for class_id in reversed(class_order):
            visible = composed[0] == class_id
            result[:, visible] = class_images[class_id][:, visible]
        return composed, result

    def create_target_mask_from_synth_anomaly(self, synth_anomaly_image):
        return target_mask_from_synthetic_anomaly(
            synth_anomaly_image,
            background_threshold=self.background_threshold,
        )

    def augment_mask(self, mask_np: np.ndarray) -> np.ndarray:
        original_spatial_shape = mask_np.shape[1:]
        augmented = pad_mask_for_transforms(
            mask_np, padding_factor=self.padding_factor
        )

        for transform_name in self.GLOBAL_TRANSFORMS:
            probability = self.global_transform_probs.get(transform_name)
            if probability is not None and self._should_apply(probability):
                augmented = self._apply_global_transform(augmented, transform_name)

        class_order = self._local_class_order(augmented)
        if self.mask_transform_local_as_global:
            for transform_name in self.LOCAL_TRANSFORMS:
                probability = self._merged_local_probability(
                    transform_name, class_order
                )
                if probability is not None and self._should_apply(probability):
                    augmented = self._apply_merged_local_transform(
                        augmented, transform_name, class_order
                    )
            return fit_mask_to_spatial_shape(augmented, original_spatial_shape)

        class_masks = {
            class_id: (augmented[0] == class_id) for class_id in class_order
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
                    class_canvas = self._apply_local_transform(
                        class_canvas, class_id, transform_name
                    )
                    any_local_transform_applied = True
            class_masks[class_id] = class_canvas[0] == class_id

        if any_local_transform_applied:
            augmented = self._compose_class_masks(
                class_masks, class_order, mask_np.dtype
            )
        return fit_mask_to_spatial_shape(augmented, original_spatial_shape)

    def set_transform_probs(self, probs: Dict[int | str, Any] | None = None) -> None:
        if probs is None:
            return
        for key, value in probs.items():
            if isinstance(key, str) and key in self.GLOBAL_TRANSFORMS:
                self.global_transform_probs[key] = self._validate_probability(value)
            elif isinstance(key, str) and key in self.LOCAL_TRANSFORMS:
                self.local_transform_probs[key] = self._validate_probability(value)
            elif isinstance(key, int):
                if not isinstance(value, dict):
                    raise TypeError(
                        "Class-specific transform probabilities must be a dict, "
                        f"got {value!r} for class {key}."
                    )
                self.class_transform_probs.setdefault(key, {})
                for transform_name, probability in value.items():
                    if transform_name not in self.LOCAL_TRANSFORMS:
                        raise KeyError(
                            "Class-specific transform probabilities are only supported "
                            f"for local transforms. Got {transform_name!r}. Available: "
                            f"{sorted(self.LOCAL_TRANSFORMS)}"
                        )
                    self.class_transform_probs[key][transform_name] = (
                        self._validate_probability(probability)
                    )
            else:
                available = sorted({*self.GLOBAL_TRANSFORMS, *self.LOCAL_TRANSFORMS})
                raise ValueError(
                    "mask_transform_probs keys must be transform names or integer "
                    f"class ids, got {key!r}. Available transforms: {available}."
                )

    def set_transform_params(
        self, params: Dict[int | str, Dict[str, Any]] | None = None
    ) -> None:
        if params is None:
            return
        for key, value in params.items():
            if isinstance(key, str) and key in self.transform_params:
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Transform params for {key!r} must be a dict, got {value!r}."
                    )
                self.transform_params[key].update(dict(value))
            elif isinstance(key, int):
                if not isinstance(value, dict):
                    raise TypeError(
                        "Class-specific transform params must be a dict, "
                        f"got {value!r} for class {key}."
                    )
                self.class_transform_params.setdefault(key, {})
                for transform_name, transform_updates in value.items():
                    if transform_name not in self.LOCAL_TRANSFORMS:
                        raise KeyError(
                            "Class-specific transform params are only supported for "
                            f"local transforms. Got {transform_name!r}. Available: "
                            f"{sorted(self.LOCAL_TRANSFORMS)}"
                        )
                    if not isinstance(transform_updates, dict):
                        raise TypeError(
                            "Class-specific transform params for "
                            f"{transform_name!r} must be a dict, got {transform_updates!r}."
                        )
                    self.class_transform_params[key].setdefault(transform_name, {})
                    self.class_transform_params[key][transform_name].update(
                        dict(transform_updates)
                    )
            else:
                raise ValueError(
                    "mask_transform_params keys must be transform names or integer "
                    f"class ids, got {key!r}."
                )

    def _apply_global_transform(
        self, mask_np: np.ndarray, transform_name: str
    ) -> np.ndarray:
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
        params.update(
            self.class_transform_params.get(class_id, {}).get(transform_name, {})
        )
        return transform(
            mask_np,
            classes=[class_id],
            params=params,
            rng=self.rng,
        )

    def _merged_local_probability(
        self, transform_name: str, class_order: list[int]
    ) -> float | None:
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

    def _merged_local_params(
        self, transform_name: str, class_order: list[int]
    ) -> dict:
        class_params = []
        for class_id in class_order:
            params = dict(self.transform_params.get(transform_name, {}))
            params.update(
                self.class_transform_params.get(class_id, {}).get(transform_name, {})
            )
            class_params.append(params)
        if not class_params:
            return dict(self.transform_params.get(transform_name, {}))

        keys = set().union(*(params.keys() for params in class_params))
        merged = {}
        for key in keys:
            values = [params[key] for params in class_params if key in params]
            neutral_value = LOCAL_PARAM_NEUTRAL_VALUES.get(key, 0)
            merged[key] = min(values, key=lambda value: abs(value - neutral_value))

        for min_key, min_value in list(merged.items()):
            if not min_key.startswith("min_"):
                continue
            max_key = f"max_{min_key[4:]}"
            if max_key in merged and min_value > merged[max_key]:
                raise ValueError(
                    "Merged local transform params for "
                    f"{transform_name!r} have no overlap: {min_key}={min_value!r} "
                    f"> {max_key}={merged[max_key]!r}."
                )
        return merged

    @staticmethod
    def _validate_probability(probability) -> float:
        probability = float(probability)
        if not 0 <= probability <= 1:
            raise ValueError(
                f"Transform probability must be between 0 and 1, got {probability}."
            )
        return probability

    def _should_apply(self, probability: float) -> bool:
        return bool(self.rng.random() < probability)

    def _local_class_order(self, mask_np: np.ndarray) -> list[int]:
        present_classes = [
            int(class_id) for class_id in np.unique(mask_np[0]) if class_id != 0
        ]
        if self.priorities is not None:
            configured = [
                class_id
                for class_id in self.priorities
                if class_id in present_classes
            ]
            missing = [
                class_id
                for class_id in present_classes
                if class_id not in configured
            ]
            return configured + sorted(missing)
        return sorted(present_classes)

    @staticmethod
    def _compose_class_masks(
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
