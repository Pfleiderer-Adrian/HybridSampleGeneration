"""Training-only spatial augmentation."""

import torch
from torch.utils.data import Dataset

from hybrid_sample_generator.configuration.augmentation import AugmentationConfiguration


class TrainingTransformDataset(Dataset):
    """Apply a transform without modifying the reusable persisted dataset."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, tuple) and item:
            return self.transform.transform_sample(item)
        if isinstance(item, list) and item:
            return list(self.transform.transform_sample(tuple(item)))
        if isinstance(item, dict):
            return self.transform.transform_sample(item)
        return self.transform(item)


class RandomSpatialOffset:
    """Translate an anomaly while keeping its foreground inside the canvas."""

    def __init__(self, *, max_fraction=1.0, foreground_threshold_rel=0.001):
        self.max_fraction = max(0.0, min(float(max_fraction), 1.0))
        self.foreground_threshold_rel = max(float(foreground_threshold_rel), 0.0)

    def __call__(self, value):
        value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        shifts = self._sample_shifts(value)
        return value if shifts is None else self._apply_shift(
            value, shifts, fill_value=torch.amin(value)
        )

    def transform_sample(self, item):
        if isinstance(item, dict):
            return self._transform_dict_sample(item)
        if not item:
            return item
        image = item[0] if isinstance(item[0], torch.Tensor) else torch.as_tensor(item[0])
        shifts = self._sample_shifts(image)
        if shifts is None:
            return item
        shifted = [self._apply_shift(image, shifts, fill_value=torch.amin(image))]
        shifted.extend(self._maybe_apply_shift(value, shifts) for value in item[1:])
        return tuple(shifted)

    def _transform_dict_sample(self, item):
        item = dict(item)
        image_key = next(
            (key for key in ("img", "x", "image", "inputs") if key in item), None
        )
        if image_key is None:
            return item
        image = item[image_key]
        image = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        shifts = self._sample_shifts(image)
        if shifts is None:
            return item
        item[image_key] = self._apply_shift(image, shifts, fill_value=torch.amin(image))
        for key in ("ori_mask", "org_mask", "mask", "tgt_mask", "target_mask"):
            if key in item:
                item[key] = self._maybe_apply_shift(item[key], shifts)
        return item

    def _sample_shifts(self, image):
        if image.ndim not in (3, 4):
            return None
        minimum, maximum = torch.amin(image), torch.amax(image)
        dynamic_range = maximum - minimum
        if not bool(torch.isfinite(dynamic_range)) or float(dynamic_range) <= 0.0:
            return None
        foreground = torch.any(
            image > minimum + dynamic_range * self.foreground_threshold_rel, dim=0
        )
        if not bool(torch.any(foreground)):
            return None
        coordinates = torch.nonzero(foreground, as_tuple=False)
        spatial_min = coordinates.min(dim=0).values.tolist()
        spatial_max = coordinates.max(dim=0).values.tolist()
        shifts = []
        for axis, size in enumerate(image.shape[1:]):
            low = int(round(-int(spatial_min[axis]) * self.max_fraction))
            high = int(round(int(size - 1 - spatial_max[axis]) * self.max_fraction))
            shifts.append(
                low if low == high else int(torch.randint(low, high + 1, ()).item())
            )
        return shifts if any(shifts) else None

    def _maybe_apply_shift(self, value, shifts):
        if not isinstance(value, torch.Tensor):
            return value
        if value.ndim == len(shifts):
            return self._apply_shift(value, shifts, fill_value=0, has_channel_dim=False)
        if value.ndim == len(shifts) + 1:
            return self._apply_shift(value, shifts, fill_value=0)
        return value

    @staticmethod
    def _apply_shift(value, shifts, *, fill_value, has_channel_dim=True):
        shifted = torch.empty_like(value)
        shifted.fill_(fill_value.item() if isinstance(fill_value, torch.Tensor) else fill_value)
        source = [slice(None)] if has_channel_dim else []
        target = [slice(None)] if has_channel_dim else []
        shape = value.shape[1:] if has_channel_dim else value.shape
        for shift, size in zip(shifts, shape):
            if shift >= 0:
                source.append(slice(0, size - shift))
                target.append(slice(shift, size))
            else:
                source.append(slice(-shift, size))
                target.append(slice(0, size + shift))
        shifted[tuple(target)] = value[tuple(source)]
        return shifted


def apply_training_offset_augmentation(dataset, config: AugmentationConfiguration):
    if not config.random_offset_enabled:
        return dataset
    transform = RandomSpatialOffset(
        max_fraction=config.random_offset_max_fraction,
        foreground_threshold_rel=config.random_offset_foreground_threshold,
    )
    return TrainingTransformDataset(dataset, transform)


__all__ = ["RandomSpatialOffset", "TrainingTransformDataset", "apply_training_offset_augmentation"]
