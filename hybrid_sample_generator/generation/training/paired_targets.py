"""Online source-to-target pair construction for paired conditional models."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from hybrid_sample_generator.imaging.masks.transform_generator import (
    TransformGenerator,
)
from hybrid_sample_generator.persistence.identifiers import stable_seed


class PairedTargetDataset(Dataset):
    """Add a jointly transformed target image and mask to source samples."""

    def __init__(
        self,
        dataset,
        mask_transform_config,
        *,
        anomaly_size,
        background_threshold: float,
        identity_probability: float,
        seed: int,
        deterministic: bool,
    ) -> None:
        probability = float(identity_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("identity_probability must be in [0, 1].")
        self.dataset = dataset
        self.mask_transform_config = mask_transform_config
        self.anomaly_size = tuple(anomaly_size)
        self.background_threshold = float(background_threshold)
        self.identity_probability = probability
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self._worker_generators: dict[int, TransformGenerator] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        source = self.dataset[index]
        if not isinstance(source, dict):
            raise TypeError("PairedTargetDataset expects dictionary samples.")
        if "img" not in source or "ori_mask" not in source:
            raise KeyError("PairedTargetDataset requires 'img' and 'ori_mask'.")
        sample = dict(source)
        generator = self._generator(index)
        identity = bool(generator.rng.random() < self.identity_probability)
        if identity:
            target_mask = _copy_value(sample["ori_mask"])
            target_image = _copy_value(sample["img"])
        else:
            target_mask, target_image = (
                generator.create_target_mask_and_transformed_image(
                    sample["ori_mask"], sample["img"]
                )
            )
        sample["tgt_mask"] = target_mask
        sample["tgt_img"] = target_image
        sample["pair_is_identity"] = identity
        return sample

    def _generator(self, index: int) -> TransformGenerator:
        if self.deterministic:
            return self._build_generator(
                stable_seed(self.seed, "paired-validation", index)
            )
        worker = get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        generator = self._worker_generators.get(worker_id)
        if generator is None:
            worker_seed = self.seed if worker is None else int(worker.seed)
            generator = self._build_generator(
                stable_seed(worker_seed, "paired-training", worker_id)
            )
            self._worker_generators[worker_id] = generator
        return generator

    def _build_generator(self, seed: int) -> TransformGenerator:
        return TransformGenerator.from_config(
            self.mask_transform_config,
            anomaly_size=self.anomaly_size,
            background_threshold=self.background_threshold,
            seed=seed,
        )


def _copy_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return np.array(value, copy=True)


def apply_paired_target_training(
    train_dataset,
    validation_dataset,
    *,
    config,
    identity_probability: float,
):
    """Wrap both splits, using stochastic training and stable validation pairs."""
    common = {
        "mask_transform_config": config.augmentation.mask_transforms,
        "anomaly_size": config.extraction.anomaly_size,
        "background_threshold": config.generation.background_threshold,
        "identity_probability": identity_probability,
        "seed": config.study.seed,
    }
    return (
        PairedTargetDataset(train_dataset, deterministic=False, **common),
        PairedTargetDataset(validation_dataset, deterministic=True, **common),
    )


__all__ = ["PairedTargetDataset", "apply_paired_target_training"]
