"""Fixed source budgets, complete real-image coverage, and reproducible crops."""

from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

from .cutmix import cutmix_defect
from .draem.synthesis import TextureSynthesizer
from .patches import crop_training_patch
from .transforms import aligned_flips, image_tensor, mask_tensor, read_image


class FixedQuotaDataset(Dataset):
    SOURCES = ("normal", "real", "draem", "cutmix", "hybrid")

    def __init__(self, healthy_samples, real_samples, hybrids, config):
        config.validate()
        self.config, self.hybrids = config, hybrids
        self.healthy_samples, self.real_samples = list(healthy_samples), list(real_samples)
        self.quotas = dict(config.data.source_quotas)
        self.hard_pool = None
        if not self.healthy_samples:
            raise ValueError("Healthy training images are required.")
        if self.quotas.get("real", 0) or self.quotas.get("cutmix", 0):
            if not self.real_samples or any(not s.mask_path for s in self.real_samples):
                raise ValueError("Real/CutMix sources require masked real training defects.")
        if self.quotas.get("real", 0) < len(self.real_samples):
            raise ValueError("Real quota must cover every real training image per epoch.")
        if self.quotas.get("hybrid", 0) and (hybrids is None or not len(hybrids)):
            raise ValueError("Hybrid quota requires materialized training hybrids.")
        self.synthesis = (TextureSynthesizer(config.data.texture_root)
                          if self.quotas.get("draem", 0) else None)
        self.set_epoch(0)

    def _real_order(self, count, source_code):
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, source_code, 831]))
        order = []
        while len(order) < count:
            order.extend(rng.permutation(len(self.real_samples)).tolist())
        return order[:count]

    def set_epoch(self, epoch):
        self.epoch = epoch
        entries = [(source, rank) for source in self.SOURCES
                   for rank in range(self.quotas.get(source, 0))]
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch, 711]))
        rng.shuffle(entries)
        self.entries = entries
        self.plan = [source for source, _ in entries]
        self.real_order = self._real_order(self.quotas.get("real", 0), 1)
        self.donor_order = self._real_order(self.quotas.get("cutmix", 0), 3)
        self.real_indices = {i: self.real_samples[self.real_order[rank]]
                             for i, (source, rank) in enumerate(entries) if source == "real"}
        assert Counter(self.plan) == Counter({s: n for s, n in self.quotas.items() if n})

    def __len__(self):
        return len(self.entries)

    def _healthy(self, rng):
        sample = self.healthy_samples[int(rng.integers(len(self.healthy_samples)))]
        array = read_image(sample.image_path)
        data = self.config.data
        size = data.image_size
        if data.mode == "patch":
            array, _, _ = crop_training_patch(array, array, np.zeros((1, *array.shape[-2:]), np.uint8),
                                              data.patch_size, rng)
            size = None
        return image_tensor(array, size, data.image_scale)

    def _real(self, sample, rng):
        image, mask = read_image(sample.image_path), read_image(sample.mask_path)
        data = self.config.data
        size = data.image_size
        if data.mode == "patch":
            image, _, mask = crop_training_patch(image, image, mask, data.patch_size, rng, anomalous=True)
            size = None
        image, mask = image_tensor(image, size, data.image_scale), mask_tensor(mask, size)
        if not mask.any():
            raise ValueError(f"Empty real defect patch: {sample.sample_id}")
        return image, mask

    def __getitem__(self, index):
        source, rank = self.entries[index]
        code = self.SOURCES.index(source)
        # Source-specific ranks keep healthy and first real draws identical across variants.
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, code, rank]))
        if source == "real":
            image, mask = self._real(self.real_samples[self.real_order[rank]], rng)
            target = image.clone()  # Placeholder: real examples never enter the paired branch.
        elif source == "hybrid":
            image, target, mask = self.hybrids.sample(int(rng.integers(len(self.hybrids))), rng)
        else:
            target = self._healthy(rng)
            if source == "normal":
                image, mask = target.clone(), torch.zeros_like(target[:1])
            elif source == "draem":
                image, mask = self.synthesis(target, rng)
            elif source == "cutmix":
                donor, donor_mask = self._real(self.real_samples[self.donor_order[rank]], rng)
                image, mask = cutmix_defect(target, donor, donor_mask, rng)
        image, target, mask = aligned_flips(image, target, mask, rng)
        return {"image": image, "target": target, "mask": mask,
                "source": source, "hard_normal": False}
