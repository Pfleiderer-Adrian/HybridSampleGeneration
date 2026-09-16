"""Paired repository hybrids and real-image downstream evaluation datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .draem.synthesis import TextureSynthesizer
from .sampling import source_plan
from .patches import crop_training_patch
from examples.mvtec_ad2.splits import verify_repository
from .transforms import aligned_flips, image_tensor, mask_tensor, read_image


class HybridPairs:
    def __init__(self, repository, store, manifest, data):
        verify_repository(repository, manifest)
        self.store, self.data = store, data
        self.pairs = []
        self.provenance = []
        for hybrid in repository.list_hybrid_samples(status="generated"):
            original = repository.get_original_sample(hybrid.original_sample_id)
            if original.has_anomaly:
                continue
            placements = repository.list_placements(hybrid.id)
            if not placements or not hybrid.image_path or not hybrid.segmentation_path:
                raise ValueError(f"Incomplete materialized hybrid {hybrid.id}.")
            if data.mode == "patch":
                mask = mask_tensor(store.load_array(hybrid.segmentation_path), None)
                if not mask.any():
                    raise ValueError(f"Hybrid {hybrid.id} has an empty stored anomaly mask (before cropping).")
            donors = []
            for placement in placements:
                synthetic = repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
                real = repository.get_real_anomaly(synthetic.real_anomaly_id)
                donors.append(real.original_sample_id)
            self.pairs.append((hybrid, original))
            self.provenance.append({"hybrid_id": hybrid.id, "original_id": original.id,
                                    "donor_original_ids": sorted(set(donors)),
                                    "placement_ids": [p.id for p in placements]})
        if not self.pairs:
            raise ValueError("No materialized hybrids on healthy training originals available.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.sample(index, np.random.default_rng(index))

    def sample(self, index, rng):
        hybrid, original = self.pairs[index]
        image = self.store.load_array(hybrid.image_path)
        target = self.store.load_array(original.image_path)
        mask = self.store.load_array(hybrid.segmentation_path)
        if image.shape != target.shape or mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(f"Unaligned hybrid pair {hybrid.id}.")
        size = self.data.image_size
        if self.data.mode == "patch":
            image, target, mask = crop_training_patch(image, target, mask, self.data.patch_size, rng, anomalous=True)
            size = None
        mask = mask_tensor(mask, size)
        if not mask.any():
            raise ValueError(f"Hybrid {hybrid.id} has no anomaly after resizing.")
        return (image_tensor(image, size, self.data.image_scale),
                image_tensor(target, size, self.data.image_scale), mask)


class MixedTrainingDataset(Dataset):
    def __init__(self, healthy_samples, hybrids, config):
        self.healthy_samples, self.hybrids, self.config = healthy_samples, hybrids, config
        if not healthy_samples:
            raise ValueError("No healthy training images.")
        self.synthesis = TextureSynthesizer(config.data.texture_root) if config.data.hybrid_fraction < 1 else None
        if config.data.hybrid_fraction > 0 and (hybrids is None or len(hybrids) == 0):
            raise ValueError("hybrid_fraction > 0 requires eligible hybrid pairs.")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.plan = source_plan(self.config.data, self.config.seed, epoch)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        source = self.plan[index]
        if source == "hybrid":
            image, target, mask = self.hybrids.sample(int(rng.integers(len(self.hybrids))), rng)
        else:
            sample = self.healthy_samples[int(rng.integers(len(self.healthy_samples)))]
            array = read_image(sample.image_path)
            size = self.config.data.image_size
            if self.config.data.mode == "patch":
                array, _, _ = crop_training_patch(array, array, np.zeros((1, *array.shape[-2:]), dtype=np.uint8),
                                                  self.config.data.patch_size, rng)
                size = None
            target = image_tensor(array, size, self.config.data.image_scale)
            if source == "draem":
                image, mask = self.synthesis(target, rng)
            else:
                image, mask = target.clone(), torch.zeros_like(target[:1])
        image, target, mask = aligned_flips(image, target, mask, rng)
        return {"image": image, "target": target, "mask": mask, "source": source}


class RealImageDataset(Dataset):
    def __init__(self, samples, data):
        self.samples, self.data = samples, data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        size = None if self.data.mode == "patch" else self.data.image_size
        image = image_tensor(read_image(sample.image_path), size, self.data.image_scale)
        mask = (mask_tensor(read_image(sample.mask_path), size)
                if sample.mask_path else torch.zeros_like(image[:1]))
        if mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(f"Unaligned evaluation image/mask: {sample.sample_id}")
        return {"image": image, "mask": mask, "label": int(sample.label != "good"),
                "sample_id": sample.sample_id}
