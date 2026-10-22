"""Paired repository hybrids and real-image downstream evaluation datasets."""

import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset

from .draem.synthesis import TextureSynthesizer
from .sampling import source_plan
from .hard_normals import load_pool
from .patches import crop_training_patch
from examples.mvtec_ad2.splits import verify_repository
from .transforms import aligned_flips, image_tensor, mask_tensor, read_image


class HybridPairs:
    def __init__(self, repository, store, manifest, data, postprocessing=None):
        verify_repository(repository, manifest)
        self.store, self.data = store, data
        self.pairs = []
        self.provenance = []
        for hybrid in repository.list_hybrid_samples(status="generated"):
            original = repository.get_original_sample(hybrid.original_sample_id)
            if original.has_anomaly:
                continue
            if postprocessing is not None and postprocessing.enabled:
                from hybrid_sample_generator.postprocessing.service import decision_for
                decision = decision_for(repository, store, hybrid, original, postprocessing)
                if postprocessing.action == "exclude" and not decision["accepted"]:
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
    def __init__(self, healthy_samples, hybrids, config, real_samples=None):
        self.healthy_samples, self.hybrids, self.config = healthy_samples, hybrids, config
        if not healthy_samples:
            raise ValueError("No healthy training images.")
        self.synthesis = TextureSynthesizer(config.data.texture_root) if config.data.hybrid_fraction < 1 else None
        if config.data.hybrid_fraction > 0 and (hybrids is None or len(hybrids) == 0):
            raise ValueError("hybrid_fraction > 0 requires eligible hybrid pairs.")
        self.hard_pool = (load_pool(config.data.hard_normal_manifest, healthy_samples, config.data.patch_size)
                          if config.data.hard_normal_fraction > 0 else None)
        self.healthy_by_id = {sample.sample_id: sample for sample in healthy_samples}
        self.real_samples = list(real_samples or []) if config.data.include_real_defects else []
        if config.data.include_real_defects and not self.real_samples:
            raise ValueError("Real defect augmentation requires masked training defects.")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.plan = source_plan(self.config.data, self.config.seed, epoch)
        # Append every real training image once, preserving the original source quotas.
        self.real_indices = {len(self.plan) + i: sample for i, sample in enumerate(self.real_samples)}
        self.plan.extend(["real"] * len(self.real_samples))
        if self.real_samples:
            order = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch, 711])).permutation(len(self.plan))
            self.real_indices = {i: self.real_indices[int(old)] for i, old in enumerate(order) if int(old) in self.real_indices}
            self.plan = [self.plan[int(old)] for old in order]
        normal_indices = [i for i, source in enumerate(self.plan) if source == "normal"]
        count = round(len(normal_indices) * self.config.data.hard_normal_fraction)
        selector = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch, 1667]))
        self.hard_indices = set(selector.choice(normal_indices, size=count, replace=False).tolist())

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        source = self.plan[index]
        if source == "real":
            sample = self.real_indices[index]
            array, mask_array = read_image(sample.image_path), read_image(sample.mask_path)
            size = self.config.data.image_size
            if self.config.data.mode == "patch":
                array, _, mask_array = crop_training_patch(array, array, mask_array, self.config.data.patch_size, rng, anomalous=True)
                size = None
            image = image_tensor(array, size, self.config.data.image_scale)
            target, mask = image.clone(), mask_tensor(mask_array, size)
            if not mask.any():
                raise ValueError("Real training defect has an empty mask.")
        elif source == "hybrid":
            image, target, mask = self.hybrids.sample(int(rng.integers(len(self.hybrids))), rng)
        else:
            record = (self.hard_pool["records"][int(rng.integers(len(self.hard_pool["records"])))]
                      if index in self.hard_indices else None)
            sample = (self.healthy_by_id[record["sample_id"]] if record is not None else
                      self.healthy_samples[int(rng.integers(len(self.healthy_samples)))])
            array = read_image(sample.image_path)
            size = self.config.data.image_size
            if record is not None:
                ph, pw = self.config.data.patch_size
                top, left = record["top"], record["left"]
                if top > max(0, array.shape[-2]-ph) or left > max(0, array.shape[-1]-pw):
                    raise ValueError("Hard-normal crop lies outside its source image.")
                array = array[:, top:top+ph, left:left+pw]
                array = np.pad(array, ((0, 0), (0, ph-array.shape[-2]), (0, pw-array.shape[-1])), mode="edge")
                size = None
            elif self.config.data.mode == "patch":
                array, _, _ = crop_training_patch(array, array, np.zeros((1, *array.shape[-2:]), dtype=np.uint8),
                                                  self.config.data.patch_size, rng)
                size = None
            target = image_tensor(array, size, self.config.data.image_scale)
            if source == "draem":
                image, mask = self.synthesis(target, rng)
            else:
                image, mask = target.clone(), torch.zeros_like(target[:1])
        image, target, mask = aligned_flips(image, target, mask, rng)
        return {"image": image, "target": target, "mask": mask, "source": source,
                "hard_normal": index in self.hard_indices}


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


class FixedPairedValidation(MixedTrainingDataset):
    """Deterministic DRAEM pairs from held-out healthy validation originals."""

    def __init__(self, healthy_samples, config):
        fixed = deepcopy(config)
        fixed.seed = config.seed + 104729
        fixed.data.include_real_defects = False
        fixed.data.normal_fraction = 0.0
        fixed.data.hard_normal_fraction = 0.0
        fixed.data.hard_normal_manifest = None
        fixed.data.hybrid_fraction = 0.0
        fixed.data.samples_per_epoch = config.dinov3.validation_samples
        super().__init__(healthy_samples, None, fixed)

    def set_epoch(self, epoch):
        # Identical image, crop, anomaly and flips at every validation pass.
        super().set_epoch(0)
