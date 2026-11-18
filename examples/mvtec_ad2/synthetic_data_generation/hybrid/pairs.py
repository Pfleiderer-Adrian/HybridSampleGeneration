"""Materialized hybrids paired with their healthy originals for downstream training."""

import numpy as np

from examples.mvtec_ad2.data.patches import crop_training_patch
from examples.mvtec_ad2.data.splits import verify_repository
from examples.mvtec_ad2.data.transforms import image_tensor, mask_tensor


class HybridPairs:
    def __init__(self, repository, store, manifest, data, postprocessing=None, *, included_ids=None):
        verify_repository(repository, manifest)
        self.store, self.data = store, data
        self.pairs = []
        self.provenance = []
        selected = None if included_ids is None else set(included_ids)
        if selected is not None and (not selected or len(selected) != len(included_ids)):
            raise ValueError("Frozen hybrid IDs must be nonempty and unique.")
        for hybrid in repository.list_hybrid_samples(status="generated"):
            if selected is not None and hybrid.id not in selected:
                continue
            original = repository.get_original_sample(hybrid.original_sample_id)
            if original.has_anomaly:
                continue
            if selected is None and postprocessing is not None and postprocessing.enabled:
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
        if selected is not None:
            by_id = {pair[0].id: (pair, source) for pair, source in zip(self.pairs, self.provenance)}
            if set(by_id) != selected:
                raise ValueError("Frozen hybrid selection contains missing or ineligible samples.")
            # Preserve the historical order as well as membership for seeded draws.
            self.pairs = [by_id[key][0] for key in included_ids]
            self.provenance = [by_id[key][1] for key in included_ids]
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
