"""Strict train-only, native-resolution cache adapter for AnomalyDiffusion."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .transforms import image_tensor, mask_tensor


class AnomalyDiffusionPairs:
    def __init__(self, path, split, data):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.data = data
        if (self.manifest.get("version") != 1
                or self.manifest.get("source") != "anomalydiffusion"
                or self.manifest.get("split_fingerprint") != split["fingerprint"]):
            raise ValueError("AnomalyDiffusion cache belongs to a different protocol/split.")
        if self.manifest.get("smoke_test"):
            raise ValueError("Smoke-test pairs cannot enter a benchmark run.")
        if data.mode != "patch" or tuple(data.patch_size) != (256, 256):
            raise ValueError("AnomalyDiffusion V3 requires native 256x256 patches.")
        train = {r["sample_id"]: r for r in split["partitions"]["train"]}
        held_out = {r["sample_id"] for p in ("validation", "test") for r in split["partitions"][p]}
        learned = set(self.manifest.get("training_sample_ids", []))
        # Exactly all generator training sources must be recorded, including mask training.
        if not learned or not learned <= train.keys() or learned & held_out:
            raise ValueError("Generator training provenance is missing or leaks held-out data.")
        self.rows = self.manifest.get("rows", [])
        if not self.rows:
            raise ValueError("AnomalyDiffusion cache has no generated pairs.")
        for row in self.rows:
            recipient = row["recipient_sample_id"]
            if recipient not in train or train[recipient]["label"] != "good" or recipient in held_out:
                raise ValueError("Generated background must be a healthy training image.")
            if recipient not in learned:
                raise ValueError("Background missing from generator provenance.")
            image, target, mask = self._arrays(row)
            if image.shape != target.shape or image.shape != (3, 256, 256) or mask.shape != (1, 256, 256):
                raise ValueError("Generated pair is not aligned at native 256x256 resolution.")
            if not mask.any() or not np.isin(mask, [0, 255]).all():
                raise ValueError("Generated mask must be binary and nonempty.")

    def _arrays(self, row):
        def load(key, mode):
            path = Path(row[key])
            if not path.is_absolute():
                path = self.path.parent / path
            value = np.array(Image.open(path).convert(mode))
            return value.transpose(2, 0, 1) if mode == "RGB" else value[None]
        return load("image_path", "RGB"), load("target_path", "RGB"), load("mask_path", "L")

    def __len__(self):
        return len(self.rows)

    def sample(self, index, rng):
        image, target, mask = self._arrays(self.rows[index])
        return (image_tensor(image, None, self.data.image_scale),
                image_tensor(target, None, self.data.image_scale), mask_tensor(mask, None))
