"""Training-only mining and validation of healthy patches with high anomaly scores."""

import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .patches import tile_starts, tiled_logits
from .transforms import read_image, image_tensor


def box_iou(first, second, size):
    ph, pw = size
    overlap = max(0, ph - abs(first[0] - second[0])) * max(0, pw - abs(first[1] - second[1]))
    return overlap / (2 * ph * pw - overlap)


def load_pool(path, samples, size):
    if any(sample.label != "good" for sample in samples):
        raise ValueError("Hard-normal sampling requires healthy training samples.")
    pool = json.loads(Path(path).read_text())
    expected = {sample.sample_id for sample in samples}
    if (pool.get("version") != 1 or pool.get("source") != "healthy_training_only"
            or pool.get("patch_size") != list(size)
            or set(pool.get("training_sample_ids", [])) != expected):
        raise ValueError("Hard-normal manifest must match the healthy training split and patch size.")
    records = pool.get("records", [])
    if not records:
        raise ValueError("Hard-normal pool is empty.")
    for record in records:
        if (record.get("sample_id") not in expected
                or any(type(record.get(key)) is not int or record[key] < 0 for key in ("top", "left"))
                or not math.isfinite(record.get("score", float("nan")))):
            raise ValueError("Invalid hard-normal patch or sample outside the healthy training split.")
    return pool


@torch.no_grad()
def mine_pool(model, samples, data, device, checkpoint, output, *, batch_size=8,
              top_fraction=.01, per_image_limit=4, max_iou=.25, capacity=1024):
    """Score native-resolution crops; limit overlapping crops from each source."""
    if not samples or any(sample.label != "good" for sample in samples):
        raise ValueError("Mining requires nonempty healthy training samples only.")
    if not 0 < top_fraction <= 1 or not 0 <= max_iou < 1 or per_image_limit < 1 or capacity < 1:
        raise ValueError("Invalid hard-normal mining settings.")
    model.eval()
    records = []
    ph, pw = data.patch_size
    for index, sample in enumerate(samples):
        image = image_tensor(read_image(sample.image_path), None, data.image_scale)
        probability = tiled_logits(model, image, device, data.patch_size, data.patch_overlap,
                                    batch_size).softmax(0)[1].numpy()
        candidates = []
        for top, left in product(tile_starts(probability.shape[0], ph, data.patch_overlap),
                                 tile_starts(probability.shape[1], pw, data.patch_overlap)):
            values = probability[top:top+ph, left:left+pw].ravel()
            count = max(1, math.ceil(values.size * top_fraction))
            score = float(np.partition(values, values.size-count)[-count:].mean())
            candidates.append((score, top, left))
        selected = []
        for score, top, left in sorted(candidates, reverse=True):
            if all(box_iou((top, left), (r["top"], r["left"]), data.patch_size) <= max_iou for r in selected):
                selected.append({"sample_id": sample.sample_id, "top": top, "left": left, "score": score})
            if len(selected) == per_image_limit:
                break
        records.extend(selected)
        print(f"Mining healthy train images {index+1}/{len(samples)}", flush=True)
    records.sort(key=lambda item: (-item["score"], item["sample_id"], item["top"], item["left"]))
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b""):
            digest.update(chunk)
    pool = {"version": 1, "source": "healthy_training_only", "patch_size": list(data.patch_size),
            "training_sample_ids": sorted(s.sample_id for s in samples),
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
            "scoring": "mean_top_fraction_anomaly_probability", "top_fraction": top_fraction,
            "per_image_limit": per_image_limit, "max_iou": max_iou, "capacity": capacity,
            "records": records[:capacity]}
    Path(output).write_text(json.dumps(pool, indent=2) + "\n")
    return pool
