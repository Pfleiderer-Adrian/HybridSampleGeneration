"""Training inputs for AnomalyDiffusion from the real defects of a split manifest.

Training windows come from the anomalous images of the ``train`` partition,
validation windows from the ``validation`` partition. Healthy images and the
test partition are never used.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import SCALE, WINDOW, defect_windows

UPSTREAM_COMMIT = "b85297f333ebfc59b3863338445b81363efb9761"
PROFILE = "scale2x_last50k_aar"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path):
    """Split manifest without downstream extras such as ``hybrid_provenance``."""
    split = json.loads(Path(path).read_text())
    split.pop("hybrid_provenance", None)
    train = {s["sample_id"] for s in split["partitions"]["train"]}
    held_out = {s["sample_id"] for p in ("validation", "test") for s in split["partitions"][p]}
    if train & held_out:
        raise ValueError("Training and held-out sample IDs overlap.")
    return split


def prepare(category, split_path, output):
    """Write ``training_patches.json``, ``validation_patches.json`` and the protocol."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared inputs: {output}")
    split = load_split(split_path)
    patches = output / "patches"
    patches.mkdir(parents=True)
    stats = {}
    for partition, stem in (("train", "training"), ("validation", "validation")):
        rows, components, images = [], 0, set()
        for sample in split["partitions"][partition]:
            if sample["label"] == "good":
                continue
            if not sample.get("mask_path"):
                raise ValueError(f"Defect image without mask: {sample['sample_id']}")
            image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
            mask = np.asarray(Image.open(sample["mask_path"]).convert("L")) > 0
            if not mask.any() or image.shape[:2] != mask.shape:
                raise ValueError(f"Invalid source mask: {sample['sample_id']}")
            image_hash, mask_hash = sha256(sample["image_path"]), sha256(sample["mask_path"])
            used = set()
            for window_image, window_mask, crop, component in defect_windows(image, mask):
                index = f"{stem}_{len(rows):06d}"
                image_path, mask_path = patches / f"{index}_image.png", patches / f"{index}_mask.png"
                Image.fromarray(window_image).save(image_path)
                Image.fromarray(window_mask.astype("uint8") * 255).save(mask_path)
                used.add(component)
                rows.append(dict(sample_id=sample["sample_id"], label="bad", image_path=str(image_path),
                                 mask_path=str(mask_path), crop_xywh=crop, scale=SCALE,
                                 native_positive_pixels=int(window_mask.sum()) // SCALE ** 2,
                                 source_component=component, source_image_sha256=image_hash,
                                 source_mask_sha256=mask_hash))
            components += len(used)
            images.add(sample["sample_id"])
        if not rows:
            raise ValueError(f"No defect windows in the {partition} partition.")
        stats[partition] = dict(images=len(images), components=components, patches=len(rows))
        write_json(output / f"{stem}_patches.json", dict(
            version=1, category=category, split_fingerprint=split["fingerprint"],
            upstream_commit=UPSTREAM_COMMIT, resolution=f"centered_{WINDOW}_to{WINDOW * SCALE}_scale{SCALE}",
            rows=rows))
    write_json(output / "split_manifest.json", split)
    write_json(output / "protocol.json", dict(
        profile=PROFILE, category=category, upstream_commit=UPSTREAM_COMMIT, scale=SCALE, window=WINDOW,
        split_fingerprint=split["fingerprint"], stats=stats,
        validation_patches_sha256=sha256(output / "validation_patches.json")))
    return stats


def read_inputs(root):
    """Load prepared inputs and verify that training windows are training-only."""
    root = Path(root).resolve()
    patches = json.loads((root / "training_patches.json").read_text())
    validation = json.loads((root / "validation_patches.json").read_text())
    split = json.loads((root / "split_manifest.json").read_text())
    train = {r["sample_id"] for r in split["partitions"]["train"]}
    held_out = {r["sample_id"] for p in ("validation", "test") for r in split["partitions"][p]}
    used = {r["sample_id"] for r in patches["rows"]}
    checked = {r["sample_id"] for r in validation["rows"]}
    if (not used or not used <= train or used & held_out or not checked
            or not checked <= {r["sample_id"] for r in split["partitions"]["validation"]}
            or patches["split_fingerprint"] != split["fingerprint"]
            or validation["split_fingerprint"] != split["fingerprint"]):
        raise ValueError("Prepared generator inputs do not match the split partitions.")
    return root, patches, validation, split
