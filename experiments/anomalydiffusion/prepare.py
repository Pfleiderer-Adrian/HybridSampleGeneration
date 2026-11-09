"""Freeze V3 inputs and export native-resolution generator patches (no training)."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

UPSTREAM_COMMIT = "b85297f333ebfc59b3863338445b81363efb9761"
REFERENCE = Path("/mnt/results/mvtec2/submissions/v3_variant2_best_20261025T144555Z")
HYBRID_REFERENCE = Path("/mnt/results/mvtec2/submissions/v3_variant5_best_20261023T130904Z")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_patch(array, x, y, size=256):
    """Preserve native pixels; pad only the right/bottom edge of small images."""
    patch = array[y:y + size, x:x + size]
    pad = [(0, size - patch.shape[0]), (0, size - patch.shape[1])]
    if patch.ndim == 3:
        pad.append((0, 0))
    return np.pad(patch, pad, mode="constant")


def latent_pixels(mask):
    """Match pinned upstream tensor Resize(32,32), bilinear, antialias=False."""
    import torch
    values = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    return int((torch.nn.functional.interpolate(values, (32, 32), mode="bilinear",
                                                align_corners=False) >= .5).sum())


def prepare_validation(output):
    """Native held-out patches from the existing V3 validation partition only."""
    output = Path(output).resolve()
    split = json.loads((output / "split_manifest.json").read_text())
    training = json.loads((output / "training_patches.json").read_text())
    path = output / "validation_patches.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data["split_fingerprint"] != split["fingerprint"]:
            raise ValueError("Validation patches belong to a different split.")
        return data
    folder = output / "validation_patches"
    folder.mkdir(exist_ok=True)
    rows = []
    for sample in split["partitions"]["validation"]:
        if sample["label"] == "good":
            continue
        image = np.array(Image.open(sample["image_path"]).convert("RGB"))
        mask = np.array(Image.open(sample["mask_path"]).convert("L")) > 0
        if mask.shape != image.shape[:2] or not mask.any():
            raise ValueError(f"Invalid held-out mask: {sample['sample_id']}")
        ys, xs = np.nonzero(mask)
        coordinates = sorted(set(zip((xs // 256 * 256).tolist(), (ys // 256 * 256).tolist())))
        for x, y in coordinates:
            name = f"{len(rows):07d}"
            image_path, mask_path = folder / f"{name}_image.png", folder / f"{name}_mask.png"
            Image.fromarray(native_patch(image, x, y)).save(image_path)
            Image.fromarray(native_patch(mask, x, y).astype(np.uint8) * 255).save(mask_path)
            rows.append({"sample_id": sample["sample_id"], "label": sample["label"],
                         "image_path": str(image_path), "mask_path": str(mask_path),
                         "crop_xywh": [x, y, 256, 256]})
    if not rows:
        raise ValueError("Generator validation requires held-out anomalous patches.")
    data = {"version": 1, "category": training["category"], "split_fingerprint": split["fingerprint"],
            "partition": "validation", "resolution": "native_256_grid_patches", "rows": rows}
    write_json(path, data)
    return data


def prepare(category, output, reference=REFERENCE, hybrid_reference=HYBRID_REFERENCE):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared inputs: {output}")
    reference, hybrid_reference = Path(reference), Path(hybrid_reference)
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    hybrid_manifest = json.loads((hybrid_reference / "manifest.json").read_text())
    job = reference_manifest["jobs"][category]
    frozen = Path(job["run"])
    split = json.loads((frozen / "split_manifest.json").read_text())
    split.pop("hybrid_provenance", None)
    cfg = json.loads((frozen / "configuration.json").read_text())
    assert cfg["data"]["mode"] == "patch" and cfg["data"]["patch_size"] == [256, 256]
    assert cfg["seed"] == 42 and not split["partitions"]["test"]
    train_ids = {s["sample_id"] for s in split["partitions"]["train"]}
    held_out = {s["sample_id"] for p in ("validation", "test") for s in split["partitions"][p]}
    if train_ids & held_out:
        raise ValueError("Training and held-out sample IDs overlap.")
    hybrid_job = hybrid_manifest["jobs"][category]
    hybrid_split = json.loads((Path(hybrid_job["run"]) / "split_manifest.json").read_text())
    hybrid_provenance = hybrid_split.get("hybrid_provenance", [])
    hybrid_split.pop("hybrid_provenance", None)
    if split != hybrid_split:
        raise ValueError("The two frozen V3 submissions have different splits.")
    status = json.loads(Path(hybrid_job["selection_status"]).read_text())
    hybrid_study = Path(status["hybrid_study"])
    generator_configuration = hybrid_study / "configuration.json"
    if not generator_configuration.is_file():
        raise FileNotFoundError(generator_configuration)
    output.mkdir(parents=True)
    write_json(output / "split_manifest.json", split)
    write_json(output / "hybrid_provenance.json", hybrid_provenance)
    rows, audit = [], []
    rng = np.random.default_rng(42)
    for sample in split["partitions"]["train"]:
        image = np.array(Image.open(sample["image_path"]).convert("RGB"))
        h, w = image.shape[:2]
        healthy = sample["label"] == "good"
        mask = (np.array(Image.open(sample["mask_path"]).convert("L")) > 0) if not healthy else np.zeros((h, w), bool)
        if mask.shape != (h, w) or (not healthy and not mask.any()):
            raise ValueError(f"Invalid training mask: {sample['sample_id']}")
        if healthy:
            coords = sorted({(int(rng.integers(max(w - 255, 1))),
                              int(rng.integers(max(h - 255, 1)))) for _ in range(4)})
        else:
            # A disjoint native grid covers every defect pixel, including large defects.
            ys, xs = np.nonzero(mask)
            coords = sorted(set(zip((xs // 256 * 256).tolist(), (ys // 256 * 256).tolist())))
        source_hash = sha256(sample["image_path"])
        mask_hash = sha256(sample["mask_path"]) if not healthy else None
        for x, y in coords:
            key = f"{len(rows):07d}"
            folder = output / "patches"
            folder.mkdir(exist_ok=True)
            image_path = folder / f"{key}_image.png"
            Image.fromarray(native_patch(image, x, y)).save(image_path)
            patch_mask = native_patch(mask, x, y)
            row = {"sample_id": sample["sample_id"], "label": sample["label"],
                   "image_path": str(image_path), "crop_xywh": [x, y, 256, 256],
                   "original_wh": [w, h], "source_image_sha256": source_hash}
            if not healthy:
                mask_path = folder / f"{key}_mask.png"
                Image.fromarray(patch_mask.astype(np.uint8) * 255).save(mask_path)
                row["mask_path"] = str(mask_path)
                row["source_mask_sha256"] = mask_hash
                audit.append({"key": key, "sample_id": sample["sample_id"],
                              "native_pixels": int(patch_mask.sum()),
                              "latent_pixels": latent_pixels(patch_mask)})
            rows.append(row)
    defects = [r for r in rows if r["label"] != "good"]
    if not defects or not any(r["label"] == "good" for r in rows):
        raise ValueError("Both healthy and anomalous training patches are required.")
    write_json(output / "training_patches.json", {"version": 1, "category": category,
               "split_fingerprint": split["fingerprint"], "upstream_commit": UPSTREAM_COMMIT,
               "resolution": "native_256_grid_patches", "rows": rows})
    write_json(output / "resolution_audit.json", {"patches": audit,
               "vanished_latent_masks": sum(r["latent_pixels"] == 0 for r in audit),
               "policy": "Default: max-pool mask support onto latent grid; native masks stay unchanged. Original bilinear mode refuses vanishing masks."})
    prepare_validation(output)
    for mixed in (False, True):
        variant = "real_anomalydiffusion_hybrid" if mixed else "real_anomalydiffusion"
        configured = deepcopy(cfg)
        configured["data"].update(source_quotas={"normal": 500, "real": 100,
            "anomalydiffusion": 200 if mixed else 400, **({"hybrid": 200} if mixed else {})},
            source_quota_rotation=None, hybrid_fraction=.2 if mixed else 0.,
            anomalydiffusion_manifest=str(output / "generated" / "manifest.json"))
        write_json(output / f"{variant}.json", configured)
    write_json(output / "protocol.json", {"category": category, "seed": 42,
               "reference_run": str(frozen), "split_fingerprint": split["fingerprint"],
               "hybrid_study": str(hybrid_study),
               "hybrid_provenance_sha256": sha256(output / "hybrid_provenance.json"),
               "hybrid_configuration_sha256": sha256(generator_configuration),
               "upstream_commit": UPSTREAM_COMMIT,
               "mask_steps": 30001, "appearance_steps": 100000,
               "training_engine": "reference_lightning_v1", "validation_monitor": "val/loss_simple_ema",
               "validation_patches_sha256": sha256(output / "validation_patches.json"),
               "checkpoint_policy": "Reference top1+last and step-numbered embeddings; full state retained for restore",
               "batch_size": 4, "base_learning_rate": .005, "learning_rate": .02,
               "latent_mask_policy": "maxpool",
               "appearance_tokens": 4, "spatial_tokens": 4, "mask_tokens": 16,
               "defaults": "Pinned appearance YAML and mask entrypoint settings; no private-test tuning.",
               "adaptations": ["MVTec AD 2 train-only split adapter",
                               "Native 256 patches instead of whole-image resizing",
                               "No upstream 512 augmentation; no geometric mask scaling",
                               "Max-pool mask support on latent grid; native labels unchanged",
                               "Lossless PNG image/mask output"]})
    return {"category": category, "patches": len(rows), "defect_patches": len(defects),
            "vanished_latent_masks": sum(r["latent_pixels"] == 0 for r in audit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--hybrid-reference", type=Path, default=HYBRID_REFERENCE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.category, args.output, args.reference, args.hybrid_reference), indent=2))


if __name__ == "__main__":
    main()
