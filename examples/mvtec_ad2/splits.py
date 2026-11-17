"""Deterministic, persisted train/validation/test splits for MVTec AD 2."""
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.mvtec_ad2.dataset import MVTecAD2Sample, discover_samples


@dataclass(frozen=True)
class SplitConfiguration:
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    seed: int = 42

    def validate(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("split seed must be a nonnegative integer.")
        if type(self.test_fraction) not in (float, int) or not 0 <= self.test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1).")
        if type(self.validation_fraction) not in (float, int) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1).")
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("split fractions must leave a nonempty training partition.")

def create_manifest(category_root, config: SplitConfiguration | None = None):
    config = config or SplitConfiguration()
    config.validate()
    groups = {}
    unique = {str(sample.image_path.resolve()): sample for sample in discover_samples(category_root)}
    for _, sample in sorted(unique.items()):
        groups.setdefault(sample.label, []).append(sample)
    if "good" not in groups or len(groups) < 2:
        raise ValueError("Both healthy images and annotated anomalies are required.")
    rng = np.random.default_rng(config.seed)
    partitions = {name: [] for name in ("train", "validation", "test")}
    for label, samples in sorted(groups.items()):
        samples = list(samples)
        rng.shuffle(samples)
        validation_count = max(1, round(len(samples) * config.validation_fraction))
        test_count = max(1, round(len(samples) * config.test_fraction)) if config.test_fraction else 0
        if validation_count + test_count >= len(samples):
            raise ValueError(f"Too few {label!r} samples for nonempty disjoint splits.")
        selected = {
            "validation": samples[:validation_count],
            "test": samples[validation_count:validation_count + test_count],
            "train": samples[validation_count + test_count:],
        }
        for partition, values in selected.items():
            partitions[partition].extend(_serialize(sample) for sample in values)
    manifest = {
        "version": 1,
        "category_root": str(Path(category_root).expanduser().resolve()),
        "split": asdict(config),
        "partitions": partitions,
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    return manifest


def capture_group(sample):
    """Keep acquisition variants together without conflating unrelated folders."""
    match = re.match(r"^(\d+)(?:_|$)", Path(sample.image_path).stem)
    if match is None:
        raise ValueError(f"Cannot determine numeric capture ID: {sample.image_path}")
    return sample.split, sample.label, int(match.group(1))


def create_grouped_manifest(category_root, config: SplitConfiguration | None = None):
    """Stratified train/validation split with disjoint acquisition groups.

    Randomize groups with the configured seed and select whole groups nearest
    to the desired per-label validation image count. Exact percentages are only
    possible when group sizes permit them. Existing image-level splits are not
    modified by this opt-in entry point.
    """
    config = config or SplitConfiguration(test_fraction=0., validation_fraction=.4)
    config.validate()
    if config.test_fraction:
        raise ValueError("Grouped manifest currently supports train/validation only.")
    unique = {str(s.image_path.resolve()): s for s in discover_samples(category_root)}
    labels = {}
    for _, sample in sorted(unique.items()):
        labels.setdefault(sample.label, {}).setdefault(capture_group(sample), []).append(sample)
    if "good" not in labels or len(labels) < 2:
        raise ValueError("Both healthy images and annotated anomalies are required.")
    rng = np.random.default_rng(config.seed)
    partitions = {name: [] for name in ("train", "validation", "test")}
    counts = {}
    for label, groups in sorted(labels.items()):
        if len(groups) < 2:
            raise ValueError(f"At least two independent {label!r} capture groups are required.")
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        target = max(1, min(total - 1, round(total * config.validation_fraction)))
        # Subset-sum keeps complete groups while approaching the target count.
        reachable = {0: ()}
        for index, key in enumerate(keys):
            size = len(groups[key])
            additions = {n + size: selected + (index,) for n, selected in list(reachable.items())
                         if n + size < total and n + size not in reachable}
            reachable.update(additions)
            if target in reachable:
                break
        selected_count = min((n for n in reachable if n), key=lambda n: (abs(n - target), n))
        held_out = set(reachable[selected_count])
        for index, key in enumerate(keys):
            part = "validation" if index in held_out else "train"
            partitions[part].extend(_serialize(s) for s in groups[key])
        counts[label] = {"total_images": total, "validation_images": selected_count,
                         "total_groups": len(keys), "validation_groups": len(held_out)}
    manifest = {"version": 1, "category_root": str(Path(category_root).expanduser().resolve()),
                "split": asdict(config), "grouping": {
                    "rule": "source_split/label/numeric_capture_id",
                    "selection": "seeded group shuffle; nearest per-label image-count subset",
                    "counts": counts}, "partitions": partitions}
    manifest["fingerprint"] = _fingerprint(manifest)
    return manifest

def load_or_create_manifest(path, category_root, config: SplitConfiguration | None = None):
    config = config or SplitConfiguration()
    config.validate()
    path = Path(path)
    if path.is_file():
        manifest = json.loads(path.read_text())
        _validate_manifest(manifest)
        if manifest["category_root"] != str(Path(category_root).expanduser().resolve()):
            raise ValueError("Saved split belongs to a different category directory.")
        if manifest["split"] != asdict(config):
            raise ValueError("Requested split settings differ from the saved manifest.")
        return manifest
    manifest = create_manifest(category_root, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def manifest_samples(manifest, partition, healthy_only=False):
    if partition not in manifest["partitions"]:
        raise KeyError(f"Unknown split partition: {partition}")
    return [
        MVTecAD2Sample(
            image_path=Path(item["image_path"]),
            mask_path=Path(item["mask_path"]) if item["mask_path"] else None,
            sample_id=item["sample_id"], split=item["split"], label=item["label"],
        )
        for item in manifest["partitions"][partition]
        if not healthy_only or item["label"] == "good"
    ]

def verify_repository(repository, manifest):
    """Ensure that the generator only ingested the training partition."""
    allowed = {item["image_path"] for item in manifest["partitions"]["train"]}
    for original in repository.list_original_samples():
        source = original.metadata.get("source_image_path")
        if source is None or str(Path(source).resolve()) not in allowed:
            raise ValueError("Study contains unknown or held-out originals.")

def _serialize(sample):
    values = asdict(sample)
    values["image_path"] = str(sample.image_path.resolve())
    values["mask_path"] = str(sample.mask_path.resolve()) if sample.mask_path else None
    return values

def _fingerprint(manifest):
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()

def _validate_manifest(manifest):
    SplitConfiguration(**manifest["split"]).validate()
    if manifest.get("fingerprint") != _fingerprint(manifest):
        raise ValueError("Split manifest is inconsistent or has been modified.")
    paths = [item["image_path"] for values in manifest["partitions"].values() for item in values]
    if len(paths) != len(set(paths)):
        raise ValueError("Split partitions contain duplicate images.")
    for values in manifest["partitions"].values():
        for item in values:
            for name in ("image_path", "mask_path"):
                if item[name] is not None and not Path(item[name]).is_file():
                    raise FileNotFoundError(f"Split source is missing: {item[name]}")
