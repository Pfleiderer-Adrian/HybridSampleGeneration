"""Custom stratified splits established before generator training."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from examples.mvtec_ad2.discovery import _collect_control_samples, _collect_public_anomaly_samples
from examples.mvtec_ad2.records import MVTecAD2Sample


def create_manifest(category_root, config):
    config.validate()
    samples = _collect_control_samples(Path(category_root), include_public_good_controls=True)
    samples += _collect_public_anomaly_samples(Path(category_root))
    unique = {str(sample.image_path.resolve()): sample for sample in samples}
    groups = {}
    for path, sample in sorted(unique.items()):
        groups.setdefault(sample.label, []).append(sample)
    if "good" not in groups or len(groups) < 2:
        raise ValueError("Both healthy images and annotated anomalies are required.")
    rng = np.random.default_rng(config.seed)
    partitions = {key: [] for key in ("train", "validation", "test")}
    for label, group in sorted(groups.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * config.validation_fraction))
        n_test = max(1, round(len(group) * config.test_fraction)) if config.test_enabled else 0
        if n_val + n_test >= len(group):
            raise ValueError(f"Too few {label!r} samples for nonempty disjoint splits.")
        for key, selected in (("validation", group[:n_val]), ("test", group[n_val:n_val+n_test]), ("train", group[n_val+n_test:])):
            for sample in selected:
                item = asdict(sample)
                item["image_path"] = str(sample.image_path.resolve())
                item["mask_path"] = str(sample.mask_path.resolve()) if sample.mask_path else None
                partitions[key].append(item)
    result = {"version": 1, "seed": config.seed, "split": asdict(config), "partitions": partitions}
    result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def manifest_samples(manifest, partition, healthy_only=False):
    return [MVTecAD2Sample(**{**item, "image_path": Path(item["image_path"]),
                            "mask_path": Path(item["mask_path"]) if item["mask_path"] else None})
            for item in manifest["partitions"][partition]
            if not healthy_only or item["label"] == "good"]


def verify_repository(repository, manifest):
    """Reject even unused held-out originals: the generator may have seen them."""
    allowed = {item["image_path"] for item in manifest["partitions"]["train"]}
    for original in repository.list_original_samples():
        source = original.metadata.get("source_image_path")
        if source is None or str(Path(source).resolve()) not in allowed:
            raise ValueError("Study contains unknown or held-out originals. Generate a fresh split-specific study.")


def load_manifest(folder):
    """Read and validate a persisted split without creating or changing files."""
    from examples.mvtec_ad2.configuration import SplitConfiguration

    manifest = json.loads((Path(folder) / "split_manifest.json").read_text())
    SplitConfiguration(**manifest["split"]).validate()
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("Split manifest is inconsistent or has been modified.")
    return manifest


def load_or_create_manifest(folder, category_root, requested=None):
    from examples.mvtec_ad2.configuration import SplitConfiguration

    folder = Path(folder)
    path = folder / "split_manifest.json"
    if path.exists():
        manifest = load_manifest(folder)
        saved = SplitConfiguration(**manifest["split"])
        if requested is not None and asdict(requested) != asdict(saved):
            raise ValueError("Split settings differ from the saved study. Use a new save_path.")
        return manifest
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Existing study has no split manifest. Use a new save_path; legacy studies cannot be split retroactively.")
    manifest = create_manifest(category_root, requested or SplitConfiguration())
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
