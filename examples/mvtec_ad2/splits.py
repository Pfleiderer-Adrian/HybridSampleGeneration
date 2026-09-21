"""Deterministic, persisted train/validation/test splits for MVTec AD 2."""
import hashlib
import json
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


def _select_groups(groups, target, rng, remaining_groups):
    """Find the closest attainable image count without dividing a group."""
    groups = list(groups)
    rng.shuffle(groups)
    candidates = {0: ()}
    for index, group in enumerate(groups):
        for count, selected in list(candidates.items()):
            if len(selected) >= len(groups) - remaining_groups:
                continue
            new_count = count + len(group)
            selection = (*selected, index)
            if new_count not in candidates or len(selection) < len(candidates[new_count]):
                candidates[new_count] = selection
    viable = [(count, selected) for count, selected in candidates.items() if selected]
    if not viable:
        raise ValueError('Too few acquisition groups for disjoint train/validation/test splits.')
    _, selected = min(viable, key=lambda item: (abs(item[0] - target), item[0]))
    selected = set(selected)
    return ([sample for index in selected for sample in groups[index]],
            [group for index, group in enumerate(groups) if index not in selected])


def create_grouped_manifest(category_root, config=None, *, test_positive_fraction=.25):
    """Split capture groups and adjust healthy test counts to the requested ratio."""
    from collections import defaultdict
    from examples.mvtec_ad2.dataset import acquisition_group
    from examples.mvtec_ad2.downstream.transforms import read_image

    config = config or SplitConfiguration()
    config.validate()
    if config.test_fraction == 0 or not 0 < test_positive_fraction < 1:
        raise ValueError('A nonempty test split and a valid positive fraction are required.')
    samples = discover_samples(category_root)
    grouped = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if sample.split == 'test_private':
            continue
        group = acquisition_group(sample)
        if sample.label != 'good':
            if sample.mask_path is None or not np.any(read_image(sample.mask_path) > 0):
                raise ValueError(f'Positive sample has no usable mask: {sample.sample_id}')
        grouped[sample.label][group].append(sample)
    if 'good' not in grouped or len(grouped) < 2:
        raise ValueError('Healthy images and annotated anomalies are required.')
    rng = np.random.default_rng(config.seed)
    partitions = {name: [] for name in ('train', 'validation', 'test')}
    for label in sorted(set(grouped) - {'good'}):
        groups = [values for _, values in sorted(grouped[label].items())]
        total = sum(map(len, groups))
        test, remaining = _select_groups(groups, max(1, round(total * config.test_fraction)), rng, 2)
        validation, training = _select_groups(remaining, max(1, round(total * config.validation_fraction)), rng, 1)
        partitions['test'].extend(test)
        partitions['validation'].extend(validation)
        partitions['train'].extend(sample for group in training for sample in group)
    positives = len(partitions['test'])
    healthy_groups = [values for _, values in sorted(grouped['good'].items())]
    healthy_count = sum(map(len, healthy_groups))
    healthy_target = round(positives * (1 - test_positive_fraction) / test_positive_fraction)
    if healthy_target > healthy_count:
        raise ValueError('Insufficient healthy samples for the requested test ratio.')
    test, remaining = _select_groups(healthy_groups, healthy_target, rng, 2)
    validation, training = _select_groups(remaining, max(1, round(healthy_count * config.validation_fraction)), rng, 1)
    partitions['test'].extend(test)
    partitions['validation'].extend(validation)
    partitions['train'].extend(sample for group in training for sample in group)
    manifest = {
        'version': 1,
        'category_root': str(Path(category_root).resolve()),
        'split': asdict(config),
        'grouping': {'rule': 'source_split/label/numeric_capture_id',
                     'test_positive_fraction': test_positive_fraction},
        'partitions': {name: [{**_serialize(sample), 'acquisition_group': acquisition_group(sample)}
                              for sample in sorted(values, key=lambda sample: str(sample.image_path))]
                       for name, values in partitions.items()},
    }
    manifest['fingerprint'] = _fingerprint(manifest)
    _validate_manifest(manifest)
    validate_grouped_manifest(manifest)
    return manifest


def validate_grouped_manifest(manifest):
    """Validate disjoint acquisition groups and permitted evaluation sources."""
    from examples.mvtec_ad2.dataset import acquisition_group
    _validate_manifest(manifest)
    owners = {}
    for partition in ('train', 'validation', 'test'):
        samples = manifest_samples(manifest, partition)
        if not samples or not any(s.label == 'good' for s in samples) or not any(s.label != 'good' for s in samples):
            raise ValueError(f'{partition} requires both healthy and positive samples.')
        for item, sample in zip(manifest['partitions'][partition], samples):
            if sample.split == 'test_private' or 'test_private' in sample.image_path.parts:
                raise ValueError('Private test data is excluded.')
            group = acquisition_group(sample)
            if item.get('acquisition_group') != group:
                raise ValueError('Acquisition group metadata is inconsistent.')
            if group in owners and owners[group] != partition:
                raise ValueError(f'Acquisition group crosses partitions: {group}')
            owners[group] = partition
