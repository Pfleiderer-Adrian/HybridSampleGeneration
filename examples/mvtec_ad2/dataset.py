"""MVTec AD 2 discovery and adapter for the hybrid generator."""
from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.common.image_io import IMAGE_EXTENSIONS, ensure_chw, load_image_array
from hybrid_sample_generator.domain.input_sample import InputSample

CONTROL_SPLITS = ("train", "validation", "val")
ANOMALY_SPLIT = "test_public"
NORMAL_LABEL = "good"

@dataclass(frozen=True)
class MVTecAD2Sample:
    image_path: Path
    mask_path: Path | None
    sample_id: str
    split: str
    label: str

class MVTecAD2Dataloader:
    """Yield channel-first MVTec samples through the library's typed boundary."""
    def __init__(self, samples: Sequence[MVTecAD2Sample]) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("MVTecAD2Dataloader requires at least one sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        for sample in self.iter_input_samples():
            yield sample.image, sample.segmentation, sample.source_name

    def iter_input_samples(self) -> Iterator[InputSample]:
        for sample in self.samples:
            image = ensure_chw(load_image_array(str(sample.image_path))).astype(np.float32, copy=False)
            if sample.mask_path is None:
                segmentation = np.zeros((1, *image.shape[1:]), dtype=np.float32)
            else:
                segmentation = ensure_chw(load_image_array(str(sample.mask_path))).astype(np.float32, copy=False)
                segmentation = np.where(segmentation[:1] > 0, 1.0, 0.0).astype(np.float32, copy=False)
            yield InputSample(
                image=image,
                segmentation=segmentation,
                source_name=sample.sample_id,
                source_image_path=str(sample.image_path),
                source_segmentation_path=None if sample.mask_path is None else str(sample.mask_path),
                metadata={"split": sample.split, "label": sample.label},
            )

def discover_samples(category_root: Path | str) -> list[MVTecAD2Sample]:
    """Find healthy controls and public annotated anomalies for one category."""
    root = Path(category_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec AD 2 category directory does not exist: {root}")
    samples = _collect_controls(root, include_public=True)
    samples.extend(_collect_anomalies(root))
    if not samples:
        raise ValueError(f"No MVTec AD 2 samples found below {root}.")
    return samples

def _collect_controls(root: Path, *, include_public: bool) -> list[MVTecAD2Sample]:
    splits = [*CONTROL_SPLITS, *([ANOMALY_SPLIT] if include_public else [])]
    return [sample for split in splits for sample in _collect_good(root, split)]

def _collect_good(root: Path, split: str) -> list[MVTecAD2Sample]:
    folder = root / split / NORMAL_LABEL
    if not folder.is_dir():
        return []
    return [MVTecAD2Sample(path, None, _sample_id(split, NORMAL_LABEL, path), split, NORMAL_LABEL) for path in _images(folder)]

def _collect_anomalies(root: Path) -> list[MVTecAD2Sample]:
    test_root = root / ANOMALY_SPLIT
    if not test_root.is_dir():
        return []
    folders = sorted(path for path in test_root.iterdir() if path.is_dir() and path.name not in {NORMAL_LABEL, "ground_truth"})
    samples = []
    for folder in folders:
        for image_path in _images(folder):
            mask_path = _mask_path(root, folder.name, image_path)
            if mask_path is None:
                print(f"Warning: missing mask for {image_path}; skipping sample.")
                continue
            samples.append(MVTecAD2Sample(image_path, mask_path, _sample_id(ANOMALY_SPLIT, folder.name, image_path), ANOMALY_SPLIT, folder.name))
    return samples

def _images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

def _mask_path(root: Path, label: str, image_path: Path) -> Path | None:
    stem = image_path.stem
    candidates = (
        root / ANOMALY_SPLIT / "ground_truth" / label / f"{stem}_mask.png",
        root / ANOMALY_SPLIT / "ground_truth" / label / f"{stem}.png",
        root / "ground_truth" / label / f"{stem}_mask.png",
        root / "ground_truth" / label / f"{stem}.png",
        root / "ground_truth_public" / label / f"{stem}_mask.png",
        root / "ground_truth_public" / label / f"{stem}.png",
        root / "test_public_ground_truth" / label / f"{stem}_mask.png",
        root / "test_public_ground_truth" / label / f"{stem}.png",
    )
    return next((path for path in candidates if path.is_file()), None)

def _sample_id(split: str, label: str, image_path: Path) -> str:
    return f"{_safe(split)}_{_safe(label)}_{_safe(image_path.stem)}{image_path.suffix.lower()}"

def _safe(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", value)).strip("_")


def acquisition_group(sample: MVTecAD2Sample) -> str:
    """Keep numbered exposure/position variants together within a source split.

    MVTec AD 2 filenames encode a capture ID before the first underscore.
    Split and label are separate namespaces: identical numbers alone do not
    establish that images from different source directories show one object.
    Unrecognized names require an explicit grouping decision before training.
    """
    match = re.fullmatch(r'(\d+)(?:_(regular|overexposed|underexposed|shift_\d+))?', sample.image_path.stem)
    if match is None:
        raise ValueError(f'Unknown acquisition naming convention: {sample.image_path}')
    return f'{sample.split}/{sample.label}/{int(match.group(1))}'
