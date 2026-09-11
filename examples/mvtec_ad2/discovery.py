"""Dataset discovery for the MVTec AD 2 example integration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from examples.common.image_io import IMAGE_EXTENSIONS
from examples.mvtec_ad2.configuration import canonical_category, safe_name
from examples.mvtec_ad2.records import MVTecAD2Sample
from examples.mvtec_ad2.settings import MVTECAD2_ROOT

CONTROL_SPLITS = ("train", "validation", "val")
ANOMALY_SPLIT = "test_public"
ANOMALY_LABEL = "bad"
NORMAL_LABEL = "good"

def discover_mvtecad2_categories(root: Path | str = MVTECAD2_ROOT) -> list[str]:
    """
    Discover available MVTec AD 2 categories under the dataset root.
    """

    root = Path(root)
    categories = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "train").is_dir() and (child / ANOMALY_SPLIT).is_dir():
            categories.append(child.name)
    return categories


def _normalize_categories(
    root: Path,
    categories: str | Iterable[str] | None,
) -> list[str]:
    if categories is None:
        return discover_mvtecad2_categories(root)

    if isinstance(categories, str):
        raw_categories = [categories]
    else:
        raw_categories = list(categories)

    normalized_categories = []
    for category in raw_categories:
        if not isinstance(category, str):
            raise TypeError(f"MVTec AD 2 category names must be strings, got {type(category).__name__}.")
        normalized_categories.append(canonical_category(category))

    return normalized_categories


def _collect_public_anomaly_samples(category_root: Path) -> list[MVTecAD2Sample]:
    samples: list[MVTecAD2Sample] = []
    test_root = category_root / ANOMALY_SPLIT
    if not test_root.is_dir():
        return samples

    anomaly_dirs = [test_root / ANOMALY_LABEL] if (test_root / ANOMALY_LABEL).is_dir() else []
    if not anomaly_dirs:
        anomaly_dirs = sorted(
            p
            for p in test_root.iterdir()
            if p.is_dir() and p.name not in {NORMAL_LABEL, "ground_truth"}
        )

    for label_dir in anomaly_dirs:
        label = label_dir.name
        for image_path in _iter_images(label_dir):
            mask_path = _find_mask_path(category_root, label, image_path)
            if mask_path is None:
                print(f"Warning: Missing mask for {image_path}. Skipping positive sample.")
                continue

            samples.append(
                MVTecAD2Sample(
                    image_path=image_path,
                    mask_path=mask_path,
                    sample_id=_make_sample_id(ANOMALY_SPLIT, label, image_path),
                    split=ANOMALY_SPLIT,
                    label=label,
                )
            )
    return samples


def _collect_control_samples(
    category_root: Path,
    *,
    include_public_good_controls: bool,
) -> list[MVTecAD2Sample]:
    samples: list[MVTecAD2Sample] = []
    for split in CONTROL_SPLITS:
        samples.extend(_collect_good_samples(category_root, split))

    if include_public_good_controls:
        samples.extend(_collect_good_samples(category_root, ANOMALY_SPLIT))

    return samples


def _collect_good_samples(category_root: Path, split: str) -> list[MVTecAD2Sample]:
    good_dir = category_root / split / NORMAL_LABEL
    if not good_dir.is_dir():
        return []

    return [
        MVTecAD2Sample(
            image_path=image_path,
            mask_path=None,
            sample_id=_make_sample_id(split, NORMAL_LABEL, image_path),
            split=split,
            label=NORMAL_LABEL,
        )
        for image_path in _iter_images(good_dir)
    ]


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_mask_path(category_root: Path, label: str, image_path: Path) -> Path | None:
    stem = image_path.stem
    candidates = [
        category_root / ANOMALY_SPLIT / "ground_truth" / label / f"{stem}_mask.png",
        category_root / ANOMALY_SPLIT / "ground_truth" / label / f"{stem}.png",
        category_root / "ground_truth" / label / f"{stem}_mask.png",
        category_root / "ground_truth" / label / f"{stem}.png",
        category_root / "ground_truth_public" / label / f"{stem}_mask.png",
        category_root / "ground_truth_public" / label / f"{stem}.png",
        category_root / "test_public_ground_truth" / label / f"{stem}_mask.png",
        category_root / "test_public_ground_truth" / label / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _make_sample_id(split: str, label: str, image_path: Path) -> str:
    filename = safe_name(image_path.stem) + image_path.suffix.lower()
    return f"{safe_name(split)}_{safe_name(label)}_{filename}"


def _validate_dataset_root(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"MVTec AD 2 root does not exist: {root}. "
            "Set MVTECAD2_ROOT via the environment variable or edit examples/mvtec_ad2/settings.py."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"MVTec AD 2 root is not a directory: {root}")
