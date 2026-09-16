"""Create, locate and restore category studies."""

from collections.abc import Iterable
from pathlib import Path

from examples.mvtec_ad2.configuration import SplitConfiguration, load_config_file
from examples.mvtec_ad2.presets import MVTECAD2_CATEGORIES, create_configuration
from examples.mvtec_ad2.discovery import (
    discover_mvtecad2_categories, normalize_categories, _validate_dataset_root,
)
from examples.mvtec_ad2.records import MVTecAD2Study
from examples.mvtec_ad2.splits import load_manifest, load_or_create_manifest


def prepare_studies(
    root: Path | str,
    categories: str | Iterable[str] | None = None,
    *,
    save_path: Path | str | None = None,
    splits: SplitConfiguration | None = None,
) -> list[MVTecAD2Study]:
    """Prepare new studies and persist their split; config is saved at execution."""
    root = Path(root)
    _validate_dataset_root(root)
    names = normalize_categories(
        discover_mvtecad2_categories(root) if categories is None else categories
    )
    if splits is not None:
        splits.validate()
    prepared = []
    for category in names:
        category_root = root / category
        if not category_root.is_dir():
            raise FileNotFoundError(f"Missing MVTec AD 2 category folder: {category_root}")
        config = create_configuration(
            category, save_path=Path(save_path) / category if save_path is not None else None,
        )
        if Path(config.study.paths.configuration_file).exists():
            raise ValueError(f"Study already exists. Open it with open_study({config.study.folder!r}).")
        prepared.append((category, category_root, config))

    studies = []
    for category, category_root, config in prepared:
        manifest = load_or_create_manifest(config.study.folder, category_root, splits)
        config.study.seed = config.matching.seed = manifest["seed"]
        studies.append(MVTecAD2Study(category, category_root, config, None, manifest))
    return studies


def open_study(study_folder: Path | str) -> MVTecAD2Study:
    """Load a study by its actual path without discovery, presets or writes."""
    folder = Path(study_folder).resolve()
    config = load_config_file(folder / "configuration.json")
    config.study.folder = str(folder)
    category = next((name for name in MVTECAD2_CATEGORIES
                     if config.study.name.startswith(f"mvtecad2_{name}_")), None)
    if category is None:
        raise ValueError(f"Not an MVTec AD 2 study: {config.study.name}")
    manifest = load_manifest(folder) if (folder / "split_manifest.json").is_file() else None
    return MVTecAD2Study(category, None, config, None, manifest)


def find_studies(save_path: Path | str, categories: str | Iterable[str] | None = None) -> list[Path]:
    """Find saved category studies without reconstructing names from presets."""
    root = Path(save_path)
    names = MVTECAD2_CATEGORIES if categories is None else normalize_categories(categories)
    folders = []
    for category in names:
        matches = sorted((root / category / "results").glob("*/configuration.json"))
        if not matches and categories is not None:
            raise FileNotFoundError(f"No saved study for {category} below {root}")
        folders.extend(path.parent for path in matches)
    if not folders:
        raise FileNotFoundError(f"No saved MVTec studies below {root}")
    return folders


def normalize_study_folders(study_folders: Iterable[Path | str]) -> list[Path]:
    """Resolve paths once and prevent repeated execution of the same study."""
    folders = [Path(folder).expanduser().resolve() for folder in study_folders]
    if not folders:
        raise ValueError("study_folders must not be empty.")
    if len(set(folders)) != len(folders):
        raise ValueError("Duplicate study folders are not allowed.")
    return folders
