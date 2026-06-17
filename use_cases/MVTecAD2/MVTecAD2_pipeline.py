from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesizer.Configuration import Configuration, load_config_file
from synthesizer.HybridDataGenerator import HybridDataGenerator
from use_cases.image_2d.ImageDataloader import ensure_chw, save_image
from use_cases.image_2d.ImageDataloader import _load_image_array as load_image_array
from use_cases.MVTecAD2.MVTecAD2_configuration import (
    CATEGORY_ALIASES,
    create_mvtecad2_configuration,
)


MVTECAD2_ROOT = Path(
    os.environ.get("MVTECAD2_ROOT", r"path/to/mvtec_ad_2_dataset")
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
CONTROL_SPLITS = ("train", "validation", "val")
ANOMALY_SPLIT = "test_public"
ANOMALY_LABEL = "bad"
NORMAL_LABEL = "good"

PREPARE_USECASE_KEYS = {
    "load_negative_controls",
    "include_public_good_controls",
    "save_path",
    "results_root",
}
DEPRECATED_GENERATION_FLAGS = ("run_evaluation", "visualize_evaluation")
DEFAULT_GENERATION_STEPS = (
    "extract_anomalies",
    "train_generator",
    "generate_synthetic_anomalies",
    "create_matching",
    "generate_hybrid_samples",
    "save_config",
)
GENERATION_STEP_ORDER = (
    "extract_anomalies",
    "load_anomalies",
    "train_generator",
    "load_generator",
    "generate_synthetic_anomalies",
    "load_synthetic_anomalies",
    "create_matching",
    "load_matching",
    "generate_hybrid_samples",
    "save_config",
)
GENERATION_STEP_ALIASES = {
    "all": "all",
    "default": "all",
    "extract": "extract_anomalies",
    "extract_anomalies": "extract_anomalies",
    "load_anomalies": "load_anomalies",
    "load_anomaly_data": "load_anomalies",
    "train": "train_generator",
    "train_generator": "train_generator",
    "load_generator": "load_generator",
    "load_model": "load_generator",
    "generate_synth": "generate_synthetic_anomalies",
    "generate_synthetic_anomalies": "generate_synthetic_anomalies",
    "synth": "generate_synthetic_anomalies",
    "load_synth": "load_synthetic_anomalies",
    "load_synthetic_anomalies": "load_synthetic_anomalies",
    "load_synth_anomalies": "load_synthetic_anomalies",
    "matching": "create_matching",
    "create_matching": "create_matching",
    "create_matching_dict": "create_matching",
    "load_matching": "load_matching",
    "load_matching_dict": "load_matching",
    "fusion": "generate_hybrid_samples",
    "hybrid": "generate_hybrid_samples",
    "generate_hybrid": "generate_hybrid_samples",
    "generate_hybrid_samples": "generate_hybrid_samples",
    "generate_samples": "generate_hybrid_samples",
    "save": "save_config",
    "save_config": "save_config",
}
GENERATION_STEP_CONFLICTS = (
    ("extract_anomalies", "load_anomalies"),
    ("train_generator", "load_generator"),
    ("generate_synthetic_anomalies", "load_synthetic_anomalies"),
    ("create_matching", "load_matching"),
)


@dataclass(frozen=True)
class MVTecAD2Sample:
    image_path: Path
    mask_path: Path | None
    sample_id: str
    split: str
    label: str


@dataclass(frozen=True)
class MVTecAD2UseCase:
    category: str
    category_root: Path
    config: Configuration
    anomaly_dataloader: "MVTecAD2Dataloader"
    control_dataloader: "MVTecAD2Dataloader"
    positive_only: bool = True


class MVTecAD2Dataloader:
    """
    Iterator adapter for MVTec AD 2 samples.

    The HybridSampleGeneration framework expects tuples of
    (image_array, segmentation_array, basename). Positive MVTec AD 2 samples
    provide real masks. Good/negative control samples do not have masks, so this
    adapter creates zero masks for them.
    """

    def __init__(self, samples: Sequence[MVTecAD2Sample]) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("MVTecAD2Dataloader requires at least one sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        for sample in self.samples:
            img = ensure_chw(load_image_array(str(sample.image_path))).astype(np.float32, copy=False)

            if sample.mask_path is None:
                seg = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
            else:
                seg = ensure_chw(load_image_array(str(sample.mask_path))).astype(np.float32, copy=False)
                if seg.shape[0] > 1:
                    seg = seg[:1]
                seg = np.where(seg > 0, 1.0, 0.0).astype(np.float32, copy=False)

            yield img, seg, sample.sample_id


def prepare_mvtecad2_usecases(
    root: Path | str = MVTECAD2_ROOT,
    categories: str | Iterable[str] | None = None,
    *,
    load_negative_controls: bool = True,
    include_public_good_controls: bool = False,
    save_path: Path | str | None = None,
    results_root: Path | str | None = None,
) -> list[MVTecAD2UseCase]:
    """
    Prepare one HybridSampleGeneration use case per MVTec AD 2 category.

    Default behavior uses test_public/bad samples with masks for anomaly
    extraction and train/validation good samples as controls for matching and
    fusion. Set load_negative_controls=False only for a positive-only debug run.
    Category-specific HybridDataGenerator settings are defined in
    MVTecAD2_configuration.py.

    If save_path is provided, every category gets its own Configuration
    save_path: <save_path>/<category>. HybridDataGenerator results are then
    written below <save_path>/<category>/results/<study_name>.
    """

    if save_path is not None and results_root is not None:
        raise ValueError("Use either save_path or results_root, not both.")

    root = Path(root)
    _validate_dataset_root(root)

    category_names = _normalize_categories(root, categories)
    if not category_names:
        raise ValueError(f"No MVTec AD 2 categories found in {root}")

    use_cases: list[MVTecAD2UseCase] = []
    for category in category_names:
        category_root = root / category
        if not category_root.is_dir():
            raise FileNotFoundError(f"Missing MVTec AD 2 category folder: {category_root}")

        positive_samples = _collect_public_anomaly_samples(category_root)
        if not positive_samples:
            raise ValueError(f"No public positive samples with masks found for category {category!r}.")

        if load_negative_controls:
            control_samples = _collect_control_samples(
                category_root,
                include_public_good_controls=include_public_good_controls,
            )
            positive_only = False
        else:
            control_samples = positive_samples
            positive_only = True

        if not control_samples:
            raise ValueError(f"No control samples found for category {category!r}.")

        config = _configuration_for_category(
            category,
            save_path=save_path,
            results_root=results_root,
        )

        use_cases.append(
            MVTecAD2UseCase(
                category=category,
                category_root=category_root,
                config=config,
                anomaly_dataloader=MVTecAD2Dataloader(positive_samples),
                control_dataloader=MVTecAD2Dataloader(control_samples),
                positive_only=positive_only,
            )
        )

    return use_cases


def run_hybrid_sample_generation_for_usecase(
    use_case: MVTecAD2UseCase,
    *,
    steps: str | Iterable[str] | None = None,
    no_of_trials: int = 1,
    train_generator: bool = True,
    load_existing_generator: bool = False,
    generate_synthetic_anomalies: bool = True,
    load_existing_synthetic_anomalies: bool = False,
    create_matching: bool = True,
    load_existing_matching: bool = False,
    generator_db_path: Path | str | None = None,
    generator_trial_id: int = -1,
) -> Configuration:
    """
    Execute HybridSampleGeneration for one prepared MVTec AD 2 use case.

    This function only creates hybrid samples. Evaluation and visualization are
    intentionally separate downstream calls.

    steps can be used to run only selected generation steps. Examples:
    ("extract", "train", "generate_synth"), ("load_synth", "matching", "fusion").
    """

    print(f"\n========== MVTec AD 2 use case: {use_case.category} ==========")

    config = use_case.config
    generator = HybridDataGenerator(config)
    selected_steps = _normalize_generation_steps(
        steps,
        train_generator=train_generator,
        load_existing_generator=load_existing_generator,
        generate_synthetic_anomalies=generate_synthetic_anomalies,
        load_existing_synthetic_anomalies=load_existing_synthetic_anomalies,
        create_matching=create_matching,
        load_existing_matching=load_existing_matching,
    )

    if "extract_anomalies" in selected_steps:
        generator.extract_anomalies(use_case.anomaly_dataloader)
    elif _needs_anomalies_loaded(selected_steps):
        generator.load_anomalies()

    if "load_anomalies" in selected_steps:
        generator.load_anomalies()

    if "train_generator" in selected_steps:
        generator.train_generator(no_of_trails=no_of_trials)
    elif "load_generator" in selected_steps or _needs_generator_loaded(selected_steps):
        generator.load_generator(
            path_to_db_file=None if generator_db_path is None else str(generator_db_path),
            trial_id=generator_trial_id,
        )

    if "generate_synthetic_anomalies" in selected_steps:
        generator.generate_synth_anomalies()
    elif "load_synthetic_anomalies" in selected_steps or _needs_synthetic_anomalies_loaded(selected_steps):
        generator.load_synth_anomalies()

    if "create_matching" in selected_steps:
        generator.create_matching_dict(use_case.control_dataloader)
    elif "load_matching" in selected_steps or _needs_matching_loaded(selected_steps):
        generator.load_matching_dict()

    if "generate_hybrid_samples" in selected_steps:
        _generate_and_save_hybrid_samples(generator, use_case)

    if "save_config" in selected_steps:
        config.save_config_file()

    return config


def run_hybrid_sample_generation_for_all_usecases(
    root: Path | str = MVTECAD2_ROOT,
    categories: str | Iterable[str] | None = None,
    **kwargs,
) -> list[Configuration]:
    """
    Prepare and execute HybridSampleGeneration for all selected MVTec AD 2 categories.

    Pass save_path=<folder> to create one output root per use case:
    <folder>/<category>/results/<study_name>/...
    """

    _reject_deprecated_generation_flags(kwargs)
    prepare_kwargs = _pop_prepare_kwargs(kwargs)

    use_cases = prepare_mvtecad2_usecases(root, categories, **prepare_kwargs)
    configs: list[Configuration] = []
    for use_case in use_cases:
        configs.append(run_hybrid_sample_generation_for_usecase(use_case, **kwargs))
    return configs


def run_evaluation_for_usecase(
    use_case: MVTecAD2UseCase,
    *,
    load_saved_config: bool = True,
) -> Configuration:
    """
    Run the evaluation pipeline for one already generated MVTec AD 2 use case.

    By default the saved configuration from the use case study folder is loaded,
    so downstream evaluation uses the exact config that produced the data.
    """

    print(f"\n========== MVTec AD 2 evaluation: {use_case.category} ==========")
    config = _downstream_config_for_usecase(use_case, load_saved_config=load_saved_config)
    generator = HybridDataGenerator(config)
    generator.run_evaluation_pipeline(use_case.anomaly_dataloader)
    return config


def run_evaluation_for_all_usecases(
    root: Path | str = MVTECAD2_ROOT,
    categories: str | Iterable[str] | None = None,
    **kwargs,
) -> list[Configuration]:
    """
    Run evaluation for all selected MVTec AD 2 categories after generation.

    Pass the same save_path/results_root that was used for generation.
    """

    load_saved_config = bool(kwargs.pop("load_saved_config", True))
    prepare_kwargs = _pop_prepare_kwargs(kwargs)
    _reject_unknown_kwargs(kwargs)

    use_cases = prepare_mvtecad2_usecases(root, categories, **prepare_kwargs)
    configs: list[Configuration] = []
    for use_case in use_cases:
        configs.append(
            run_evaluation_for_usecase(
                use_case,
                load_saved_config=load_saved_config,
            )
        )
    return configs


def visualize_evaluation_for_usecase(
    use_case: MVTecAD2UseCase,
    *,
    load_saved_config: bool = True,
) -> Configuration:
    """
    Open the evaluation/outlier visualization for one generated use case.
    """

    print(f"\n========== MVTec AD 2 visualization: {use_case.category} ==========")
    config = _downstream_config_for_usecase(use_case, load_saved_config=load_saved_config)
    generator = HybridDataGenerator(config)
    generator.visualize_evaluation_results()
    return config


def visualize_evaluation_for_all_usecases(
    root: Path | str = MVTECAD2_ROOT,
    categories: str | Iterable[str] | None = None,
    **kwargs,
) -> list[Configuration]:
    """
    Open the evaluation/outlier visualization for all selected categories.

    The viewer is interactive and blocks per category until the GUI is closed.
    Pass the same save_path/results_root that was used for generation.
    """

    load_saved_config = bool(kwargs.pop("load_saved_config", True))
    prepare_kwargs = _pop_prepare_kwargs(kwargs)
    _reject_unknown_kwargs(kwargs)

    use_cases = prepare_mvtecad2_usecases(root, categories, **prepare_kwargs)
    configs: list[Configuration] = []
    for use_case in use_cases:
        configs.append(
            visualize_evaluation_for_usecase(
                use_case,
                load_saved_config=load_saved_config,
            )
        )
    return configs


def run_evaluation_and_visualization_for_usecase(
    use_case: MVTecAD2UseCase,
    *,
    load_saved_config: bool = True,
) -> Configuration:
    """
    Run evaluation for one generated use case and open the viewer afterwards.
    """

    print(f"\n========== MVTec AD 2 evaluation + visualization: {use_case.category} ==========")
    config = _downstream_config_for_usecase(use_case, load_saved_config=load_saved_config)
    generator = HybridDataGenerator(config)
    generator.run_evaluation_pipeline(use_case.anomaly_dataloader)
    generator.visualize_evaluation_results()
    return config


def run_evaluation_and_visualization_for_all_usecases(
    root: Path | str = MVTECAD2_ROOT,
    categories: str | Iterable[str] | None = None,
    **kwargs,
) -> list[Configuration]:
    """
    Run evaluation and then visualization for all selected MVTec AD 2 categories.

    categories can be a single category string, e.g. "can", or an iterable such
    as ("can", "fabric"). The viewer is interactive and blocks until closed.
    """

    load_saved_config = bool(kwargs.pop("load_saved_config", True))
    prepare_kwargs = _pop_prepare_kwargs(kwargs)
    _reject_unknown_kwargs(kwargs)

    use_cases = prepare_mvtecad2_usecases(root, categories, **prepare_kwargs)
    configs: list[Configuration] = []
    for use_case in use_cases:
        configs.append(
            run_evaluation_and_visualization_for_usecase(
                use_case,
                load_saved_config=load_saved_config,
            )
        )
    return configs


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
        normalized_categories.append(_canonical_category(category))

    return normalized_categories


def _configuration_for_category(
    category: str,
    *,
    save_path: Path | str | None,
    results_root: Path | str | None,
) -> Configuration:
    base_save_path = save_path if save_path is not None else results_root
    category_save_path = _category_save_path(base_save_path, category)

    return create_mvtecad2_configuration(
        category,
        save_path=category_save_path,
    )


def _category_save_path(base_save_path: Path | str | None, category: str) -> Path | None:
    if base_save_path is None:
        return None
    path = Path(base_save_path) / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def _downstream_config_for_usecase(
    use_case: MVTecAD2UseCase,
    *,
    load_saved_config: bool,
) -> Configuration:
    if load_saved_config:
        config_path = Path(use_case.config.study_folder) / "configuration.json"
        if config_path.is_file():
            return load_config_file(str(config_path))
        print(f"Warning: No saved configuration found at {config_path}. Using prepared config.")
    return use_case.config


def _pop_prepare_kwargs(kwargs: dict) -> dict:
    return {key: kwargs.pop(key) for key in list(kwargs) if key in PREPARE_USECASE_KEYS}


def _reject_deprecated_generation_flags(kwargs: dict) -> None:
    deprecated_values = {key: kwargs.pop(key) for key in DEPRECATED_GENERATION_FLAGS if key in kwargs}
    enabled = [key for key, value in deprecated_values.items() if value]
    if enabled:
        raise ValueError(
            "Evaluation and visualization are now downstream steps. "
            "Run run_evaluation_for_all_usecases(...) and "
            "visualize_evaluation_for_all_usecases(...) after generation instead, "
            "or use run_evaluation_and_visualization_for_all_usecases(...)."
        )


def _reject_unknown_kwargs(kwargs: Mapping[str, object]) -> None:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")


def _normalize_generation_steps(
    steps: str | Iterable[str] | None,
    *,
    train_generator: bool,
    load_existing_generator: bool,
    generate_synthetic_anomalies: bool,
    load_existing_synthetic_anomalies: bool,
    create_matching: bool,
    load_existing_matching: bool,
) -> tuple[str, ...]:
    if steps is None:
        selected_steps = _legacy_generation_steps(
            train_generator=train_generator,
            load_existing_generator=load_existing_generator,
            generate_synthetic_anomalies=generate_synthetic_anomalies,
            load_existing_synthetic_anomalies=load_existing_synthetic_anomalies,
            create_matching=create_matching,
            load_existing_matching=load_existing_matching,
        )
    else:
        selected_steps = _canonical_generation_steps(steps)

    _validate_generation_step_conflicts(selected_steps)
    return selected_steps


def _legacy_generation_steps(
    *,
    train_generator: bool,
    load_existing_generator: bool,
    generate_synthetic_anomalies: bool,
    load_existing_synthetic_anomalies: bool,
    create_matching: bool,
    load_existing_matching: bool,
) -> tuple[str, ...]:
    steps = ["extract_anomalies"]

    if generate_synthetic_anomalies:
        if train_generator:
            steps.append("train_generator")
        elif load_existing_generator:
            steps.append("load_generator")
        else:
            raise ValueError("Either train_generator or load_existing_generator must be True.")
        steps.append("generate_synthetic_anomalies")
    elif load_existing_synthetic_anomalies:
        steps.append("load_synthetic_anomalies")
    else:
        raise ValueError(
            "Either generate_synthetic_anomalies or load_existing_synthetic_anomalies must be True."
        )

    if create_matching:
        steps.append("create_matching")
    elif load_existing_matching:
        steps.append("load_matching")
    else:
        raise ValueError("Either create_matching or load_existing_matching must be True.")

    steps.extend(["generate_hybrid_samples", "save_config"])
    return tuple(steps)


def _canonical_generation_steps(steps: str | Iterable[str]) -> tuple[str, ...]:
    raw_steps = [steps] if isinstance(steps, str) else list(steps)
    if not raw_steps:
        raise ValueError("steps must contain at least one pipeline step.")

    canonical_steps = []
    for step in raw_steps:
        if not isinstance(step, str):
            raise TypeError(f"Pipeline step names must be strings, got {type(step).__name__}.")
        step_key = step.strip().lower().replace("-", "_").replace(" ", "_")
        canonical_step = GENERATION_STEP_ALIASES.get(step_key)
        if canonical_step is None:
            allowed_steps = ", ".join(sorted(GENERATION_STEP_ALIASES))
            raise ValueError(f"Unknown generation step {step!r}. Allowed steps: {allowed_steps}")
        if canonical_step == "all":
            return DEFAULT_GENERATION_STEPS
        canonical_steps.append(canonical_step)

    canonical_set = set(canonical_steps)
    return tuple(step for step in GENERATION_STEP_ORDER if step in canonical_set)


def _validate_generation_step_conflicts(steps: Sequence[str]) -> None:
    step_set = set(steps)
    for left, right in GENERATION_STEP_CONFLICTS:
        if left in step_set and right in step_set:
            raise ValueError(f"Pipeline steps {left!r} and {right!r} cannot be used together.")


def _needs_anomalies_loaded(steps: Sequence[str]) -> bool:
    step_set = set(steps)
    return (
        bool({"train_generator", "generate_synthetic_anomalies"} & step_set)
        and "extract_anomalies" not in step_set
        and "load_anomalies" not in step_set
    )


def _needs_generator_loaded(steps: Sequence[str]) -> bool:
    step_set = set(steps)
    return (
        "generate_synthetic_anomalies" in step_set
        and "train_generator" not in step_set
        and "load_generator" not in step_set
    )


def _needs_synthetic_anomalies_loaded(steps: Sequence[str]) -> bool:
    step_set = set(steps)
    return (
        bool({"create_matching", "generate_hybrid_samples"} & step_set)
        and "generate_synthetic_anomalies" not in step_set
        and "load_synthetic_anomalies" not in step_set
    )


def _needs_matching_loaded(steps: Sequence[str]) -> bool:
    step_set = set(steps)
    return (
        "generate_hybrid_samples" in step_set
        and "create_matching" not in step_set
        and "load_matching" not in step_set
    )


def _generate_and_save_hybrid_samples(
    generator: HybridDataGenerator,
    use_case: MVTecAD2UseCase,
) -> None:
    config = use_case.config
    study_folder = Path(config.study_folder)
    save_folder = study_folder / "generated_hybrid_samples"
    synth_roi_folder = study_folder / "synth_roi_data"

    _reset_fusion_output_folder(save_folder, study_folder)
    _reset_fusion_output_folder(synth_roi_folder, study_folder)

    img_folder = save_folder / "images"
    seg_folder = save_folder / "segmentations"
    img_folder.mkdir(parents=True, exist_ok=True)
    seg_folder.mkdir(parents=True, exist_ok=True)

    for control_image, control_seg, basename in use_case.control_dataloader:
        if basename not in config.matching_dict:
            print(f"{basename} not found in matching dict")
            continue

        base_mask = _mask_like_image(control_seg, control_image) if use_case.positive_only else None
        img, seg = generator.fusion_synth_anomalies(
            control_image,
            basename,
            base_mask=base_mask,
            save_npy=True,
        )

        save_image(img, img_folder / basename)
        save_image(_segmentation_for_png(seg), seg_folder / basename)


def _reset_fusion_output_folder(folder: Path, study_folder: Path) -> None:
    folder = folder.resolve()
    study_folder = study_folder.resolve()
    _ensure_child_path(folder, study_folder)

    if folder.exists():
        shutil.rmtree(folder)


def _ensure_child_path(path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete fusion output outside study folder: {path}") from exc


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


def _segmentation_for_png(seg: np.ndarray) -> np.ndarray:
    mask = np.asarray(seg)
    if mask.ndim != 3:
        raise ValueError(f"Expected segmentation with shape (C,H,W), got {mask.shape}")
    return np.where(mask[:1] > 0, 255, 0).astype(np.uint8)


def _mask_like_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    image = np.asarray(image)
    if mask.shape == image.shape:
        return mask
    if mask.ndim == 3 and image.ndim == 3 and mask.shape[0] == 1 and mask.shape[1:] == image.shape[1:]:
        return np.repeat(mask, image.shape[0], axis=0)
    raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")


def _make_sample_id(split: str, label: str, image_path: Path) -> str:
    filename = _safe_name(image_path.stem) + image_path.suffix.lower()
    return f"{_safe_name(split)}_{_safe_name(label)}_{filename}"


def _safe_name(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "sample"


def _canonical_category(category: str) -> str:
    category = category.strip().lower()
    return CATEGORY_ALIASES.get(category, category)


def _validate_dataset_root(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"MVTec AD 2 root does not exist: {root}. "
            "Set MVTECAD2_ROOT in pipeline_MVTecAD2.py or via the MVTECAD2_ROOT environment variable."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"MVTec AD 2 root is not a directory: {root}")


if __name__ == "__main__":

    save_path = r"path/to/save/generated/data"
    dataset_root = r"path/to/mvtec_ad_2_dataset"

    run_hybrid_sample_generation_for_all_usecases(
        dataset_root,
        categories="can",
        no_of_trials=1,
        steps=("extract", "train", "generate_synth", "matching", "fusion", "save"),
        #steps=("matching", "fusion", "save"),
        save_path=save_path
    )

    # Downstream call after generation:
    run_evaluation_and_visualization_for_all_usecases(
         dataset_root,
         categories="can",
         save_path=save_path,
    )
