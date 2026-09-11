"""Orchestration for the MVTec AD 2 example integration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration, load_config_file
from hybrid_sample_generator.evaluation.service import evaluate_study
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from hybrid_sample_generator.visualization import run_hybrid_visualizer
from examples.image_2d.image_dataloader import save_image
from examples.mvtec_ad2.configuration import create_mvtecad2_configuration
from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader
from examples.mvtec_ad2.discovery import (
    _collect_control_samples,
    _collect_public_anomaly_samples,
    _normalize_categories,
    _validate_dataset_root,
)
from examples.mvtec_ad2.records import MVTecAD2UseCase
from examples.mvtec_ad2.steps import (
    _needs_generator_loaded,
    _normalize_generation_steps,
    _pop_prepare_kwargs,
    _reject_deprecated_generation_flags,
    _reject_unknown_kwargs,
)

MVTECAD2_ROOT = Path(os.environ.get("MVTECAD2_ROOT", r"/mnt/results/mvtec2/mvtec_ad_2"))
MVTECAD2_SAVE = Path(os.environ.get("MVTECAD2_SAVE", r"/mnt/results/mvtec2/experiments/test_datarepo_v4"))

def prepare_mvtecad2_usecases(
    root: Path | str = MVTECAD2_ROOT,
    categories: str | Iterable[str] | None = None,
    *,
    include_public_good_controls: bool = False,
    save_path: Path | str | None = None,
    results_root: Path | str | None = None,
) -> list[MVTecAD2UseCase]:
    """
    Prepare one HybridSampleGeneration use case per MVTec AD 2 category.

    Default behavior uses test_public/bad samples with masks for anomaly
    extraction and train/validation good samples as controls for matching and
    fusion. Both groups are exposed through one mixed sample dataloader.
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

        control_samples = _collect_control_samples(
            category_root,
            include_public_good_controls=include_public_good_controls,
        )

        if not control_samples:
            raise ValueError(f"No control samples found for category {category!r}.")

        config = _configuration_for_category(
            category,
            save_path=save_path,
            results_root=results_root,
        )

        samples_by_path = {
            sample.image_path.resolve(): sample
            for sample in [*positive_samples, *control_samples]
        }
        use_cases.append(
            MVTecAD2UseCase(
                category=category,
                category_root=category_root,
                config=config,
                sample_dataloader=MVTecAD2Dataloader(list(samples_by_path.values())),
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
    plan_hybrids: bool = True,
    generator_db_path: Path | str | None = None,
    generator_trial_id: int = -1,
) -> Configuration:
    """
    Execute HybridSampleGeneration for one prepared MVTec AD 2 use case.

    This function only creates hybrid samples. Evaluation and visualization are
    intentionally separate downstream calls.

    steps can be used to run only selected generation steps. Examples:
    ("ingest", "extract", "train", "generate_synth"),
    ("plan", "materialize") for already persisted synthetic anomalies.
    generator_trial_id selects the model to load: -1 best model, -2 newest model,
    otherwise the concrete Optuna trial/model number.
    """

    print(f"\n========== MVTec AD 2 use case: {use_case.category} ==========")

    config = use_case.config
    generator = HybridDataGenerator(config)
    selected_steps = _normalize_generation_steps(
        steps,
        train_generator=train_generator,
        load_existing_generator=load_existing_generator,
        generate_synthetic_anomalies=generate_synthetic_anomalies,
        plan_hybrids=plan_hybrids,
    )

    if "ingest_dataset" in selected_steps:
        generator.ingest_dataset(use_case.sample_dataloader)

    if "extract_anomalies" in selected_steps:
        generator.extract_anomalies()

    if "train_generator" in selected_steps:
        generator.train_generator(no_of_trials=no_of_trials)
    elif "load_generator" in selected_steps or _needs_generator_loaded(selected_steps):
        generator.load_generator(
            path_to_db_file=None if generator_db_path is None else str(generator_db_path),
            trial_id=generator_trial_id,
        )

    if "generate_synthetic_anomalies" in selected_steps:
        generator.generate_synthetic_anomalies()

    if "plan_hybrid_samples" in selected_steps:
        generator.plan_hybrid_samples()

    if "materialize_hybrid_samples" in selected_steps:
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
    evaluate_study(config)
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
    run_hybrid_visualizer(config)
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
    evaluate_study(config)
    run_hybrid_visualizer(config)
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
        config_path = Path(use_case.config.study.folder) / "configuration.json"
        if config_path.is_file():
            return load_config_file(str(config_path))
        print(f"Warning: No saved configuration found at {config_path}. Using prepared config.")
    return use_case.config

def _generate_and_save_hybrid_samples(
    generator: HybridDataGenerator,
    use_case: MVTecAD2UseCase,
) -> None:
    config = use_case.config
    img_folder = Path(config.study.paths.generated_images)
    seg_folder = Path(config.study.paths.generated_segmentations)
    img_folder.mkdir(parents=True, exist_ok=True)
    seg_folder.mkdir(parents=True, exist_ok=True)
    for hybrid in generator.materialize_hybrid_samples():
        original = generator.repository.get_original_sample(hybrid.original_sample_id)
        image = generator.artifact_store.load_array(hybrid.image_path)
        segmentation = generator.artifact_store.load_array(hybrid.segmentation_path)
        source = Path(original.source_name)
        suffix = source.suffix or ".png"
        export_name = f"{source.stem}__hybrid_{hybrid.variant_index}{suffix}"
        save_image(image, img_folder / export_name)
        save_image(_segmentation_for_png(segmentation), seg_folder / export_name)

def _segmentation_for_png(seg: np.ndarray) -> np.ndarray:
    mask = np.asarray(seg)
    if mask.ndim != 3:
        raise ValueError(f"Expected segmentation with shape (C,H,W), got {mask.shape}")
    return np.where(mask[:1] > 0, 255, 0).astype(np.uint8)
