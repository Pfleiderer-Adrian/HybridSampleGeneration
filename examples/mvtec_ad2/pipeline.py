"""Execute explicit steps for new experiments or saved studies."""

from collections.abc import Iterable
from pathlib import Path

from examples.mvtec_ad2.configuration import Experiment
from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader
from examples.mvtec_ad2.records import MVTecAD2Study, WorkflowResult
from examples.mvtec_ad2.splits import manifest_samples, verify_repository
from examples.mvtec_ad2.steps import (
    FULL_EXPERIMENT, GENERATE_HYBRIDS, GENERATOR_STEPS, TRAIN_DOWNSTREAM, normalize_steps,
)
from examples.mvtec_ad2.studies import open_study, prepare_studies, normalize_study_folders
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator

def run_study(
    study: MVTecAD2Study,
    *,
    steps: str | Iterable[str] | None = None,
    downstream_run_id: str | None = None,
) -> WorkflowResult:
    """Execute selected steps using the explicitly supplied study configuration."""
    from examples.mvtec_ad2.workflow import preflight

    selected = normalize_steps(steps)
    hybrids = preflight(study, selected, downstream_run_id=downstream_run_id)
    config = study.config
    generator_steps = set(selected) & set(GENERATOR_STEPS)
    # Training-only saves after texture resolution; combined generation saves
    # once here. The downstream run always gets its own resolved snapshot.
    if generator_steps:
        config.save_config_file()
    generator = HybridDataGenerator(config) if generator_steps else None
    if "ingest" in selected:
        loader = study.sample_dataloader
        if loader is None:
            loader = MVTecAD2Dataloader(manifest_samples(study.split_manifest, "train"))
        generator.ingest_dataset(loader)
        verify_repository(generator.repository, study.split_manifest)
    if "extract" in selected:
        generator.extract_anomalies()
    if "train_generator" in selected:
        generator.train_generator()
    elif "load_generator" in selected or "generate_synthetic" in selected:
        generator.load_generator()
    if "generate_synthetic" in selected:
        generator.generate_synthetic_anomalies()
    if "plan" in selected:
        generator.plan_hybrid_samples()
    if "materialize" in selected:
        generator.materialize_hybrid_samples()
    if "export" in selected:
        from examples.mvtec_ad2.export import export_hybrids
        export_hybrids(config)
    output = None
    if "train_downstream" in selected:
        from examples.mvtec_ad2.downstream.runner import train_downstream
        output = train_downstream(study, hybrids=hybrids, save_study_config=not generator_steps)
        downstream_run_id = output.name
    if "evaluate_downstream" in selected:
        from examples.mvtec_ad2.downstream.runner import evaluate_downstream
        output = evaluate_downstream(study, downstream_run_id)
    return WorkflowResult(Path(config.study.folder), output)


def run_new_experiment(
    experiment: Experiment, *, steps: str | Iterable[str] = FULL_EXPERIMENT,
) -> list[WorkflowResult]:
    """Create category studies using current presets and a fixed dataset split."""
    selected = normalize_steps(steps)
    if not (set(selected) & set(GENERATOR_STEPS) or "train_downstream" in selected):
        raise ValueError("Export/evaluation requires existing studies.")
    if not experiment.categories:
        raise ValueError("An experiment must select at least one category.")
    experiment.split.validate()
    studies = prepare_studies(
        experiment.dataset_root, experiment.categories,
        save_path=experiment.output_root, splits=experiment.split,
    )
    return [run_study(study, steps=selected) for study in studies]


def run_existing_studies(
    study_folders: Iterable[Path | str], *, steps: str | Iterable[str],
    downstream_run_id: str | None = None,
) -> list[WorkflowResult]:
    """Execute on saved configurations; current presets are never applied implicitly."""
    selected = normalize_steps(steps)
    studies = [open_study(folder) for folder in normalize_study_folders(study_folders)]
    return [run_study(study, steps=selected, downstream_run_id=downstream_run_id)
            for study in studies]
