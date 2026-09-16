"""Step-specific prerequisite checks before expensive work."""

from pathlib import Path

from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from examples.mvtec_ad2.splits import manifest_samples, verify_repository
from examples.mvtec_ad2.steps import GENERATOR_STEPS


def preflight(study, steps, *, downstream_run_id=None):
    """Return reusable hybrid pairs only when no earlier step will mutate them."""
    selected = set(steps)
    config, manifest = study.config, study.split_manifest
    if downstream_run_id is not None and ("evaluate_downstream" not in selected or "train_downstream" in selected):
        raise ValueError("downstream_run_id is only valid for evaluation without training.")
    needs_split = bool(selected & (set(GENERATOR_STEPS) | {"train_downstream", "evaluate_downstream"}))
    if needs_split and manifest is None:
        raise ValueError("This workflow requires a saved split manifest; legacy studies remain viewable/exportable.")
    generator_steps = selected & set(GENERATOR_STEPS)
    if generator_steps:
        config.validate()
    needs_hybrids = "train_downstream" in selected and config.downstream.data.hybrid_fraction > 0
    needs_repo = bool(generator_steps or "export" in selected or needs_hybrids)
    database = Path(config.study.paths.artifact_database)
    repo = StudyRepository(database) if needs_repo and database.is_file() else None
    # HybridPairs verifies provenance itself. Avoid verifying twice on a pure
    # downstream run; generation needs verification before it changes records.
    if generator_steps and repo is not None:
        verify_repository(repo, manifest)
    for consumers, producer, invalidators, method, message in (
        ({"extract", "plan"}, "ingest", set(), "list_original_samples", "Ingest originals first."),
        ({"train_generator", "generate_synthetic"}, "extract", {"ingest"}, "list_real_anomalies", "Extract anomalies first."),
        ({"plan"}, "generate_synthetic", {"ingest", "extract"}, "list_synthetic_anomalies", "Generate synthetic anomalies first."),
        ({"materialize"}, "plan", {"ingest", "extract", "generate_synthetic"}, "list_hybrid_samples", "Plan hybrids first."),
    ):
        if selected & consumers and producer not in selected:
            if selected & invalidators or repo is None or not getattr(repo, method)():
                raise ValueError(message)
    if selected & {"load_generator", "generate_synthetic"} and "train_generator" not in selected:
        if not Path(config.study.paths.optuna_db_file).is_file():
            raise ValueError("No saved generator. Select train_generator first.")
    if "ingest" in selected and study.sample_dataloader is None:
        if not manifest_samples(manifest, "train"):
            raise ValueError("No training originals in the split manifest.")
    needs_materialized = "export" in selected or needs_hybrids
    if needs_materialized and "materialize" not in selected:
        if selected & {"ingest", "extract", "generate_synthetic", "plan"}:
            raise ValueError("Earlier steps invalidate existing hybrids; select materialize before export/downstream training.")
    if "export" in selected and "materialize" not in selected:
        if repo is None or not repo.list_hybrid_samples(status="generated"):
            raise ValueError("No materialized hybrids to export.")
    hybrids = None
    if "train_downstream" in selected:
        from examples.mvtec_ad2.downstream.sampling import source_plan
        from examples.mvtec_ad2.downstream.draem.synthesis import TextureSynthesizer
        from examples.mvtec_ad2.downstream.datasets import HybridPairs
        config.downstream.validate()
        data = config.downstream.data
        source_plan(data, config.downstream.seed, 0)
        if not manifest_samples(manifest, "train", healthy_only=True) or not manifest_samples(manifest, "validation"):
            raise ValueError("Downstream training requires healthy training images and validation samples.")
        if data.hybrid_fraction < 1 and data.texture_root is not None:
            TextureSynthesizer(data.texture_root)
        if needs_hybrids:
            if config.matching.routine == "fixed_from_extraction_anomaly_fusion" and "plan" in selected:
                raise ValueError("DRAEM requires healthy hybrid backgrounds.")
            if "materialize" not in selected:
                if repo is None:
                    raise ValueError("No materialized hybrids. Select materialize first.")
                # Planning may replace records; do not cache pairs across it.
                if generator_steps:
                    if not repo.list_hybrid_samples(status="generated"):
                        raise ValueError("No materialized hybrids. Select materialize first.")
                else:
                    hybrids = HybridPairs(repo, ArtifactStore(config.study.folder), manifest, data)
    if "evaluate_downstream" in selected and "train_downstream" not in selected:
        from examples.mvtec_ad2.downstream.runner import load_run
        load_run(study, downstream_run_id)
    return hybrids
