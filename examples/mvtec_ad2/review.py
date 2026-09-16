"""Review persisted generator results independently of training workflows."""

from collections.abc import Iterable
from pathlib import Path

from examples.mvtec_ad2.studies import open_study, normalize_study_folders


def review_studies(study_folders: Iterable[Path | str], *, actions=("visualize",)) -> None:
    """Generator-quality evaluation/viewing; no split or source dataset required."""
    actions = tuple(actions)
    if not actions or any(action not in ("evaluate_generator", "visualize") for action in actions):
        raise ValueError("Review actions are evaluate_generator and/or visualize.")
    for folder in normalize_study_folders(study_folders):
        case = open_study(folder)
        if not Path(case.config.study.paths.artifact_database).is_file():
            raise FileNotFoundError(f"No study repository: {case.config.study.paths.artifact_database}")
        if "evaluate_generator" in actions:
            from hybrid_sample_generator.evaluation.service import evaluate_study
            evaluate_study(case.config)
        if "visualize" in actions:
            from hybrid_sample_generator.visualization import run_hybrid_visualizer
            run_hybrid_visualizer(case.config)
