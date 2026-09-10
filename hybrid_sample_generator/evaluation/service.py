"""Orchestrate evaluation of persisted real/synthetic anomaly relations."""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from hybrid_sample_generator.configuration.evaluation import EvaluationConfiguration
from hybrid_sample_generator.evaluation.metrics import (
    compute_glcm,
    get_glcm_feature_diffs,
    get_glcm_roi_feature_diffs,
    get_volume_feature_diffs,
    glcm_features,
    relative_foreground_mask,
)
from hybrid_sample_generator.evaluation.outliers import find_outliers
from hybrid_sample_generator.evaluation.pairs import EvaluationPair, cutout_pairs, roi_pairs
from hybrid_sample_generator.evaluation.reporting import (
    analyze_results,
    print_overlap_summary,
    save_difference_histograms,
)
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_paths import StudyPaths
from hybrid_sample_generator.persistence.study_repository import StudyRepository


def evaluate_study(config):
    """Evaluate one persisted study without constructing the pipeline facade."""
    paths = config.study.paths
    return evaluation_pipeline(
        StudyRepository(paths.artifact_database),
        ArtifactStore(paths.study_folder),
        paths,
        config.evaluation,
    )


def run_feature_calculator(
    pairs: list[EvaluationPair],
    feature_calculator_func,
    artifact_store: ArtifactStore,
    paths: StudyPaths,
    config: EvaluationConfiguration,
    *,
    use_recorded_masks: bool,
):
    real_totals = defaultdict(float)
    synthetic_totals = defaultdict(float)
    differences = defaultdict(list)
    differences_with_ids = defaultdict(list)
    csv_rows = []
    for pair in pairs:
        real_array = artifact_store.load_array(pair.real_image_path)
        synthetic_array = artifact_store.load_array(pair.synthetic_image_path)
        if use_recorded_masks and pair.real_mask_path and pair.synthetic_mask_path:
            real_mask = artifact_store.load_array(pair.real_mask_path) > 0
            synthetic_mask = artifact_store.load_array(pair.synthetic_mask_path) > 0
        elif use_recorded_masks and config.foreground_threshold is not None:
            real_mask = relative_foreground_mask(
                real_array, config.foreground_threshold
            )
            synthetic_mask = relative_foreground_mask(
                synthetic_array, config.foreground_threshold
            )
        else:
            real_mask = np.ones_like(real_array, dtype=bool)
            synthetic_mask = np.ones_like(synthetic_array, dtype=bool)
        real_features, synthetic_features, pair_differences = feature_calculator_func(
            real_array, real_mask, synthetic_array, synthetic_mask
        )
        csv_rows.append(
            {
                "pair_id": pair.id,
                "real_anomaly_id": pair.real_anomaly_id,
                "synthetic_anomaly_id": pair.synthetic_anomaly_id,
                "placement_id": pair.placement_id,
                "feature_calculator": feature_calculator_func.__name__,
                "metric_diffs": json.dumps(
                    {name: _scalar(value) for name, value in pair_differences.items()}
                ),
            }
        )
        for name, difference in pair_differences.items():
            real_totals[name] += real_features[name]
            synthetic_totals[name] += synthetic_features[name]
            differences[name].append(difference)
            differences_with_ids[name].append({"value": difference, "sample": pair.id})

    count = len(pairs)
    outliers = {
        name: find_outliers(values, differences_with_ids[name], config, name)
        for name, values in differences.items()
    }
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(
            paths.metric_diffs_csv,
            mode="a",
            index=False,
            header=not os.path.exists(paths.metric_diffs_csv),
        )
    return {
        "sample_counter": count,
        "mean_real": ({name: value / count for name, value in real_totals.items()} if count else {}),
        "mean_synth": ({name: value / count for name, value in synthetic_totals.items()} if count else {}),
        "outliers": outliers,
        "all_diffs": dict(differences),
    }


def evaluation_pipeline(repository, artifact_store, paths, config):
    """Evaluate cutout and placement pairs joined through normalized relations."""
    config.validate()
    os.makedirs(paths.evaluation_results, exist_ok=True)
    if os.path.exists(paths.metric_diffs_csv):
        os.remove(paths.metric_diffs_csv)
    cutouts = cutout_pairs(repository)
    rois = roi_pairs(repository)
    results = {
        "glcm_cutout": run_feature_calculator(
            cutouts, get_glcm_feature_diffs, artifact_store, paths, config,
            use_recorded_masks=True,
        ),
        "volume_cutout": run_feature_calculator(
            cutouts, get_volume_feature_diffs, artifact_store, paths, config,
            use_recorded_masks=True,
        ),
        "glcm_roi": run_feature_calculator(
            rois, get_glcm_roi_feature_diffs, artifact_store, paths, config,
            use_recorded_masks=False,
        ),
    }
    histogram_paths = {
        "glcm_cutout": paths.glcm_cutout_difference_histograms,
        "volume_cutout": paths.volume_cutout_difference_histograms,
        "glcm_roi": paths.glcm_roi_difference_histograms,
    }
    for name, result in results.items():
        analyze_results(result)
        if result["all_diffs"]:
            save_difference_histograms(result["all_diffs"], histogram_paths[name])
    print_overlap_summary([result["outliers"] for result in results.values()])
    return results


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


__all__ = [
    "EvaluationPair", "compute_glcm", "evaluate_study", "evaluation_pipeline",
    "get_glcm_feature_diffs", "get_glcm_roi_feature_diffs",
    "get_volume_feature_diffs", "glcm_features", "run_feature_calculator",
]
