from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyPaths import StudyPaths
from synthesizer.StudyRepository import StudyRepository
from synthesizer.configuration.evaluation import EvaluationConfiguration


@dataclass(frozen=True)
class EvaluationPair:
    id: str
    real_anomaly_id: str
    synthetic_anomaly_id: str
    real_image_path: str
    synthetic_image_path: str
    real_mask_path: str | None = None
    synthetic_mask_path: str | None = None
    placement_id: str | None = None


def evaluate_study(config):
    """Evaluate one persisted study without constructing a workflow orchestrator."""
    paths = config.study.paths
    repository = StudyRepository(paths.artifact_database)
    artifact_store = ArtifactStore(paths.study_folder)
    return evaluation_pipeline(
        repository,
        artifact_store,
        paths,
        config.evaluation,
    )


def _relative_foreground_mask(array, background_threshold):
    threshold = float(background_threshold)
    if threshold < 0:
        raise ValueError("background_threshold must be non-negative.")
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    return array > minimum + threshold * (maximum - minimum)


def compute_glcm(volume, mask, levels=32):
    volume = np.asarray(volume)
    mask = np.asarray(mask)
    if volume.ndim not in (3, 4):
        raise ValueError(f"Expected channel-first 2D/3D data, got {volume.shape}.")
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for channel_index in range(volume.shape[0]):
        if mask.ndim == volume.ndim:
            mask_index = min(channel_index, mask.shape[0] - 1)
            channel_mask = mask[mask_index]
        else:
            channel_mask = mask
        glcm += _compute_glcm_for_channel(
            volume[channel_index], channel_mask > 0, levels
        )
    total = glcm.sum()
    return (glcm / total).astype(np.float32) if total > 0 else glcm.astype(np.float32)


def _compute_glcm_for_channel(volume, mask, levels=32):
    lesion = volume[mask]
    glcm = np.zeros((levels, levels), dtype=np.float64)
    if lesion.size == 0:
        return glcm
    minimum = float(np.nanmin(lesion))
    value_range = float(np.nanmax(lesion)) - minimum
    if not np.isfinite(value_range) or value_range <= 0:
        quantized = np.zeros_like(volume, dtype=np.int64)
    else:
        quantized = np.clip(
            ((volume - minimum) / value_range * (levels - 1)).astype(np.int64),
            0,
            levels - 1,
        )
    displacements = (
        ((0, 1), (1, 1), (1, 0), (1, -1))
        if volume.ndim == 2
        else (
            (1, 0, 1), (0, 0, 1), (-1, 0, 1), (1, 1, 1), (0, 1, 1),
            (-1, 1, 1), (1, 1, 0), (0, 1, 0), (-1, 1, 0), (1, 1, -1),
            (0, 1, -1), (-1, 1, -1), (1, 0, 0),
        )
    )
    for displacement in displacements:
        source_slices = []
        target_slices = []
        for delta in displacement:
            if delta == 0:
                source_slices.append(slice(None))
                target_slices.append(slice(None))
            elif delta > 0:
                source_slices.append(slice(0, -delta))
                target_slices.append(slice(delta, None))
            else:
                source_slices.append(slice(-delta, None))
                target_slices.append(slice(0, delta))
        source_slices = tuple(source_slices)
        target_slices = tuple(target_slices)
        valid = mask[source_slices] & mask[target_slices]
        if not np.any(valid):
            continue
        source = quantized[source_slices][valid]
        target = quantized[target_slices][valid]
        np.add.at(glcm, (source, target), 1)
        np.add.at(glcm, (target, source), 1)
    return glcm


def glcm_features(glcm, roi=False):
    levels = glcm.shape[0]
    row = np.arange(levels).reshape((-1, 1))
    column = np.arange(levels).reshape((1, -1))
    contrast = float(np.sum((row - column) ** 2 * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + (row - column) ** 2)))
    energy = float(np.sqrt(np.sum(glcm**2)))
    mean_row = np.sum(row * glcm)
    mean_column = np.sum(column * glcm)
    std_row = np.sqrt(np.sum((row - mean_row) ** 2 * glcm))
    std_column = np.sqrt(np.sum((column - mean_column) ** 2 * glcm))
    correlation = (
        1.0
        if std_row * std_column == 0
        else float(
            np.sum((row - mean_row) * (column - mean_column) * glcm)
            / (std_row * std_column)
        )
    )
    prefix = "roi_" if roi else ""
    return {
        prefix + "Contrast": contrast,
        prefix + "Homogeneity": homogeneity,
        prefix + "Energy": energy,
        prefix + "Correlation": correlation,
    }


def get_glcm_feature_diffs(real_array, real_mask, synthetic_array, synthetic_mask):
    real = glcm_features(compute_glcm(real_array, real_mask))
    synthetic = glcm_features(compute_glcm(synthetic_array, synthetic_mask))
    return real, synthetic, {
        name: abs(real[name] - synthetic[name]) for name in real
    }


def get_glcm_roi_feature_diffs(real_array, real_mask, synthetic_array, synthetic_mask):
    real = glcm_features(compute_glcm(real_array, real_mask), roi=True)
    synthetic = glcm_features(
        compute_glcm(synthetic_array, synthetic_mask), roi=True
    )
    return real, synthetic, {
        name: abs(real[name] - synthetic[name]) for name in real
    }


def get_volume_feature_diffs(real_array, real_mask, synthetic_array, synthetic_mask):
    real_mask = _spatial_mask(real_mask)
    synthetic_mask = _spatial_mask(synthetic_mask)
    real_center = center_of_mass(real_mask)
    synthetic_center = center_of_mass(synthetic_mask)
    names = ("H-center", "W-center") if real_mask.ndim == 2 else (
        "D-center", "H-center", "W-center"
    )
    real = {"Volume": int(real_mask.sum())}
    synthetic = {"Volume": int(synthetic_mask.sum())}
    real.update(dict(zip(names, real_center)))
    synthetic.update(dict(zip(names, synthetic_center)))
    return real, synthetic, {
        name: abs(real[name] - synthetic[name]) for name in real
    }


def _spatial_mask(mask):
    mask = np.asarray(mask) > 0
    if mask.ndim in (3, 4):
        mask = np.any(mask, axis=0)
    if mask.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D spatial mask, got {mask.shape}.")
    return mask


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
            real_mask = _relative_foreground_mask(real_array, config.foreground_threshold)
            synthetic_mask = _relative_foreground_mask(
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
            differences_with_ids[name].append(
                {"value": difference, "sample": pair.id}
            )

    sample_count = len(pairs)
    outliers = {
        name: _outliers(values, differences_with_ids[name], config, name)
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
        "sample_counter": sample_count,
        "mean_real": {
            name: value / sample_count for name, value in real_totals.items()
        } if sample_count else {},
        "mean_synth": {
            name: value / sample_count for name, value in synthetic_totals.items()
        } if sample_count else {},
        "outliers": outliers,
        "all_diffs": dict(differences),
    }


def _outliers(values, entries, config, metric_name):
    if not values:
        return []
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    custom = config.outlier_thresholds.get(metric_name, {})
    lower = custom.get("min") if custom.get("min") is not None else lower
    upper = custom.get("max") if custom.get("max") is not None else upper
    return sorted(
        [entry for entry in entries if entry["value"] < lower or entry["value"] > upper],
        key=lambda entry: abs(entry["value"]),
        reverse=True,
    )


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _cutout_pairs(repository: StudyRepository) -> list[EvaluationPair]:
    pairs = []
    for synthetic in repository.list_synthetic_anomalies():
        real = repository.get_real_anomaly(synthetic.real_anomaly_id)
        pairs.append(
            EvaluationPair(
                id=synthetic.id,
                real_anomaly_id=real.id,
                synthetic_anomaly_id=synthetic.id,
                real_image_path=real.image_path,
                synthetic_image_path=synthetic.image_path,
                real_mask_path=real.segmentation_path,
                synthetic_mask_path=synthetic.segmentation_path,
            )
        )
    return pairs


def _roi_pairs(repository: StudyRepository) -> list[EvaluationPair]:
    pairs = []
    for entry in repository.hierarchy():
        placement = entry.placement
        if placement.roi_image_path is None:
            continue
        pairs.append(
            EvaluationPair(
                id=placement.id,
                real_anomaly_id=entry.real_anomaly.id,
                synthetic_anomaly_id=entry.synthetic_anomaly.id,
                real_image_path=entry.real_anomaly.roi_image_path,
                synthetic_image_path=placement.roi_image_path,
                real_mask_path=entry.real_anomaly.roi_segmentation_path,
                synthetic_mask_path=placement.roi_segmentation_path,
                placement_id=placement.id,
            )
        )
    return pairs


def evaluation_pipeline(
    repository: StudyRepository,
    artifact_store: ArtifactStore,
    paths: StudyPaths,
    config: EvaluationConfiguration,
):
    """Evaluate pairs joined through SyntheticAnomaly.real_anomaly_id."""
    config.validate()
    os.makedirs(paths.evaluation_results, exist_ok=True)
    if os.path.exists(paths.metric_diffs_csv):
        os.remove(paths.metric_diffs_csv)
    cutouts = _cutout_pairs(repository)
    rois = _roi_pairs(repository)
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


def analyze_results(results):
    if not results or not results["sample_counter"]:
        print("No results.")
        return
    print(f"Analysed {results['sample_counter']} explicit real/synthetic pairs.")
    for name, value in results["mean_real"].items():
        print(f"Real {name}: {value:.4f}")
    for name, value in results["mean_synth"].items():
        print(f"Synthetic {name}: {value:.4f}")


def save_difference_histograms(differences, save_path):
    if not differences:
        return
    columns = 2
    rows = (len(differences) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, rows * 4), squeeze=False)
    flat_axes = axes.flatten()
    for axis, (name, values) in zip(flat_axes, differences.items()):
        axis.hist(values, bins=32, edgecolor="black", alpha=0.7)
        axis.set_title(f"Difference Histogram: {name}")
        axis.set_xlabel("Absolute difference")
        axis.set_ylabel("Pairs")
    for axis in flat_axes[len(differences):]:
        axis.set_axis_off()
    figure.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def print_overlap_summary(outlier_groups):
    sample_metrics = defaultdict(set)
    for group in outlier_groups:
        for metric, entries in group.items():
            for entry in entries:
                sample_metrics[entry["sample"]].add(metric)
    if not sample_metrics:
        print("No metric outlier overlaps.")
        return
    counts = defaultdict(int)
    for metrics in sample_metrics.values():
        counts[len(metrics)] += 1
    print("Outlier overlaps: " + ", ".join(
        f"{sample_count} samples in {metric_count} metrics"
        for metric_count, sample_count in sorted(counts.items())
    ))
