from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.feature import match_template

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.InputSample import InputSample, iter_input_samples
from synthesizer.StudyRecords import HybridSample, OriginalSample, Placement, RealAnomaly
from synthesizer.StudyRepository import StudyRepository, stable_id, stable_seed
from synthesizer.configuration.matching import MatchingConfiguration


@dataclass(frozen=True)
class _Candidate:
    real_anomaly: RealAnomaly
    score: float | None
    position: tuple[float, ...]
    center: tuple[float, ...]
    roi_shape: tuple[int, ...]


def persist_original_sample(
    sample: InputSample,
    repository: StudyRepository,
    artifact_store: ArtifactStore,
) -> OriginalSample:
    image = np.asarray(sample.image)
    segmentation = np.asarray(sample.segmentation)
    if image.ndim not in (3, 4):
        raise ValueError(
            f"Input image must be (C,H,W) or (C,D,H,W), got {image.shape}."
        )
    if (
        segmentation.ndim != image.ndim
        or segmentation.shape[1:] != image.shape[1:]
        or segmentation.shape[0] not in (1, image.shape[0])
    ):
        raise ValueError(
            "Image and segmentation must share their spatial shape, and the "
            f"segmentation needs one or {image.shape[0]} channels: "
            f"{image.shape} vs {segmentation.shape}."
        )

    source_identity = (
        str(Path(sample.source_image_path).expanduser().resolve())
        if sample.source_image_path
        else sample.source_name
    )
    record_id = stable_id("original", source_identity)
    existing = repository.find_original_sample(record_id)
    if existing is not None:
        stored_image = artifact_store.load_array(existing.image_path, mmap_mode="r")
        if existing.spatial_dimensions != image.ndim - 1 or not np.array_equal(
            stored_image, image
        ):
            raise ValueError(
                f"Source identity {source_identity!r} refers to different image data. "
                "Use a unique source path or source_name for every original sample."
            )
        return existing

    image_path = artifact_store.save_entity_array(
        "original_samples", record_id, "image", image
    )
    segmentation_path = artifact_store.save_entity_array(
        "original_samples", record_id, "segmentation", segmentation
    )
    metadata = dict(sample.metadata)
    if sample.source_image_path:
        metadata["source_image_path"] = str(sample.source_image_path)
    if sample.source_segmentation_path:
        metadata["source_segmentation_path"] = str(sample.source_segmentation_path)

    record = OriginalSample(
        id=record_id,
        source_name=sample.source_name,
        image_path=image_path,
        segmentation_path=segmentation_path,
        spatial_dimensions=image.ndim - 1,
        metadata=metadata,
    )
    repository.upsert_original_sample(record)
    return record


def plan_hybrid_samples(
    control_samples,
    repository: StudyRepository,
    artifact_store: ArtifactStore,
    config: MatchingConfiguration,
) -> list[HybridSample]:
    """Create normalized hybrid and placement records without performing fusion."""
    config.validate()
    real_anomalies = repository.list_real_anomalies()
    synthetic_anomalies = repository.list_synthetic_anomalies()
    if not real_anomalies:
        raise ValueError("No real anomalies registered. Run extract_anomalies first.")
    if not synthetic_anomalies:
        raise ValueError(
            "No synthetic anomalies registered. Run generate_synthetic_anomalies first."
        )

    variants_by_real = {record.id: [] for record in real_anomalies}
    for synthetic in synthetic_anomalies:
        variants_by_real.setdefault(synthetic.real_anomaly_id, []).append(synthetic)
    real_anomalies = [record for record in real_anomalies if variants_by_real.get(record.id)]
    if not real_anomalies:
        raise ValueError("No real anomaly has a generated synthetic variant.")

    repository.clear_hybrid_plans()
    used_synthetic_ids: set[str] = set()
    planned_original_ids: set[str] = set()
    planned: list[HybridSample] = []

    for sample_index, sample in enumerate(iter_input_samples(control_samples)):
        original = persist_original_sample(sample, repository, artifact_store)
        if original.id in planned_original_ids:
            raise ValueError(
                f"Control source {sample.source_name!r} occurs more than once. "
                "Each original may be planned only once per call."
            )
        planned_original_ids.add(original.id)
        candidates = _match_real_anomalies(
            sample,
            original,
            real_anomalies,
            artifact_store,
            config,
        )
        if not candidates:
            print(f"Warning: no matching real anomaly found for {sample.source_name}.")
            continue

        for hybrid_index in range(int(config.hybrids_per_original)):
            hybrid_id = stable_id("hybrid", original.id, hybrid_index)
            desired_count = _placement_count(
                config,
                stable_seed(config.seed, original.id, hybrid_index, "placement_count"),
            )
            ordered_candidates = candidates
            if config.routine in {"local", "fixed_from_extraction_control_fusion"}:
                shift = (
                    sample_index * int(config.hybrids_per_original) + hybrid_index
                ) % len(candidates)
                ordered_candidates = candidates[shift:] + candidates[:shift]
            options = _variant_options(
                ordered_candidates,
                variants_by_real,
                hybrid_index=hybrid_index,
            )

            selected = []
            selected_synthetic_ids: set[str] = set()
            selected_real_ids: set[str] = set()
            used_positions: list[tuple[tuple[float, ...], tuple[int, ...], str]] = []
            for candidate, synthetic in options:
                if len(selected) >= desired_count:
                    break
                if synthetic.id in selected_synthetic_ids:
                    continue
                if not config.reuse_synthetic_across_hybrids and synthetic.id in used_synthetic_ids:
                    continue
                is_sibling = candidate.real_anomaly.id in selected_real_ids
                if is_sibling and not config.allow_sibling_variants_in_same_hybrid:
                    continue
                if not is_sibling and check_roi_overlap(
                    candidate.center,
                    candidate.roi_shape,
                    [(center, shape) for center, shape, _ in used_positions],
                ):
                    continue
                selected.append((candidate, synthetic))
                selected_synthetic_ids.add(synthetic.id)
                selected_real_ids.add(candidate.real_anomaly.id)
                used_positions.append(
                    (candidate.center, candidate.roi_shape, candidate.real_anomaly.id)
                )

            if not selected:
                print(
                    f"Warning: no unused synthetic variant available for {sample.source_name} "
                    f"hybrid variant {hybrid_index}."
                )
                continue

            hybrid = HybridSample(
                id=hybrid_id,
                original_sample_id=original.id,
                variant_index=hybrid_index,
            )
            repository.upsert_hybrid_sample(hybrid)
            planned.append(hybrid)

            for order_index, (candidate, synthetic) in enumerate(selected):
                position_z, position_y, position_x = _position_columns(candidate.position)
                placement = Placement(
                    id=stable_id("placement", hybrid.id, order_index),
                    hybrid_sample_id=hybrid.id,
                    synthetic_anomaly_id=synthetic.id,
                    order_index=order_index,
                    spatial_dimensions=len(candidate.position),
                    position_z=position_z,
                    position_y=position_y,
                    position_x=position_x,
                    score=candidate.score,
                    method=config.routine,
                )
                repository.upsert_placement(placement)
                used_synthetic_ids.add(synthetic.id)

            if len(selected) < desired_count:
                print(
                    f"Warning: planned {len(selected)} of {desired_count} requested placements "
                    f"for {sample.source_name}, hybrid variant {hybrid_index}."
                )

    return planned


def _match_real_anomalies(
    sample: InputSample,
    original: OriginalSample,
    real_anomalies: list[RealAnomaly],
    artifact_store: ArtifactStore,
    config: MatchingConfiguration,
) -> list[_Candidate]:
    control = np.asarray(sample.image)
    spatial_shape = np.asarray(control.shape[1:], dtype=float)
    routine = config.routine

    if routine == "fixed_from_extraction_anomaly_fusion":
        source_reals = [
            record for record in real_anomalies if record.original_sample_id == original.id
        ]
        return [
            _fixed_candidate(record, spatial_shape, artifact_store) for record in source_reals
        ]

    pool = list(real_anomalies)
    if routine == "batchwise" and len(pool) > int(config.batch_size):
        rng = np.random.default_rng(stable_seed(config.seed, original.id, "batchwise"))
        indices = sorted(
            rng.choice(len(pool), size=int(config.batch_size), replace=False).tolist()
        )
        pool = [pool[index] for index in indices]

    if routine == "fixed_from_extraction_control_fusion":
        return [_fixed_candidate(record, spatial_shape, artifact_store) for record in pool]

    candidates = []
    for record in pool:
        roi = artifact_store.load_array(record.roi_image_path)
        score, center = template_matching(roi, control, config)
        if center is None or not np.isfinite(score) or score < -1:
            continue
        position = tuple(
            float(value) for value in (np.asarray(center, dtype=float) / spatial_shape)
        )
        candidates.append(
            _Candidate(
                real_anomaly=record,
                score=float(score),
                position=position,
                center=tuple(float(value) for value in center),
                roi_shape=tuple(int(value) for value in roi.shape[1:]),
            )
        )

    if routine in {"global", "batchwise"}:
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.real_anomaly.id))
    return candidates


def _fixed_candidate(
    record: RealAnomaly,
    spatial_shape: np.ndarray,
    artifact_store: ArtifactStore,
) -> _Candidate:
    position = tuple(float(value) for value in record.source_position)
    center = tuple(float(value) for value in (np.asarray(position) * spatial_shape))
    roi = artifact_store.load_array(record.roi_image_path, mmap_mode="r")
    return _Candidate(
        real_anomaly=record,
        score=None,
        position=position,
        center=center,
        roi_shape=tuple(int(value) for value in roi.shape[1:]),
    )


def _variant_options(
    candidates,
    variants_by_real,
    *,
    hybrid_index: int,
):
    options = []
    for candidate in candidates:
        variants = variants_by_real[candidate.real_anomaly.id]
        shift = hybrid_index % len(variants)
        for synthetic in variants[shift:] + variants[:shift]:
            options.append((candidate, synthetic))
    return options


def _placement_count(config: MatchingConfiguration, seed: int) -> int:
    count = int(config.anomalies_per_hybrid)
    deviation = int(config.max_anomalies_per_hybrid_deviation)
    if deviation:
        count += int(np.random.default_rng(seed).integers(-deviation, deviation + 1))
    return max(1, count)


def _position_columns(position: tuple[float, ...]):
    if len(position) == 2:
        return None, float(position[0]), float(position[1])
    if len(position) == 3:
        return float(position[0]), float(position[1]), float(position[2])
    raise ValueError(f"Position must be 2D or 3D, got {position!r}")


def _to_spatial(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim not in (3, 4):
        raise ValueError(f"Expected channel-first 2D/3D data, got {array.shape}.")
    if array.shape[0] == 1:
        return array[0]
    return np.max(array, axis=0)


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    magnitude = np.zeros_like(array, dtype=np.float32)
    for gradient in np.gradient(array):
        magnitude += gradient.astype(np.float32) ** 2
    return np.sqrt(magnitude)


def template_matching(template, control, config: MatchingConfiguration):
    template = _to_spatial(template)
    control = _to_spatial(control)
    if any(template_size > control_size for template_size, control_size in zip(template.shape, control.shape)):
        return -2.0, None

    score_maps = []
    weights = []
    if float(config.intensity_weight) > 0:
        score_maps.append(match_template(control, template))
        weights.append(float(config.intensity_weight))
    if float(config.gradient_weight) > 0:
        template_gradient = _gradient_magnitude(template)
        control_gradient = _gradient_magnitude(control)
        if np.std(template_gradient) > 1e-8 and np.std(control_gradient) > 1e-8:
            score_maps.append(match_template(control_gradient, template_gradient))
            weights.append(float(config.gradient_weight))
    if not score_maps:
        return -2.0, None

    result = np.zeros_like(score_maps[0], dtype=np.float32)
    for score_map, weight in zip(score_maps, weights):
        result += score_map * (weight / sum(weights))
    top_left = np.unravel_index(np.argmax(result), result.shape)
    center = tuple(
        float(offset + size / 2.0) for offset, size in zip(top_left, template.shape)
    )
    return float(np.max(result)), center


def check_roi_overlap(center, roi_shape, used_positions) -> bool:
    for used_center, used_shape in used_positions:
        overlaps = all(
            abs(float(used_center[axis]) - float(center[axis]))
            < (float(used_shape[axis]) + float(roi_shape[axis])) / 2.0
            for axis in range(len(center))
        )
        if overlaps:
            return True
    return False


def combine_label_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    overwrite: bool = True,
    return_dtype=None,
) -> np.ndarray:
    if not isinstance(mask_a, np.ndarray) or not isinstance(mask_b, np.ndarray):
        raise TypeError("mask_a and mask_b must be NumPy arrays.")
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shapes must match, got {mask_a.shape} vs {mask_b.shape}.")
    if mask_a.ndim not in (3, 4):
        raise ValueError(f"Expected channel-first 2D/3D masks, got {mask_a.shape}.")
    result = mask_a.copy()
    foreground = mask_b > 0
    if not overwrite:
        foreground &= result == 0
    result[foreground] = mask_b[foreground]
    return result if return_dtype is None else result.astype(return_dtype, copy=False)


def ssim_01(x, y, data_range=None, k1=0.01, k2=0.03):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim >= 3:
        return float(np.mean([ssim_01(a, b, data_range, k1, k2) for a, b in zip(x, y)]))
    if data_range is None:
        data_range = max(x.max(), y.max()) - min(x.min(), y.min())
        if data_range == 0:
            return 1.0 if np.allclose(x, y) else 0.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mu_x) * (y - mu_y)).mean()
    value = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(np.clip(value, 0.0, 1.0))
