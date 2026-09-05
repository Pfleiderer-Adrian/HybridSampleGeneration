from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from skimage.feature import match_template
from tqdm import tqdm

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyRecords import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
)
from synthesizer.StudyRepository import StudyRepository, stable_id, stable_seed
from synthesizer.configuration.matching import MatchingConfiguration


MATCHER_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class _Candidate:
    real_anomaly: RealAnomaly
    score: float | None
    position: tuple[float, ...]
    center: tuple[float, ...]
    roi_shape: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedArray:
    intensity: np.ndarray
    gradient: np.ndarray | None
    gradient_is_variable: bool


@dataclass
class _MatchingStats:
    controls: int = 0
    computed_pairs: int = 0
    cache_hits: int = 0
    preparation_seconds: float = 0.0
    matching_seconds: float = 0.0
    persistence_seconds: float = 0.0

    def log(self, total_seconds: float) -> None:
        print(
            "Matching summary: "
            f"originals={self.controls}, computed_pairs={self.computed_pairs}, "
            f"cache_hits={self.cache_hits}, "
            f"prepare={self.preparation_seconds:.3f}s, "
            f"match={self.matching_seconds:.3f}s, "
            f"persist={self.persistence_seconds:.3f}s, total={total_seconds:.3f}s"
        )


def plan_hybrid_samples(
    repository: StudyRepository,
    artifact_store: ArtifactStore,
    config: MatchingConfiguration,
) -> list[HybridSample]:
    """Create normalized hybrid and placement records without performing fusion."""
    started_at = perf_counter()
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

    originals = _matching_originals(repository, config.routine)
    if not originals:
        target_kind = (
            "anomalous"
            if config.routine == "fixed_from_extraction_anomaly_fusion"
            else "control"
        )
        raise ValueError(
            f"No {target_kind} original samples found. Run ingest_dataset first."
        )

    stats = _MatchingStats()
    prepared_rois: dict[str, _PreparedArray] = {}
    roi_shapes = _real_anomaly_roi_shapes(real_anomalies, artifact_store, stats)

    used_synthetic_ids: set[str] = set()
    planned: list[HybridSample] = []
    placements: list[Placement] = []

    for sample_index, original in enumerate(
        tqdm(originals, desc="Planning hybrid samples", unit="sample")
    ):
        stats.controls += 1
        candidates = _match_real_anomalies(
            original,
            real_anomalies,
            prepared_rois,
            roi_shapes,
            repository,
            artifact_store,
            config,
            stats,
        )
        if not candidates:
            print(f"Warning: no matching real anomaly found for {original.source_name}.")
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
                    f"Warning: no unused synthetic variant available for {original.source_name} "
                    f"hybrid variant {hybrid_index}."
                )
                continue

            hybrid = HybridSample(
                id=hybrid_id,
                original_sample_id=original.id,
                variant_index=hybrid_index,
            )
            planned.append(hybrid)

            for order_index, (candidate, synthetic) in enumerate(selected):
                position_z, position_y, position_x = _position_columns(candidate.position)
                placements.append(
                    Placement(
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
                )
                used_synthetic_ids.add(synthetic.id)

            if len(selected) < desired_count:
                print(
                    f"Warning: planned {len(selected)} of {desired_count} requested placements "
                    f"for {original.source_name}, hybrid variant {hybrid_index}."
                )

    persistence_started = perf_counter()
    repository.replace_hybrid_plan(planned, placements)
    stats.persistence_seconds += perf_counter() - persistence_started
    stats.log(perf_counter() - started_at)
    return planned


def _match_real_anomalies(
    original: OriginalSample,
    real_anomalies: list[RealAnomaly],
    prepared_rois: dict[str, _PreparedArray],
    roi_shapes: dict[str, tuple[int, ...]],
    repository: StudyRepository,
    artifact_store: ArtifactStore,
    config: MatchingConfiguration,
    stats: _MatchingStats,
) -> list[_Candidate]:
    routine = config.routine

    if routine == "fixed_from_extraction_anomaly_fusion":
        preparation_started = perf_counter()
        control = artifact_store.load_array(original.image_path, mmap_mode="r")
        spatial_shape = np.asarray(control.shape[1:], dtype=float)
        stats.preparation_seconds += perf_counter() - preparation_started
        source_reals = [
            record for record in real_anomalies if record.original_sample_id == original.id
        ]
        return [
            _fixed_candidate(record, spatial_shape, roi_shapes[record.id])
            for record in source_reals
        ]

    pool = list(real_anomalies)
    if routine == "batchwise" and len(pool) > int(config.batch_size):
        rng = np.random.default_rng(stable_seed(config.seed, original.id, "batchwise"))
        indices = sorted(
            rng.choice(len(pool), size=int(config.batch_size), replace=False).tolist()
        )
        pool = [pool[index] for index in indices]

    if routine == "fixed_from_extraction_control_fusion":
        preparation_started = perf_counter()
        control = artifact_store.load_array(original.image_path, mmap_mode="r")
        spatial_shape = np.asarray(control.shape[1:], dtype=float)
        stats.preparation_seconds += perf_counter() - preparation_started
        return [
            _fixed_candidate(record, spatial_shape, roi_shapes[record.id])
            for record in pool
        ]

    matcher_signature = _matcher_signature(config)
    cached_by_real_id = {
        candidate.real_anomaly_id: candidate
        for candidate in repository.list_match_candidates(
            original.id, matcher_signature
        )
    }
    missing_records = [record for record in pool if record.id not in cached_by_real_id]
    control_prepared = None
    spatial_shape = None
    if missing_records:
        preparation_started = perf_counter()
        control = artifact_store.load_array(original.image_path)
        spatial_shape = np.asarray(control.shape[1:], dtype=float)
        control_prepared = _prepare_matching_array(
            control,
            with_gradient=float(config.gradient_weight) > 0,
        )
        stats.preparation_seconds += perf_counter() - preparation_started

    candidates = []
    new_cache_records: list[MatchCandidate] = []
    for record in pool:
        cached = cached_by_real_id.get(record.id)
        if cached is None:
            if control_prepared is None or spatial_shape is None:
                raise RuntimeError("Missing prepared control for an uncached match pair.")
            preparation_started = perf_counter()
            prepared_roi = prepared_rois.get(record.id)
            if prepared_roi is None:
                roi = artifact_store.load_array(record.roi_image_path)
                prepared_roi = _prepare_matching_array(
                    roi,
                    with_gradient=float(config.gradient_weight) > 0,
                )
                prepared_rois[record.id] = prepared_roi
            stats.preparation_seconds += perf_counter() - preparation_started
            matching_started = perf_counter()
            score, center = _template_matching_prepared(
                prepared_roi, control_prepared, config
            )
            stats.matching_seconds += perf_counter() - matching_started
            stats.computed_pairs += 1
            is_valid = center is not None and np.isfinite(score) and score >= -1
            position = (
                tuple(
                    float(value)
                    for value in (np.asarray(center, dtype=float) / spatial_shape)
                )
                if is_valid
                else None
            )
            cached = MatchCandidate(
                original_sample_id=original.id,
                real_anomaly_id=record.id,
                matcher_signature=matcher_signature,
                is_valid=is_valid,
                score=float(score) if np.isfinite(score) else None,
                position=position,
                center=(
                    tuple(float(value) for value in center)
                    if center is not None
                    else None
                ),
                roi_shape=roi_shapes[record.id],
            )
            new_cache_records.append(cached)
        else:
            stats.cache_hits += 1

        if not cached.is_valid or cached.position is None or cached.center is None:
            continue
        candidates.append(
            _Candidate(
                real_anomaly=record,
                score=cached.score,
                position=cached.position,
                center=cached.center,
                roi_shape=cached.roi_shape,
            )
        )

    if new_cache_records:
        persistence_started = perf_counter()
        repository.upsert_match_candidates(new_cache_records)
        stats.persistence_seconds += perf_counter() - persistence_started

    if routine in {"global", "batchwise"}:
        candidates.sort(
            key=lambda candidate: (
                -float(candidate.score),
                candidate.real_anomaly.id,
            )
        )
    return candidates


def _fixed_candidate(
    record: RealAnomaly,
    spatial_shape: np.ndarray,
    roi_shape: tuple[int, ...],
) -> _Candidate:
    position = tuple(float(value) for value in record.source_position)
    center = tuple(float(value) for value in (np.asarray(position) * spatial_shape))
    return _Candidate(
        real_anomaly=record,
        score=None,
        position=position,
        center=center,
        roi_shape=roi_shape,
    )


def _matching_originals(
    repository: StudyRepository,
    routine: str,
) -> list[OriginalSample]:
    return repository.list_original_samples(
        has_anomaly=routine == "fixed_from_extraction_anomaly_fusion"
    )


def _real_anomaly_roi_shapes(
    real_anomalies: list[RealAnomaly],
    artifact_store: ArtifactStore,
    stats: _MatchingStats,
) -> dict[str, tuple[int, ...]]:
    roi_shapes = {}
    for record in real_anomalies:
        stored_shape = record.metadata.get("roi_shape")
        if stored_shape is not None:
            roi_shapes[record.id] = tuple(int(value) for value in stored_shape)
            continue
        preparation_started = perf_counter()
        roi = artifact_store.load_array(record.roi_image_path, mmap_mode="r")
        roi_shapes[record.id] = tuple(int(value) for value in roi.shape[1:])
        stats.preparation_seconds += perf_counter() - preparation_started
    return roi_shapes


def _matcher_signature(config: MatchingConfiguration) -> str:
    return stable_id(
        "matcher",
        MATCHER_ALGORITHM_VERSION,
        float(config.intensity_weight),
        float(config.gradient_weight),
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


def _prepare_matching_array(array, *, with_gradient: bool) -> _PreparedArray:
    intensity = _to_spatial(array)
    gradient = _gradient_magnitude(intensity) if with_gradient else None
    return _PreparedArray(
        intensity=intensity,
        gradient=gradient,
        gradient_is_variable=bool(
            gradient is not None and np.std(gradient) > 1e-8
        ),
    )


def template_matching(template, control, config: MatchingConfiguration):
    """Match full images while preparing each input once for this call."""
    with_gradient = float(config.gradient_weight) > 0
    return _template_matching_prepared(
        _prepare_matching_array(template, with_gradient=with_gradient),
        _prepare_matching_array(control, with_gradient=with_gradient),
        config,
    )


def _template_matching_prepared(
    template: _PreparedArray,
    control: _PreparedArray,
    config: MatchingConfiguration,
):
    if any(
        template_size > control_size
        for template_size, control_size in zip(
            template.intensity.shape, control.intensity.shape
        )
    ):
        return -2.0, None

    score_maps = []
    weights = []
    if float(config.intensity_weight) > 0:
        score_maps.append(match_template(control.intensity, template.intensity))
        weights.append(float(config.intensity_weight))
    if (
        float(config.gradient_weight) > 0
        and template.gradient_is_variable
        and control.gradient_is_variable
    ):
        score_maps.append(match_template(control.gradient, template.gradient))
        weights.append(float(config.gradient_weight))
    if not score_maps:
        return -2.0, None

    result = np.zeros_like(score_maps[0], dtype=np.float32)
    weight_sum = sum(weights)
    for score_map, weight in zip(score_maps, weights):
        result += score_map * (weight / weight_sum)
    top_left = np.unravel_index(np.argmax(result), result.shape)
    center = tuple(
        float(offset + size / 2.0)
        for offset, size in zip(top_left, template.intensity.shape)
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
