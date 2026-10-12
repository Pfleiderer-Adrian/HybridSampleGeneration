"""Preparation and weighted template matching for 2D images and 3D volumes."""

from dataclasses import dataclass

import numpy as np
from skimage.feature import match_template

from hybrid_sample_generator.configuration.matching import MatchingConfiguration


@dataclass(frozen=True)
class PreparedArray:
    intensity: np.ndarray
    gradient: np.ndarray | None
    gradient_is_variable: bool


def _to_spatial(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim not in (3, 4):
        raise ValueError(f"Expected channel-first 2D/3D data, got {array.shape}.")
    if array.shape[0] == 1:
        return array[0]
    return np.max(array, axis=0)


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    magnitude = np.zeros_like(array, dtype=np.float64)
    for gradient in np.gradient(array):
        magnitude += gradient ** 2
    return np.sqrt(magnitude)


def prepare_matching_array(array, *, with_gradient: bool) -> PreparedArray:
    intensity = np.asarray(_to_spatial(array), dtype=np.float64)
    gradient = _gradient_magnitude(intensity) if with_gradient else None
    return PreparedArray(
        intensity=intensity,
        gradient=gradient,
        gradient_is_variable=bool(
            gradient is not None and np.std(gradient) > 1e-8
        ),
    )


def _window_sum(array, shape):
    """Sum every valid window without constructing a sliding-window tensor."""
    result = array
    for axis, size in enumerate(shape):
        padding = [(0, 0)] * result.ndim
        padding[axis] = (1, 0)
        cumulative = np.pad(np.cumsum(result, axis=axis, dtype=np.float64), padding)
        upper = [slice(None)] * result.ndim
        lower = [slice(None)] * result.ndim
        upper[axis] = slice(size, None)
        lower[axis] = slice(None, -size)
        result = cumulative[tuple(upper)] - cumulative[tuple(lower)]
    return result


def _correlation(template, control):
    """Float64 NCC with explicit rejection of numerically flat windows."""
    template = np.asarray(template, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    shape = template.shape
    output_shape = tuple(n - w + 1 for n, w in zip(control.shape, shape))
    invalid = np.full(output_shape, np.nan, dtype=np.float64)
    if not np.isfinite(template).all() or not np.isfinite(control).all():
        return invalid
    # Centering leaves NCC unchanged and reduces cancellation in window moments.
    template = template - template.mean()
    control = control - control.mean()
    tolerance = 64 * np.finfo(np.float64).eps
    energy = np.mean(control ** 2)
    if np.var(template) <= max(1e-16, tolerance * np.mean(template ** 2)):
        return invalid
    volume = template.size
    mean = _window_sum(control, shape) / volume
    second_moment = _window_sum(control ** 2, shape) / volume
    variance = second_moment - mean ** 2
    floor = np.maximum(1e-16, tolerance * np.maximum(energy, second_moment))
    valid = variance > floor
    if not valid.any():
        return invalid
    scores = match_template(control, template)
    valid &= np.isfinite(scores) & (np.abs(scores) <= 1 + 1e-6)
    # Only round-off-sized excursions may be clipped; invalid peaks are rejected.
    return np.where(valid, np.clip(scores, -1, 1), np.nan)


def template_matching(template, control, config: MatchingConfiguration, *, source_position=None):
    """Match full images while preparing each input once for this call."""
    with_gradient = float(config.gradient_weight) > 0
    return template_matching_prepared(
        prepare_matching_array(template, with_gradient=with_gradient),
        prepare_matching_array(control, with_gradient=with_gradient),
        config,
        source_position=source_position,
    )


def template_matching_prepared(
    template: PreparedArray,
    control: PreparedArray,
    config: MatchingConfiguration,
    *,
    source_position=None,
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
        score_maps.append(_correlation(template.intensity, control.intensity))
        weights.append(float(config.intensity_weight))
    if (
        float(config.gradient_weight) > 0
        and template.gradient_is_variable
        and control.gradient_is_variable
    ):
        score_maps.append(_correlation(template.gradient, control.gradient))
        weights.append(float(config.gradient_weight))
    if float(config.distance_weight) > 0:
        position = np.asarray(source_position, dtype=np.float64)
        shape = np.asarray(control.intensity.shape, dtype=np.float64)
        roi_shape = np.asarray(template.intensity.shape, dtype=np.float64)
        if position.shape != shape.shape or not np.isfinite(position).all() or np.any((position < 0) | (position > 1)):
            raise ValueError("Distance matching requires a normalized source_position in [0, 1] for each spatial axis.")
        lower = roi_shape / (2 * shape)
        upper = 1 - lower
        max_distance = np.linalg.norm(np.maximum(np.abs(lower - position), np.abs(upper - position)))
        output_shape = tuple(int(n - w + 1) for n, w in zip(shape, roi_shape))
        squared_distance = np.zeros(output_shape, dtype=np.float64)
        for axis, size in enumerate(output_shape):
            centers = (np.arange(size, dtype=np.float64) + roi_shape[axis] / 2) / shape[axis]
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = size
            squared_distance += ((centers - position[axis]) ** 2).reshape(broadcast_shape)
        proximity = np.ones(output_shape, dtype=np.float64) if max_distance == 0 else 1 - np.clip(np.sqrt(squared_distance) / max_distance, 0, 1)
        score_maps.append(proximity)
        weights.append(float(config.distance_weight))
    if not score_maps:
        return -2.0, None

    result = np.zeros_like(score_maps[0], dtype=np.float64)
    weight_sum = sum(weights)
    for score_map, weight in zip(score_maps, weights):
        result += score_map * (weight / weight_sum)
    valid = np.isfinite(result)
    if not valid.any():
        return -2.0, None
    result = np.where(valid, result, -np.inf)
    top_left = np.unravel_index(np.argmax(result), result.shape)
    center = tuple(
        float(offset + size / 2.0)
        for offset, size in zip(top_left, template.intensity.shape)
    )
    return float(np.max(result)), center


__all__ = [
    "PreparedArray",
    "prepare_matching_array",
    "template_matching",
    "template_matching_prepared",
]
