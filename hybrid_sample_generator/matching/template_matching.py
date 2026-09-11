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
    array = np.asarray(array, dtype=np.float32)
    magnitude = np.zeros_like(array, dtype=np.float32)
    for gradient in np.gradient(array):
        magnitude += gradient.astype(np.float32) ** 2
    return np.sqrt(magnitude)


def prepare_matching_array(array, *, with_gradient: bool) -> PreparedArray:
    intensity = _to_spatial(array)
    gradient = _gradient_magnitude(intensity) if with_gradient else None
    return PreparedArray(
        intensity=intensity,
        gradient=gradient,
        gradient_is_variable=bool(
            gradient is not None and np.std(gradient) > 1e-8
        ),
    )


def template_matching(template, control, config: MatchingConfiguration):
    """Match full images while preparing each input once for this call."""
    with_gradient = float(config.gradient_weight) > 0
    return template_matching_prepared(
        prepare_matching_array(template, with_gradient=with_gradient),
        prepare_matching_array(control, with_gradient=with_gradient),
        config,
    )


def template_matching_prepared(
    template: PreparedArray,
    control: PreparedArray,
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


__all__ = [
    "PreparedArray",
    "prepare_matching_array",
    "template_matching",
    "template_matching_prepared",
]
