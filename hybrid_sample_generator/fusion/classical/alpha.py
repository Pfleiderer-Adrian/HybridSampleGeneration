"""Edge-aware alpha-mask construction for classical fusion."""

import numpy as np
import scipy.ndimage

from hybrid_sample_generator.fusion.classical.configuration import CONFIDENCE_LEVELS


def sample_alpha_params(config):
    max_alpha, sq, steepness = config.max_alpha, config.sq, config.steepness_factor
    if config.fusion_variation:
        z_score = confidence_z_score(config)
        max_alpha = float(np.clip(np.random.normal(max_alpha, config.alpha_variation / z_score), 0.0, 1.0))
        sq = float(np.maximum(0.1, np.random.normal(sq, config.sq_variation / z_score)))
        steepness = float(np.maximum(0.1, np.random.normal(steepness, config.steepness_variation / z_score)))
    return max_alpha, sq, steepness


def get_alpha_mask_2d(anomaly, config, valid_mask):
    max_alpha, sq, steepness = sample_alpha_params(config)
    alpha = np.zeros_like(anomaly, dtype=np.float32)
    if not np.any(valid_mask > 0):
        return alpha
    clean = clean_edge_mask(anomaly, valid_mask, config) if config.fusion_use_sobel_for_alpha_mask else valid_mask > 0
    if np.any(clean):
        alpha[:, :] = distance_alpha(clean, max_alpha, sq, steepness, config)
    return alpha


def get_alpha_mask_3d(anomaly, config, valid_mask):
    max_alpha, sq, steepness = sample_alpha_params(config)
    alpha = np.zeros_like(anomaly, dtype=np.float32)
    for depth in range(anomaly.shape[0]):
        valid = valid_mask[depth]
        if not np.any(valid > 0):
            continue
        clean = clean_edge_mask(anomaly[depth], valid, config) if config.fusion_use_sobel_for_alpha_mask else valid > 0
        if np.any(clean):
            alpha[depth] = distance_alpha(clean, max_alpha, sq, steepness, config)
    return alpha


def clean_edge_mask(image, valid_mask, config):
    size = config.dilation_size
    y, x = np.ogrid[-size:size + 1, -size:size + 1]
    brush = x**2 + y**2 <= size**2
    shave_structure = scipy.ndimage.generate_binary_structure(2, 2)
    grad_y = scipy.ndimage.sobel(image, axis=0)
    grad_x = scipy.ndimage.sobel(image, axis=1)
    magnitude = np.hypot(grad_y, grad_x)
    if magnitude.max() > 0:
        magnitude /= magnitude.max()
    edges = magnitude > config.sobel_threshold
    thick = scipy.ndimage.binary_dilation(edges, structure=brush)
    body = scipy.ndimage.binary_fill_holes(thick)
    clean = scipy.ndimage.binary_erosion(body, structure=brush, iterations=1)
    if config.shave_pixels > 0:
        clean = scipy.ndimage.binary_erosion(clean, structure=shave_structure, iterations=config.shave_pixels)
    clean = clean.copy()
    clean[valid_mask <= 0] = False
    return clean


def distance_alpha(clean_mask, max_alpha, sq, steepness, config):
    factor = config.upsampling_factor
    if factor > 1:
        large = scipy.ndimage.zoom(clean_mask, factor, order=0)
        distance = scipy.ndimage.zoom(scipy.ndimage.distance_transform_edt(large), 1 / factor, order=1)
    else:
        distance = scipy.ndimage.distance_transform_edt(clean_mask)
    if distance.max() > 0:
        distance /= distance.max()
    distance = np.clip(distance * steepness, 0, 1.0) ** sq * max_alpha
    distance[~clean_mask] = 0.0
    return distance.astype(np.float32, copy=False)


def confidence_z_score(params):
    try:
        return CONFIDENCE_LEVELS[params.selected_confidence]
    except KeyError as exc:
        raise ValueError(
            f"Unknown selected_confidence {params.selected_confidence!r}. "
            f"Supported values: {list(CONFIDENCE_LEVELS)}"
        ) from exc


__all__ = ["get_alpha_mask_2d", "get_alpha_mask_3d", "sample_alpha_params"]
