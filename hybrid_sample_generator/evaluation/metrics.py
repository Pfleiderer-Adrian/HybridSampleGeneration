"""Texture and volume metrics for real/synthetic anomaly pairs."""

import numpy as np
from scipy.ndimage import center_of_mass


def relative_foreground_mask(array, background_threshold):
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
        channel_mask = (
            mask[min(channel_index, mask.shape[0] - 1)]
            if mask.ndim == volume.ndim
            else mask
        )
        glcm += _compute_glcm_for_channel(volume[channel_index], channel_mask > 0, levels)
    total = glcm.sum()
    return (glcm / total).astype(np.float32) if total > 0 else glcm.astype(np.float32)


def _compute_glcm_for_channel(volume, mask, levels=32):
    lesion = volume[mask]
    glcm = np.zeros((levels, levels), dtype=np.float64)
    if lesion.size == 0:
        return glcm
    minimum = float(np.nanmin(lesion))
    value_range = float(np.nanmax(lesion)) - minimum
    quantized = (
        np.zeros_like(volume, dtype=np.int64)
        if not np.isfinite(value_range) or value_range <= 0
        else np.clip(
            ((volume - minimum) / value_range * (levels - 1)).astype(np.int64),
            0,
            levels - 1,
        )
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
        if np.any(valid):
            source = quantized[source_slices][valid]
            target = quantized[target_slices][valid]
            np.add.at(glcm, (source, target), 1)
            np.add.at(glcm, (target, source), 1)
    return glcm


def glcm_features(glcm, roi=False):
    levels = glcm.shape[0]
    row = np.arange(levels).reshape((-1, 1))
    column = np.arange(levels).reshape((1, -1))
    mean_row = np.sum(row * glcm)
    mean_column = np.sum(column * glcm)
    std_row = np.sqrt(np.sum((row - mean_row) ** 2 * glcm))
    std_column = np.sqrt(np.sum((column - mean_column) ** 2 * glcm))
    correlation = 1.0 if std_row * std_column == 0 else float(
        np.sum((row - mean_row) * (column - mean_column) * glcm)
        / (std_row * std_column)
    )
    prefix = "roi_" if roi else ""
    return {
        prefix + "Contrast": float(np.sum((row - column) ** 2 * glcm)),
        prefix + "Homogeneity": float(np.sum(glcm / (1.0 + (row - column) ** 2))),
        prefix + "Energy": float(np.sqrt(np.sum(glcm**2))),
        prefix + "Correlation": correlation,
    }


def get_glcm_feature_diffs(real_array, real_mask, synthetic_array, synthetic_mask):
    real = glcm_features(compute_glcm(real_array, real_mask))
    synthetic = glcm_features(compute_glcm(synthetic_array, synthetic_mask))
    return real, synthetic, {name: abs(real[name] - synthetic[name]) for name in real}


def get_glcm_roi_feature_diffs(real_array, real_mask, synthetic_array, synthetic_mask):
    real = glcm_features(compute_glcm(real_array, real_mask), roi=True)
    synthetic = glcm_features(compute_glcm(synthetic_array, synthetic_mask), roi=True)
    return real, synthetic, {name: abs(real[name] - synthetic[name]) for name in real}


def get_volume_feature_diffs(real_array, real_mask, synthetic_array, synthetic_mask):
    real_mask = _spatial_mask(real_mask)
    synthetic_mask = _spatial_mask(synthetic_mask)
    names = ("H-center", "W-center") if real_mask.ndim == 2 else (
        "D-center", "H-center", "W-center"
    )
    real = {"Volume": int(real_mask.sum())}
    synthetic = {"Volume": int(synthetic_mask.sum())}
    real.update(dict(zip(names, center_of_mass(real_mask))))
    synthetic.update(dict(zip(names, center_of_mass(synthetic_mask))))
    return real, synthetic, {name: abs(real[name] - synthetic[name]) for name in real}


def _spatial_mask(mask):
    mask = np.asarray(mask) > 0
    if mask.ndim in (3, 4):
        mask = np.any(mask, axis=0)
    if mask.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D spatial mask, got {mask.shape}.")
    return mask


__all__ = ["compute_glcm", "get_glcm_feature_diffs", "get_glcm_roi_feature_diffs", "get_volume_feature_diffs", "glcm_features", "relative_foreground_mask"]
