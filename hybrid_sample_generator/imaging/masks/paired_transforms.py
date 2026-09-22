"""Apply sampled spatial transforms to a label mask and its image together."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from hybrid_sample_generator.imaging.masks.geometry import (
    DEFAULT_PADDING_MODE,
    sample_uniform,
    validate_channel_first_mask,
)
from hybrid_sample_generator.imaging.masks.interpolation import interpolate_masked_regions


def pad_pair(mask: np.ndarray, image: np.ndarray, padding_factor: int):
    """Place both arrays on the same canvas, filling image padding with background."""
    validate_channel_first_mask(mask)
    if image.ndim != mask.ndim or image.shape[1:] != mask.shape[1:]:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}.")
    factor = int(padding_factor)
    if factor < 1:
        raise ValueError("padding_factor must be >= 1.")
    spatial = np.asarray(mask.shape[1:])
    total = spatial * factor - spatial
    pads = [(0, 0)] + [(int(n // 2), int(n - n // 2)) for n in total]
    padded_mask = np.pad(mask, pads, mode="constant")
    padded_image = np.pad(image, pads, mode="constant")
    if factor > 1:
        valid = np.pad(np.ones(mask.shape[1:], dtype=bool), pads[1:], mode="constant")
        for channel_index, channel in enumerate(image):
            background = channel[mask[0] == 0]
            finite = background[np.isfinite(background)]
            fill = float(finite.min()) if finite.size else 0.0
            padded_image[channel_index, ~valid] = fill
    return padded_mask, padded_image


def _axis_values(value, ndim: int, name: str) -> tuple[float, ...]:
    if np.isscalar(value):
        values = (float(value),) * ndim
    else:
        if len(value) != ndim:
            raise ValueError(f"{name} must contain {ndim} values.")
        values = tuple(float(part) for part in value)
    if not all(np.isfinite(part) for part in values):
        raise ValueError(f"{name} must be finite.")
    return values


def sample_warp(name: str, mask: np.ndarray, params: dict, rng: np.random.Generator):
    """Draw one transform and return a callable shared by mask and image."""
    shape = mask.shape
    ndim = mask.ndim
    if name in {"zoom", "stretch"}:
        if name == "zoom":
            low = float(params.get("min_zoom", 0.9))
            high = float(params.get("max_zoom", 0.9))
            if not 0 < low <= high <= 1:
                raise ValueError("Zoom bounds must satisfy 0 < min_zoom <= max_zoom <= 1.")
            scales = np.full(ndim, sample_uniform(low, high, rng=rng))
        else:
            scales = sample_uniform(
                params.get("min_stretch", 1.0),
                params.get("max_stretch", 1.2),
                rng=rng,
                size=ndim,
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("Stretch scales must be finite and positive.")
        matrix = np.diag(1.0 / np.asarray(scales))
        center = np.asarray(shape, dtype=float) / 2.0
        offset = center - matrix @ center
    elif name == "rotate":
        angle = sample_uniform(max_value=params.get("max_rotation", 5.0), rng=rng)
        foreground = np.where(mask != 0)
        if foreground[0].size == 0:
            return lambda array, order: array.copy()
        center = np.asarray(
            [(axis.min() + axis.max()) / 2.0 for axis in foreground], dtype=float
        )
        radians = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(radians), np.sin(radians)],
             [-np.sin(radians), np.cos(radians)]],
            dtype=float,
        )
        matrix = np.eye(ndim)
        matrix[-2:, -2:] = rotation
        offset = np.zeros(ndim)
        offset[-2:] = center[-2:] - rotation @ center[-2:]
    elif name == "elastic":
        sigma = _axis_values(params.get("sigma", 30), ndim, "sigma")
        magnitude = _axis_values(params.get("magnitude", 20), ndim, "magnitude")
        coordinates = np.meshgrid(
            *[np.arange(size, dtype=np.float32) for size in shape], indexing="ij"
        )
        displaced = []
        for axis in range(ndim):
            random_field = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            smooth = ndi.gaussian_filter(random_field, sigma=sigma, mode="reflect")
            maximum = float(np.max(np.abs(smooth)))
            if maximum > 0:
                smooth = smooth / maximum
            displaced.append(coordinates[axis] + smooth * magnitude[axis])

        def warp(array, order):
            return ndi.map_coordinates(
                array, displaced, order=order, mode=DEFAULT_PADDING_MODE,
                cval=0, prefilter=order > 1,
            )

        return warp
    else:
        raise ValueError(f"Unknown spatial transform: {name!r}")

    def warp(array, order):
        return ndi.affine_transform(
            array, matrix=matrix, offset=offset, output_shape=shape,
            order=order, mode=DEFAULT_PADDING_MODE, cval=0,
            prefilter=order > 1,
        )

    return warp


def apply_warp_pair(mask: np.ndarray, image: np.ndarray, warp):
    """Warp labels by nearest neighbour and intensities without boundary mixing."""
    foreground = mask[0] != 0
    warped_mask = warp(mask[0], 0).astype(mask.dtype, copy=False)[None, ...]
    warped_image = interpolate_masked_regions(
        image, foreground,
        warp=lambda array: warp(array, 1),
        nearest_warp=lambda array: warp(array, 0),
    )
    return warped_mask, warped_image


def dilate_class_pair(
    class_mask: np.ndarray,
    class_image: np.ndarray,
    iterations: int,
):
    """Expand a class and copy newly covered pixels from its nearest source pixel."""
    if iterations <= 0 or not np.any(class_mask):
        return class_mask.copy(), class_image.copy()
    enlarged = ndi.binary_dilation(class_mask, iterations=iterations)
    added = enlarged & ~class_mask
    result = class_image.copy()
    nearest = ndi.distance_transform_edt(
        ~class_mask, return_distances=False, return_indices=True
    )
    result[:, added] = class_image[
        (slice(None), *(axis[added] for axis in nearest))
    ]
    return enlarged, result


def fit_pair(mask: np.ndarray, image: np.ndarray, target_shape):
    """Use the mask's fit scale and final crop for both outputs."""
    target = np.asarray(target_shape, dtype=int)
    spatial = np.asarray(mask.shape[1:], dtype=int)
    if target.shape != spatial.shape or np.any(target <= 0) or np.any(target > spatial):
        raise ValueError(f"Invalid target shape {tuple(target)} for mask {mask.shape}.")
    crop_start = (spatial - target) // 2
    crop_end = crop_start + target - 1
    foreground = mask[0] != 0
    scale = 1.0
    if np.any(foreground):
        center = spatial.astype(float) / 2.0
        for axis, coordinates in enumerate(np.where(foreground)):
            low, high = float(coordinates.min()), float(coordinates.max())
            if low < crop_start[axis]:
                scale = min(scale, (center[axis] - crop_start[axis]) / (center[axis] - low))
            if high > crop_end[axis]:
                scale = min(scale, (crop_end[axis] - center[axis]) / (high - center[axis]))
    if scale < 1.0:
        scale = max(np.nextafter(scale, 0.0), np.finfo(float).eps)
        matrix = np.eye(len(spatial)) / scale
        center = spatial.astype(float) / 2.0
        offset = center - matrix @ center

        def warp(array, order):
            return ndi.affine_transform(
                array, matrix=matrix, offset=offset, output_shape=tuple(spatial),
                order=order, mode=DEFAULT_PADDING_MODE, cval=0,
                prefilter=order > 1,
            )

        mask, image = apply_warp_pair(mask, image, warp)
    crop = tuple(slice(int(start), int(start + size)) for start, size in zip(crop_start, target))
    return mask[(slice(None), *crop)], image[(slice(None), *crop)]
