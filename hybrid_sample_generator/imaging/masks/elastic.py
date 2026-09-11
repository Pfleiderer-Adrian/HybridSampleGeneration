"""Elastic mask transformations and their size-dependent defaults."""

import numpy as np
import scipy.ndimage as ndi

from hybrid_sample_generator.imaging.masks.geometry import (
    DEFAULT_PADDING_MODE,
    validate_channel_first_mask,
)


def random_elastic_transform(
    mask_np: np.ndarray,
    sigma=30,
    magnitude=20,
    rng=None,
):
    """Apply a smooth random displacement field to a channel-first label mask."""
    validate_channel_first_mask(mask_np)
    transformed_mask = mask_np[0].copy()
    spatial_ndim = transformed_mask.ndim
    sigma = _as_axis_tuple(sigma, spatial_ndim, "sigma")
    magnitude = _as_axis_tuple(magnitude, spatial_ndim, "magnitude")
    rng = rng if rng is not None else np.random.default_rng()

    coordinates = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in transformed_mask.shape],
        indexing="ij",
    )
    displaced_coordinates = []
    for axis in range(spatial_ndim):
        random_field = rng.uniform(
            -1.0, 1.0, size=transformed_mask.shape
        ).astype(np.float32)
        smooth_field = ndi.gaussian_filter(random_field, sigma=sigma, mode="reflect")
        max_abs = np.max(np.abs(smooth_field))
        if max_abs > 0:
            smooth_field = smooth_field / max_abs
        displaced_coordinates.append(
            coordinates[axis] + smooth_field * magnitude[axis]
        )

    transformed_mask = ndi.map_coordinates(
        transformed_mask,
        displaced_coordinates,
        order=0,
        mode=DEFAULT_PADDING_MODE,
        cval=0,
        prefilter=False,
    )
    return transformed_mask.astype(mask_np.dtype)[None, ...]


def default_elastic_params_from_anomaly_size(anomaly_size):
    """Derive conservative elastic defaults from a channel-first anomaly size."""
    if anomaly_size is None:
        return {}
    spatial_shape = tuple(int(size) for size in anomaly_size)
    if len(spatial_shape) in (3, 4):
        spatial_shape = spatial_shape[1:]
    if len(spatial_shape) not in (2, 3) or any(size <= 0 for size in spatial_shape):
        return {}
    sigma = tuple(max(2, int(round(size * 0.2))) for size in spatial_shape)
    magnitude = tuple(max(1, int(round(size * 0.2))) for size in spatial_shape)
    return {"sigma": sigma, "magnitude": magnitude}


def _as_axis_tuple(value, ndim, name):
    if np.isscalar(value):
        return (float(value),) * ndim
    if len(value) != ndim:
        raise ValueError(
            f"{name} must be scalar or contain exactly {ndim} values, got {value!r}."
        )
    return tuple(float(v) for v in value)
