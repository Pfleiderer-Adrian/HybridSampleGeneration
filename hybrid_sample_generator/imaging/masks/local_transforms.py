"""Class-local label-mask transformations."""

import numpy as np
import scipy.ndimage as ndi

from hybrid_sample_generator.imaging.masks.elastic import random_elastic_transform
from hybrid_sample_generator.imaging.masks.geometry import (
    rotate_spatial_mask,
    sample_uniform,
    stretch_spatial_mask,
    validate_channel_first_mask,
)


def random_local_stretch_transform(
    mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None
):
    """Stretch selected classes in a 2D or 3D channel-first label mask."""
    validate_channel_first_mask(mask_np)
    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()
    classes, priorities = _classes_and_priorities(
        transformed_mask, classes, priorities
    )
    scales = sample_uniform(
        params.get("min_stretch", 0.95),
        params.get("max_stretch", 1.05),
        rng=rng,
        size=transformed_mask.ndim,
    )
    class_masks = {}
    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask):
            binary_mask = stretch_spatial_mask(binary_mask, scales=scales).astype(bool)
        class_masks[cls] = binary_mask
    return _compose(transformed_mask, class_masks, classes, priorities, mask_np.dtype)


def random_local_rotation_transform(
    mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None
):
    """Rotate selected classes in a 2D or 3D channel-first label mask."""
    validate_channel_first_mask(mask_np)
    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()
    classes, priorities = _classes_and_priorities(
        transformed_mask, classes, priorities
    )
    angle = sample_uniform(max_value=params.get("max_rotation", 5.0), rng=rng)
    class_masks = {}
    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask):
            binary_mask = rotate_spatial_mask(
                binary_mask, angle=angle, center_mask=binary_mask
            ).astype(bool)
        class_masks[cls] = binary_mask
    return _compose(transformed_mask, class_masks, classes, priorities, mask_np.dtype)


def random_local_dilate_transform(
    mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None
):
    """Dilate selected classes in a 2D or 3D channel-first label mask."""
    validate_channel_first_mask(mask_np)
    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()
    classes, priorities = _classes_and_priorities(
        transformed_mask, classes, priorities
    )
    iterations = sample_uniform(
        params.get("min_iterations", 0),
        params.get("max_iterations", 2),
        rng=rng,
        integer=True,
    )
    class_masks = {}
    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask) and iterations > 0:
            binary_mask = ndi.binary_dilation(binary_mask, iterations=iterations)
        class_masks[cls] = binary_mask
    return _compose(transformed_mask, class_masks, classes, priorities, mask_np.dtype)


def random_local_elastic_transform(
    mask_np: np.ndarray, classes=None, priorities=None, params=None, rng=None
):
    """Apply elastic deformation to selected classes in a label mask."""
    validate_channel_first_mask(mask_np)
    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()
    classes, priorities = _classes_and_priorities(
        transformed_mask, classes, priorities
    )
    class_masks = {}
    for cls in classes:
        binary_mask = transformed_mask == cls
        if np.any(binary_mask):
            binary_mask = random_elastic_transform(
                binary_mask[None, ...],
                sigma=params.get("sigma", 30),
                magnitude=params.get("magnitude", 20),
                rng=rng,
            )[0].astype(bool)
        class_masks[cls] = binary_mask
    return _compose(transformed_mask, class_masks, classes, priorities, mask_np.dtype)


def _classes_and_priorities(mask, classes, priorities):
    if classes is None:
        classes = [cls for cls in np.unique(mask) if cls != 0]
    if priorities is None:
        priorities = classes
    return classes, priorities


def _compose(mask, class_masks, classes, priorities, dtype):
    final_mask = mask.copy()
    for cls in classes:
        final_mask[final_mask == cls] = 0
    for cls in reversed(priorities):
        if cls in class_masks:
            final_mask[class_masks[cls]] = cls
    return final_mask.astype(dtype)[None, ...]
