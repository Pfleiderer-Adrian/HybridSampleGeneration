"""Training-sample and alpha preprocessing for learned fusion."""

import numpy as np
import scipy.ndimage
import torch
from scipy.ndimage import binary_dilation


def to_tensor(array, device):
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device).unsqueeze(0)


def unpack_sample(sample):
    if isinstance(sample, dict):
        image = sample.get("img", sample.get("image"))
        mask = sample.get("seg", sample.get("mask", sample.get("ori_mask")))
        basename = sample.get("fname", sample.get("basename", "sample"))
        if image is None or mask is None:
            raise ValueError("Fusion backend training samples must contain image and mask data.")
        return image, mask, basename
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1], sample[2] if len(sample) >= 3 else "sample"
    raise ValueError("Expected training sample as dict or tuple/list (img, seg, basename).")


def spatial_label_mask(mask, spatial_dims):
    mask = np.asarray(mask)
    if mask.ndim == spatial_dims:
        return mask
    if mask.ndim == spatial_dims + 1:
        return np.max(mask, axis=0)
    raise ValueError(
        f"mask must have {spatial_dims} or {spatial_dims + 1} dims. Got {mask.shape}."
    )


def bbox_slices(mask, *, margin, shape):
    coordinates = np.where(mask)
    return tuple(
        slice(
            max(0, int(axis.min()) - margin),
            min(int(shape[index]), int(axis.max()) + margin + 1),
        )
        for index, axis in enumerate(coordinates)
    )


def channel_min(image):
    return np.min(image, axis=tuple(range(1, image.ndim)), keepdims=True)


def pseudo_inpaint(target, mask, *, sigma):
    sigma_tuple = (0.0, *([max(float(sigma), 0.1)] * (target.ndim - 1)))
    blurred = scipy.ndimage.gaussian_filter(target, sigma=sigma_tuple)
    support = support_mask(mask.astype(np.float32), 2, mask.ndim) > 0
    return np.where(support[None, ...], blurred, target).astype(np.float32, copy=False)


def soft_alpha(mask, params, spatial_dims):
    mask = mask.astype(np.float32, copy=False)
    sigma = float(params.base_alpha_blur_sigma)
    alpha = scipy.ndimage.gaussian_filter(mask, sigma=sigma) if sigma > 0 else mask
    alpha = np.clip(alpha, 0.0, 1.0)
    maximum = float(np.max(alpha))
    if maximum > 0:
        alpha = alpha / maximum
    alpha = alpha * float(params.base_alpha)
    support = support_mask(mask, params.residual_border_width, spatial_dims)
    return np.where(support > 0, alpha, 0.0).astype(np.float32, copy=False)


def support_mask(mask, border_width: int, spatial_dims):
    support = mask > 0
    if int(border_width) > 0 and np.any(support):
        support = binary_dilation(
            support,
            structure=np.ones((3,) * spatial_dims, dtype=bool),
            iterations=int(border_width),
        )
    return support.astype(np.float32, copy=False)


def safe_scale(array):
    scale = float(np.nanstd(array))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanmax(array) - np.nanmin(array))
    return scale if np.isfinite(scale) and scale >= 1e-6 else 1.0


__all__ = ["bbox_slices", "channel_min", "pseudo_inpaint", "safe_scale", "soft_alpha", "spatial_label_mask", "support_mask", "to_tensor", "unpack_sample"]
