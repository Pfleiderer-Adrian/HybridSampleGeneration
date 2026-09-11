"""Target-mask generation for conditional and unconditional models."""

import numpy as np
import torch


def target_mask_from_original_mask(original_mask, augment_mask):
    if original_mask is None:
        raise ValueError(
            "original_mask is required for conditional target-mask generation."
        )
    if torch.is_tensor(original_mask):
        device = original_mask.device
        dtype = original_mask.dtype
        augmented = augment_mask(original_mask.detach().cpu().numpy())
        return torch.as_tensor(augmented, device=device, dtype=dtype)
    return augment_mask(np.asarray(original_mask))


def target_mask_from_synthetic_anomaly(
    synthetic_anomaly_image,
    *,
    background_threshold,
):
    if synthetic_anomaly_image is None:
        raise ValueError(
            "synth_anomaly_image is required for threshold target-mask generation."
        )

    threshold_rel = 0.0 if background_threshold is None else float(background_threshold)
    if threshold_rel < 0.0:
        raise ValueError(
            f"background_threshold must be >= 0, got {background_threshold}."
        )

    if torch.is_tensor(synthetic_anomaly_image):
        threshold_source = synthetic_anomaly_image
        if not torch.is_floating_point(threshold_source):
            threshold_source = threshold_source.to(torch.float32)
        finite_values = threshold_source[torch.isfinite(threshold_source)]
        if finite_values.numel() == 0:
            return torch.zeros_like(
                torch.amax(threshold_source, dim=0), dtype=torch.uint8
            )
        min_val = torch.min(finite_values)
        max_val = torch.max(finite_values)
        threshold = min_val + threshold_rel * (max_val - min_val)
        projection = torch.amax(threshold_source, dim=0)
        return (projection > threshold).to(torch.uint8)

    min_val = float(np.nanmin(synthetic_anomaly_image))
    max_val = float(np.nanmax(synthetic_anomaly_image))
    threshold = min_val + threshold_rel * (max_val - min_val)
    projection = np.max(synthetic_anomaly_image, axis=0)
    return (projection > threshold).astype(np.uint8)


__all__ = [
    "target_mask_from_original_mask",
    "target_mask_from_synthetic_anomaly",
]
