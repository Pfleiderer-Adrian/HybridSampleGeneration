"""Dimension-independent spatial workflow for classical anomaly fusion."""

from __future__ import annotations

import numpy as np

from hybrid_sample_generator.fusion.classical.intensity import match_local_intensity
from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.fusion.spatial import (
    finalize_spatial_fusion,
    prepare_spatial_fusion,
)


def fuse_spatial(
    control,
    anomaly,
    anomaly_meta,
    position,
    target_mask,
    extraction_config,
    *,
    params,
    spatial_ndim: int,
    crop_roi,
    dynamic_roi_size,
    alpha_builder,
    anomaly_roi=None,
    anomaly_roi_mask=None,
) -> FusionOutput:
    """Prepare, alpha-blend, and finalize one 2D or 3D anomaly."""
    prepared = prepare_spatial_fusion(
        control,
        anomaly,
        anomaly_meta,
        position,
        target_mask,
        params=params,
        spatial_ndim=spatial_ndim,
        backend_name="ClassicalFusionBackend",
    )

    anomaly_projection = np.max(prepared.anomaly, axis=0)
    alpha_mask = alpha_builder(
        anomaly_projection,
        params,
        prepared.target_mask,
    )
    matched_anomaly = match_local_intensity(
        prepared.anomaly,
        prepared.control,
        prepared.background_slice,
        prepared.valid_mask,
        prepared.target_mask,
        anomaly_roi,
        anomaly_roi_mask,
        alpha_mask,
        params,
        normalization_eps=extraction_config.normalization_eps,
    )
    anomaly_crop = matched_anomaly[
        (slice(None), *prepared.crop_to_background)
    ]
    alpha_crop = alpha_mask[prepared.crop_to_background][None, ...]
    fused_region = (
        anomaly_crop * alpha_crop
        + prepared.background_slice * (1.0 - alpha_crop)
    )
    return finalize_spatial_fusion(
        prepared,
        fused_region,
        extraction_config,
        crop_roi=crop_roi,
        dynamic_roi_size=dynamic_roi_size,
    )


__all__ = ["fuse_spatial"]
