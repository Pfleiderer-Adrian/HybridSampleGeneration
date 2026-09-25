"""Poisson-specific composition on shared spatial fusion preparation."""

from __future__ import annotations

import numpy as np

from hybrid_sample_generator.fusion.classical.intensity import (
    infer_output_intensity_bounds,
    match_local_intensity,
)
from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.fusion.poisson.solver import solve_poisson
from hybrid_sample_generator.fusion.spatial import (
    finalize_spatial_fusion,
    prepare_spatial_fusion,
)


def fuse_spatial_poisson(
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
    anomaly_roi=None,
    anomaly_roi_mask=None,
    performance_warning=None,
) -> FusionOutput:
    """Prepare, solve, and finalize one 2D or 3D Poisson placement."""
    prepared = prepare_spatial_fusion(
        control,
        anomaly,
        anomaly_meta,
        position,
        target_mask,
        params=params,
        spatial_ndim=spatial_ndim,
        backend_name="PoissonFusionBackend",
    )
    matched_anomaly = match_local_intensity(
        prepared.anomaly,
        prepared.control,
        prepared.background_slice,
        prepared.valid_mask,
        prepared.target_mask,
        anomaly_roi,
        anomaly_roi_mask,
        None,
        params,
        normalization_eps=extraction_config.normalization_eps,
    )
    anomaly_crop = matched_anomaly[
        (slice(None), *prepared.crop_to_background)
    ]
    mask_crop = prepared.target_mask[prepared.crop_to_background] > 0

    if performance_warning is not None:
        performance_warning(
            int(np.count_nonzero(mask_crop)),
            prepared.channels,
        )

    patch_slices = tuple(
        slice(
            max(0, output_slice.start - 1),
            min(
                prepared.control.shape[axis + 1],
                output_slice.stop + 1,
            ),
        )
        for axis, output_slice in enumerate(prepared.output_slices)
    )
    inner_slices = tuple(
        slice(
            output_slice.start - patch_slice.start,
            output_slice.stop - patch_slice.start,
        )
        for output_slice, patch_slice in zip(
            prepared.output_slices,
            patch_slices,
        )
    )
    target_patch = prepared.control[(slice(None), *patch_slices)].copy()
    source_patch = target_patch.copy()
    target_inner = target_patch[(slice(None), *inner_slices)]
    source_patch[(slice(None), *inner_slices)] = np.where(
        mask_crop[None, ...],
        anomaly_crop,
        target_inner,
    )
    patch_mask = np.zeros(target_patch.shape[1:], dtype=bool)
    patch_mask[inner_slices] = mask_crop

    fused_patch, solver_metrics = solve_poisson(
        source_patch,
        target_patch,
        patch_mask,
        guidance_mode=params.guidance_mode,
        rtol=params.solver_rtol,
        atol=params.solver_atol,
        max_iterations=params.solver_max_iterations,
    )
    if params.clip_output:
        bounds = infer_output_intensity_bounds(prepared.control)
        if bounds is not None:
            fused_patch = np.clip(fused_patch, *bounds)
    fused_region = fused_patch[(slice(None), *inner_slices)]
    metrics = {
        "solver": "cg",
        "guidance_mode": params.guidance_mode,
        "unknowns": solver_metrics.unknowns,
        "iterations": solver_metrics.iterations,
        "anchored_components": solver_metrics.anchored_components,
    }
    return finalize_spatial_fusion(
        prepared,
        fused_region,
        extraction_config,
        crop_roi=crop_roi,
        dynamic_roi_size=dynamic_roi_size,
        metrics=metrics,
    )


__all__ = ["fuse_spatial_poisson"]
