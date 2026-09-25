"""Backend-independent spatial preparation and finalization for fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from hybrid_sample_generator.fusion.background import (
    control_background_mask,
    keep_control_background_after_fusion,
)
from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.fusion.preprocessing import (
    denormalize_anomaly,
    inverse_extraction_scale,
    spatial_label_mask,
    validate_position,
)
from hybrid_sample_generator.imaging.masks.interpolation import interpolate_masked_regions


@dataclass(slots=True)
class PreparedSpatialFusion:
    """Validated and placed arrays shared by deterministic fusion backends."""

    control: np.ndarray
    anomaly: np.ndarray
    target_mask: np.ndarray
    background_slice: np.ndarray
    valid_mask: np.ndarray
    control_background_mask: np.ndarray | None
    output_slices: tuple[slice, ...]
    crop_to_background: tuple[slice, ...]
    crop_shape: tuple[int, ...]
    offset: np.ndarray
    control_spatial_shape: tuple[int, ...]
    channels: int
    spatial_ndim: int


def prepare_spatial_fusion(
    control,
    anomaly,
    anomaly_meta,
    position,
    target_mask,
    *,
    params,
    spatial_ndim: int,
    backend_name: str,
) -> PreparedSpatialFusion:
    """Validate, denormalize, resize, and place one anomaly."""
    if anomaly_meta is None:
        raise ValueError("anomaly_meta must be provided (needs at least 'scale_factor').")
    scale_factor = anomaly_meta.get("scale_factor")
    if scale_factor is None:
        raise ValueError("anomaly_meta is missing required key 'scale_factor'.")
    if target_mask is None:
        raise ValueError(
            f"{backend_name} requires target_mask. Create or load tgt_mask before fusion."
        )

    ctrl = np.asarray(control).astype(np.float32, copy=False)
    anom = np.asarray(anomaly).astype(np.float32, copy=False)
    anom = denormalize_anomaly(anom, anomaly_meta)
    target_mask = np.asarray(target_mask)
    expected_ndim = spatial_ndim + 1
    if ctrl.ndim != expected_ndim or anom.ndim != expected_ndim:
        expected = "(C,H,W)" if spatial_ndim == 2 else "(C,D,H,W)"
        raise ValueError(
            f"Both images must be {expected}. Got ctrl={ctrl.shape}, anomaly={anom.shape}"
        )
    if target_mask.ndim not in (spatial_ndim, expected_ndim):
        raise ValueError(
            f"target_mask must have {spatial_ndim} or {expected_ndim} dims. "
            f"Got {target_mask.shape}"
        )
    if target_mask.shape[-spatial_ndim:] != anom.shape[-spatial_ndim:]:
        raise ValueError(
            f"target mask spatial shape {target_mask.shape[-spatial_ndim:]} "
            f"does not match anomaly {anom.shape[-spatial_ndim:]}"
        )

    channels = int(ctrl.shape[0])
    ctrl_spatial = np.asarray(ctrl.shape[1:], dtype=int)
    foreground = spatial_label_mask(target_mask, spatial_ndim) > 0
    background_values = anom[:, ~foreground]
    finite_background = background_values[np.isfinite(background_values)]
    bg_min = (
        float(np.min(finite_background))
        if finite_background.size
        else float(np.nanmin(anom))
    )
    anom = np.where(foreground[None, ...], anom, bg_min)

    if np.any(foreground):
        coordinates = np.where(foreground)
        crop_slices = tuple(
            slice(axis.min(), axis.max() + 1) for axis in coordinates
        )
        anom = anom[(slice(None), *crop_slices)]
        if target_mask.ndim == expected_ndim:
            target_mask = target_mask[(slice(None), *crop_slices)]
        else:
            target_mask = target_mask[crop_slices]

    scale = inverse_extraction_scale(scale_factor, ndim=spatial_ndim)
    spatial_target_mask = spatial_label_mask(target_mask, spatial_ndim)
    anom = interpolate_masked_regions(
        anom,
        spatial_target_mask > 0,
        warp=lambda spatial: zoom(spatial, scale, order=1),
        nearest_warp=lambda spatial: zoom(spatial, scale, order=0),
        interpolate_background=False,
        background_fill=bg_min,
    )
    target_mask = zoom(spatial_target_mask, scale, order=0).astype(
        np.uint8, copy=False
    )

    position = validate_position(position, spatial_ndim)
    anomaly_spatial = np.asarray(anom.shape[1:], dtype=int)
    offset = np.asarray(
        [
            round(ctrl_spatial[axis] * position[axis] - anomaly_spatial[axis] / 2)
            for axis in range(spatial_ndim)
        ],
        dtype=int,
    )
    offset_end = offset + anomaly_spatial
    for axis, (start, end, limit) in enumerate(
        zip(offset, offset_end, ctrl_spatial)
    ):
        if end > limit:
            shift = end - limit
            offset[axis] -= shift
            offset_end[axis] -= shift
        if offset[axis] < 0:
            shift = -offset[axis]
            offset[axis] += shift
            offset_end[axis] += shift

    insert_slices = tuple(
        slice(int(start), int(end)) for start, end in zip(offset, offset_end)
    )
    background_slice = ctrl[(slice(None), *insert_slices)]
    crop_shape = tuple(int(value) for value in background_slice.shape[1:])
    crop_to_background = tuple(slice(0, size) for size in crop_shape)

    detected_background = None
    if params.fusion_keep_bg:
        detected_background = control_background_mask(
            ctrl,
            params.fusion_bg_value,
            params.fusion_relative_bg_threshold,
            params.fusion_bg_exterior_only,
        )
        background_mask = detected_background[insert_slices]
        target_mask = target_mask.copy()
        target_mask[crop_to_background] = np.where(
            background_mask, 0, target_mask[crop_to_background]
        )

    output_slices = tuple(
        slice(int(offset[axis]), int(offset[axis] + crop_shape[axis]))
        for axis in range(spatial_ndim)
    )
    return PreparedSpatialFusion(
        control=ctrl,
        anomaly=anom,
        target_mask=target_mask,
        background_slice=background_slice,
        valid_mask=target_mask > 0,
        control_background_mask=detected_background,
        output_slices=output_slices,
        crop_to_background=crop_to_background,
        crop_shape=crop_shape,
        offset=offset,
        control_spatial_shape=tuple(int(value) for value in ctrl_spatial),
        channels=channels,
        spatial_ndim=spatial_ndim,
    )


def finalize_spatial_fusion(
    prepared: PreparedSpatialFusion,
    fused_region: np.ndarray,
    extraction_config,
    *,
    crop_roi,
    dynamic_roi_size,
    metrics: dict[str, Any] | None = None,
) -> FusionOutput:
    """Write a fused region and construct segmentation and ROI outputs."""
    if fused_region.shape != prepared.background_slice.shape:
        raise ValueError(
            f"fused region shape {fused_region.shape} does not match target region "
            f"{prepared.background_slice.shape}."
        )

    fused_image = prepared.control.copy()
    fused_image[(slice(None), *prepared.output_slices)] = fused_region
    segmentation = np.zeros(prepared.control_spatial_shape, dtype=np.uint8)
    segmentation[prepared.output_slices] = prepared.target_mask[
        prepared.crop_to_background
    ].astype(np.uint8, copy=False)
    segmentation = segmentation[None, ...]
    if prepared.channels != 1:
        segmentation = np.repeat(segmentation, prepared.channels, axis=0)

    if prepared.control_background_mask is not None:
        fused_image, segmentation = keep_control_background_after_fusion(
            fused_image,
            segmentation,
            prepared.control,
            prepared.control_background_mask,
        )
    if np.sum(segmentation) == 0:
        return FusionOutput(
            image=prepared.control,
            segmentation=segmentation,
            metrics=metrics,
        )

    centroid = tuple(
        float(prepared.offset[axis]) + float(prepared.crop_shape[axis]) / 2.0
        for axis in range(prepared.spatial_ndim)
    )
    roi_config = extraction_config.roi
    if roi_config.fixed_size is None:
        roi_size = dynamic_roi_size(
            prepared.crop_shape,
            roi_config.min_padding,
            roi_config.padding_ratio,
            roi_config.min_size,
        )
    else:
        roi_size = roi_config.fixed_size

    return FusionOutput(
        image=fused_image,
        segmentation=segmentation,
        roi=crop_roi(
            fused_image, centroid, roi_size, centroid_is_normalized=False
        ),
        roi_mask=crop_roi(
            segmentation, centroid, roi_size, centroid_is_normalized=False
        ),
        metrics=metrics,
    )


__all__ = [
    "PreparedSpatialFusion",
    "finalize_spatial_fusion",
    "prepare_spatial_fusion",
]
