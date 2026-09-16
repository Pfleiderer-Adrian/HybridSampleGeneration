"""Dimension-independent anomaly extraction."""

import numpy as np
from scipy.ndimage import find_objects, label

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.extraction.normalization import (
    add_background_noise_floor,
    normalize_anomaly,
)
from hybrid_sample_generator.imaging.resampling import resize_and_pad, spatial_target_size
from hybrid_sample_generator.imaging.roi import crop_spatial_clip, dynamic_roi_size


def crop_and_center_anomalies(
    image: np.ndarray,
    segmentation: np.ndarray | None,
    config: ExtractionConfiguration,
):
    """Extract and normalize connected anomalies from a 2D image or 3D volume."""
    config.validate()
    spatial_dims = image.ndim - 1
    if spatial_dims not in (2, 3):
        raise ValueError(
            "image must have shape (C,H,W) or (C,D,H,W). "
            f"Got {image.shape}"
        )
    if segmentation is None or np.all(segmentation == 0):
        return None, None, None, None
    if segmentation.ndim != image.ndim:
        raise ValueError(
            f"segmentation must have {image.ndim} dimensions. Got {segmentation.shape}"
        )
    if (
        segmentation.shape[1:] != image.shape[1:]
        or segmentation.shape[0] not in (1, image.shape[0])
    ):
        raise ValueError(
            "image and segmentation must share their spatial shape and segmentation "
            f"must have one or {image.shape[0]} channels. Got image={image.shape}, "
            f"segmentation={segmentation.shape}"
        )

    target_size = spatial_target_size(config.anomaly_size, spatial_dims)
    binary_mask = np.any(segmentation > 0, axis=0).astype(np.uint8)
    if config.separate_components:
        labeled, _ = label(binary_mask)
        regions = [region for region in find_objects(labeled) if region is not None]
    else:
        labeled = binary_mask
        indices = np.where(binary_mask > 0)
        regions = [
            tuple(
                slice(int(axis.min()), int(axis.max()) + 1)
                for axis in indices
            )
        ]

    anomalies = []
    anomaly_rois = []
    masks = []
    roi_masks = []
    minimum_size = int(config.min_coverage_ratio * np.prod(target_size))

    for component_index, region in enumerate(regions, start=1):
        region_mask = labeled[region] == component_index
        region_size = int(region_mask.sum())
        if region_size < minimum_size:
            print(
                f"Anomaly region {component_index} omitted! "
                f"size={region_size} < {minimum_size}"
            )
            continue

        cropped = image[(slice(None), *region)]
        cropped = np.where(region_mask, cropped, np.min(image))
        if config.add_background_noise:
            cropped = add_background_noise_floor(cropped)

        centroid = tuple(
            int(round((axis.start + axis.stop - 1) / 2))
            for axis in region
        )
        centroid_normalized = tuple(
            coordinate / axis_size
            for coordinate, axis_size in zip(centroid, image.shape[1:])
        )
        normalized, scale_factor = resize_and_pad(
            cropped,
            target_size,
            order=1,
            foreground_mask=region_mask,
        )
        normalized, normalization_metadata = normalize_anomaly(
            normalized,
            normalization=config.normalization,
            eps=float(config.normalization_eps),
        )
        metadata = {
            "label": float(np.max(segmentation).round(0)),
            "scale_factor": tuple(round(float(value), 4) for value in scale_factor),
            "centroid_voxel": centroid,
            "centroid_norm": centroid_normalized,
            "shape": image.shape,
            **normalization_metadata,
        }

        roi_size = (
            dynamic_roi_size(
                cropped.shape[1:],
                config.roi.min_padding,
                config.roi.padding_ratio,
                config.roi.min_size,
            )
            if config.roi.fixed_size is None
            else config.roi.fixed_size
        )
        anomaly_rois.append(
            crop_spatial_clip(
                image, centroid, roi_size, centroid_is_normalized=False
            )
        )
        roi_masks.append(
            crop_spatial_clip(
                segmentation, centroid, roi_size, centroid_is_normalized=False
            )
        )
        anomalies.append((normalized, metadata))

        cropped_mask = segmentation[(slice(None), *region)]
        cropped_mask = np.where(region_mask, cropped_mask, 0)
        normalized_mask, _ = resize_and_pad(
            cropped_mask,
            target_size,
            order=0,
        )
        masks.append(normalized_mask)

    return anomalies, anomaly_rois, masks, roi_masks
