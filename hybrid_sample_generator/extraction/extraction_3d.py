"""Three-dimensional anomaly extraction."""

import numpy as np
from scipy.ndimage import find_objects, label

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.extraction.normalization import (
    add_background_noise_floor,
    normalize_anomaly,
)
from hybrid_sample_generator.imaging.resampling import resize_and_pad_3d, spatial_target_size
from hybrid_sample_generator.imaging.roi import crop_cube_clip, dynamic_roi_size


def crop_and_center_anomaly_3d(
    img,
    seg,
    config: ExtractionConfiguration,
):
    """
    Extract connected anomaly regions from a 3D segmentation mask and return:
      - normalized-size anomaly cutouts (C, tD, tH, tW) via resize+pad
      - ROI cutouts around the anomaly centroid (variable size)

    Pipeline:
      1) Collapse seg across channel axis -> binary 3D mask (D,H,W)
      2) Connected-component labeling -> individual anomaly regions (if separated_anomaly=True in config)
      3) For each region above min_region_voxels:
         - crop the region from img
         - compute centroid (center of mass)
         - resize+pad the region crop to target_size
         - compute ROI crop around centroid (region size + margin)
         - store meta_data (label, scale_factor, centroid, original shape)

    Inputs
    ------
    img:
        np.ndarray with shape (C, D, H, W).
    seg:
        np.ndarray with shape (C, D, H, W).
        Convention:
          - 0 = background
          - >0 = anomaly (any positive value is treated as anomaly)
    config:
        ExtractionConfiguration containing target size, normalization and ROI rules.
    min_region_voxels:
        Minimum voxel count for a connected component to be kept.
        If <= 0, defaults to 5% of target volume (0.05 * tD * tH * tW).

    Outputs
    -------
    anomalies:
        list[tuple[np.ndarray, dict]]
        Each item: (padded_arr, meta_data)
          - padded_arr: np.ndarray of shape (C, tD, tH, tW)
          - meta_data: dict with keys:
              - "label": float, max label value in seg (rounded)
              - "scale_factor": tuple[float,float,float], (D,H,W) resize factor
              - "centroid_voxel": tuple[int,int,int], centroid in voxel coords (d,h,w)
              - "centroid_norm": tuple[float,float,float], centroid normalized by (D,H,W)
              - "shape": tuple[int,int,int,int], original image shape
              - "norm_type": str or None ("zscore" | "zscore_median" | None)
              - "norm_mean": float (zscore only)
              - "norm_std": float (zscore only)
              - "norm_median": float (zscore_median only)
              - "norm_mad": float (zscore_median only)
    anomalies_roi:
        list[np.ndarray]
        ROI crops around anomaly centroid, shape (C, d', h', w') (variable).
    org_masks:
        list[np.ndarray]
        Segmentation crops around anomaly centroid, shape (C, tD, tH, tW).

    Notes
    -----
    - If seg is None or completely empty, the function returns None in the original code.

    Raises
    ------
    ValueError:
        If img/seg are not 4D or shapes do not match.
    """
    config.validate()
    target_size = spatial_target_size(config.anomaly_size, 3)
    if seg is None or np.all(seg == 0):
        return None, None, None

    if img.ndim != 4:
        raise ValueError(f"img must be (C,D,H,W). Got {img.shape}")
    if seg.ndim != 4:
        raise ValueError(f"seg must be (C,D,H,W). Got {seg.shape}")
    if seg.shape[1:] != img.shape[1:] or seg.shape[0] not in (1, img.shape[0]):
        raise ValueError(
            "img and seg must share their spatial shape and seg must have one or "
            f"{img.shape[0]} channels. Got img={img.shape}, seg={seg.shape}"
        )

    C, D, H, W = img.shape
    shape = img.shape

    binary3d = np.any(seg > 0, axis=0).astype(np.uint8)  # (D,H,W)

    if config.separate_components:
        labeled, num = label(binary3d)
        regions = [r for r in find_objects(labeled) if r is not None]
    else:
        # whole mask as one region
        labeled = binary3d
        
        d_indices, h_indices, w_indices = np.where(binary3d > 0)
        dsl = slice(int(np.min(d_indices)), int(np.max(d_indices)) + 1)
        hsl = slice(int(np.min(h_indices)), int(np.max(h_indices)) + 1)
        wsl = slice(int(np.min(w_indices)), int(np.max(w_indices)) + 1)
        regions = [(dsl, hsl, wsl)]

    anomalies = []
    anomalies_roi = []
    org_masks = []
    roi_masks = []

    min_region_voxels = int(config.min_coverage_ratio * (target_size[0] * target_size[1] * target_size[2]))

    for ridx, region in enumerate(regions, start=1):
        dsl, hsl, wsl = region
        region_mask = (labeled[region] == ridx)
        voxels = int(region_mask.sum())

        if voxels < min_region_voxels:
            print(f"Anomaly region {ridx} omitted! voxels={voxels} < {min_region_voxels}")
            continue

        result = img[:, dsl, hsl, wsl]  # (C,d,h,w)
        result = np.where(region_mask, result, np.min(img))

        if config.add_background_noise:
            result = add_background_noise_floor(result)

        # geometric middle like in 2D
        cd = (dsl.start + dsl.stop - 1) / 2
        ch = (hsl.start + hsl.stop - 1) / 2
        cw = (wsl.start + wsl.stop - 1) / 2
        
        centroid_voxel = (int(round(cd)), int(round(ch)), int(round(cw)))
        centroid_norm = (centroid_voxel[0] / D, centroid_voxel[1] / H, centroid_voxel[2] / W)

        padded_arr, scale_factor = resize_and_pad_3d(
            result,
            target_size=target_size,
            order=1,
            foreground_mask=region_mask,
        )
        padded_arr, norm_meta = normalize_anomaly(
            padded_arr, normalization=config.normalization, eps=float(config.normalization_eps)
        )
        scale_factor = tuple(round(float(ele), 4) for ele in scale_factor)

        label_tmp = float(np.max(seg).round(0))

        meta_data = {
            "label": label_tmp,
            "scale_factor": scale_factor,
            "centroid_voxel": centroid_voxel,
            "centroid_norm": centroid_norm,
            "shape": shape
        }
        meta_data.update(norm_meta)
        
        if config.roi.fixed_size is None:
            size_spatial = dynamic_roi_size(
                result.shape[-3:],
                config.roi.min_padding,
                config.roi.padding_ratio,
                config.roi.min_size,
            )
        else:
            size_spatial = config.roi.fixed_size

        anomalies_roi.append(crop_cube_clip(img, centroid_voxel, size_spatial, centroid_is_normalized=False))
        roi_masks.append(crop_cube_clip(seg, centroid_voxel, size_spatial, centroid_is_normalized=False))

        anomalies.append((padded_arr, meta_data))

        # cutout like in img
        m_result = seg[:, dsl, hsl, wsl]
        m_result = np.where(region_mask, m_result, 0)

        # order=0 for nearest neighbor
        padded_mask, _ = resize_and_pad_3d(
            m_result,
            target_size=target_size,
            order=0,
        )
        org_masks.append(padded_mask)

    return anomalies, anomalies_roi, org_masks, roi_masks
