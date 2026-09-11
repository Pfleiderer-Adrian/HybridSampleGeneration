"""Two-dimensional anomaly extraction."""

import numpy as np
from scipy.ndimage import find_objects, label

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.extraction.normalization import (
    add_background_noise_floor,
    normalize_anomaly,
)
from hybrid_sample_generator.imaging.resampling import resize_and_pad_2d, spatial_target_size
from hybrid_sample_generator.imaging.roi import crop_square_clip, dynamic_roi_size


def crop_and_center_anomaly_2d(
    img,
    seg,
    config: ExtractionConfiguration,
):
    """
    Extract connected anomaly regions from a 2D segmentation mask and return:
      - normalized-size anomaly cutouts (C, tH, tW) via resize+pad
      - ROI cutouts around the anomaly centroid (variable size)
      - Segmentation (multiclass) crops around anomaly centroid, shape (C, tH, tW)

    Pipeline:
      1) Collapse seg across channel axis -> binary 2D mask (H,W)
      2) Connected-component labeling -> individual anomaly regions (if separated_anomaly=True in config)
      3) For each region above min_region_pixels:
         - crop the region from img
         - compute centroid (center of mass)
         - resize+pad the region crop to target_size
         - compute ROI crop around centroid (region size + margin)
         - store meta_data (label, scale_factor, centroid, original shape)

    Inputs
    ------
    img:
        np.ndarray (C, H, W)
    seg:
        np.ndarray (C, H, W)  (segmentation / anomaly mask)
    config:
        ExtractionConfiguration containing target size, normalization and ROI rules.
    min_region_pixels:
        Minimum number of pixels for a connected component to be kept.
        If <=0, defaults to 5% of the target crop area.

    Returns
    -------
    anomalies:
        list[tuple[np.ndarray, dict]]
        Each item: (padded_arr, meta_data)
          - padded_arr: np.ndarray of shape (C, tH, tW)
          - meta_data: dict with keys:
              - "label": float, max label value in seg (rounded)
              - "scale_factor": tuple[float,float], (H,W) resize factor
              - "centroid_voxel": tuple[int,int], centroid in pixel coords (h,w)
              - "centroid_norm": tuple[float,float], centroid normalized by (H,W)
              - "shape": tuple[int,int,int], original image shape
              - "norm_type": str or None ("zscore" | "zscore_median" | None)
              - "norm_mean": float (zscore only)
              - "norm_std": float (zscore only)
              - "norm_median": float (zscore_median only)
              - "norm_mad": float (zscore_median only)
    anomalies_roi:
        list[np.ndarray]
        ROI crops around anomaly centroid, shape (C, h', w') (variable).
    org_masks:
        list[np.ndarray]
        Segmentation crops around anomaly centroid, shape (C, tH, tW).
    """
    config.validate()
    target_size = spatial_target_size(config.anomaly_size, 2)

    if seg is None or np.all(seg == 0):
        return None, None, None

    if img.ndim != 3:
        raise ValueError(f"img must be 3D (C,H,W). Got {img.shape}")
    if seg.ndim != 3:
        raise ValueError(f"seg must be 3D (C,H,W). Got {seg.shape}")
    if seg.shape[1:] != img.shape[1:] or seg.shape[0] not in (1, img.shape[0]):
        raise ValueError(
            "img and seg must share their spatial shape and seg must have one or "
            f"{img.shape[0]} channels. Got img={img.shape}, seg={seg.shape}"
        )

    C, H, W = img.shape
    shape = img.shape

    binary2d = np.any(seg > 0, axis=0).astype(np.uint8)  # (H,W)

    if config.separate_components:
        labeled, num = label(binary2d)
        regions = [r for r in find_objects(labeled) if r is not None]
    else:
        # whole mask as one region
        labeled = binary2d
        
        h_indices, w_indices = np.where(binary2d > 0)
        hsl = slice(int(np.min(h_indices)), int(np.max(h_indices)) + 1)
        wsl = slice(int(np.min(w_indices)), int(np.max(w_indices)) + 1)
        regions = [(hsl, wsl)]

    anomalies = []
    anomalies_roi = []
    org_masks = []
    roi_masks = []

    min_region_pixels = int(config.min_coverage_ratio * (target_size[0] * target_size[1]))

    for ridx, region in enumerate(regions, start=1):

        hsl, wsl = region
        region_mask = (labeled[region] == ridx)
        pixels = int(region_mask.sum())

        if pixels < min_region_pixels:
            print(f"Anomaly region {ridx} omitted! pixels={pixels} < {min_region_pixels}")
            continue

        result = img[:, hsl, wsl]  # (C,h,w)
        result = np.where(region_mask, result, np.min(img))

        if config.add_background_noise:
            result = add_background_noise_floor(result)

        #ch, cw = center_of_mass(binary2d, labeled, ridx)
        ch = (hsl.start + hsl.stop - 1) / 2
        cw = (wsl.start + wsl.stop - 1) / 2
        #ch, cw = hsl[0]+((hsl[1]-hsl[0])/2), wsl[0]+((wsl[1]-wsl[0])/2)
        centroid_voxel = (ch, cw)
        centroid_norm = (ch / (H - 1), cw / (W - 1))
        # centroid_norm = (centroid_voxel[0] / H, centroid_voxel[1] / W)


        padded_arr, scale_factor = resize_and_pad_2d(
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
                result.shape[-2:],
                config.roi.min_padding,
                config.roi.padding_ratio,
                config.roi.min_size,
            )
        else:
            size_spatial = config.roi.fixed_size
        
        anomalies_roi.append(crop_square_clip(img, centroid_voxel, size_spatial, centroid_is_normalized=False))
        roi_masks.append(crop_square_clip(seg, centroid_voxel, size_spatial, centroid_is_normalized=False))

        anomalies.append((padded_arr, meta_data))

        # cutout like in img
        m_result = seg[:, hsl, wsl]
        m_result = np.where(region_mask, m_result, 0)

        # order=0 for nearest neighbor
        padded_mask, _ = resize_and_pad_2d(
            m_result,
            target_size=target_size,
            order=0,
        )
        org_masks.append(padded_mask)

    return anomalies, anomalies_roi, org_masks, roi_masks
