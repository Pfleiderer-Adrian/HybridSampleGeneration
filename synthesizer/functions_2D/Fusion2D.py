import numpy as np
import scipy.ndimage
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes


def _denormalize_anomaly(anom, normalization_meta):
    if not normalization_meta:
        return anom

    norm_type = normalization_meta.get("norm_type")
    if norm_type == "zscore":
        mean = normalization_meta.get("norm_mean")
        std = normalization_meta.get("norm_std")
        if mean is None or std is None:
            return anom
        return anom * float(std) + float(mean)
    if norm_type == "zscore_median":
        median = normalization_meta.get("norm_median")
        mad = normalization_meta.get("norm_mad")
        if median is None or mad is None:
            return anom
        return anom * float(mad) + float(median)

    return anom


def fusion2d(
    control_image,
    synth_anomaly_image,
    anomaly_meta,
    position_factor,
    fusion_mask_params,
    background_threshold=None,
):
    """
    Fuse a synthetic anomaly into a control image (2D) using alpha blending.

    High-level steps (2D adaptation of fusion3d):
      1) Remove anomaly background via thresholding.
      2) Trim anomaly padding (crop to foreground bounding box).
      3) Zoom anomaly (spatial only) by `scale_factor`.
      4) Compute the insertion box in control coords using `position_factor`.
      5) Normalize anomaly intensities locally to match control region (per channel).
      6) Create an alpha mask (edge-aware distance-transform mask).
      7) Alpha-blend anomaly into the control region.
      8) Create a binary segmentation mask for the inserted anomaly.

    Inputs
    ------
    control_image:
        np.ndarray, shape (C, H, W)
        The target image into which the anomaly is inserted.
    synth_anomaly_image:
        np.ndarray, shape (C, H, W)
        The synthetic anomaly cutout/image.
    anomaly_meta:
        dict with metadata for this anomaly (typically from config.syn_anomaly_transformations[anomaly_basename]).
        Must contain:
          - "scale_factor": scalar or tuple/list of length 2
        May contain normalization keys:
          - "norm_type" and corresponding parameters (see _denormalize_anomaly).
    position_factor:
        Iterable length 2 (H,W), typically in [0,1]:
        Example: (0.5, 0.5) inserts anomaly centered in the middle of the image.
    fusion_mask_params:
        dict containing parameters for alpha mask creation (see get_alpha_mask_sobel_final_2d):
          - "max_alpha"
          - "sq"
          - "steepness_factor"
          - "upsampling_factor"
          - "sobel_threshold"
          - "dilation_size"
          - "shave_pixels"
    background_threshold:
        float or None.
        If None: defaults to min(anomaly) + 0.001.
        Used to decide what is foreground vs background in the anomaly.
    Outputs
    -------
    fused_image:
        np.ndarray, shape (C, H, W)
        The control image with the anomaly blended in.
    segmentation:
        np.ndarray, shape (C, H, W), dtype uint8
        Binary mask where 1 indicates the inserted anomaly region.

    Raises
    ------
    ValueError:
        If inputs are not 3D (C,H,W).
    """
    # Ensure float32 for arithmetic stability (no copy if already float32)
    ctrl = control_image.astype(np.float32, copy=False)
    if anomaly_meta is None:
        raise ValueError("anomaly_meta must be provided (needs at least 'scale_factor').")

    scale_factor = anomaly_meta.get("scale_factor")
    if scale_factor is None:
        raise ValueError("anomaly_meta is missing required key 'scale_factor'.")

    anom = synth_anomaly_image.astype(np.float32, copy=False)
    anom = _denormalize_anomaly(anom, anomaly_meta)

    # Validate dimensionality (expects channel-first 2D images)
    if ctrl.ndim != 3 or anom.ndim != 3:
        raise ValueError(f"Both images must be 2D (C,H,W). Got ctrl={ctrl.shape}, anom={anom.shape}")

    # Unpack control shape
    C, H, W = ctrl.shape

    # ------------------------------------------------------------
    # 1) Choose background threshold if not provided
    # ------------------------------------------------------------
    if background_threshold is None:
        background_threshold = float(np.nanmin(anom)) + 0.001

    # ------------------------------------------------------------
    # 2) Remove anomaly background by pushing values below threshold to global min
    #    This increases contrast between foreground and background and stabilizes mask creation.
    # ------------------------------------------------------------
    anom_min = float(np.nanmin(anom))
    anom = np.where(anom > background_threshold, anom, anom_min)

    # ------------------------------------------------------------
    # 3) Trim spatial padding by cropping to the foreground bounding box (same crop for all channels)
    # ------------------------------------------------------------
    # Foreground mask over spatial dims: a pixel is foreground if ANY channel is above threshold
    fg = np.any(anom > background_threshold, axis=0)  # (H,W)

    # Only crop if there is any foreground
    if np.any(fg):
        yy, xx = np.where(fg)
        y0, y1 = yy.min(), yy.max() + 1
        x0, x1 = xx.min(), xx.max() + 1
        anom = anom[:, y0:y1, x0:x1]
    # If no foreground exists, `anom` remains unchanged. (You may optionally early-return.)

    # ------------------------------------------------------------
    # 4) Zoom anomaly spatially (keep channel axis unchanged)
    # ------------------------------------------------------------
    if np.isscalar(scale_factor):
        sf = (float(scale_factor),) * 2
    else:
        sf = tuple(float(v) for v in scale_factor)  # (scale_H, scale_W)

    # Zoom factors: (C unchanged, spatial scaled)
    anom = zoom(anom, (1.0, sf[0], sf[1]), order=1)

    # ------------------------------------------------------------
    # 5) Compute insertion offset from normalized position_factor
    # ------------------------------------------------------------
    ctrl_spatial = np.array([H, W], dtype=int)            # control spatial dims
    anom_spatial = np.array(anom.shape[1:], dtype=int)    # anomaly spatial dims after zoom

    # Offset is chosen so anomaly center is placed at position_factor * control_size
    offset = np.array(
        [
            round(ctrl_spatial[0] * position_factor[0] - anom_spatial[0] / 2),
            round(ctrl_spatial[1] * position_factor[1] - anom_spatial[1] / 2),
        ],
        dtype=int,
    )
    offset_end = offset + anom_spatial

    # ------------------------------------------------------------
    # 6) Clamp insertion box to control bounds
    # ------------------------------------------------------------
    for i, (o, e, s) in enumerate(zip(offset, offset_end, ctrl_spatial)):
        # If end exceeds bounds, shift box up/left
        if e > s:
            shift = e - s
            offset[i] -= shift
            offset_end[i] -= shift
        # If start is negative, shift box down/right
        if offset[i] < 0:
            shift = -offset[i]
            offset[i] += shift
            offset_end[i] += shift

    h0, w0 = offset
    h1, w1 = offset_end

    # ------------------------------------------------------------
    # 7) Compute per-channel min/max of the control region where we will fuse
    #    Used for local intensity normalization of the anomaly.
    # ------------------------------------------------------------
    bg_slice = ctrl[:, h0:h1, w0:w1]                 # (C, h, w)
    img_region_max = np.nanmax(bg_slice, axis=(1, 2))  # (C,)
    img_region_min = np.nanmin(bg_slice, axis=(1, 2))  # (C,)

    # ------------------------------------------------------------
    # 8) Create a spatial valid mask for anomaly foreground
    # ------------------------------------------------------------
    # Use max projection over channels to define foreground robustly
    anom_proj = np.max(anom, axis=0)                         # (h,w)
    valid_mask = anom_proj > background_threshold            # (h,w)

    # ------------------------------------------------------------
    # 9) Locally normalize anomaly values (per channel) to match control region stats
    #    - Only normalize values where valid_mask is True
    #    - Match anomaly's foreground range to control region's per-channel range
    # ------------------------------------------------------------
    if np.any(valid_mask):
        anom = anom.copy()  # we will modify in place
        for c in range(C):
            vp = anom[c][valid_mask]  # foreground values for this channel
            if vp.size == 0:
                continue

            a_min = float(vp.min())
            a_max = float(vp.max())
            src_range = a_max - a_min
            if src_range == 0:
                src_range = 1.0

            tgt_range = float(img_region_max[c] - img_region_min[c])

            # Map anomaly foreground values into control region intensity range
            vp = ((vp - a_min) / src_range) * tgt_range + float(img_region_min[c])
            anom[c][valid_mask] = vp

    # ------------------------------------------------------------
    # 10) Create alpha mask from Sobel+distance transform mask (edge-aware)
    # ------------------------------------------------------------
    alpha_mask = get_alpha_mask_sobel_final_2d(
        anom_proj, fusion_mask_params, background_threshold
    )  # (h,w)

    # Broadcast alpha from (h,w) to (1,h,w); it will broadcast across channels during blending
    alpha = alpha_mask[None, :, :]  # (1,h,w)

    # ------------------------------------------------------------
    # 11) Crop anomaly + alpha to exactly match the clamped insertion region
    # ------------------------------------------------------------
    anom_crop = anom[:, : (h1 - h0), : (w1 - w0)]
    alpha_crop = alpha[:, : (h1 - h0), : (w1 - w0)]
    bg_slice = ctrl[:, h0:h1, w0:w1]

    # If clamp shortened region, also crop to background slice actual shape
    hh, ww = bg_slice.shape[1:]
    anom_crop = anom_crop[:, :hh, :ww]
    alpha_crop = alpha_crop[:, :hh, :ww]

    # ------------------------------------------------------------
    # 12) Alpha blending
    #    fused = anomaly * alpha + background * (1 - alpha)
    # ------------------------------------------------------------
    fused_region = anom_crop * alpha_crop + bg_slice * (1.0 - alpha_crop)

    # Write fused region back into a copy of the control image
    fused_image = ctrl.copy()
    fused_image[:, h0:h1, w0:w1] = fused_region

    # ------------------------------------------------------------
    # 13) Create segmentation mask in control coordinates
    # ------------------------------------------------------------
    segmentation = np.zeros((H, W), dtype=np.uint8)

    # valid_mask is anomaly-local (h,w) -> crop to actual inserted region size
    vm_crop = valid_mask[:hh, :ww]
    segmentation[h0 : h0 + hh, w0 : w0 + ww] = vm_crop.astype(np.uint8)

    # Fill holes for a cleaner binary segmentation
    segmentation = binary_fill_holes(segmentation).astype(np.uint8)

    segmentation = segmentation[None, ...]                 # (1, H, W)
    if C != 1:
        segmentation = np.repeat(segmentation, C, axis=0)  # (C, H, W)

    return fused_image, segmentation


def get_alpha_mask_sobel_final_2d(anomaly_arr, config, background_threshold):
    """
    Build an alpha blending mask for a 2D anomaly image using:
      - Sobel edges to detect boundaries
      - morphological operations to close gaps and remove noise
      - distance transform to produce a smooth mask interior
      - non-linear shaping to control blending strength and falloff

    Inputs
    ------
    anomaly_arr:
        np.ndarray, shape (H, W)
        A single-channel representation of the anomaly (e.g. max projection across channels).
    config:
        dict with keys:
          - "dilation_size": int
          - "sobel_threshold": float
          - "upsampling_factor": int
          - "steepness_factor": float
          - "sq": int/float
          - "max_alpha": float
          - "shave_pixels": int
    background_threshold:
        float threshold to decide foreground vs background.

    Outputs
    -------
    alpha_mask:
        np.ndarray, shape (H, W), dtype float32
        Values are in [0, max_alpha] inside anomaly area, 0 outside.
    """
    # Initialize alpha mask
    alpha_mask = np.zeros_like(anomaly_arr, dtype=np.float32)

    # ------------------------------------------------------------
    # Build structuring elements for morphology operations
    # ------------------------------------------------------------
    dilation_size = config["dilation_size"]

    # Circular 2D structuring element ("round brush") used to close gaps
    y, x = np.ogrid[-dilation_size: dilation_size + 1, -dilation_size: dilation_size + 1]
    struct_brush = x ** 2 + y ** 2 <= dilation_size ** 2

    # 3x3-ish structure used for "shaving" boundary pixels
    struct_shave = scipy.ndimage.generate_binary_structure(2, 2)

    slice_img = anomaly_arr

    # Skip if no foreground content
    if not np.any(slice_img > background_threshold):
        return alpha_mask

    # ------------------------------------------------------------
    # 1) Sobel edge detection
    # ------------------------------------------------------------
    sx = scipy.ndimage.sobel(slice_img, axis=0)
    sy = scipy.ndimage.sobel(slice_img, axis=1)
    grad_mag = np.hypot(sx, sy)

    # Normalize gradient magnitude to [0,1]
    if grad_mag.max() > 0:
        grad_mag /= grad_mag.max()

    # Edge mask = pixels with gradient magnitude above threshold
    edge_mask = grad_mag > config["sobel_threshold"]

    # ------------------------------------------------------------
    # 2) Close gaps and fill the interior
    # ------------------------------------------------------------
    thick_edge = scipy.ndimage.binary_dilation(edge_mask, structure=struct_brush)
    filled_body = scipy.ndimage.binary_fill_holes(thick_edge)

    # Erode once to compensate for dilation thickening (shape reconstruction)
    restored_mask = scipy.ndimage.binary_erosion(filled_body, structure=struct_brush, iterations=1)

    # ------------------------------------------------------------
    # 3) Optional "shaving" to remove boundary pixels (reduce blending artifacts)
    # ------------------------------------------------------------
    shave_pixels = config["shave_pixels"]
    if shave_pixels > 0:
        shaved_mask = scipy.ndimage.binary_erosion(
            restored_mask, structure=struct_shave, iterations=shave_pixels
        )
    else:
        shaved_mask = restored_mask

    final_clean_mask = shaved_mask

    # Skip empty mask
    if not np.any(final_clean_mask):
        return alpha_mask

    # Enforce background threshold after morphology:
    # Remove any pixels that are still background in the original slice.
    bg_mask = slice_img <= background_threshold
    final_clean_mask = final_clean_mask.copy()
    final_clean_mask[bg_mask] = False

    # ------------------------------------------------------------
    # 4) Distance transform to create smooth interior weights
    # ------------------------------------------------------------
    upsampling_factor = config["upsampling_factor"]

    if upsampling_factor > 1:
        # Upsample mask for smoother distance transform, then downsample
        large_mask = scipy.ndimage.zoom(final_clean_mask, upsampling_factor, order=0)
        large_dist = scipy.ndimage.distance_transform_edt(large_mask)
        dist_map = scipy.ndimage.zoom(large_dist, 1 / upsampling_factor, order=1)
    else:
        dist_map = scipy.ndimage.distance_transform_edt(final_clean_mask)

    # Normalize to [0,1]
    curr_max = dist_map.max()
    if curr_max > 0:
        dist_map /= curr_max

    # ------------------------------------------------------------
    # 5) Apply shaping parameters to control falloff and maximum alpha
    # ------------------------------------------------------------
    dist_map *= config["steepness_factor"]
    dist_map = np.clip(dist_map, 0, 1.0)
    dist_map = (dist_map ** config["sq"]) * config["max_alpha"]

    # Enforce zero outside the clean mask
    dist_map[~final_clean_mask] = 0.0

    alpha_mask[:, :] = dist_map.astype(np.float32, copy=False)
    return alpha_mask


def trim_zeros(arr):
    """
    Trim a numpy array to the bounding box of its non-zero values.

    Inputs
    ------
    arr:
        np.ndarray of any dimensionality.

    Outputs
    -------
    np.ndarray:
        A view/copy of the array cropped to the min/max indices where arr != 0.

    Notes
    -----
    - If arr is all zeros, np.nonzero(arr) is empty and this will raise an error.
      Consider guarding with `if not np.any(arr): return arr[:0]` or similar if needed.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))

    return arr[slices]
