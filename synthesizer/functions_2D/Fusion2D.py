import numpy as np
import scipy.ndimage
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation

from synthesizer.functions_2D.Anomaly_Extraction2D import crop_square_clip


def _inverse_extraction_scale(scale_factor, ndim):
    """Return the spatial zoom that restores an extraction resize."""
    if np.isscalar(scale_factor):
        sf = np.full(ndim, float(scale_factor), dtype=np.float32)
    else:
        sf = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
        if sf.size != ndim:
            raise ValueError(f"scale_factor must have len {ndim}. Got {scale_factor!r}")

    if np.any(sf <= 0):
        raise ValueError(f"scale_factor values must be > 0. Got {scale_factor!r}")

    return tuple(float(1.0 / value) for value in sf)


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


def _load_conditional_target_mask(config, anomaly_basename, target_mask_loader):
    if not getattr(config, "conditional", False):
        return None
    if anomaly_basename is None or target_mask_loader is None:
        raise ValueError("conditional fusion needs anomaly_basename and target_mask_loader to load tgt_mask.")
    try:
        return np.asarray(target_mask_loader(anomaly_basename, artifact="tgt_mask"))
    except (FileNotFoundError, KeyError) as exc:
        raise ValueError(f"conditional fusion needs a saved tgt_mask for {anomaly_basename!r}.") from exc


def _spatial_label_mask(mask, spatial_ndim):
    mask = np.asarray(mask)
    if mask.ndim == spatial_ndim:
        return mask
    if mask.ndim == spatial_ndim + 1:
        return np.max(mask, axis=0)
    raise ValueError(f"target mask must have {spatial_ndim} or {spatial_ndim + 1} dims. Got {mask.shape}")


def fusion2d(
    control_image,
    synth_anomaly_image,
    anomaly_meta,
    position_factor,
    config,
    anomaly_basename=None,
    target_mask_loader=None,
    background_threshold=None,
):
    """
    Fuse a synthetic anomaly into a control image (2D) using alpha blending.

    High-level steps (2D adaptation of fusion3d):
      1) Remove anomaly background via thresholding.
      2) Trim anomaly padding (crop to foreground bounding box).
      3) Restore anomaly size by zooming spatially with inverse `scale_factor`.
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
    target_mask = _load_conditional_target_mask(config, anomaly_basename, target_mask_loader)

    # Validate dimensionality (expects channel-first 2D images)
    if ctrl.ndim != 3 or anom.ndim != 3:
        raise ValueError(f"Both images must be 2D (C,H,W). Got ctrl={ctrl.shape}, anom={anom.shape}")
    if target_mask is not None and target_mask.ndim not in (2, 3):
        raise ValueError(f"target mask must be 2D (H,W) or 3D (C,H,W). Got {target_mask.shape}")
    if target_mask is not None and target_mask.shape[-2:] != anom.shape[-2:]:
        raise ValueError(f"target mask spatial shape {target_mask.shape[-2:]} does not match anomaly {anom.shape[-2:]}")

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
    # Foreground mask over spatial dims: conditional targets define the intended label footprint.
    if target_mask is not None:
        fg = _spatial_label_mask(target_mask, spatial_ndim=2) > 0
    else:
        fg = np.any(anom > background_threshold, axis=0)  # (H,W)

    # Only crop if there is any foreground
    if np.any(fg):
        yy, xx = np.where(fg)
        y0, y1 = yy.min(), yy.max() + 1
        x0, x1 = xx.min(), xx.max() + 1
        anom = anom[:, y0:y1, x0:x1]
        if target_mask is not None:
            if target_mask.ndim == 3:
                target_mask = target_mask[:, y0:y1, x0:x1]
            else:
                target_mask = target_mask[y0:y1, x0:x1]
    # If no foreground exists, `anom` remains unchanged. (You may optionally early-return.)

    # ------------------------------------------------------------
    # 4) Restore anomaly footprint from extraction scale (keep channel axis unchanged)
    # ------------------------------------------------------------
    sf = _inverse_extraction_scale(scale_factor, ndim=2)

    # Zoom factors: (C unchanged, spatial scaled)
    anom = zoom(anom, (1.0, sf[0], sf[1]), order=1)
    if target_mask is not None:
        if target_mask.ndim == 3:
            target_mask = zoom(target_mask, (1.0, sf[0], sf[1]), order=0)
        else:
            target_mask = zoom(target_mask, sf, order=0)
        target_mask = _spatial_label_mask(target_mask, spatial_ndim=2).astype(np.uint8, copy=False)

    # ------------------------------------------------------------
    # 5) Compute insertion offset from normalized position_factor
    # ------------------------------------------------------------
    ctrl_spatial = np.array([H, W], dtype=int)            # control spatial dims
    anom_spatial = np.array(anom.shape[1:], dtype=int)    # anomaly spatial dims after zoom

    # Offset is chosen so anomaly center is placed at position_factor * control_size
    offset = np.array(
        [
        int(round((H - 1) * position_factor[0] - anom_spatial[0] / 2)),
        int(round((W - 1) * position_factor[1] - anom_spatial[1] / 2)),
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
    # 9) Locally normalize anomaly values (per channel) to match the target area plus border
    # ------------------------------------------------------------
    anom = anom.copy() 
    binary_mask = (valid_mask > 0) 

    if np.any(binary_mask):
        dilation_kernel_size = config.fusion_normalization_border_width * 2 + 1
        dilation_structure = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=bool)
        dilated_mask = binary_dilation(binary_mask, structure=dilation_structure)
        
        if np.any(dilated_mask):
            for c in range(C):
                # anomaly intensity
                vp = anom[c][binary_mask]  
                if vp.size == 0:
                    continue
                
                # background intensity
                bg_local = bg_slice[c][dilated_mask]
                if bg_local.size == 0:
                    continue
                
                # mean and std
                a_mean = np.mean(vp)
                a_std = np.std(vp)
                if a_std == 0: 
                    a_std = 1e-5
                    
                bg_mean = np.mean(bg_local)
                bg_std = np.std(bg_local)
                
                # z score matching
                vp_matched = ((vp - a_mean) / a_std) * bg_std + bg_mean
                
                anom[c][binary_mask] = vp_matched

    # ------------------------------------------------------------
    # 10) Create alpha mask from Sobel+distance transform mask (edge-aware)
    # ------------------------------------------------------------
    alpha_mask = get_alpha_mask_sobel_final_2d(
        anom_proj, config, background_threshold
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

    if target_mask is not None:
        multiclass_seg_local = target_mask.astype(np.uint8, copy=False)
    else:
        seg_local_binary = alpha_mask > 0.05
        seg_local_binary = binary_fill_holes(seg_local_binary)
        multiclass_seg_local = (valid_mask * seg_local_binary).astype(np.uint8)

    vm_crop = multiclass_seg_local[:hh, :ww]
    segmentation[h0 : h0 + hh, w0 : w0 + ww] = vm_crop

    if target_mask is None:
        for class_id in np.unique(segmentation):
            if class_id == 0:
                continue
            class_mask = segmentation == class_id
            filled_class_mask = binary_fill_holes(class_mask)
            filled_background_holes = filled_class_mask & (segmentation == 0)
            segmentation[filled_background_holes] = class_id

    segmentation = segmentation[None, ...]                 # (1, H, W)
    if C != 1:
        segmentation = np.repeat(segmentation, C, axis=0)  # (C, H, W)

    # Ensuring no anomalies are fused in and an empty mask is provided -> could this confuse the model? Better to simply return ctrl
    # This is why synth ROI is missing for some samples! (roi becomes None here)
    if np.sum(segmentation) == 0:
        return ctrl, segmentation, None
    
    # ------------------------------------------------------------
    # 14) Extract ROI around the inserted anomaly
    # ------------------------------------------------------------
    
    # centroid
    ch = h0 + (hh / 2.0)
    cw = w0 + (ww / 2.0)
    centroid_voxel = (ch, cw)
    
    if config.fixed_roi_size is None:
        size_spatial = [int(s + max(mp, s * pr)) for s, mp, pr in zip((hh, ww), config.min_pad, config.pad_ratio)]    
    else:
        size_spatial = config.fixed_roi_size

    # crop roi from fused image
    fused_roi = crop_square_clip(
        fused_image, 
        centroid_voxel, 
        size_spatial, 
        centroid_is_normalized=False
    )

    return fused_image, segmentation, fused_roi


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

    
    max_alpha = config.fusion_mask_params["max_alpha"]
    sq = config.fusion_mask_params["sq"]
    steepness_factor = config.fusion_mask_params["steepness_factor"]

    # add seed if you want it reproducible
    if config.fusion_variation:
        # max alpha
        std_alpha = config.fusion_variation_params["alpha_variation"] / config.confidence_z_score
        sampled_alpha = np.random.normal(max_alpha, std_alpha)
        max_alpha = float(np.clip(sampled_alpha, 0.0, 1.0))

        # sq
        std_sq = config.fusion_variation_params["sq_variation"] / config.confidence_z_score
        sampled_sq = np.random.normal(sq, std_sq)
        sq = float(np.maximum(0.1, sampled_sq))

        # steepness
        std_steepness = config.fusion_variation_params["steepness_variation"] / config.confidence_z_score
        sampled_steepness = np.random.normal(steepness_factor, std_steepness)
        steepness_factor = float(np.maximum(0.1, sampled_steepness))
    


    # Initialize alpha mask
    alpha_mask = np.zeros_like(anomaly_arr, dtype=np.float32)

    # ------------------------------------------------------------
    # Build structuring elements for morphology operations
    # ------------------------------------------------------------
    dilation_size = config.fusion_mask_params["dilation_size"]

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
    edge_mask = grad_mag > config.fusion_mask_params["sobel_threshold"]

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
    shave_pixels = config.fusion_mask_params["shave_pixels"]
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
    upsampling_factor = config.fusion_mask_params["upsampling_factor"]

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
    dist_map *= steepness_factor
    dist_map = np.clip(dist_map, 0, 1.0)
    dist_map = (dist_map ** sq) * max_alpha

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
