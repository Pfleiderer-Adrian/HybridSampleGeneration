import numpy as np
from scipy.ndimage import zoom, label, find_objects


def _as_axis_tuple(value, ndim, name):
    if np.isscalar(value):
        return (value,) * ndim

    values = tuple(value)
    if len(values) < ndim:
        raise ValueError(f"{name} must have at least len {ndim}. Got {value!r}")
    return values[:ndim]


def dynamic_roi_size(spatial_shape, min_pad, pad_ratio, min_roi_size):
    spatial_shape = tuple(int(size) for size in spatial_shape)
    min_pad = _as_axis_tuple(min_pad, len(spatial_shape), "min_pad")
    pad_ratio = _as_axis_tuple(pad_ratio, len(spatial_shape), "pad_ratio")
    min_roi_size = _as_axis_tuple(min_roi_size, len(spatial_shape), "min_roi_size")

    return [
        max(int(size + max(axis_min_pad, size * axis_pad_ratio)), int(axis_min_roi))
        for size, axis_min_pad, axis_pad_ratio, axis_min_roi
        in zip(spatial_shape, min_pad, pad_ratio, min_roi_size)
    ]


def resize_and_pad_2d(arr, target_size, order=1):
    """
    Resize (downscale only) and center-pad a 3D tensor (C, H, W) to a target spatial size.

    Behavior:
    - Only *downscales* if an input spatial dimension exceeds the target (scale factor < 1).
    - Never upscales (scale factors are capped at 1.0).
    - Pads with the minimum value of `arr` to keep background consistent.
    - Pads symmetrically so the anomaly stays centered in the saved cutout.
    - Returns the padded array cropped to exactly (C, tH, tW).

    Inputs
    ------
    arr:
        np.ndarray with shape (C, H, W).
    target_size:
        Target spatial size (tH, tW).
    order:
        Interpolation order for scipy.ndimage.zoom (1=linear).
    Outputs
    -------
    arr_padded:
        np.ndarray
        Array of shape (C, tH, tW).
    scale_factor:
        tuple[float, float]
        Per-axis scale factor used for (H, W). Values are in (0, 1] due to "no upscaling".

    Raises
    ------
    ValueError:
        If arr.ndim != 3.
    """
    if arr.ndim != 3:
        raise ValueError(f"resize_and_pad_3d expects 3D (C,h,w). Got {arr.shape}")

    C, h, w = arr.shape
    tH, tW = target_size

    scale_spatial = [min(t / s, 1.0) for s, t in zip((h, w), (tH, tW))]

    if any(sf < 1.0 for sf in scale_spatial):
        arr = zoom(arr, (1.0, scale_spatial[0], scale_spatial[1]), order=order)

    _, h2, w2 = arr.shape
    pad_total_h = max(tH - h2, 0)
    pad_total_w = max(tW - w2, 0)

    pad_h0 = pad_total_h // 2
    pad_w0 = pad_total_w // 2

    pad_h = (pad_h0, pad_total_h - pad_h0)
    pad_w = (pad_w0, pad_total_w - pad_w0)

    pad_widths = ((0, 0), pad_h, pad_w)

    fill = float(np.min(arr))
    arr_padded = np.pad(arr, pad_widths, mode="constant", constant_values=fill)
    arr_padded = arr_padded[:, :tH, :tW]

    return arr_padded, tuple(scale_spatial)


def _normalize_anomaly(arr, normalization, eps):
    """
    Normalize a cutout for training and return normalization metadata.

    Supported normalization:
      - "zscore": (x - mean) / std
      - "zscore_median": (x - median) / mad
    """
    if normalization is None or str(normalization).lower() in ("none", "null"):
        return arr, {"norm_type": None}

    norm = str(normalization).lower()
    if norm in ("zscore", "z-score", "z_score"):
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < eps:
            std = eps
        return (arr - mean) / std, {"norm_type": "zscore", "norm_mean": mean, "norm_std": std}

    if norm in ("zscore_median", "z-score-median", "zscore-median"):
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad < eps:
            mad = eps
        return (arr - median) / mad, {"norm_type": "zscore_median", "norm_median": median, "norm_mad": mad}

    raise ValueError(f"Unknown normalization: {normalization!r}")


def crop_square_clip(arr, centroid, size, centroid_is_normalized=None):
    """
    Crop a square/rect subregion from a 3D (C, H, W) array, clipping to image bounds.

    Inputs
    ------
    arr:
        np.ndarray with shape (C, H, W).
    centroid:
        Center location of the crop. Accepted formats:
          - length 2: (h, w)
          - length 3: (c, h, w)  -> channel index ignored
        Values may be:
          - pixel coordinates (ints/floats)
          - or normalized coordinates in [0,1] (if centroid_is_normalized=True or auto-detected)
    size:
        Requested crop size. Accepted formats:
          - (H, W)
          - (C, H, W) -> last 2 are used
    centroid_is_normalized:
        If None, auto-detect: if 0<=h<=~1 and 0<=w<=~1, centroid is interpreted as normalized.

    Returns
    -------
    np.ndarray:
        Cropped view of shape (C, h', w') where h'<=H and w'<=W.

    Raises
    ------
    ValueError:
        If arr.ndim != 3.
    """
    if arr.ndim != 3:
        raise ValueError(f"crop_square_clip expects 3D (C,H,W). Got {arr.shape}")

    C, H, W = arr.shape

    centroid = tuple(centroid)
    if len(centroid) == 2:
        ch, cw = centroid
    elif len(centroid) == 3:
        ch, cw = centroid[1], centroid[2]
    else:
        raise ValueError(f"centroid must be length 2 or 3. Got {centroid}")

    size = tuple(size)
    sh, sw = size[-2], size[-1]

    if centroid_is_normalized is None:
        # allow a small margin above 1.0 to avoid false negatives due to rounding
        centroid_is_normalized = (0.0 <= ch <= 1.2) and (0.0 <= cw <= 1.2)

    if centroid_is_normalized:
        ch = ch * H
        cw = cw * W

    ch, cw = int(round(ch)), int(round(cw))
    sh, sw = int(sh), int(sw)

    h0 = ch - sh // 2
    w0 = cw - sw // 2
    h1 = h0 + sh
    w1 = w0 + sw

    # shift y axis
    if h0 < 0:
        h1 = h1 - h0
        h0 = 0
    elif h1 > H:
        h0 = h0 - (h1 - H)
        h1 = H

    # shift x axis
    if w0 < 0:
        w1 = w1 - w0
        w0 = 0
    elif w1 > W:
        w0 = w0 - (w1 - W)
        w1 = W

    h0c, w0c = max(h0, 0), max(w0, 0)
    h1c, w1c = min(h1, H), min(w1, W)

    return arr[:, h0c:h1c, w0c:w1c]


def _spatial_target_size(target_size):
    """
    Normalize target_size to a pure spatial (H, W) tuple.

    Inputs
    ------
    target_size:
        Either:
          - (H, W)
          - (C, H, W)  -> last 2 are used

    Outputs
    -------
    tuple[int, int]
        Spatial target size (H, W).
    """
    if len(target_size) == 2:
        return tuple(target_size)
    if len(target_size) == 3:
        return tuple(target_size[-2:])
    raise ValueError(f"target_size must be (H,W) or (C,H,W). Got {target_size}")

def add_bg_noise_floor(img, sigma_rel=0.003, eps=1e-8):
    """
    sigma_rel: relative Stärke zum Dynamikbereich (0.1% - 1% ist typisch)
    """
    img = img.copy()
    bg = img.min()

    # Maske: überall wo wirklich Background ist (oder fast Background)
    mask = np.isclose(img, bg, atol=eps)

    # Dynamikbereich schätzen
    dyn = img.max() - img.min()
    sigma = sigma_rel * (dyn + 1e-12)

    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(img.dtype)

    img[mask] = bg + noise[mask]
    return img


def crop_and_center_anomaly_2d(
    img,
    seg,
    config,
    target_size,
    *,
    normalization=None,
    normalization_eps=1e-8,
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
    target_size:
        Target spatial size (tH, tW) or (C,tH,tW)
    min_region_pixels:
        Minimum number of pixels for a connected component to be kept.
        If <=0, defaults to 5% of the target crop area.
    normalization:
        Normalization mode for anomaly cutouts: "zscore", "zscore_median", or None.
    normalization_eps:
        Small epsilon to avoid division by zero in normalization.

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
    target_size = _spatial_target_size(target_size)

    if seg is None or np.all(seg == 0):
        return None, None, None

    if img.ndim != 3:
        raise ValueError(f"img must be 3D (C,H,W). Got {img.shape}")
    if seg.ndim != 3:
        raise ValueError(f"seg must be 3D (C,H,W). Got {seg.shape}")
    #if img.shape != seg.shape:
    #    raise ValueError(f"img and seg must have same shape. Got img={img.shape}, seg={seg.shape}")

    C, H, W = img.shape
    shape = img.shape

    binary2d = np.any(seg > 0, axis=0).astype(np.uint8)  # (H,W)

    if config.separated_anomaly:
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

    min_region_pixels = int(config.min_anomaly_percentage * (target_size[0] * target_size[1]))

    for ridx, region in enumerate(regions, start=1):

        hsl, wsl = region
        region_mask = (labeled[region] == ridx)
        pixels = int(region_mask.sum())

        if pixels < min_region_pixels:
            print(f"Anomaly region {ridx} omitted! pixels={pixels} < {min_region_pixels}")
            continue

        result = img[:, hsl, wsl]  # (C,h,w)
        result = np.where(region_mask, result, np.min(img))

        if config.add_bg_noise:
            result = add_bg_noise_floor(result)

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
        )
        padded_arr, norm_meta = _normalize_anomaly(
            padded_arr, normalization=normalization, eps=float(normalization_eps)
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

        if config.fixed_roi_size is None:
            size_spatial = dynamic_roi_size(result.shape[-2:], config.min_pad, config.pad_ratio, config.min_roi_size)
        else:
            size_spatial = config.fixed_roi_size
        
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
