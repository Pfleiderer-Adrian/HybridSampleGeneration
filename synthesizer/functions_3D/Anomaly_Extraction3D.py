import numpy as np
from scipy.ndimage import zoom, label, find_objects, center_of_mass

def resize_and_pad_3d(arr, target_size, order=1, *, random_offset=False, rng=None):
    """
    Resize (downscale only) and center-pad a 4D tensor (C, D, H, W) to a target spatial size.

    Behavior:
    - Only *downscales* if an input spatial dimension exceeds the target (scale factor < 1).
    - Never upscales (scale factors are capped at 1.0).
    - Pads with the minimum value of `arr` to keep background consistent.
    - If random_offset=True, distributes padding asymmetrically to randomize anomaly placement.
    - Returns the padded array cropped to exactly (C, tD, tH, tW).

    Inputs
    ------
    arr:
        np.ndarray with shape (C, D, H, W).
    target_size:
        Target spatial size (tD, tH, tW).
    order:
        Interpolation order for scipy.ndimage.zoom (1=linear).
    random_offset:
        If True, apply a random spatial offset by using asymmetric padding.
    rng:
        Optional numpy random Generator for reproducible offsets.

    Outputs
    -------
    arr_padded:
        np.ndarray with shape (C, tD, tH, tW).
    scale_spatial:
        tuple[float, float, float]
        Per-axis scale factor used for (D, H, W). Values are in (0, 1] due to "no upscaling".

    Raises
    ------
    ValueError:
        If arr.ndim != 4.
    """
    if arr.ndim != 4:
        raise ValueError(f"resize_and_pad_4d expects 4D (C,d,h,w). Got {arr.shape}")

    C, d, h, w = arr.shape
    tD, tH, tW = target_size

    scale_spatial = [min(t / s, 1.0) for s, t in zip((d, h, w), (tD, tH, tW))]

    if any(sf < 1.0 for sf in scale_spatial):
        arr = zoom(arr, (1.0, scale_spatial[0], scale_spatial[1], scale_spatial[2]), order=order)

    _, d2, h2, w2 = arr.shape
    pad_total_d = max(tD - d2, 0)
    pad_total_h = max(tH - h2, 0)
    pad_total_w = max(tW - w2, 0)

    if random_offset:
        if rng is None:
            rng = np.random.default_rng()
        pad_d0 = int(rng.integers(0, pad_total_d + 1))
        pad_h0 = int(rng.integers(0, pad_total_h + 1))
        pad_w0 = int(rng.integers(0, pad_total_w + 1))
    else:
        pad_d0 = pad_total_d // 2
        pad_h0 = pad_total_h // 2
        pad_w0 = pad_total_w // 2

    pad_d = (pad_d0, pad_total_d - pad_d0)
    pad_h = (pad_h0, pad_total_h - pad_h0)
    pad_w = (pad_w0, pad_total_w - pad_w0)

    pad_widths = ((0, 0), pad_d, pad_h, pad_w)

    fill = float(np.min(arr))
    arr_padded = np.pad(arr, pad_widths, mode="constant", constant_values=fill)
    arr_padded = arr_padded[:, :tD, :tH, :tW]

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


def crop_cube_clip(arr, centroid, size, centroid_is_normalized=None):
    """
    Crop a cube-like subvolume from a 4D (C, D, H, W) array, clipping to image bounds.

    Inputs
    ------
    arr:
        np.ndarray with shape (C, D, H, W).
    centroid:
        Center location of the crop. Accepted formats:
          - length 3: (d, h, w)
          - length 4: (c, d, h, w)  -> channel index ignored
        Values may be:
          - voxel coordinates (ints/floats)
          - or normalized coordinates in [0,1] (if centroid_is_normalized=True or auto-detected)
    size:
        Crop size. Accepted formats:
          - (D, H, W)
          - (C, D, H, W) (only the last 3 values are used)
    centroid_is_normalized:
        If True, centroid is interpreted as normalized and multiplied by (D,H,W).
        If None, auto-detects normalized centroid if all components are in [0, ~1.2].

    Outputs
    -------
    np.ndarray:
        Cropped subvolume with shape (C, d', h', w') where d'/h'/w' may be smaller if crop hits boundaries.

    Raises
    ------
    ValueError:
        If arr.ndim != 4 or centroid length is not 3/4.
    """
    if arr.ndim != 4:
        raise ValueError(f"crop_cube_clip expects 4D (C,D,H,W). Got {arr.shape}")

    C, D, H, W = arr.shape

    centroid = tuple(centroid)
    if len(centroid) == 4:
        cd, ch, cw = centroid[1], centroid[2], centroid[3]
    elif len(centroid) == 3:
        cd, ch, cw = centroid
    else:
        raise ValueError(f"centroid must be len 3 or 4, got {centroid}")

    size = tuple(size)
    sd, sh, sw = size[-3], size[-2], size[-1]

    if centroid_is_normalized is None:
        centroid_is_normalized = (0.0 <= cd <= 1.2) and (0.0 <= ch <= 1.2) and (0.0 <= cw <= 1.2)

    if centroid_is_normalized:
        cd = cd * D
        ch = ch * H
        cw = cw * W

    cd, ch, cw = int(round(cd)), int(round(ch)), int(round(cw))
    sd, sh, sw = int(sd), int(sh), int(sw)

    d0 = cd - sd // 2
    h0 = ch - sh // 2
    w0 = cw - sw // 2

    d1 = d0 + sd
    h1 = h0 + sh
    w1 = w0 + sw

    d0c, h0c, w0c = max(d0, 0), max(h0, 0), max(w0, 0)
    d1c, h1c, w1c = min(d1, D), min(h1, H), min(w1, W)

    return arr[:, d0c:d1c, h0c:h1c, w0c:w1c]


def _spatial_target_size(target_size):
    """
    Normalize target_size to a pure spatial (D, H, W) tuple.

    Inputs
    ------
    target_size:
        Either:
          - (D, H, W)
          - (C, D, H, W)  -> last 3 are used

    Outputs
    -------
    tuple[int, int, int]
        Spatial target size (D, H, W).

    Raises
    ------
    ValueError:
        If target_size is not length 3 or 4.
    """
    # accept (D,H,W) or (C,D,H,W)
    if len(target_size) == 3:
        return tuple(target_size)
    if len(target_size) == 4:
        return tuple(target_size[-3:])
    raise ValueError(f"target_size must be (D,H,W) or (C,D,H,W), got {target_size}")


def crop_and_center_anomaly_3d(
    img,
    seg,
    config,
    target_size,
    separated_anomaly=True,
    *,
    random_offset=False,
    rng=None,
    normalization=None,
    normalization_eps=1e-8,
):
    """
    Extract connected anomaly regions from a 3D segmentation mask and return:
      - normalized-size anomaly cutouts (C, tD, tH, tW) via resize+pad
      - ROI cutouts around the anomaly centroid (variable size)

    Pipeline:
      1) Collapse seg across channel axis -> binary 3D mask (D,H,W)
      2) Connected-component labeling -> individual anomaly regions
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
    target_size:
        Either (tD, tH, tW) or (C, tD, tH, tW). Only spatial dims are used.
    separated_anomaly:
        not implemented -> next updates
    min_region_voxels:
        Minimum voxel count for a connected component to be kept.
        If <= 0, defaults to 5% of target volume (0.05 * tD * tH * tW).
    random_offset:
        If True, apply random spatial offsets to anomaly cutouts after resize+pad.
    rng:
        Optional numpy random Generator for reproducible offsets.
    normalization:
        Normalization mode for anomaly cutouts: "zscore", "zscore_median", or None.
    normalization_eps:
        Small epsilon to avoid division by zero in normalization.

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

    Notes
    -----
    - If seg is None or completely empty, the function returns None in the original code.

    Raises
    ------
    ValueError:
        If img/seg are not 4D or shapes do not match.
    """
    target_size = _spatial_target_size(target_size)
    if seg is None or np.all(seg == 0):
        return None

    if img.ndim != 4:
        raise ValueError(f"img must be (C,D,H,W). Got {img.shape}")
    if seg.ndim != 4:
        raise ValueError(f"seg must be (C,D,H,W). Got {seg.shape}")
    if img.shape != seg.shape:
        raise ValueError(f"img and seg must have same shape. Got img={img.shape}, seg={seg.shape}")

    C, D, H, W = img.shape
    shape = img.shape

    binary3d = np.any(seg > 0, axis=0).astype(np.uint8)  # (D,H,W)

    labeled, num = label(binary3d)
    regions = find_objects(labeled)

    anomalies = []
    anomalies_roi = []

    min_region_voxels = int(config.min_anomaly_percentage * (target_size[0] * target_size[1] * target_size[2]))

    for ridx, region in enumerate(regions, start=1):
        if region is None:
            continue

        dsl, hsl, wsl = region
        region_mask = (labeled[region] == ridx)
        voxels = int(region_mask.sum())

        if voxels < min_region_voxels:
            print(f"Anomaly region {ridx} omitted! voxels={voxels} < {min_region_voxels}")
            continue

        result = img[:, dsl, hsl, wsl]  # (C,d,h,w)
        result = np.where(region_mask, result, np.min(img))

        cd, ch, cw = center_of_mass(binary3d, labeled, ridx)
        centroid_voxel = (int(round(cd)), int(round(ch)), int(round(cw)))
        centroid_norm = (centroid_voxel[0] / D, centroid_voxel[1] / H, centroid_voxel[2] / W)

        padded_arr, scale_factor = resize_and_pad_3d(
            result,
            target_size=target_size,
            order=1,
            random_offset=random_offset,
            rng=rng,
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

        size_spatial = [int(s + max(mp, s * pr)) for s, mp, pr in zip(result.shape[-3:], config.min_pad, config.pad_ratio)]

        anomalies_roi.append(crop_cube_clip(img, centroid_voxel, size_spatial, centroid_is_normalized=False))

        anomalies.append((padded_arr, meta_data))

    return anomalies, anomalies_roi
