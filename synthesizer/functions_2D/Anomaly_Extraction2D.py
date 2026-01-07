import numpy as np
from scipy.ndimage import zoom, label, find_objects, center_of_mass


def resize_and_pad_2d(arr, target_size, order=1):
    """
    Resize (downscale only) and center-pad a 3D tensor (C, H, W) to a target spatial size.

    Behavior:
    - Only *downscales* if an input spatial dimension exceeds the target (scale factor < 1).
    - Never upscales (scale factors are capped at 1.0).
    - Pads with the minimum value of `arr` to keep background consistent.
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
    pad_h = (max((tH - h2) // 2, 0), max((tH - h2) - (tH - h2) // 2, 0))
    pad_w = (max((tW - w2) // 2, 0), max((tW - w2) - (tW - w2) // 2, 0))

    pad_widths = ((0, 0), pad_h, pad_w)

    fill = float(np.min(arr))
    arr_padded = np.pad(arr, pad_widths, mode="constant", constant_values=fill)
    arr_padded = arr_padded[:, :tH, :tW]

    return arr_padded, tuple(scale_spatial)


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


def crop_and_center_anomaly_2d(img, seg, target_size, separated_anomaly=True, min_region_pixels=0):
    """
    Extract connected anomaly regions from a 2D segmentation mask and return:
      - normalized-size anomaly cutouts (C, tH, tW) via resize+pad
      - ROI cutouts around the anomaly centroid (variable size)

    Pipeline:
      1) Collapse seg across channel axis -> binary 2D mask (H,W)
      2) Connected-component labeling -> individual anomaly regions
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
    separated_anomaly:
        Currently kept for API-compatibility with 3D version.
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
    anomalies_roi:
        list[np.ndarray]
        ROI crops around anomaly centroid, shape (C, h', w') (variable).

    Notes
    -----
    - `seg` is treated as anomaly mask; any value > 0 counts as anomaly.
    - This function expects **channel-first** arrays: (C, H, W).
    """
    target_size = _spatial_target_size(target_size)

    if seg is None or np.all(seg == 0):
        return None

    if img.ndim != 3:
        raise ValueError(f"img must be 3D (C,H,W). Got {img.shape}")
    if seg.ndim != 3:
        raise ValueError(f"seg must be 3D (C,H,W). Got {seg.shape}")
    #if img.shape != seg.shape:
    #    raise ValueError(f"img and seg must have same shape. Got img={img.shape}, seg={seg.shape}")

    C, H, W = img.shape
    shape = img.shape

    binary2d = np.any(seg > 0, axis=0).astype(np.uint8)  # (H,W)

    labeled, num = label(binary2d)
    regions = find_objects(labeled)

    anomalies = []
    anomalies_roi = []

    if min_region_pixels <= 0:
        min_region_pixels = int(0.05 * (target_size[0] * target_size[1]))

    for ridx, region in enumerate(regions, start=1):
        if region is None:
            continue

        hsl, wsl = region
        region_mask = (labeled[region] == ridx)
        pixels = int(region_mask.sum())

        if pixels < min_region_pixels:
            print(f"Anomaly region {ridx} omitted! pixels={pixels} < {min_region_pixels}")
            continue

        result = img[:, hsl, wsl]  # (C,h,w)

        ch, cw = center_of_mass(binary2d, labeled, ridx)
        centroid_voxel = (int(round(ch)), int(round(cw)))
        centroid_norm = (centroid_voxel[0] / H, centroid_voxel[1] / W)

        padded_arr, scale_factor = resize_and_pad_2d(result, target_size=target_size, order=1)
        scale_factor = tuple(round(float(ele), 4) for ele in scale_factor)

        label_tmp = float(np.max(seg).round(0))

        meta_data = {
            "label": label_tmp,
            "scale_factor": scale_factor,
            "centroid_voxel": centroid_voxel,
            "centroid_norm": centroid_norm,
            "shape": shape
        }

        size_spatial = tuple(max(1, s + 10) for s in result.shape[-2:])
        anomalies_roi.append(crop_square_clip(img, centroid_voxel, size_spatial, centroid_is_normalized=False))

        anomalies.append((padded_arr, meta_data))

    return anomalies, anomalies_roi
