import numpy as np
from tqdm import tqdm
from skimage.feature import match_template

def _to_spatial(arr: np.ndarray) -> np.ndarray:

    arr = np.asarray(arr)
    if arr.shape[0] == 1:
        return arr[0]
    return np.max(arr, axis=0)


def template_matching(template, control):

    template = _to_spatial(template)
    control = _to_spatial(control)

    # Check if template fits in control sample
    if any(t_dim > c_dim for t_dim, c_dim in zip(template.shape, control.shape)):
        return -2, None
    
    # Compute correlation map (same dimensionality as control minus template extents)
    result = match_template(control, template)

    # Best similarity score is max of correlation map
    similarity_score = float(np.max(result))

    # Index of best match corresponds to the template's top-left-front corner position
    idx_tuple = np.unravel_index(np.argmax(result), result.shape)

    center_coords = tuple(
        top_left + (dim_size // 2) 
        for top_left, dim_size in zip(idx_tuple, template.shape)
    )

    return similarity_score, center_coords

import numpy as np

def ssim_01(x, y, data_range=None, k1=0.01, k2=0.03):
    """
    x,y: HxW oder HxWxC, dtype float oder uint8.
    Gibt SSIM in [0,1] zurück (vereinfacht global, ohne Sliding Window).
    """
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    # pro Kanal rechnen und mitteln
    if x.ndim == 3:
        return float(np.mean([ssim_01(x[..., c], y[..., c], data_range, k1, k2) for c in range(x.shape[2])]))

    if data_range is None:
        # robust: Range aus beiden Bildern
        data_range = max(x.max(), y.max()) - min(x.min(), y.min())
        if data_range == 0:
            return 1.0 if np.allclose(x, y) else 0.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    # SSIM kann minimal außerhalb liegen durch Numerik
    return float(np.clip(ssim, 0.0, 1.0))

import numpy as np
from scipy import ndimage

import numpy as np
from scipy import ndimage



def center_foreground_com(
    img,
    threshold,
    fg_is_brighter=True,
    fill_value=0,
    largest_only=False,
    connectivity=2,
    min_size=1,
    order=0,
):
    """
    Centers the foreground by its center of mass (CoM).

    Parameters
    ----------
    img : ndarray
        Input image, either 2D (H, W) or 3D (H, W, C).
    threshold : float
        Background threshold used to create the foreground mask.
    fg_is_brighter : bool, default=True
        If True, foreground is defined as gray > threshold.
        If False, foreground is defined as gray < threshold.
    fill_value : scalar, default=0
        Constant value used to fill newly introduced borders after shifting.
    largest_only : bool, default=False
        If True and multiple connected components exist, keep only the largest one.
    connectivity : int, default=2
        Connectivity for connected-component labeling in 2D:
        1 = 4-neighborhood, 2 = 8-neighborhood.
    min_size : int, default=1
        Minimum component size (in pixels) to be considered when filtering components.
        Only used when largest_only=True.
    order : int, default=0
        Interpolation order for ndimage.shift (e.g., 0, 1, 3, ...).

    Returns
    -------
    shifted : ndarray
        Shifted image with the foreground centered.
    shift : tuple(float, float)
        Applied shift (shift_y, shift_x).
    com : tuple(float, float)
        Original center of mass (cy, cx) computed from the mask.
    mask : ndarray (bool)
        Foreground mask used for the center-of-mass computation.
    """
    # Convert to 2D intensity image for mask computation
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img

    # Build foreground mask based on threshold and polarity
    mask = gray > threshold if fg_is_brighter else gray < threshold
    if mask.sum() == 0:
        raise ValueError("Foreground mask is empty. Check threshold/fg_is_brighter.")

    # Optional: keep only the largest connected component
    if largest_only:
        # Create a 2D connectivity structure (4- or 8-connected)
        structure = ndimage.generate_binary_structure(rank=2, connectivity=connectivity)
        labeled, n = ndimage.label(mask, structure=structure)

        if n == 0:
            raise ValueError("No components found (mask might be empty?).")

        # Compute component sizes per label (label 0 is background)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0

        # Filter out small components if requested
        if min_size > 1:
            keep = np.where(sizes >= min_size)[0]
            if len(keep) == 0:
                raise ValueError(f"No component >= min_size={min_size} found.")
            # Pick the largest among the remaining components
            largest_label = keep[np.argmax(sizes[keep])]
        else:
            # Pick the overall largest component
            largest_label = np.argmax(sizes)

        mask = (labeled == largest_label)

    # Center of mass in (y, x) coordinates
    cy, cx = ndimage.center_of_mass(mask.astype(np.float32))

    # Target is the image center (pixel-centered for odd/even sizes)
    H, W = gray.shape
    target_y, target_x = (H - 1) / 2.0, (W - 1) / 2.0
    shift_y, shift_x = target_y - cy, target_x - cx

    # Apply shift (no wrap-around; constant padding)
    if img.ndim == 2:
        shifted = ndimage.shift(
            img, shift=(shift_y, shift_x),
            order=order, mode="constant", cval=fill_value
        )
    else:
        # Shift each channel independently and stack back to (H, W, C)
        shifted = np.stack([
            ndimage.shift(
                img[..., c], shift=(shift_y, shift_x),
                order=order, mode="constant", cval=fill_value
            )
            for c in range(img.shape[2])
        ], axis=2)

    return shifted, (shift_y, shift_x), (cy, cx), mask





import numpy as np

def crop_border(arr: np.ndarray, bg_thresh: float, margin: int = 0) -> np.ndarray:
    """
    Crop away black/background borders by extracting the tight bounding box
    around foreground pixels/voxels (values > bg_thresh).

    Parameters
    ----------
    arr : np.ndarray
        Input image/volume with shape (C, H, W) or (C, D, H, W).
    bg_thresh : float
        Background threshold. Values <= bg_thresh are treated as background.
    margin : int
        Optional extra padding (in pixels/voxels) to keep around the detected box.

    Returns
    -------
    np.ndarray
        Cropped array in the same layout as the input.
        If no foreground is found, the original array is returned unchanged.
    """
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected (C,H,W) or (C,D,H,W), got shape={arr.shape}")

    # Build a foreground mask by aggregating across channels:
    # if any channel at a position is > bg_thresh => foreground
    fg_mask = np.any(arr > bg_thresh, axis=0)  # (H,W) or (D,H,W)

    # If nothing exceeds the threshold, there's nothing to crop
    if not np.any(fg_mask):
        return arr

    # Find min/max indices of foreground coordinates (tight bounding box)
    coords = np.argwhere(fg_mask)            # shape (N, 2) or (N, 3)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1            # +1 for exclusive slicing end

    # Apply optional margin and clamp to valid bounds
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, fg_mask.shape)

    if arr.ndim == 3:
        # arr: (C,H,W), mask indices: (y,x)
        y0, x0 = mins
        y1, x1 = maxs
        return arr[:, y0:y1, x0:x1]
    else:
        # arr: (C,D,H,W), mask indices: (z,y,x)
        z0, y0, x0 = mins
        z1, y1, x1 = maxs
        return arr[:, z0:z1, y0:y1, x0:x1]


def create_matching_dictionary(control_sample_dataloader, roi_dataloader, config, matching_routine="local", anomaly_duplicates=False):

    # template_output_dir übergeben, wenn templates_path in config noch nicht überschrieben wurde (also die templates noch nicht generiert wurden)
    allowed_matchings_routines = ["local", "global", "fixed_from_extraction_anomaly_fusion", "fixed_from_extraction_control_fusion"]
    if matching_routine not in allowed_matchings_routines:
        raise ValueError("Not a allowed matching routine.")
    
    # check (for one sample) if control_sample_dataloader loads samples with shape [C,H,W] or [C,D,H,W]
    shape_checked = False
    
    # Rows to be written later to a CSV by the caller
    matching_data = []

    checked_roi_names = set()
    checked_control_names = set()

    # Tracks ROI filenames already used (only relevant for "global")
    excluded_roi_sample_names = []


    # ------------------------------------------------------------
    # Routine: fusion real anomaly and synthetic anomaly into one sample
    # ------------------------------------------------------------
    if matching_routine == "fixed_from_extraction_anomaly_fusion":
        i = 0
        for control, _, control_filename, *ignored in tqdm(control_sample_dataloader):
            checked_control_names.add(control_filename)
            if not shape_checked:
                if control.ndim not in [3, 4]:
                    raise ValueError("Control sample has to be 3D [C,H,W] or 4D [C,D,H,W]")
                if control.shape[0] >= np.min(control.shape[1:]):
                    print(f"Warning: First dimension of first control sample {control_filename} is larger than another one. Shape: {control.shape}. Channel dimension must be first.")
                shape_checked = True
            j = 0
            anomaly_list = []
            while True:
                try:
                    roi = roi_dataloader.load_numpy_by_basename(control_filename+"_"+str(j)+".npy")
                    centroid = config.syn_anomaly_transformations[control_filename+"_"+str(j)+".npy"]["centroid_norm"]

                    anomaly_list.append(
                        (control_filename+"_"+str(j)+".npy", centroid)
                    )
                    j += 1
                except Exception as e:
                    print(e.with_traceback(None))
                    print(f"Finished loading {j-1} synthetic anomalies for control sample {control_filename}.")
                    break
            matching_data.append([control_filename,anomaly_list])


    # ------------------------------------------------------------
    # Routine: fixed_from_extraction (sequential pairing, centroid from metadata)
    # ------------------------------------------------------------
    if matching_routine == "fixed_from_extraction_control_fusion":
        i = 0
        for control, _, control_filename, *ignored in tqdm(control_sample_dataloader):
            checked_control_names.add(control_filename)
            if not shape_checked:
                if control.ndim not in [3, 4]:
                    raise ValueError("Control sample has to be 3D [C,H,W] or 4D [C,D,H,W]")
                if control.shape[0] >= np.min(control.shape[1:]):
                    print(f"Warning: First dimension of first control sample {control_filename} is larger than another one. Shape: {control.shape}. Channel dimension must be first.")
                shape_checked = True

            # Stop if ROI dataset is exhausted
            if i >= roi_dataloader.__len__():
                if anomaly_duplicates:
                    i = 0
                    roi, roi_filename = roi_dataloader[i]
                    i += 1
                else:
                    break
            else:
                roi, roi_filename = roi_dataloader[i]
                checked_roi_names.add(roi_filename)
                i += 1

            if roi_filename is None:
                centroid = None
            else:
                centroid = config.syn_anomaly_transformations[roi_filename]["centroid_norm"]

            matching_data.append(
                [control_filename, [(roi_filename, centroid)]]
            )

    # ------------------------------------------------------------
    # Routine: local (sequential ROI assignment, per-control template matching)
    # ------------------------------------------------------------
    if matching_routine == "local":
        i = 0
        skipped_rois = {}   # want no duplicates here
        for control, _, control_filename, *ignored in tqdm(control_sample_dataloader):
            checked_control_names.add(control_filename)
            if not shape_checked:
                if control.ndim not in [3, 4]:
                    raise ValueError("Control sample has to be 3D [C,H,W] or 4D [C,D,H,W]")
                if control.shape[0] >= np.min(control.shape[1:]):
                    print(f"Warning: First dimension of control sample {control_filename} is larger than another one. Shape: {control.shape}. Channel dimension must be first.")
                shape_checked = True
            highest_sim_position_factor = None
            spatial_shape = np.array(control.shape[1:], dtype=float)  # aus config oder aus shape?

            # always check previously skipped rois first for new control
            for roi_filename, roi in skipped_rois.items():
                sim, opt_center = template_matching(roi, control)
                if sim >= -1:
                    highest_sim_position_factor = (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                    matching_data.append([control_filename, [(roi_filename, highest_sim_position_factor)]])
                    skipped_rois.pop(roi_filename)
                    break
            
            # go to next control sample if match was found in skipped_rois
            if highest_sim_position_factor is not None:
                continue

            # if no match was found load new roi
            if i >= roi_dataloader.__len__():
                if anomaly_duplicates:
                    i = 0
                    roi, roi_filename = roi_dataloader[i]

                else:
                    if skipped_rois:
                        continue    # check if skipped rois fit in another control
                    else: 
                        break   # done, no roi left to match
            else:
                roi, roi_filename = roi_dataloader[i]
                checked_roi_names.add(roi_filename)
            
            i_start = i
            i += 1

            if roi_filename is not None:
                # Only compute match if ROI not excluded (list is empty in this routine by default)
                if roi_filename not in excluded_roi_sample_names:
                    sim, opt_center = template_matching(roi, control)
                    # while roi doesn't fit in control try next roi
                    while sim < -1:
                        if i >= roi_dataloader.__len__():
                            if anomaly_duplicates:
                                i = 0
                            else:
                                break   # no more rois to check (break for anomaly_duplicates=False)
                        if i == i_start:
                            break   # break after checking every roi once (break for anomaly_duplicates=True)
                        skipped_rois[roi_filename] = roi
                        roi, roi_filename = roi_dataloader[i]
                        checked_roi_names.add(roi_filename)
                        i += 1
                        
                        sim, opt_center = template_matching(roi, control)

                    # no match possible for current control
                    if sim < -1:
                        continue

                    # found match
                    # Convert center (row,col)/(slice,row,col) to normalized position factor (H,W)/(D,H,W).
                    highest_sim_position_factor = (np.array(opt_center, dtype=float) / spatial_shape).tolist()

                matching_data.append([control_filename, [(roi_filename, highest_sim_position_factor)]])


    # ------------------------------------------------------------
    # Routine: global (search best ROI for each control; avoid reusing ROI)
    # ------------------------------------------------------------
    if matching_routine == "global":
        skipped_controls = []
        for control, _, control_filename, *ignored in control_sample_dataloader:
            checked_control_names.add(control_filename)
            if not shape_checked:
                if control.ndim not in [3, 4]:
                    raise ValueError("Control sample has to be 3D [C,H,W] or 4D [C,D,H,W]")
                if control.shape[0] >= np.min(control.shape[1:]):
                    print(f"Warning: First dimension of control sample {control_filename} is larger than another one. Shape: {control.shape}. Channel dimension must be first.")
                shape_checked = True
            highest_sim = -np.inf
            highest_sim_roi_name = None
            highest_sim_position_factor = None
            spatial_shape = np.array(control.shape[1:], dtype=float)
            
            for roi, roi_filename in tqdm(roi_dataloader):
                if roi_filename in excluded_roi_sample_names:
                    continue
                if anomaly_duplicates and len(excluded_roi_sample_names) == roi_dataloader.__len__():
                    excluded_roi_sample_names = []

                sim, opt_center = template_matching(roi, control)

                if sim > highest_sim:
                    highest_sim = sim
                    highest_sim_roi_name = roi_filename

                    # Normalize by spatial shape only (H,W)/(D,H,W).
                    highest_sim_position_factor = (
                        (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                    )
            
            # no matching template in remaining rois
            if highest_sim < -1:
                skipped_controls.append((control, control_filename))
                continue

            matching_data.append([control_filename, [(highest_sim_roi_name, highest_sim_position_factor)]])
            excluded_roi_sample_names.append(highest_sim_roi_name)

        if anomaly_duplicates and skipped_controls:
            # try to find matches for skipped controls
            for control, control_name in tqdm(skipped_controls):
                # try all rois (no exclusion)
                highest_sim = -np.inf
                highest_sim_roi_name = None
                highest_sim_position_factor = None

                for roi, roi_filename in roi_dataloader:
                    checked_roi_names.add(roi_filename)
                    sim, opt_center = template_matching(roi, control)
                    if sim > highest_sim:
                        highest_sim = sim
                        highest_sim_roi_name = roi_filename
                        spatial_shape = np.array(control.shape[1:], dtype=float)
                        highest_sim_position_factor = (
                            (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                        )                    
                if highest_sim_roi_name is not None and highest_sim >= -1:
                    matching_data.append([control_name, [(highest_sim_roi_name, highest_sim_position_factor)]])
                
                else:
                    print(f"WARNING: No match found for control sample {control_name}")


    used_roi_names = {roi_name for _, list in matching_data for roi_name, _ in list}
    skipped_roi_names = checked_roi_names - used_roi_names

    used_control_names = {ctrl_name for ctrl_name, _ in matching_data}
    skipped_control_names = checked_control_names - used_control_names


    print("\n---------- Matching Summary ----------")
    print(f"\n{len(matching_data)} fused pairs.")

    if skipped_roi_names:
        print(f"{len(skipped_roi_names)} skipped rois:")
        for name in sorted(skipped_roi_names):
            print(f"\t{name}")
    else:
        print("No skipped rois.")

    if skipped_control_names:
        print(f"{len(skipped_control_names)} skipped controls:")
        for name in sorted(skipped_control_names):
            print(f"\t{name}")
    else:
        print("No skipped controls.")

    return matching_data



def combine_binary_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    mode: str = "or",
    return_dtype=None,
) -> np.ndarray:
    """
    Combine two binary masks (NumPy arrays) with shape (C,H,W) or (C,D,H,W).

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        Binary masks of identical shape. Values may be boolean or {0,1}.
        Any non-zero value is treated as True.
    mode : str
        "or"        : union (A ∪ B)
        "and"       : intersection (A ∩ B)
        "xor"       : exclusive OR
        "a_minus_b" : A \\ B
        "b_minus_a" : B \\ A
    return_dtype : dtype or None
        None => return dtype like mask_a, otherwise cast to this dtype (e.g. np.uint8 or bool).

    Returns
    -------
    np.ndarray
        Combined mask with the same shape as the inputs.
    """
    # Basic type checks
    if not isinstance(mask_a, np.ndarray) or not isinstance(mask_b, np.ndarray):
        raise TypeError("mask_a and mask_b must be NumPy arrays.")

    # Require identical shapes
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shapes must match, got {mask_a.shape} vs {mask_b.shape}.")

    # Only allow (C,H,W) or (C,D,H,W)
    if mask_a.ndim not in (3, 4):
        raise ValueError(
            f"Expected ndim 3 or 4 for (C,H,W) or (C,D,H,W), got ndim={mask_a.ndim}."
        )

    # Interpret any non-zero as True (robust to uint8/float/etc.)
    a = mask_a.astype(bool, copy=False)
    b = mask_b.astype(bool, copy=False)

    # Select combination operation
    mode = mode.lower()
    if mode == "or":
        out = np.logical_or(a, b)
    elif mode == "and":
        out = np.logical_and(a, b)
    elif mode == "xor":
        out = np.logical_xor(a, b)
    elif mode in ("a_minus_b", "a-b", "a\\b"):
        out = np.logical_and(a, np.logical_not(b))
    elif mode in ("b_minus_a", "b-a", "b\\a"):
        out = np.logical_and(b, np.logical_not(a))
    else:
        raise ValueError(
            f"Unknown mode='{mode}'. Supported: or/and/xor/a_minus_b/b_minus_a."
        )


    out = np.where(out > 0, 1, 0)
    return out
    # Control output dtype
    if return_dtype is None:
        # Return as bool if original was bool, otherwise cast to mask_a dtype
        if mask_a.dtype == bool:
            return out
        return out.astype(mask_a.dtype, copy=False)
    else:
        return out.astype(return_dtype, copy=False)

