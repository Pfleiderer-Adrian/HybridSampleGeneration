import numpy as np
from tqdm import tqdm
from skimage.feature import match_template


def create_matching_dict2d(control_sample_dataloader, roi_dataloader, config, matching_routine="local", anomaly_duplicates=False):
    """
    Create a list of matchings between control samples and ROI anomaly samples (2D only).

    Matching output format (rows):
      [control_filename, roi_filename, position_factor]

    Where:
      - control_filename: basename from the control dataloader
      - roi_filename: basename from ROI dataset
      - position_factor: list[float] normalized to control *image* shape (H,W)
        - for "local"/"global": derived from template matching center (row,col) -> [row/H, col/W]
        - for "fixed_from_extraction": taken from config.syn_anomaly_transformations[roi]["centroid_norm"] (len 2)

    Inputs
    ------
    control_sample_dataloader:
        Iterable yielding (control, _, control_filename)
        where control is typically a numpy array image and _ is ignored here.
        Expected control shape: (C,H,W) (channel-first). If C>1, max-projection is used for matching.
    roi_dataloader:
        Iterable/dataset yielding (roi, roi_filename) (often AnomalyDataset2D with numpy_mode=True).
        Expected roi shape: (C,h,w) (channel-first). If C>1, max-projection is used for matching.
    config:
        Configuration containing syn_anomaly_transformations (needed for fixed_from_extraction).
    matching_routine:
        One of: "local", "global", "fixed_from_extraction"
          - "local": match ith control with ith ROI (sequential) and compute best template position (resource middle)
          - "global": for each control, search over all ROI and pick best similarity (resource heavy)
          - "fixed_from_extraction": sequential mapping, take centroid from extraction metadata (resource lite)

    Outputs
    -------
    matching_data:
        list[list]
        Each row: [control_filename, roi_filename, position_factor]

    Notes
    -----
    - The "local" routine as implemented uses ROI i for control i (no global best search).
    - Position factors are **spatial only** in 2D: [y/H, x/W].
    """
    # template_output_dir übergeben, wenn templates_path in config noch nicht überschrieben wurde (also die templates noch nicht generiert wurden)
    allowed_matchings_routines = ["local", "global", "fixed_from_extraction"]
    if matching_routine not in allowed_matchings_routines:
        raise ValueError("Not a allowed matching routine.")
        # Rows to be written later to a CSV by the caller
    matching_data = []

    # Tracks ROI filenames already used (only relevant for "global")
    excluded_roi_sample_names = []

    # ------------------------------------------------------------
    # Routine: fixed_from_extraction (sequential pairing, centroid from metadata)
    # ------------------------------------------------------------
    if matching_routine == "fixed_from_extraction":
        i = 0
        for control, _, control_filename, *ignored in tqdm(control_sample_dataloader):
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
                i += 1

            if roi_filename is None:
                centroid = None
            else:
                centroid = config.syn_anomaly_transformations[roi_filename]["centroid_norm"]

            matching_data.append(
                [control_filename, roi_filename, centroid]
            )

    # ------------------------------------------------------------
    # Routine: local (sequential ROI assignment, per-control template matching)
    # ------------------------------------------------------------
    if matching_routine == "local":
        i = 0
        skipped_rois = {}   # want no duplicates here
        for control, _, control_filename, *ignored in tqdm(control_sample_dataloader):
            highest_sim_position_factor = None

            # always check previously skipped rois first for new control
            for roi_filename, roi in skipped_rois.items():
                sim, opt_center = template_matching(roi, control)
                if sim >= -1:
                    spatial_shape = np.array(control.shape[-2:], dtype=float)
                    highest_sim_position_factor = (
                        (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                    )
                    matching_data.append([control_filename, roi_filename, highest_sim_position_factor])
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
                        i += 1
                        
                        sim, opt_center = template_matching(roi, control)

                    # no match possible for current control
                    if sim < -1:
                        continue

                    # found match
                    # Convert center (row,col) to normalized position factor (y/H, x/W).
                    # NOTE: We normalize by the spatial shape only (H,W); channel C is ignored.
                    spatial_shape = np.array(control.shape[-2:], dtype=float)
                    highest_sim_position_factor = (
                        (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                    )

                matching_data.append([control_filename, roi_filename, highest_sim_position_factor])

    # ------------------------------------------------------------
    # Routine: global (search best ROI for each control; avoid reusing ROI)
    # ------------------------------------------------------------
    if matching_routine == "global":
        skipped_controls = []
        for control, _, control_filename, *ignored in control_sample_dataloader:
            highest_sim = -np.inf
            highest_sim_roi_name = None
            highest_sim_position_factor = None

            for roi, roi_filename in tqdm(roi_dataloader):
                if roi_filename in excluded_roi_sample_names:
                    continue
                if anomaly_duplicates and len(excluded_roi_sample_names) == roi_dataloader.__len__():
                    excluded_roi_sample_names = []

                sim, opt_center = template_matching(roi, control)

                if sim > highest_sim:
                    highest_sim = sim
                    highest_sim_roi_name = roi_filename

                    # Normalize by spatial shape only (H,W).
                    spatial_shape = np.array(control.shape[-2:], dtype=float)
                    highest_sim_position_factor = (
                        (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                    )

            if highest_sim_roi_name is None:
                raise RuntimeError(f"No match found for {control_filename}")
            
            # no matching template in remaining rois
            if highest_sim < -1:
                skipped_controls.append((control, control_filename))
                continue

            matching_data.append([control_filename, highest_sim_roi_name, highest_sim_position_factor])
            excluded_roi_sample_names.append(highest_sim_roi_name)

        if anomaly_duplicates and skipped_controls:
            # try to find matches for skipped controls
            for control, control_name in tqdm(skipped_controls):
                # try all rois (no exclusion)
                highest_sim = -np.inf
                highest_sim_roi_name = None
                highest_sim_position_factor = None

                for roi, roi_filename in roi_dataloader:
                    sim, opt_center = template_matching(roi, control)
                    if sim > highest_sim:
                        highest_sim = sim
                        highest_sim_roi_name = roi_filename
                        spatial_shape = np.array(control.shape[-2:], dtype=float)
                        highest_sim_position_factor = (
                            (np.array(opt_center, dtype=float) / spatial_shape).tolist()
                        )                    
                if highest_sim_roi_name is not None and highest_sim >= -1:
                    matching_data.append([control_name, highest_sim_roi_name, highest_sim_position_factor])

    return matching_data


def _to_2d_spatial(arr: np.ndarray) -> np.ndarray:
    """
    Convert an array to a 2D spatial image (H,W) for template matching.

    Supported inputs:
      - (C,H,W): channel-first image -> returns max-projection over C (or arr[0] if C==1)
      - (H,W): already 2D -> returned as-is

    Notes
    -----
    - `skimage.feature.match_template` works with 2D arrays for this use case.
    - If you want true multi-channel matching, you need a different strategy (e.g. per-channel matching and aggregation).
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        # (C,H,W)
        if arr.shape[0] == 1:
            return arr[0]
        return np.max(arr, axis=0)

    if arr.ndim == 2:
        return arr

    raise ValueError(f"Expected (H,W) or (C,H,W), got {arr.shape}")


def template_matching(template, control):
    """
    2D template matching using `skimage.feature.match_template`.

    This performs 2D template matching to find the best match position
    of the ROI template inside the control image.

    Inputs
    ------
    template:
        np.ndarray, expected shapes:
          - (C,h,w) or (h,w)
    control:
        np.ndarray, expected shapes:
          - (C,H,W) or (H,W)

    Outputs
    -------
    similarity_score:
        float
        Maximum match score (higher is better).
    center:
        tuple[int, int]
        (center_row, center_col) of the best match position in control coordinates.

    Notes
    -----
    - `match_template` expects `control` to be >= template shape in each dimension.
    - If arrays contain a channel dimension (C>1), this implementation uses max-projection across channels.
      For C==1, it uses the single channel directly.
    """
    template2d = _to_2d_spatial(template)
    control2d = _to_2d_spatial(control)

    # Check if template fits in control sample
    if any(t_dim > c_dim for t_dim, c_dim in zip(template.shape, control.shape)):
        return -2, None

    # Compute correlation map (same dimensionality as control minus template extents)
    result = match_template(control2d, template2d)

    # Best similarity score is max of correlation map
    similarity_score = float(np.max(result))

    # Index of best match corresponds to the template's top-left corner position
    idx_tuple = np.unravel_index(np.argmax(result), result.shape)
    top_left_row, top_left_col = idx_tuple

    # Template shape
    y, x = template2d.shape

    # Convert top-left corner to center coordinate
    center_row = top_left_row + (y // 2)
    center_col = top_left_col + (x // 2)

    return similarity_score, (center_row, center_col)
