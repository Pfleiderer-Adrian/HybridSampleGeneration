import numpy as np
from tqdm import tqdm
from skimage.feature import match_template


def create_matching_dict3d(control_sample_dataloader, roi_dataloader, config, matching_routine="local", anomaly_duplicates=False):
    """
    Create a list of matchings between control samples and ROI anomaly samples.

    Matching output format (rows):
      [control_filename, roi_filename, position_factor]

    Where:
      - control_filename: basename from the control dataloader
      - roi_filename: basename from ROI dataset
      - position_factor: list[float] normalized to control volume shape
        - for "local"/"global": derived from template matching center
        - for "fixed_from_extraction": taken from config.syn_anomaly_transformations[roi]["centroid"]

    Inputs
    ------
    control_sample_dataloader:
        Iterable yielding (control, _, control_filename)
        where control is typically a numpy array volume and _ is ignored here.
    roi_dataloader:
        Iterable/dataset yielding (roi, roi_filename) (often AnomalyDataset3D with numpy_mode=True).
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

            # In your Anomaly_Extraction3D.py metadata, you used "centroid_voxel" and "centroid_norm".
            # Ensure this matches your actual stored metadata keys.
            matching_data.append(
                [control_filename, roi_filename, centroid]
            )

    # ------------------------------------------------------------
    # Routine: local (sequential ROI assignment, per-control template matching)
    # ------------------------------------------------------------
    if matching_routine == "local":
        i = 0
        for control, _, control_filename, *ignored in tqdm(control_sample_dataloader):
            highest_sim_position_factor = None

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

            if roi_filename is not None:
                # Only compute match if ROI not excluded (list is empty in this routine by default)
                if roi_filename not in excluded_roi_sample_names:
                    sim, opt_center = template_matching(roi, control)

                    # Convert center (slice,row,col) to normalized position factor.
                    # NOTE: The original code prepends (1,) then divides by control.shape.
                    # This implies control.shape likely includes a leading channel dimension.
                    highest_sim_position_factor = (
                        (np.array((1,) + opt_center) / np.array(control.shape)).astype(float).tolist()
                    )

                matching_data.append([control_filename, roi_filename, highest_sim_position_factor])

    # ------------------------------------------------------------
    # Routine: global (search best ROI for each control; avoid reusing ROI)
    # ------------------------------------------------------------
    if matching_routine == "global":
        for control, _, control_filename, *ignored in control_sample_dataloader:
            highest_sim = -np.inf
            highest_sim_roi_name = None
            highest_sim_position_factor = None

            for roi, roi_filename in tqdm(roi_dataloader):
                if roi_filename in excluded_roi_sample_names:
                    continue

                sim, opt_center = template_matching(roi, control)

                if sim > highest_sim:
                    highest_sim = sim
                    highest_sim_roi_name = roi_filename
                    highest_sim_position_factor = (
                        (np.array((1,) + opt_center) / np.array(control.shape)).astype(float).tolist()
                    )
            
            if highest_sim_roi_name is None:
                raise RuntimeError(f"No match found for {control_filename}")

            matching_data.append([control_filename, highest_sim_roi_name, highest_sim_position_factor])
            excluded_roi_sample_names.append(highest_sim_roi_name)

    return matching_data


def template_matching(template, control):
    """
    Template matching using `skimage.feature.match_template`.

    This performs 3D template matching (after squeezing singleton dims) to find the best match position
    of the ROI template inside the control volume.

    Inputs
    ------
    template:
        np.ndarray, expected to become shape (D, H, W) after squeeze.
    control:
        np.ndarray, expected to become shape (D, H, W) after squeeze.

    Outputs
    -------
    similarity_score:
        float
        Maximum match score (higher is better).
    center:
        tuple[int, int, int]
        (center_slice, center_row, center_col) of the best match position in control coordinates.

    Notes
    -----
    - `match_template` expects `control` to be >= template shape in each dimension.
    - If arrays still contain a channel dimension, `np.squeeze` removes it (assuming C==1).
      For multi-channel inputs, this needs adaptation.
    """
    # Remove singleton dimensions (commonly removes channel dimension when C==1)
    template = np.squeeze(template)
    control = np.squeeze(control)

    # Compute correlation map (same dimensionality as control minus template extents)
    result = match_template(control, template)

    # Best similarity score is max of correlation map
    similarity_score = float(np.max(result))

    # Index of best match corresponds to the template's top-left-front corner position
    idx_tuple = np.unravel_index(np.argmax(result), result.shape)
    top_left_slice, top_left_row, top_left_col = idx_tuple

    # Template shape
    z, y, x = template.shape

    # Convert top-left corner to center coordinate
    center_slice = top_left_slice + (z // 2)
    center_row = top_left_row + (y // 2)
    center_col = top_left_col + (x // 2)

    return similarity_score, (center_slice, center_row, center_col)
