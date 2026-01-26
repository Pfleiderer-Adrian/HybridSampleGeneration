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


def create_matching_dictionary(control_sample_dataloader, roi_dataloader, config, matching_routine="local", anomaly_duplicates=False):

    # template_output_dir übergeben, wenn templates_path in config noch nicht überschrieben wurde (also die templates noch nicht generiert wurden)
    allowed_matchings_routines = ["local", "global", "fixed_from_extraction"]
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
    # Routine: fixed_from_extraction (sequential pairing, centroid from metadata)
    # ------------------------------------------------------------
    if matching_routine == "fixed_from_extraction":
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
                [control_filename, roi_filename, centroid]
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

                matching_data.append([control_filename, roi_filename, highest_sim_position_factor])


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
                    matching_data.append([control_name, highest_sim_roi_name, highest_sim_position_factor])
                
                else:
                    print(f"WARNING: No match found for control sample {control_name}")


    used_roi_names = {roi_name for _, roi_name, _ in matching_data}
    skipped_roi_names = checked_roi_names - used_roi_names

    used_control_names = {ctrl_name for ctrl_name, _, _ in matching_data}
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
