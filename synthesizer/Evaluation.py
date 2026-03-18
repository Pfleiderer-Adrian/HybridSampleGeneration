import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.stats import entropy
from scipy.ndimage import center_of_mass
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


from synthesizer.functions_2D.Anomaly_Extraction2D import crop_and_center_anomaly_2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_and_center_anomaly_3d
from data_handler.AnomalyDataset import save_numpy_as_npy
from data_handler.Visualizer import OutlierGUI


def compare_intensity_range(sample_dataloader, synth_dir):

    global_min = np.inf
    global_max = -np.inf
    for arr, seg, basename in sample_dataloader:
        scan_min = arr.min()
        scan_max = arr.max()
        global_min = round(min(global_min, scan_min), 2)
        global_max = round(max(global_max, scan_max), 2)
    real_range_min, real_range_max = float(global_min), float(global_max)

    if not os.path.exists(synth_dir):
        print("Directory doesn't exist.")
        return
    global_min = np.inf
    global_max = -np.inf
    for f in Path(synth_dir).rglob("*.npy"):
        arr = np.load(f, mmap_mode="r")
        scan_min = arr.min()
        scan_max = arr.max()
        global_min = round(min(global_min, scan_min), 2)
        global_max = round(max(global_max, scan_max), 2)
    synth_range_min, synth_range_max = float(global_min), float(global_max)


    print(f"Intensity range of real data with anomalies: ({real_range_min}, {real_range_max})")
    print(f"Intensity range of synthetic data with anomalies: ({synth_range_min}, {synth_range_max})")



# -- local analysis --
# glcm

def compute_glcm(volume, mask, levels=32):

    glcm_total = np.zeros((levels, levels), dtype=np.float32)
    for c in range(volume.shape[0]):
        # mask can be with or without channel
        current_mask = mask[c] if mask.ndim == volume.ndim else mask
        glcm_total += _compute_glcm_for_channel(volume[c], current_mask, levels) # one glcm computation per channel

    # norm
    total_sum = glcm_total.sum()
    if total_sum > 0:
        glcm_total /= total_sum
    return glcm_total


def _compute_glcm_for_channel(volume, mask, levels=32):

    lesion = volume[mask]
    if lesion.size == 0:
        print("Mask empty. GLCM is 0.")
        return np.zeros((levels, levels), dtype=np.float32)

    # quantization
    vol_min = lesion.min()
    vol_max = lesion.max()
    delta = vol_max - vol_min
    
    if delta == 0:
        volume_q = np.zeros_like(volume, dtype=int)
    else:
        volume_q = ((volume - vol_min) / delta * (levels - 1)).astype(int)
        volume_q = np.clip(volume_q, 0, levels - 1)
    
    # vectorized computation
    glcm = np.zeros((levels, levels), dtype=np.float32)

    if volume.ndim == 2:
        displacements = [(0, 1), (1, 1), (1, 0), (1, -1)]
    
    elif volume.ndim == 3:
        displacements = [(1,0,1), (0,0,1), (-1,0,1), (1,1,1), (0,1,1), (-1,1,1), 
                        (1,1,0), (0,1,0), (-1,1,0), (1,1,-1), (0,1,-1), (-1,1,-1), (1,0,0)]
        
    else:
        raise ValueError(f"Data must be 2D or 3D without channel. (here {volume.ndim})")
    
    # bool mask for fast indexing
    mask_bool = (mask >= 1) # only works for 0, 1 masks

    for displacement in displacements:
        # volume[0:-1] vs volume[1:] compares i with i+1 without loop
        slices_src = []
        slices_dst = []
        
        for d in displacement:
            if d == 0:
                slices_src.append(slice(None))
                slices_dst.append(slice(None))
            elif d > 0:
                slices_src.append(slice(0, -d))
                slices_dst.append(slice(d, None))
            else: # d < 0
                slices_src.append(slice(-d, None))
                slices_dst.append(slice(0, d))
        
        slices_src = tuple(slices_src)
        slices_dst = tuple(slices_dst)

        # cut out shifted array
        src_vals = volume_q[slices_src]
        dst_vals = volume_q[slices_dst]
        
        # cut out shifted mask
        src_mask = mask_bool[slices_src]
        dst_mask = mask_bool[slices_dst]
        
        # only where pixel/voxel and neighbor are both in mask
        valid_pairs = src_mask & dst_mask
        
        if not np.any(valid_pairs):
            continue
            
        # value pairs
        i_vals = src_vals[valid_pairs]
        j_vals = dst_vals[valid_pairs]
        
        # fill glcm (symmetric)
        np.add.at(glcm, (i_vals, j_vals), 1)
        np.add.at(glcm, (j_vals, i_vals), 1)
    
    return glcm


def glcm_features(glcm, roi=False):
    levels = glcm.shape[0]
    i = np.arange(levels).reshape((-1,1))   # col vector
    j = np.arange(levels).reshape((1,-1))   # row vector
    
    # Contrast
    contrast = np.sum((i-j)**2 * glcm)
    
    # Homogeneity
    homogeneity = np.sum(glcm / (1.0 + (i-j)**2))
    
    # Energy
    energy = np.sqrt(np.sum(glcm**2))
    
    # Correlation
    mean_i = np.sum(i * glcm)   
    mean_j = np.sum(j * glcm)
    std_i = np.sqrt(np.sum((i - mean_i)**2 * glcm))    
    std_j = np.sqrt(np.sum((j - mean_j)**2 * glcm))
    if std_i * std_j == 0:
        correlation = 1.0
    else:
        correlation = np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
    
    if roi:
        return {
            "roi_Contrast": contrast,
            "roi_Homogeneity": homogeneity,
            "roi_Energy": energy,
            "roi_Correlation": correlation
        }
        
    return {
        "Contrast": contrast,
        "Homogeneity": homogeneity,
        "Energy": energy,
        "Correlation": correlation
    }

def get_glcm_feature_diffs(real_arr, real_mask, synth_arr, synth_mask):
    """
    computes glcm features for real and synth and their differences
    """

    real_features = glcm_features(compute_glcm(real_arr, real_mask))
    synth_features = glcm_features(compute_glcm(synth_arr, synth_mask))
    
    diff_features = {}

    for name in real_features.keys():
        if name in synth_features:
            diff_features[name] = abs(real_features[name] - synth_features[name])
    
    return real_features, synth_features, diff_features

def get_glcm_roi_feature_diffs(real_arr, real_mask, synth_arr, synth_mask):
    """
    computes glcm features for real and synth and their differences (roi)
    """

    real_features = glcm_features(compute_glcm(real_arr, real_mask), roi=True)
    synth_features = glcm_features(compute_glcm(synth_arr, synth_mask), roi=True)
    
    diff_features = {}

    for name in real_features.keys():
        if name in synth_features:
            diff_features[name] = abs(real_features[name] - synth_features[name])
    
    return real_features, synth_features, diff_features


# center of mass and volume
def get_volume_feature_diffs(real_anomaly, real_mask, synth_anomaly, synth_mask):
    
    mask_dim = real_mask.ndim
    if mask_dim == real_anomaly.ndim:
        # mask has channel dim
        real_mask = np.max(real_mask, axis=0)
        mask_dim -=1

    if synth_mask.ndim == real_anomaly.ndim:
        # synth mask has channel dim
        synth_mask = np.max(synth_mask, axis=0)


    real_vol = np.sum(real_mask).astype(np.int64)
    synth_vol = np.sum(synth_mask).astype(np.int64)

    real_center = center_of_mass(real_mask)    # use anomaly instead of mask, if center should be influenced by anomaly intensity
    synth_center = center_of_mass(synth_mask)

    if mask_dim == 2:
        real_features = {
            "Volume": real_vol,
            "H-center": real_center[0],
            "W-center": real_center[1]
        }
        
        synth_features = {
            "Volume": synth_vol,
            "H-center": synth_center[0],
            "W-center": synth_center[1]
        }
        
        diff_features = {
            "Volume": abs(real_vol - synth_vol),
            "H-center": abs(real_center[0] - synth_center[0]),
            "W-center": abs(real_center[1] - synth_center[1])
        }

    elif mask_dim == 3:
        real_features = {
            "Volume": real_vol,
            "D-center": real_center[0],
            "H-center": real_center[1],
            "W-center": real_center[2]
        }
        
        synth_features = {
            "Volume": synth_vol,
            "D-center": synth_center[0],
            "H-center": synth_center[1],
            "W-center": synth_center[2]
        }
        
        diff_features = {
            "Volume": abs(real_vol - synth_vol),
            "D-center": abs(real_center[0] - synth_center[0]),
            "H-center": abs(real_center[1] - synth_center[1]),
            "W-center": abs(real_center[2] - synth_center[2])
        }

    else:
        raise ValueError(f"Data must be 2D or 3D without channel. (here {mask_dim})")

    return real_features, synth_features, diff_features


def save_difference_histograms(differences_dict, save_path):
    """
    creates one hist for every metric in differenes_dict and saves it to save_path as png.
    """
    num_features = len(differences_dict)
    
    cols = 2
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
    axes = axes.flatten()
    
    for i, (name, diff_list) in enumerate(differences_dict.items()):
        if not diff_list:
            print(f"Found no data for histogram {name}.")
            continue
            
        ax = axes[i]
        bins = 32
        ax.hist(diff_list, bins, edgecolor='black', alpha=0.7)
        ax.set_title(f'Difference Histogram: {name}')
        ax.set_xlabel('Absolute Difference (Real - Synthetic)')
        ax.set_ylabel('# Pairs')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nSaved histograms: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def get_overlapping_samples(outliers_data):
    
    sample_tracker = defaultdict(list)
    
    for metric_name, sample_list in outliers_data.items():
        for entry in sample_list:
            sample_name = entry["sample"]
            # append metric to this samples list
            sample_tracker[sample_name].append(metric_name)
            
    overlapping_samples = {
        sample: metrics for sample, metrics in sample_tracker.items() if len(metrics) > 1
    }
    return overlapping_samples


def print_overlap_summary(overlaps_list):
    
    sample_to_metrics = {}
    
    for metric_group in overlaps_list:
        # metric_group {'Contrast': [...], 'Homogeneity': [...]}
        for metric_name, occurrences in metric_group.items():
            # occurrences:  {'value': ..., 'sample': ...}
            for entry in occurrences:
                sample_name = entry['sample']
                
                if sample_name not in sample_to_metrics:
                    sample_to_metrics[sample_name] = set()
                
                sample_to_metrics[sample_name].add(metric_name)

    stats = {}
    for metrics_set in sample_to_metrics.values():
        n = len(metrics_set)
        if n > 0:
            stats[n] = stats.get(n, 0) + 1

    print("- Summary of Overlaps -")
    if not stats:
        print("Found no overlaps.")
    else:
        for n in sorted(stats.keys(), reverse=True):
            count = stats[n]
            label = "sample" if count == 1 else "samples"
            label2 = "metric" if n == 1 else "metrics"
            print(f"outlier in {n} {label2}: {count} {label}")


def analyze_results(results):
    
    if results is None or results["sample_counter"] == 0:
        print("No results.")
        return

    mean_real = results["mean_real"]
    mean_synth = results["mean_synth"]
    outliers = results["outliers"]

    print("\n---------- REAL DATA (Means) ----------")
    for name, value in mean_real.items():
        print(f"{name}: {value:.4f}")

    print("\n---------- SYNTHETIC DATA (Means) ----------")
    for name, value in mean_synth.items():
        print(f"{name}: {value:.4f}")
        
    # Outlier
    print(f"\n ---------- Metric difference outliers ----------")
    for name, top_list in outliers.items():
        list_len = len(top_list)
        if list_len == 1:
            print(f"{name} -> 1 outlier")
        else:
            print(f"{name} -> {list_len} outliers")
    

def run_feature_calculator(anomaly_dir, synth_anomaly_dir, feature_calculator_func, config, cutout=True):
    """
    if cutout=True create mask (anomaly > bg_thresh or 0), else take whole images (for roi)
    """
    sample_counter = 0
    real_totals = {}
    synth_totals = {}
    feature_differences = {}

    all_diffs_and_sample = {}
    diff_outliers = {}
    # smallest_diffs = {}

    csv_rows = []
    csv_path = os.path.join(config.study_folder, "evaluation_results", "metric_diffs.csv")

    grouped_roi_data = {}

    for f in Path(synth_anomaly_dir).rglob("*.npy"):    # also checks all sub folders (for roi)
        anomaly_name = f.name
        real_file_path = os.path.join(anomaly_dir, anomaly_name)    # same name matching
        if not os.path.exists(real_file_path):
            print(f"Warning: Missing synth file {real_file_path}. Skipping.")
            continue
        
        try:
            synth_arr = np.load(f)
            real_arr = np.load(real_file_path)
        except Exception as e:
            print(f"Error trying to load {f} and {real_file_path}: {e}. Skipping.")
            continue

        if cutout:
            if config.background_threshold:
                real_mask = real_arr > config.background_threshold
                synth_mask = synth_arr > config.background_threshold
            else:
                real_mask = real_arr > 0
                synth_mask = synth_arr > 0
        else: # roi
            real_mask = np.ones_like(real_arr, dtype=bool)
            synth_mask = np.ones_like(synth_arr, dtype=bool)

        sample_counter += 1

        real_features, synth_features, diff_features = feature_calculator_func(real_arr, real_mask, synth_arr, synth_mask)

        if isinstance(diff_features, dict):
            diff_features_serializable = {k: (v.item() if hasattr(v, 'item') else v) for k, v in diff_features.items()}
        else:
            diff_features_serializable = diff_features

        if cutout:
            csv_rows.append({
                "sample_name": anomaly_name,
                "feature_calculator": feature_calculator_func.__name__,
                "metric_diffs": json.dumps(diff_features_serializable)
            })
        else:   # use control name as key for ROI metrics (and anomaly name as sub key)
            control_name = f.parent.name
            if control_name not in grouped_roi_data:
                grouped_roi_data[control_name] = {}
            grouped_roi_data[control_name][anomaly_name] = diff_features_serializable

        for name, diff in diff_features.items():
            # if name not in real_features or name not in synth_features:
            #     continue

            real_val = real_features[name]
            synth_val = synth_features[name]
            
            real_totals[name] = real_totals.get(name, 0.0) + real_val
            synth_totals[name] = synth_totals.get(name, 0.0) + synth_val
            
            if name not in feature_differences:
                feature_differences[name] = []
            feature_differences[name].append(diff)
            
            if name not in all_diffs_and_sample:
                all_diffs_and_sample[name] = []
            relative_path = str(f.relative_to(synth_anomaly_dir))
            new_entry = {"value": diff, "sample": relative_path}
            all_diffs_and_sample[name].append(new_entry)
        
    for control_name, anomalies_dict in grouped_roi_data.items():
        csv_rows.append({
            "sample_name": control_name,
            "feature_calculator": "roi_" + feature_calculator_func.__name__,
            "differences_dict": json.dumps(anomalies_dict)
        })

    if sample_counter == 0:
        print("WARNING: Found no pairs. Return.")
        return None
    
    print(f"Calculation done. Analysed {sample_counter} pairs.")
    
    feature_names_list = list(real_totals.keys())

    mean_real_features = {name: real_totals[name] / sample_counter for name in feature_names_list}
    mean_synth_features = {name: synth_totals[name] / sample_counter for name in feature_names_list}

    for name, all_values_dicts in all_diffs_and_sample.items():
        # smallest_diffs (smallest first)
        # sorted_by_diff = sorted(all_values_dicts, key=lambda x: x["value"])
        # smallest_diffs[name] = sorted_by_diff[:config.n_smallest_diffs]
        
        # Outlier
        raw_values = feature_differences[name]
        if not raw_values:
            diff_outliers[name] = []
            continue

        q1, q3 = np.percentile(raw_values, [25, 75])
        iqr = q3 - q1
        lower_threshold = q1 - (1.5 * iqr)
        upper_threshold = q3 + (1.5 * iqr)

        custom = config.custom_outlier_thresholds.get(name)
        if custom:
            custom_min = custom.get("min")
            custom_max = custom.get("max")
            if custom_min is not None:
                lower_threshold = custom_min
            if custom_max is not None:
                upper_threshold = custom_max

        outliers_for_metric = []
        for sample in all_values_dicts:
            if sample["value"] > upper_threshold or sample["value"] < lower_threshold:
                outliers_for_metric.append(sample)

        # biggest first
        outliers_for_metric.sort(key=lambda x: abs(x["value"]), reverse=True)
        diff_outliers[name] = outliers_for_metric

    if csv_rows:
        df_new = pd.DataFrame(csv_rows)
        file_exists = os.path.isfile(csv_path)
        
        # creates "metric_diffs.csv" in directory "evaluation_results"
        df_new.to_csv(csv_path, mode='a', index=False, header=not file_exists)

    return {
        "sample_counter": sample_counter,
        "mean_real": mean_real_features,
        "mean_synth": mean_synth_features,
        "outliers": diff_outliers,
        # "smallest_diffs": smallest_diffs,
        "all_diffs": feature_differences
    }


def evaluation_pipeline(sample_dataloader, config):

    study_folder = config.study_folder
    eval_results_folder = os.path.join(study_folder, "evaluation_results")
    os.makedirs(eval_results_folder, exist_ok=True)

    csv_path = os.path.join(eval_results_folder, "metric_diffs.csv")
    
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Old results file {csv_path} deleted.")
    
    ghs_dir = os.path.join(study_folder, "generated_hybrid_samples")
    ghs_img_dir = os.path.join(ghs_dir, "images_npy")
    ghs_seg_dir = os.path.join(ghs_dir, "segmentations_npy")

    # print("--- Global comparison ---")
    # if config.global_intensity_check:
    #     print("- Computing histograms -")
    #     compare_hist_and_metrics(img_dir, ghs_img_dir)
    
    print("\n- Computing intensity ranges -")
    compare_intensity_range(sample_dataloader, ghs_img_dir)

    print("\n--- Compare anomaly cutouts ---")
    print("- Computing glcm on cutouts -")
    anomaly_dir = os.path.join(study_folder, "anomaly_data")
    synth_anomaly_dir = os.path.join(study_folder, "synth_anomaly_data")

    glcm_cutout_results = run_feature_calculator(anomaly_dir, synth_anomaly_dir, get_glcm_feature_diffs, config, cutout=True)
    glcm_cutout_hist_path = os.path.join(eval_results_folder, "glcm_cutout_difference_histograms.png")
    analyze_results(glcm_cutout_results)
    save_difference_histograms(glcm_cutout_results["all_diffs"], glcm_cutout_hist_path)

    print("\n- Computing volume features on cutouts -")
    volume_cutout_results = run_feature_calculator(anomaly_dir, synth_anomaly_dir, get_volume_feature_diffs, config, cutout=True)
    volume_cutout_hist_path = os.path.join(eval_results_folder, "volume_cutout_difference_histograms.png")
    analyze_results(volume_cutout_results)
    save_difference_histograms(volume_cutout_results["all_diffs"], volume_cutout_hist_path)

    print("\n--- Compare ROIs ---")

    print("- Computing glcm on ROIs (VAE input-output pair) -")
    roi_dir = os.path.join(study_folder, "anomaly_roi_data")
    synth_roi_dir = os.path.join(study_folder, "synth_roi_data") 

    glcm_roi_results = run_feature_calculator(roi_dir, synth_roi_dir, get_glcm_roi_feature_diffs, config, cutout=False)
    glcm_roi_hist_path = os.path.join(eval_results_folder, "glcm_roi_difference_histograms.png")
    analyze_results(glcm_roi_results)
    save_difference_histograms(glcm_roi_results["all_diffs"], glcm_roi_hist_path)


    print("\n--- Examining Outliers ---")
    overlaps_list = [glcm_cutout_results["outliers"], volume_cutout_results["outliers"], glcm_roi_results["outliers"]]
    print_overlap_summary(overlaps_list)

    auto_remove_outliers(overlaps_list, config)

def auto_remove_outliers(overlaps_list, config):
    """
    Deletes statistical outliers.
    Only deletes synth_anomaly for anomaly outliers and synth_roi for roi outliers.
    """
    anomaly_outlier_paths = set()
    roi_outlier_paths = set()

    for group in overlaps_list:
        for metric, items in group.items():
            is_roi = "roi" in metric.lower()
            for item in items:
                path = item['sample']
                if is_roi:
                    roi_outlier_paths.add(path)
                else:
                    anomaly_outlier_paths.add(path)

    if not anomaly_outlier_paths and not roi_outlier_paths:
        print("No outliers found.")
        return False

    synth_anomaly_dir = os.path.join(config.study_folder, "synth_anomaly_data")
    synth_roi_dir = os.path.join(config.study_folder, "synth_roi_data")

    def count_npy_files(directory):
        if not os.path.exists(directory):
            return 0
        return sum(len([f for f in files if f.endswith('.npy')]) for _, _, files in os.walk(directory))

    total_anomalies = count_npy_files(synth_anomaly_dir)
    total_rois = count_npy_files(synth_roi_dir)

    files_to_delete = []
    
    for rel_path in anomaly_outlier_paths:
        full_path = os.path.join(synth_anomaly_dir, rel_path)
        if os.path.exists(full_path):
            files_to_delete.append(full_path)

    for rel_path in roi_outlier_paths:
        full_path = os.path.join(synth_roi_dir, rel_path)
        if os.path.exists(full_path):
            files_to_delete.append(full_path)

    if not files_to_delete:
        print("Outliers identified, but no corresponding files found on disk.")
        return False

    tk_root = tk.Tk()
    tk_root.withdraw() 
    tk_root.attributes('-topmost', True)

    message = (
        f"Outlier Statistics:\n"
        f"------------------------------\n"
        f"Anomalies: {len(anomaly_outlier_paths)} outliers found (Total: {total_anomalies})\n"
        f"ROIs: {len(roi_outlier_paths)} outliers found (Total: {total_rois})\n\n"
        f"Total files to be deleted: {len(files_to_delete)} (This only deletes the specific outlier files)\n\n"
        f"Do you want to proceed with the deletion?"
    )

    confirm = messagebox.askyesno(
        title="Delete Outliers",
        message=message,
        parent=tk_root
    )

    if confirm:
        deleted_count = 0
        for filepath in files_to_delete:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    deleted_count += 1
            except Exception as e:
                print(f"Error trying to remove {filepath}: {e}")
        
        messagebox.showinfo(title="Success", message=f"Successfully removed {deleted_count} files.", parent=tk_root)
        print(f"Auto-removed {deleted_count} files.")
    
    tk_root.destroy()
    return True
