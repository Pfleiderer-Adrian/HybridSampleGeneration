# Hybrid Sample Generation
![Python](https://img.shields.io/badge/Python-14354C?style=flat&logo=python&logoColor=green) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![DOI:AMLDS63918.2025.11159383](http://img.shields.io/badge/DOI-AMLDS63918.2025.11159383-B31B1B.svg)](https://doi.org/10.1109/AMLDS63918.2025.11159383)

This repository implements a hybrid sample generation pipeline for imaging data to extend existing training datasets.
The implementation is based on the IEEE paper: https://doi.org/10.1109/AMLDS63918.2025.11159383
.

It extracts real anomaly cutouts from labeled images/volumes, trains a **VAE Model**, generates
**synthetic anomaly cutouts**, matches them to control samples (images w/o anomalies) via ROI template matching, and finally
**fuses** synthetic anomalies into control images/volumes to produce hybrid data **and** a corresponding
segmentation mask.

Core modules:
- `synthesizer/HybridDataGenerator.py` — pipeline wrapper
- `synthesizer/Configuration.py` — configuration + metadata persistence
- `synthesizer/function_XD/Anomaly_Extraction.py` — anomaly + ROI cutouts
- `synthesizer/function_XD/MatchingXD.py` — matching (syn. anomaly x control image)
- `synthesizer/function_XD/FusionXD.py` — fusion (syn. anomaly x control image)
- `synthesizer/Trainer.py` — Optuna training loop
---

### How it works

✅ **Input**  
- Control samples: image/volume (Samples without anomalies)  
- Anomaly samples: image/volume + anomaly mask (Samples with anomalies)
- A Dataloader/Iterator for your dataset

✅ **Output**  
- Fused control volume containing inserted synthetic anomaly  
- Segmentation mask for the inserted anomaly

✅ **Supported shapes**
- 3D: `(C, D, H, W)`
- 2D: `(C, H, W)`

---

### Main Requirements

✅ **Packages:**
- `Python`
- `torch`
- `optuna`
- `numpy`, `scipy`, `pandas`
- small packages are listed in requirements.txt

> Exact versions depend on your CUDA / PyTorch setup.

✅ **Dataloader / Iterator:**

Implement a custom dataloader/iterator for your dataset (e.g., NIfTI .nii/.nii.gz files stored in a single folder).

Each iteration must yield:
- img_arr
- seg_arr
- basename

**and**
- img_arr.shape == seg_arr.shape
- Both arrays must be in channels-first format: (Channels, Depth, Height, Width) or (Channels, Height, Width)

---

### Usage
```python
    # create config
    config = Configuration("Study01", "VAE_ResNet_3D", anomaly_size=(1, 32, 96, 96))

    HDG = HybridDataGenerator(config)
    # 1) Extract anomaly cutouts + ROI cutouts from anomaly-labeled samples
    HDG.extract_anomalies(your_dataloader)
    # 1) Or load already extracted anomalies
    HDG.load_anomalies()

    # 2) Train generator via Optuna
    HDG.train_generator(no_of_trials=5)
    # 2) Or only load a trained model
    HDG.load_generator()

    # 3) Generate and load synthetic anomalies
    HDG.generate_synth_anomalies()
    # 3) Or load already generated synthetic anomalies
    HDG.load_synth_anomalies()

    # 4) Create matching between control samples and anomaly ROIs
    HDG.create_matching_dict(your_dataloader)
    # 4) Or load already created matching dict
    HDG.load_matching_dict()

    # 5) Generate new Hybrid Training Samples and synthetic ROIs
    img, seg = HDG.fusion_synth_anomalies(control_image, basename_of_image)

    # 6) Compute (textural and morphological) metric differences for real-synthetic pairs
    HDG.run_evaluation_pipeline(dataloader_samples_with_anomalies)

    # 7) Start Outlier Viewer for manual inspection of generated samples
    HDG.visualize_evaluation_results()

```

---

### Matching logic
  Pair control samples and synthetic anomalies for fusion and save the results in matching_dict.csv: control, [(anomaly, fusion_position), ...]

  Choose one of four possible matching_routines:

  Matching count parameters used by local and global:
  - fusions_per_control:
      Target number of synthetic ROIs that should be matched with one control sample.
      With fusions_per_control = 1, each control receives one anomaly if a valid match is found.
      With higher values, several non-overlapping anomalies can be fused into the same control sample.
  - max_fusions_per_control_deviation:
      Adds a random integer deviation in [-max_fusions_per_control_deviation, +max_fusions_per_control_deviation] to fusions_per_control for each control sample.
      The result is clipped to at least 1.
      Example: fusions_per_control = 2 and max_fusions_per_control_deviation = 1 produces 1, 2 or 3 requested fusions per control.
      Use 0 for a fixed number of fusions per control.

  Matching score parameters used by local and global:
  - matching_intensity_weight:
      Weight for the standard intensity-based template matching score. Default: 0.5.
  - matching_gradient_weight:
      Weight for the gradient-magnitude template matching score. Default: 0.5.
      The gradient score is computed from the spatial gradient magnitude of ROI and control sample and is only used when the gradient images contain usable structure.
      If the gradient score is not usable, matching falls back to the available score components.

  For local and global matching, the final similarity map is a weighted combination of intensity matching and gradient-magnitude matching.
  This helps prefer insertion positions that match both local brightness/texture and local edge structure.
  If a ROI is larger than the control sample in any spatial dimension, it is skipped and no template matching is computed for that pair.

  - local: (pairing not optimal, but fast)  
      `Fusion synthetic anomaly into a random control sample, finds best position within the sample.`  
      Iterates through every control sample and attempts to find the configured number of ROI matches.
      Always only checks the next ROI from the dataloader (until enough matches were found).
      Uses the combined intensity and gradient-magnitude template matching score to find the best position for a given ROI within the control sample.
      ROIs matched with the same control sample can not spatially overlap.
      If a ROI does not fit in the current control (ROI bigger than control or overlap), it is added to skipped_rois.
      Before loading new ROIs, the routine always checks skipped_rois to see if they fit the current control sample.
      If all new ROIs are exhausted and anomaly_duplicates is enabled, the routine restarts from the beginning of the ROI list.
      If the target number of matches cannot be reached, it issues a warning and moves to the next control sample.

  - global: (optimal pairing, but slow)  
      `Fusion synthetic anomaly into the best control sample within the dataset + finds best position within the sample.`  
      Iterates through every control sample and attempts to find the configured number of ROI matches.
      Performs combined intensity and gradient-magnitude template matching for all available ROIs against the current control sample to find the best possible positions for every ROI and save info in all_matches list.
      ROIs matched with the same control sample can not spatially overlap.
      Prioritizes matches based on the highest similarity score (sort all_matches by similarity descending and then always try to match the next index until no further matches are needed)
      ROIs that have been successfully matched are added to excluded_roi_samples and are excluded from future controls to avoid reuse.
      If no more matches are possible for the current control and anomaly_duplicates is enabled, also check excluded_roi_samples and select those with highest similarity.
      If all available ROIs have been excluded and anomaly_duplicates is enabled, the exclusion list is cleared to allow a full restart of the ROI pool.
      If the target number of matches cannot be reached, it issues a warning and moves to the next control sample.

  - fixed_from_extraction_control_fusion: (if images/dataset are aligned and homogeneous)  
      `Fusion synthetic anomaly into a random control sample at extraction position.`  
      Iterates through every control sample and pairs it with exactly one ROI from the dataloader in a sequential 1:1 relationship.
      Retrieves the exact fusion positions (centroids) from the metadata.
      If the ROI dataloader is exhausted and anomaly_duplicates is enabled, the routine restarts from the beginning of the ROI list to continue pairing.
      If the ROI dataloader is exhausted and anomaly_duplicates is disabled, the process stops entirely.
      Bypasses similarity scores, assuming the pre-extracted anomalies are already valid for the target control.

  - fixed_from_extraction_anomaly_fusion: (if no control samples exists)  
      `Fusion synthetic anomaly into the original anomaly sample directly over the extracted real anomaly.`  
      Iterates through every control sample and attempts to find a specific set of synthetic ROIs pre-assigned to it.
      Uses a naming convention (control_filename + index) to identify and load matching ROI files.
      Retrieves the exact fusion positions (centroids) from the metadata.
      Continues to load and append ROIs for a single control sample until no further matching filenames are found.
      Bypasses overlap checks and similarity scores, assuming the pre-extracted anomalies are already valid for the target control.

  Matching Summary:
    After the matching process, the system provides a summary to evaluate the efficiency of the chosen routine and parameters:
    Utilization Rate: Tracks how many of the available synthetic ROIs were actually fused into control samples.
    Unused ROIs: If some ROIs were never matched, the system calculates a suggested fusions_per_control value.
    Optimization Tip: To achieve a ~100% utilization rate, the summary suggests increasing the fusions_per_control based on the ratio of available ROIs to processed control samples.

---

### Fusion logic
  Fusion inserts the matched synthetic anomaly into the target control sample at the position stored in matching_dict.csv and creates the corresponding segmentation mask.
  Before blending, the anomaly is cropped to its foreground area, rescaled with the saved scale_factor and locally intensity-normalized to the insertion region of the control sample.
  The actual fusion uses an edge-aware alpha mask based on Sobel edges, morphology and a distance transform, so the anomaly interior can be blended more strongly than its boundary.

  Fusion parameters in config:
  - fusion_mask_params:
      max_alpha controls the maximum blending weight of the anomaly. For example, max_alpha = 0.8 means that fully weighted pixels use 0.8 * anomaly intensity + 0.2 * background intensity.
      sq and steepness_factor shape the alpha falloff from anomaly interior to boundary.
      upsampling_factor increases mask resolution during distance-transform computation for smoother alpha masks.
      sobel_threshold controls which gradients are treated as anomaly edges.
      dilation_size closes edge gaps and refines the foreground body before alpha creation.
      shave_pixels erodes boundary pixels to reduce visible blending artifacts.
  - fusion_variation:
      If enabled, max_alpha, sq and steepness_factor are sampled around their configured values for each fusion.
  - fusion_variation_params:
      alpha_variation, sq_variation and steepness_variation define the allowed one-sided deviation used for gaussian sampling.
  - selected_confidence / confidence_z_score:
      Converts the variation values into standard deviations; for example, selected_confidence = "90%" means samples stay within the configured one-sided deviation in about 90% of cases.
  - background_threshold:
      Defines which anomaly pixels are treated as foreground during cropping and alpha-mask creation. If None, the fusion code derives a threshold from the anomaly minimum.
  - fusion_normalization_border_width:
      Background region used to estimate control intensity for anomaly normalization. None disables fusion-time intensity normalization; -1 uses the entire image; >= 0 uses a local border around the anomaly mask.
  - fusion_restore_anomaly_bg_relation:
      If enabled, local border normalization preserves the extracted intensity relation between an anomaly and its original surrounding background. Disable it to use only direct intensity normalization.

---

### Evaluation details
  Pairwise comparison of real vs synthetic samples (pairing VAE Input with corresponding generated Output):

  - Textural Comparison: GLCM-based features (Contrast, Homogeneity, Energy, Correlation).

    GLCM (Gray-Level Co-occurrence Matrix):
        - encodes spatial relationships between intensity values by counting how often pairs of gray levels occur at given directional offsets (2D: 4 directions, 3D: 13 directions)
        - in this case, a distance of 1 is used, so only immediate neighbors are considered
        - a GLCM is computed separately for each channel by aggregating over all directional offsets after quantizing intensity values into discrete gray levels
    
    Computing differences between real and synthetic anomaly cutouts and between real and synthetic ROIs.
        Contrast:     Measures local intensity variations; high values indicate sharp edges or coarse textures.
        Homogeneity:  Measures the similarity of neighboring pixels; high values indicate smooth transitions.
        Energy:       Measures textural uniformity and order; high values indicate constant or repetitive patterns.
        Correlation:  Measures the linear dependency of neighboring gray levels; high levels indicate a strong linear relationship.

  - Morphological Comparison: Volume and Center of Mass (CoM). Computing differences between real and synthetic anomaly cutouts.
        Volume:       Sum of voxels within the segmentation mask to ensure size-preservation during synthesis.
        CoM:          The spatial centroid of the anomaly, used to detect positional shifts or shape-asymmetry. Calculated separately for each dimension.

  - Outlier Detection: (Optional) automated removal of low-quality synthetic anomalies and synthetic ROIs (statistical outliers or based on optional fixed thresholds in config).
      Manual removal of synthetic samples, synthetic ROIs, real anomalies + ROIs and whole generated hybrid samples + segmentations also possible within the Outlier Viewer.

  - Execution Summary: Outputs key statistics to the console like metric averages for real and synthetic datasets, per-metric outlier counts,
      and the intersection of outliers across different metrics (outlier overlaps).

  - Saves results to the evaluation_results directory within study_folder:
      metric_diffs.csv containing all computed metric difference values
      Histograms (distributions of the metric differences) for every metric, grouped into three .png files: GLCM on cutouts, Volume (morphological metrics) on cutouts, and GLCM on ROIs

---

### Visualize and Debug anomalies
  To fine-tune the configuration, debug the generation pipeline or delete samples, you can use the Outlier Viewer.

  **Key Capabilities:**
  - Hierarchical Navigation: parent-child relationship between generated hybrid samples and fused anomalies within them (treeview).
      Navigate between samples (and Slices (Depth) in case of 3D) with on-screen buttons or keyboard arrow keys.
      Shows generated hybrid sample + segmentation if control name is selected in treeview.
      Shows synthetic and real anomalies and ROIs if anomaly name (within a control) is selected in treeview.

  - Metric-based filtering and sorting: dynamically filter and sort dataset by selecting different metrics via checkboxes and adjusting the outlier threshold t via a mouse-draggable slider.
      Filtering: For each selected metric, only anomalies that fall within the top t% of highest differences of the metric are included in the tree view.
      Sorting: Ranks samples in descending order (average of their min-max normalized differences across all checked metrics). -> highest differences first

  - Deletion:
      Delete button; always asks for confirmation before deletion
      Delete whole generated hybrid sample + segmentation and all its ROIs if control name is selected in treeview
      If anomaly name is selected in treeview the delete button opens a window with checkboxes where you can check what you want to delete:
          Real Anomaly (VAE input) + real ROI
          Synthetic ROI (just this fusion)
          Synthetic Anomaly + all its ROIs (may affect other fusions)
          Hybrid Sample + all ROIs inside

  - Extra Features: 
      Contrast control via a mouse-draggable slider
      Slices (Depth) navigation via mouse wheel scrolling

  >The viewer expects .npy arrays with shape (C, D, H, W) or (C, H, W). It loads data directly from the study_folder where it was saved during the generation process.

---

### Cite this work
```tex
@INPROCEEDINGS{11159383,
  author={Pfleiderer, Adrian and Bauer, Bernhard},
  booktitle={2025 International Conference on Advanced Machine Learning and Data Science (AMLDS)}, 
  title={Fused Hybrid Training Samples through Synthetic Anomaly Generation for Optimized Model Training}, 
  year={2025},
  pages={248-256},
  doi={10.1109/AMLDS63918.2025.11159383}}
```


### License
This project is licensed under the GNU General Public License v3.0.
