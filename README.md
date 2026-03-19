# Hybrid Sample Generation
![Python](https://img.shields.io/badge/Python-14354C?style=flat&logo=python&logoColor=green) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![DOI:AMLDS63918.2025.11159383](http://img.shields.io/badge/DOI-AMLDS63918.2025.11159383-B31B1B.svg)](https://doi.org/10.1109/AMLDS63918.2025.11159383)

This repository implements a hybrid sample generation pipeline for imaging data to extend existing training datasets.
The implementation is based on the IEEE paper: https://doi.org/10.1109/AMLDS63918.2025.11159383
.

It extracts real anomaly cutouts from labeled volumes, trains a **ResNet-VAE**, generates
**synthetic anomaly cutouts**, matches them to control samples via ROI template matching, and finally
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
- Binary segmentation mask for the inserted anomaly

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
    HDG.train_generator(no_of_trails=5)
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
