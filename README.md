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

    # 5) Generate new Hybrid Training Samples
    img, seg = HDG.fusion_synth_anomalies(control_image, basename_of_image)
```

---

### Visualize and Debug anomalies

To fine-tune the configuration or debug the generation pipeline, you can visualize:

- **Extracted input anomaly**: `./anomaly_data/<filename>.npy` (train input)
- **Extracted ROI of anomaly**: `./anomaly_roi_data/<filename>.npy` (only for Matching)
- **Generated synthetic anomaly**: `./synt_anomaly_data/<filename>.npy` (model output)

Use the lightweight project viewer located at `./data_handler/Visualizer.py`:

```bash
python ./data_handler/Visualizer.py /filepath/to/your/file.npy
```

**Features:**
- Depth (D) navigation via a mouse-draggable slider (and mouse wheel scrolling over the image)
- Contrast control via a mouse-draggable slider (window/level style)

>The viewer expects .npy arrays with shape (C, D, H, W).

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
