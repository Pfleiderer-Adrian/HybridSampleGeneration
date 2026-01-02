# Hybrid Sample Generation

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
- `synthesizer/Anomaly_Extraction.py` — anomaly + ROI cutouts
- `synthesizer/Fusion3D.py` — fusion + matching (template matching)
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
- Currently only 3D: `(C, D, H, W)`
- Also planned for 2D: `(C, H, W)`

---

### Main Requirements

✅ **Packages:**
- `Python 14+`
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
- Both arrays must be in channels-first format: (Channels, Depth, Height, Width)

---

### Usage
```python
    # create config
    config = Configuration("Study01", "VAE_ResNet_3D", anomaly_size=(1, 32, 96, 96))

    HDG = HybridDataGenerator(config)
    # 1) Extract anomaly cutouts + ROI cutouts from anomaly-labeled samples
    HDG.extract_anomalies(dataloader, None)
    # 1) Or load already extracted anomalies
    HDG.load_anomalies()

    # 2) Train generator via Optuna
    HDG.train_generator(1)
    # 2) Or only load a trained model
    HDG.load_generator()

    # 3) Generate and load synthetic anomalies
    HDG.generate_synth_anomalies()
    # 3) Or load already generated synthetic anomalies
    HDG.load_synth_anomalies()

    # 4) Create matching between control samples and anomaly ROIs
    HDG.create_matching_dict(dataloader_samples_with_anomalies)
    # 4) Or load already created matching dict
    HDG.load_matching_dict()

    # 5) Generate new Hybrid Training Samples
    img, seg = HDG.fusion_synth_anomalies(control_image, basename_of_image)
```
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