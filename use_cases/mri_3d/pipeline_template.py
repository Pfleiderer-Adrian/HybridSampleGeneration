import os
from dataclasses import asdict
import nibabel as nib
import numpy as np

from use_cases.mri_3d.NiftiDataloader import NiftiDataloader
from synthesizer.Configuration import Configuration
from synthesizer.HybridDataGenerator import HybridDataGenerator

path_to_img = "add image path here for samples with anomalies"
path_to_seg = "add segmentation path here for samples with anomalies"

path_to_control_img = "add image path here for samples without anomalies"
path_to_control_seg = "add segmentation path here for samples without anomalies"

# keep in mind you need to create your on dataloader/iterator for your dataset (e.g. nii-files in one folder)
# Iterator/Dataloder must yielding - (img_arr, seg_arr, basename)
# img_arr.shape == seg_arr.shape and (Channels, Depth, Height, Width)
# load samples with anomalies
dataloader_samples_with_anomalies = NiftiDataloader(path_to_img, path_to_seg, "t1")
# load samples without anomalies
dataloader_samples_without_anomalies = NiftiDataloader(path_to_control_img, path_to_control_seg, "t1")

if __name__ == "__main__":

    # define a basic configuration
    config = Configuration("brain_T1", "VAE_ConvNeXt_3D", (1, 32, 96, 96))
    # Optional trainable fusion backend:
    # config.set_fusion_backend("learned_residual_alpha")

    HDG = HybridDataGenerator(config)
    # 1) Extract anomaly cutouts + ROI cutouts from anomaly-labeled samples
    HDG.extract_anomalies(dataloader_samples_with_anomalies)
    # 1) Or load already extracted anomalies
    HDG.load_anomalies()

    # 2) Train generator via Optuna
    HDG.train_generator(no_of_trials=1)
    # 2) Or only load a trained model
    HDG.load_generator()

    # 3) Generate and load synthetic anomalies
    HDG.generate_synth_anomalies()
    # 3) Or load already generated synthetic anomalies
    HDG.load_synth_anomalies()

    # 4) Create matching between control samples and anomaly ROIs
    HDG.create_matching_dict(dataloader_samples_without_anomalies)
    # 4) Or load already created matching dict
    HDG.load_matching_dict()

    # Optional when using a trainable fusion backend:
    # HDG.train_fusion_backend(dataloader_samples_with_anomalies)
    # 5) Initialize the configured fusion backend
    HDG.load_fusion_backend()

    # set result folder
    paths = config.get_paths()
    img_folder = paths.generated_images
    seg_folder = paths.generated_segmentations
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)

    # iterate over your control samples
    for control_image, _, basename, img_affine, seg_affine in dataloader_samples_without_anomalies:

        if basename not in config.matching_dict:
            print(basename+" not found in matching dict")
            continue

        # 6) Fuse synthetic anomaly into one control sample
        img, seg = HDG.fusion_synth_anomalies(control_image, basename)

        # your own saving routine
        tmp_obj = nib.Nifti1Image(img, img_affine, dtype=np.int32)
        filepath = os.path.join(img_folder, basename)
        nib.save(tmp_obj, filepath)
        tmp_obj = nib.Nifti1Image(seg, seg_affine, dtype=np.int32)
        filepath = os.path.join(seg_folder, basename)
        nib.save(tmp_obj, filepath)

    # save the actual configuration
    config.save_config_file()

    dataloader_samples_with_anomalies = NiftiDataloader(path_to_img, path_to_seg, "t2w")

    # 7) Compute (textural and morphological) metric differences for real-synthetic pairs
    HDG.run_evaluation_pipeline(dataloader_samples_with_anomalies)
    
    # 8) Start Outlier Viewer for manual inspection of generated samples
    HDG.visualize_evaluation_results()
