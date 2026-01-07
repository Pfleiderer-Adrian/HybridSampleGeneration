import os
from dataclasses import asdict
import nibabel as nib
import numpy as np

from use_cases.image_2d.ImageDataloader import ImageDataloader, save_image
from use_cases.mri_3d.NiftiDataloader import NiftiDataloader
from synthesizer.Configuration import Configuration
from synthesizer.HybridDataGenerator import HybridDataGenerator
from PIL import Image


path_to_img = "/mnt/dataset-nassfliess/train_set"
path_to_seg = "/mnt/dataset-nassfliess/train_seg"

path_to_control_img = "/mnt/dataset-nassfliess/train_set"
path_to_control_seg = "/mnt/dataset-nassfliess/train_seg"

# keep in mind you need to create your on dataloader/iterator for your dataset (e.g. png-files in one folder)
# Iterator/Dataloder must yielding - (img_arr, seg_arr, basename)
# img_arr.shape == seg_arr.shape and (Channels, Depth, Height, Width)
# load samples with anomalies
dataloader_samples_with_anomalies = ImageDataloader(path_to_img, path_to_seg)
print(dataloader_samples_with_anomalies.sample_infos)
# load samples without anomalies
dataloader_samples_without_anomalies = ImageDataloader(path_to_control_img, path_to_control_seg)

if __name__ == "__main__":

    # define a basic configuration
    config = Configuration("images", "VAE_ResNet_2D", (3, 32, 32))
    config.epochs = 1
    HDG = HybridDataGenerator(config)
    # 1) Extract anomaly cutouts + ROI cutouts from anomaly-labeled samples
    HDG.extract_anomalies(dataloader_samples_with_anomalies, None)
    # 1) Or load already extracted anomalies
    HDG.load_anomalies()

    # 2) Train generator via Optuna
    HDG.train_generator(no_of_trails=1)
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

    # set result folder
    save_folder = os.path.join(config.study_folder, "generated_hybrid_samples")
    img_folder = os.path.join(save_folder, "images")
    seg_folder = os.path.join(save_folder, "segmentations")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)

    # iterate over your control samples
    for control_image, _, basename in dataloader_samples_without_anomalies:

        # 5) Fuse synthetic anomaly into one control sample
        img, seg = HDG.fusion_synth_anomalies(control_image, basename)

        filepath = os.path.join(img_folder, basename)
        save_image(img, filepath)

        filepath = os.path.join(seg_folder, basename)
        save_image(seg, filepath)

    # save the actual configuration
    config.save_config_file()