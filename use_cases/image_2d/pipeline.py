import os
from PIL import Image

from synthesizer.Matching import combine_binary_masks
from use_cases.image_2d.ImageDataloader import ImageDataloader, save_image
from synthesizer.Configuration import Configuration
from synthesizer.HybridDataGenerator import HybridDataGenerator

path_to_img = "/mnt/results/Osram/images/werk_berlin_glass/xbopolarisation/feasibility"
path_to_seg = "/mnt/results/Osram/images/werk_berlin_glass/segmentations"

path_to_control_img = "/mnt/results/Osram/images/werk_berlin_glass/xbopolarisation/feasibility"
path_to_control_seg = "/mnt/results/Osram/images/werk_berlin_glass/segmentations"

# keep in mind you need to create your on dataloader/iterator for your dataset (e.g. png-files in one folder)
# Iterator/Dataloder must yielding - (img_arr, seg_arr, basename)
# img_arr.shape == seg_arr.shape and (Channels, Height, Width)
# load samples with anomalies
dataloader_samples_with_anomalies = ImageDataloader(path_to_img, path_to_seg)
# load samples without anomalies
dataloader_samples_without_anomalies = ImageDataloader(path_to_control_img, path_to_control_seg)

if __name__ == "__main__":

    # define a basic configuration
    config = Configuration("berlin_5", "VAE_ConvNeXt_2D", (1, 96, 96), "/mnt/results/")

    HDG = HybridDataGenerator(config)
    # 1) Extract anomaly cutouts + ROI cutouts from anomaly-labeled samples
    #HDG.extract_anomalies(dataloader_samples_with_anomalies, None)
    # 1) Or load already extracted anomalies
    HDG.load_anomalies()

    # 2) Train generator via Optuna
    #HDG.train_generator(no_of_trails=1)
    # 2) Or only load a trained model
    HDG.load_generator(trial_id=-1)

    # 3) Generate and load synthetic anomalies
    #HDG.generate_synth_anomalies()
    # 3) Or load already generated synthetic anomalies
    HDG.load_synth_anomalies()


    # 4) Create matching between control samples and anomaly ROIs
    #HDG.create_matching_dict(dataloader_samples_without_anomalies)
    # 4) Or load already created matching dict
    HDG.load_matching_dict()
    # set result folder
    save_folder = os.path.join(config.study_folder, "generated_hybrid_samples")
    img_folder = os.path.join(save_folder, "images_synth")
    seg_folder = os.path.join(save_folder, "segmentations_synth")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)
    
    # iterate over your control samples
    for control_image, control_seg, basename in dataloader_samples_without_anomalies:

        #if basename not in config.matching_dict:
        #    print(basename+" not found in matching dict")
        #    continue

        # 5) Fuse synthetic anomaly into one control sample
        img, seg = HDG.fusion_synth_anomalies(control_image, basename)

        filepath = os.path.join(img_folder, "synth-" + basename)
        save_image(img, filepath)

        seg_final = combine_binary_masks(control_seg, seg, mode = "or")

        filepath = os.path.join(seg_folder, "synth-" + basename)
        save_image(seg_final, filepath)
    """

    # save the actual configuration
    save_folder_ori = os.path.join(config.study_folder, path_to_seg)

    seg_loader = ImageDataloader(seg_folder, save_folder_ori)

    seg_comb_folder = os.path.join(save_folder, "seg_combined")
    os.makedirs(seg_comb_folder, exist_ok=True)
    for seg1, seg2, basename in seg_loader:
        save_path = os.path.join(seg_comb_folder, basename)
        seg_comb = combine_binary_masks(seg1, seg2, mode = "or")
        save_image(seg_comb, save_path)
"""
    # save the actual configuration
    config.save_config_file()