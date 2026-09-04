from use_cases.mri_3d.NiftiDataloader import NiftiDataloader
from synthesizer.Configuration import Configuration
from synthesizer.Evaluation import evaluate_study
from synthesizer.HybridDataGenerator import HybridDataGenerator
from data_handler.Visualizer import run_hybrid_visualizer

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
    # config.fusion.set_backend("learned_residual_alpha")

    generator = HybridDataGenerator(config)
    # 1) Extract anomaly cutouts + ROI cutouts from anomaly-labeled samples
    generator.extract_anomalies(dataloader_samples_with_anomalies)

    # 2) Train generator via Optuna
    generator.train_generator(no_of_trials=1)
    # To reuse a trained model instead: omit training and call generator.load_generator().

    # 3) Generate synthetic anomalies
    generator.generate_synthetic_anomalies()

    # 4) Plan multiple hybrid variants and normalized placements
    generator.plan_hybrid_samples(dataloader_samples_without_anomalies)

    # Optional when using a trainable fusion backend:
    # generator.train_fusion_backend(dataloader_samples_with_anomalies)

    # 5) Materialize every planned hybrid; the fusion backend is initialized lazily
    generator.materialize_hybrid_samples()
    hybrid_dataset = generator.datasets.hybrid_samples(
        load_to_ram=False,
        numpy_mode=True,
    )

    # save the actual configuration
    config.save_config_file()

    # 6) Compute metric differences directly from persisted study relationships
    evaluate_study(config)
    
    # 7) Browse the persisted study hierarchy
    run_hybrid_visualizer(config)
