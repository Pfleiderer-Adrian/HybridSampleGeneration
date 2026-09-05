from use_cases.image_2d.ImageDataloader import ImageDataloader
from synthesizer.Configuration import Configuration
from synthesizer.Evaluation import evaluate_study
from synthesizer.HybridDataGenerator import HybridDataGenerator
from data_handler.Visualizer import run_hybrid_visualizer

path_to_img = "add image path here for all original samples"
path_to_seg = "add segmentation path here for all original samples"

# keep in mind you need to create your on dataloader/iterator for your dataset (e.g. png-files in one folder)
# The iterator/dataloader yields (img_arr, seg_arr_or_none, basename).
# Images use (Channels, Height, Width); annotated masks share the spatial shape.
# Controls may have an empty mask or no annotation at all.
dataloader_all_samples = ImageDataloader(path_to_img, path_to_seg)

if __name__ == "__main__":

    # define a basic configuration
    config = Configuration("images", "VAE_ResNet_2D", (3, 32, 32))
    # Optional trainable fusion backend:
    # config.fusion.set_backend("learned_residual_alpha")

    generator = HybridDataGenerator(config)
    # 1) Persist and classify every original exactly once
    generator.ingest_dataset(dataloader_all_samples)

    # 2) Extract anomaly cutouts + ROI cutouts from persisted anomalous originals
    generator.extract_anomalies()

    # 3) Train generator via Optuna
    generator.train_generator(no_of_trials=1)
    # To reuse a trained model instead: omit training and call generator.load_generator().

    # 4) Generate synthetic anomalies
    generator.generate_synthetic_anomalies()

    # 5) Plan controls from persisted originals
    generator.plan_hybrid_samples()

    # Optional when using a trainable fusion backend:
    # generator.train_fusion_backend()

    # 6) Materialize every planned hybrid; the fusion backend is initialized lazily
    generator.materialize_hybrid_samples()
    hybrid_dataset = generator.datasets.hybrid_samples(
        load_to_ram=False,
        numpy_mode=True,
    )

    # save the actual configuration
    config.save_config_file()

    # 7) Compute metric differences directly from persisted study relationships
    evaluate_study(config)

    # 8) Browse the persisted study hierarchy
    run_hybrid_visualizer(config)
