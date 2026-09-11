"""Run the two-dimensional hybrid sample generation example."""

from examples.image_2d.image_dataloader import ImageDataloader
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.evaluation.service import evaluate_study
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from hybrid_sample_generator.visualization import run_hybrid_visualizer

def main() -> None:
    path_to_img = "add image path here for all original samples"
    path_to_seg = "add segmentation path here for all original samples"

    # Replace this with an iterator for the dataset being processed.
    # Images use (Channels, Height, Width); controls may have an empty mask or None.
    dataloader_all_samples = ImageDataloader(path_to_img, path_to_seg)

    # Define a basic configuration
    config = Configuration("images", "cVAE_ConvNeXt_2D", (3, 32, 32))

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


if __name__ == "__main__":
    main()
