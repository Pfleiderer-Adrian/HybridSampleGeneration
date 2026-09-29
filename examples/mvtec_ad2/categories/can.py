"""Complete MVTec AD 2 workflow for the can category."""
from examples.mvtec_ad2.common import (
    create_downstream_configuration,
    create_generator_configuration,
)
from examples.mvtec_ad2.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.downstream.runner import evaluate_downstream, train_downstream
from examples.mvtec_ad2.settings import category_root, study_folder
from examples.mvtec_ad2.splits import (
    SplitConfiguration,
    load_or_create_manifest,
    manifest_samples,
)
from hybrid_sample_generator import HybridDataGenerator

CATEGORY = "can"
ANOMALY_SIZE = (3, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    config.generation.variation_strength = 1.5
    config.extraction.roi.min_size = (256, 256)

    config.matching.anomalies_per_hybrid = 1
    config.matching.max_anomalies_per_hybrid_deviation = 1

    # Fusion settings
    config.fusion.set_backend("classical")
    config.fusion.parameters.max_alpha = 0.8
    config.fusion.parameters.sq = 1.0
    config.fusion.parameters.steepness_factor = 3.0
    config.fusion.parameters.upsampling_factor = 4
    config.fusion.parameters.fusion_use_sobel_for_alpha_mask = False
    config.fusion.parameters.shave_pixels = 0
    config.fusion.parameters.fusion_variation = True
    config.fusion.parameters.alpha_variation = 0.01
    config.fusion.parameters.sq_variation = 0.01
    config.fusion.parameters.steepness_variation = 1.0
    config.fusion.parameters.selected_confidence = "90%"

    config.extraction.roi.fixed_size = None
    config.extraction.roi.min_size = (1, 128, 128)
    config.matching.intensity_weight = 0.3
    config.matching.gradient_weight = 0.7
    config.extraction.roi.preserve_aspect_ratio = True


    return config


def main() -> None:
    config = create_configuration()
    manifest = load_or_create_manifest(
        study_folder(CATEGORY) / "split_manifest.json",
        category_root(CATEGORY),
        SplitConfiguration(test_fraction=0.2, validation_fraction=0.2, seed=42),
    )

    generator = HybridDataGenerator(config)
    generator.ingest_dataset(MVTecAD2Dataloader(manifest_samples(manifest, "train")))
    generator.extract_anomalies()
    generator.train_generator()
    generator.generate_synthetic_anomalies()
    generator.plan_hybrid_samples()
    generator.materialize_hybrid_samples()
    config.save_config_file()

    run_folder = train_downstream(config, manifest, create_downstream_configuration())
    metrics = evaluate_downstream(run_folder, manifest)
    print(metrics)


if __name__ == "__main__":
    main()
