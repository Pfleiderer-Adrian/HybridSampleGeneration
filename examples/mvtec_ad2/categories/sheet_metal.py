"""Complete MVTec AD 2 workflow for the sheet_metal category."""
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
from hybrid_sample_generator.generation.model_settings import IntRange


CATEGORY = "sheet_metal"
ANOMALY_SIZE = (1, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)

    # extraction settings
    config.extraction.preserve_aspect_ratio = True

    # training settings
    config.model.parameters.beta_kl_warmup_epochs = 200
    config.model.search.n_levels = IntRange(4, 5)

    # Fusion settings
    config.fusion.set_backend("classical")
    config.fusion.parameters.max_alpha = 1.0
    config.fusion.parameters.sq = 0.1
    config.fusion.parameters.steepness_factor = 5.0
    config.fusion.parameters.upsampling_factor = 2
    config.fusion.parameters.sobel_threshold = 0.01
    config.fusion.parameters.dilation_size = 1
    config.fusion.parameters.shave_pixels = 0
    config.fusion.parameters.fusion_use_sobel_for_alpha_mask = False
    config.fusion.parameters.fusion_variation = True
    config.fusion.parameters.alpha_variation = 0.05
    config.fusion.parameters.sq_variation = 0.1
    config.fusion.parameters.steepness_variation = 1.0
    config.fusion.parameters.selected_confidence = "90%"
    config.fusion.parameters.fusion_normalization_border_width = None

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
