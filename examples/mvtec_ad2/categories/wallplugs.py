"""Complete MVTec AD 2 workflow for the wallplugs category."""
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

CATEGORY = "wallplugs"
ANOMALY_SIZE = (1, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    config.augmentation.mask_transforms.use_mask_transform = False
    config.augmentation.mask_transforms.mask_transform_probs = {
        "zoom": 0.5,
        "stretch": 0.5,
        "rotate": 0.8,
        "elastic": 0.5,
        "local_dilate": 0,
        "local_stretch": 0,
        "local_rotate": 0,
        "local_elastic": 0,
    }
    config.augmentation.mask_transforms.mask_transform_params = {
        "zoom": {"min_zoom": 0.97, "max_zoom": 1.0},
        "stretch": {"min_stretch": 0.98, "max_stretch": 1.05},
        "rotate": {"max_rotation": 2.0},
        "elastic": {"sigma": 30, "magnitude": 5},
    }

    config.extraction.roi.fixed_size = None
    config.extraction.roi.min_size = (1, 256, 256)
    config.matching.intensity_weight = 0.3
    config.matching.gradient_weight = 0.7
    config.extraction.roi.preserve_aspect_ratio = True
    config.extraction.preserve_aspect_ratio = True

    config.matching.batch_size = 64

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
