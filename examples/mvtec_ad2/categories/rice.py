"""Complete MVTec AD 2 workflow for the rice category."""
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

CATEGORY = "rice"
ANOMALY_SIZE = (3, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)

    # Keep the latent feature map close to the size used for 64 x 64 inputs.
    # Five levels reduce 128 x 128 inputs to 4 x 4; the search may also test 8 x 8.
    config.model.parameters.n_levels = 5
    config.model.search.n_levels = IntRange(4, 5)

    return config


def main() -> None:
    config = create_configuration()
    manifest = load_or_create_manifest(
        study_folder(CATEGORY) / "split_manifest.json",
        category_root(CATEGORY),
        SplitConfiguration(test_fraction=0.2, validation_fraction=0.2, seed=42),
    )

    config.generation.variation_strength = 1.5
    config.extraction.roi.min_size = (128, 128)

    config.matching.anomalies_per_hybrid = 2
    config.matching.hybrids_per_original = 1
    config.generation.variants_per_real_anomaly = 5
    config.matching.reuse_synthetic_across_hybrids = True
    config.matching.max_anomalies_per_hybrid_deviation = 1

    # Fusion settings
    config.fusion.set_backend("classical")
    config.fusion.parameters.max_alpha = 0.8
    config.fusion.parameters.segmentation_min_alpha = 0.10
    config.fusion.parameters.sq = 1.0
    config.fusion.parameters.steepness_factor = 3.0
    config.fusion.parameters.upsampling_factor = 4
    config.fusion.parameters.fusion_use_sobel_for_alpha_mask = False
    config.fusion.parameters.shave_pixels = 0
    config.fusion.parameters.fusion_variation = True
    config.fusion.parameters.alpha_variation = 0.01
    config.fusion.parameters.sq_variation = 0.02
    config.fusion.parameters.steepness_variation = 0.15
    config.fusion.parameters.selected_confidence = "90%"

    config.fusion.parameters.image_harmonization = "LBM"

    config.extraction.roi.fixed_size = None
    config.extraction.roi.min_size = (1, 128, 128)
    config.matching.intensity_weight = 0.3
    config.matching.gradient_weight = 0.7
    config.extraction.roi.preserve_aspect_ratio = True

    config.augmentation.mask_transforms.use_mask_transform = True

    

    config.augmentation.mask_transforms.mask_transform_probs = {
        "zoom": 0.5,
        "stretch": 0.5,
        "rotate": 0.5,
        "elastic": 0.5,
        "local_dilate": 0,
        "local_stretch": 0,
        "local_rotate": 0,
        "local_elastic": 0,
    }
    config.augmentation.mask_transforms.mask_transform_params = {
        "zoom": {"min_zoom": 0.97, "max_zoom": 1.0},
        "stretch": {"min_stretch": 0.95, "max_stretch": 1.05},
        "rotate": {"max_rotation": 2.0},
        "elastic": {"sigma": 30, "magnitude": 5},
    }
    """
    # Fusion settings
    config.fusion.set_backend("classical")
    config.fusion.parameters.max_alpha = 0.8
    # Preserve narrow defect passages while retaining an inward edge transition.
    config.fusion.parameters.sq = 1.0
    config.fusion.parameters.steepness_factor = 4.0
    config.fusion.parameters.upsampling_factor = 4
    config.fusion.parameters.fusion_use_sobel_for_alpha_mask = False
    config.fusion.parameters.fusion_variation = True
    config.fusion.parameters.alpha_variation = 0.02
    config.fusion.parameters.sq_variation = 0.05
    config.fusion.parameters.steepness_variation = 0.25
    config.fusion.parameters.selected_confidence = "90%"
    """

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
