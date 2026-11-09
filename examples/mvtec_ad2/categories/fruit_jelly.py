"""Complete MVTec AD 2 workflow for the fruit_jelly category."""
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

CATEGORY = "fruit_jelly"
ANOMALY_SIZE = (3, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    config.model.parameters.skip_alpha = 0.1
    config.model.parameters.skip_dropout_p = 0.85

    config.matching.hybrids_per_original = 5
    config.matching.reuse_synthetic_across_hybrids = True
    config.matching.allow_sibling_variants_in_same_hybrid = False
    config.matching.anomalies_per_hybrid = 1
    config.matching.max_anomalies_per_hybrid_deviation = 1

    config.fusion.parameters.image_harmonization = "LBM"

    # Exclude hybrids whose defect mask extends too far outside the object.
    post = config.postprocessing
    post.enabled = True
    post.action = "exclude"
    post.save_previews = True
    post.foreground.channel_reduction = "minimum"
    post.foreground.comparison = "below"
    post.foreground.threshold = 240
    post.foreground.components = "largest"
    post.foreground.fill_holes = True
    post.rules.minimum_object_overlap.enabled = True
    post.rules.minimum_object_overlap.threshold = 0.90

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
    generator.postprocess_hybrid_samples()
    config.save_config_file()

    run_folder = train_downstream(config, manifest, create_downstream_configuration())
    metrics = evaluate_downstream(run_folder, manifest)
    print(metrics)


if __name__ == "__main__":
    main()
