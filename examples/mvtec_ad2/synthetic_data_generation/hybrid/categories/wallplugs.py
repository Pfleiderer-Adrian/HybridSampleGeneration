"""Generator configuration for the MVTec AD 2 wallplugs category."""
from examples.mvtec_ad2.synthetic_data_generation.hybrid.configuration import create_generator_configuration
from examples.mvtec_ad2.data.splits import capture_variants
from hybrid_sample_generator.generation.model_settings import Choice

CATEGORY = "wallplugs"
ANOMALY_SIZE = (1, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    # Weak encoder-to-decoder skips with strong dropout during training.
    config.model.parameters.skip_alpha = 0.1
    config.model.parameters.skip_dropout_p = 0.85
    config.augmentation.mask_transforms.use_mask_transform = True
    config.augmentation.mask_transforms.mask_transform_probs = {
        "zoom": 0.5,
        "stretch": 0.5,
        "rotate": 0.8,
        "elastic": 0.0,
        "local_dilate": 0,
        "local_stretch": 0,
        "local_rotate": 0,
        "local_elastic": 0,
    }
    config.augmentation.mask_transforms.mask_transform_params = {
        "zoom": {"min_zoom": 0.97, "max_zoom": 1.0},
        "stretch": {"min_stretch": 0.98, "max_stretch": 1.05},
        "rotate": {"max_rotation": 180.0},
        "elastic": {"sigma": 10, "magnitude": 5},
    }

    config.extraction.roi.fixed_size = None
    config.extraction.roi.min_size = (1, 128, 128)
    config.matching.intensity_weight = 0.4
    config.matching.gradient_weight = 0.6
    config.matching.distance_weight = 0.0
    config.extraction.roi.preserve_aspect_ratio = True
    config.extraction.preserve_aspect_ratio = True

    config.matching.anomalies_per_hybrid = 2
    config.matching.hybrids_per_original = 5
    config.generation.variants_per_real_anomaly = 10
    config.matching.reuse_synthetic_across_hybrids = True
    config.matching.max_anomalies_per_hybrid_deviation = 2

    config.matching.batch_size = 64
    config.model.search.bottleneck_dim = Choice((32, 64, 128))

    # Quality decisions remain separate from materialized image artifacts.
    post = config.postprocessing
    post.enabled = True
    post.save_previews = True
    post.action = "exclude"
    post.foreground.channel_reduction = "grayscale"
    post.foreground.comparison = "above"
    post.foreground.threshold = 40
    post.foreground.components = "all"
    post.foreground.min_component_area = 100
    post.foreground.fill_holes = False
    post.rules.minimum_object_overlap.enabled = True
    post.rules.minimum_object_overlap.threshold = 0.95
    # Defects 010 and 012 legitimately extend beyond the plug outline.
    post.rules.minimum_object_overlap.exempt_source_names = capture_variants(
        ["test_public_bad_010", "test_public_bad_012"]
    )

    return config
