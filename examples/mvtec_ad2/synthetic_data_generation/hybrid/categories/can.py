"""Generator configuration for the MVTec AD 2 can category."""
from examples.mvtec_ad2.synthetic_data_generation.hybrid.configuration import create_generator_configuration

CATEGORY = "can"
ANOMALY_SIZE = (3, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    config.generation.variation_strength = 1.5
    config.extraction.roi.min_size = (128, 128)

    config.matching.anomalies_per_hybrid = 3
    config.matching.hybrids_per_original = 2
    config.generation.variants_per_real_anomaly = 10
    config.matching.reuse_synthetic_across_hybrids = True
    config.matching.max_anomalies_per_hybrid_deviation = 0

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
    config.matching.intensity_weight = 0.5
    config.matching.gradient_weight = 0.3
    config.matching.distance_weight = 0.2
    config.extraction.roi.preserve_aspect_ratio = True

    config.augmentation.mask_transforms.use_mask_transform = True

    config.augmentation.mask_transforms.mask_transform_probs = {
        "zoom": 0.5,
        "stretch": 0.5,
        "rotate": 0.5,
        "elastic": 0.0,
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

    # Quality decisions remain separate from materialized image artifacts.
    post = config.postprocessing
    post.enabled = True
    post.save_previews = True
    post.action = "exclude"
    post.foreground.channel_reduction = "minimum"
    post.foreground.comparison = "below"
    post.foreground.threshold = 240
    post.foreground.components = "largest"
    post.foreground.fill_holes = True
    post.rules.minimum_object_overlap.enabled = True
    post.rules.minimum_object_overlap.threshold = 1.0

    return config
