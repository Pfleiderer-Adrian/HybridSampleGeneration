"""Generator configuration for the MVTec AD 2 sheet_metal category."""
from examples.mvtec_ad2.synthetic_data_generation.hybrid.configuration import create_generator_configuration
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
    config.generation.posterior_skip_source = "transformed"

    config.matching.distance_weight = 0.3

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
