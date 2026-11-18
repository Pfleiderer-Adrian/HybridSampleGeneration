"""Generator configuration for the MVTec AD 2 vial category."""
from examples.mvtec_ad2.synthetic_data_generation.hybrid.configuration import create_generator_configuration

CATEGORY = "vial"
ANOMALY_SIZE = (1, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)

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
        "rotate": {"max_rotation": 1.0},
        "elastic": {"sigma": 30, "magnitude": 5},
    }

    config.extraction.roi.min_size = (1, 128, 128)

    return config
