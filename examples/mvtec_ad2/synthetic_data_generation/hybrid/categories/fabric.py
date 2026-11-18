"""Generator configuration for the MVTec AD 2 fabric category."""
from examples.mvtec_ad2.synthetic_data_generation.hybrid.configuration import create_generator_configuration

CATEGORY = "fabric"
ANOMALY_SIZE = (3, 128, 128)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    config.extraction.roi.min_size = (128, 128)
    config.fusion.parameters.segmentation_min_alpha = 0.2

    config.matching.anomalies_per_hybrid = 1
    config.matching.max_anomalies_per_hybrid_deviation = 0

    return config
