"""Generator configuration for the MVTec AD 2 walnuts category."""
from examples.mvtec_ad2.synthetic_data_generation.hybrid.configuration import create_generator_configuration

CATEGORY = "walnuts"
ANOMALY_SIZE = (3, 64, 64)


def create_configuration():
    config = create_generator_configuration(CATEGORY, ANOMALY_SIZE)
    return config
