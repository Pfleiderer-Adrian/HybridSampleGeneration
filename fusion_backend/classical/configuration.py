from dataclasses import asdict, dataclass

from fusion_backend.fusion_configuration import FusionConfiguration


CONFIDENCE_LEVELS = {
    "68%": 1.0,
    "80%": 1.28,
    "90%": 1.645,
    "95%": 1.960,
    "99%": 2.576,
}


@dataclass
class Config:
    max_alpha: float = 0.8
    sq: float = 2
    steepness_factor: float = 3
    upsampling_factor: int = 2
    sobel_threshold: float = 0.05
    dilation_size: int = 2
    shave_pixels: int = 1
    fusion_variation: bool = True
    alpha_variation: float = 0.05
    sq_variation: float = 0.1
    steepness_variation: float = 0.1
    selected_confidence: str = "90%"
    # Border pixels used to normalize anomaly intensity to the surrounding
    # fusion region: None skips normalization, 0 uses only the anomaly mask,
    # >0 uses a wider surrounding region, -1 uses the entire image.
    fusion_normalization_border_width: int | None = 2


def get_classical_fusion_configuration():
    return FusionConfiguration(asdict(Config()))
