from dataclasses import asdict, dataclass

from fusion_backend.fusion_configuration import FusionConfiguration


@dataclass
class Config:
    """
    Parameters for the trainable residual-alpha fusion backend.

    The backend first builds a deterministic alpha-blend proposal and then uses
    a small CNN to predict an alpha correction and a residual image correction.
    """

    spatial_dims: int | None = None
    base_channels: int = 32
    depth: int = 4
    alpha_delta_scale: float = 0.25
    residual_scale: float = 0.25
    base_alpha: float = 0.85
    base_alpha_blur_sigma: float = 1.0
    residual_border_width: int = 2
    fusion_normalization_border_width: int | None = None
    clamp_output: bool = False

    train_epochs: int = 25
    train_lr: float = 1e-3
    train_weight_decay: float = 1e-5
    train_max_samples_per_epoch: int | None = None
    train_crop_margin: int = 24
    train_inpaint_blur_sigma: float = 8.0
    foreground_loss_weight: float = 4.0
    support_loss_weight: float = 1.0
    alpha_delta_l1: float = 1e-4
    residual_l1: float = 1e-4
    grad_clip_norm: float | None = 1.0
    log_every: int | None = 10


def get_learned_residual_alpha_fusion_configuration():
    return FusionConfiguration(asdict(Config()))
