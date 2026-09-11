"""Configuration model for learned residual-alpha fusion."""

from dataclasses import dataclass

from hybrid_sample_generator.fusion.config_validation import validate_parameters


@dataclass(slots=True)
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
    fusion_keep_bg: bool = False
    fusion_bg_value: float | None = None
    fusion_relative_bg_threshold: float | None = 0.01
    fusion_bg_exterior_only: bool = True

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

    def validate(self) -> None:
        validate_parameters(
            self,
            positive=("base_channels", "depth", "train_epochs", "train_lr",
                      "train_max_samples_per_epoch", "grad_clip_norm"),
            nonnegative=("log_every", "alpha_delta_scale", "residual_scale", "base_alpha_blur_sigma",
                         "residual_border_width", "fusion_relative_bg_threshold",
                         "train_weight_decay", "train_crop_margin", "train_inpaint_blur_sigma",
                         "foreground_loss_weight", "support_loss_weight", "alpha_delta_l1", "residual_l1"),
            unit_interval=("base_alpha",),
        )
        if self.spatial_dims not in (None, 2, 3):
            raise ValueError("fusion.parameters.spatial_dims must be None, 2 or 3.")
