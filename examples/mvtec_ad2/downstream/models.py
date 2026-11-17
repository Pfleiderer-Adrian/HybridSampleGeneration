"""Optional downstream model selection; old snapshots default to DRAEM."""

from .draem.model import DRAEM


def build_model(config, *, backbone_config=None):
    if config.model == "superadd_dual_bank":
        from .memory_bank import SuperADDDualBank
        return SuperADDDualBank(config, backbone_config=backbone_config)
    if config.model == "draem":
        return DRAEM(config.training.reconstruction_width, config.training.segmentation_width)
    if config.model in ("dinov3_paired", "dinov3_segmentation"):
        from .dinov3 import DINOv3Paired, DINOv3Segmentation
        model = DINOv3Paired if config.model == "dinov3_paired" else DINOv3Segmentation
        return model(config.dinov3, backbone_config=backbone_config)
    raise ValueError(f"Unknown downstream model {config.model!r}.")
