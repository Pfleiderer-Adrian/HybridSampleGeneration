"""Optional downstream model selection; old snapshots default to DRAEM."""

from examples.mvtec_ad2.downstream.models.draem import DRAEM


def build_model(config, *, backbone_config=None):
    if config.model == "superadd_dual_bank":
        from examples.mvtec_ad2.downstream.models.memory_bank import SuperADDDualBank
        return SuperADDDualBank(config, backbone_config=backbone_config)
    if config.model == "draem":
        return DRAEM(config.training.reconstruction_width, config.training.segmentation_width)
    if config.model in ("dinov3_paired", "dinov3_segmentation"):
        if config.dinov3.backbone_type == "convnext":
            from examples.mvtec_ad2.downstream.models.dinov3_convnext import DINOv3ConvNeXtPaired, DINOv3ConvNeXtSegmentation
            model = DINOv3ConvNeXtPaired if config.model == "dinov3_paired" else DINOv3ConvNeXtSegmentation
            return model(config.dinov3, backbone_config=backbone_config)
        from examples.mvtec_ad2.downstream.models.dinov3 import DINOv3Paired, DINOv3Segmentation
        model = DINOv3Paired if config.model == "dinov3_paired" else DINOv3Segmentation
        return model(config.dinov3, backbone_config=backbone_config)
    raise ValueError(f"Unknown downstream model {config.model!r}.")
