"""Downstream configurations of the V3 MVTec AD 2 benchmark submissions.

All submitted variants share one DINOv3 configuration and differ only in the
per-epoch source draws (1000 draws per epoch).
"""

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.settings import TEXTURE_ROOT

MIXTURES = {
    "real_only": {"normal": 500, "real": 500},
    "real_draem": {"normal": 500, "real": 100, "draem": 400},
    "real_cutmix": {"normal": 500, "real": 100, "cutmix": 400},
    "real_hybrid": {"normal": 500, "real": 100, "hybrid": 400},
    "real_draem_hybrid": {"normal": 500, "real": 100, "hybrid": 200, "draem": 200},
    "real_anomalydiffusion": {"normal": 500, "real": 100, "anomalydiffusion": 400},
    "real_anomalydiffusion_hybrid": {"normal": 500, "real": 100, "anomalydiffusion": 200, "hybrid": 200},
    "real_anomalydiffusion_draem": {"normal": 500, "real": 100, "anomalydiffusion": 200, "draem": 200},
    "real_anomalydiffusion_draem_hybrid": {
        "normal": 500, "real": 100, "anomalydiffusion": 134, "draem": 133, "hybrid": 133,
    },
}
# Rotates the rounding remainder of the synthetic budget across epochs.
ROTATIONS = {"real_anomalydiffusion_draem_hybrid": ["anomalydiffusion", "draem", "hybrid"]}
DEFAULT_MIXTURE = "real_draem_hybrid"
DINOV3_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"


def create_downstream_configuration(
    mixture: str = DEFAULT_MIXTURE, *, anomalydiffusion_manifest=None
) -> DownstreamConfiguration:
    """Return the submitted V3 downstream configuration for one source mixture."""
    if mixture not in MIXTURES:
        raise ValueError(f"Unknown mixture {mixture!r}; choose one of {sorted(MIXTURES)}.")
    quotas = dict(MIXTURES[mixture])
    if quotas.get("anomalydiffusion") and anomalydiffusion_manifest is None:
        raise ValueError(f"Mixture {mixture!r} requires an AnomalyDiffusion pair manifest.")

    config = DownstreamConfiguration(model="dinov3_paired", seed=42)
    data = config.data
    data.source_quotas = quotas
    data.source_quota_rotation = list(ROTATIONS[mixture]) if mixture in ROTATIONS else None
    data.anomalydiffusion_manifest = None if anomalydiffusion_manifest is None else str(anomalydiffusion_manifest)
    data.include_real_defects = True
    data.samples_per_epoch = 1000
    data.mode = "patch"
    data.image_size = (256, 256)
    data.patch_size = (256, 256)
    data.patch_overlap = 0.5
    data.image_scale = 255.0
    data.texture_root = TEXTURE_ROOT

    training = config.training
    training.epochs = 100
    training.validation_interval = 5
    training.selection_metric = "pixel_aupro_0.05"
    training.batch_size = 8
    training.learning_rate = 1e-4
    training.num_workers = 2
    training.device = "auto"
    training.reconstruction_width = 128
    training.segmentation_width = 64

    dinov3 = config.dinov3
    dinov3.pair_healthy_samples = False
    dinov3.backbone_type = "vit"
    dinov3.backbone = DINOV3_BACKBONE
    dinov3.revision = DINOV3_REVISION
    dinov3.decoder = "cnn"
    dinov3.decoder_width = 64
    dinov3.decoder_blocks = 2
    dinov3.healthy_loss_weight = 1.0
    dinov3.consistency_weight = 1.0
    dinov3.dice_weight = 1.0
    dinov3.validation_samples = 128

    config.validate()
    return config
