"""DRAEM configuration types and run snapshots."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfiguration:
    # Explicit quotas override legacy fractions and appended real examples.
    source_quotas: dict[str, int] | None = None
    include_real_defects: bool = False
    hybrid_fraction: float = 0.5
    normal_fraction: float = 0.5
    hard_normal_fraction: float = 0.0
    hard_normal_manifest: str | None = None
    samples_per_epoch: int = 1000
    image_size: tuple[int, int] = (256, 256)
    mode: str = "patch"
    patch_size: tuple[int, int] = (512, 512)
    patch_overlap: float = 0.5
    texture_root: str | None = None
    # MVTec image arrays (including persisted originals/hybrids) use 0..255.
    image_scale: float = 255.0


@dataclass
class TrainingConfiguration:
    epochs: int = 100
    validation_interval: int = 1
    selection_metric: str = "auto"
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_workers: int = 4
    device: str = "auto"
    reconstruction_width: int = 128
    segmentation_width: int = 64


@dataclass
class DINOv3Configuration:
    pair_healthy_samples: bool = True
    backbone: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    revision: str = "main"
    decoder_width: int = 64
    decoder: str = "cnn"
    decoder_blocks: int = 2
    healthy_loss_weight: float = 1.0
    consistency_weight: float = 1.0
    dice_weight: float = 1.0
    validation_samples: int = 128


@dataclass
class MemoryBankConfiguration:
    layers: tuple[int, ...] = (3, 6, 9, 12)
    normal_capacity: int = 20000
    defect_capacity: int = 5000
    tokens_per_region: int = 64
    query_chunk_size: int = 256
    reference_chunk_size: int = 2048
    defect_weight: float = 1.0
    calibration_images: int = 16


@dataclass
class DownstreamConfiguration:
    data: DataConfiguration = field(default_factory=DataConfiguration)
    training: TrainingConfiguration = field(default_factory=TrainingConfiguration)
    seed: int = 42
    model: str = "draem"
    dinov3: DINOv3Configuration = field(default_factory=DINOv3Configuration)
    memory_bank: MemoryBankConfiguration = field(default_factory=MemoryBankConfiguration)

    def validate(self):
        quotas = self.data.source_quotas
        if quotas is not None:
            if (not isinstance(quotas, dict) or not quotas
                    or set(quotas) - {"normal", "real", "draem", "cutmix", "hybrid"}
                    or any(type(n) is not int or n < 0 for n in quotas.values())
                    or sum(quotas.values()) != self.data.samples_per_epoch):
                raise ValueError("source_quotas must contain nonnegative source counts summing to samples_per_epoch.")
            if self.model not in ("dinov3_paired", "dinov3_segmentation"):
                raise ValueError("Explicit source quotas require a DINOv3 segmentation model.")
            if (quotas.get("real", 0) or quotas.get("cutmix", 0)) and not self.data.include_real_defects:
                raise ValueError("Real/CutMix quotas require include_real_defects.")
            if self.data.hard_normal_fraction:
                raise ValueError("Explicit source quotas currently exclude hard-normal sampling.")
        if type(self.dinov3.pair_healthy_samples) is not bool:
            raise ValueError("pair_healthy_samples must be boolean.")
        if self.training.selection_metric not in ("auto", "pixel_aupro_0.05"):
            raise ValueError("selection_metric must be auto or pixel_aupro_0.05.")
        if self.model not in ("draem", "dinov3_paired", "dinov3_segmentation", "superadd_dual_bank"):
            raise ValueError("Unknown downstream model.")
        if type(self.data.include_real_defects) is not bool:
            raise ValueError("include_real_defects must be boolean.")
        if self.data.include_real_defects and self.model not in ("dinov3_paired", "dinov3_segmentation"):
            raise ValueError("Real defect augmentation is supported for DINOv3 segmentation models.")
        bank = self.memory_bank
        if not bank.layers or any(type(x) is not int or x < 1 for x in bank.layers) or len(set(bank.layers)) != len(bank.layers):
            raise ValueError("Memory-bank layers must be distinct positive transformer layer numbers.")
        for name in ("normal_capacity", "defect_capacity", "tokens_per_region", "query_chunk_size", "reference_chunk_size", "calibration_images"):
            if type(getattr(bank, name)) is not int or getattr(bank, name) < 1:
                raise ValueError(f"Memory-bank {name} must be a positive integer.")
        if not math.isfinite(bank.defect_weight) or bank.defect_weight < 0:
            raise ValueError("Memory-bank defect_weight must be finite and nonnegative.")
        if not self.dinov3.backbone or not self.dinov3.revision:
            raise ValueError("DINOv3 backbone and revision must be nonempty.")
        if type(self.dinov3.decoder_width) is not int or self.dinov3.decoder_width <= 0:
            raise ValueError("DINOv3 decoder_width must be a positive integer.")
        if self.dinov3.decoder not in ("cnn", "dpt", "residual_fpn", "convnext_fpn", "cnn_residual", "cnn_multilayer"):
            raise ValueError("Unknown DINOv3 decoder.")
        if type(self.dinov3.decoder_blocks) is not int or self.dinov3.decoder_blocks < 1:
            raise ValueError("DINOv3 decoder_blocks must be a positive integer.")
        if type(self.dinov3.validation_samples) is not int or self.dinov3.validation_samples <= 0:
            raise ValueError("DINOv3 validation_samples must be a positive integer.")
        for name in ("healthy_loss_weight", "consistency_weight", "dice_weight"):
            value = getattr(self.dinov3, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"DINOv3 {name} must be finite and nonnegative.")
        for name, value in (
            ("hybrid_fraction", self.data.hybrid_fraction),
            ("normal_fraction", self.data.normal_fraction),
            ("hard_normal_fraction", self.data.hard_normal_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.data.hard_normal_fraction > 0:
            if self.data.mode != "patch" or not self.data.hard_normal_manifest or self.data.normal_fraction == 0:
                raise ValueError("Hard-normal sampling requires patch mode, a manifest and normal examples.")
        for name, value in (
            ("samples_per_epoch", self.data.samples_per_epoch),
            ("epochs", self.training.epochs), ("batch_size", self.training.batch_size),
            ("validation_interval", self.training.validation_interval),
            ("reconstruction_width", self.training.reconstruction_width),
            ("segmentation_width", self.training.segmentation_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if type(self.training.num_workers) is not int or self.training.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        for value in (self.training.learning_rate, self.data.image_scale):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("learning_rate and image_scale must be finite and positive.")
        if self.data.mode not in ("patch", "image"):
            raise ValueError("data.mode must be patch or image.")
        for name in ("image_size", "patch_size"):
            size = getattr(self.data, name)
            if len(size) != 2 or any(type(n) is not int or n < 64 or n % 32 for n in size):
                raise ValueError(f"{name} dimensions must be multiples of 32, at least 64.")
        if type(self.data.patch_overlap) not in (int, float) or not 0 <= self.data.patch_overlap < 1:
            raise ValueError("patch_overlap must be in [0, 1).")
        if self.data.normal_fraction == 1 and self.model != "superadd_dual_bank":
            raise ValueError("Training requires anomalous examples (normal_fraction < 1).")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, kind in (("data", DataConfiguration), ("training", TrainingConfiguration),
                          ("dinov3", DINOv3Configuration), ("memory_bank", MemoryBankConfiguration)):
            values[key] = kind(**values.get(key, {}))
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
