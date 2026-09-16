"""DRAEM configuration types and run snapshots."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfiguration:
    hybrid_fraction: float = 0.5
    normal_fraction: float = 0.5
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
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    reconstruction_width: int = 128
    segmentation_width: int = 64


@dataclass
class DownstreamConfiguration:
    data: DataConfiguration = field(default_factory=DataConfiguration)
    training: TrainingConfiguration = field(default_factory=TrainingConfiguration)
    seed: int = 42

    def validate(self):
        for name, value in (
            ("hybrid_fraction", self.data.hybrid_fraction),
            ("normal_fraction", self.data.normal_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for name, value in (
            ("samples_per_epoch", self.data.samples_per_epoch),
            ("epochs", self.training.epochs), ("batch_size", self.training.batch_size),
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
        if self.data.normal_fraction == 1:
            raise ValueError("Training requires anomalous examples (normal_fraction < 1).")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, kind in (("data", DataConfiguration), ("training", TrainingConfiguration)):
            values[key] = kind(**values.get(key, {}))
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

