from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from synthesizer.mask_manipulation import TransformGenerator


@dataclass
class AugmentationConfiguration:
    """Training offsets and target-mask transformations."""

    mask_transforms: TransformGenerator.Config = field(default_factory=TransformGenerator.Config)
    random_offset_enabled: bool = True
    random_offset_max_fraction: float = 1.0
    random_offset_foreground_threshold: float = 0.001

    def validate(self) -> None:
        if not 0.0 <= float(self.random_offset_max_fraction) <= 1.0:
            raise ValueError("augmentation.random_offset_max_fraction must be in [0, 1].")
        if float(self.random_offset_foreground_threshold) < 0:
            raise ValueError("augmentation.random_offset_foreground_threshold must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_transforms": asdict(self.mask_transforms),
            "random_offset_enabled": self.random_offset_enabled,
            "random_offset_max_fraction": self.random_offset_max_fraction,
            "random_offset_foreground_threshold": self.random_offset_foreground_threshold,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AugmentationConfiguration":
        values = dict(values)
        mask_values = dict(values.pop("mask_transforms", {}))
        values["mask_transforms"] = TransformGenerator.Config(**mask_values)
        return cls(**values)

