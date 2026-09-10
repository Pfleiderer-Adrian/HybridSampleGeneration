from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoiConfiguration:
    """Rules for constructing inspection ROIs around an anomaly."""

    fixed_size: tuple[int, ...] | None = None
    min_size: tuple[int, ...] | int = 0
    min_padding: tuple[int, ...] = (20, 20, 20)
    padding_ratio: tuple[float, ...] = (0.5, 0.5, 0.5)

    def validate(self) -> None:
        if self.fixed_size is not None and any(int(value) <= 0 for value in self.fixed_size):
            raise ValueError("extraction.roi.fixed_size values must be positive.")
        if any(int(value) < 0 for value in self.min_padding):
            raise ValueError("extraction.roi.min_padding values must be non-negative.")
        if any(float(value) < 0 for value in self.padding_ratio):
            raise ValueError("extraction.roi.padding_ratio values must be non-negative.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RoiConfiguration":
        values = dict(values)
        for key in ("fixed_size", "min_padding", "padding_ratio"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        if isinstance(values.get("min_size"), list):
            values["min_size"] = tuple(values["min_size"])
        return cls(**values)


@dataclass
class ExtractionConfiguration:
    """Settings used only while extracting real anomaly artifacts."""

    anomaly_size: tuple[int, ...]
    separate_components: bool = True
    min_coverage_ratio: float = 0.05
    add_background_noise: bool = True
    normalization: str = "z-score"
    normalization_eps: float = 1e-6
    roi: RoiConfiguration = field(default_factory=RoiConfiguration)

    def __post_init__(self) -> None:
        self.anomaly_size = tuple(int(value) for value in self.anomaly_size)

    def validate(self) -> None:
        if len(self.anomaly_size) not in (3, 4) or any(value <= 0 for value in self.anomaly_size):
            raise ValueError(
                "extraction.anomaly_size must be positive and have shape (C,H,W) or (C,D,H,W)."
            )
        if not 0.0 <= float(self.min_coverage_ratio) <= 1.0:
            raise ValueError("extraction.min_coverage_ratio must be in [0, 1].")
        if float(self.normalization_eps) <= 0:
            raise ValueError("extraction.normalization_eps must be positive.")
        self.roi.validate()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExtractionConfiguration":
        values = dict(values)
        values["anomaly_size"] = tuple(values["anomaly_size"])
        values["roi"] = RoiConfiguration.from_dict(values.get("roi", {}))
        return cls(**values)

