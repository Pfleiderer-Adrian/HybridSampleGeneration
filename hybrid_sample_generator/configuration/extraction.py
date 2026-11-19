"""Configuration models for anomaly extraction and ROI construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


@dataclass
class RoiConfiguration:
    """Rules for constructing inspection ROIs around an anomaly."""

    fixed_size: tuple[int, ...] | None = param(
        None, ge=1, doc="Fixed spatial ROI size; None selects dynamic sizing.",
    )
    min_size: tuple[int, ...] | int = param(
        0, ge=0, doc="Minimum dynamic ROI size, either scalar or per axis.",
    )
    min_padding: tuple[int, ...] = param(
        (10, 10, 10), ge=0, doc="Minimum ROI padding around the anomaly on each axis.",
    )
    padding_ratio: tuple[float, ...] = param(
        (0.5, 0.5, 0.5),
        ge=0.0,
        doc="Additional ROI padding relative to anomaly size on each axis.",
    )
    preserve_aspect_ratio: bool = param(
        False,
        doc="Expand dynamic ROIs to match the spatial aspect ratio of the full image sample.",
    )

    def validate(self) -> None:
        validate_parameters(self, "extraction.roi")

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

    anomaly_size: tuple[int, ...] = param(
        (3, 64, 64),
        ge=1,
        min_length=3,
        max_length=4,
        doc="Target anomaly crop shape (C,H,W) or (C,D,H,W); must match the model dimensions.",
    )
    separate_components: bool = param(
        True, doc="Extract each connected anomaly component separately.",
    )
    min_coverage_ratio: float = param(
        0.01, ge=0.0, le=1.0, doc="Minimum mask coverage within a crop.",
    )
    add_background_noise: bool = param(
        True, doc="Add a small amount of noise to otherwise constant backgrounds.",
    )
    preserve_aspect_ratio: bool = param(
        True,
        doc="Use one spatial scale for every axis and center-pad the remaining area; False stretches axes independently.",
    )
    normalization: str | None = param(
        "z-score",
        schema={"type": "choice", "choices": ["z-score", "zscore_median"], "nullable": True},
        validated=False,
        doc="Intensity normalization method; None disables normalization. Spelling variants of the listed names are accepted.",
    )
    normalization_eps: float = param(
        1e-6, gt=0.0, doc="Lower bound for numerically stable normalization.",
    )
    roi: RoiConfiguration = param(default_factory=RoiConfiguration)

    def __post_init__(self) -> None:
        self.anomaly_size = tuple(int(value) for value in self.anomaly_size)

    def validate(self) -> None:
        validate_parameters(self, "extraction")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExtractionConfiguration":
        values = dict(values)
        values["anomaly_size"] = tuple(values["anomaly_size"])
        values["roi"] = RoiConfiguration.from_dict(values.get("roi", {}))
        return cls(**values)
