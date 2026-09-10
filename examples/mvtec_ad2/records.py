"""Records used by the MVTec AD 2 example integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hybrid_sample_generator.configuration.root import Configuration

if TYPE_CHECKING:
    from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader


@dataclass(frozen=True)
class MVTecAD2Sample:
    image_path: Path
    mask_path: Path | None
    sample_id: str
    split: str
    label: str


@dataclass(frozen=True)
class MVTecAD2UseCase:
    category: str
    category_root: Path
    config: Configuration
    sample_dataloader: "MVTecAD2Dataloader"
