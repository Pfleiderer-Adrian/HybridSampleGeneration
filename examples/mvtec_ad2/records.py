"""Records used by the MVTec AD 2 example integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examples.mvtec_ad2.configuration import Configuration
    from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader


@dataclass(frozen=True)
class MVTecAD2Sample:
    image_path: Path
    mask_path: Path | None
    sample_id: str
    split: str
    label: str


@dataclass(frozen=True)
class MVTecAD2Study:
    category: str
    category_root: Path | None
    config: Configuration
    sample_dataloader: "MVTecAD2Dataloader | None"
    split_manifest: dict | None = None


@dataclass(frozen=True)
class WorkflowResult:
    study_folder: Path
    downstream_run_folder: Path | None = None
