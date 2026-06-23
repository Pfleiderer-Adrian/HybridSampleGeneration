from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class FusionOutput:
    """Return value for one fusion operation."""

    image: np.ndarray
    segmentation: np.ndarray
    roi: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    metrics: dict[str, Any] | None = None


@runtime_checkable
class FusionBackend(Protocol):
    """Capability interface consumed by HybridDataGenerator for final sample fusion."""

    def warmup(self, shape, device=None, dtype=None, config=None):
        ...

    def load_checkpoint(self, path: str, **kwargs) -> None:
        ...

    def train_model(
        self,
        sample_dataloader,
        *,
        epochs: int | None = None,
        lr: float | None = None,
        checkpoint_path: str | None = None,
        device=None,
        config=None,
    ) -> dict:
        ...

    def fuse(
        self,
        control,
        anomaly,
        anomaly_meta,
        position,
        *,
        target_mask,
        config=None,
        **kwargs,
    ) -> FusionOutput:
        ...
