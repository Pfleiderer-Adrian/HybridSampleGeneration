"""Shared interfaces and result types for fusion backends."""

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
    """Capability interface consumed by FusionService for final sample fusion."""

    def warmup(self, shape, device=None, dtype=None, config=None):
        ...

    def fuse(
        self,
        sample: dict[str, Any],
        control_img: np.ndarray,
        position: Any,
        *,
        extraction_config=None,
    ) -> FusionOutput:
        ...
