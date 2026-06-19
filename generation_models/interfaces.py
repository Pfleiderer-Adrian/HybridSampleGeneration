from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class StepOutput:
    """Return value for one train/validation step."""

    loss: torch.Tensor
    metrics: dict[str, Any]


@runtime_checkable
class TrainableModule(Protocol):
    """Capability interface consumed by synthesizer.Trainer."""

    def warmup(self, shape, device=None, dtype=None, config=None):
        ...

    def training_step(self, batch, batch_idx: int, config=None) -> StepOutput:
        ...

    def validation_step(self, batch, batch_idx: int, config=None) -> StepOutput:
        ...

    def on_epoch_start(self, epoch: int, config=None) -> None:
        ...

    def configure_optimizers(self, config) -> tuple[torch.optim.Optimizer, Any | None]:
        ...

    def save_checkpoint(self, path: str, **state) -> None:
        ...

    def load_checkpoint(self, path: str, **kwargs) -> None:
        ...


@runtime_checkable
class GenerativeBackend(Protocol):
    """Capability interface consumed by HybridDataGenerator."""

    def warmup(self, shape, device=None, dtype=None, config=None):
        ...

    def load_checkpoint(self, path: str, **kwargs) -> None:
        ...

    def generate(self, sample, *, mode: str, **kwargs):
        ...
