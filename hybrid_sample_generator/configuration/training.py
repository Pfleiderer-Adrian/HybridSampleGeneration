from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainingConfiguration:
    """Model-independent optimization and stopping settings."""

    validation_ratio: float = 0.2
    batch_size: int = 64
    epochs: int = 3000
    learning_rate: float = 1e-3
    log_every: int | None = None
    dtype: torch.dtype | None = None
    gradient_clip_norm: float | None = None
    monitor_metric: str | None = None
    early_stopping_enabled: bool = True
    early_stopping: dict[str, Any] = field(
        default_factory=lambda: {"patience": 2000, "delta": 0.0001}
    )
    lr_scheduler_enabled: bool = True
    lr_scheduler: dict[str, Any] = field(
        default_factory=lambda: {"patience": 1000, "factor": 0.1, "threshold": 1e-5}
    )

    def validate(self) -> None:
        if not 0.0 <= float(self.validation_ratio) < 1.0:
            raise ValueError("training.validation_ratio must be in [0, 1).")
        if int(self.batch_size) <= 0 or int(self.epochs) <= 0:
            raise ValueError("training.batch_size and training.epochs must be positive.")
        if float(self.learning_rate) <= 0:
            raise ValueError("training.learning_rate must be positive.")

    def to_dict(self) -> dict[str, Any]:
        values = dict(self.__dict__)
        if self.dtype is not None:
            values["dtype"] = str(self.dtype).removeprefix("torch.")
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfiguration":
        values = dict(values)
        dtype = values.get("dtype")
        if isinstance(dtype, str):
            try:
                values["dtype"] = getattr(torch, dtype.removeprefix("torch."))
            except AttributeError as exc:
                raise ValueError(f"Unknown torch dtype {dtype!r}.") from exc
        return cls(**values)

