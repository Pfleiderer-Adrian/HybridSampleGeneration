"""Model-independent training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


TrialSelection = Literal["best", "last"] | int

# Offered by the configuration schema; any torch dtype name is accepted when loading.
TRAINING_DTYPES = ("float16", "bfloat16", "float32", "float64")


@dataclass
class TrainingConfiguration:
    """Model-independent optimization and stopping settings."""

    num_trials: int = param(10, ge=1, doc="Number of Optuna trials.")
    trial_selection: TrialSelection = param(
        "best", ge=0, doc="Trial to use: 'best', 'last', or a non-negative trial ID.",
    )
    validation_ratio: float = param(
        0.2, ge=0.0, lt=1.0, doc="Fraction of data reserved for validation.",
    )
    batch_size: int = param(64, ge=1, doc="Training batch size.")
    epochs: int = param(1000, ge=1, doc="Maximum number of training epochs.")
    learning_rate: float = param(1e-3, gt=0.0, doc="Initial learning rate.")
    dtype: torch.dtype | None = param(
        None,
        schema={"type": "choice", "choices": list(TRAINING_DTYPES), "nullable": True},
        validated=False,
        doc="PyTorch dtype for training; None uses the model default.",
    )
    gradient_clip_norm: float | None = param(
        None, gt=0.0, doc="Optional upper bound for the gradient norm.",
    )
    monitor_metric: str | None = param(
        "selection", doc="Metric used for model selection and early stopping.",
    )
    early_stopping_enabled: bool = param(True, doc="Enable early stopping.")
    early_stopping: dict[str, Any] = param(
        default_factory=lambda: {"patience": 100, "delta": 0.0001},
        schema={
            "type": "map",
            "values": {"type": "any"},
            "keys": {
                "patience": {"type": "int", "min": 0, "doc": "Epochs without improvement before stopping."},
                "delta": {"type": "float", "min": 0.0, "doc": "Minimum improvement that resets the patience counter."},
            },
        },
        doc="Early stopping settings.",
    )
    lr_scheduler_enabled: bool = param(True, doc="Enable the learning rate scheduler.")
    lr_scheduler: dict[str, Any] = param(
        default_factory=lambda: {"patience": 30, "factor": 0.1, "threshold": 1e-5},
        schema={
            "type": "map",
            "values": {"type": "any"},
            "keys": {
                "patience": {"type": "int", "min": 0, "doc": "Epochs without improvement before the learning rate is reduced."},
                "factor": {"type": "float", "min": 0.0, "min_exclusive": True, "max": 1.0, "max_exclusive": True, "doc": "Learning rate multiplier."},
                "threshold": {"type": "float", "min": 0.0, "doc": "Minimum improvement that counts as progress."},
            },
        },
        doc="Learning rate scheduler settings passed to ReduceLROnPlateau; additional keyword arguments are allowed.",
    )

    def validate(self) -> None:
        validate_parameters(self, "training")

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

