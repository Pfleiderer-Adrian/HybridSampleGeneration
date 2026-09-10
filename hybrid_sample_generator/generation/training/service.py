"""Backward-compatible facade for generation training."""

from hybrid_sample_generator.generation.training.loop import train
from hybrid_sample_generator.generation.training.optuna import (
    objective,
    optimize,
    sample_model_params,
)

__all__ = ["objective", "optimize", "sample_model_params", "train"]
