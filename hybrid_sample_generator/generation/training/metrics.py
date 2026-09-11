"""Metric conversion, aggregation, scheduling, and logging helpers."""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def current_lr(optimizer, scheduler=None) -> float:
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        lrs = scheduler.get_last_lr()
        if lrs:
            return float(lrs[0])
    if optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


def step_scheduler(scheduler, metric: float):
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


def to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return float(value.item())
        return float(value.float().mean().item())
    return float(value)


def metric_value(metrics: dict, *, preferred_key=None) -> float:
    if not metrics:
        raise ValueError("Model returned no training metrics.")
    for key in (
        preferred_key,
        "total",
        "loss",
        "objective",
        "val_loss",
        "train_loss",
    ):
        if key and key in metrics:
            return to_float(metrics[key])
    for value in metrics.values():
        try:
            return to_float(value)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Could not find a scalar metric in: {metrics}")


def format_metrics(metrics: dict) -> str:
    parts = []
    for key, value in metrics.items():
        try:
            parts.append(f"{key}: {to_float(value):.4f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(parts) if parts else "no scalar metrics"


def format_epoch_log(epoch, lr, train_metrics, val_metrics):
    return (
        f"\nEpoch {epoch:03d}: lr: {lr:0.5f}, "
        f"train [{format_metrics(train_metrics)}], "
        f"val [{format_metrics(val_metrics)}]"
    )


def metrics_to_float(metrics: dict) -> dict:
    converted = {}
    for key, value in metrics.items():
        try:
            converted[key] = to_float(value)
        except (TypeError, ValueError):
            continue
    return converted


def average_metric_dicts(metric_dicts: list[dict]) -> dict:
    if not metric_dicts:
        return {}
    sums = {}
    counts = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


__all__ = [
    "average_metric_dicts",
    "current_lr",
    "format_epoch_log",
    "format_metrics",
    "metric_value",
    "metrics_to_float",
    "step_scheduler",
    "to_float",
]
