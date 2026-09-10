"""Batch and epoch loops for trainable generation models."""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from hybrid_sample_generator.configuration.training import TrainingConfiguration
from hybrid_sample_generator.generation.interfaces import StepOutput
from hybrid_sample_generator.generation.training.metrics import (
    average_metric_dicts,
    current_lr,
    format_epoch_log,
    metric_value,
    metrics_to_float,
    step_scheduler,
    to_float,
)


class EarlyStoppingTracker:
    """Track non-improving epochs without implicitly writing checkpoints."""

    def __init__(self, *, patience=0, delta=0.0, **_ignored):
        self.patience = int(patience)
        self.delta = float(delta)
        self.best = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def extract_step_output(output):
    if not isinstance(output, StepOutput):
        raise TypeError(
            "training_step/validation_step must return "
            "hybrid_sample_generator.generation.interfaces.StepOutput."
        )
    return output.loss, dict(output.metrics)


def run_epoch(
    model,
    loader,
    optimizer,
    config: TrainingConfiguration,
    device,
    *,
    training: bool,
) -> dict:
    model.train(training)
    step_fn = model.training_step if training else model.validation_step
    metric_dicts = []
    iterator = tqdm(
        loader,
        desc="train" if training else "val",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(iterator):
        batch = move_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = step_fn(batch, batch_idx, config)
            loss, metrics = extract_step_output(output)
            if training:
                loss.backward()
                if config.gradient_clip_norm is not None and config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()
        metrics = metrics_to_float(metrics)
        if "loss" not in metrics:
            metrics["loss"] = to_float(loss)
        metric_dicts.append(metrics)
        try:
            value = metric_value(metrics, preferred_key=config.monitor_metric)
            iterator.set_postfix(value=f"{value:.6f}")
        except ValueError:
            pass
    return average_metric_dicts(metric_dicts)


def train(
    model,
    train_loader,
    val_loader,
    config: TrainingConfiguration,
    *,
    anomaly_size,
    best_model_path=None,
):
    """Train through the common batch-level generative model interface."""
    train_history = []
    val_history = []
    best_epoch = 0
    best_val = float("inf")

    with tqdm(desc="epoch", total=config.epochs) as progress:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.warmup(anomaly_size, device=device, dtype=config.dtype, config=config)
        optimizer, scheduler = model.configure_optimizers(config)
        if optimizer is None:
            raise ValueError(
                f"{model.__class__.__name__}.configure_optimizers() returned no optimizer."
            )
        if scheduler is None and config.lr_scheduler_enabled:
            scheduler = ReduceLROnPlateau(optimizer, "min", **config.lr_scheduler)
        early_stopping = (
            EarlyStoppingTracker(**config.early_stopping)
            if config.early_stopping_enabled
            else None
        )

        for epoch in range(config.epochs):
            model.on_epoch_start(epoch, config=config)
            train_metrics = run_epoch(
                model, train_loader, optimizer, config, device, training=True
            )
            val_metrics = {}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = run_epoch(
                        model, val_loader, optimizer, config, device, training=False
                    )
            if not val_metrics:
                val_metrics = train_metrics

            train_value = metric_value(train_metrics, preferred_key=config.monitor_metric)
            val_value = metric_value(val_metrics, preferred_key=config.monitor_metric)
            train_history.append(train_value)
            val_history.append(val_value)
            if val_value < best_val:
                best_val = float(val_value)
                print("New best epoch! val_loss: " + str(best_val))
                best_epoch = epoch + 1
                if best_model_path is not None:
                    model.save_checkpoint(
                        best_model_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=best_epoch,
                        metrics={"train": train_metrics, "val": val_metrics},
                    )

            lr = current_lr(optimizer, scheduler)
            tqdm.write(format_epoch_log(epoch + 1, lr, train_metrics, val_metrics))
            progress.set_postfix(
                lr=f"{lr:.5f}", train=f"{train_value:.4f}", val=f"{val_value:.4f}"
            )
            progress.update(1)
            step_scheduler(scheduler, val_value)
            if early_stopping is not None and early_stopping.step(val_value):
                print("Early stopping")
                break
    return train_history, val_history, best_epoch, best_val


__all__ = ["EarlyStoppingTracker", "extract_step_output", "move_to_device", "run_epoch", "train"]
