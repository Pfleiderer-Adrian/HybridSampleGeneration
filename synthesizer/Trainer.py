import os
import optuna
import torch
from optuna import Trial
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from generation_models.interfaces import StepOutput
from generation_models.model_configuration import ModelConfiguration
from generation_models.model_registry import get_model_spec
from synthesizer.Configuration import Configuration


class _TrainingTransformDataset(Dataset):
    """
    Lightweight wrapper that applies a transform only for model training.

    The underlying anomaly dataset is also reused later for synthetic anomaly
    generation, so the persisted samples themselves must remain unmodified.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and item:
            return self.transform.transform_sample(item)
        if isinstance(item, list) and item:
            return list(self.transform.transform_sample(tuple(item)))
        if isinstance(item, dict):
            return self.transform.transform_sample(item)
        return self.transform(item)


class _RandomSpatialOffset:
    """
    Randomly translate an anomaly inside its fixed canvas without cropping it.

    The foreground bbox is estimated from the current tensor, then the allowed
    shift range is restricted so the foreground remains fully inside the canvas.
    This replaces the former extraction-time offset while keeping saved cutouts
    centered for generation and fusion.
    """

    def __init__(
        self,
        *,
        max_fraction: float = 1.0,
        foreground_threshold_rel: float = 0.001,
    ) -> None:
        self.max_fraction = max(0.0, min(float(max_fraction), 1.0))
        self.foreground_threshold_rel = max(float(foreground_threshold_rel), 0.0)

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        shifts = self._sample_shifts(x)
        if shifts is None:
            return x

        return self._apply_shift(x, shifts, fill_value=torch.amin(x))

    # gets whole item (masks too if multiclass)
    def transform_sample(self, item):
        if isinstance(item, dict):
            return self._transform_dict_sample(item)
        if not item:
            return item

        x = item[0]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        shifts = self._sample_shifts(x)
        if shifts is None:
            return item

        shifted = [self._apply_shift(x, shifts, fill_value=torch.amin(x))]
        for value in item[1:]:
            shifted.append(self._maybe_apply_shift(value, shifts))
        return tuple(shifted)

    def _transform_dict_sample(self, item):
        item = dict(item)
        x_key = next((key for key in ("img", "x", "image", "inputs") if key in item), None)
        if x_key is None:
            return item

        x = item[x_key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        shifts = self._sample_shifts(x)
        if shifts is None:
            return item

        item[x_key] = self._apply_shift(x, shifts, fill_value=torch.amin(x))
        for key in ("ori_mask", "org_mask", "mask", "tgt_mask", "target_mask"):
            if key in item:
                item[key] = self._maybe_apply_shift(item[key], shifts)
        return item

    # get exact offset
    def _sample_shifts(self, x):
        if x.ndim not in (3, 4):
            return None

        x_min = torch.amin(x)
        x_max = torch.amax(x)
        dynamic_range = x_max - x_min
        if not bool(torch.isfinite(dynamic_range)) or float(dynamic_range) <= 0.0:
            return None

        threshold = x_min + dynamic_range * self.foreground_threshold_rel
        foreground = torch.any(x > threshold, dim=0)
        if not bool(torch.any(foreground)):
            return None

        coords = torch.nonzero(foreground, as_tuple=False)
        spatial_min = coords.min(dim=0).values.tolist()
        spatial_max = coords.max(dim=0).values.tolist()

        shifts: list[int] = []
        for axis, size in enumerate(x.shape[1:]):
            min_shift = -int(spatial_min[axis])
            max_shift = int(size - 1 - spatial_max[axis])
            min_shift = int(round(min_shift * self.max_fraction))
            max_shift = int(round(max_shift * self.max_fraction))

            if min_shift == max_shift:
                shifts.append(min_shift)
            else:
                shifts.append(int(torch.randint(min_shift, max_shift + 1, ()).item()))

        if not any(shifts):
            return None

        return shifts

    def _maybe_apply_shift(self, value, shifts):
        if not isinstance(value, torch.Tensor):
            return value    # only shift torch.Tensor

        spatial_ndim = len(shifts)
        if value.ndim == spatial_ndim:
            return self._apply_shift(value, shifts, fill_value=0, has_channel_dim=False)
        if value.ndim == spatial_ndim + 1:
            return self._apply_shift(value, shifts, fill_value=0, has_channel_dim=True)
        return value

    def _apply_shift(self, x, shifts, *, fill_value, has_channel_dim=True):
        shifted = torch.empty_like(x)
        if isinstance(fill_value, torch.Tensor):
            fill_value = fill_value.item()
        shifted.fill_(fill_value)

        src_slices = [slice(None)] if has_channel_dim else []
        dst_slices = [slice(None)] if has_channel_dim else []
        spatial_shape = x.shape[1:] if has_channel_dim else x.shape
        for shift, size in zip(shifts, spatial_shape):
            if shift >= 0:
                src_slices.append(slice(0, size - shift))
                dst_slices.append(slice(shift, size))
            else:
                src_slices.append(slice(-shift, size))
                dst_slices.append(slice(0, size + shift))

        shifted[tuple(dst_slices)] = x[tuple(src_slices)]
        return shifted


def optimize(no_of_trials, config:Configuration, dataset):
    """
    Run Optuna hyperparameter optimization for a generative model.

    This function:
      1) Creates (or reuses) an Optuna study stored via `config.get_paths()`.
      2) Runs `objective(...)` for `no_of_trials` trials.
      3) Prints summary information about the best trial found.

    Inputs
    ------
    no_of_trials:
        Number of Optuna trials to run (each trial trains one model instance).
    config:
        Global configuration containing:
          - study_folder, study_name
          - model_name, model_params search space
          - training params (epochs, lr, batch_size, val_ratio, ...)
    dataset:
        PyTorch-style dataset (must implement __len__ and __getitem__),
        expected to yield anomaly cutouts (inputs for training).

    Outputs
    -------
    None
        Side effects:
          - Creates `<study_folder>/<study_name>.db` (Optuna SQLite storage)
          - Trains and saves models per trial into `<study_folder>/trained_models/`
          - Prints best-trial statistics to stdout.
    """
    paths = config.get_paths()
    os.makedirs(paths.study_folder, exist_ok=True)

    study = optuna.create_study(study_name=config.study_name, direction='minimize', load_if_exists=True,
                                storage=paths.optuna_storage_url)

    # Run optimization process
    func = lambda trial: objective(trial, config, dataset)
    study.optimize(func, n_trials=no_of_trials)  # calls objective function multiple time and tries to minimize the return value of this function

    # Print Results
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



def objective(trial: Trial, config: Configuration, dataset):
    """
    Optuna objective function: samples hyperparameters, trains a model, saves it, and returns a score.

    The objective:
      1) Builds a hyperparameter set from config.model_params.
      2) Instantiates the model architecture specified by `config.model_name`.
      3) Splits dataset into train/val and trains for up to `config.epochs`.
      4) Stores hyperparameters + paths as Optuna user attributes for reproducibility.
      5) Returns the minimum validation loss observed during training.

    Inputs
    ------
    trial:
        Optuna Trial object (used to sample hyperparameters and store metadata).
    config:
        Configuration (search space + training params).
    dataset:
        Dataset used for training/validation split.

    Outputs
    -------
    float
        The objective value (lower is better): min(validation_loss_over_epochs).

    Side effects
    ------------
    - Saves model weights to `<study_folder>/trained_models/model_trial_<trial.number>.pth`
    - Writes trial metadata into the Optuna DB (user_attrs)
    """
    # 1. set hyperparameter tuning space via optuna
    params = sample_model_params(trial, config.model_params)


    # 2. Create the model with chosen hyperparameters
    model = get_model_spec(config.model_name).build(params)

    n_val = int(len(dataset) * config.val_ratio)
    n_train = len(dataset) - n_val

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    train_ds = _apply_training_offset_augmentation(train_ds, config)

    _anomaly_train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
    _anomaly_val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    # 3. Start training of the model
    directory = config.get_paths().trained_models
    os.makedirs(directory, exist_ok=True)
    best_model_path = os.path.join(directory, f"model_trial_{trial.number}_best.pth")

    train_losses, val_losses, best_epoch, best_val = train(
        model=model,
        train_loader=_anomaly_train_loader,
        val_loader=_anomaly_val_loader,
        config=config,
        best_model_path=best_model_path,
    )

    for key, value in params.items():
        trial.set_user_attr(key, value)

    # save trained model (best epoch)
    model_path = best_model_path
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", float(best_val))
    trial.set_user_attr("params", params)
    trial.set_user_attr("model_name", config.model_name)

    # lowest validation/monitor error over all epochs
    return min(val_losses) if val_losses else float(best_val)


def _apply_training_offset_augmentation(dataset, config: Configuration):
    if not config.random_offset:
        return dataset

    transform = _RandomSpatialOffset(
        max_fraction=config.random_offset_max_fraction,
        foreground_threshold_rel=config.random_offset_foreground_threshold_rel,
    )
    return _TrainingTransformDataset(dataset, transform)


def sample_model_params(trial: Trial, model_params):
    """
    Build concrete model params from a ModelConfiguration search space.

    Equal min/max values are treated as fixed constants. Strings and booleans
    become categorical choices when min/max differ.
    """
    model_params = ModelConfiguration.from_value(model_params)
    min_params = model_params.min
    max_params = model_params.max
    params = {}

    for key in sorted(set(min_params) | set(max_params)):
        min_value = min_params.get(key)
        max_value = max_params.get(key, min_value)
        params[key] = _sample_or_fix_param(trial, key, min_value, max_value)

    return params


def _sample_or_fix_param(trial: Trial, key, min_value, max_value):
    if min_value == max_value:
        return min_value

    if isinstance(min_value, bool) and isinstance(max_value, bool):
        return trial.suggest_categorical(key, _unique_choices([min_value, max_value]))

    if isinstance(min_value, int) and isinstance(max_value, int):
        low, high = sorted((min_value, max_value))
        return trial.suggest_int(key, low, high)

    if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
        low, high = sorted((float(min_value), float(max_value)))
        return trial.suggest_float(key, low, high)

    if isinstance(min_value, (list, tuple, set)) and max_value is None:
        return trial.suggest_categorical(key, list(min_value))

    return trial.suggest_categorical(key, _unique_choices([min_value, max_value]))


def _unique_choices(values):
    choices = []
    for value in values:
        if value not in choices:
            choices.append(value)
    return choices


class _EarlyStoppingTracker:
    """Small early-stopping helper that does not implicitly checkpoint the model."""

    def __init__(self, *, patience=0, delta=0.0, **_ignored):
        self.patience = int(patience)
        self.delta = float(delta)
        self.best = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < (self.best - self.delta):
            self.best = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def _current_lr(optimizer, scheduler=None) -> float:
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        lrs = scheduler.get_last_lr()
        if lrs:
            return float(lrs[0])
    if optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


def _step_scheduler(scheduler, metric: float):
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return float(value.item())
        return float(value.float().mean().item())
    return float(value)


def _metric_value(metrics: dict, *, preferred_key=None) -> float:
    if not metrics:
        raise ValueError("Model returned no training metrics.")

    preferred_keys = [
        preferred_key,
        "total",
        "loss",
        "objective",
        "val_loss",
        "train_loss",
    ]
    for key in preferred_keys:
        if key and key in metrics:
            return _to_float(metrics[key])

    for value in metrics.values():
        try:
            return _to_float(value)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Could not find a scalar metric in: {metrics}")


def _format_metrics(metrics: dict) -> str:
    parts = []
    for key, value in metrics.items():
        try:
            parts.append(f"{key}: {_to_float(value):.4f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(parts) if parts else "no scalar metrics"


def _format_epoch_log(epoch, lr, train_metrics, val_metrics):
    return (
        f"\nEpoch {epoch:03d}: lr: {lr:0.5f}, "
        f"train [{_format_metrics(train_metrics)}], "
        f"val [{_format_metrics(val_metrics)}]"
    )


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _extract_step_output(output):
    if not isinstance(output, StepOutput):
        raise TypeError("training_step/validation_step must return generation_models.interfaces.StepOutput.")
    return output.loss, dict(output.metrics)


def _metrics_to_float(metrics: dict) -> dict:
    converted = {}
    for key, value in metrics.items():
        try:
            converted[key] = _to_float(value)
        except (TypeError, ValueError):
            continue
    return converted


def _average_metric_dicts(metric_dicts: list[dict]) -> dict:
    if not metric_dicts:
        return {}

    sums = {}
    counts = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def _run_epoch(model, loader, optimizer, config, device, *, training: bool) -> dict:
    model.train(training)
    step_fn = model.training_step if training else model.validation_step

    metric_dicts = []
    grad_clip_norm = config.grad_clip_norm
    iterator = tqdm(loader, desc=("train" if training else "val"), leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(iterator):
        batch = _move_to_device(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = step_fn(batch, batch_idx, config)
            loss, metrics = _extract_step_output(output)

            if training:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        metrics = _metrics_to_float(metrics)
        if "loss" not in metrics:
            metrics["loss"] = _to_float(loss)
        metric_dicts.append(metrics)

        monitor_key = config.monitor_metric
        try:
            iterator.set_postfix(value=f"{_metric_value(metrics, preferred_key=monitor_key):.6f}")
        except ValueError:
            pass

    return _average_metric_dicts(metric_dicts)


def train(model, train_loader, val_loader, config, *, best_model_path=None):
    """
    Train a model through the TrainableModule batch-level interface.

    Required model methods:
      - warmup(shape, device, dtype, config)
      - training_step(batch, batch_idx, config)
      - validation_step(batch, batch_idx, config)
      - on_epoch_start(epoch, config)
      - configure_optimizers(config) -> (optimizer, scheduler)
      - save_checkpoint(path, **state)
    """
    train_history = []
    val_history = []
    best_epoch = 0
    best_val = float("inf")
    monitor_key = config.monitor_metric

    with tqdm(desc="epoch", total=config.epochs) as pbar_outer:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        model.to(device)
        model.warmup(
            config.anomaly_size,
            device=device,
            dtype=config.training_dtype,
            config=config,
        )

        optimizer, scheduler = model.configure_optimizers(config)
        if optimizer is None:
            raise ValueError(f"{model.__class__.__name__}.configure_optimizers() returned no optimizer.")

        if scheduler is None and config.lr_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, "min", **config.lr_scheduler_params)

        early_stopping = None
        if config.early_stopping:
            early_stopping = _EarlyStoppingTracker(**config.early_stopping_params)

        for epoch in range(config.epochs):
            model.on_epoch_start(epoch, config=config)

            train_metrics = _run_epoch(model, train_loader, optimizer, config, device, training=True)
            val_metrics = {}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = _run_epoch(model, val_loader, optimizer, config, device, training=False)
            if not val_metrics:
                val_metrics = train_metrics

            train_value = _metric_value(train_metrics, preferred_key=monitor_key)
            val_value = _metric_value(val_metrics, preferred_key=monitor_key)
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

            lr = _current_lr(optimizer, scheduler)
            tqdm.write(_format_epoch_log(epoch + 1, lr, train_metrics, val_metrics))
            pbar_outer.set_postfix(
                lr=f"{lr:.5f}",
                train=f"{train_value:.4f}",
                val=f"{val_value:.4f}",
            )
            pbar_outer.update(1)

            _step_scheduler(scheduler, val_value)

            if early_stopping is not None and early_stopping.step(val_value):
                print("Early stopping")
                break

    return train_history, val_history, best_epoch, best_val
