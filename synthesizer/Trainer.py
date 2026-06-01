import os
import optuna
import torch
from optuna import Trial
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping_pytorch import EarlyStopping

from models.model_loader import model_loader
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
        x_key = next((key for key in ("x", "image", "inputs") if key in item), None)
        if x_key is None:
            return item

        x = item[x_key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        shifts = self._sample_shifts(x)
        if shifts is None:
            return item

        item[x_key] = self._apply_shift(x, shifts, fill_value=torch.amin(x))
        for key in ("org_mask", "mask", "tgt_mask", "target_mask"):
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


def optimize(no_of_trails, config:Configuration, dataset):
    """
    Run Optuna hyperparameter optimization for a generative model.

    This function:
      1) Creates (or reuses) an Optuna study stored as a SQLite DB in `config.study_folder`.
      2) Runs `objective(...)` for `no_of_trails` trials.
      3) Prints summary information about the best trial found.

    Inputs
    ------
    no_of_trails:
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
    os.makedirs(config.study_folder, exist_ok=True)

    study = optuna.create_study(study_name=config.study_name, direction='minimize', load_if_exists=True,
                                storage="sqlite:///" + str(os.path.join(config.study_folder, config.study_name + ".db")))  # Speicherort der Datenbank

    # Run optimization process
    func = lambda trial: objective(trial, config, dataset)
    study.optimize(func, n_trials=no_of_trails)  # calls objective function multiple time and tries to minimize the return value of this function

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
      1) Builds a hyperparameter set from config.model_params["min"/"max"].
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
    model = model_loader(config.model_name, params)

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
    directory = os.path.join(config.study_folder, 'trained_models')
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

    # lowest validation error over all epochs
    return min(val_losses)


def _apply_training_offset_augmentation(dataset, config: Configuration):
    if not getattr(config, "random_offset", False):
        return dataset

    transform = _RandomSpatialOffset(
        max_fraction=getattr(config, "random_offset_max_fraction", 1.0),
        foreground_threshold_rel=getattr(config, "random_offset_foreground_threshold_rel", 0.001),
    )
    return _TrainingTransformDataset(dataset, transform)


def sample_model_params(trial: Trial, model_params):
    """
    Build concrete model params from a {"min": ..., "max": ...} search space.

    Equal min/max values are treated as fixed constants. Strings and booleans
    become categorical choices when min/max differ.
    """
    min_params = model_params["min"]
    max_params = model_params["max"]
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


def beta_schedule(epoch, beta_start, beta_max, warmup_start, warmup_epochs):
    if warmup_start >= epoch:
        return 0.0
    if warmup_epochs <= 0:
        return beta_max
    t = min(1.0, max(0.0, epoch / warmup_epochs))
    return beta_start + t * (beta_max - beta_start)


def train(model, train_loader, val_loader, config, *, best_model_path=None):
    """
    Train a model for up to `config.epochs` epochs using `model.fit_epoch(...)`.

    This function assumes the model implements:
      model.fit_epoch(train_loader, val_loader, optimizer) -> (train_loss_dict, val_loss_dict)
    where each dict contains at least a "total" key.

    Inputs
    ------
    model:
        PyTorch module-like object with `.parameters()` and `.fit_epoch(...)`.
    train_loader:
        DataLoader for training set.
    val_loader:
        DataLoader for validation set.
    config:
        Training configuration:
          - epochs, lr, batch_size
          - lr_scheduler (bool), early_stopping (bool)
          - plus other flags/params

    Outputs
    -------
    (train_history, val_history, best_epoch, best_val):
        train_history: list[float]
            Per-epoch training loss (train_loss["total"]).
        val_history: list[float]
            Per-epoch validation loss (val_loss["total"]).
        best_epoch: int
            Epoch index (1-based) with the lowest validation loss.
        best_val: float
            Lowest validation loss observed.

    Side effects
    ------------
    - Prints training progress to stdout via tqdm.
    - May stop early if early stopping triggers.
    """
    train_history = []
    val_history = []
    best_epoch = 0
    best_val = float("inf")
    log_template = "\nEpoch {ep:03d}: lr: {learning_rate:0.5f}, train_loss: {t_loss:0.4f}, val_loss {v_loss:0.4f}, val_recon {v_recon:0.4f}, val_recon_weighted {v_recon_w:0.4f}, val_kl {v_kl:0.4f}, val_kl_weighted {v_kl_w:0.4f}, kl_raw {v_kl_raw:0.4f}"

    with tqdm(desc="epoch", total=config.epochs) as pbar_outer:

        # define optimizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        model.to(device)
        model.warmup(config.anomaly_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # weight_decay für L2-Regularisierung

        # define loss function
        # criterion = nn.MSELoss()  # nn.BCELoss()

        # define early stopping
        early_stopping = EarlyStopping(**config.early_stopping_params)

        # define learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min', **config.lr_scheduler_params)

        # train model for specified number of epochs
        for epoch in range(config.epochs):
            
            # update beta_kl for current epoch
            if hasattr(model.cfg, 'beta_kl'):
                model.cfg.beta_kl = beta_schedule(
                    epoch,
                    model.cfg.beta_kl_start,
                    model.cfg.beta_kl_max,
                    model.cfg.beta_kl_warmup_start,
                    model.cfg.beta_kl_warmup_epochs
                    )

            # train step
            train_loss, val_loss = model.fit_epoch(train_loader, val_loader, optimizer, device=device)

            # store results
            train_history.append(train_loss["total"])
            val_history.append(val_loss["total"])

            # Track best epoch and optionally save best checkpoint
            if val_loss["total"] < best_val:
                best_val = float(val_loss["total"])
                print("New best epoch! val_loss: "+str(best_val))
                best_epoch = epoch + 1
                if best_model_path is not None:
                    torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), best_model_path)


            # update progress bar
            tqdm.write(log_template.format(ep=epoch + 1, learning_rate=scheduler.get_last_lr()[0], t_loss=train_loss["total"],
                                           v_loss=val_loss["total"], v_recon=val_loss["recon"], v_recon_w=val_loss["recon_weighted"], v_kl=val_loss["kl"], v_kl_w=val_loss["kl_weighted"], v_kl_raw=val_loss["kl_raw"]))
            pbar_outer.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.5f}", t_loss=f"{val_loss['total']:.4f}", v_loss=f"{val_loss['total']:.4f}")
            pbar_outer.update(1)

            # Learning Rate Scheduling
            if config.lr_scheduler:
                scheduler.step(val_loss["total"])

            # Early stopping
            if config.early_stopping:
                early_stopping(val_loss["total"], model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    return train_history, val_history, best_epoch, best_val
