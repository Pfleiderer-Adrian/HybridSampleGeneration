import os
import optuna
import torch
from optuna import Trial
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping_pytorch import EarlyStopping

from models.VAE_ConvNeXt_2D import ConvNeXtVAE2D
from models.VAE_ConvNeXt_3D import ConvNeXtVAE3D
from models.VAE_ResNet_2D import ResNetVAE2D
from models.VAE_ResNet_3D import ResNetVAE3D, Config
from synthesizer.Configuration import Configuration


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
    min_params = config.model_params["min"]
    max_params = config.model_params["max"]
    params = {}
    for key, value in min_params.items():
        if isinstance(value, bool):
            params[key] = trial.suggest_categorical(key, [True, False])
        elif isinstance(value, int):
            params[key] = trial.suggest_int(key, value, max_params[key])
        elif isinstance(value, float):
            params[key] = trial.suggest_float(key, value, max_params[key])


    # 2. Create the model with chosen hyperparameters
    model = None
    if config.model_name == "VAE_ResNet_3D":
        model = ResNetVAE3D(config.anomaly_size[0], Config(**params))
    elif config.model_name == "VAE_ResNet_2D":
        model = ResNetVAE2D(config.anomaly_size[0], Config(**params))
    elif config.model_name == "VAE_ConvNeXt_3D":
        model = ConvNeXtVAE3D(config.anomaly_size[0], Config(**params))
    elif config.model_name == "VAE_ConvNeXt_2D":
        model = ConvNeXtVAE2D(config.anomaly_size[0], Config(**params))
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    n_val = int(len(dataset) * config.val_ratio)
    n_train = len(dataset) - n_val

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

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
    trial.set_user_attr("anomaly_size", config.anomaly_size)
    trial.set_user_attr("model_name", config.model_name)

    # lowest validation error over all epochs
    return min(val_losses)


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
    log_template = "\nEpoch {ep:03d}: lr: {learning_rate:0.5f}, train_loss: {t_loss:0.4f}, val_loss {v_loss:0.4f}, val_recon {v_recon:0.4f}, val_kl {v_kl:0.4f}, val_recon_weighted {v_recon_w:0.4f}, val_kl_weighted {v_kl_w:0.4f}"

    with tqdm(desc="epoch", total=config.epochs) as pbar_outer:

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # weight_decay für L2-Regularisierung

        # define loss function
        # criterion = nn.MSELoss()  # nn.BCELoss()

        # define early stopping
        early_stopping = EarlyStopping(**config.early_stopping_params)

        # define learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min', **config.lr_scheduler_params)

        # train model for specified number of epochs
        for epoch in range(config.epochs):

            # train step
            train_loss, val_loss = model.fit_epoch(train_loader, val_loader, optimizer)

            # store results
            train_history.append(train_loss["total"])
            val_history.append(val_loss["total"])

            # Track best epoch and optionally save best checkpoint
            if val_loss["total"] < best_val:
                best_val = float(val_loss["total"])
                best_epoch = epoch + 1
                if best_model_path is not None:
                    torch.save(model.state_dict(), best_model_path)

            # update progress bar
            tqdm.write(log_template.format(ep=epoch + 1, learning_rate=scheduler.get_last_lr()[0], t_loss=train_loss["total"],
                                           v_loss=val_loss["total"], v_recon=val_loss["recon"], v_kl=val_loss["kl"], v_recon_w=val_loss["recon_weighted"], v_kl_w=val_loss["kl_weighted"]))
            pbar_outer.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.5f}", t_loss=f"{val_loss["total"]:.4f}", v_loss=f"{val_loss["total"]:.4f}")
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
