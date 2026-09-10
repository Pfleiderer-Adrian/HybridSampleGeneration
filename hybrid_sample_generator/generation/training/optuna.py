"""Optuna study orchestration and model hyperparameter sampling."""

import os

import optuna
import torch
from optuna import Trial
from torch.utils.data import DataLoader, random_split

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.model_settings import ModelHyperparameterSpace
from hybrid_sample_generator.generation.registry import get_model_spec
from hybrid_sample_generator.generation.training.augmentation import (
    apply_training_offset_augmentation,
)
from hybrid_sample_generator.generation.training.loop import train


def optimize(no_of_trials, config: Configuration, dataset):
    """Run or resume an Optuna study for the configured generator model."""
    paths = config.study.paths
    os.makedirs(paths.study_folder, exist_ok=True)
    study = optuna.create_study(
        study_name=config.study.name,
        direction="minimize",
        load_if_exists=True,
        storage=paths.optuna_storage_url,
    )
    study.optimize(
        lambda trial: objective(trial, config, dataset), n_trials=no_of_trials
    )
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


def objective(trial: Trial, config: Configuration, dataset):
    """Sample parameters, train one model, and return its best validation loss."""
    params = sample_model_params(trial, config.model.parameters)
    model = get_model_spec(config.model.name).build(params)
    training = config.training
    validation_size = int(len(dataset) * training.validation_ratio)
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(config.study.seed)
    train_dataset, validation_dataset = random_split(
        dataset, [training_size, validation_size], generator=generator
    )
    train_dataset = apply_training_offset_augmentation(
        train_dataset, config.augmentation
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    os.makedirs(config.study.paths.trained_models, exist_ok=True)
    model_path = os.path.join(
        config.study.paths.trained_models, f"model_trial_{trial.number}_best.pth"
    )
    _, validation_losses, best_epoch, best_validation = train(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        config=training,
        anomaly_size=config.extraction.anomaly_size,
        best_model_path=model_path,
    )
    for key, value in params.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", float(best_validation))
    trial.set_user_attr("params", params)
    trial.set_user_attr("model_name", config.model.name)
    return min(validation_losses) if validation_losses else float(best_validation)


def sample_model_params(trial: Trial, model_params):
    """Build concrete parameters from a model hyperparameter search space."""
    model_params = ModelHyperparameterSpace.from_value(model_params)
    params = {}
    for key in sorted(set(model_params.min) | set(model_params.max)):
        minimum = model_params.min.get(key)
        maximum = model_params.max.get(key, minimum)
        params[key] = _sample_or_fix_param(trial, key, minimum, maximum)
    return params


def _sample_or_fix_param(trial, key, minimum, maximum):
    if minimum == maximum:
        return minimum
    if isinstance(minimum, bool) and isinstance(maximum, bool):
        return trial.suggest_categorical(key, _unique_choices([minimum, maximum]))
    if isinstance(minimum, int) and isinstance(maximum, int):
        return trial.suggest_int(key, *sorted((minimum, maximum)))
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        return trial.suggest_float(key, *sorted((float(minimum), float(maximum))))
    if isinstance(minimum, (list, tuple, set)) and maximum is None:
        return trial.suggest_categorical(key, list(minimum))
    return trial.suggest_categorical(key, _unique_choices([minimum, maximum]))


def _unique_choices(values):
    choices = []
    for value in values:
        if value not in choices:
            choices.append(value)
    return choices


__all__ = ["objective", "optimize", "sample_model_params"]
