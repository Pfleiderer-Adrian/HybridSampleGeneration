"""Optuna study orchestration and model hyperparameter sampling."""

import os
from dataclasses import asdict, replace

import optuna
import torch
from optuna import Trial
from torch.utils.data import DataLoader, random_split

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.model_settings import (
    Choice,
    FloatRange,
    IntRange,
    SearchSpace,
)
from hybrid_sample_generator.generation.registry import get_model_spec
from hybrid_sample_generator.generation.training.augmentation import (
    apply_training_offset_augmentation,
)
from hybrid_sample_generator.generation.training.paired_targets import (
    apply_paired_target_training,
)
from hybrid_sample_generator.generation.training.loop import train


def optimize(
    num_trials: int,
    config: Configuration,
    dataset,
    *,
    num_anomaly_classes: int | None,
):
    """Run or resume an Optuna study for the configured generator model."""
    paths = config.study.paths
    os.makedirs(paths.study_folder, exist_ok=True)
    study = optuna.create_study(
        study_name=config.study.name,
        direction="minimize",
        load_if_exists=True,
        storage=paths.optuna_storage_url,
        sampler=optuna.samplers.TPESampler(seed=config.study.seed),
    )
    study.optimize(
        lambda trial: objective(
            trial, config, dataset, num_anomaly_classes=num_anomaly_classes
        ),
        n_trials=num_trials
    )
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


def objective(
    trial: Trial,
    config: Configuration,
    dataset,
    *,
    num_anomaly_classes: int | None,
):
    """Sample parameters, train one model, and return its best validation loss."""
    parameters = sample_model_params(
        trial, config.model.parameters, config.model.search
    )
    model_spec = get_model_spec(config.model.name)
    model = model_spec.build(
        parameters,
        in_channels=config.extraction.anomaly_size[0],
        num_anomaly_classes=num_anomaly_classes,
    )
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
    if getattr(model_spec, "training_target_mode", "identity") == "paired":
        train_dataset, validation_dataset = apply_paired_target_training(
            train_dataset,
            validation_dataset,
            config=config,
            identity_probability=parameters.identity_pair_probability,
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
    params = asdict(parameters)
    for key, value in params.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", float(best_validation))
    trial.set_user_attr("params", params)
    trial.set_user_attr("model_name", config.model.name)
    trial.set_user_attr("in_channels", int(config.extraction.anomaly_size[0]))
    trial.set_user_attr("num_anomaly_classes", num_anomaly_classes)
    return min(validation_losses) if validation_losses else float(best_validation)


def sample_model_params(
    trial: Trial,
    parameters,
    search: SearchSpace,
):
    """Return a fresh parameter dataclass with explicit distributions sampled."""
    sampled = replace(parameters)
    for name, distribution in search.items():
        if isinstance(distribution, IntRange):
            value = trial.suggest_int(
                name,
                distribution.low,
                distribution.high,
                step=distribution.step,
                log=distribution.log,
            )
        elif isinstance(distribution, FloatRange):
            value = trial.suggest_float(
                name,
                distribution.low,
                distribution.high,
                step=distribution.step,
                log=distribution.log,
            )
        elif isinstance(distribution, Choice):
            value = trial.suggest_categorical(name, distribution.values)
        else:
            raise TypeError(f"Unsupported search distribution for {name!r}.")
        setattr(sampled, name, value)
    return sampled


__all__ = ["objective", "optimize", "sample_model_params"]
