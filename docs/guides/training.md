# Generator training

`generator.train_generator()` trains the generative model on the extracted real
anomaly cutouts. It runs an Optuna hyperparameter search, stores one checkpoint
per trial and finally loads the selected trial, so
`generate_synthetic_anomalies()` can follow directly.

```python
config.model.set_model("cVAE_ConvNeXt_2D")
config.training.num_trials = 5
config.training.epochs = 500
config.training.batch_size = 32

generator = HybridDataGenerator(config)
generator.train_generator()
```

Training needs real anomalies in the study, so run
[extraction](extraction.md) first. A CUDA GPU is used when available; otherwise
training runs on the CPU.

## What a trial does

Each of the `config.training.num_trials` trials:

1. samples the model parameters that have a distribution in
   `config.model.search`; all other parameters keep their value from
   `config.model.parameters` (see [generator models](../configuration/models.md));
2. splits the real anomalies into training and validation data according to
   `config.training.validation_ratio`. The split is seeded with
   `config.study.seed` and therefore identical in every trial;
3. trains with Adam at `config.training.learning_rate` for at most
   `config.training.epochs` epochs;
4. saves the checkpoint of the epoch with the lowest validation value of
   `config.training.monitor_metric`.

The default monitor metric `selection` is the mask-balanced reconstruction
loss without the KL term. It also is the value Optuna minimizes across trials.
`total` monitors the complete weighted training loss instead.

## Stopping and learning rate

- **Early stopping** (`early_stopping_enabled`, default on) ends a trial when
  the validation value has not improved by more than
  `early_stopping["delta"]` for `early_stopping["patience"]` epochs.
- **Learning-rate scheduling** (`lr_scheduler_enabled`, default on) uses
  `ReduceLROnPlateau` on the validation value with the `patience`, `factor`
  and `threshold` from `config.training.lr_scheduler`.
- `gradient_clip_norm` optionally clips the gradient norm in every step.

```python
config.training.early_stopping = {"patience": 50, "delta": 1e-4}
config.training.lr_scheduler = {"patience": 15, "factor": 0.5, "threshold": 1e-5}
```

## Training augmentation

Two augmentations in `config.augmentation` act during training:

- **Random offset** (`random_offset_enabled`, default on) translates each
  training cutout and its mask by a random amount. The anomaly foreground
  always stays inside the cutout; `random_offset_max_fraction` scales the
  available range. Validation data is not shifted.
- **Mask transforms** (`config.augmentation.mask_transforms`) create the
  source-to-target pairs for the paired models. For all mask-conditioned
  models they also produce the target masks during
  [generation](generation.md#target-masks).

## Outputs

```text
study/
  <study_name>.db                          # Optuna study with all trials
  trained_models/
    model_trial_<n>_best.pth               # best checkpoint of trial n
    model_trial_<n>_history.csv            # per-epoch metrics of trial n
```

The history CSV contains the learning rate, the KL weight and every training
and validation metric per epoch. It is rewritten after each epoch, so a running
trial can be inspected.

## Selecting a trial

`config.training.trial_selection` decides which trial `train_generator()` and
`load_generator()` load:

| Value | Loaded trial |
| --- | --- |
| `"best"` (default) | trial with the lowest objective value |
| `"last"` | trial with the highest trial number |
| a non-negative integer | trial with exactly this number |

To reuse a trained generator in a later process, load it instead of training
again:

```python
generator = HybridDataGenerator(config)
generator.load_generator()
generator.generate_synthetic_anomalies()
```

`load_generator(path_to_db_file=...)` reads the trials from another Optuna
database, for example to reuse a generator trained in a different study folder.
That database must contain an Optuna study named like `config.study.name`, and
the checkpoint paths stored in it must still exist.

## Repeating training

`train_generator()` continues the existing Optuna study of the study folder:
every call adds `num_trials` further trials to the earlier ones and keeps
their checkpoints. Existing synthetic anomalies and hybrids are not changed
until `generate_synthetic_anomalies()` runs again.

The complete parameter list is available in the
[training reference](../configuration/reference/training.md) and the
[augmentation reference](../configuration/reference/augmentation.md).
