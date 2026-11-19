# Training

| Parameter | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `config.training.num_trials` | `int` | `10` | `>= 1` | Number of Optuna trials. |
| `config.training.trial_selection` | `str \| int` | `'best'` | `'best'`, `'last'` or `>= 0` | Trial to use: 'best', 'last', or a non-negative trial ID. |
| `config.training.validation_ratio` | `float` | `0.2` | `[0.0, 1.0)` | Fraction of data reserved for validation. |
| `config.training.batch_size` | `int` | `64` | `>= 1` | Training batch size. |
| `config.training.epochs` | `int` | `1000` | `>= 1` | Maximum number of training epochs. |
| `config.training.learning_rate` | `float` | `0.001` | `> 0.0` | Initial learning rate. |
| `config.training.dtype` | `str \| None` | `None` | `'float16'`, `'bfloat16'`, `'float32'`, `'float64'`, `None` | PyTorch dtype for training; None uses the model default. |
| `config.training.gradient_clip_norm` | `float \| None` | `None` | `> 0.0` | Optional upper bound for the gradient norm. |
| `config.training.monitor_metric` | `str \| None` | `'selection'` | — | Metric used for model selection and early stopping. |
| `config.training.early_stopping_enabled` | `bool` | `True` | — | Enable early stopping. |
| `config.training.early_stopping` | `dict` | `{'patience': 100, 'delta': 0.0001}` | — | Early stopping settings. |
| `config.training.lr_scheduler_enabled` | `bool` | `True` | — | Enable the learning rate scheduler. |
| `config.training.lr_scheduler` | `dict` | `{'patience': 30, 'factor': 0.1, 'threshold': 1e-05}` | — | Learning rate scheduler settings passed to ReduceLROnPlateau; additional keyword arguments are allowed. |

## Nested training settings

- `config.training.early_stopping['patience']`: Epochs without improvement before stopping. Allowed: `>= 0`.
- `config.training.early_stopping['delta']`: Minimum improvement that resets the patience counter. Allowed: `>= 0.0`.
- `config.training.lr_scheduler['patience']`: Epochs without improvement before the learning rate is reduced. Allowed: `>= 0`.
- `config.training.lr_scheduler['factor']`: Learning rate multiplier. Allowed: `(0.0, 1.0)`.
- `config.training.lr_scheduler['threshold']`: Minimum improvement that counts as progress. Allowed: `>= 0.0`.

Their defaults are shown in the table above.

