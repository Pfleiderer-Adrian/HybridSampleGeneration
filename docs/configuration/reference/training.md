# Training

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.training.num_trials` | `int` | `10` | Number of Optuna trials; positive. |
| `config.training.trial_selection` | `TrialSelection` | `'best'` | Trial to use: 'best', 'last', or a non-negative trial ID. |
| `config.training.validation_ratio` | `float` | `0.2` | Fraction of data reserved for validation; range [0, 1). |
| `config.training.batch_size` | `int` | `64` | Training batch size; positive. |
| `config.training.epochs` | `int` | `1000` | Maximum number of training epochs; positive. |
| `config.training.learning_rate` | `float` | `0.001` | Initial learning rate; positive. |
| `config.training.dtype` | `torch.dtype \| None` | `None` | PyTorch dtype for training; None uses the model default. |
| `config.training.gradient_clip_norm` | `float \| None` | `None` | Optional upper bound for the gradient norm. |
| `config.training.monitor_metric` | `str \| None` | `'selection'` | Metric used for model selection and early stopping. |
| `config.training.early_stopping_enabled` | `bool` | `True` | Enable early stopping. |
| `config.training.early_stopping` | `dict[str, Any]` | `{'patience': 100, 'delta': 0.0001}` | Early stopping settings, including patience and delta. |
| `config.training.lr_scheduler_enabled` | `bool` | `True` | Enable the learning rate scheduler. |
| `config.training.lr_scheduler` | `dict[str, Any]` | `{'patience': 30, 'factor': 0.1, 'threshold': 1e-05}` | Learning rate scheduler settings, including patience and factor. |

## Nested training settings

`config.training.early_stopping` contains `patience` (epochs without improvement) and `delta` (minimum improvement). `config.training.lr_scheduler` contains `patience`, `factor` (learning rate multiplier), and `threshold` (minimum improvement). Their defaults are shown in the table above.

