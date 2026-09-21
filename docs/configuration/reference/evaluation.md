# Evaluation

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.evaluation.foreground_threshold` | `float \| None` | `0.01` | Foreground threshold used during evaluation; None disables it. |
| `config.evaluation.outlier_thresholds` | `dict[str, dict[str, float \| None]]` | `12 metrics with min=None and max=None` | Minimum and maximum limits per evaluation metric; None leaves a limit open. |

## Defaults

`outlier_thresholds` contains `min: None` and `max: None` for Contrast, Homogeneity, Energy, Correlation, their four `roi_` variants, Volume, D-center, H-center, and W-center.

