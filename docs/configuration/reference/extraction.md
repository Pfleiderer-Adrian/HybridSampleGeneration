# Extraction

| Parameter | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `config.extraction.anomaly_size` | `list[int]` | `[3, 64, 64]` | 3 to 4 elements, each `>= 1` | Target anomaly crop shape (C,H,W) or (C,D,H,W); must match the model dimensions. |
| `config.extraction.separate_components` | `bool` | `True` | — | Extract each connected anomaly component separately. |
| `config.extraction.min_coverage_ratio` | `float` | `0.01` | `[0.0, 1.0]` | Minimum mask coverage within a crop. |
| `config.extraction.add_background_noise` | `bool` | `True` | — | Add a small amount of noise to otherwise constant backgrounds. |
| `config.extraction.preserve_aspect_ratio` | `bool` | `True` | — | Use one spatial scale for every axis and center-pad the remaining area; False stretches axes independently. |
| `config.extraction.normalization` | `str \| None` | `'z-score'` | `'z-score'`, `'zscore_median'`, `None` | Intensity normalization method; None disables normalization. Spelling variants of the listed names are accepted. |
| `config.extraction.normalization_eps` | `float` | `1e-06` | `> 0.0` | Lower bound for numerically stable normalization. |
| `config.extraction.roi.fixed_size` | `list[int] \| None` | `None` | each `>= 1` | Fixed spatial ROI size; None selects dynamic sizing. |
| `config.extraction.roi.min_size` | `list[int] \| int` | `0` | each `>= 0` or `>= 0` | Minimum dynamic ROI size, either scalar or per axis. |
| `config.extraction.roi.min_padding` | `list[int]` | `[10, 10, 10]` | each `>= 0` | Minimum ROI padding around the anomaly on each axis. |
| `config.extraction.roi.padding_ratio` | `list[float]` | `[0.5, 0.5, 0.5]` | each `>= 0.0` | Additional ROI padding relative to anomaly size on each axis. |
| `config.extraction.roi.preserve_aspect_ratio` | `bool` | `False` | — | Expand dynamic ROIs to match the spatial aspect ratio of the full image sample. |

