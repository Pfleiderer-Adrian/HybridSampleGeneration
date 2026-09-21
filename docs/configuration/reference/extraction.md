# Extraction

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.extraction.anomaly_size` | `tuple[int, ...]` | `(3, 64, 64)` | Target anomaly crop shape (C,H,W) or (C,D,H,W); must match the model dimensions. |
| `config.extraction.separate_components` | `bool` | `True` | Extract each connected anomaly component separately. |
| `config.extraction.min_coverage_ratio` | `float` | `0.01` | Minimum mask coverage within a crop; range [0, 1]. |
| `config.extraction.add_background_noise` | `bool` | `True` | Add a small amount of noise to otherwise constant backgrounds. |
| `config.extraction.normalization` | `str` | `'z-score'` | Intensity normalization method; defaults to z-score. |
| `config.extraction.normalization_eps` | `float` | `1e-06` | Positive lower bound for numerically stable normalization. |
| `config.extraction.roi.fixed_size` | `tuple[int, ...] \| None` | `None` | Fixed spatial ROI size; None selects dynamic sizing. |
| `config.extraction.roi.min_size` | `tuple[int, ...] \| int` | `0` | Minimum dynamic ROI size, either scalar or per axis. |
| `config.extraction.roi.min_padding` | `tuple[int, ...]` | `(10, 10, 10)` | Minimum ROI padding around the anomaly on each axis. |
| `config.extraction.roi.padding_ratio` | `tuple[float, ...]` | `(0.5, 0.5, 0.5)` | Additional ROI padding relative to anomaly size on each axis. |

