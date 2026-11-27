# Evaluation

Evaluation measures how closely synthetic anomalies resemble the real anomalies
they were generated from. It compares texture and shape per pair, flags pairs
that deviate unusually, and writes the differences to the study folder.

```python
from hybrid_sample_generator.evaluation.service import evaluate_study

results = evaluate_study(config)
```

`evaluate_study(config)` reads the study repository and stored arrays directly.
It does not construct a `HybridDataGenerator`, so it can run in a separate
process after generation. It only writes to `evaluation_results/`.

## What is compared

Evaluation runs three comparisons. Each one produces a set of metrics per pair.

| Comparison | Pairs | Region | Metrics |
| --- | --- | --- | --- |
| Cutout texture | every synthetic anomaly and its real parent | anomaly mask | `Contrast`, `Homogeneity`, `Energy`, `Correlation` |
| Cutout morphology | the same pairs | anomaly mask | `Volume`, `H-center`, `W-center`, and `D-center` for 3D |
| Placement ROI texture | every placement with a stored ROI and the ROI of its real anomaly | whole ROI | `roi_Contrast`, `roi_Homogeneity`, `roi_Energy`, `roi_Correlation` |

Each synthetic anomaly is joined to its real parent through `real_anomaly_id`.
Placement ROI comparisons use the full
Original → Hybrid → Placement → Synthetic → Real join. Because every pair is
identified by explicit IDs, multiple variants of one real anomaly cannot
overwrite or masquerade as one pair.

The cutout comparisons need synthetic anomalies, so they are available after
`generate_synthetic_anomalies()`. The placement ROI comparison only includes
placements whose ROI image has been stored; without any, that comparison is
empty and the other two still run.

### Masks for the cutout comparisons

The cutout comparisons restrict each cutout to its anomaly region:

1. If both the real and the synthetic anomaly have a stored segmentation mask,
   those masks are used.
2. Otherwise, a foreground mask is derived from the image: pixels above
   `minimum + foreground_threshold * (maximum - minimum)` count as foreground.
   `config.evaluation.foreground_threshold` defaults to `0.01`.
3. If `foreground_threshold` is `None`, the whole cutout is used.

The placement ROI comparison always uses the whole ROI.

## Metrics

### Texture

Texture is described with a gray-level co-occurrence matrix (GLCM), which counts
how often two intensity levels occur next to each other.

- Each channel is quantized to 32 levels between the minimum and maximum of its
  masked values. A constant region maps to a single level.
- Immediate-neighbor pairs are counted in four directions for 2D data and in
  thirteen directions for 3D data. A pair counts only if both pixels lie inside
  the mask. Counts are symmetric.
- The matrices of all channels are summed and normalized to probabilities.

Four features are derived from the matrix `p(i, j)`:

| Feature | Definition | Reading |
| --- | --- | --- |
| Contrast | sum of `(i - j)² · p(i, j)` | higher for strong local intensity changes |
| Homogeneity | sum of `p(i, j) / (1 + (i - j)²)` | higher when neighbors have similar intensity |
| Energy | square root of the sum of `p(i, j)²` | higher for uniform or repetitive texture |
| Correlation | normalized covariance of neighboring levels | higher when neighboring levels vary together; `1.0` for a constant region |

### Morphology

- `Volume` is the number of pixels or voxels in the anomaly mask. Masks with
  several channels are combined, so a pixel counts once if any channel marks it.
- `H-center`, `W-center` and, for 3D data, `D-center` are the coordinates of the
  mask's center of mass within the cutout.

### Differences

For every pair and metric, evaluation records the absolute difference between
the real and the synthetic value. Small differences mean the synthetic anomaly
resembles its real parent in that respect. All outputs below refer to these
differences, not to the raw metric values.

## Outliers

Outliers are determined per metric over the differences of all pairs. By
default a pair is an outlier when its difference lies outside
`Q1 - 1.5 · IQR` to `Q3 + 1.5 · IQR`, where `Q1` and `Q3` are the first and
third quartile and `IQR = Q3 - Q1`. Non-finite differences are ignored.

`config.evaluation.outlier_thresholds` replaces these limits per metric. Each
entry has a `min` and a `max`; `None` keeps the IQR limit for that side:

```python
# Flag pairs whose contrast differs by more than 0.5, keep the IQR lower limit.
config.evaluation.outlier_thresholds["Contrast"]["max"] = 0.5

# Flag pairs whose mask size differs by more than 200 pixels.
config.evaluation.outlier_thresholds["Volume"] = {"min": None, "max": 200}
```

The limits apply to the absolute differences. The available names are the
metric names from the table above, including the `roi_` variants.

## Results

### Return value

`evaluate_study` returns one entry per comparison: `glcm_cutout`,
`volume_cutout` and `glcm_roi`. Each entry contains:

| Key | Content |
| --- | --- |
| `sample_counter` | number of evaluated pairs |
| `mean_real` | mean of each metric over the real side of all pairs |
| `mean_synth` | mean of each metric over the synthetic side of all pairs |
| `all_diffs` | per metric, the list of absolute differences |
| `outliers` | per metric, the outlier pairs as `{"value": ..., "sample": ...}`, largest difference first |

`sample` is the synthetic anomaly ID for cutout comparisons and the placement ID
for the placement ROI comparison.

```python
results = evaluate_study(config)

contrast_outliers = results["glcm_cutout"]["outliers"]["Contrast"]
for entry in contrast_outliers[:5]:
    print(entry["sample"], entry["value"])
```

### Files

Each run writes to `<study folder>/evaluation_results/`. `metric_diffs.csv` is
recreated on every run; a histogram image is overwritten when its comparison
has at least one pair.

| File | Content |
| --- | --- |
| `metric_diffs.csv` | one row per pair and comparison |
| `glcm_cutout_difference_histograms.png` | histograms of the cutout texture differences |
| `volume_cutout_difference_histograms.png` | histograms of the cutout morphology differences |
| `glcm_roi_difference_histograms.png` | histograms of the placement ROI texture differences |

`metric_diffs.csv` has these columns:

| Column | Content |
| --- | --- |
| `pair_id` | synthetic anomaly ID, or placement ID for the ROI comparison |
| `real_anomaly_id` | real parent anomaly |
| `synthetic_anomaly_id` | synthetic variant |
| `placement_id` | placement, only for the ROI comparison |
| `feature_calculator` | `get_glcm_feature_diffs`, `get_volume_feature_diffs` or `get_glcm_roi_feature_diffs` |
| `metric_diffs` | JSON object with the absolute difference per metric |

### Console output

For each comparison, evaluation prints the number of pairs and the real and
synthetic mean of every metric. A final line summarizes how many pairs are
outliers in one, two or more metrics at once. Pairs that are outliers in
several metrics are the first candidates for a closer look.

## Inspecting results

The Evaluation tab of the [study browser](visualization.md) shows the
metric-based evaluation and previews linked fused placement ROIs for cutout
metrics.

## Limits

- The metrics compare a synthetic anomaly with its own real parent. They do not
  measure the variety among the variants of one real anomaly.
- A small difference indicates similar texture and shape statistics. It does
  not show that a sample is useful for training a downstream model.
- The cutout texture comparison depends on the mask. With derived foreground
  masks, `foreground_threshold` changes which pixels are compared.

For segmentation metrics of the benchmark models, see
[MVTec AD 2 downstream evaluation](../mvtec-ad2/downstream-evaluation.md).
