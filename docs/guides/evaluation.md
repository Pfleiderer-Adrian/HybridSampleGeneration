# Evaluation

Evaluation joins each synthetic anomaly to its real parent through
`real_anomaly_id`. Placement ROI comparisons use the full
Original → Hybrid → Placement → Synthetic → Real join. The CSV output contains
all relevant IDs, so multiple variants cannot overwrite or masquerade as one
pair.

`evaluate_study(config)` compares each explicit real/synthetic cutout pair using
GLCM contrast, homogeneity, energy and correlation, plus mask volume and center
of mass. GLCMs quantize each channel to 32 levels and aggregate immediate-neighbor
pairs over four 2D or thirteen 3D directions. When placement ROI artifacts are
available, the same GLCM features are also compared between the original real
ROI and the fused placement ROI.

For every metric the evaluator records the absolute pair difference. Outliers
default to the `1.5 * IQR` rule and can be overridden per metric with
`config.evaluation.outlier_thresholds`. Each run replaces
`evaluation_results/metric_diffs.csv`, writes up to three histogram images
(cutout texture, cutout morphology and placement-ROI texture), prints real and
synthetic means, and summarizes outlier overlaps.

Evaluation reads the normalized repository relations directly and does not
construct a generation orchestrator.
