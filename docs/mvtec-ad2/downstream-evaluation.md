# Downstream evaluation

Part of the [MVTec AD 2 benchmark](index.md). This page covers the metrics and
checkpoint selection of the downstream segmentation models. It is separate from
the package's [evaluation step](../guides/evaluation.md), which compares
synthetic anomalies with their real parents. The `training.*` settings below
belong to `DownstreamConfiguration`.

## AUC-PRO

Besides image and pixel AUROC/AP, downstream evaluation reports
`pixel_aupro_0.05` in validation and test metrics. It averages the detected fraction of each
8-connected ground-truth defect region equally, integrates the PRO curve up to
a background-pixel false positive rate of 0.05, and divides by 0.05 to normalize
the result to [0, 1]. The endpoint is linearly interpolated. It uses the same
`pixel_metric_bins` score quantization as pixel AUROC/AP (4096 bins by default),
so it approximates an evaluation over all distinct scores. Images without defects
contribute background false positives; splits without defects or background
return `null`. Saved results change only when a run is evaluated again.

## Checkpoint selection

`training.selection_metric` decides which validated epoch becomes `best.pt`.
With the default `"auto"`, each model uses its own criterion:

| Model | Criterion with `"auto"` | Direction |
| --- | --- | --- |
| `draem` | validation `pixel_aupro_0.05` | highest |
| `dinov3_paired` | `paired_total_loss` on fixed synthetic validation pairs | lowest |
| `dinov3_segmentation` | `segmentation_total_loss` (focal + weighted Dice) on real validation data | lowest |

Set `training.selection_metric = "pixel_aupro_0.05"` to select a DINOv3
checkpoint by the highest validation AUC-PRO instead; the submitted preset does
this. The criterion uses real held-out validation images and masks, so paired
synthetic validation is not needed. Paired training still uses the full paired
loss.

Selection by AUC-PRO keeps the earlier epoch on ties. Validation must contain
both defect regions and background pixels; an undefined AUC-PRO raises an error
instead of silently switching criteria. Focal loss is reported for monitoring
only. Checkpoints and training history record `selection_metric`,
`selection_mode` and `selection_score`. Test data never selects the checkpoint.

## Validation interval

`training.validation_interval` controls the full-validation frequency (default
`1`, every epoch). For example, `5` validates at epochs 5, 10, 15, and so on.
The final epoch is always validated, even when it is not a multiple of the
interval. Only validated epochs can replace `best.pt`; `last.pt` is saved every
epoch. History leaves validation values blank on skipped epochs.
