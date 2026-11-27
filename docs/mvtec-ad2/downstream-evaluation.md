# Downstream evaluation

Part of the [MVTec AD 2 benchmark](index.md). This page covers the metrics and
checkpoint selection of the downstream segmentation models. It is separate from
the package's [evaluation step](../guides/evaluation.md), which compares
synthetic anomalies with their real parents. The `training.*` settings below
belong to `DownstreamConfiguration`.

## AUC-PRO

MVTec AD 2 downstream evaluation additionally reports `pixel_aupro_0.05` in
validation and test metrics. It averages the detected fraction of each
8-connected ground-truth defect region equally, integrates the PRO curve up to
a background-pixel false positive rate of 0.05, and divides by 0.05 to normalize
the result to [0, 1]. The endpoint is linearly interpolated. It uses the same
`pixel_metric_bins` score quantization as pixel AUROC/AP (4096 bins by default),
so it approximates an evaluation over all distinct scores. Images without defects
contribute background false positives; splits without defects or background
return `null`. Existing saved results are updated only by re-evaluation.

## Checkpoint selection

New DRAEM downstream runs select `best.pt` by the **highest validation
`pixel_aupro_0.05`**, retaining the earlier epoch on ties. Focal loss is still
reported for monitoring but does not select the checkpoint. Validation must
contain both defect regions and background pixels; an undefined AUC-PRO causes
an explicit error rather than silently switching criteria. Checkpoints and
training history record `selection_metric`, `selection_mode` and
`selection_score`. Existing runs and their selected checkpoints remain unchanged.
DINOv3 models retain their documented loss-based selection criteria.
To select a DINOv3 checkpoint by validation AUC-PRO instead, set
`training.selection_metric = "pixel_aupro_0.05"` (default `"auto"` retains
the model's criterion). This uses real held-out validation images and masks;
paired synthetic validation is not needed for this criterion. Paired training
still uses the full paired loss. Test data never selects the checkpoint.

## Validation interval

`training.validation_interval` controls the full-validation frequency (default
`1`, every epoch). For example, `5` validates at epochs 5, 10, 15, and so on.
The final epoch is always validated, even when it is not a multiple of the
interval. Only validated epochs can replace `best.pt`; `last.pt` is saved every
epoch. History leaves validation values blank on skipped epochs.
