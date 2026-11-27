# Downstream training

Part of the [MVTec AD 2 benchmark](index.md). All settings on this page belong
to `DownstreamConfiguration`, not to the package's `Configuration`.

`downstream/presets.py` returns the configuration of the submitted runs: frozen
DINOv3 ViT-S/16 (pinned revision) with a CNN decoder, native 256×256 patches,
batch size 8, 100 epochs, validation every 5 epochs and checkpoint selection by
validation pixel AUPRO@0.05. Each run stores its configuration, split snapshot
with hybrid provenance, checkpoints, history, predictions and metrics below:

```text
<study>/downstream/dinov3_paired/<timestamp>_<id>/
```

To change settings, start from `create_downstream_configuration(mixture)` and
call `train_downstream()` and `evaluate_downstream()` directly.

## Adding real training defects to DINOv3

Set `data.include_real_defects = true` (default: `false`) for
`dinov3_paired` or `dinov3_segmentation`. Every masked defect image in the
persisted **training partition** contributes one defect-containing crop per
epoch, shuffled together with the existing mixture. These examples are added
to `samples_per_epoch`; the normal/DRAEM/hybrid quotas are not reduced.
Validation and test partitions are not affected and must not overlap the real
training image paths. No private images are added.

Real defect images use Focal + Dice supervision. They have no healthy partner,
so partner and consistency losses are skipped for these samples. All other
samples keep the paired losses. The run records the included image IDs
in `real_training_sources.json` and the observed `real` count in
`training_history.csv`. Use `training.selection_metric = "pixel_aupro_0.05"`
for selection on real validation defects without synthetic validation pairs;
see [downstream evaluation](downstream-evaluation.md#checkpoint-selection).

## Fixed source quotas for downstream comparisons

`config.data.source_quotas` can specify exact per-epoch counts for `normal`,
`real`, `draem`, `cutmix`, `hybrid` and `anomalydiffusion`. Counts must sum to
`config.data.samples_per_epoch`. When set, these quotas replace the
fraction-based plan (`normal_fraction`, `hybrid_fraction`) and the additional
once-per-image real-defect entries.
Every real training image is sampled before repeats within each epoch; the real
quota must therefore cover the training defect-image count. Hard-normal sampling
is disabled for this mode.

Set `config.dinov3.pair_healthy_samples = False` to supervise healthy examples
with segmentation loss only (Focal, with Dice omitted for empty masks). Real
examples always receive Focal + Dice only. Synthetic examples keep the healthy
partner supervision and background probability consistency. With the default
`True`, healthy examples also receive the paired losses.

`cutmix` is a defect-aware rectangular 2D adaptation: donor defect pixels guide
the rectangle center, Beta(1,1) sets its area, image pixels and ground-truth labels
are copied together, and the healthy receiver is retained as the partner. The
rectangle itself is **not** the defect label. Donors and receivers must come
from the training partition. This mode needs no hybrid-generation study.

`config.data.source_quota_rotation` lists synthetic sources whose combined
quota is split equally per epoch; the rounding remainder rotates across them.
Healthy and real quotas stay fixed.

## Fixed hybrids and separate output folders

`train_downstream(..., hybrid_ids=[...])` trains on exactly the listed hybrids
in the given order, for example the `hybrid_provenance` of an earlier run.
This selection replaces the postprocessing filter. `output_root=...` writes
the run below another folder while still loading hybrids from the study.

## Resuming a run

Training can resume in the same output directory with
`train(..., resume_checkpoint=path_to_last_checkpoint)`. This restores model,
optimizer, scheduler, completed epoch and the previous best selection score;
the existing CSV history is appended. The saved training configuration must
match. Checkpoints persist the Python, NumPy, Torch and training-loader random
states and are replaced atomically. A checkpoint without stored random states
can still be resumed, but an interrupted epoch may then be repeated. Bitwise
equivalence across a GPU migration is not guaranteed. Crop sampling is
deterministic per epoch and index.
