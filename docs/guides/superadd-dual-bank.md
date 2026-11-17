# SuperADD-inspired dual memory banks

`superadd_dual_bank` is an independent implementation inspired by
[SuperADD](https://github.com/LukasRoom/SuperADD), not a reproduction of its
published results. It uses the existing Transformers DINOv3 loader, normalized
features, bounded priority-reservoir sampling and chunked exact nearest-neighbor
search. It does not include upstream coreset selection, brightness augmentation
or morphological postprocessing. DINOv3 weights remain frozen; no decoder or
gradient optimizer is used.

```python
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration

config = DownstreamConfiguration(model="superadd_dual_bank")
config.data.mode = "patch"
config.data.patch_size = (256, 256)
config.data.normal_fraction = 0.75
config.data.hybrid_fraction = 0.5  # 0.0 for DRAEM-only defect references
config.memory_bank.layers = (3, 6, 9, 12)  # one-based ViT-S block outputs
config.memory_bank.normal_capacity = 20000
config.memory_bank.defect_capacity = 5000
config.memory_bank.defect_weight = 1.0
```

The **normal bank** always uses all healthy training originals with overlapping
tiles, independently of the synthetic mixture. Baseline and hybrid runs can thus
use the same normal references. Normal-distance scaling is determined from held-out
healthy validation images (up to `calibration_images`, default 16), never test
images or masks.

The **defect bank** uses one deterministic pass through the existing training
mixture, with `samples_per_epoch` defining the total sample budget. At 1000 samples,
the example uses 750 normal / 125 DRAEM / 125 hybrid samples; with
`hybrid_fraction=0.0` it uses 750 normal / 250 DRAEM. Only non-normal examples
contribute defect features. Image and mask transformations remain aligned.
8-connected mask regions select any overlapping feature cell via max pooling;
this avoids dropping tiny defects by mask downsampling, but selected features
can still contain normal context. Each region supplies at most
`tokens_per_region` candidates (default 64). Banks retain a bounded random
selection independently per layer. Equal image quotas do not imply equal token
quotas, and duplicate donor variants may still influence the reference distribution.

For a pure normal-bank baseline, set `normal_fraction=1.0` and
`hybrid_fraction=0.0`. For a score ablation with constructed defect banks but no
defect contribution, set `defect_weight=0.0`.

Per layer, let `dn` be the nearest normal distance, `da` the nearest defect
distance and `s` the 95th percentile of normal validation distances. The score is:

```text
base = dn / (dn + s)
support = sigmoid((dn - da) / s)
score = base * (1 + defect_weight * support) / (1 + defect_weight)
```

With no defect bank or zero weight the score is `base`. The added evidence favors
known defect features while retaining a normal-distance contribution for unknown
defects. Layers are averaged and scores are interpolated to image resolution.
Scores lie in [0, 1], but are not calibrated probabilities; a binary threshold
should be chosen on validation data. Inference uses only the input image.

`query_chunk_size` (256) and `reference_chunk_size` (2048) bound each distance
matrix. `training.epochs` and learning rate do not affect bank construction.
The usual `train_downstream` / `evaluate_downstream` functions, persisted splits,
native-image tiled evaluation and AUROC/AP/AUC-PRO0.05 metrics are supported.
The reported focal `segmentation_loss` is a diagnostic, not a fitted objective.

Runs live in `study/downstream/superadd_dual_bank/<run_id>/`. `bank_sources.json`
records normal/calibration sources; `bank_build.json` records sampling counts,
reference counts and distance scales. Both `best.pt` and `last.pt` contain the
same fitted state, including backbone configuration/weights and bank buffers,
allowing offline reload without another pretrained download. Install the same
`dinov3` optional extra used by the other DINOv3 downstream models.
