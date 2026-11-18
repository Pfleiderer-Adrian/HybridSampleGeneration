# MVTec AD 2 example

This example reproduces the V3 MVTec AD 2 benchmark pipeline: a grouped
train/validation split, synthetic anomalies from several sources, and a
DINOv3 segmentation model trained on a fixed mixture of healthy, real and
synthetic examples. The split, generator and downstream settings correspond
to the submitted V3 runs.

This is a custom anomaly-supervised protocol, not the official unsupervised
MVTec benchmark.

## Setup

Set the dataset, output and optional DTD texture locations in `settings.py` or
through `MVTECAD2_ROOT`, `MVTECAD2_OUTPUT` and `MVTECAD2_TEXTURES`. The dataset
root contains one directory per category. The DINOv3 models need the optional
dependencies:

```bash
pip install -e ".[examples,dinov3]"
```

## Running a category

Run from the repository root:

```bash
python -m examples.mvtec_ad2.run --category can
python -m examples.mvtec_ad2.run --category can --mixture real_draem
```

`run.py` creates or reuses the split, generates hybrids when the mixture draws
hybrids, trains the downstream model and evaluates it on the validation
partition. `--skip-generation` reuses an existing hybrid study.

| Mixture (`--mixture`) | Draws per epoch |
| --- | --- |
| `real_only` | 500 healthy + 500 real |
| `real_draem` | 500 healthy + 100 real + 400 DRAEM |
| `real_cutmix` | 500 healthy + 100 real + 400 CutMix |
| `real_hybrid` | 500 healthy + 100 real + 400 hybrid |
| `real_draem_hybrid` (default) | 500 healthy + 100 real + 200 hybrid + 200 DRAEM |
| `real_anomalydiffusion` | 500 healthy + 100 real + 400 AnomalyDiffusion |
| `real_anomalydiffusion_hybrid` | … + 200 AnomalyDiffusion + 200 hybrid |
| `real_anomalydiffusion_draem` | … + 200 AnomalyDiffusion + 200 DRAEM |
| `real_anomalydiffusion_draem_hybrid` | … + 400 synthetic, rotated equally across the three sources |

AnomalyDiffusion mixtures need generated pairs; pass their manifest with
`--anomalydiffusion-manifest` (see below).

## Layout

```text
examples/mvtec_ad2/
├── run.py                       end-to-end entry point per category
├── settings.py                  dataset and output locations
├── data/                        dataset discovery, splits, transforms, patches
├── synthetic_data_generation/   one package per synthetic source
│   ├── hybrid/                  HybridSampleGenerator
│   │   ├── configuration.py     defaults shared by all categories
│   │   ├── categories/          one configuration per category
│   │   ├── pipeline.py          ingestion → … → materialization → postprocessing
│   │   └── pairs.py             hybrid/healthy pairs for training
│   ├── draem/                   Perlin-mask texture synthesis, DTD textures
│   ├── cutmix/                  defect-aware CutMix from real training defects
│   └── anomalydiffusion/        AnomalyDiffusion generation (own runtime) and loader
└── downstream/                  DINOv3 downstream model and pipeline
    ├── presets.py               submitted configuration and mixtures
    ├── configuration.py         DownstreamConfiguration
    ├── runner.py                train_downstream / evaluate_downstream
    ├── training.py, evaluation.py
    ├── data/                    training mixtures, sampling, hard normals
    └── models/                  DINOv3 ViT/ConvNeXt, decoders, SuperADD, DRAEM baseline
```

## Split

`run.py` uses a grouped split: 60 % train and 40 % validation per label, seed
42, without a test partition. All acquisition variants of a capture
(`regular`, `overexposed`, `underexposed`, `shift_1`–`shift_3`) stay in the same
partition. The manifest is saved as `<study>/split_manifest.json` and reused;
its dataset root, settings and grouping mode must match. Only the training
partition is ingested into the hybrid study.

## Synthetic sources

**Hybrid.** `categories/<category>.py` contains only the generator
configuration; `create_generator_configuration()` applies the shared defaults
(`cVAE_ConvNeXt_2D` and its search space, extraction, matching, fusion and
training). `can` and `wallplugs` enable postprocessing with action `exclude`
to drop hybrids whose anomaly lies outside the object; `wallplugs` exempts the
captures `test_public_bad_010` and `test_public_bad_012`, whose defects extend
beyond the plug outline (`capture_variants()` expands a capture to its six
variants).

**DRAEM** draws Perlin masks filled with DTD textures on healthy training
patches. Without `MVTECAD2_TEXTURES` the textures are downloaded into the study.

**CutMix** copies rectangles around real training defects onto healthy
training patches.

**AnomalyDiffusion** runs in its own Python environment and reuses the hybrid
masks; see
[`synthetic_data_generation/anomalydiffusion/README.md`](synthetic_data_generation/anomalydiffusion/README.md).

## Downstream

`downstream/presets.py` returns the submitted configuration: frozen
DINOv3 ViT-S/16 (pinned revision) with a CNN decoder, native 256×256 patches,
batch size 8, 100 epochs, validation every 5 epochs and checkpoint selection by
validation pixel AUPRO@0.05. Each run stores its configuration, split snapshot
with hybrid provenance, checkpoints, history, predictions and metrics below:

```text
<study>/downstream/dinov3_paired/<timestamp>_<id>/
```

To change settings, start from `create_downstream_configuration(mixture)` and
call `train_downstream()` and `evaluate_downstream()` directly.

### Adding real training defects to DINOv3

Set `data.include_real_defects = true` (default: `false`) for
`dinov3_paired` or `dinov3_segmentation`. Every masked defect image in the
persisted **training partition** contributes one defect-containing crop per
epoch, shuffled together with the existing mixture. These examples are added
to `samples_per_epoch`; the original normal/DRAEM/hybrid quotas remain intact.
Validation and test partitions are unchanged and must not overlap the real
training image paths. No private images are added.

Real defect images use Focal + Dice supervision. They have no healthy partner,
so partner and consistency losses are skipped for these samples. Other paired
samples retain their original losses. The run records the included image IDs
in `real_training_sources.json` and the observed `real` count in
`training_history.csv`. Use `training.selection_metric = "pixel_aupro_0.05"`
for selection on real validation defects without synthetic validation pairs.

Training can resume in the same output directory with
`train(..., resume_checkpoint=path_to_last_checkpoint)`. This restores model,
optimizer, scheduler, completed epoch and the previous best selection score;
the existing CSV history is appended. The saved training configuration must
match. New checkpoints also persist Python, NumPy, Torch and training-loader
random states and are replaced atomically. Legacy checkpoints lack those random
states, so an interrupted epoch may be repeated and bitwise equivalence across
a GPU migration is not guaranteed. Deterministic epoch/index-based crop sampling
is retained.

### Fixed source quotas for downstream comparisons

`config.data.source_quotas` can specify exact per-epoch counts for `normal`,
`real`, `draem`, `cutmix`, `hybrid` and `anomalydiffusion`. Counts must sum to
`config.data.samples_per_epoch`. When set, these quotas replace the legacy
fraction-based plan and the additional once-per-image real-defect entries.
Every real training image is sampled before repeats within each epoch; the real
quota must therefore cover the training defect-image count. Hard-normal sampling
is disabled for this mode.

Set `config.dinov3.pair_healthy_samples = False` to supervise healthy examples
with segmentation loss only (Focal, with Dice omitted for empty masks). Real
examples always receive Focal + Dice only. Synthetic examples retain healthy
partner supervision and background probability consistency. The default `True`
preserves the historical healthy paired-loss behavior.

`cutmix` is a defect-aware rectangular 2D adaptation: donor defect pixels guide
the rectangle center, Beta(1,1) sets its area, image pixels and ground-truth labels
are copied together, and the healthy receiver is retained as the partner. The
rectangle itself is **not** the defect label. Donors and receivers must come
from the training partition. This mode needs no hybrid-generation study.

`config.data.source_quota_rotation` lists synthetic sources whose combined
quota is split equally per epoch; the rounding remainder rotates across them.
Healthy and real quotas stay fixed.

### Fixed hybrids and separate output folders

`train_downstream(..., hybrid_ids=[...])` trains on exactly the listed hybrids
in the given order, for example the `hybrid_provenance` of an earlier run.
This selection replaces the postprocessing filter. `output_root=...` writes
the run below another folder while still loading hybrids from the study.
