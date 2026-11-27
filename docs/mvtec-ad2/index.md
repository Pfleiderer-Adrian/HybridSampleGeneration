# MVTec AD 2 benchmark

This example reproduces the V3 MVTec AD 2 benchmark pipeline: a grouped
train/validation split, synthetic anomalies from several sources, and a
DINOv3 segmentation model trained on a fixed mixture of healthy, real and
synthetic examples. The split, generator and downstream settings correspond
to the submitted V3 runs.

This is a custom anomaly-supervised protocol, not the official unsupervised
MVTec benchmark.

The benchmark lives in
[`examples/mvtec_ad2/`](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/blob/main/examples/mvtec_ad2) and is not part of the
`hybrid_sample_generator` package. Its downstream models are configured through
`DownstreamConfiguration`, which is separate from the package's `Configuration`;
its settings are not listed in the [configuration reference](../configuration/index.md).

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
[`synthetic_data_generation/anomalydiffusion/README.md`](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/blob/main/examples/mvtec_ad2/synthetic_data_generation/anomalydiffusion/README.md).

## Downstream

The downstream step trains a segmentation model on a fixed mixture of healthy,
real and synthetic examples and evaluates it on the validation partition.

- [Downstream training](downstream-training.md): submitted preset, run folders, real training defects, source quotas and fixed hybrids
- [Paired DINOv3](dinov3-paired.md), [DINOv3 segmentation decoder](dinov3-segmentation.md) and [SuperADD-inspired dual banks](superadd-dual-bank.md): the available models
- [Downstream evaluation](downstream-evaluation.md): AUC-PRO, checkpoint selection and validation interval
