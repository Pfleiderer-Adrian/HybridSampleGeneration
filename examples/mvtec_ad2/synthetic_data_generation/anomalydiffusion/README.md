# AnomalyDiffusion pairs for MVTec AD 2

This package generates image/healthy-partner/mask pairs with the original
[AnomalyDiffusion](https://github.com/sjtuplayer/anomalydiffusion) code (AAAI
2024, commit `b85297f`) for the downstream mixtures `real_anomalydiffusion*`.
Its defaults are the profile of the V3 benchmark submissions: **2x windows,
last checkpoint after 50,000 appearance steps, adaptive attention (AAR)**.

The integration is an adaptation to MVTec AD 2, not the original MVTec AD
protocol of the paper.

## Pipeline

1. **prepare** – For every real defect of the split's `train` partition, native
   128×128 windows are centered on each mask component (64 px stride for
   components larger than 128 px) and upscaled 2x to 256×256 (RGB bicubic,
   mask nearest). Every defect pixel is covered. The `validation` partition
   provides validation windows. Healthy and test images are not used.
2. **train** – The appearance stage (text embeddings and spatial encoder) is
   trained for 50,000 steps on the frozen base latent diffusion model with the
   original Lightning module: batch size 4, effective learning rate 0.02,
   seed 42. The final checkpoint is selected (`selection.json`). There is no
   separate mask training: masks come from the hybrid study.
3. **generate** – Masks and healthy backgrounds come from a materialized hybrid
   study. For each of 1,000 pairs, one mask component is drawn, a native
   256×256 crop is taken around it on the healthy training image, every
   covering 128 window is inpainted at 2x (DDIM 200, η = 1, AAR) and the
   windows are scaled back and averaged. Outside the windows the healthy image
   is kept; inside, the generated output is used without mask compositing.

Documented adaptations: masks in the 32×32 latent space use max pooling
(`maxpool`), so every occupied latent cell is kept; images are PNG; the
original 512 px `data_enhance` scaling is disabled.

## Setup

Generation and training use a separate Python 3.10 environment (Torch 2.0.1,
PyTorch Lightning 1.5). Run from the repository root:

```bash
bash examples/mvtec_ad2/synthetic_data_generation/anomalydiffusion/setup.sh
```

The script clones the pinned upstream code and taming-transformers and
installs `requirements.lock` into `.venv/` next to this file.
`ANOMALYDIFFUSION_HOME` selects a different location for `upstream/`,
`.tools/` and `.venv/`.

## Usage

```bash
AD=examples/mvtec_ad2/synthetic_data_generation/anomalydiffusion/run.sh
STUDY=$MVTECAD2_OUTPUT/can/hybrid_generation      # hybrid study from examples.mvtec_ad2.run

$AD fetch-weights --output <cache>/model.ckpt      # 6.2 GB, verified by SHA256
$AD prepare  --category can --split $STUDY/split_manifest.json --output <work>/can
CUDA_VISIBLE_DEVICES=0 $AD train    --inputs <work>/can --checkpoint <cache>/model.ckpt
CUDA_VISIBLE_DEVICES=0 $AD generate --inputs <work>/can --checkpoint <cache>/model.ckpt --hybrid-study $STUDY
```

`generate` writes `<work>/can/generated/manifest.json`. Pass it to the
downstream run:

```bash
python -m examples.mvtec_ad2.run --category can --mixture real_anomalydiffusion_hybrid \
    --skip-generation --anomalydiffusion-manifest <work>/can/generated/manifest.json
```

Without `--hybrid-ids`, all generated hybrids on healthy training images are
used; hybrids rejected by postprocessing with action `exclude` are skipped.
`--hybrid-ids` restricts and orders the masks, for example to the hybrids of a
previous downstream run: its `split_manifest.json` contains `hybrid_provenance`
and can be passed directly. The V3 submissions used the hybrids of the
submitted `real_draem_hybrid` runs.

The split of `prepare`, the hybrid study and the downstream run must be the
same; the downstream loader checks the split fingerprint, training-only
provenance and binary nonempty 256×256 masks.
