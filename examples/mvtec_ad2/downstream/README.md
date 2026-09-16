# MVTec AD 2 / DRAEM workflow

See [the example guide](../README.md) for configuration, study APIs and CLI usage.
All shared generator and DRAEM settings live in `../presets.py` in
`apply_global_defaults(config)`. Category functions such as `configure_can(config)`
can override the same fields:

```python
config.downstream.data.hybrid_fraction = 1.0
config.downstream.data.normal_fraction = 0.5
config.downstream.data.mode = "patch"
config.downstream.data.patch_size = (512, 512)
config.downstream.training.batch_size = 2
config.downstream.training.epochs = 100
```

The configuration types and backwards-compatible run serialization are defined in
`downstream/configuration.py`. The example's root `Configuration` composes them
as `config.downstream`; the library remains independent of DRAEM.

For a new experiment, use `run_new_experiment(EXPERIMENT, steps=FULL_EXPERIMENT)`.
To train on an existing study with its saved settings:

```python
from examples.mvtec_ad2.pipeline import TRAIN_DOWNSTREAM, run_existing_studies

results = run_existing_studies(["/path/to/study"], steps=TRAIN_DOWNSTREAM)
```

To apply current presets explicitly or make local changes:

```python
from examples.mvtec_ad2.presets import apply_downstream_preset
from examples.mvtec_ad2.studies import open_study
from examples.mvtec_ad2.pipeline import TRAIN_DOWNSTREAM, run_study

study = open_study("/path/to/study")
apply_downstream_preset(study.config, study.category)
study.config.downstream.training.epochs = 20
result = run_study(study, steps=TRAIN_DOWNSTREAM)
```

New DRAEM runs save their resolved configuration and split separately. Evaluation
always uses that run snapshot, regardless of changes to the study or current
presets. Select a concrete run with `downstream_run_id` when evaluating without
training. Evaluation needs the real holdout images, but no training textures,
hybrid artifacts or generator model.

The ordered step vocabulary is:

```text
ingest → extract → train_generator OR load_generator → generate_synthetic
       → plan → materialize → export → train_downstream → evaluate_downstream
```

`generate_synthetic` loads the saved generator when training/loading was not
selected. Other omitted steps are not performed automatically. Missing
prerequisites fail before expensive execution. A Perlin-only baseline uses
`hybrid_fraction=0` and `TRAIN_DOWNSTREAM` on a new experiment, without generator
artifacts. Export and review can also operate on legacy studies without splits.

## Splits and provenance

With `test_enabled=False`, `test_fraction` is ignored and all discovered healthy
images and annotated anomalies are assigned to training/validation. Training
anomalies serve as generator donors, not as healthy reconstruction targets.
No test metrics or test predictions are produced. With testing enabled, default
fractions yield approximately 60/20/20 training/validation/test. Fractions are
relative to the whole dataset and stratified per label, with nonempty partitions.
Too few images cause an explicit error.

Discovery pools healthy images from train/validation/val/test_public and annotated
public anomalies. The previous `include_public_good_controls` runner flag is
removed: the manifest now controls partition membership. This is a custom
anomaly-supervised protocol, not the official unsupervised MVTec benchmark.
Splitting operates per image; correlated frames need an additional grouped split
before making benchmark claims. Missing anomaly masks are warned about and skipped
by the existing discovery adapter.

Only training originals enter the generator repository. Healthy backgrounds and
all anomaly donor relationships are verified. The manifest is saved during
preparation and configuration once at execution. `save`/`save_config` are not steps.
On reuse, split membership is loaded rather than rediscovered. A changed split or seed requires a new
save_path. Legacy studies without a manifest cannot be made independent after
generator training and are rejected for training workflows. They remain viewable
via `review_studies([study_folder])` from `examples.mvtec_ad2.review` and usable for generator-quality evaluation
with `actions=("evaluate_generator", "visualize")`, without a manifest or original
dataset. `find_studies(save_path, categories)` finds actual saved folders without
reconstructing study names from presets. The saved configuration and repository
must still use a supported schema. External generator database overrides are not
offered because their training provenance is unknown.

The generator's own `config.training.validation_ratio` divides training donors
internally; those validation cutouts do not belong to the downstream holdout.

## Mixture and outputs

`data.hybrid_fraction` is the hybrid share of anomalous training examples;
`data.normal_fraction` is the healthy share of all examples. At 0.5 for both,
1,000 examples contain 500 healthy, 250 Perlin and 250 hybrid examples. Set
`data.texture_root` to an existing texture collection, or leave it `None` to
automatically download DTD R1.0.1 from Oxford when Perlin training starts.
The approximately 625 MB archive is downloaded with a progress bar, checked and
its 5,640 images extracted under `<study_folder>/downstream/textures/dtd/images`.
The resolved path is saved in the run configuration; the study configuration keeps
the configured `None` or explicit path. Subsequent runs reuse
the prepared dataset. Temporary downloads are removed on failure; no partial
dataset is marked ready. Explicit invalid paths fail instead of downloading.
Hybrid-only training and evaluation-only runs never download textures.
Quotas are rounded per epoch and shuffled deterministically, with replacement.
100% hybrids needs no textures; 0% hybrids needs no hybrid study.

Pairs contain the hybrid, its healthy original and its binary mask, jointly
cropped/flipped in patch mode. Both images use `image_scale=255` for this adapter. Grayscale
images become RGB. Hybrids on anomalous backgrounds are excluded.

Results live under
`<save_path>/<category>/results/<study_name>/downstream/draem/<run_id>/`:
configuration, split/provenance manifest, training/source-count CSV, best/last
checkpoints, metrics, NumPy anomaly maps and selected PNG previews. Image AUROC/AP
are exact; pixel AUROC/AP use 4,096 bins and original-resolution masks in patch mode
(resized masks in image mode). Single-class metrics
are null. Image scores use 21x21 average pooling followed by a maximum. Model
selection minimizes focal segmentation loss on real validation images; these
have no paired healthy target. No classification threshold is fitted. Checkpoints
contain optimizer state, but automatic training resume is not exposed.

Materialization writes arrays to the study Artifact Store. The optional `export`
step writes lossless PNG images and binary 0/255 masks to `exports/images` and
`exports/segmentations`, using matching filenames with the hybrid record ID to
avoid name collisions. It can run on existing materialized studies without loading
the generator or materializing again. DRAEM reads the Artifact Store, not exports.

## Native-resolution patch training

New configurations default to `data.mode = "patch"`, `patch_size = (512, 512)`,
`patch_overlap = 0.5` and `training.batch_size = 8` in the shared MVTec presets (the raw dataclass default is 2). Patch dimensions must be
multiples of 32 and at least 64. `image_size` is ignored in patch mode.
The core generator and stored hybrid/original arrays are not modified.

- Hybrid training crops use identical coordinates for hybrid, healthy target and
  mask. A random foreground pixel anchors each crop, so even a one-pixel anomaly
  remains represented. Empty stored masks fail with the hybrid ID before training.
- Healthy training crops are random. Perlin synthesis is applied *after* cropping
  a healthy image. The existing normal/Perlin/hybrid quotas now count patches.
- Crops use original pixels, without spatial interpolation. Images smaller than
  a patch are edge-padded; masks are zero-padded. Aligned flips remain enabled.
- Validation and test use a deterministic sliding grid over the entire image,
  without consulting masks for tile selection. The right/bottom edges are always
  covered. Tiles are processed in batches of `training.batch_size`; logits are
  averaged in overlapping regions on CPU, then converted to probabilities.
  Padding is removed before losses, metrics and saved predictions are computed.
- Every full image contributes once to image metrics. Pixel metrics and the
  validation loss use the reconstructed original-resolution map, not duplicated
  overlapping tiles. The 21x21 image-score pooling is in original pixels.

Only patches go to the GPU. Full-resolution maps are assembled in CPU memory.
Validation is run over complete images each epoch and can therefore take longer
than the previous resized evaluation. Model selection still uses validation loss;
the test partition is only evaluated by `evaluate_downstream`.

For the former whole-image baseline, explicitly set `data.mode = "image"` and
`data.image_size`. Saved run snapshots predating patch mode retain this baseline
when loaded; current category presets select patch mode for new studies. Metrics
from different spatial modes are not directly equivalent. Split membership and
training-donor provenance checks are unchanged.

## CLI

```bash
python -m examples.mvtec_ad2 continue --study-folder /path/to/study \
  --steps train_downstream evaluate_downstream
python -m examples.mvtec_ad2 evaluate --study-folder /path/to/study --run-id RUN_ID
```

To reuse local textures, configure `config.downstream.data.texture_root`;
`None` enables the study-local download when required. There is no separate
training JSON input. Existing studies keep saved settings unless explicitly changed.

## Tests

MVTec workflow and downstream tests live in `tests/example_cases/mvtec_ad2/`.
They are included in the regular project test command:

```bash
python -m unittest discover -s tests -v
```

To run only these MVTec integration tests:

```bash
python -m unittest discover -s tests/example_cases/mvtec_ad2 -t . -v
```

## Architecture

The independent local implementation follows the DRAEM topology of
[Zavrtanik et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html),
referencing the [official implementation](https://github.com/VitjanZ/DRAEM).
MSE, Gaussian-window SSIM and focal loss train both networks jointly. Native
NumPy/Torch texture augmentation differs from the upstream imgaug pipeline;
this is not a bit-for-bit reproduction. Upstream checkpoints use different names.
No upstream source is vendored and no pretrained models are downloaded. DTD is
downloaded only when needed and no texture path is configured. Reference
widths are 128/64, with smaller widths available for CPU smoke tests.
