# MVTec AD 2 examples

This directory contains one directly executable end-to-end recipe per MVTec AD 2 category. Each recipe creates a deterministic train/validation/test split, generates hybrid samples, trains DRAEM on the training partition, and evaluates it on the held-out partitions.

Set the local paths with environment variables or edit `settings.py`:

```bash
export MVTECAD2_ROOT=/data/mvtec_ad_2
export MVTECAD2_OUTPUT=/results/mvtec_ad_2
# Optional. If omitted, DTD is downloaded below the study folder.
export MVTECAD2_TEXTURES=/data/dtd/images
```

Run a category from the repository root:

```bash
python -m examples.mvtec_ad2.categories.can
python -m examples.mvtec_ad2.categories.fabric
python -m examples.mvtec_ad2.categories.fruit_jelly
python -m examples.mvtec_ad2.categories.rice
python -m examples.mvtec_ad2.categories.sheet_metal
python -m examples.mvtec_ad2.categories.vial
python -m examples.mvtec_ad2.categories.wallplugs
python -m examples.mvtec_ad2.categories.walnuts
```

Every category module shows the complete workflow explicitly:

1. Load or create its persisted split manifest.
2. Ingest only the training partition.
3. Extract anomalies and train the generator.
4. Generate, plan, and materialize hybrid samples.
5. Train DRAEM using healthy and hybrid training samples.
6. Evaluate the selected checkpoint on validation and test data.

`common.py` holds settings shared by the recipes. Category-specific differences remain in the corresponding category module. `dataset.py` adapts MVTec images to `InputSample`; `splits.py` owns the reproducible split; `downstream/` contains the DRAEM implementation. There is intentionally no workflow CLI, step selector, study registry, or continuation manager.

Set `test_fraction=0` in a category recipe to evaluate only the validation holdout.

The saved split manifest is reused on subsequent executions and must match the requested split settings. A downstream run stores its own configuration, split snapshot, checkpoint, predictions, and metrics below:

```text
<study>/downstream/draem/<timestamp>_<id>/
```

This is a custom anomaly-supervised experiment protocol and not the official unsupervised MVTec benchmark.
