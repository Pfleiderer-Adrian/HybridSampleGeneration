# MVTec AD 2 example

The MVTec AD 2 example builds a deterministic train/validation/test split,
generates hybrid samples from the training partition, trains DRAEM with healthy
and hybrid samples, and evaluates the selected checkpoint on the held-out
partitions.

Set the dataset, study, and optional DTD texture locations in `settings.py` or
through the `MVTECAD2_ROOT`, `MVTECAD2_OUTPUT`, and `MVTECAD2_TEXTURES`
environment variables. The dataset root must contain one directory per MVTec
AD 2 category.

## Complete Python example for `can`

The following code is the complete workflow used by
`categories/can.py`:

```python
from examples.mvtec_ad2.common import (
    create_downstream_configuration,
    create_generator_configuration,
)
from examples.mvtec_ad2.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.downstream.runner import (
    evaluate_downstream,
    train_downstream,
)
from examples.mvtec_ad2.settings import category_root, study_folder
from examples.mvtec_ad2.splits import (
    SplitConfiguration,
    load_or_create_manifest,
    manifest_samples,
)
from hybrid_sample_generator import HybridDataGenerator


category = "can"
config = create_generator_configuration(category, anomaly_size=(3, 64, 64))

# Settings specific to the can category. Shared MVTec defaults, including the
# cVAE_ConvNeXt_2D model and its search space, are applied by the factory above.
config.generation.variation_strength = 1.5
config.fusion.parameters.max_alpha = 0.9
config.fusion.parameters.sobel_threshold = 0.05
config.extraction.roi.min_size = (256, 256)

manifest = load_or_create_manifest(
    study_folder(category) / "split_manifest.json",
    category_root(category),
    SplitConfiguration(
        test_fraction=0.2,
        validation_fraction=0.2,
        seed=42,
    ),
)

training_samples = manifest_samples(manifest, "train")
generator = HybridDataGenerator(config)
generator.ingest_dataset(MVTecAD2Dataloader(training_samples))
generator.extract_anomalies()
generator.train_generator()
generator.generate_synthetic_anomalies()
generator.plan_hybrid_samples()
generator.materialize_hybrid_samples()
config.save_config_file()

downstream_config = create_downstream_configuration()
run_folder = train_downstream(config, manifest, downstream_config)
metrics = evaluate_downstream(run_folder, manifest)
print(metrics)
```

`create_generator_configuration()` selects `cVAE_ConvNeXt_2D`, applies the
shared MVTec model parameters and `SearchSpace`, and configures extraction,
generation, matching, fusion, and training defaults. The concrete `can`
settings above override only the values that differ for this category.

The manifest is persisted beside the study and reused on later executions. Its
dataset root and split settings must still match. Set `test_fraction=0` to keep
only a validation holdout. Only the training partition is ingested into the
hybrid-generation study; validation and test samples are reserved for the
downstream evaluation.

The supporting modules have these responsibilities:

- `dataset.py` discovers MVTec files and adapts them to `InputSample` records.
- `splits.py` creates, persists, and validates the reproducible split.
- `common.py` contains defaults shared by all category examples.
- `categories/` contains the small category-specific configurations.
- `downstream/` contains DRAEM training, checkpoints, and evaluation.

Each downstream run stores its configuration, split snapshot, checkpoint,
predictions, and metrics below:

```text
<study>/downstream/draem/<timestamp>_<id>/
```

This is a custom anomaly-supervised experiment protocol and not the official
unsupervised MVTec benchmark.

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
