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

## Paired baseline/hybrid comparison across categories

Validate all eight categories and persist the exact grouped splits first:

```bash
python -m examples.mvtec_ad2.run_comparison \
  --categories all --seed 42 \
  --output /mnt/results/mvtec2/experiments/comparison_v1 \
  --dry-run
```

Start preparation, training and evaluation by running the same command without
`--dry-run`. Missing hybrid data is prepared automatically, including generator
training. The existing category generator settings apply; `--epochs` overrides
DRAEM epochs only. DTD textures are prepared once for the whole experiment;
`--texture-root /path/to/dtd/images` uses an existing texture collection instead.
Use `--categories can vial` to run a subset.

Each category has one shared train/validation/test manifest and one saved DRAEM
initial state. Both variants use the same seed, initial weights, training settings,
textures, validation set and test set. Baseline uses DRAEM synthesis only;
the hybrid variant replaces 50% of anomalous training examples with hybrids.
Both variants use 50% normal training examples. Real annotated anomalies are
used to prepare hybrids and to evaluate the models; they are not direct baseline
training examples. Only the training partition is available to hybrid preparation.

The test allocation uses 20% of annotated public anomaly images, rounded to whole
acquisition groups, and adds normal images to target 25% positives. It imposes no
minimum of 30 positive images. On the current dataset, `can` yields 18 positive
and 54 normal test images. Validation retains the 20% allocation per label;
it is not forced to have 25% positives. Whole groups can cause deviations;
`split_summary.csv` records the actual counts, group counts and pixel prevalence.
`test_private` is excluded throughout.

An acquisition group is inferred from the numeric filename prefix: `000_regular`,
`000_overexposed`, `000_underexposed` and `000_shift_1` belong together within the
same original source split and label. Numeric identifiers are scoped to those
folders; they do not establish a physical identity across different source folders.
Unknown naming patterns fail validation rather than silently becoming independent
samples. No inferred group crosses train, validation and test.

Outputs include:

- `experiment.json` and `environment.json`: settings, software versions and texture hashes.
- `<category>/split_manifest.json` and `status.json`: shared split and completed stages.
- `<category>/hybrid_generation/`: automatically prepared generator study.
- `<category>/draem_initial.pt`: shared initial model weights.
- `<category>/baseline/` and `<category>/hybrid/`: checkpoints, history, predictions and metrics.
- `comparison.csv` and `comparison.md`: paired validation/test metrics, hybrid minus baseline,
  and equally weighted category means for completed pairs.

Re-run the identical command to skip completed preparation stages and completed
model runs. An interrupted model run restarts from the shared initial weights;
its previous directory is preserved under `<category>/interrupted/`. Category
failures are recorded and the remaining categories continue. Changes to experiment
settings, saved initial weights, software versions or textures require a new output
directory. This comparison is a custom supervised split, not the official MVTec
AD 2 benchmark protocol. A single seed supplies a paired comparison, not uncertainty
estimates across training seeds.
