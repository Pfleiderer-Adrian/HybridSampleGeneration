# Hybrid Sample Generation

This project extracts real anomalies from labelled 2D images or 3D volumes,
trains a generative model, creates multiple synthetic variants and places them
into original control samples. It is based on the IEEE paper
[AMLDS63918.2025.11159383](https://doi.org/10.1109/AMLDS63918.2025.11159383).

## Data model

Study metadata and relationships are stored in `artifacts.sqlite`. NumPy arrays
remain normal files below `artifacts/`; the database stores paths relative to
the study folder.

```text
OriginalSample  1 ── 0..n  RealAnomaly  1 ── 0..n  SyntheticAnomaly
       │                                             │
       └── 0..n  HybridSample  1 ── 1..n  Placement ┘
```

A placement is an independent record. It identifies one synthetic anomaly,
one hybrid sample, an insertion order, a matching method and score, and an
explicit normalized center position. Positions use `(y, x)` for 2D and
`(z, y, x)` for 3D. This removes the old one-to-one and filename-based
relationship between anomalies and generated samples.

Database constraints enforce unique component/variant/order combinations and
foreign-key integrity. Entity IDs are deterministic hashes of stable source
identity and variant indices; filenames do not carry identity.

One study has this layout:

```text
study/
  configuration.json
  artifacts.sqlite
  artifacts/
    original_samples/<id>/{image,segmentation}.npy
    real_anomalies/<id>/{image,segmentation,roi_image,roi_segmentation}.npy
    synthetic_anomalies/<id>/{image,segmentation}.npy
    hybrid_samples/<id>/{image,segmentation}.npy
    placements/<id>/{roi_image,roi_segmentation}.npy
  evaluation_results/
  exports/
```

## Input

The pipeline accepts channel-first arrays:

- 2D: `(C, H, W)`
- 3D: `(C, D, H, W)`

A dataloader may yield the compact tuple `(image, segmentation, source_name)`.
For unambiguous source identity and provenance, yield `InputSample` records or
implement `iter_input_samples()`:

```python
from synthesizer.InputSample import InputSample

yield InputSample(
    image=image,
    segmentation=mask,
    source_name="sample-001",
    source_image_path="/dataset/images/sample-001.png",
    source_segmentation_path="/dataset/masks/sample-001.png",
)
```

The bundled image, NIfTI and MVTec AD 2 loaders expose this typed boundary.
Input arrays are snapshotted into the study, so materialization does not depend
on iterating the original dataloader again.

## Usage

```python
from synthesizer.Configuration import Configuration
from synthesizer.Evaluation import evaluate_study
from synthesizer.HybridDataGenerator import HybridDataGenerator
from data_handler.Visualizer import run_hybrid_visualizer

config = Configuration(
    "study-01",
    "VAE_ResNet_2D",
    anomaly_size=(3, 32, 32),
    study_folder="results/study-01",
)

config.generation.variants_per_real_anomaly = 5
config.matching.hybrids_per_original = 3
config.matching.anomalies_per_hybrid = 2
config.matching.reuse_synthetic_across_hybrids = True
config.matching.allow_sibling_variants_in_same_hybrid = False
config.matching.routine = "global"

generator = HybridDataGenerator(config)
generator.extract_anomalies(anomaly_dataloader)
generator.train_generator(no_of_trials=5)
generator.generate_synthetic_anomalies()

# Planning only writes HybridSample and Placement records.
generator.plan_hybrid_samples(control_dataloader)

# Fusion consumes the stored plan and writes generated payloads.
generator.materialize_hybrid_samples()

hybrid_dataset = generator.datasets.hybrid_samples(
    load_to_ram=False,
    numpy_mode=True,
)
evaluation = evaluate_study(config)
run_hybrid_visualizer(config)
config.save_config_file()
```

Repository-backed phases need no load step. A new `HybridDataGenerator(config)`
can immediately plan from persisted synthetic anomalies or materialize a
persisted hybrid plan. Only the generator model has to be loaded explicitly
before producing new variants, because it is an in-memory runtime component.

## Configuration

`Configuration` contains requested behavior only. Generated entities, matching
results and extraction metadata live in the study repository.

```text
config.study        identity, location, reproducibility seed
config.extraction   cutout, normalization and ROI rules
config.augmentation target-mask and training augmentation
config.generation   model sampling, feedback and variant count
config.matching     hybrid count, placement count and reuse policies
config.training     optimizer and dataloader behavior
config.evaluation   metric/outlier settings
config.model        generator choice and model-specific parameters
config.fusion       fusion backend and backend-specific parameters
```

The current configuration schema is version 3. Older filename/CSV layouts are
intentionally unsupported.

### Synthetic variants

`config.generation.variants_per_real_anomaly` controls how many children are
generated for every `RealAnomaly`. Each child has its own deterministic ID,
variant index, seed, image and target mask. Feedback generation is bounded by
`config.generation.feedback.max_attempts`.

### Hybrid planning

- `hybrids_per_original`: number of planned hybrid variants per input original.
- `anomalies_per_hybrid`: target placement count in each hybrid.
- `max_anomalies_per_hybrid_deviation`: deterministic random deviation around
  the placement count.
- `reuse_synthetic_across_hybrids`: whether the same synthetic ID may be used
  by more than one hybrid.
- `allow_sibling_variants_in_same_hybrid`: whether variants with the same real
  parent may occur together in one hybrid.
- `intensity_weight` and `gradient_weight`: weights for template matching.
- `seed`: reproducibility seed owned by the matching phase.

`local`, `global` and `batchwise` compute a placement candidate once per real
anomaly ROI and original sample. A concrete synthetic child is chosen only
after candidate ranking. `fixed_from_extraction_control_fusion` reuses source
centers on arbitrary controls;
`fixed_from_extraction_anomaly_fusion` joins originals and real anomalies by
foreign key and places variants back at their extraction positions.

## Datasets, evaluation and visualization

`StudyDatasets` creates short-lived `RealAnomalyDataset`,
`SyntheticAnomalyDataset` and `HybridSampleDataset` views over repository
records. They do not scan folders or align files by basename. Dataset objects
are not persistent state of `HybridDataGenerator`; callers choose explicitly
whether a view should load arrays into RAM.

Evaluation joins each synthetic anomaly to its real parent through
`real_anomaly_id`. Placement ROI comparisons use the full
Original → Hybrid → Placement → Synthetic → Real join. The CSV output contains
all relevant IDs, so multiple variants cannot overwrite or masquerade as one
pair.

`run_hybrid_visualizer(config)` opens the same relationship as a tree and
resolves every displayed array through the repository. `evaluate_study(config)`
does the same for evaluation, without constructing a generation orchestrator.

## Tests

```bash
python -m unittest discover -s tests -v
```

The integration tests cover multiple real components, multiple synthetic and
hybrid variants, normalized multi-placement records, unique artifacts,
foreign-key traversal, 2D/3D coordinates, materialization and FK-based
evaluation.
