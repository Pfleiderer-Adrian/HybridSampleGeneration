# Quick start

```python
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.evaluation.service import evaluate_study
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from hybrid_sample_generator.visualization import run_hybrid_visualizer

config = Configuration(
    "study-01",
    study_folder="results/study-01",
)
config.extraction.anomaly_size = (1, 32, 32)
config.model.set_model("cVAE_ConvNeXt_2D")

config.generation.variants_per_real_anomaly = 5
config.training.num_trials = 5
config.training.trial_selection = "best"
config.matching.hybrids_per_original = 3
config.matching.anomalies_per_hybrid = 2
config.matching.reuse_synthetic_across_hybrids = True
config.matching.allow_sibling_variants_in_same_hybrid = False
config.matching.routine = "local"

generator = HybridDataGenerator(config)
summary = generator.ingest_dataset(all_samples_dataloader)
generator.extract_anomalies()
generator.train_generator()
generator.generate_synthetic_anomalies()

# Planning writes HybridSample/Placement records and updates the matching cache.
generator.plan_hybrid_samples()

# Fusion consumes the stored plan and writes generated payloads.
generator.materialize_hybrid_samples()

hybrid_dataset = generator.datasets.hybrid_samples(
    load_to_ram=False,
    numpy_mode=True,
)
evaluation = evaluate_study(config)
config.save_config_file()
run_hybrid_visualizer(config)
```

Ingest is the only phase that accepts the source dataloader. Extraction and
planning select anomalous or normal originals by database fields.
Repository-backed phases need no load step: a new
`HybridDataGenerator(config)` can immediately continue from persisted records.
Only the generator model has to be loaded explicitly before producing new
variants, because it is an in-memory runtime component. The classical fusion
backend is created on demand from the validated fusion parameters.
Save the configuration before opening the visualizer: its configuration view
reads the saved JSON, and the GUI call blocks until the window closes.

Run the bundled examples from the repository root so their package imports are
resolved consistently:

```bash
python -m examples.image_2d.main
python -m examples.nifti_3d.main
python -m examples.mvtec_ad2.categories.can
```

The [MVTec AD 2 guide](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/blob/main/examples/mvtec_ad2/README.md)
describes the category recipes, persisted data splits, hybrid generation, and
downstream DRAEM evaluation.
