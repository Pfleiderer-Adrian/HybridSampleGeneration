# Continuing and repeating a study

For a study with synthetic variants and a saved hybrid plan, continue directly
with materialization:

```python
from hybrid_sample_generator.configuration.root import load_config_file
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator

config = load_config_file("results/study-01/configuration.json")
generator = HybridDataGenerator(config)
generator.materialize_hybrid_samples()
```

To build a new plan from existing synthetic variants, call
`generator.plan_hybrid_samples()` before materialization. To regenerate synthetic
variants with a saved generator, set `config.training.trial_selection` to
`"best"`, `"last"`, or a concrete nonnegative trial ID and call
`generator.load_generator()` followed by
`generator.generate_synthetic_anomalies()`.

Repeating a phase has the following effects:

| Phase | Effect on existing results |
|---|---|
| `ingest_dataset()` | Replaces all originals and removes real/synthetic anomalies, hybrids, placements and matching-cache entries. |
| `extract_anomalies()` | Replaces real anomalies and removes synthetic anomalies, hybrids, placements and matching-cache entries. |
| `generate_synthetic_anomalies()` | Replaces synthetic variants and removes hybrids and placements; the real-ROI matching cache remains available. |
| `plan_hybrid_samples()` | Replaces all hybrid and placement records, including generated statuses and their artifact references; retains and updates the matching cache. |
| `materialize_hybrid_samples()` | Processes every stored hybrid, including already generated or failed ones, and rewrites its generated outputs. |

These resets remove database records; old array files can remain on disk without
repository references. Repeating a phase is not an incremental append or an
automatic skip of completed work. After changing generated data, rerun evaluation
to replace its previous CSV results.
