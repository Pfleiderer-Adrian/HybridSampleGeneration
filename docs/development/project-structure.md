# Project structure

- `pyproject.toml` — package metadata, stable dependencies and experimental extras
- `hybrid_sample_generator/configuration/` — validated, section-based configuration
- `hybrid_sample_generator/domain/` — input and persisted study records
- `hybrid_sample_generator/persistence/` — repository, study paths and artifacts
- `hybrid_sample_generator/pipeline/` — ingestion and the public orchestration facade
- `hybrid_sample_generator/imaging/` — shared image, similarity and mask operations
- `hybrid_sample_generator/extraction/` — dimension-independent anomaly extraction
- `hybrid_sample_generator/matching/` — hybrid planning and matching cache
- `hybrid_sample_generator/generation/` — generation service, model registry,
  training and supported VAEs
- `hybrid_sample_generator/fusion/` — fusion service, shared preprocessing and
  the classical alpha-blending and Poisson backends
- `hybrid_sample_generator/evaluation/` — pairwise metrics, outliers and reports
- `hybrid_sample_generator/datasets/` — repository-backed training datasets
- `hybrid_sample_generator/visualization/` — study browser and maintenance UI
- `examples/` — shared example helpers plus 2D image, 3D NIfTI and MVTec AD 2 workflows
- `tests/` — tests grouped by the same feature boundaries
- `experiments/` — unsupported prototypes, optional dependencies and isolated tests
