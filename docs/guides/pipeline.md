# Pipeline overview

A study runs through the stages below. Every stage reads its inputs from the
study repository and writes its results back, so stages can run in separate
processes and a study can be [continued](continuing-studies.md) at any point.
The [quick start](../getting-started/quick-start.md) shows all calls in order.

| # | Stage | Call | Result | Guide |
| --- | --- | --- | --- | --- |
| 1 | Ingest | `generator.ingest_dataset(dataloader)` | originals, classified as anomalous or control | [Input data](input-data.md) |
| 2 | Extraction | `generator.extract_anomalies()` | real anomaly cutouts, masks and ROIs | [Extraction](extraction.md) |
| 3 | Training | `generator.train_generator()` | trained generator checkpoints | [Generator training](training.md) |
| 4 | Generation | `generator.generate_synthetic_anomalies()` | synthetic variants of every real anomaly | [Synthetic variants](generation.md) |
| 5 | Planning | `generator.plan_hybrid_samples()` | hybrid and placement records | [Hybrid planning](matching.md) |
| 6 | Fusion | `generator.materialize_hybrid_samples()` | hybrid images and label masks | [Fusion](fusion.md) |
| 7 | Postprocessing (optional) | `generator.postprocess_hybrid_samples()` | accept/reject decision per hybrid | [Postprocessing](postprocessing.md) |
| 8 | Evaluation (optional) | `evaluate_study(config)` | metric differences and outliers | [Evaluation](evaluation.md) |

`generator` is a `HybridDataGenerator(config)`. Only ingest takes the source
dataloader; all later stages select their inputs by database fields. The
generator model is the only in-memory component: load it with
`generator.load_generator()` before stage 4 when stage 3 ran in an earlier
process.

Each stage is configured by its own section of
[`Configuration`](../configuration/index.md):

| Stage | Configuration section |
| --- | --- |
| Extraction | `config.extraction` |
| Training | `config.training`, `config.model`, `config.augmentation` |
| Generation | `config.generation`, `config.augmentation.mask_transforms` |
| Planning | `config.matching` |
| Fusion | `config.fusion` |
| Postprocessing | `config.postprocessing` |
| Evaluation | `config.evaluation` |

The results can be inspected at any stage in the
[study browser](visualization.md). Terms such as *control*, *cutout* or
*placement* are defined in the [glossary](../concepts/glossary.md).
