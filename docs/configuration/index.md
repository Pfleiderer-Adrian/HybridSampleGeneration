# Configuration

Create settings through `Configuration`. The reference pages list every stored parameter with its type, default, and a short explanation. Use the search box above to find a full path such as `config.training.batch_size` or a topic such as "learning rate".

```python
from hybrid_sample_generator.configuration import Configuration

config = Configuration("my-study")
config.extraction.anomaly_size = (1, 32, 32)
config.model.set_model("cVAE_ConvNeXt_2D")
config.training.batch_size = 32
config.validate()

path = config.save_config_file()        # <study folder>/configuration.json
```

A saved configuration is loaded with `load_config_file`:

```python
from hybrid_sample_generator.configuration import load_config_file

config = load_config_file("results/my-study/configuration.json")
```

By default the study folder is `results/<study name>` below the working
directory. `Configuration(name, save_path=root)` uses
`<root>/results/<study name>`, and `Configuration(name, study_folder=path)`
uses exactly `path`.

## Configuration structure

`Configuration` contains requested behavior only. Generated entities, matching
results and extraction metadata live in the study repository.

```text
config.study        identity, location, reproducibility seed
config.extraction   cutout, normalization and ROI rules
config.augmentation target-mask and training augmentation
config.generation   model sampling, feedback and variant count
config.matching     hybrid count, placement count and reuse policies
config.postprocessing quality checks for materialized hybrids
config.training     optimizer and dataloader behavior
config.evaluation   metric/outlier settings
config.model        generator choice and model-specific parameters
config.fusion       fusion backend and backend-specific parameters
```

The configuration schema is version 10 and the artifact database schema is
version 2. Configuration files and study databases of other versions cannot be
loaded; recreate the study and run `ingest_dataset()` again.

`config.study.seed` controls reproducibility for training, synthetic variant
generation, and hybrid planning. Set it before running the pipeline:

```python
config.study.seed = 123
```

## Sections

- [Study](reference/study.md): name, storage location, and seed
- [Extraction](reference/extraction.md): crops, ROIs, and normalization
- [Augmentation](reference/augmentation.md): mask transforms and training offsets
- [Generation](reference/generation.md): sampling, variants, and feedback
- [Matching](reference/matching.md): candidate selection and placement
- [Postprocessing](reference/postprocessing.md): quality rules and actions
- [Training](reference/training.md): optimization and model selection
- [Evaluation](reference/evaluation.md): foreground and outliers
- [Models](models.md): model choice, parameters, and Optuna search
- [Fusion](fusion.md): backend choice and fusion parameters

`Configuration` provides defaults, but some settings depend on each other. Run `config.validate()` after changing the model, image size, or backend. `config.schema_version` identifies the file format and is set by the code; do not change it manually.

The reference pages are generated from the configuration classes; see
[documentation](../development/documentation.md) for how to update them.
