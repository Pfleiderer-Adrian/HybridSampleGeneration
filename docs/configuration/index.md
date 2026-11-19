# Configuration

Create settings through `Configuration`. The reference pages list every stored parameter with its type, default, and a short explanation. Use the search box above to find a full path such as `config.training.batch_size` or a topic such as "learning rate".

```python
from hybrid_sample_generator.configuration import Configuration

config = Configuration("my-study")
config.extraction.anomaly_size = (1, 32, 32)
config.model.set_model("cVAE_ConvNeXt_2D")
config.training.batch_size = 32
config.validate()
```

## Configuration structure

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

The current configuration schema is version 9 and the artifact database schema
is version 2. Older study databases and filename/CSV layouts are intentionally
unsupported; recreate the study and run `ingest_dataset()` again.

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
- [Training](reference/training.md): optimization and model selection
- [Evaluation](reference/evaluation.md): foreground and outliers
- [Models](models.md): model choice, parameters, and Optuna search
- [Fusion](fusion.md): backend choice and fusion parameters

`Configuration` provides defaults, but some settings depend on each other. Run `config.validate()` after changing the model, image size, or backend. `config.schema_version` identifies the file format and is set by the code; do not change it manually.

## Parameter schema

Every parameter is declared once, on its configuration dataclass, together with its allowed range and description:

```python
batch_size: int = param(32, ge=1, doc="Batch size for candidate scoring.")
```

`param()` accepts `ge`/`gt`/`le`/`lt` for numeric bounds (applied to each element of a sequence), `choices` for a fixed set of values, `min_length`/`max_length` for sequences, and `doc`. Validation, the reference pages, and the schema below all read these declarations. Rules that relate several parameters, such as "at least one matching weight must be positive", stay in the `validate()` methods.

`Configuration.schema()` returns the declarations as JSON-compatible data. Graphical front ends build their forms from it, so new parameters, generator models, and fusion backends appear without changes to the front end:

```python
from hybrid_sample_generator.configuration import Configuration

schema = Configuration.schema()
schema["sections"]["matching"]["fields"]["batch_size"]
# {"type": "int", "min": 1, "doc": "Batch size for candidate scoring.", "default": 32}
```

| Key | Meaning |
| --- | --- |
| `type` | `bool`, `int`, `float`, `str`, `choice`, `list`, `map`, `any_of`, `any`, or `group` for a nested section |
| `default` | Default value in its JSON form; absent for `required` parameters and those with a `default_label` |
| `min`, `max` | Inclusive bounds unless `min_exclusive` or `max_exclusive` is true |
| `choices` | Allowed values of a `choice` |
| `items`, `min_length`, `max_length` | Element descriptor and length bounds of a `list` |
| `values`, `keys` | Value descriptor of a `map` and descriptors of its known keys |
| `options` | Alternatives of an `any_of`, for example a fixed name or a trial ID |
| `nullable` | `None` is allowed |
| `validated` | `false` when `choices` lists recommended values and other spellings are also accepted |
| `custom` | The group needs a dedicated editor; generic forms should skip it. Currently set for `augmentation.mask_transforms` |
| `doc` | Explanation shown in the reference |

`schema["sections"]["model"]` and `schema["sections"]["fusion"]` have the type `variant`. Their `variants` hold one parameter group per registered model or backend, with the defaults of that variant; `selector` names the key that chooses between them (`name` or `backend`). Each model variant also lists its default Optuna search space and, per parameter, the distribution types that `config.model.search` accepts.

The schema describes single parameters. To check a complete set of values, including rules across parameters, call `Configuration.from_dict(values)`; errors name the parameter path.

## Updating the reference

The pages in `configuration/reference/` are generated from `Configuration.schema()`. When parameters, ranges, or explanations change, update the `param(...)` declarations on the configuration classes and regenerate:

```bash
python scripts/generate_config_reference.py
python scripts/generate_config_reference.py --check
```

The check reports new fields without explanations and outdated reference pages. To preview the site locally, install `mkdocs-material` and run `mkdocs serve`. GitHub Actions publishes the same Markdown files.
