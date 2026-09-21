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

## Updating the reference

The pages in `configuration/reference/` are generated. When parameters or explanations change, update the configuration classes and `scripts/generate_config_reference.py`:

```bash
python scripts/generate_config_reference.py
python scripts/generate_config_reference.py --check
```

The check reports new fields without explanations and outdated reference pages. To preview the site locally, install `mkdocs-material` and run `mkdocs serve`. GitHub Actions publishes the same Markdown files.
