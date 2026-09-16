# MVTec AD 2 example

Run from the repository root. Edit `settings.py` for dataset/output paths,
categories and the experiment split. Edit `presets.py` for model settings:

1. A new `Configuration(study_name, save_path=...)` gets its anomaly size and
   model selection from the category preset via `config.extraction.anomaly_size`
   and `config.model.set_model(...)`.
2. `apply_global_defaults(config)` sets shared generator, fusion and DRAEM settings.
3. `configure_can(config)`, `configure_fabric(config)`, etc. override category differences.
4. Local changes to a prepared/opened study take precedence for that execution.

`CATEGORY_PRESETS` contains model, anomaly shape and the optional override function
for every category. Categories without an override inherit all shared settings.
`configuration.py` defines the study/experiment types; `downstream/configuration.py`
defines DRAEM types and saved-run serialization.

## Run an experiment

```bash
python -m examples.mvtec_ad2.main
```

`main.py` runs `EXPERIMENT` from `settings.py` with `FULL_EXPERIMENT`. Select
`GENERATE_HYBRIDS` or an explicit ordered step tuple to run only part of it.
There is no `MODE` switch. The recipes are:

| Recipe | Steps |
| --- | --- |
| `GENERATE_HYBRIDS` | ingest, extract, train_generator, generate_synthetic, plan, materialize |
| `TRAIN_DOWNSTREAM` | train_downstream, evaluate_downstream |
| `FULL_EXPERIMENT` | GENERATE_HYBRIDS, export, TRAIN_DOWNSTREAM |

```python
from pathlib import Path
from examples.mvtec_ad2.configuration import Experiment, SplitConfiguration
from examples.mvtec_ad2.pipeline import GENERATE_HYBRIDS, run_new_experiment

experiment = Experiment(
    dataset_root=Path('/data/mvtec_ad_2'),
    output_root=Path('/results/my_experiment'),
    categories=('can', 'fabric'),
    split=SplitConfiguration(test_enabled=False, validation_fraction=0.2),
)
results = run_new_experiment(experiment, steps=GENERATE_HYBRIDS)
```

A new experiment uses current presets and establishes a saved split before
training. Existing study paths are rejected; use a fresh output root or continue
an existing study. This is a custom anomaly-supervised protocol, not the official
unsupervised MVTec benchmark. See [split and downstream details](downstream/README.md).

## Continue or review a study

```python
from examples.mvtec_ad2.pipeline import run_existing_studies
from examples.mvtec_ad2.studies import find_studies

folders = find_studies('/results/my_experiment', categories=('can',))
results = run_existing_studies(folders, steps=('plan', 'materialize', 'export'))
```

Existing studies always load their saved configuration and split. Downstream
training no longer replaces saved settings with current presets automatically.
For an explicit replacement or local customization:

```python
from examples.mvtec_ad2.studies import open_study
from examples.mvtec_ad2.presets import apply_downstream_preset
from examples.mvtec_ad2.pipeline import TRAIN_DOWNSTREAM, run_study

study = open_study('/path/to/study')
apply_downstream_preset(study.config, study.category)
study.config.downstream.training.epochs = 20
result = run_study(study, steps=TRAIN_DOWNSTREAM)
```

Use `prepare_studies(root, categories, save_path=..., splits=...)` from `studies.py`
when customizing each new study before calling `run_study`.

Review is independent of processing and supports saved studies without a split:

```python
from examples.mvtec_ad2.review import review_studies

review_studies(['/path/to/study'], actions=('evaluate_generator', 'visualize'))
```

Opening and reviewing do not rewrite the study configuration. A downstream run
stores its own configuration and split snapshot; later evaluation uses that
snapshot. `run_study` defaults to generation steps; both batch APIs accept explicit
steps, with `run_new_experiment` defaulting to `FULL_EXPERIMENT`.

## CLI

```bash
python -m examples.mvtec_ad2 generate --categories can fabric
python -m examples.mvtec_ad2 generate --categories can --steps ingest extract
python -m examples.mvtec_ad2 continue --study-folder /path/to/study --steps plan materialize export
python -m examples.mvtec_ad2 continue --study-folder /path/to/study --steps train_downstream evaluate_downstream
python -m examples.mvtec_ad2 review --study-folder /path/to/study
python -m examples.mvtec_ad2 evaluate --study-folder /path/to/study --run-id RUN_ID
```

`generate` accepts `--dataset-root` and `--output-root`; defaults and split settings
come from `EXPERIMENT`. `continue` requires explicit steps. `evaluate` requires one
study and its concrete downstream run ID. Use `--help` on any command.

The previous `runner.py` APIs and `downstream.main` entry point are replaced by
`studies.py`, `pipeline.py`, `review.py` and the CLI above. Saved study/run JSON
formats are unchanged.

## Tests

```bash
python -m unittest discover -s tests/example_cases/mvtec_ad2 -t . -q
```

Step recipes are defined centrally in `steps.py` and also available from
`pipeline.py`. Dataset discovery and study preparation take an explicit dataset
root; only the entry points use `settings.EXPERIMENT` defaults. Category aliases
are normalized before preparation. Empty, unknown or repeated category selections
are rejected, and every category directory and target configuration is checked
before writing the first split. This is not an atomic batch: later split-data or
execution errors can still leave earlier prepared studies in place. Repeated
study paths (including equivalent resolved paths) are rejected before continuing
or reviewing a batch.
