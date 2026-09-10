# Refactoring baseline

Recorded on 2026-09-10 before moving modules into the new package structure.

## Repository state

- Branch: `refactor_datastructure`
- Commit: `3b8fc2a`
- Working tree before the baseline checks: clean
- Existing source roots: `synthesizer`, `data_handler`, `generation_models`,
  and `fusion_backend`

No commit was created as part of this baseline.

## Verification

The project uses Python's built-in `unittest` runner. `pytest` is not installed in
the project environment and is not required by the documented test workflow.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result: **45 tests passed**.

Additional checks:

```bash
.venv/bin/python -m compileall -q \
  synthesizer data_handler generation_models fusion_backend use_cases tests
.venv/bin/python -m pip check
```

Results:

- All Python modules compiled successfully.
- No broken requirements were found.

## Compatibility boundary

The following imports are used by the README and example pipelines and therefore
form the de facto public API during the migration:

```python
from synthesizer.Configuration import Configuration, load_config_file
from synthesizer.Evaluation import evaluate_study
from synthesizer.HybridDataGenerator import HybridDataGenerator
from synthesizer.InputSample import InputSample
from data_handler.Visualizer import run_hybrid_visualizer
```

These paths must continue to work through compatibility modules until the final
migration phase updates the documentation and examples. Internal imports used by
the test suite are also kept working until their corresponding feature package is
migrated.

## Acceptance criterion for each migration step

After each move or split:

1. The full 45-test baseline passes.
2. The compilation check passes.
3. `pip check` reports no broken requirements.
4. The five documented public imports above remain importable, unless a later
   migration step explicitly replaces and documents them.
5. Unrelated working-tree changes are not modified.
