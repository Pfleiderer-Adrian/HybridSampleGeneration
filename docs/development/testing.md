# Tests

The tests use `unittest` and need no dataset or GPU. Run all of them from the
repository root:

```bash
python -m unittest discover -s tests -v
```

Run the tests of one area or one file:

```bash
python -m unittest discover -s tests/matching -t . -v
python -m unittest tests.fusion.test_service -v
```

The `tests/` folder mirrors the package: `configuration`, `datasets`, `domain`,
`evaluation`, `extraction`, `fusion`, `generation`, `imaging`, `matching`,
`persistence`, `pipeline`, `postprocessing` and `visualization`.
`tests/example_cases` covers the bundled examples, including the MVTec AD 2
benchmark code.

The integration tests in `tests/pipeline` run the stages on small generated
data. They cover the one-time mixed dataset ingest, multiple real components,
multiple synthetic and hybrid variants, multi-placement records, unique
artifacts, foreign-key traversal, 2D/3D coordinates, materialization,
evaluation and cached full-image `local` matching.

## Continuous integration

The `Tests` workflow runs on every pull request and on `main` with Python 3.12
and the CPU build of PyTorch. It first checks that the generated
[configuration reference](documentation.md#configuration-reference) is current
and then runs all tests.

## Coverage

The workflow measures line and branch coverage of the package with
[coverage.py](https://coverage.readthedocs.io/). The study browser is excluded
because it needs a display; the settings are in `[tool.coverage.run]` of
`pyproject.toml`. To measure locally:

```bash
python -m pip install coverage
python -m coverage run -m unittest discover -s tests
python -m coverage report
```

The workflow fails when the coverage is below 90 %. The limit is
`MINIMUM_COVERAGE` in `.github/workflows/tests.yml`; check it locally with
`python -m coverage report --fail-under=90`.

The coverage badge in the README shows the value of the last run on `main`.
After such a run, the workflow writes the percentage to `coverage.json` on the
`badges` branch, and the badge reads that file.
