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
