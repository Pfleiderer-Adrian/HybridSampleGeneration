# Tests

```bash
python -m unittest discover -s tests -v
```

The integration tests cover the one-time mixed dataset ingest, multiple real
components, multiple synthetic and hybrid variants, normalized multi-placement
records, unique artifacts, foreign-key traversal, 2D/3D coordinates,
materialization, FK-based evaluation and cached full-image `local` matching.
