# Postprocessing materialized hybrids

```python
generator.materialize_hybrid_samples()
summary = generator.postprocess_hybrid_samples()
```

The method reads the final anomaly masks and estimates object masks from the
unmodified original partner. Activate `config.postprocessing.enabled`; choose
`report` to inspect results or `exclude` to omit failed hybrids from MVTec
HybridPairs downstream training. A disabled call is a no-op.

Can's example uses the largest component with minimum RGB intensity below 240,
fills holes, and requires all anomaly pixels inside it. It excludes failed
hybrids. Wallplugs' example retains all grayscale components above 40 of at least
100 pixels, without hole filling, and checks 95% overlap in **report mode**.
These Wallplugs thresholds must be inspected before enabling exclusion; dark
object structures and legitimate geometric defects can fail the heuristic.

Each execution writes `postprocessing/<timestamp_id>/configuration.json`,
`summary.json`, `decisions.csv`, and optionally 2D previews. Current decisions are
stored separately in the additive `postprocessing_decisions` SQLite table.
Materialization status remains unchanged. Excluded images remain available in
the visualizer, whose hybrid details show the saved decision and reasons.
No physical deletion is performed. Historical reports remain when rerunning
checks with different settings, allowing previously failed hybrids to pass.

The MVTec downstream loader verifies the configuration fingerprint and original,
hybrid-image and mask file signatures (paths, size and nanosecond modification
time). Enabled postprocessing with absent/stale decisions stops loading with an
instruction to run the step again. Changing thresholds/action requires rerunning
checks. Plan replacement cascades decision deletion; rematerialization clears
current decisions before fusion starts. Hand-modified files are detected through
the file signatures; these signatures are not content hashes.

The first version checks each entire hybrid's final union mask. It supports
2D/3D grayscale and RGB inputs, with previews only for 2D images. Placement-type
classification, individual-placement rules, alternative learned object masks,
and deletion actions are not part of this version.
