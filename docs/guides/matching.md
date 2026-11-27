# Hybrid planning

`generator.plan_hybrid_samples()` decides which synthetic anomalies are placed
into which target original and where. It writes `HybridSample` and `Placement`
records but no images; [fusion](fusion.md) materializes the plan afterwards.
Planning is reproducible through `config.study.seed`.

## Settings

- `hybrids_per_original`: requested number of hybrid variants per eligible
  target original.
- `anomalies_per_hybrid`: target placement count in each hybrid.
- `max_anomalies_per_hybrid_deviation`: deterministic random deviation around
  the placement count.
- `reuse_synthetic_across_hybrids`: whether the same synthetic ID may be used
  by more than one hybrid.
- `allow_sibling_variants_in_same_hybrid`: whether variants with the same real
  parent may occur together in one hybrid.
- `intensity_weight` and `gradient_weight`: weights for template matching.
- `distance_weight`: optional weight for proximity to the original anomaly's normalized
  source centroid (default `0.0`, which disables it). For `local`,
  `global` and `batchwise`, proximity is `1 - distance / maximum_distance`, using
  normalized window centers and the farthest valid window center as the maximum.
  It ranges from 0 to 1 and is averaged with the active intensity/gradient scores
  before selecting a position. Invalid correlation windows remain invalid.
  Fixed-placement modes do not use this weight.

The complete parameter list is available in the
[matching reference](../configuration/reference/matching.md).

## Routines

`config.matching.routine` selects how real anomaly ROIs are assigned to target
originals. The default is `batchwise`.

| Routine | Target originals | ROIs evaluated per target | Position | Selection order |
| --- | --- | --- | --- | --- |
| `local` | controls | only as many as needed, in sequence | template matching | sequential ROI order |
| `global` | controls | all | template matching | least-used ROI first, then best score |
| `batchwise` | controls | seeded subset of at most `batch_size` | template matching | least-used ROI first, then best score |
| `fixed_from_extraction_control_fusion` | controls | all, no matching | source position of the real anomaly | rotating ROI order |
| `fixed_from_extraction_anomaly_fusion` | anomalous originals | the original's own anomalies, no matching | source position of the real anomaly | extraction order |

`local`, `global`, `batchwise` and `fixed_from_extraction_control_fusion` target
originals with `has_anomaly=False`. `fixed_from_extraction_anomaly_fusion` targets
anomalous originals. Only real anomalies with synthetic variants are candidates.
A hybrid can contain fewer placements than requested; if no eligible placement
is found, that hybrid is omitted entirely.

### `local`

`local` assigns real anomaly ROIs sequentially across hybrids and controls.
It searches the full control image only for the next ROI with an eligible
synthetic variant, trying another ROI if the match is invalid or overlaps an
existing placement. Matching stops as soon as the requested placement count is
reached. Each hybrid tries at most one pass through the ROI pool; unused ROIs
are not loaded or matched. The ROI sequence restarts on each planning run.

### `global` and `batchwise`

`global` evaluates all real anomaly ROIs for each control and selects placements
by ascending accepted-placement count per real anomaly ROI, then descending
match score. Counts span all target images and hybrids in the current planning
run and update after each accepted placement. This favors unused sources before
reusing sources, even when their matching scores are lower. Invalid or
overlapping candidates do not increment counts. `batchwise` applies the same
ordering within a seeded subset of at most `batch_size` ROIs per control;
therefore equal usage across the entire pool is not guaranteed. Counts restart
on replanning.

### Fixed positions

`fixed_from_extraction_control_fusion` reuses the source centers of the real
anomalies on arbitrary controls. `fixed_from_extraction_anomaly_fusion` joins
originals and real anomalies by foreign key and places variants back at their
extraction positions. Neither routine computes a matching score.

## Variant selection

Within each real anomaly source, synthetic variants with the fewest accepted
placements in the current planning run are preferred. Ties are shuffled
reproducibly using the study seed, target original ID, source ID and hybrid
index. This also distributes variants when `hybrids_per_original=1`.
Counts restart on each planning run; rejected candidates do not count.
Reuse restrictions, sibling restrictions and overlap checks still apply.
Variant balancing operates within each real anomaly ROI; source balancing in
`global` and `batchwise` operates separately across those ROIs, not across
capture groups or defect classes.

## Template matching

Template matching uses Float64 correlation and rejects constant or numerically
flat templates and search windows. Window variance must exceed
`max(1e-16, 64 * eps64 * max(global centered energy, local second moment))`.
This numerical safeguard is not configurable and can exclude suitable placements
on very uniform surfaces; an invalid active intensity or gradient score also
invalidates the combined score.

## Matching cache

All three modes prepare control and ROI gradients on demand and reuse them
within the planning run. Pair results, including rejected pairs, are cached in
SQLite by matcher signature. Repeated planning with unchanged inputs and weights
reuses evaluated pairs, including when changing modes; new pairs are computed
only as needed.
