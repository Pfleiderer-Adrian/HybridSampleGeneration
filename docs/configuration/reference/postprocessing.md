# Postprocessing

`config.postprocessing` is stored with all other sections in the study's
`configuration.json`. Existing schema-10 files without the section load with
postprocessing disabled. This is an optional extension of schema 10; other
schema versions remain unsupported.

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `False` | Require and run quality checks when explicitly called. |
| `action` | `"report"` | `report` records failures; `exclude` also omits failures in the MVTec downstream loader. Neither deletes artifacts. |
| `save_previews` | `False` | Write 2D original overlays with foreground in blue and anomalies in red. |
| `foreground.method` | `"threshold"` | Currently supported object-mask estimator. |
| `foreground.channel_reduction` | `"minimum"` | Minimum channel value, or `grayscale` (RGB luminance). |
| `foreground.comparison` | `"below"` | Threshold direction: `below` or `above`, strict comparison. |
| `foreground.threshold` | `240` | Threshold in original image units, expected 0–255; no per-image normalization. |
| `foreground.components` | `"largest"` | Keep largest component or `all`. |
| `foreground.min_component_area` | `1` | Minimum connected-component pixel/voxel count, before hole filling. |
| `foreground.fill_holes` | `True` | Fill enclosed holes after component selection. |
| `rules.reject_empty_mask` | `True` | Mark empty materialized anomaly masks as failed. |
| `rules.minimum_object_overlap.enabled` | `False` | Measure anomaly coverage by estimated object mask. |
| `rules.minimum_object_overlap.threshold` | `1.0` | Minimum fraction of anomaly pixels inside object; inclusive bound. |
| `rules.minimum_object_overlap.exempt_source_groups` | `[]` | Source capture groups exempt from object-overlap checking; all other rules remain active. Requires masks saved during materialization. |
| `rules.mask_area_fraction.enabled` | `False` | Enable whole-image anomaly area bounds. |
| `rules.mask_area_fraction.minimum` | `0.0` | Inclusive lower bound. |
| `rules.mask_area_fraction.maximum` | `1.0` | Inclusive upper bound. |

Any failed rule marks the entire hybrid as failed quality checking. Measurements
refer to the union of the final materialized anomaly masks. With source-group
exemptions, overlap is measured only on the union of nonexempt placement masks;
mask-area and empty-mask rules still use the entire final mask.
Empty estimated object masks stop evaluation rather than silently rejecting all
samples. Object overlap requires a healthy original partner. Threshold masks are
heuristics, not semantic ground truth. Multiple defect types are not classified
automatically; automatic type-specific rules are not yet supported.
