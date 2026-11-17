# Postprocessing

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.postprocessing.enabled` | `bool` | `False` | Require and run quality checks when explicitly called. |
| `config.postprocessing.action` | `str` | `'report'` | `report` records failures; `exclude` also omits failures in the MVTec downstream loader. Neither deletes artifacts. |
| `config.postprocessing.save_previews` | `bool` | `False` | Write 2D original overlays with foreground in blue and anomalies in red. |
| `config.postprocessing.foreground.method` | `str` | `'threshold'` | Object-mask estimator; currently only `threshold`. |
| `config.postprocessing.foreground.channel_reduction` | `str` | `'minimum'` | Minimum channel value (`minimum`) or RGB luminance (`grayscale`). |
| `config.postprocessing.foreground.comparison` | `str` | `'below'` | Threshold direction: `below` or `above`, strict comparison. |
| `config.postprocessing.foreground.threshold` | `float` | `240.0` | Threshold in original image units, range [0, 255]; no per-image normalization. |
| `config.postprocessing.foreground.components` | `str` | `'largest'` | Keep the largest component (`largest`) or `all`. |
| `config.postprocessing.foreground.min_component_area` | `int` | `1` | Minimum connected-component pixel/voxel count, before hole filling; positive. |
| `config.postprocessing.foreground.fill_holes` | `bool` | `True` | Fill enclosed holes after component selection. |
| `config.postprocessing.rules.minimum_object_overlap.enabled` | `bool` | `False` | Measure anomaly coverage by the estimated object mask. |
| `config.postprocessing.rules.minimum_object_overlap.threshold` | `float` | `1.0` | Minimum fraction of anomaly pixels inside the object; inclusive bound in [0, 1]. |
| `config.postprocessing.rules.mask_area_fraction.enabled` | `bool` | `False` | Enable whole-image anomaly area bounds. |
| `config.postprocessing.rules.mask_area_fraction.minimum` | `float` | `0.0` | Inclusive lower bound in [0, 1]. |
| `config.postprocessing.rules.mask_area_fraction.maximum` | `float` | `1.0` | Inclusive upper bound in [0, 1]. |
| `config.postprocessing.rules.reject_empty_mask` | `bool` | `True` | Mark empty materialized anomaly masks as failed. |

Run with `generator.postprocess_hybrid_samples()` after materialization; see the postprocessing guide. Configuration files without this section load with postprocessing disabled.

Any failed rule marks the entire hybrid as failed quality checking. Measurements refer to the union of the final materialized anomaly masks, not individual ROIs. Empty estimated object masks stop evaluation rather than silently rejecting all samples. Object overlap requires a healthy original partner. Threshold masks are heuristics, not semantic ground truth. Multiple defect types are not classified automatically; type-specific rules are not yet supported.

