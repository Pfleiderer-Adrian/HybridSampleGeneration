# Matching

| Parameter | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `config.matching.routine` | `str` | `'batchwise'` | `'batchwise'`, `'fixed_from_extraction_anomaly_fusion'`, `'fixed_from_extraction_control_fusion'`, `'global'`, `'local'` | Candidate selection and placement method; see the matching guide. |
| `config.matching.hybrids_per_original` | `int` | `1` | `>= 1` | Number of hybrid images per eligible original. |
| `config.matching.anomalies_per_hybrid` | `int` | `1` | `>= 1` | Target number of placed anomalies per hybrid. |
| `config.matching.max_anomalies_per_hybrid_deviation` | `int` | `0` | `>= 0` | Allowed random deviation from the target placement count. |
| `config.matching.reuse_synthetic_across_hybrids` | `bool` | `True` | — | Allow a synthetic variant to appear in multiple hybrids. |
| `config.matching.allow_sibling_variants_in_same_hybrid` | `bool` | `False` | — | Allow variants of one real anomaly in the same hybrid. |
| `config.matching.batch_size` | `int` | `32` | `>= 1` | Batch size for candidate scoring. |
| `config.matching.intensity_weight` | `float` | `1.0` | — | Weight of intensity similarity during matching. |
| `config.matching.gradient_weight` | `float` | `2.0` | — | Weight of gradient similarity during matching. |
| `config.matching.distance_weight` | `float` | `0.0` | `>= 0.0` | Weight of proximity to the original anomaly's normalized source position. Zero preserves intensity/gradient-only matching. |

