# Matching

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.matching.routine` | `str` | `'fixed_from_extraction_anomaly_fusion'` | Candidate selection and placement method; see the matching guide. |
| `config.matching.hybrids_per_original` | `int` | `1` | Number of hybrid images per eligible original; positive. |
| `config.matching.anomalies_per_hybrid` | `int` | `1` | Target number of placed anomalies per hybrid; positive. |
| `config.matching.max_anomalies_per_hybrid_deviation` | `int` | `0` | Allowed random deviation from the target placement count; non-negative. |
| `config.matching.reuse_synthetic_across_hybrids` | `bool` | `True` | Allow a synthetic variant to appear in multiple hybrids. |
| `config.matching.allow_sibling_variants_in_same_hybrid` | `bool` | `False` | Allow variants of one real anomaly in the same hybrid. |
| `config.matching.batch_size` | `int` | `64` | Batch size for candidate scoring; positive. |
| `config.matching.intensity_weight` | `float` | `1.0` | Weight of intensity similarity during matching. |
| `config.matching.gradient_weight` | `float` | `2.0` | Weight of gradient similarity during matching. |

Allowed `matching.routine` values: `local`, `global`, `batchwise`, `fixed_from_extraction_anomaly_fusion`, and `fixed_from_extraction_control_fusion`.

