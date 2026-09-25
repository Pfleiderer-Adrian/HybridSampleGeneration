# Fusion: poisson

Select with `config.fusion.set_backend('poisson')`.

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.fusion.parameters.guidance_mode` | `str` | `'source'` | Gradient guidance: 'source' preserves anomaly gradients; 'mixed' selects the stronger source or target gradient. |
| `config.fusion.parameters.solver_rtol` | `float` | `1e-05` | Relative convergence tolerance for the conjugate-gradient solver. |
| `config.fusion.parameters.solver_atol` | `float` | `0.0` | Absolute convergence tolerance for the conjugate-gradient solver. |
| `config.fusion.parameters.solver_max_iterations` | `int` | `2000` | Maximum conjugate-gradient iterations per image channel. |
| `config.fusion.parameters.clip_output` | `bool` | `True` | Clip results to recognized control-image ranges such as [0, 1] or [0, 255]. |
| `config.fusion.parameters.fusion_normalization_border_width` | `int \| None` | `2` | Context border width: None disables normalization; -1 uses the whole image. |
| `config.fusion.parameters.fusion_restore_anomaly_bg_relation` | `bool` | `True` | Restore the intensity relationship between anomaly and background. |
| `config.fusion.parameters.fusion_relation_mode` | `str` | `'delta'` | Intensity relationship method: 'delta' or 'ratio'. |
| `config.fusion.parameters.fusion_relation_norm_classes_separately` | `bool` | `False` | Normalize the intensity relationship separately for each anomaly class. |
| `config.fusion.parameters.fusion_relation_min_context_size` | `int` | `8` | Minimum number of context pixels for the intensity relationship. |
| `config.fusion.parameters.fusion_keep_bg` | `bool` | `False` | Keep the synthetic anomaly background. |
| `config.fusion.parameters.fusion_bg_value` | `float \| None` | `None` | Explicit background value; None uses automatic detection. |
| `config.fusion.parameters.fusion_relative_bg_threshold` | `float \| None` | `0.01` | Relative threshold for detecting background pixels. |
| `config.fusion.parameters.fusion_bg_exterior_only` | `bool` | `True` | Consider only background pixels connected to the exterior. |
