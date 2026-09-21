# Fusion: classical

Select with `config.fusion.set_backend('classical')`.

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.fusion.parameters.max_alpha` | `float` | `0.8` | Maximum anomaly mixing weight; range [0, 1]. |
| `config.fusion.parameters.sq` | `float` | `2` | Shape parameter for the spatial alpha mask. |
| `config.fusion.parameters.steepness_factor` | `float` | `3` | Steepness of the transition at the anomaly boundary. |
| `config.fusion.parameters.upsampling_factor` | `int` | `2` | Factor for finer alpha mask computation. |
| `config.fusion.parameters.sobel_threshold` | `float` | `0.05` | Threshold for Sobel edge detection. |
| `config.fusion.parameters.dilation_size` | `int` | `2` | Width of mask dilation. |
| `config.fusion.parameters.shave_pixels` | `int` | `1` | Number of boundary pixels removed. |
| `config.fusion.parameters.fusion_use_sobel_for_alpha_mask` | `bool` | `False` | Use Sobel edges for the alpha mask. |
| `config.fusion.parameters.fusion_variation` | `bool` | `True` | Enable random variation of fusion parameters. |
| `config.fusion.parameters.alpha_variation` | `float` | `0.05` | Variation strength for max_alpha. |
| `config.fusion.parameters.sq_variation` | `float` | `0.1` | Variation strength for sq. |
| `config.fusion.parameters.steepness_variation` | `float` | `0.1` | Variation strength for steepness_factor. |
| `config.fusion.parameters.selected_confidence` | `str` | `'90%'` | Confidence level: 68%, 80%, 90%, 95%, or 99%. |
| `config.fusion.parameters.fusion_normalization_border_width` | `int \| None` | `2` | Context border width: None disables normalization; -1 uses the whole image. |
| `config.fusion.parameters.fusion_restore_anomaly_bg_relation` | `bool` | `True` | Restore the intensity relationship between anomaly and background. |
| `config.fusion.parameters.fusion_relation_mode` | `str` | `'delta'` | Intensity relationship method: 'delta' or 'ratio'. |
| `config.fusion.parameters.fusion_relation_norm_classes_separately` | `bool` | `False` | Normalize the intensity relationship separately for each anomaly class. |
| `config.fusion.parameters.fusion_relation_min_context_size` | `int` | `8` | Minimum number of context pixels for the intensity relationship. |
| `config.fusion.parameters.fusion_keep_bg` | `bool` | `False` | Keep the synthetic anomaly background. |
| `config.fusion.parameters.fusion_bg_value` | `float \| None` | `None` | Explicit background value; None uses automatic detection. |
| `config.fusion.parameters.fusion_relative_bg_threshold` | `float \| None` | `0.01` | Relative threshold for detecting background pixels. |
| `config.fusion.parameters.fusion_bg_exterior_only` | `bool` | `True` | Consider only background pixels connected to the exterior. |
