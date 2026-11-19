# Fusion: classical

Select with `config.fusion.set_backend('classical')`.

| Parameter | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `config.fusion.parameters.max_alpha` | `float` | `0.8` | `[0.0, 1.0]` | Maximum anomaly mixing weight. |
| `config.fusion.parameters.segmentation_min_alpha` | `float` | `0.0` | `[0.0, 1.0]` | Minimum anomaly mixing weight required for a pixel to remain in the output segmentation mask; must not exceed max_alpha. |
| `config.fusion.parameters.image_harmonization` | `str \| None` | `None` | `'PCTNet'`, `'LBM'`, `None` | Optional final 2D harmonization step. Requires a separately installed libcom dependency and one or three image channels. |
| `config.fusion.parameters.sq` | `float` | `2` | `> 0.0` | Shape parameter for the spatial alpha mask. |
| `config.fusion.parameters.steepness_factor` | `float` | `3` | `> 0.0` | Steepness of the transition at the anomaly boundary. |
| `config.fusion.parameters.upsampling_factor` | `int` | `2` | `>= 1` | Factor for finer alpha mask computation. |
| `config.fusion.parameters.sobel_threshold` | `float` | `0.05` | `>= 0.0` | Threshold for Sobel edge detection. |
| `config.fusion.parameters.dilation_size` | `int` | `2` | `>= 0` | Width of mask dilation. |
| `config.fusion.parameters.shave_pixels` | `int` | `1` | `>= 0` | Number of boundary pixels removed. |
| `config.fusion.parameters.fusion_use_sobel_for_alpha_mask` | `bool` | `False` | — | Use Sobel edges for the alpha mask. |
| `config.fusion.parameters.fusion_variation` | `bool` | `True` | — | Enable random variation of fusion parameters. |
| `config.fusion.parameters.alpha_variation` | `float` | `0.05` | `>= 0.0` | Variation strength for max_alpha. |
| `config.fusion.parameters.sq_variation` | `float` | `0.1` | `>= 0.0` | Variation strength for sq. |
| `config.fusion.parameters.steepness_variation` | `float` | `0.1` | `>= 0.0` | Variation strength for steepness_factor. |
| `config.fusion.parameters.selected_confidence` | `str` | `'90%'` | `'68%'`, `'80%'`, `'90%'`, `'95%'`, `'99%'` | Confidence level. |
| `config.fusion.parameters.fusion_normalization_border_width` | `int \| None` | `2` | `>= -1` | Context border width for intensity normalization: None disables normalization, -1 uses the whole image, 0 uses the fallback context directly. |
| `config.fusion.parameters.fusion_restore_anomaly_bg_relation` | `bool` | `True` | — | Restore the intensity relationship between anomaly and background. |
| `config.fusion.parameters.fusion_relation_mode` | `str` | `'delta'` | `'delta'`, `'ratio'` | Intensity relationship method; use 'ratio' only if intensities are not close to 0 and have matching signs. |
| `config.fusion.parameters.fusion_relation_norm_classes_separately` | `bool` | `False` | — | Normalize the intensity relationship separately for each anomaly class. |
| `config.fusion.parameters.fusion_relation_min_context_size` | `int` | `8` | `>= 1` | Minimum number of context pixels for the intensity relationship. |
| `config.fusion.parameters.fusion_keep_bg` | `bool` | `False` | — | Keep the synthetic anomaly background. |
| `config.fusion.parameters.fusion_bg_value` | `float \| None` | `None` | — | Explicit background value; None uses automatic detection. |
| `config.fusion.parameters.fusion_relative_bg_threshold` | `float \| None` | `0.01` | `>= 0.0` | Relative threshold for detecting background pixels. |
| `config.fusion.parameters.fusion_bg_exterior_only` | `bool` | `True` | — | Consider only background pixels connected to the exterior. |
