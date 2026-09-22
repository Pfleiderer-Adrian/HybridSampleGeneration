# Generation

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.generation.sampling_mode` | `str` | `'posterior'` | Latent sampling source: 'posterior' or 'prior'. |
| `config.generation.variation_strength` | `float` | `0.5` | Strength of random variation during generation; non-negative. |
| `config.generation.clamp_output` | `bool` | `False` | Clamp generated image values to [0, 1]. |
| `config.generation.background_threshold` | `float` | `0.01` | Relative threshold for deriving a mask from generated images. |
| `config.generation.variants_per_real_anomaly` | `int` | `3` | Number of synthetic variants per real anomaly; positive. |
| `config.generation.feedback.enabled` | `bool` | `False` | Enable image similarity feedback and repeated generation attempts. |
| `config.generation.feedback.similarity_threshold` | `float` | `0.8` | Minimum similarity for accepting a variant; range [0, 1]. |
| `config.generation.feedback.threshold_relaxation_factor` | `float` | `0.9` | Factor for relaxing the threshold; range (0, 1]. |
| `config.generation.feedback.max_attempts` | `int` | `100` | Maximum number of generation attempts; positive. |

