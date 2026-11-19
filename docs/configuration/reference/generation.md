# Generation

| Parameter | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `config.generation.sampling_mode` | `str` | `'posterior'` | `'posterior'`, `'prior'` | Latent sampling source. |
| `config.generation.posterior_skip_source` | `str` | `'transformed'` | `'original'`, `'transformed'` | Encoder skip source for models supporting transformed posterior skips: 'original' uses the input image; 'transformed' uses an image paired with the generated target mask. Ignored by other models and in prior mode. |
| `config.generation.variation_strength` | `float` | `0.5` | `>= 0.0` | Strength of random variation during generation. |
| `config.generation.clamp_output` | `bool` | `False` | — | Clamp generated image values to [0, 1]. |
| `config.generation.background_threshold` | `float` | `0.01` | `>= 0.0` | Relative threshold for deriving a mask from generated images. |
| `config.generation.variants_per_real_anomaly` | `int` | `3` | `>= 1` | Number of synthetic variants per real anomaly. |
| `config.generation.feedback.enabled` | `bool` | `False` | — | Enable image similarity feedback and repeated generation attempts. |
| `config.generation.feedback.similarity_threshold` | `float` | `0.8` | `[0.0, 1.0]` | Minimum similarity for accepting a variant. |
| `config.generation.feedback.threshold_relaxation_factor` | `float` | `0.9` | `(0.0, 1.0]` | Factor for relaxing the threshold. |
| `config.generation.feedback.max_attempts` | `int` | `100` | `>= 1` | Maximum number of generation attempts. |

