# Synthetic variants

`generator.generate_synthetic_anomalies()` creates
`config.generation.variants_per_real_anomaly` synthetic variants for every real
anomaly. Each variant is a `SyntheticAnomaly` record with its own deterministic
ID, variant index, seed, image and target mask. A generator must be loaded
first, either by [`train_generator()`](training.md) or by `load_generator()`.

```python
config.generation.variants_per_real_anomaly = 5
config.generation.sampling_mode = "posterior"
config.generation.variation_strength = 0.5

generator.load_generator()
generator.generate_synthetic_anomalies()
```

The seed of a variant is derived from `config.study.seed`, the real anomaly ID
and the variant index. Repeating generation with unchanged inputs therefore
reproduces the same variants.

## Sampling

`config.generation.sampling_mode` selects where the latent vector comes from:

- `posterior` (default) encodes the real anomaly and samples around it:
  `z = mu + variation_strength * sigma * eps`. A `variation_strength` of `0`
  reproduces the reconstruction of the real anomaly, values around `0.2`–`0.5`
  give small variations and values of `1.0` or more can drift far from the
  source.
- `prior` ignores the real image and samples `z` from a standard normal
  distribution scaled by `variation_strength`.

`clamp_output` clamps the generated image to `[0, 1]`; leave it disabled when
[extraction](extraction.md) normalizes cutouts with z-scores.

## Target masks

How the mask of a variant is obtained depends on the model:

- **Unconditioned models** (`VAE_ResNet`, `VAE_ConvNeXt`) generate an image
  only. The mask is derived from that image: pixels above
  `config.generation.background_threshold`, relative to the image's intensity
  range, belong to the anomaly.
- **Mask-conditioned models** (`cVAE_ConvNeXt`, `paired_cVAE_ConvNeXt`) first
  transform the mask of the real anomaly into a new target mask and then
  generate the image for it. The variant therefore differs from its parent in
  shape as well as in texture.

The mask transforms are configured in `config.augmentation.mask_transforms`.
Global transforms (`zoom`, `stretch`, `rotate`, `elastic`) act on the whole
mask; local transforms (`local_dilate`, `local_stretch`, `local_rotate`,
`local_elastic`) act on each anomaly class separately:

```python
masks = config.augmentation.mask_transforms
masks.setGlobalParam("rotate", probability=0.5, max_rotation=15.0)
masks.setGlobalParam("elastic", probability=0.0)
masks.setAllClassParams("local_dilate", probability=0.3, max_iterations=3)
masks.setClassParam(2, "local_stretch", probability=1.0, max_stretch=1.1)
```

`use_mask_transform = False` disables all transforms, so each variant keeps the
mask of its parent. The default probabilities and parameters are listed in the
[augmentation reference](../configuration/reference/augmentation.md#transform-defaults).

For `cVAE_ConvNeXt` in posterior mode,
`config.generation.posterior_skip_source` chooses the image that feeds the
encoder skip connections: `transformed` (default) warps the real image with the
same transform as the mask, `original` uses the unchanged image. The setting
only has an effect when skip connections are active (`skip_alpha` above `0`).

## Feedback generation

Feedback generation rejects variants that differ too much from their real
parent. It is disabled by default:

```python
config.generation.feedback.enabled = True
config.generation.feedback.similarity_threshold = 0.8
config.generation.feedback.max_attempts = 20
```

For every variant, generation is repeated until the structural similarity
(a global SSIM score, clipped to `[0, 1]`) between the generated and the real cutout reaches
`similarity_threshold` or `max_attempts` is exhausted. If no attempt reaches
the threshold, the attempt with the highest similarity is kept, so every
variant is always produced. After every 100 unsuccessful attempts, the
threshold is multiplied by `threshold_relaxation_factor`.

Feedback increases generation time by up to `max_attempts` per variant and
biases the variants towards their parents. It is most useful with a high
`variation_strength` or prior sampling, where single samples can fail. With a
mask-conditioned model the comparison also penalizes the intended change of
shape, so use a lower threshold there.

## Repeating generation

Each call replaces all synthetic variants and removes the hybrids and
placements that depended on them; see
[continuing a study](continuing-studies.md).

The complete parameter list is available in the
[generation reference](../configuration/reference/generation.md).
