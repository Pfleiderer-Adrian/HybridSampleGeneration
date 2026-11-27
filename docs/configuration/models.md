# Generator models

`config.model.name` holds the selected model name. Call `config.model.set_model(name)` to choose a registered variant; this resets `config.model.parameters` and the initial Optuna search to that variant's defaults. The reference shows values **per variant**, because 2D and 3D models can have different defaults.

| Architecture | 2D | 3D |
| --- | --- | --- |
| ResNet VAE | [VAE_ResNet_2D](reference/model-VAE_ResNet_2D.md) | [VAE_ResNet_3D](reference/model-VAE_ResNet_3D.md) |
| ConvNeXt VAE | [VAE_ConvNeXt_2D](reference/model-VAE_ConvNeXt_2D.md) | [VAE_ConvNeXt_3D](reference/model-VAE_ConvNeXt_3D.md) |
| Mask-conditioned ConvNeXt VAE | [cVAE_ConvNeXt_2D](reference/model-cVAE_ConvNeXt_2D.md) | [cVAE_ConvNeXt_3D](reference/model-cVAE_ConvNeXt_3D.md) |
| Paired source-to-target ConvNeXt VAE | [paired_cVAE_ConvNeXt_2D](reference/model-paired_cVAE_ConvNeXt_2D.md) | [paired_cVAE_ConvNeXt_3D](reference/model-paired_cVAE_ConvNeXt_3D.md) |

## Choosing a model

Every architecture is registered once for 2D and once for 3D data. Both
entries share one dimension-independent implementation and differ only in
their defaults and default search space.

| Architecture | Mask conditioning | Trained on | Use it when |
| --- | --- | --- | --- |
| `VAE_ResNet` | none | reconstruction | a compact baseline is sufficient |
| `VAE_ConvNeXt` | none | reconstruction | variants should keep the shape of their parent |
| `cVAE_ConvNeXt` (default) | SPADE blocks in the decoder | reconstruction | variants should also vary in shape |
| `paired_cVAE_ConvNeXt` | SPADE blocks in the decoder, no skip connections | source-to-target pairs | the model should learn the change of shape explicitly |

The unconditioned models generate an image and derive its mask from the image.
The mask-conditioned models generate the image for a given target mask, which
[generation](../guides/generation.md#target-masks) obtains by transforming the
mask of the real anomaly.

`cVAE_ConvNeXt` is trained to reconstruct each cutout for its own mask and
only meets changed masks during generation. `paired_cVAE_ConvNeXt` is trained
on pairs instead: the encoder receives a real cutout with its mask, and the
decoder has to produce a jointly transformed version of that cutout for the
transformed mask. The pairs are created on the fly with the transforms in
`config.augmentation.mask_transforms`, so these settings affect training as
well as generation. `identity_pair_probability` (default `0.2`) is the share
of training pairs whose target equals the source. Validation pairs are fixed
by the study seed.

## Model setup

`Configuration(study_name, save_path=None, *, study_folder=None)` only accepts
study identity and storage location. New configurations default to
`cVAE_ConvNeXt_2D` and `config.extraction.anomaly_size = (3, 64, 64)`.
Use registered names such as `VAE_ResNet_2D`, `VAE_ConvNeXt_3D` or
`paired_cVAE_ConvNeXt_2D` with `config.model.set_model(name)`.
Set the anomaly size and select the model before customizing its parameter space:

```python
from hybrid_sample_generator.configuration import Configuration
from hybrid_sample_generator.generation.model_settings import Choice, FloatRange, IntRange

config = Configuration("volume-study")
config.extraction.anomaly_size = (1, 32, 64, 64)
config.model.set_model("VAE_ConvNeXt_3D")
config.model.parameters.z_channels = 32
config.model.search.clear()
config.model.search.n_res_blocks = IntRange(4, 6)
config.model.search.dropout = FloatRange(0.0, 0.2)
config.model.search.recon_loss = Choice(("mse", "smoothl1"))
```

Parameters absent from `config.model.search` remain fixed for every trial. Use `clear()` to make every
parameter fixed and, for example, `del config.model.search.dropout` to remove
one distribution. Each model module owns a concrete `Config` dataclass plus
factories for its dimension-specific defaults and search space. `set_model`
uses those factories to initialize fresh model parameters and a validated
`SearchSpace` bound to them. Runtime values such as the input channel count and
number of anomaly classes are derived from the extracted data and are not part
of the saved model parameters. Model dimensionality and the full configuration
are validated when constructing `HybridDataGenerator`, serializing, or
explicitly calling `config.validate()`.

## Mask-balanced reconstruction

All VAE variants receive the extracted `ori_mask` during training. Foreground
and background reconstruction errors are averaged separately per sample and
combined with `foreground_weight` and `background_weight`. The weights are
relative and must be non-negative; at least one must be positive. ResNet and
regular ConvNeXt use the mask only for the loss and remain unconditioned. The
paired models weight the loss with the target mask. The `selection` metric,
which [training](../guides/training.md) monitors by default, is this balanced
loss.

## Fixed values and Optuna search

`config.model.parameters.<name>` sets a concrete value. You can assign an Optuna distribution to `config.model.search.<name>`. Without a search distribution, the concrete value stays fixed across trials. The "Default search" column shows which parameters already have a distribution for the selected model.

- `IntRange(low, high, step=1, log=False)` selects integers.
- `FloatRange(low, high, step=None, log=False)` selects floating-point values.
- `Choice(values)` selects from a non-empty list of values.

A distribution must match the type of its concrete parameter. The model's spatial dimensions must also match `config.extraction.anomaly_size`.
