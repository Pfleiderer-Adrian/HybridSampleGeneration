# Generator models

`config.model.name` holds the selected model name. Call `config.model.set_model(name)` to choose a registered variant; this resets `config.model.parameters` and the initial Optuna search to that variant's defaults. The reference shows values **per variant**, because 2D and 3D models can have different defaults.

| Architecture | 2D | 3D |
| --- | --- | --- |
| ResNet VAE | [VAE_ResNet_2D](reference/model-VAE_ResNet_2D.md) | [VAE_ResNet_3D](reference/model-VAE_ResNet_3D.md) |
| ConvNeXt VAE | [VAE_ConvNeXt_2D](reference/model-VAE_ConvNeXt_2D.md) | [VAE_ConvNeXt_3D](reference/model-VAE_ConvNeXt_3D.md) |
| Mask-conditioned ConvNeXt VAE | [cVAE_ConvNeXt_2D](reference/model-cVAE_ConvNeXt_2D.md) | [cVAE_ConvNeXt_3D](reference/model-cVAE_ConvNeXt_3D.md) |

## Fixed values and Optuna search

`config.model.parameters.<name>` sets a concrete value. You can assign an Optuna distribution to `config.model.search.<name>`. Without a search distribution, the concrete value stays fixed across trials. The "Default search" column shows which parameters already have a distribution for the selected model.

```python
from hybrid_sample_generator.generation.model_settings import Choice, FloatRange, IntRange

config.model.set_model("VAE_ConvNeXt_3D")
config.model.parameters.dropout = 0.05
config.model.search.clear()
config.model.search.n_res_blocks = IntRange(5, 6)
config.model.search.dropout = FloatRange(0.0, 0.2)
config.model.search.recon_loss = Choice(("mse", "smoothl1"))
```

- `IntRange(low, high, step=1, log=False)` selects integers.
- `FloatRange(low, high, step=None, log=False)` selects floating-point values.
- `Choice(values)` selects from a non-empty list of values.

A distribution must match the type of its concrete parameter. The model's spatial dimensions must also match `config.extraction.anomaly_size`.
