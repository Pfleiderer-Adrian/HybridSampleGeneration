# Augmentation

| Parameter | Type | Default | Meaning / values |
| --- | --- | --- | --- |
| `config.augmentation.mask_transforms.use_mask_transform` | `bool` | `True` | Enable the default mask transform probabilities. |
| `config.augmentation.mask_transforms.mask_transform_probs` | `dict[int \| str, Any]` | `{}` | Probabilities for global, local, and class-specific transforms. |
| `config.augmentation.mask_transforms.mask_transform_params` | `dict[int \| str, dict[str, Any]]` | `{}` | Parameter ranges for individual transforms. |
| `config.augmentation.mask_transforms.priorities` | `list[int] \| tuple[int, ...] \| None` | `None` | Class priority when transformed masks overlap. |
| `config.augmentation.mask_transforms.local_as_global` | `bool` | `False` | Apply local transforms jointly to all anomaly classes. |
| `config.augmentation.mask_transforms.padding_factor` | `int` | `2` | Scale factor for the temporary transform canvas. |
| `config.augmentation.random_offset_enabled` | `bool` | `True` | Enable random offsets during training. |
| `config.augmentation.random_offset_max_fraction` | `float` | `1.0` | Maximum offset as a fraction of available space; range [0, 1]. |
| `config.augmentation.random_offset_foreground_threshold` | `float` | `0.001` | Foreground threshold used for training offsets. |

Masks use nearest-neighbor interpolation; jointly transformed images use linear interpolation. `mask_transform_probs` and `mask_transform_params` accept global or local transform names and class IDs. With `use_mask_transform=True`, the default anomaly size `(3, 64, 64)` gives these effective values; global elastic parameters change with `anomaly_size`.

## Transform defaults

| Transform | Default probability | Effective default parameters |
| --- | --- | --- |
| `zoom` | `1.0` | `{'min_zoom': 0.9, 'max_zoom': 0.9}` |
| `stretch` | `1.0` | `{'min_stretch': 1.0, 'max_stretch': 1.2}` |
| `rotate` | `0.0` | `{'max_rotation': 5.0}` |
| `elastic` | `1.0` | `{'sigma': (13, 13), 'magnitude': (13, 13)}` |
| `local_dilate` | `0.0` | `{'min_iterations': 0, 'max_iterations': 2}` |
| `local_stretch` | `0.0` | `{'min_stretch': 0.95, 'max_stretch': 1.05}` |
| `local_rotate` | `0.0` | `{'max_rotation': 5.0}` |
| `local_elastic` | `0.0` | `{'sigma': 30, 'magnitude': 20}` |

