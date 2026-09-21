# VAE_ResNet_3D

Select with `config.model.set_model('VAE_ResNet_3D')`. The table shows the actual defaults for this model variant. A dash in Default search means the parameter stays fixed unless you assign a compatible distribution through `config.model.search`.

| Parameter | Type | Default | Meaning / values | Default search |
| --- | --- | --- | --- | --- |
| `config.model.parameters.n_res_blocks` | `int` | `4` | Number of residual blocks per level. | `IntRange(low=4, high=5, step=1, log=False)` |
| `config.model.parameters.n_levels` | `int` | `4` | Number of encoder and decoder levels. | — |
| `config.model.parameters.z_channels` | `int` | `64` | Channel count in the spatial bottleneck. | `Choice(values=(64, 128))` |
| `config.model.parameters.bottleneck_dim` | `int` | `128` | Dimension of the latent vector. | `Choice(values=(128, 256))` |
| `config.model.parameters.use_multires_skips` | `bool` | `True` | Use encoder features from multiple resolutions as skip connections. | — |
| `config.model.parameters.recon_weight` | `float` | `100.0` | Weight of the reconstruction loss. | — |
| `config.model.parameters.beta_kl_start` | `float` | `0.0` | Initial weight of the KL loss. | — |
| `config.model.parameters.beta_kl_max` | `float` | `0.05` | Maximum weight of the KL loss. | — |
| `config.model.parameters.beta_kl_warmup_start` | `int` | `20` | Epoch at which the KL weight begins to increase. | — |
| `config.model.parameters.beta_kl_warmup_epochs` | `int` | `30` | Number of epochs needed to reach beta_kl_max. | — |
| `config.model.parameters.free_bits` | `float` | `0.0` | KL free-bits allowance for latent dimensions. | — |
| `config.model.parameters.recon_loss` | `str` | `'mse'` | Reconstruction loss, such as 'mse' or 'smoothl1'. | — |
| `config.model.parameters.recon_smoothl1_beta` | `float` | `1.0` | Transition point of the Smooth L1 loss. | — |
| `config.model.parameters.use_transpose_conv` | `bool` | `False` | Use transposed convolutions for upsampling. | — |
| `config.model.parameters.fg_weight` | `float` | `1.0` | Additional weight for foreground pixels in the reconstruction loss. | — |
| `config.model.parameters.fg_threshold` | `float` | `0.0` | Threshold for selecting foreground pixels. | — |
