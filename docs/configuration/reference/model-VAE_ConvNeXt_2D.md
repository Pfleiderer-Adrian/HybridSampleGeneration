# VAE_ConvNeXt_2D

Select with `config.model.set_model('VAE_ConvNeXt_2D')`. The table shows the actual defaults for this model variant. A dash in Default search means the parameter stays fixed unless you assign a compatible distribution through `config.model.search`.

| Parameter | Type | Default | Meaning / values | Default search |
| --- | --- | --- | --- | --- |
| `config.model.parameters.n_res_blocks` | `int` | `4` | Number of residual blocks per level. | `IntRange(low=4, high=5, step=1, log=False)` |
| `config.model.parameters.n_levels` | `int` | `4` | Number of encoder and decoder levels. | — |
| `config.model.parameters.z_channels` | `int` | `32` | Channel count in the spatial bottleneck. | `Choice(values=(32, 64))` |
| `config.model.parameters.bottleneck_dim` | `int` | `64` | Dimension of the latent vector. | `Choice(values=(64, 128))` |
| `config.model.parameters.recon_weight` | `float` | `10.0` | Weight of the reconstruction loss. | — |
| `config.model.parameters.beta_kl_start` | `float` | `0.0` | Initial weight of the KL loss. | — |
| `config.model.parameters.beta_kl_max` | `float` | `0.08` | Maximum weight of the KL loss. | — |
| `config.model.parameters.beta_kl_warmup_start` | `int` | `0` | Epoch at which the KL weight begins to increase. | — |
| `config.model.parameters.beta_kl_warmup_epochs` | `int` | `200` | Number of epochs needed to reach beta_kl_max. | — |
| `config.model.parameters.free_bits` | `float` | `0.001` | KL free-bits allowance for latent dimensions. | — |
| `config.model.parameters.latent_recon_weight` | `float` | `0.0` | Weight of the latent reconstruction Smooth L1 loss; 0 disables the loss. | — |
| `config.model.parameters.latent_recon_noise_scale` | `float` | `1.0` | Scale of Gaussian noise added to the detached latent mean for the reconstruction cycle; positive. | — |
| `config.model.parameters.latent_recon_image_noise_std` | `float` | `0.03` | Standard deviation of Gaussian noise added to the cycle image before re-encoding during training; non-negative. | — |
| `config.model.parameters.recon_loss` | `str` | `'smoothl1'` | Reconstruction loss, such as 'mse' or 'smoothl1'. | — |
| `config.model.parameters.recon_smoothl1_beta` | `float` | `1.0` | Transition point of the Smooth L1 loss. | — |
| `config.model.parameters.use_transpose_conv` | `bool` | `False` | Use transposed convolutions for upsampling. | — |
| `config.model.parameters.fg_weight` | `float` | `1.0` | Additional weight for foreground pixels in the reconstruction loss. | — |
| `config.model.parameters.fg_threshold` | `float` | `0.0` | Threshold for selecting foreground pixels. | — |
| `config.model.parameters.drop_path_rate` | `float` | `0.001` | Stochastic depth rate in ConvNeXt blocks. | — |
| `config.model.parameters.dropout` | `float` | `0.001` | Dropout probability within the model. | — |
| `config.model.parameters.skip_dropout_p` | `float` | `1.0` | Shared dropout probability for skip connections. | — |
| `config.model.parameters.skip_dropout_ps` | `Optional[List[float]]` | `None` | Dropout per skip level; overrides skip_dropout_p and requires n_levels values. | — |
| `config.model.parameters.skip_alpha` | `float` | `0.0` | Shared skip connection scale; range [0, 1]. | — |
| `config.model.parameters.skip_alphas` | `Optional[List[float]]` | `None` | Scale per skip level; overrides skip_alpha and requires n_levels values. | — |
