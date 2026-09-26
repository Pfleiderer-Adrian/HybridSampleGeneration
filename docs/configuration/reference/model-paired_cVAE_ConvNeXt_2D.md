# paired_cVAE_ConvNeXt_2D

Select with `config.model.set_model('paired_cVAE_ConvNeXt_2D')`. The table shows the actual defaults for this model variant. A dash in Default search means the parameter stays fixed unless you assign a compatible distribution through `config.model.search`.

| Parameter | Type | Default | Meaning / values | Default search |
| --- | --- | --- | --- | --- |
| `config.model.parameters.n_res_blocks` | `int` | `4` | Number of residual blocks per level. | `IntRange(low=4, high=5, step=1, log=False)` |
| `config.model.parameters.n_spade_blocks` | `int` | `2` | Number of mask-conditioned SPADE blocks. | — |
| `config.model.parameters.n_levels` | `int` | `4` | Number of encoder and decoder levels. | — |
| `config.model.parameters.z_channels` | `int` | `32` | Channel count in the spatial bottleneck. | `Choice(values=(32, 64))` |
| `config.model.parameters.bottleneck_dim` | `int` | `64` | Dimension of the latent vector. | `Choice(values=(64, 128))` |
| `config.model.parameters.recon_weight` | `float` | `10.0` | Weight of the reconstruction loss. | — |
| `config.model.parameters.beta_kl_start` | `float` | `0.0` | Initial weight of the KL loss. | — |
| `config.model.parameters.beta_kl_max` | `float` | `0.08` | Maximum weight of the KL loss. | — |
| `config.model.parameters.beta_kl_warmup_start` | `int` | `0` | Epoch at which the KL weight begins to increase. | — |
| `config.model.parameters.beta_kl_warmup_epochs` | `int` | `200` | Number of epochs needed to reach beta_kl_max. | — |
| `config.model.parameters.free_bits` | `float` | `0.001` | KL free-bits allowance for latent dimensions. | — |
| `config.model.parameters.recon_loss` | `str` | `'smoothl1'` | Reconstruction loss, such as 'mse' or 'smoothl1'. | — |
| `config.model.parameters.recon_smoothl1_beta` | `float` | `1.0` | Transition point of the Smooth L1 loss. | — |
| `config.model.parameters.use_transpose_conv` | `bool` | `False` | Use transposed convolutions for upsampling. | — |
| `config.model.parameters.foreground_weight` | `float` | `0.8` | Relative weight of the mask foreground mean in the reconstruction loss; non-negative. | — |
| `config.model.parameters.background_weight` | `float` | `0.2` | Relative weight of the mask background mean in the reconstruction loss; non-negative. | — |
| `config.model.parameters.drop_path_rate` | `float` | `0.001` | Stochastic depth rate in ConvNeXt blocks. | — |
| `config.model.parameters.dropout` | `float` | `0.001` | Dropout probability within the model. | — |
| `config.model.parameters.identity_pair_probability` | `float` | `0.2` | Probability of using an unchanged source/target pair during paired source-to-target training; range [0, 1]. | — |
