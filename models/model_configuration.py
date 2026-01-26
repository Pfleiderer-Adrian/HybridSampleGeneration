# model specific hyperparameter for dynamic tuning via optuna
        
from dataclasses import asdict
from models import VAE_ConvNeXt_2D, VAE_ConvNeXt_3D, VAE_ResNet_2D, VAE_ResNet_3D


def get_model_configuration(model_name, in_channels):
        model_params = None
        # VAE3D parameter
        if model_name == "VAE_ResNet_3D":
            _VAE3D_min_params = asdict(VAE_ResNet_3D.Config(
                in_channels=in_channels,
                n_res_blocks=4,
                n_levels=4,
                z_channels=64,
                bottleneck_dim=128,
                use_multires_skips = True,
                recon_weight = 100.0,
                beta_kl = 0.05,
                fg_weight=1.0,
                fg_threshold=0.0,
                recon_loss="mse",
                use_transpose_conv = False))
            _VAE3D_max_params = asdict(VAE_ResNet_3D.Config(
                in_channels=in_channels,
                n_res_blocks=5,
                n_levels=5,
                z_channels=128,
                bottleneck_dim=256,
                use_multires_skips = True,
                recon_weight = 300.0,
                beta_kl = 0.1,
                fg_weight=2.0,
                fg_threshold=0.0,
                recon_loss="mse",
                use_transpose_conv=False))
            model_params = {"min": _VAE3D_min_params, "max": _VAE3D_max_params}

        if model_name == "VAE_ConvNeXt_3D":
            _VAE3D_min_params = asdict(VAE_ConvNeXt_3D.Config(
                in_channels=in_channels,
                n_res_blocks=5,
                n_levels=5,
                z_channels=128,
                bottleneck_dim=256,
                use_multires_skips = True,
                recon_weight = 1.0,
                beta_kl = 4.0,
                fg_weight=1.0,
                fg_threshold=0.0,
                recon_loss="mse",
                skip_dropout_p=0.6,
                skip_alpha=0.2,
                use_transpose_conv = False))
            _VAE3D_max_params = asdict(VAE_ConvNeXt_3D.Config(
                in_channels=in_channels,
                n_res_blocks=6,
                n_levels=6,
                z_channels=128,
                bottleneck_dim=256,
                use_multires_skips = True,
                recon_weight = 1.0,
                beta_kl = 4.0,
                fg_weight=2.0,
                fg_threshold=0.0,
                skip_dropout_p=0.6,
                skip_alpha=0.2,
                recon_loss="mse",
                use_transpose_conv=False))
            model_params = {"min": _VAE3D_min_params, "max": _VAE3D_max_params}

        # VAE2D parameter
        if model_name == "VAE_ResNet_2D":
            _VAE2D_min_params = asdict(VAE_ResNet_2D.Config(
                in_channels=in_channels,
                n_res_blocks=4,
                n_levels=4,
                z_channels=32,
                bottleneck_dim=64,
                use_multires_skips = False,
                recon_weight = 5.0,
                beta_kl = 0.1,
                use_transpose_conv=False))
            _VAE2D_max_params = asdict(VAE_ResNet_2D.Config(
                in_channels=in_channels,
                n_res_blocks=5,
                n_levels=5,
                z_channels=64,
                bottleneck_dim=128,
                use_multires_skips = False,
                recon_weight = 100.0,
                beta_kl = 0.5,
                use_transpose_conv=False))
            model_params = {"min": _VAE2D_min_params, "max": _VAE2D_max_params}
        
        if model_name == "VAE_ConvNeXt_2D":
            _VAE2D_min_params = asdict(VAE_ConvNeXt_2D.Config(
                in_channels=in_channels,
                n_res_blocks=3,
                n_levels=4,
                z_channels=64,
                bottleneck_dim=96,
                use_multires_skips = False,
                recon_weight = 5.0,
                beta_kl = 2.0,
                recon_loss="smoothl1",
                use_transpose_conv=False,
                drop_path_rate = 0.1,  # Stochastic depth max rate (0.0 disables)
                dropout = 0.05,
                skip_dropout_p = 0.6,  # Drop entire skip-tensors per sample during training (0.0 disables)
                skip_alpha = 0.05,
                beta_kl_max = 0.5 ,          # target KL weight (defaults to beta_kl)
                beta_kl_start = 0.0,         # starting KL weight
                beta_kl_warmup_epochs = 50,
                free_bits=0.01   ))
            _VAE2D_max_params = asdict(VAE_ConvNeXt_2D.Config(                
                in_channels=in_channels,
                n_res_blocks=3,
                n_levels=4,
                z_channels=64,
                bottleneck_dim=96,
                use_multires_skips = False,
                recon_weight = 5.0,
                beta_kl = 2.0,
                recon_loss="smoothl1",
                use_transpose_conv=False,
                drop_path_rate = 0.1,  # Stochastic depth max rate (0.0 disables)
                dropout = 0.05,
                skip_dropout_p = 0.6,  # Drop entire skip-tensors per sample during training (0.0 disables)
                skip_alpha = 0.05,
                beta_kl_max = 0.5 ,          # target KL weight (defaults to beta_kl)
                beta_kl_start = 0.0,         # starting KL weight
                beta_kl_warmup_epochs = 50,
                free_bits=0.01  ))
            model_params = {"min": _VAE2D_min_params, "max": _VAE2D_max_params}
        return model_params