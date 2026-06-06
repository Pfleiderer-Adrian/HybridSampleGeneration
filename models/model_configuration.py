from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

from models import VAE_ConvNeXt_2D, cVAE_ConvNeXt_2D, VAE_ConvNeXt_3D, cVAE_ConvNeXt_3D, VAE_ResNet_2D, VAE_ResNet_3D


DEFAULT_INPUT_ARTEFACTS = ("img", "fname")
IMMUTABLE_MODEL_PARAMS = {"in_channels"}


MODEL_CONFIG_CLASSES = {
    "VAE_ResNet_3D": VAE_ResNet_3D.Config,
    "VAE_ResNet_2D": VAE_ResNet_2D.Config,
    "VAE_ConvNeXt_3D": VAE_ConvNeXt_3D.Config,
    "cVAE_ConvNeXt_3D": cVAE_ConvNeXt_3D.Config,
    "VAE_ConvNeXt_2D": VAE_ConvNeXt_2D.Config,
    "cVAE_ConvNeXt_2D": cVAE_ConvNeXt_2D.Config,
}


class ModelConfiguration:
    """
    Model-specific hyperparameter search space.

    The min/max dictionaries define the Optuna search space. Equal min/max
    values are treated as fixed parameters by the trainer.
    """

    def __init__(
        self,
        min_config,
        max_config=None,
        *,
        input_artefacts=DEFAULT_INPUT_ARTEFACTS,
        immutable_params=IMMUTABLE_MODEL_PARAMS,
    ):
        self.min = self.model_config_to_dict(min_config)
        self.max = self.model_config_to_dict(max_config if max_config is not None else min_config)
        self.input_artefacts = tuple(input_artefacts)
        self.immutable_params = set(immutable_params)

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            if not hasattr(value, "input_artefacts"):
                value.input_artefacts = DEFAULT_INPUT_ARTEFACTS
            if not hasattr(value, "immutable_params"):
                value.immutable_params = set(IMMUTABLE_MODEL_PARAMS)
            else:
                value.immutable_params = set(value.immutable_params)
            return value
        if isinstance(value, Mapping):
            try:
                min_config = value["min"]
                max_config = value["max"]
            except KeyError as exc:
                raise KeyError("Model configuration mapping must contain 'min' and 'max'.") from exc
            return cls(
                min_config,
                max_config,
                input_artefacts=value.get(
                    "input_artefacts",
                    value.get("default_input_artefacts", DEFAULT_INPUT_ARTEFACTS),
                ),
                immutable_params=value.get("immutable_params", IMMUTABLE_MODEL_PARAMS),
            )
        raise TypeError("model_params must be a ModelConfiguration or mapping.")

    def set_hyperparameter_space(self, min_config, max_config):
        """
        Override the hyperparameter search space used by Optuna.
        """
        self.min = self.model_config_to_dict(min_config)
        self.max = self.model_config_to_dict(max_config)

    def set_model_param(self, name, value):
        """
        Fix one model parameter to a concrete value for every Optuna trial.
        """
        self.validate_model_param_name(name)
        self.min[name] = value
        self.max[name] = value

    def set_model_params(self, params=None, **kwargs):
        """
        Fix multiple model parameters to concrete values.
        """
        params = {} if params is None else dict(params)
        params.update(kwargs)
        for name, value in params.items():
            self.set_model_param(name, value)

    def set_model_param_range(self, name, min_value, max_value):
        """
        Set one Optuna search range/categorical choice pair for a parameter.
        """
        self.validate_model_param_name(name)
        self.min[name] = min_value
        self.max[name] = max_value

    def update_model_param_ranges(self, ranges=None, **kwargs):
        """
        Update multiple model parameter ranges.
        """
        ranges = {} if ranges is None else dict(ranges)
        ranges.update(kwargs)
        for name, value_range in ranges.items():
            if not isinstance(value_range, (tuple, list)) or len(value_range) != 2:
                raise ValueError(f"Range for {name!r} must be a (min_value, max_value) pair.")
            self.set_model_param_range(name, value_range[0], value_range[1])

    def validate_model_param_name(self, name):
        valid_names = set(self.min) | set(self.max)
        if name not in valid_names:
            raise KeyError(f"Unknown model parameter {name!r}. Available parameters: {sorted(valid_names)}")
        if name in self.immutable_params:
            raise AttributeError(
                f"Model parameter {name!r} is derived from anomaly_size and cannot be changed afterwards."
            )

    def to_dict(self):
        return {
            "min": self.min.copy(),
            "max": self.max.copy(),
            "input_artefacts": tuple(self.input_artefacts),
            "immutable_params": tuple(sorted(self.immutable_params)),
        }

    def __getitem__(self, key):
        if key == "min":
            return self.min
        if key == "max":
            return self.max
        if key == "input_artefacts":
            return self.input_artefacts
        if key == "immutable_params":
            return self.immutable_params
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    @staticmethod
    def model_config_to_dict(config):
        if is_dataclass(config):
            return asdict(config)
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError("Model config must be a dataclass instance or mapping.")


def get_model_config_class(model_name):
        try:
            return MODEL_CONFIG_CLASSES[model_name]
        except KeyError:
            raise ValueError(f"Unknown model: {model_name}. Supported models: {list(MODEL_CONFIG_CLASSES)}")


def get_model_configuration(model_name, in_channels, debug=False):
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
            model_params = ModelConfiguration(_VAE3D_min_params, _VAE3D_max_params, input_artefacts=["img","fname"])


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
                beta_kl_start=0.0,
                beta_kl_max=7.0,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=100,
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
                beta_kl_start=0.0,
                beta_kl_max=7.0,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=100,
                fg_weight=2.0,
                fg_threshold=0.0,
                skip_dropout_p=0.6,
                skip_alpha=0.2,
                recon_loss="mse",
                use_transpose_conv=False))
            model_params = ModelConfiguration(_VAE3D_min_params, _VAE3D_max_params, input_artefacts=["img","fname"])

        if model_name == "cVAE_ConvNeXt_3D":
            _cVAE3D_min_params = asdict(cVAE_ConvNeXt_3D.Config(
                in_channels=in_channels,
                num_anomaly_classes=None,
                n_res_blocks=5,
                n_spade_blocks=2,
                n_levels=5,
                z_channels=128,
                bottleneck_dim=256,
                use_multires_skips=True,
                recon_weight=1.0,
                beta_kl=4.0,
                beta_kl_start=0.0,
                beta_kl_max=7.0,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=100,
                fg_weight=1.0,
                fg_threshold=0.0,
                recon_loss="mse",
                skip_dropout_p=0.6,
                skip_alpha=0.2,
                use_transpose_conv=False))
            _cVAE3D_max_params = asdict(cVAE_ConvNeXt_3D.Config(
                in_channels=in_channels,
                num_anomaly_classes=None,
                n_res_blocks=6,
                n_spade_blocks=2,
                n_levels=6,
                z_channels=128,
                bottleneck_dim=256,
                use_multires_skips=True,
                recon_weight=1.0,
                beta_kl=4.0,
                beta_kl_start=0.0,
                beta_kl_max=7.0,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=100,
                fg_weight=2.0,
                fg_threshold=0.0,
                skip_dropout_p=0.6,
                skip_alpha=0.2,
                recon_loss="mse",
                use_transpose_conv=False))
            model_params = ModelConfiguration(_cVAE3D_min_params, _cVAE3D_max_params, input_artefacts=["img","fname","ori_mask","tgt_mask"])

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
            model_params = ModelConfiguration(_VAE2D_min_params, _VAE2D_max_params, input_artefacts=["img","fname"])
        

        if model_name == "VAE_ConvNeXt_2D":
            _VAE2D_min_params = asdict(VAE_ConvNeXt_2D.Config(
                in_channels=in_channels,
                n_res_blocks=4,
                n_levels=4,
                z_channels=32,
                bottleneck_dim=64,
                use_multires_skips=False,

                recon_loss="smoothl1",
                recon_weight=10.0,          

                drop_path_rate=0.001,       
                dropout=0.001,              
                skip_dropout_p=1.0,        
                skip_alpha=0.0,
                use_transpose_conv=False,

                beta_kl=0.05,             
                beta_kl_start=0.0,
                beta_kl_max=0.08,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=1000, 

                free_bits=0.001,

                fg_weight=1.0,
                fg_threshold=0.0  ))
            
            _VAE2D_max_params = asdict(VAE_ConvNeXt_2D.Config(                
                in_channels=in_channels,
                n_res_blocks=4,
                n_levels=4,
                z_channels=32,
                bottleneck_dim=64,
                use_multires_skips=False,

                recon_loss="smoothl1",
                recon_weight=10.0,          

                drop_path_rate=0.001,       
                dropout=0.001,              
                skip_dropout_p=1.0,        
                skip_alpha=0.0,
                use_transpose_conv=False,

                beta_kl=0.05,             
                beta_kl_start=0.0,
                beta_kl_max=0.08,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=1000, 

                free_bits=0.001,

                fg_weight=1.0,
                fg_threshold=0.0
                ))
            model_params = ModelConfiguration(_VAE2D_min_params, _VAE2D_max_params, input_artefacts=["img","fname"])

        if model_name == "cVAE_ConvNeXt_2D":
            _cVAE2D_min_params = asdict(cVAE_ConvNeXt_2D.Config(
                in_channels=in_channels,
                num_anomaly_classes=None,
                n_res_blocks=4,
                n_spade_blocks=2,
                n_levels=4,
                z_channels=32,
                bottleneck_dim=64,
                use_multires_skips=False,
                recon_loss="smoothl1",
                recon_weight=10.0,          
                drop_path_rate=0.001,       
                dropout=0.001,              
                skip_dropout_p=1.0,        
                skip_alpha=0.0,
                use_transpose_conv=False,
                beta_kl=0.05,             
                beta_kl_start=0.0,
                beta_kl_max=0.08,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=1000, 
                free_bits=0.001,
                fg_weight=1.0,
                fg_threshold=0.0))
            
            _cVAE2D_max_params = asdict(cVAE_ConvNeXt_2D.Config(                
                in_channels=in_channels,
                num_anomaly_classes=None,
                n_res_blocks=4,
                n_levels=4,
                n_spade_blocks=2,
                z_channels=32,
                bottleneck_dim=64,
                use_multires_skips=False,
                recon_loss="smoothl1",
                recon_weight=10.0,          
                drop_path_rate=0.001,       
                dropout=0.001,              
                skip_dropout_p=1.0,        
                skip_alpha=0.0,
                use_transpose_conv=False,
                beta_kl=0.05,             
                beta_kl_start=0.0,
                beta_kl_max=0.08,
                beta_kl_warmup_start=0,
                beta_kl_warmup_epochs=1000, 
                free_bits=0.001,
                fg_weight=1.0,
                fg_threshold=0.0))
            model_params = ModelConfiguration(_cVAE2D_min_params, _cVAE2D_max_params, input_artefacts=["img","fname","ori_mask","tgt_mask"])

        if model_params is None:
            raise ValueError(f"Unknown model: {model_name}. Supported models: {list(MODEL_CONFIG_CLASSES)}")

        if debug:
            model_params.set_hyperparameter_space(model_params.max, model_params.max)

        return model_params
