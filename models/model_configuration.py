from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

from models.VAEs.VAE_ConvNeXt import VAE_ConvNeXt_2D, VAE_ConvNeXt_3D
from models.VAEs.VAE_ResNet import VAE_ResNet_2D, VAE_ResNet_3D
from models.VAEs.cVAE_ConvNeXt import cVAE_ConvNeXt_2D, cVAE_ConvNeXt_3D


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
    from models.VAEs.VAE_ConvNeXt.configuration import (
        MODEL_CONFIGURATION_FACTORIES as CONVNEXT_VAE_CONFIGURATION_FACTORIES,
    )
    from models.VAEs.VAE_ResNet.configuration import (
        MODEL_CONFIGURATION_FACTORIES as RESNET_VAE_CONFIGURATION_FACTORIES,
    )
    from models.VAEs.cVAE_ConvNeXt.configuration import (
        MODEL_CONFIGURATION_FACTORIES as CONVNEXT_CVAE_CONFIGURATION_FACTORIES,
    )

    model_configuration_factories = {
        **RESNET_VAE_CONFIGURATION_FACTORIES,
        **CONVNEXT_VAE_CONFIGURATION_FACTORIES,
        **CONVNEXT_CVAE_CONFIGURATION_FACTORIES,
    }

    try:
        model_params = model_configuration_factories[model_name](in_channels)
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}. Supported models: {list(MODEL_CONFIG_CLASSES)}")

    if debug:
        model_params.set_hyperparameter_space(model_params.max, model_params.max)

    return model_params
