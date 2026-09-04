from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass


class FusionConfiguration:
    """Fusion-backend-specific fixed parameters."""

    def __init__(self, params):
        self.params = self.fusion_config_to_dict(params)

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "params" in value:
                return cls(value["params"])
            return cls(value)
        raise TypeError("fusion_params must be a FusionConfiguration or mapping.")

    def set_fusion_param(self, name, value):
        self.validate_fusion_param_name(name)
        self.params[name] = value

    def set_fusion_params(self, params=None, **kwargs):
        params = {} if params is None else dict(params)
        params.update(kwargs)
        for name, value in params.items():
            self.set_fusion_param(name, value)

    def validate_fusion_param_name(self, name):
        if name not in self.params:
            raise KeyError(f"Unknown fusion parameter {name!r}. Available parameters: {sorted(self.params)}")

    def fixed_params(self):
        return self.params.copy()

    def to_dict(self):
        return {"params": self.params.copy()}

    def __getitem__(self, key):
        return self.params[key]

    def get(self, key, default=None):
        return self.params.get(key, default)

    @staticmethod
    def fusion_config_to_dict(config):
        if is_dataclass(config):
            return asdict(config)
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError("Fusion config must be a dataclass instance or mapping.")


@dataclass
class FusionSettings:
    """Selected fusion backend, its parameters and optional checkpoint."""

    backend: str
    parameters: FusionConfiguration
    checkpoint: str | None = None

    @classmethod
    def for_backend(cls, backend: str) -> "FusionSettings":
        from fusion_backend.fusion_registry import get_fusion_backend_spec

        spec = get_fusion_backend_spec(backend)
        return cls(backend=backend, parameters=spec.build_configuration())

    def set_backend(self, backend: str, *, reset_parameters: bool = True) -> None:
        from fusion_backend.fusion_registry import get_fusion_backend_spec

        spec = get_fusion_backend_spec(backend)
        self.backend = backend
        self.checkpoint = None
        if reset_parameters:
            self.parameters = spec.build_configuration()

    def to_dict(self):
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "parameters": self.parameters.to_dict(),
        }

    @classmethod
    def from_dict(cls, values):
        return cls(
            backend=values["backend"],
            checkpoint=values.get("checkpoint"),
            parameters=FusionConfiguration.from_value(values["parameters"]),
        )
