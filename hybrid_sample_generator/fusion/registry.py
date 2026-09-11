"""Registry and construction helpers for fusion backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from hybrid_sample_generator.fusion.classical import ClassicalFusionBackend
from hybrid_sample_generator.fusion.interfaces import FusionBackend
from hybrid_sample_generator.fusion.classical import Config as ClassicalFusionConfig
from hybrid_sample_generator.fusion.settings import FusionParameters


@dataclass(frozen=True)
class FusionBackendSpec:
    name: str
    backend_cls: Type[FusionBackend]
    config_cls: Type[FusionParameters]

    def validate_configuration(self, parameters: FusionParameters) -> None:
        if not isinstance(parameters, self.config_cls):
            raise TypeError(f"Fusion backend {self.name!r} requires {self.config_cls.__module__}.Config.")
        parameters.validate()

    def build(self, parameters: FusionParameters | None = None) -> FusionBackend:
        parameters = self.build_configuration() if parameters is None else parameters
        self.validate_configuration(parameters)
        return self.backend_cls(fusion_params=parameters)

    def build_configuration(self, values=None) -> FusionParameters:
        parameters = self.config_cls(**({} if values is None else values))
        self.validate_configuration(parameters)
        return parameters


FUSION_BACKEND_REGISTRY: dict[str, FusionBackendSpec] = {
    "classical": FusionBackendSpec(
        name="classical",
        backend_cls=ClassicalFusionBackend,
        config_cls=ClassicalFusionConfig,
    ),
}


def get_fusion_backend_spec(name: str) -> FusionBackendSpec:
    try:
        return FUSION_BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown fusion backend: {name}. Supported backends: {list(FUSION_BACKEND_REGISTRY)}") from exc


def registered_fusion_backend_names() -> list[str]:
    return list(FUSION_BACKEND_REGISTRY)
