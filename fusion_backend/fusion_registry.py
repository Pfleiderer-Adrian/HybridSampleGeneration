from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from fusion_backend.interfaces import FusionBackend
from fusion_backend import ClassicalFusionBackend, LearnedResidualAlphaFusionBackend
from fusion_backend.classical import Config as ClassicalFusionConfig
from fusion_backend.classical import get_classical_fusion_configuration
from fusion_backend.learned_residual_alpha import Config as LearnedResidualAlphaFusionConfig
from fusion_backend.learned_residual_alpha import get_learned_residual_alpha_fusion_configuration


@dataclass(frozen=True)
class FusionBackendSpec:
    name: str
    backend_cls: Type[FusionBackend]
    config_cls: Type
    config_factory: Callable[[], object]
    spatial_dims: int | None = None
    trainable: bool = False

    def build(self, params: dict | None = None) -> FusionBackend:
        return self.backend_cls(**(params or {}))

    def build_configuration(self):
        return self.config_factory()


FUSION_BACKEND_REGISTRY: dict[str, FusionBackendSpec] = {
    "classical": FusionBackendSpec(
        name="classical",
        backend_cls=ClassicalFusionBackend,
        config_cls=ClassicalFusionConfig,
        config_factory=get_classical_fusion_configuration,
    ),
    "learned_residual_alpha": FusionBackendSpec(
        name="learned_residual_alpha",
        backend_cls=LearnedResidualAlphaFusionBackend,
        config_cls=LearnedResidualAlphaFusionConfig,
        config_factory=get_learned_residual_alpha_fusion_configuration,
        trainable=True,
    ),
}


def get_fusion_backend_spec(name: str) -> FusionBackendSpec:
    try:
        return FUSION_BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown fusion backend: {name}. Supported backends: {list(FUSION_BACKEND_REGISTRY)}") from exc


def registered_fusion_backend_names() -> list[str]:
    return list(FUSION_BACKEND_REGISTRY)
