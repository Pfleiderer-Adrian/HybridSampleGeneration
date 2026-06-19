from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from fusion_models.interfaces import FusionBackend
from fusion_models import ClassicalFusionBackend
from fusion_models.classical import Config as ClassicalFusionConfig
from fusion_models.classical import get_classical_fusion_configuration


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
}


def get_fusion_backend_spec(name: str) -> FusionBackendSpec:
    try:
        return FUSION_BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown fusion backend: {name}. Supported backends: {list(FUSION_BACKEND_REGISTRY)}") from exc


def registered_fusion_backend_names() -> list[str]:
    return list(FUSION_BACKEND_REGISTRY)
