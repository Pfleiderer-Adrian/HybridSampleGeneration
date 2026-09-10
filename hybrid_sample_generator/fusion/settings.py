from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


class FusionParameters(Protocol):
    """Common capability required from backend-specific parameter objects."""

    def validate(self) -> None: ...


@dataclass
class FusionSettings:
    """Selected fusion backend, its parameters and optional checkpoint."""

    backend: str
    parameters: FusionParameters
    checkpoint: str | None = None

    @classmethod
    def for_backend(cls, backend: str) -> "FusionSettings":
        from hybrid_sample_generator.fusion.registry import get_fusion_backend_spec

        spec = get_fusion_backend_spec(backend)
        return cls(backend=backend, parameters=spec.build_configuration())

    def set_backend(self, backend: str) -> None:
        from hybrid_sample_generator.fusion.registry import get_fusion_backend_spec

        spec = get_fusion_backend_spec(backend)
        parameters = spec.build_configuration()
        self.backend = backend
        self.checkpoint = None
        self.parameters = parameters

    def validate(self) -> None:
        from hybrid_sample_generator.fusion.registry import get_fusion_backend_spec

        get_fusion_backend_spec(self.backend).validate_configuration(self.parameters)

    def to_dict(self):
        self.validate()
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "parameters": asdict(self.parameters),
        }

    @classmethod
    def from_dict(cls, values):
        from hybrid_sample_generator.fusion.registry import get_fusion_backend_spec

        spec = get_fusion_backend_spec(values["backend"])
        parameter_values = values["parameters"]
        return cls(
            backend=values["backend"],
            checkpoint=values.get("checkpoint"),
            parameters=spec.build_configuration(parameter_values),
        )
