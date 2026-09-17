"""Typed generator settings and explicit Optuna search distributions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from numbers import Real
from typing import Any, Iterator


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (self.low, self.high, self.step)):
            raise TypeError("IntRange bounds and step must be integers.")
        if not isinstance(self.log, bool):
            raise TypeError("IntRange.log must be a boolean.")
        if self.low > self.high:
            raise ValueError("IntRange.low must not exceed high.")
        if self.step <= 0:
            raise ValueError("IntRange.step must be positive.")
        if self.log and self.low <= 0:
            raise ValueError("A logarithmic IntRange requires positive bounds.")
        if self.log and self.step != 1:
            raise ValueError("A logarithmic IntRange cannot define a step other than 1.")


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, Real)
               for value in (self.low, self.high)):
            raise TypeError("FloatRange bounds must be numeric.")
        if self.step is not None and (
            isinstance(self.step, bool) or not isinstance(self.step, Real)
        ):
            raise TypeError("FloatRange.step must be numeric or None.")
        if not isinstance(self.log, bool):
            raise TypeError("FloatRange.log must be a boolean.")
        if self.low > self.high:
            raise ValueError("FloatRange.low must not exceed high.")
        if self.step is not None and self.step <= 0:
            raise ValueError("FloatRange.step must be positive.")
        if self.log and self.step is not None:
            raise ValueError("A logarithmic FloatRange cannot define a step.")
        if self.log and self.low <= 0:
            raise ValueError("A logarithmic FloatRange requires positive bounds.")


@dataclass(frozen=True)
class Choice:
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("Choice requires at least one value.")


SearchDistribution = IntRange | FloatRange | Choice


class SearchSpace:
    """Validated search distributions bound to one parameter dataclass."""

    def __init__(self, parameters, **distributions: SearchDistribution) -> None:
        if not is_dataclass(parameters) or isinstance(parameters, type):
            raise TypeError("SearchSpace parameters must be a dataclass instance.")
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(
            self,
            "_parameter_names",
            {field.name for field in fields(parameters)},
        )
        object.__setattr__(self, "_distributions", {})
        for name, distribution in distributions.items():
            self._set_distribution(name, distribution)

    def __getattr__(self, name: str) -> SearchDistribution:
        try:
            return self._distributions[name]
        except KeyError as exc:
            raise AttributeError(
                f"Parameter {name!r} is not part of this search space."
            ) from exc

    def __setattr__(self, name: str, value: SearchDistribution) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._set_distribution(name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            raise AttributeError(f"Cannot delete internal attribute {name!r}.")
        try:
            del self._distributions[name]
        except KeyError as exc:
            raise AttributeError(
                f"Parameter {name!r} is not part of this search space."
            ) from exc

    def __contains__(self, name: object) -> bool:
        return name in self._distributions

    def __iter__(self) -> Iterator[str]:
        return iter(self._distributions)

    def __len__(self) -> int:
        return len(self._distributions)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SearchSpace)
            and type(self._parameters) is type(other._parameters)
            and self._distributions == other._distributions
        )

    def clear(self) -> None:
        self._distributions.clear()

    def items(self):
        return self._distributions.items()

    def names(self) -> tuple[str, ...]:
        return tuple(self._distributions)

    def validate(self) -> None:
        for name, distribution in self._distributions.items():
            self._validate_distribution(name, distribution)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        self.validate()
        return {
            name: _distribution_to_dict(distribution)
            for name, distribution in self._distributions.items()
        }

    @classmethod
    def from_dict(cls, parameters, values: dict[str, Any]) -> "SearchSpace":
        search = cls(parameters)
        for name, distribution in values.items():
            search._set_distribution(name, _distribution_from_dict(distribution))
        return search

    def _set_distribution(self, name: str, distribution: SearchDistribution) -> None:
        self._validate_distribution(name, distribution)
        self._distributions[name] = distribution

    def _validate_distribution(
        self,
        name: str,
        distribution: SearchDistribution,
    ) -> None:
        if name not in self._parameter_names:
            raise AttributeError(f"Unknown model search parameter {name!r}.")
        if not isinstance(distribution, (IntRange, FloatRange, Choice)):
            raise TypeError(
                f"Search parameter {name!r} must use IntRange, FloatRange, or Choice."
            )
        _validate_distribution_type(
            name,
            getattr(self._parameters, name),
            distribution,
        )


class GeneratorModelSettings:
    """Selected model, concrete parameters, and optional search distributions."""

    def __init__(self, name: str = "cVAE_ConvNeXt_2D") -> None:
        self.set_model(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self):
        return self._parameters

    @property
    def search(self) -> SearchSpace:
        return self._search

    def set_model(self, name: str) -> None:
        """Select a model and reset its concrete parameters and default search."""
        from hybrid_sample_generator.generation.registry import get_model_spec

        spec = get_model_spec(name)
        parameters = spec.build_configuration()
        search = spec.build_search_space(parameters)
        self._name = name
        self._parameters = parameters
        self._search = search

    def validate(self) -> None:
        from hybrid_sample_generator.generation.registry import get_model_spec

        spec = get_model_spec(self.name)
        if not isinstance(self.parameters, spec.config_cls):
            raise TypeError(
                f"Parameters for {self.name!r} must be {spec.config_cls.__name__}."
            )
        self.search.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "parameters": asdict(self.parameters),
            "search": self.search.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GeneratorModelSettings":
        from hybrid_sample_generator.generation.registry import get_model_spec

        unknown = set(values) - {"name", "parameters", "search"}
        if unknown:
            raise TypeError(f"Unknown model setting(s): {sorted(unknown)}")
        spec = get_model_spec(values["name"])
        parameters = spec.config_cls(**values["parameters"])
        search = SearchSpace.from_dict(parameters, values.get("search", {}))
        settings = cls.__new__(cls)
        settings._name = values["name"]
        settings._parameters = parameters
        settings._search = search
        settings.validate()
        return settings


def _validate_distribution_type(name, default, distribution) -> None:
    if isinstance(distribution, IntRange):
        valid = isinstance(default, int) and not isinstance(default, bool)
    elif isinstance(distribution, FloatRange):
        valid = isinstance(default, Real) and not isinstance(default, bool)
    else:
        valid = all(_same_value_type(default, value) for value in distribution.values)
    if not valid:
        raise TypeError(
            f"Search distribution for {name!r} is incompatible with "
            f"parameter value {default!r}."
        )


def _same_value_type(default, value) -> bool:
    if default is None:
        return True
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, Real) and not isinstance(value, bool)
    return isinstance(value, type(default))


def _distribution_to_dict(distribution: SearchDistribution) -> dict[str, Any]:
    values = asdict(distribution)
    values["type"] = {
        IntRange: "int",
        FloatRange: "float",
        Choice: "choice",
    }[type(distribution)]
    return values


def _distribution_from_dict(values: dict[str, Any]) -> SearchDistribution:
    values = dict(values)
    kind = values.pop("type")
    if kind == "int":
        return IntRange(**values)
    if kind == "float":
        return FloatRange(**values)
    if kind == "choice":
        return Choice(**values)
    raise ValueError(f"Unknown search distribution type {kind!r}.")


__all__ = [
    "Choice",
    "FloatRange",
    "GeneratorModelSettings",
    "IntRange",
    "SearchSpace",
]
