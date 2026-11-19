"""Machine-readable description of every configuration parameter.

``configuration_schema()`` reports type, default, value range and description
for each parameter, derived from the ``param(...)`` declarations on the
configuration dataclasses. Graphical front ends build their forms from it, so
new parameters, generator models and fusion backends appear without changes
to the front end. The result contains only JSON-compatible values.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from hybrid_sample_generator.configuration.parameters import (
    describe_instance,
    iter_parameters,
)


# Search distributions that SearchSpace accepts for a parameter of each type.
_SEARCH_DISTRIBUTIONS = {
    "int": ["int", "choice"],
    "float": ["float", "choice"],
    "bool": ["choice"],
    "str": ["choice"],
    "choice": ["choice"],
}


def configuration_schema() -> dict[str, Any]:
    """Describe all configuration sections, model variants and fusion backends.

    Layout::

        {
          "schema_version": 10,
          "sections": {
            "matching": {"type": "group", "fields": {"batch_size": {...}, ...}},
            ...
            "model":  {"type": "variant", "selector": "name", "default": ...,
                       "variants": {"<model>": {"spatial_dims": 2,
                                                "parameters": <group>,
                                                "search": {...}}}},
            "fusion": {"type": "variant", "selector": "backend", "default": ...,
                       "variants": {"<backend>": {"parameters": <group>}}},
          },
        }

    A parameter descriptor has a ``type`` (``bool``, ``int``, ``float``,
    ``str``, ``choice``, ``list``, ``map``, ``any_of`` or ``any``) and, where
    applicable, ``default``, ``min``/``max`` with ``min_exclusive``/
    ``max_exclusive``, ``choices``, ``items``, ``min_length``/``max_length``,
    ``nullable``, ``required`` and ``doc``. Groups flagged ``custom`` need a
    dedicated editor and should be skipped by generic forms. Descriptors
    flagged ``validated: false`` list the recommended values only.

    Rules relating several parameters are not part of the schema; check a
    complete configuration with ``Configuration.from_dict(values)``.
    """
    from hybrid_sample_generator.configuration.root import Configuration
    from hybrid_sample_generator.fusion.registry import FUSION_BACKEND_REGISTRY
    from hybrid_sample_generator.generation.registry import MODEL_REGISTRY

    config = Configuration("example", study_folder="results/example")
    defaults = config.to_dict()

    sections: dict[str, Any] = {}
    for name, value in vars(config).items():
        if name in ("schema_version", "model", "fusion"):
            continue
        if not is_dataclass(value):
            raise TypeError(f"Configuration section {name!r} is not a dataclass.")
        sections[name] = describe_instance(type(value), defaults[name])

    model_variants = {}
    for name, spec in MODEL_REGISTRY.items():
        parameters = spec.build_configuration()
        group = describe_instance(type(parameters), _plain(asdict(parameters)))
        model_variants[name] = {
            "spatial_dims": spec.spatial_dims,
            "parameters": group,
            "search": {
                "default": _plain(spec.build_search_space(parameters).to_dict()),
                "distributions": {
                    path: _SEARCH_DISTRIBUTIONS[entry["type"]]
                    for path, entry in iter_parameters(group)
                    if entry["type"] in _SEARCH_DISTRIBUTIONS
                },
            },
        }
    sections["model"] = {
        "type": "variant",
        "selector": "name",
        "default": config.model.name,
        "variants": model_variants,
    }

    sections["fusion"] = {
        "type": "variant",
        "selector": "backend",
        "default": config.fusion.backend,
        "variants": {
            name: {
                "parameters": describe_instance(
                    spec.config_cls, _plain(asdict(spec.build_configuration()))
                )
            }
            for name, spec in FUSION_BACKEND_REGISTRY.items()
        },
    }
    return {"schema_version": Configuration.SCHEMA_VERSION, "sections": sections}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = ["configuration_schema", "iter_parameters"]
