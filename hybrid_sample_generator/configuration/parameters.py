"""Declarative parameter metadata shared by validation, the GUI schema and the docs.

Configuration dataclasses declare each field with ``param(...)``. The value
range, allowed values and description live next to the field, so validation,
``configuration_schema()`` and the generated reference cannot drift apart.

This module only depends on the standard library and can be imported from any
package without creating import cycles.
"""

from __future__ import annotations

import math
import types
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


_METADATA_KEY = "parameter"


@dataclass(frozen=True)
class ParameterSpec:
    """Constraints and description attached to one configuration field."""

    doc: str = ""
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None
    choices: tuple[Any, ...] | None = None
    min_length: int | None = None
    max_length: int | None = None
    schema: dict[str, Any] | None = None
    validated: bool = True
    required: bool = False
    default_label: str | None = None
    custom: bool = False


def param(
    default=MISSING,
    *,
    default_factory=MISSING,
    doc: str = "",
    ge=None,
    gt=None,
    le=None,
    lt=None,
    choices=None,
    min_length: int | None = None,
    max_length: int | None = None,
    schema: dict[str, Any] | None = None,
    validated: bool = True,
    required: bool = False,
    default_label: str | None = None,
    custom: bool = False,
):
    """Declare a configuration field together with its constraints.

    ``ge``/``gt``/``le``/``lt`` bound numeric values; on sequence fields they
    bound every element. ``choices`` restricts the value to a fixed set.
    ``min_length``/``max_length`` bound sequence lengths. ``schema`` replaces
    the descriptor derived from the type annotation, and ``validated=False``
    leaves checking of that field to the owning class. ``custom=True`` marks a
    field that needs a dedicated editor and is skipped by generic consumers.
    """
    if choices is not None:
        choices = tuple(sorted(choices) if isinstance(choices, (set, frozenset)) else choices)
    spec = ParameterSpec(
        doc=doc, ge=ge, gt=gt, le=le, lt=lt, choices=choices,
        min_length=min_length, max_length=max_length, schema=schema,
        validated=validated, required=required, default_label=default_label,
        custom=custom,
    )
    kwargs: dict[str, Any] = {"metadata": {_METADATA_KEY: spec}}
    if default is not MISSING:
        kwargs["default"] = default
    if default_factory is not MISSING:
        kwargs["default_factory"] = default_factory
    return field(**kwargs)


@lru_cache(maxsize=None)
def describe_class(cls) -> dict[str, dict[str, Any]]:
    """Return one descriptor per field of a configuration dataclass.

    Descriptors are plain JSON-compatible dictionaries. Nested dataclasses
    become ``{"type": "group", "fields": {...}}``. Defaults are not included
    because they can depend on the instance; see ``describe_instance``.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass.")
    hints = get_type_hints(cls)
    result: dict[str, dict[str, Any]] = {}
    for item in fields(cls):
        spec = item.metadata.get(_METADATA_KEY, ParameterSpec())
        annotation = hints[item.name]
        if is_dataclass(annotation):
            entry: dict[str, Any] = {"type": "group", "fields": describe_class(annotation)}
        elif spec.schema is not None:
            entry = _copy(spec.schema)
        else:
            entry = _describe_type(annotation)
            _apply_constraints(entry, spec)
        if spec.doc:
            entry["doc"] = spec.doc
        if spec.required:
            entry["required"] = True
        if spec.default_label is not None:
            entry["default_label"] = spec.default_label
        if spec.custom:
            entry["custom"] = True
        if not spec.validated:
            entry["validated"] = False
        result[item.name] = entry
    return result


def describe_instance(cls, defaults: dict[str, Any]) -> dict[str, Any]:
    """Return a group descriptor for ``cls`` with defaults from a plain dict."""
    return {"type": "group", "fields": _with_defaults(describe_class(cls), defaults)}


def iter_parameters(group: dict[str, Any], prefix: str = ""):
    """Yield ``(dotted_path, descriptor)`` for every non-group entry."""
    for name, entry in group["fields"].items():
        path = f"{prefix}.{name}" if prefix else name
        if entry["type"] == "group":
            yield from iter_parameters(entry, path)
        else:
            yield path, entry


def validate_parameters(instance, prefix: str) -> None:
    """Check every declared field of a configuration dataclass instance.

    Raises ``ValueError`` naming the dotted parameter path. Rules that relate
    several fields stay in the ``validate()`` method of the owning class.
    """
    _validate_group(instance, describe_class(type(instance)), prefix)


def check_value(value, descriptor: dict[str, Any], path: str) -> None:
    """Check one value against a descriptor; raise ``ValueError`` on mismatch."""
    _check(value, descriptor, path)


def range_text(descriptor: dict[str, Any]) -> str:
    """Return a short human-readable range such as ``[0, 1)`` or ``>= 1``."""
    low, high = descriptor.get("min"), descriptor.get("max")
    if low is None and high is None:
        return ""
    low_open = descriptor.get("min_exclusive", False)
    high_open = descriptor.get("max_exclusive", False)
    if high is None:
        return f"{'>' if low_open else '>='} {low}"
    if low is None:
        return f"{'<' if high_open else '<='} {high}"
    return f"{'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}"


def _copy(value):
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return value


def _with_defaults(entries: dict[str, dict[str, Any]], defaults: dict[str, Any]):
    result = {}
    for name, entry in entries.items():
        entry = _copy(entry)
        if entry["type"] == "group":
            entry["fields"] = _with_defaults(entry["fields"], defaults.get(name, {}))
        elif not entry.get("required") and "default_label" not in entry and name in defaults:
            entry["default"] = _copy(defaults[name])
        result[name] = entry
    return result


def _describe_type(annotation) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        arguments = get_args(annotation)
        options = [_describe_type(item) for item in arguments if item is not type(None)]
        entry = options[0] if len(options) == 1 else {"type": "any_of", "options": options}
        if len(options) != len(arguments):
            entry["nullable"] = True
        return entry
    if origin is Literal:
        return {"type": "choice", "choices": list(get_args(annotation))}
    if origin in (tuple, list):
        arguments = [item for item in get_args(annotation) if item is not Ellipsis]
        item = _describe_type(arguments[0]) if arguments else {"type": "any"}
        return {"type": "list", "items": item}
    if origin is dict:
        arguments = get_args(annotation)
        values = _describe_type(arguments[1]) if len(arguments) == 2 else {"type": "any"}
        return {"type": "map", "values": values}
    for python_type, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if annotation is python_type:
            return {"type": name}
    return {"type": "any"}


def _numeric_targets(entry: dict[str, Any]):
    if entry["type"] in ("int", "float"):
        yield entry
    elif entry["type"] == "list":
        yield from _numeric_targets(entry["items"])
    elif entry["type"] == "any_of":
        for option in entry["options"]:
            yield from _numeric_targets(option)


def _list_targets(entry: dict[str, Any]):
    if entry["type"] == "list":
        yield entry
    elif entry["type"] == "any_of":
        for option in entry["options"]:
            yield from _list_targets(option)


def _apply_constraints(entry: dict[str, Any], spec: ParameterSpec) -> None:
    if spec.choices is not None:
        nullable = entry.get("nullable", False)
        entry.clear()
        entry.update({"type": "choice", "choices": list(spec.choices)})
        if nullable:
            entry["nullable"] = True
        return
    for target in _numeric_targets(entry):
        if spec.gt is not None:
            target.update({"min": spec.gt, "min_exclusive": True})
        elif spec.ge is not None:
            target["min"] = spec.ge
        if spec.lt is not None:
            target.update({"max": spec.lt, "max_exclusive": True})
        elif spec.le is not None:
            target["max"] = spec.le
    for target in _list_targets(entry):
        if spec.min_length is not None:
            target["min_length"] = spec.min_length
        if spec.max_length is not None:
            target["max_length"] = spec.max_length


def _validate_group(instance, entries: dict[str, dict[str, Any]], prefix: str) -> None:
    for name, entry in entries.items():
        path = f"{prefix}.{name}"
        value = getattr(instance, name)
        if entry["type"] == "group":
            if not entry.get("custom"):
                _validate_group(value, entry["fields"], path)
        elif entry.get("validated", True):
            _check(value, entry, path)


def _in_range(value, entry: dict[str, Any]) -> bool:
    low, high = entry.get("min"), entry.get("max")
    if low is not None and not (value > low if entry.get("min_exclusive") else value >= low):
        return False
    if high is not None and not (value < high if entry.get("max_exclusive") else value <= high):
        return False
    return True


def _expectation(entry: dict[str, Any]) -> str:
    kind = entry["type"]
    if kind in ("int", "float"):
        noun = "an integer" if kind == "int" else "a number"
        limits = range_text(entry)
        return f"{noun} {'in ' if limits[:1] in ('[', '(') else ''}{limits}".strip()
    if kind == "bool":
        return "a boolean"
    if kind == "str":
        return "a string"
    if kind == "choice":
        return "one of " + ", ".join(repr(choice) for choice in entry["choices"])
    if kind == "list":
        return f"a sequence of {_expectation(entry['items'])}"
    if kind == "map":
        return "a dictionary"
    if kind == "any_of":
        return " or ".join(_expectation(option) for option in entry["options"])
    return "valid"


def _fail(path: str, entry: dict[str, Any], value) -> None:
    suffix = " or None" if entry.get("nullable") else ""
    raise ValueError(f"{path} must be {_expectation(entry)}{suffix}, got {value!r}.")


def _check(value, entry: dict[str, Any], path: str) -> None:
    if value is None:
        if entry.get("nullable"):
            return
        _fail(path, entry, value)
    kind = entry["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            _fail(path, entry, value)
    elif kind in ("int", "float"):
        expected = Integral if kind == "int" else Real
        if isinstance(value, bool) or not isinstance(value, expected):
            _fail(path, entry, value)
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value!r}.")
        if not _in_range(value, entry):
            _fail(path, entry, value)
    elif kind == "str":
        if not isinstance(value, str):
            _fail(path, entry, value)
    elif kind == "choice":
        if not any(type(value) is type(choice) and value == choice for choice in entry["choices"]):
            _fail(path, entry, value)
    elif kind == "list":
        if not isinstance(value, (list, tuple)):
            _fail(path, entry, value)
        low, high = entry.get("min_length"), entry.get("max_length")
        if (low is not None and len(value) < low) or (high is not None and len(value) > high):
            limits = f"{low}" if low == high else f"{low if low is not None else 0} to {high if high is not None else 'any number of'}"
            raise ValueError(f"{path} must have {limits} elements, got {len(value)}.")
        for index, item in enumerate(value):
            _check(item, entry["items"], f"{path}[{index}]")
    elif kind == "map":
        if not isinstance(value, dict):
            _fail(path, entry, value)
        known = entry.get("keys", {})
        for key, item in value.items():
            _check(item, known.get(key, entry.get("values", {"type": "any"})), f"{path}[{key!r}]")
    elif kind == "any_of":
        for option in entry["options"]:
            try:
                _check(value, option, path)
                return
            except ValueError:
                continue
        _fail(path, entry, value)


__all__ = [
    "ParameterSpec",
    "check_value",
    "describe_class",
    "describe_instance",
    "iter_parameters",
    "param",
    "range_text",
    "validate_parameters",
]
