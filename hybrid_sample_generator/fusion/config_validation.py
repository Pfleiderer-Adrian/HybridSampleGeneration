"""Validation shared by the concrete fusion parameter dataclasses."""

import math
from dataclasses import fields
from numbers import Integral, Real
from types import UnionType
from typing import get_args, get_origin, get_type_hints


def validate_parameters(config, *, positive=(), nonnegative=(), unit_interval=()):
    hints = get_type_hints(type(config))
    for field in fields(config):
        value = getattr(config, field.name)
        annotation = hints[field.name]
        types = get_args(annotation) if get_origin(annotation) is UnionType else (annotation,)
        valid = any(
            (expected is type(None) and value is None)
            or (expected is bool and isinstance(value, bool))
            or (expected is int and isinstance(value, Integral) and not isinstance(value, bool))
            or (expected is float and isinstance(value, Real) and not isinstance(value, bool))
            or (expected is str and isinstance(value, str))
            for expected in types
        )
        if not valid:
            raise ValueError(f"fusion.parameters.{field.name} must be {annotation}, got {value!r}.")
        if isinstance(value, Real) and not math.isfinite(value):
            raise ValueError(f"fusion.parameters.{field.name} must be finite.")
    for names, predicate, description in (
        (positive, lambda value: value > 0, "positive"),
        (nonnegative, lambda value: value >= 0, "non-negative"),
        (unit_interval, lambda value: 0 <= value <= 1, "in [0, 1]"),
    ):
        for name in names:
            value = getattr(config, name)
            if value is not None and not predicate(value):
                raise ValueError(f"fusion.parameters.{name} must be {description}.")
    border = config.fusion_normalization_border_width
    if border is not None and border < -1:
        raise ValueError("fusion.parameters.fusion_normalization_border_width must be None, -1 or non-negative.")
