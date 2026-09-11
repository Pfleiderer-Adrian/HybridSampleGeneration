"""Configuration models for generation-time data augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GLOBAL_MASK_TRANSFORMS = frozenset({"zoom", "elastic", "stretch", "rotate"})
LOCAL_MASK_TRANSFORMS = frozenset(
    {"local_dilate", "local_stretch", "local_rotate", "local_elastic"}
)


@dataclass
class MaskTransformConfiguration:
    """Configuration for generation-time target-mask transformations."""

    use_mask_transform: bool = True
    mask_transform_probs: dict[int | str, Any] = field(default_factory=dict)
    mask_transform_params: dict[int | str, dict[str, Any]] = field(
        default_factory=dict
    )
    priorities: list[int] | tuple[int, ...] | None = None
    local_as_global: bool = False
    padding_factor: int = 2

    def setGlobalParam(self, transform_name: str, probability=None, **params):
        if transform_name not in GLOBAL_MASK_TRANSFORMS:
            raise ValueError(f"{transform_name!r} is not a global transform.")
        return self._set_transform_config(transform_name, probability, params)

    def setClassParam(
        self, class_id: int, transform_name: str, probability=None, **params
    ):
        if transform_name not in LOCAL_MASK_TRANSFORMS:
            raise ValueError(f"{transform_name!r} is not a local transform.")
        return self._set_transform_config(
            transform_name, probability, params, class_id=class_id
        )

    def setAllClassParams(self, transform_name: str, probability=None, **params):
        if transform_name not in LOCAL_MASK_TRANSFORMS:
            raise ValueError(f"{transform_name!r} is not a local transform.")
        return self._set_transform_config(transform_name, probability, params)

    def _set_transform_config(
        self,
        transform_name: str,
        probability,
        params: dict,
        class_id: int | None = None,
    ):
        if probability is not None:
            if class_id is None:
                self.mask_transform_probs[transform_name] = probability
            else:
                self.mask_transform_probs.setdefault(class_id, {})[
                    transform_name
                ] = probability

        if params:
            if class_id is None:
                self.mask_transform_params.setdefault(transform_name, {}).update(params)
            else:
                self.mask_transform_params.setdefault(class_id, {}).setdefault(
                    transform_name, {}
                ).update(params)

        return self


@dataclass
class AugmentationConfiguration:
    """Training offsets and target-mask transformations."""

    mask_transforms: MaskTransformConfiguration = field(
        default_factory=MaskTransformConfiguration
    )
    random_offset_enabled: bool = True
    random_offset_max_fraction: float = 1.0
    random_offset_foreground_threshold: float = 0.001

    def validate(self) -> None:
        if not 0.0 <= float(self.random_offset_max_fraction) <= 1.0:
            raise ValueError("augmentation.random_offset_max_fraction must be in [0, 1].")
        if float(self.random_offset_foreground_threshold) < 0:
            raise ValueError("augmentation.random_offset_foreground_threshold must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_transforms": asdict(self.mask_transforms),
            "random_offset_enabled": self.random_offset_enabled,
            "random_offset_max_fraction": self.random_offset_max_fraction,
            "random_offset_foreground_threshold": self.random_offset_foreground_threshold,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AugmentationConfiguration":
        values = dict(values)
        mask_values = dict(values.pop("mask_transforms", {}))
        values["mask_transforms"] = MaskTransformConfiguration(**mask_values)
        return cls(**values)

