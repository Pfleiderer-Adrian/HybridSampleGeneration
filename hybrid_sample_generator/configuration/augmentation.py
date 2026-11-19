"""Configuration models for generation-time data augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


GLOBAL_MASK_TRANSFORMS = frozenset({"zoom", "elastic", "stretch", "rotate"})
LOCAL_MASK_TRANSFORMS = frozenset(
    {"local_dilate", "local_stretch", "local_rotate", "local_elastic"}
)


@dataclass
class MaskTransformConfiguration:
    """Configuration for generation-time target-mask transformations."""

    use_mask_transform: bool = param(
        True,
        doc="Master switch for all mask transformations. When disabled, configured probabilities and parameters are retained but no mask transformation is applied.",
    )
    mask_transform_probs: dict[int | str, Any] = param(
        default_factory=dict,
        doc="Probabilities for global, local, and class-specific transforms.",
    )
    mask_transform_params: dict[int | str, dict[str, Any]] = param(
        default_factory=dict, doc="Parameter ranges for individual transforms.",
    )
    priorities: list[int] | tuple[int, ...] | None = param(
        None, doc="Class priority when transformed masks overlap.",
    )
    local_as_global: bool = param(
        False, doc="Apply local transforms jointly to all anomaly classes.",
    )
    padding_factor: int = param(2, doc="Scale factor for the temporary transform canvas.")

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

    # custom=True: the nested, class-specific layout needs a dedicated editor;
    # generic schema consumers skip it and it is not checked field by field.
    mask_transforms: MaskTransformConfiguration = param(
        default_factory=MaskTransformConfiguration, custom=True,
    )
    random_offset_enabled: bool = param(
        True, doc="Enable random offsets during training.",
    )
    random_offset_max_fraction: float = param(
        1.0, ge=0.0, le=1.0, doc="Maximum offset as a fraction of available space.",
    )
    random_offset_foreground_threshold: float = param(
        0.001, ge=0.0, doc="Foreground threshold used for training offsets.",
    )

    def validate(self) -> None:
        validate_parameters(self, "augmentation")

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

