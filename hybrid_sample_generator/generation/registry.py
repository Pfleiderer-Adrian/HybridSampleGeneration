"""Registry and construction helpers for generator models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from torch import nn

from hybrid_sample_generator.generation.model_settings import SearchSpace
from hybrid_sample_generator.generation.vae.conditional_convnext.configuration import (
    Config as ConditionalConvNeXtConfig,
    get_convnext_cvae_configuration,
    get_convnext_cvae_search,
)
from hybrid_sample_generator.generation.vae.conditional_convnext.model import (
    ConditionalConvNeXtVAE,
)
from hybrid_sample_generator.generation.vae.convnext.configuration import (
    Config as ConvNeXtConfig,
    get_convnext_vae_configuration,
    get_convnext_vae_search,
)
from hybrid_sample_generator.generation.vae.convnext.model import ConvNeXtVAE
from hybrid_sample_generator.generation.vae.paired_conditional_convnext.configuration import (
    Config as PairedConditionalConvNeXtConfig,
    get_paired_convnext_cvae_configuration,
    get_paired_convnext_cvae_search,
)
from hybrid_sample_generator.generation.vae.paired_conditional_convnext.model import (
    PairedConditionalConvNeXtVAE,
)
from hybrid_sample_generator.generation.vae.resnet.configuration import (
    Config as ResNetConfig,
    get_resnet_vae_configuration,
    get_resnet_vae_search,
)
from hybrid_sample_generator.generation.vae.resnet.model import ResNetVAE


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_cls: type[nn.Module]
    config_cls: type
    config_factory: Callable[[int], object]
    search_factory: Callable[[object, int], SearchSpace]
    spatial_dims: int
    input_artefacts: tuple[str, ...]
    uses_masks: bool = False
    supports_transformed_posterior_skips: bool = False
    training_target_mode: Literal["identity", "paired"] = "identity"

    def build(
        self,
        parameters,
        *,
        in_channels: int,
        num_anomaly_classes: int | None = None,
    ):
        config = (
            self.config_cls(**parameters)
            if isinstance(parameters, Mapping)
            else replace(parameters)
        )
        kwargs = {
            "spatial_dims": self.spatial_dims,
            "in_channels": int(in_channels),
        }
        if self.uses_masks:
            if num_anomaly_classes is None or int(num_anomaly_classes) <= 0:
                raise ValueError("Conditional models require num_anomaly_classes > 0.")
            kwargs["num_anomaly_classes"] = int(num_anomaly_classes)
        return self.model_cls(config, **kwargs)

    def build_configuration(self):
        return self.config_factory(self.spatial_dims)

    def build_search_space(self, parameters) -> SearchSpace:
        return self.search_factory(parameters, self.spatial_dims)


VAE_ARTEFACTS = ("img", "fname", "ori_mask")
CONDITIONAL_VAE_ARTEFACTS = VAE_ARTEFACTS

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "VAE_ResNet_3D": ModelSpec(
        "VAE_ResNet_3D", ResNetVAE, ResNetConfig,
        get_resnet_vae_configuration, get_resnet_vae_search,
        3, VAE_ARTEFACTS,
    ),
    "VAE_ResNet_2D": ModelSpec(
        "VAE_ResNet_2D", ResNetVAE, ResNetConfig,
        get_resnet_vae_configuration, get_resnet_vae_search,
        2, VAE_ARTEFACTS,
    ),
    "VAE_ConvNeXt_3D": ModelSpec(
        "VAE_ConvNeXt_3D", ConvNeXtVAE, ConvNeXtConfig,
        get_convnext_vae_configuration, get_convnext_vae_search,
        3, VAE_ARTEFACTS,
    ),
    "VAE_ConvNeXt_2D": ModelSpec(
        "VAE_ConvNeXt_2D", ConvNeXtVAE, ConvNeXtConfig,
        get_convnext_vae_configuration, get_convnext_vae_search,
        2, VAE_ARTEFACTS,
    ),
    "cVAE_ConvNeXt_3D": ModelSpec(
        "cVAE_ConvNeXt_3D", ConditionalConvNeXtVAE, ConditionalConvNeXtConfig,
        get_convnext_cvae_configuration, get_convnext_cvae_search,
        3, CONDITIONAL_VAE_ARTEFACTS,
        uses_masks=True,
        supports_transformed_posterior_skips=True,
    ),
    "cVAE_ConvNeXt_2D": ModelSpec(
        "cVAE_ConvNeXt_2D", ConditionalConvNeXtVAE, ConditionalConvNeXtConfig,
        get_convnext_cvae_configuration, get_convnext_cvae_search,
        2, CONDITIONAL_VAE_ARTEFACTS,
        uses_masks=True,
        supports_transformed_posterior_skips=True,
    ),
    "paired_cVAE_ConvNeXt_3D": ModelSpec(
        "paired_cVAE_ConvNeXt_3D",
        PairedConditionalConvNeXtVAE,
        PairedConditionalConvNeXtConfig,
        get_paired_convnext_cvae_configuration,
        get_paired_convnext_cvae_search,
        3,
        CONDITIONAL_VAE_ARTEFACTS,
        uses_masks=True,
        training_target_mode="paired",
    ),
    "paired_cVAE_ConvNeXt_2D": ModelSpec(
        "paired_cVAE_ConvNeXt_2D",
        PairedConditionalConvNeXtVAE,
        PairedConditionalConvNeXtConfig,
        get_paired_convnext_cvae_configuration,
        get_paired_convnext_cvae_search,
        2,
        CONDITIONAL_VAE_ARTEFACTS,
        uses_masks=True,
        training_target_mode="paired",
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model: {model_name}. Supported models: {list(MODEL_REGISTRY)}"
        ) from exc


def registered_model_names() -> list[str]:
    return list(MODEL_REGISTRY)
