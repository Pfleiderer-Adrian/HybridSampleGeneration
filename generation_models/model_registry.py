from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from torch import nn

from generation_models.VAEs.VAE_ConvNeXt.VAE_ConvNeXt_2D import ConvNeXtVAE2D, Config as ConvNeXtVAE2D_Config
from generation_models.VAEs.VAE_ConvNeXt.VAE_ConvNeXt_3D import ConvNeXtVAE3D, Config as ConvNeXtVAE3D_Config
from generation_models.VAEs.VAE_ResNet.VAE_ResNet_2D import ResNetVAE2D, Config as ResNetVAE2D_Config
from generation_models.VAEs.VAE_ResNet.VAE_ResNet_3D import ResNetVAE3D, Config as ResNetVAE3D_Config
from generation_models.VAEs.cVAE_ConvNeXt.cVAE_ConvNeXt_2D import ConvNeXtcVAE2D, Config as ConvNeXtcVAE2D_Config
from generation_models.VAEs.cVAE_ConvNeXt.cVAE_ConvNeXt_3D import ConvNeXtcVAE3D, Config as ConvNeXtcVAE3D_Config
from generation_models.Diffusion.LatentDiffusionLoRA.LatentDiffusionLoRA_2D import (
    LatentDiffusionLoRA2D,
    Config as LatentDiffusionLoRA2D_Config,
)
from generation_models.VAEs.VAE_ConvNeXt.configuration import (
    get_convnext_vae_2d_configuration,
    get_convnext_vae_3d_configuration,
)
from generation_models.VAEs.VAE_ResNet.configuration import (
    get_resnet_vae_2d_configuration,
    get_resnet_vae_3d_configuration,
)
from generation_models.VAEs.cVAE_ConvNeXt.configuration import (
    get_convnext_cvae_2d_configuration,
    get_convnext_cvae_3d_configuration,
)
from generation_models.Diffusion.LatentDiffusionLoRA.configuration import (
    get_latent_diffusion_lora_2d_configuration,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_cls: Type[nn.Module]
    config_cls: Type
    config_factory: Callable[[int], object]
    uses_masks: bool = False
    trainable: bool = True
    generative: bool = True
    spatial_dims: int | None = None

    def build(self, params: dict):
        return self.model_cls(self.config_cls(**params))

    def build_configuration(self, in_channels: int):
        return self.config_factory(in_channels)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "VAE_ResNet_3D": ModelSpec(
        "VAE_ResNet_3D",
        ResNetVAE3D,
        ResNetVAE3D_Config,
        get_resnet_vae_3d_configuration,
        spatial_dims=3,
    ),
    "VAE_ResNet_2D": ModelSpec(
        "VAE_ResNet_2D",
        ResNetVAE2D,
        ResNetVAE2D_Config,
        get_resnet_vae_2d_configuration,
        spatial_dims=2,
    ),
    "VAE_ConvNeXt_3D": ModelSpec(
        "VAE_ConvNeXt_3D",
        ConvNeXtVAE3D,
        ConvNeXtVAE3D_Config,
        get_convnext_vae_3d_configuration,
        spatial_dims=3,
    ),
    "VAE_ConvNeXt_2D": ModelSpec(
        "VAE_ConvNeXt_2D",
        ConvNeXtVAE2D,
        ConvNeXtVAE2D_Config,
        get_convnext_vae_2d_configuration,
        spatial_dims=2,
    ),
    "cVAE_ConvNeXt_3D": ModelSpec(
        "cVAE_ConvNeXt_3D",
        ConvNeXtcVAE3D,
        ConvNeXtcVAE3D_Config,
        get_convnext_cvae_3d_configuration,
        uses_masks=True,
        spatial_dims=3,
    ),
    "cVAE_ConvNeXt_2D": ModelSpec(
        "cVAE_ConvNeXt_2D",
        ConvNeXtcVAE2D,
        ConvNeXtcVAE2D_Config,
        get_convnext_cvae_2d_configuration,
        uses_masks=True,
        spatial_dims=2,
    ),
    "LatentDiffusionLoRA_2D": ModelSpec(
        "LatentDiffusionLoRA_2D",
        LatentDiffusionLoRA2D,
        LatentDiffusionLoRA2D_Config,
        get_latent_diffusion_lora_2d_configuration,
        uses_masks=True,
        spatial_dims=2,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}. Supported models: {list(MODEL_REGISTRY)}") from exc


def registered_model_names() -> list[str]:
    return list(MODEL_REGISTRY)
