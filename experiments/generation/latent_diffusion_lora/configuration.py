"""Configuration model for latent-diffusion LoRA generation."""

from dataclasses import asdict

from experiments.generation.latent_diffusion_lora.model_2d import Config
from hybrid_sample_generator.generation.model_settings import ModelHyperparameterSpace


DIFFUSION_INPUT_ARTEFACTS = ("img", "fname", "ori_mask")


def get_latent_diffusion_lora_2d_configuration(in_channels):
    base = asdict(Config(in_channels=in_channels))
    return ModelHyperparameterSpace(
        base,
        base,
        input_artefacts=DIFFUSION_INPUT_ARTEFACTS,
    )
