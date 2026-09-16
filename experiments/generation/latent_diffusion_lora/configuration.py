"""Configuration defaults for latent-diffusion LoRA generation."""

from experiments.generation.latent_diffusion_lora.model_2d import Config


DIFFUSION_INPUT_ARTEFACTS = ("img", "fname", "ori_mask")


def get_latent_diffusion_lora_2d_configuration(in_channels):
    return Config(in_channels=in_channels)
