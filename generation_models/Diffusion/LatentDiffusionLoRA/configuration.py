from dataclasses import asdict

from generation_models.Diffusion.LatentDiffusionLoRA.LatentDiffusionLoRA_2D import Config
from generation_models.model_configuration import ModelConfiguration


DIFFUSION_INPUT_ARTEFACTS = ("img", "fname", "ori_mask")


def get_latent_diffusion_lora_2d_configuration(in_channels):
    base = asdict(Config(in_channels=in_channels))
    return ModelConfiguration(
        base,
        base,
        input_artefacts=DIFFUSION_INPUT_ARTEFACTS,
    )
