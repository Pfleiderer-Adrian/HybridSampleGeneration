"""Network-free smoke tests for the experimental diffusion prototype."""

import unittest

from experiments.generation.latent_diffusion_lora.configuration import (
    DIFFUSION_INPUT_ARTEFACTS,
    get_latent_diffusion_lora_2d_configuration,
)
from experiments.generation.latent_diffusion_lora.model_2d import (
    Config,
    LatentDiffusionLoRA2D,
)


class LatentDiffusionLoRAPrototypeTests(unittest.TestCase):
    def test_configuration_and_construction_do_not_load_remote_weights(self):
        model = LatentDiffusionLoRA2D(Config(in_channels=1))
        settings = get_latent_diffusion_lora_2d_configuration(1)

        self.assertIsNone(model.pipeline)
        self.assertEqual(settings.input_artefacts, DIFFUSION_INPUT_ARTEFACTS)
        self.assertEqual(settings.min["in_channels"], 1)


if __name__ == "__main__":
    unittest.main()
