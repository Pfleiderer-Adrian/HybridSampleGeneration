"""Tests for registered generator model architectures."""

import unittest

import torch

from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.generation.vae.conditional_convnext import model_2d as conditional_2d
from hybrid_sample_generator.generation.vae.conditional_convnext import model_3d as conditional_3d
from hybrid_sample_generator.generation.vae.conditional_convnext import spade_2d, spade_3d
from hybrid_sample_generator.generation.vae.convnext import layers_2d as convnext_layers_2d
from hybrid_sample_generator.generation.vae.convnext import layers_3d as convnext_layers_3d


MODEL_CASES = (
    (
        "VAE_ResNet_2D",
        {
            "in_channels": 1,
            "n_res_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
            "use_multires_skips": False,
        },
        (1, 1, 8, 8),
        False,
    ),
    (
        "VAE_ResNet_3D",
        {
            "in_channels": 1,
            "n_res_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
            "use_multires_skips": False,
        },
        (1, 1, 4, 4, 4),
        False,
    ),
    (
        "VAE_ConvNeXt_2D",
        {
            "in_channels": 1,
            "n_res_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
            "drop_path_rate": 0.0,
            "dropout": 0.0,
        },
        (1, 1, 8, 8),
        False,
    ),
    (
        "VAE_ConvNeXt_3D",
        {
            "in_channels": 1,
            "n_res_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
        },
        (1, 1, 4, 4, 4),
        False,
    ),
    (
        "cVAE_ConvNeXt_2D",
        {
            "in_channels": 1,
            "num_anomaly_classes": 1,
            "n_res_blocks": 1,
            "n_spade_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
            "drop_path_rate": 0.0,
            "dropout": 0.0,
        },
        (1, 1, 8, 8),
        True,
    ),
    (
        "cVAE_ConvNeXt_3D",
        {
            "in_channels": 1,
            "num_anomaly_classes": 1,
            "n_res_blocks": 1,
            "n_spade_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
        },
        (1, 1, 4, 4, 4),
        True,
    ),
)


class ModelCompatibilityTests(unittest.TestCase):
    def test_registered_models_forward_and_reload_state_dict_strictly(self):
        for name, params, input_shape, conditional in MODEL_CASES:
            with self.subTest(model=name):
                spec = MODEL_REGISTRY[name]
                model = spec.build(params).eval()
                image = torch.randn(input_shape)
                args = (image, torch.zeros(input_shape, dtype=torch.long)) if conditional else (image,)

                with torch.no_grad():
                    output = model(*args)

                self.assertEqual(output["recon"].shape, image.shape)
                self.assertEqual(output["x_ref"].shape, image.shape)
                self.assertEqual(output["mu"].shape, (input_shape[0], params["bottleneck_dim"]))
                self.assertEqual(output["logvar"].shape, output["mu"].shape)

                restored = spec.build(params).eval()
                with torch.no_grad():
                    restored(*args)
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(list(restored.state_dict()), list(model.state_dict()))

    def test_experimental_diffusion_model_is_not_registered(self):
        self.assertNotIn("LatentDiffusionLoRA_2D", MODEL_REGISTRY)


    def test_conditional_models_reuse_base_convnext_layers(self):
        self.assertIs(spade_2d.ConvNeXtBlock2D, convnext_layers_2d.ConvNeXtBlock2D)
        self.assertIs(conditional_2d.ConvNeXtUNetEncoder2D, convnext_layers_2d.ConvNeXtUNetEncoder2D)
        self.assertIs(spade_3d.ConvNeXtBlock3D, convnext_layers_3d.ConvNeXtBlock3D)
        self.assertIs(conditional_3d.ConvNeXtUNetEncoder3D, convnext_layers_3d.ConvNeXtUNetEncoder3D)

    def test_conditional_spade_blocks_support_stochastic_depth(self):
        block_2d = spade_2d.ConvNeXtSPADEBlock2D(4, 1, drop_path=0.1)
        block_3d = spade_3d.ConvNeXtSPADEBlock3D(4, 1, drop_path=0.1)

        self.assertIsInstance(block_2d.drop_path, convnext_layers_2d.DropPath)
        self.assertIsInstance(block_3d.drop_path, convnext_layers_3d.DropPath)


if __name__ == "__main__":
    unittest.main()
