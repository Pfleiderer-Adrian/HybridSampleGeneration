"""Tests for registered generator model architectures."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.generation.vae.conditional_convnext.model import (
    ConditionalConvNeXtVAE,
)
from hybrid_sample_generator.generation.vae.conditional_convnext.spade import (
    ConvNeXtSPADEBlock,
)
from hybrid_sample_generator.generation.vae.convnext.layers import (
    ConvNeXtUNetEncoder,
    DropPath,
)


MODEL_CASES = (
    (
        "VAE_ResNet_2D",
        {
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
                model = spec.build(
                    params,
                    in_channels=input_shape[1],
                    num_anomaly_classes=1 if conditional else None,
                ).eval()
                image = torch.randn(input_shape)
                args = (image, torch.zeros(input_shape, dtype=torch.long)) if conditional else (image,)

                with torch.no_grad():
                    output = model(*args)

                self.assertEqual(output["recon"].shape, image.shape)
                self.assertEqual(output["x_ref"].shape, image.shape)
                self.assertEqual(output["mu"].shape, (input_shape[0], params["bottleneck_dim"]))
                self.assertEqual(output["logvar"].shape, output["mu"].shape)

                restored = spec.build(
                    params,
                    in_channels=input_shape[1],
                    num_anomaly_classes=1 if conditional else None,
                ).eval()
                with torch.no_grad():
                    restored(*args)
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(list(restored.state_dict()), list(model.state_dict()))

    def test_registered_models_generate_in_both_dimensions(self):
        for name, params, input_shape, conditional in MODEL_CASES:
            with self.subTest(model=name):
                model = MODEL_REGISTRY[name].build(
                    params,
                    in_channels=input_shape[1],
                    num_anomaly_classes=1 if conditional else None,
                )
                shape = input_shape[1:]
                model.warmup(shape)
                image = np.random.default_rng(1).random(shape, dtype=np.float32)
                mask = np.zeros(shape, dtype=np.uint8)
                mask[(0, *[slice(1, 3) for _ in shape[1:]])] = 1
                sample = {"img": image, "ori_mask": mask}

                posterior, _ = model.generate(
                    sample,
                    mode="posterior",
                    variation_strength=0.0,
                    clamp_01=False,
                )
                prior, _ = model.generate(
                    sample,
                    mode="prior",
                    variation_strength=0.0,
                    clamp_01=False,
                )

                self.assertEqual(posterior.shape, shape)
                self.assertEqual(prior.shape, shape)

    def test_experimental_diffusion_model_is_not_registered(self):
        self.assertNotIn("LatentDiffusionLoRA_2D", MODEL_REGISTRY)


    def test_conditional_models_reuse_base_convnext_layers(self):
        for name, params, input_shape, conditional in MODEL_CASES:
            if not conditional:
                continue
            with self.subTest(model=name):
                model = MODEL_REGISTRY[name].build(
                    params,
                    in_channels=input_shape[1],
                    num_anomaly_classes=1 if conditional else None,
                )
                self.assertIsInstance(model, ConditionalConvNeXtVAE)
                self.assertIsInstance(model.encoder, ConvNeXtUNetEncoder)

    def test_conditional_spade_blocks_support_both_dimensions(self):
        block_2d = ConvNeXtSPADEBlock(4, 1, drop_path=0.1, spatial_dims=2)
        block_3d = ConvNeXtSPADEBlock(4, 1, drop_path=0.1, spatial_dims=3)
        self.assertIsInstance(block_2d.drop_path, DropPath)
        self.assertIsInstance(block_3d.drop_path, DropPath)
        self.assertEqual(block_2d.dwconv.weight.ndim, 4)
        self.assertEqual(block_3d.dwconv.weight.ndim, 5)


if __name__ == "__main__":
    unittest.main()
