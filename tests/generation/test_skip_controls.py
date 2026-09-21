"""Per-level ConvNeXt skip controls."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.generation.vae.conditional_convnext.spade import ConvNeXtSPADEUNetDecoder
from hybrid_sample_generator.generation.vae.convnext.layers import ConvNeXtUNetDecoder


CONVNEXT_MODELS = (
    "VAE_ConvNeXt_2D",
    "VAE_ConvNeXt_3D",
    "cVAE_ConvNeXt_2D",
    "cVAE_ConvNeXt_3D",
)


def build_model(name, **overrides):
    spec = MODEL_REGISTRY[name]
    params = {
        "n_levels": 2,
        "n_res_blocks": 1,
        "z_channels": 4,
        "bottleneck_dim": 3,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
        **overrides,
    }
    if spec.uses_masks:
        params["n_spade_blocks"] = 0
    return spec.build(
        params,
        in_channels=1,
        num_anomaly_classes=1 if spec.uses_masks else None,
    )


class SkipControlTests(unittest.TestCase):
    def test_list_lengths_and_value_ranges_are_validated(self):
        invalid = (
            ("skip_alphas", [0.5]),
            ("skip_alphas", [0.1, 0.2, 0.3]),
            ("skip_dropout_ps", [0.5]),
            ("skip_dropout_ps", [0.1, 0.2, 0.3]),
            ("skip_alphas", [-0.1, 0.5]),
            ("skip_alphas", [1.1, 0.5]),
            ("skip_alphas", [float("nan"), 0.5]),
            ("skip_alphas", [float("inf"), 0.5]),
            ("skip_alpha", -0.1),
            ("skip_alpha", 1.1),
            ("skip_alpha", float("nan")),
            ("skip_dropout_ps", [0.5, 1.1]),
            ("skip_dropout_p", -0.1),
        )
        for name in CONVNEXT_MODELS:
            for field, value in invalid:
                with self.subTest(model=name, field=field, value=value):
                    with self.assertRaises(ValueError):
                        build_model(name, **{field: value})

    def test_scalar_fallback_and_valid_per_level_values(self):
        for name in CONVNEXT_MODELS:
            with self.subTest(model=name):
                model = build_model(name, skip_alpha=0.25, skip_dropout_p=0.5)
                self.assertEqual(model.decoder.skip_alphas, [0.25, 0.25])
                self.assertEqual(model.decoder.skip_dropout_ps, [0.5, 0.5])
                model = build_model(
                    name,
                    skip_alpha=0.25,
                    skip_alphas=[0.0, 1.0],
                    skip_dropout_p=0.5,
                    skip_dropout_ps=[1.0, 0.0],
                )
                self.assertEqual(model.decoder.skip_alphas, [0.0, 1.0])
                self.assertEqual(model.decoder.skip_dropout_ps, [1.0, 0.0])

    def test_decoder_applies_alphas_in_encoder_order(self):
        for spatial_dims in (2, 3):
            for decoder_type in (ConvNeXtUNetDecoder, ConvNeXtSPADEUNetDecoder):
                with self.subTest(spatial_dims=spatial_dims, decoder=decoder_type.__name__):
                    kwargs = {
                        "out_channels": 1,
                        "n_res_blocks": 0,
                        "n_levels": 2,
                        "z_channels": 4,
                        "skip_alpha": 1.0,
                        "skip_alphas": [0.0, 0.25],
                        "spatial_dims": spatial_dims,
                    }
                    if decoder_type is ConvNeXtSPADEUNetDecoder:
                        kwargs.update(n_spade_blocks=0, num_anomaly_classes=1)
                    decoder = decoder_type(**kwargs).eval()
                    decoder.set_skips([
                        torch.ones((1, 8, *((8,) * spatial_dims))),
                        torch.ones((1, 16, *((4,) * spatial_dims))),
                    ])
                    fuse_inputs = []
                    hooks = [
                        layer.register_forward_pre_hook(
                            lambda _layer, inputs: fuse_inputs.append(inputs[0].detach())
                        )
                        for layer in decoder.fuse
                    ]
                    try:
                        z = torch.zeros((1, 4, *((2,) * spatial_dims)))
                        if decoder_type is ConvNeXtSPADEUNetDecoder:
                            mask = torch.zeros((1, 1, *((8,) * spatial_dims)))
                            decoder(z, mask)
                        else:
                            decoder(z)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    self.assertEqual(len(fuse_inputs), 2)
                    torch.testing.assert_close(
                        fuse_inputs[0][:, 16:],
                        torch.full_like(fuse_inputs[0][:, 16:], 0.25),
                    )
                    torch.testing.assert_close(
                        fuse_inputs[1][:, 8:],
                        torch.zeros_like(fuse_inputs[1][:, 8:]),
                    )

    def test_posterior_uses_effective_per_level_alphas(self):
        for name in CONVNEXT_MODELS:
            spec = MODEL_REGISTRY[name]
            spatial_shape = (8, 8) if spec.spatial_dims == 2 else (4, 4, 4)
            sample = {
                "img": np.ones((1, *spatial_shape), dtype=np.float32),
                "ori_mask": np.zeros((1, *spatial_shape), dtype=np.uint8),
            }
            for alphas, expect_skips in (([0.0, 0.25], True), ([0.0, 0.0], False)):
                with self.subTest(model=name, alphas=alphas):
                    model = build_model(
                        name,
                        skip_alpha=0.0 if expect_skips else 1.0,
                        skip_alphas=alphas,
                        skip_dropout_p=0.0,
                    )
                    model.warmup((1, *spatial_shape))
                    model.generate(
                        sample,
                        mode="posterior",
                        variation_strength=0.0,
                        clamp_01=False,
                    )
                    self.assertEqual(model.decoder._skips is not None, expect_skips)


if __name__ == "__main__":
    unittest.main()
