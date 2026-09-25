"""Optional latent reconstruction loss for ConvNeXt VAEs."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hybrid_sample_generator.generation.registry import MODEL_REGISTRY


CONVNEXT_MODELS = (
    "VAE_ConvNeXt_2D",
    "VAE_ConvNeXt_3D",
    "cVAE_ConvNeXt_2D",
    "cVAE_ConvNeXt_3D",
)


def build_model(name, **overrides):
    spec = MODEL_REGISTRY[name]
    parameters = {
        "n_levels": 1,
        "n_res_blocks": 1,
        "z_channels": 4,
        "bottleneck_dim": 3,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
        "skip_dropout_p": 0.0,
        **overrides,
    }
    if spec.uses_masks:
        parameters["n_spade_blocks"] = 1
    return spec.build(
        parameters,
        in_channels=1,
        num_anomaly_classes=1 if spec.uses_masks else None,
    )


def forward_model(model, name, image):
    if MODEL_REGISTRY[name].uses_masks:
        return model(image, torch.ones_like(image, dtype=torch.long))
    return model(image)


class LatentReconstructionTests(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        invalid = (
            ("latent_recon_weight", -0.1),
            ("latent_recon_weight", float("nan")),
            ("latent_recon_weight", float("inf")),
            ("latent_recon_noise_scale", 0.0),
            ("latent_recon_noise_scale", -0.1),
            ("latent_recon_noise_scale", float("inf")),
            ("latent_recon_image_noise_std", -0.1),
            ("latent_recon_image_noise_std", float("nan")),
        )
        for name in CONVNEXT_MODELS:
            for field, value in invalid:
                with self.subTest(model=name, field=field, value=value):
                    with self.assertRaises(ValueError):
                        build_model(name, **{field: value})

    def test_non_numeric_settings_are_rejected(self):
        for name in CONVNEXT_MODELS:
            for field in (
                "latent_recon_weight",
                "latent_recon_noise_scale",
                "latent_recon_image_noise_std",
            ):
                with self.subTest(model=name, field=field):
                    with self.assertRaises(TypeError):
                        build_model(name, **{field: "0.1"})

    def test_zero_weight_skips_extra_passes(self):
        for name in CONVNEXT_MODELS:
            with self.subTest(model=name):
                model = build_model(name, latent_recon_weight=0.0)
                shape = (8, 8) if MODEL_REGISTRY[name].spatial_dims == 2 else (4, 4, 4)
                image = torch.rand((1, 1, *shape))
                calls = {"encoder": 0, "decoder": 0}
                hooks = [
                    model.encoder.register_forward_hook(
                        lambda *_: calls.__setitem__("encoder", calls["encoder"] + 1)
                    ),
                    model.decoder.register_forward_hook(
                        lambda *_: calls.__setitem__("decoder", calls["decoder"] + 1)
                    ),
                ]
                try:
                    output = forward_model(model, name, image)
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertEqual(calls, {"encoder": 1, "decoder": 1})
                self.assertNotIn("latent_recon", output)
                output["reconstruction_mask"] = torch.ones_like(image)
                losses = model.loss(output)
                self.assertEqual(losses["latent_recon"].item(), 0.0)
                self.assertEqual(losses["latent_recon_weighted"].item(), 0.0)

    def test_active_cycle_adds_weighted_loss_and_gradients(self):
        for name in CONVNEXT_MODELS:
            with self.subTest(model=name):
                model = build_model(
                    name,
                    latent_recon_weight=0.2,
                    latent_recon_noise_scale=0.5,
                    latent_recon_image_noise_std=0.0,
                )
                shape = (8, 8) if MODEL_REGISTRY[name].spatial_dims == 2 else (4, 4, 4)
                image = torch.rand((1, 1, *shape))
                calls = {"encoder": 0, "decoder": 0}
                hooks = [
                    model.encoder.register_forward_hook(
                        lambda *_: calls.__setitem__("encoder", calls["encoder"] + 1)
                    ),
                    model.decoder.register_forward_hook(
                        lambda *_: calls.__setitem__("decoder", calls["decoder"] + 1)
                    ),
                ]
                try:
                    output = forward_model(model, name, image)
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertEqual(calls, {"encoder": 2, "decoder": 2})
                self.assertFalse(output["latent_target"].requires_grad)
                self.assertEqual(output["latent_recon"].shape, output["mu"].shape)
                output["reconstruction_mask"] = torch.ones_like(image)
                losses = model.loss(output)
                torch.testing.assert_close(
                    losses["latent_recon"],
                    F.smooth_l1_loss(
                        output["latent_recon"], output["latent_target"], beta=1.0
                    ),
                )
                torch.testing.assert_close(
                    losses["total"],
                    losses["recon_weighted"]
                    + losses["kl_weighted"]
                    + losses["latent_recon_weighted"],
                )
                losses["total"].backward()
                self.assertTrue(torch.isfinite(model.fc_decode.weight.grad).all())
                self.assertTrue(torch.isfinite(model.fc_mu.weight.grad).all())

    def test_image_noise_is_training_only(self):
        for name in ("VAE_ConvNeXt_2D", "cVAE_ConvNeXt_2D"):
            with self.subTest(model=name):
                model = build_model(
                    name,
                    latent_recon_weight=0.1,
                    latent_recon_image_noise_std=0.03,
                )
                image = torch.rand((1, 1, 8, 8))
                real_randn_like = torch.randn_like
                with patch("torch.randn_like", wraps=real_randn_like) as randn_like:
                    forward_model(model.train(), name, image)
                    self.assertEqual(randn_like.call_count, 3)
                    randn_like.reset_mock()
                    forward_model(model.eval(), name, image)
                    self.assertEqual(randn_like.call_count, 2)

    def test_conditional_cycle_reencodes_with_target_mask(self):
        for name in ("cVAE_ConvNeXt_2D", "cVAE_ConvNeXt_3D"):
            with self.subTest(model=name):
                model = build_model(name, latent_recon_weight=0.1)
                shape = (8, 8) if MODEL_REGISTRY[name].spatial_dims == 2 else (4, 4, 4)
                image = torch.rand((1, 1, *shape))
                original_mask = torch.zeros_like(image, dtype=torch.long)
                target_mask = torch.ones_like(image, dtype=torch.long)
                encoder_inputs = []
                hook = model.encoder.register_forward_pre_hook(
                    lambda _, args: encoder_inputs.append(args[0].detach())
                )
                try:
                    model(image, original_mask, target_mask)
                finally:
                    hook.remove()
                self.assertEqual(len(encoder_inputs), 2)
                torch.testing.assert_close(
                    encoder_inputs[0][:, 1:],
                    torch.zeros_like(encoder_inputs[0][:, 1:]),
                )
                torch.testing.assert_close(
                    encoder_inputs[1][:, 1:],
                    torch.ones_like(encoder_inputs[1][:, 1:]),
                )


if __name__ == "__main__":
    unittest.main()
