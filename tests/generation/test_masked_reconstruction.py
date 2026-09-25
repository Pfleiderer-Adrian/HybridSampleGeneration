"""Mask-balanced reconstruction loss shared by all VAE variants."""

import unittest

import torch

from hybrid_sample_generator.generation.registry import MODEL_REGISTRY


class MaskedReconstructionTests(unittest.TestCase):
    @staticmethod
    def _model(*, model_name="VAE_ResNet_2D", in_channels=1, **overrides):
        parameters = {
            "n_res_blocks": 1,
            "n_levels": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
            "use_multires_skips": False,
            "recon_loss": "mse",
            "recon_weight": 1.0,
            "beta_kl_start": 0.0,
            "beta_kl_max": 0.0,
            "free_bits": 0.0,
            "foreground_weight": 0.8,
            "background_weight": 0.2,
            **overrides,
        }
        return MODEL_REGISTRY[model_name].build(
            parameters,
            in_channels=in_channels,
        )

    @staticmethod
    def _output(reconstruction, mask):
        batch_size = reconstruction.shape[0]
        return {
            "recon": reconstruction,
            "x_ref": torch.zeros_like(reconstruction),
            "mu": torch.zeros((batch_size, 3)),
            "logvar": torch.zeros((batch_size, 3)),
            "reconstruction_mask": mask,
        }

    def test_loss_balances_region_means_instead_of_pixel_counts(self):
        model = self._model()
        reconstruction = torch.ones((1, 1, 4, 4))
        reconstruction[..., 0, 0] = 2.0
        mask = torch.zeros_like(reconstruction)
        mask[..., 0, 0] = 1

        losses = model.loss(self._output(reconstruction, mask))

        torch.testing.assert_close(losses["foreground_recon"], torch.tensor(4.0))
        torch.testing.assert_close(losses["background_recon"], torch.tensor(1.0))
        torch.testing.assert_close(losses["recon"], torch.tensor(3.4))
        torch.testing.assert_close(losses["selection"], losses["recon"])

    def test_reconstruction_weights_are_relative(self):
        model = self._model(foreground_weight=8.0, background_weight=2.0)
        reconstruction = torch.ones((1, 1, 4, 4))
        reconstruction[..., 0, 0] = 2.0
        mask = torch.zeros_like(reconstruction)
        mask[..., 0, 0] = 1

        losses = model.loss(self._output(reconstruction, mask))

        torch.testing.assert_close(losses["recon"], torch.tensor(3.4))

    def test_only_regions_present_in_each_sample_contribute_to_weight_sum(self):
        model = self._model()
        reconstruction = torch.ones((2, 1, 2, 2))
        reconstruction[1] = 2.0
        mask = torch.zeros_like(reconstruction)
        mask[1] = 1

        losses = model.loss(self._output(reconstruction, mask))

        torch.testing.assert_close(losses["foreground_recon"], torch.tensor(4.0))
        torch.testing.assert_close(losses["background_recon"], torch.tensor(1.0))
        torch.testing.assert_close(losses["recon"], torch.tensor(2.5))

    def test_loss_collapses_multichannel_mask_to_spatial_foreground(self):
        model = self._model(in_channels=3)
        reconstruction = torch.ones((1, 3, 2, 2))
        reconstruction[..., 0, 0] = 2.0
        mask = torch.zeros_like(reconstruction)
        mask[:, 1, 0, 0] = 1

        losses = model.loss(self._output(reconstruction, mask))

        torch.testing.assert_close(losses["foreground_recon"], torch.tensor(4.0))
        torch.testing.assert_close(losses["background_recon"], torch.tensor(1.0))
        torch.testing.assert_close(losses["recon"], torch.tensor(3.4))

    def test_mask_balanced_loss_supports_3d_tensors(self):
        model = self._model(model_name="VAE_ResNet_3D")
        reconstruction = torch.ones((1, 1, 2, 2, 2))
        reconstruction[..., 0, 0, 0] = 2.0
        mask = torch.zeros_like(reconstruction)
        mask[..., 0, 0, 0] = 1

        losses = model.loss(self._output(reconstruction, mask))

        torch.testing.assert_close(losses["foreground_recon"], torch.tensor(4.0))
        torch.testing.assert_close(losses["background_recon"], torch.tensor(1.0))
        torch.testing.assert_close(losses["recon"], torch.tensor(3.4))

    def test_loss_rejects_sample_without_a_positive_weighted_region(self):
        model = self._model(foreground_weight=1.0, background_weight=0.0)
        reconstruction = torch.ones((1, 1, 2, 2))
        mask = torch.zeros_like(reconstruction)

        with self.assertRaisesRegex(ValueError, "positive weight"):
            model.loss(self._output(reconstruction, mask))

    def test_loss_requires_the_extracted_anomaly_mask(self):
        model = self._model()
        output = {
            "recon": torch.zeros((1, 1, 4, 4)),
            "x_ref": torch.zeros((1, 1, 4, 4)),
            "mu": torch.zeros((1, 3)),
            "logvar": torch.zeros((1, 3)),
        }

        with self.assertRaisesRegex(KeyError, "reconstruction_mask"):
            model.loss(output)

    def test_reconstruction_weights_are_validated(self):
        for overrides, error in (
            ({"foreground_weight": -0.1}, ValueError),
            ({"background_weight": float("nan")}, ValueError),
            ({"foreground_weight": 0.0, "background_weight": 0.0}, ValueError),
            ({"foreground_weight": "0.8"}, TypeError),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(error):
                    self._model(**overrides)

    def test_every_registered_vae_requests_the_original_mask(self):
        for name, spec in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                self.assertIn("ori_mask", spec.input_artefacts)


if __name__ == "__main__":
    unittest.main()
