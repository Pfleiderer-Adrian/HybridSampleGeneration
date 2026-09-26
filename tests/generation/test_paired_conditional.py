"""Paired source-to-target conditional VAE behavior."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from hybrid_sample_generator.configuration.augmentation import (
    MaskTransformConfiguration,
)
from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.generation.training.paired_targets import PairedTargetDataset


SMALL_PARAMETERS = {
    "n_res_blocks": 1,
    "n_spade_blocks": 1,
    "n_levels": 1,
    "z_channels": 4,
    "bottleneck_dim": 3,
    "drop_path_rate": 0.0,
    "dropout": 0.0,
    "recon_loss": "mse",
    "recon_weight": 1.0,
    "beta_kl_start": 0.5,
    "beta_kl_max": 0.5,
    "free_bits": 0.0,
}


class PairedTargetDatasetTests(unittest.TestCase):
    @staticmethod
    def _sample():
        image = torch.zeros((1, 8, 8))
        image[:, 2:5, 2:5] = 1.0
        mask = torch.zeros((1, 8, 8), dtype=torch.long)
        mask[:, 2:5, 2:5] = 1
        return {"img": image, "ori_mask": mask, "fname": "sample"}

    def _dataset(self, identity_probability, *, deterministic=True):
        transforms = MaskTransformConfiguration(use_mask_transform=False)
        transforms.setGlobalParam("rotate", probability=1.0, max_rotation=10.0)
        return PairedTargetDataset(
            [self._sample()],
            transforms,
            anomaly_size=(1, 8, 8),
            background_threshold=0.01,
            identity_probability=identity_probability,
            seed=7,
            deterministic=deterministic,
        )

    def test_identity_pair_copies_without_aliasing_the_source(self):
        dataset = self._dataset(1.0)
        sample = dataset[0]

        self.assertTrue(sample["pair_is_identity"])
        torch.testing.assert_close(sample["tgt_img"], sample["img"])
        torch.testing.assert_close(sample["tgt_mask"], sample["ori_mask"])
        self.assertIsNot(sample["tgt_img"], sample["img"])
        self.assertIsNot(sample["tgt_mask"], sample["ori_mask"])

    def test_validation_pair_is_deterministic_and_jointly_transformed(self):
        dataset = self._dataset(0.0)
        first = dataset[0]
        second = dataset[0]

        self.assertFalse(first["pair_is_identity"])
        torch.testing.assert_close(first["tgt_img"], second["tgt_img"])
        torch.testing.assert_close(first["tgt_mask"], second["tgt_mask"])
        self.assertEqual(first["tgt_img"].shape, first["img"].shape)
        self.assertEqual(first["tgt_mask"].shape, first["ori_mask"].shape)


class PairedConditionalModelTests(unittest.TestCase):
    @staticmethod
    def _model(dims=2):
        return MODEL_REGISTRY[f"paired_cVAE_ConvNeXt_{dims}D"].build(
            SMALL_PARAMETERS,
            in_channels=1,
            num_anomaly_classes=1,
        )

    def test_training_uses_target_image_mask_and_includes_kl(self):
        model = self._model()
        source = torch.zeros((1, 1, 8, 8))
        source_mask = torch.zeros((1, 1, 8, 8), dtype=torch.long)
        target = torch.ones_like(source)
        target_mask = torch.ones_like(source_mask)
        batch = {
            "img": source,
            "ori_mask": source_mask,
            "tgt_img": target,
            "tgt_mask": target_mask,
        }

        output = model._forward_from_batch(batch)
        torch.testing.assert_close(output["x_ref"], target)
        torch.testing.assert_close(
            model._reconstruction_mask_from_batch(batch), target_mask
        )
        output["reconstruction_mask"] = target_mask
        losses = model.loss(output)

        expected_kl = 0.5 * (
            output["mu"].pow(2) + output["logvar"].exp() - 1.0 - output["logvar"]
        ).sum(dim=1).mean()
        torch.testing.assert_close(losses["kl_raw"], expected_kl)
        torch.testing.assert_close(losses["kl_weighted"], expected_kl * 0.5)

    def test_posterior_uses_exactly_one_encoder_pass_and_no_skip_api(self):
        model = self._model().eval()
        model.warmup((1, 8, 8))
        sample = {
            "img": np.zeros((1, 8, 8), dtype=np.float32),
            "ori_mask": np.zeros((1, 8, 8), dtype=np.uint8),
        }

        with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encode:
            image, mask = model.generate(
                sample,
                mode="posterior",
                variation_strength=0.0,
                clamp_01=False,
                device="cpu",
            )

        self.assertEqual(encode.call_count, 1)
        self.assertFalse(hasattr(model.decoder, "set_skips"))
        self.assertEqual(image.shape, sample["img"].shape)
        self.assertEqual(mask.shape, sample["ori_mask"].shape)

    def test_registry_marks_only_the_new_models_for_paired_training(self):
        for dims in (2, 3):
            spec = MODEL_REGISTRY[f"paired_cVAE_ConvNeXt_{dims}D"]
            self.assertEqual(spec.training_target_mode, "paired")
            self.assertFalse(spec.supports_transformed_posterior_skips)
        self.assertEqual(
            MODEL_REGISTRY["cVAE_ConvNeXt_2D"].training_target_mode,
            "identity",
        )

    def test_both_dimensions_forward_and_reload_strictly(self):
        for dims, spatial in ((2, (8, 8)), (3, (4, 4, 4))):
            with self.subTest(dims=dims):
                model = self._model(dims).eval()
                source = torch.zeros((1, 1, *spatial))
                source_mask = torch.zeros((1, 1, *spatial), dtype=torch.long)
                target = torch.ones_like(source)
                target_mask = torch.ones_like(source_mask)
                with torch.no_grad():
                    output = model(source, source_mask, target_mask, target)
                self.assertEqual(output["recon"].shape, source.shape)
                self.assertEqual(
                    output["mu"].shape,
                    (1, SMALL_PARAMETERS["bottleneck_dim"]),
                )

                restored = self._model(dims).eval()
                with torch.no_grad():
                    restored(source, source_mask, target_mask, target)
                restored.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
