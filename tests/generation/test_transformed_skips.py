"""Selection of original or transformed skips during conditional generation."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from hybrid_sample_generator.configuration.generation import GenerationConfiguration
from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.generation.service import GenerationService
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


def build_model(dims, skip_alpha=1.0):
    name = f"cVAE_ConvNeXt_{dims}D"
    spatial = (8, 8) if dims == 2 else (4, 4, 4)
    model = MODEL_REGISTRY[name].build(
        {
            "n_levels": 1,
            "n_res_blocks": 1,
            "n_spade_blocks": 1,
            "z_channels": 4,
            "bottleneck_dim": 3,
            "drop_path_rate": 0.0,
            "dropout": 0.0,
            "skip_dropout_p": 0.0,
            "skip_alpha": skip_alpha,
        },
        in_channels=1,
        num_anomaly_classes=1,
    )
    model.warmup((1, *spatial))
    return model, spatial


class TransformedSkipTests(unittest.TestCase):
    def test_transformed_mode_rejects_explicit_target_mask(self):
        for skip_alpha in (0.0, 1.0):
            with self.subTest(skip_alpha=skip_alpha):
                model, spatial = build_model(2, skip_alpha=skip_alpha)
                image = np.ones((1, *spatial), dtype=np.float32)
                mask = np.ones_like(image, dtype=np.uint8)
                for supplied_mask in ("sample", "argument"):
                    with self.subTest(supplied_mask=supplied_mask):
                        sample = {"img": image, "ori_mask": mask}
                        kwargs = {}
                        if supplied_mask == "sample":
                            sample["tgt_mask"] = mask
                        else:
                            kwargs["target_mask"] = mask
                        with self.assertRaisesRegex(ValueError, "Explicit target_mask"):
                            model.generate(
                                sample,
                                mode="posterior",
                                posterior_skip_source="transformed",
                                target_mask_generator=TransformGenerator(),
                                **kwargs,
                            )

    def test_transformed_mode_requires_generator(self):
        model, spatial = build_model(2)
        image = np.ones((1, *spatial), dtype=np.float32)
        mask = np.ones_like(image, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "target_mask_generator"):
            model.generate(
                {"img": image, "ori_mask": mask},
                mode="posterior",
                posterior_skip_source="transformed",
            )

    def test_original_mode_accepts_explicit_target_mask(self):
        model, spatial = build_model(2)
        image = np.ones((1, *spatial), dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.uint8)
        target_mask = np.ones_like(mask)
        _, returned_mask = model.generate(
            {"img": image, "ori_mask": mask, "tgt_mask": target_mask},
            mode="posterior",
            posterior_skip_source="original",
            variation_strength=0.0,
        )
        np.testing.assert_array_equal(returned_mask, target_mask)

    def test_automatic_pair_rejects_batched_input(self):
        model, spatial = build_model(2)
        image = np.ones((1, 1, *spatial), dtype=np.float32)
        mask = np.ones((1, 1, *spatial), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "unbatched"):
            model.generate(
                {"img": image, "ori_mask": mask},
                mode="posterior",
                posterior_skip_source="transformed",
                target_mask_generator=TransformGenerator(),
            )

    def test_zero_alphas_skip_the_second_encoder(self):
        model, spatial = build_model(2, skip_alpha=0.0)
        image = np.ones((1, *spatial), dtype=np.float32)
        mask = np.ones_like(image, dtype=np.uint8)
        with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder:
            model.generate(
                {"img": image, "ori_mask": mask},
                mode="posterior",
                posterior_skip_source="transformed",
                target_mask_generator=TransformGenerator(),
                variation_strength=0.0,
            )
        self.assertEqual(encoder.call_count, 1)

    def test_internal_pair_generates_mask_and_image_together(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                model, spatial = build_model(dims)
                image = np.ones((1, *spatial), dtype=np.float32)
                mask = np.zeros_like(image, dtype=np.uint8)
                mask[(0, *((slice(2, 4),) * dims))] = 1
                settings = {"zoom": 1.0}
                params = {"zoom": {"min_zoom": 0.8, "max_zoom": 0.8}}
                expected_mask, expected_image = TransformGenerator(
                    settings, transform_params=params, rng=np.random.default_rng(19)
                ).create_target_mask_and_transformed_image(mask, image)
                encoder_inputs = []
                hook = model.encoder.register_forward_pre_hook(
                    lambda _, args: encoder_inputs.append(args[0].detach().clone())
                )
                try:
                    result, actual_mask = model.generate(
                        {"img": image, "ori_mask": mask},
                        mode="posterior",
                        posterior_skip_source="transformed",
                        target_mask_generator=TransformGenerator(
                            settings, transform_params=params,
                            rng=np.random.default_rng(19),
                        ),
                        variation_strength=0.0,
                        clamp_01=False,
                        n=2,
                    )
                finally:
                    hook.remove()
                np.testing.assert_array_equal(actual_mask, expected_mask)
                self.assertEqual(result.shape, (2, *image.shape))
                self.assertEqual(model.decoder._skips[0].shape[0], 2)
                self.assertEqual(len(encoder_inputs), 2)
                np.testing.assert_allclose(
                    encoder_inputs[0][0, 0].cpu().numpy(), image[0], atol=1e-6
                )
                np.testing.assert_allclose(
                    encoder_inputs[1][0, 0].cpu().numpy(), expected_image[0], atol=1e-6
                )

    def test_service_passes_strategy_only_to_conditional_model(self):
        config = Configuration("skip-service", study_folder="/tmp/skip-service")
        config.model.set_model("cVAE_ConvNeXt_2D")
        config.generation.posterior_skip_source = "transformed"
        service = GenerationService(config, None, None, None)
        model = Mock()
        model.generate.return_value = (
            np.zeros((1, 8, 8), dtype=np.float32),
            np.zeros((1, 8, 8), dtype=np.uint8),
        )
        service._model = model
        service._generate_variant({}, np.zeros((1, 8, 8)), TransformGenerator())
        self.assertEqual(
            model.generate.call_args.kwargs["posterior_skip_source"],
            "transformed",
        )

        config.model.set_model("VAE_ConvNeXt_2D")
        with self.assertRaisesRegex(ValueError, "conditional ConvNeXt"):
            service._generate_variant({}, np.zeros((1, 8, 8)), TransformGenerator())

    def test_configuration_round_trip_and_model_restriction(self):
        config = Configuration("transformed-config")
        config.generation.posterior_skip_source = "transformed"
        restored = Configuration.from_dict(config.to_dict())
        self.assertEqual(restored.generation.posterior_skip_source, "transformed")
        config.model.set_model("VAE_ConvNeXt_2D")
        with self.assertRaisesRegex(ValueError, "conditional ConvNeXt"):
            config.validate()
        config.model.set_model("cVAE_ConvNeXt_2D")
        config.validate()

    def test_generation_configuration_validates_skip_source(self):
        config = GenerationConfiguration(posterior_skip_source="invalid")
        with self.assertRaisesRegex(ValueError, "posterior_skip_source"):
            config.validate()
        config = GenerationConfiguration(
            sampling_mode="prior", posterior_skip_source="transformed"
        )
        with self.assertRaisesRegex(ValueError, "posterior sampling"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
