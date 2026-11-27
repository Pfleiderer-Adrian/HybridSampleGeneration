"""Tests for prior and posterior generation across every registered model."""

import unittest

import numpy as np
import torch

from hybrid_sample_generator.generation.registry import MODEL_REGISTRY, registered_model_names
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator
from tests.pipeline.test_trained_generators import SMALL_PARAMETERS


def _build(name):
    spec = MODEL_REGISTRY[name]
    shape = (1, *((8,) * spec.spatial_dims)) if spec.spatial_dims == 2 else (1, 4, 4, 4)
    model = spec.build(
        SMALL_PARAMETERS[name.rsplit("_", 1)[0]],
        in_channels=1,
        num_anomaly_classes=1 if spec.uses_masks else None,
    )
    model.warmup(shape, device="cpu")
    image = np.random.default_rng(1).random(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(0, *[slice(1, 3) for _ in shape[1:]])] = 1
    return spec, model, shape, {"img": image, "ori_mask": mask}


def _generate(model, sample, mode, **kwargs):
    kwargs.setdefault("device", "cpu")
    return model.generate(sample, mode=mode, **kwargs)


class GenerationModeTests(unittest.TestCase):
    def test_prior_sampling_returns_image_and_mask_for_every_model(self):
        for name in sorted(registered_model_names()):
            with self.subTest(model=name):
                spec, model, shape, sample = _build(name)
                torch.manual_seed(0)
                image, mask = _generate(model, sample, "prior", variation_strength=1.0)

                self.assertIsInstance(image, np.ndarray)
                self.assertEqual(image.shape, shape)
                self.assertEqual(tuple(mask.shape[-len(shape) + 1:]), shape[1:])
                self.assertTrue(np.isfinite(image).all())
                # Clamping is on by default.
                self.assertGreaterEqual(float(image.min()), 0.0)
                self.assertLessEqual(float(image.max()), 1.0)
                if spec.uses_masks:
                    # Without transforms, the target mask is the mask of the sample.
                    np.testing.assert_array_equal(mask, sample["ori_mask"])

    def test_prior_sampling_follows_the_random_state_and_strength(self):
        for name in sorted(registered_model_names()):
            with self.subTest(model=name):
                _, model, _, sample = _build(name)
                torch.manual_seed(3)
                first, _ = _generate(model, sample, "prior", variation_strength=2.0, clamp_01=False)
                torch.manual_seed(3)
                repeated, _ = _generate(model, sample, "prior", variation_strength=2.0, clamp_01=False)
                torch.manual_seed(4)
                other, _ = _generate(model, sample, "prior", variation_strength=2.0, clamp_01=False)
                zero_a, _ = _generate(model, sample, "prior", variation_strength=0.0, clamp_01=False)
                zero_b, _ = _generate(model, sample, "prior", variation_strength=0.0, clamp_01=False)

                np.testing.assert_array_equal(first, repeated)
                self.assertFalse(np.array_equal(first, other))
                np.testing.assert_array_equal(zero_a, zero_b)
                with self.assertRaises(ValueError):
                    _generate(model, sample, "prior", variation_strength=-1.0)

    def test_posterior_sampling_follows_the_random_state_and_strength(self):
        for name in sorted(registered_model_names()):
            with self.subTest(model=name):
                _, model, shape, sample = _build(name)
                torch.manual_seed(5)
                first, _ = _generate(model, sample, "posterior", variation_strength=3.0, clamp_01=False)
                torch.manual_seed(5)
                repeated, _ = _generate(model, sample, "posterior", variation_strength=3.0, clamp_01=False)
                torch.manual_seed(6)
                other, _ = _generate(model, sample, "posterior", variation_strength=3.0, clamp_01=False)
                mean_a, _ = _generate(model, sample, "posterior", variation_strength=0.0, clamp_01=False)
                mean_b, _ = _generate(model, sample, "posterior", variation_strength=0.0, clamp_01=False)

                self.assertEqual(first.shape, shape)
                np.testing.assert_array_equal(first, repeated)
                self.assertFalse(np.array_equal(first, other))
                np.testing.assert_array_equal(mean_a, mean_b)
                with self.assertRaises(ValueError):
                    _generate(model, sample, "posterior", variation_strength=-1.0)

    def test_tensor_output_and_mode_aliases(self):
        for name in sorted(registered_model_names()):
            with self.subTest(model=name):
                _, model, shape, sample = _build(name)
                for mode in ("prior", "prior_sampling", "posterior", "posterior_sampling", "img2img", "PRIOR"):
                    image, _ = _generate(model, sample, mode, return_torch=True)
                    self.assertIsInstance(image, torch.Tensor)
                    self.assertEqual(tuple(image.shape), shape)
                with self.assertRaisesRegex(ValueError, "Unknown generation mode"):
                    _generate(model, sample, "diffusion")

    def test_conditional_models_decode_the_requested_target_mask(self):
        for name in sorted(registered_model_names()):
            spec, model, shape, sample = _build(name)
            if not spec.uses_masks:
                continue
            with self.subTest(model=name):
                target = np.zeros(shape, dtype=np.uint8)
                target[(0, *[slice(0, 2) for _ in shape[1:]])] = 1
                explicit = dict(sample, tgt_mask=target)

                for mode in ("prior", "posterior"):
                    kwargs = {"posterior_skip_source": "original"} if (
                        mode == "posterior" and spec.supports_transformed_posterior_skips
                    ) else {}
                    _, returned = _generate(model, explicit, mode, **kwargs)
                    np.testing.assert_array_equal(returned, target)

                # A mask generator supplies the target mask when the sample has none.
                generator = TransformGenerator(
                    {"zoom": 1.0},
                    use_mask_transform=True,
                    transform_params={"zoom": {"min_zoom": 0.5, "max_zoom": 0.5}},
                    rng=np.random.default_rng(0),
                    anomaly_size=shape,
                )
                _, generated = _generate(model, sample, "prior", target_mask_generator=generator)
                self.assertEqual(generated.shape, shape)
                self.assertLess(int(np.count_nonzero(generated)), int(np.count_nonzero(sample["ori_mask"])))

    def test_conditional_models_require_a_dictionary_with_a_mask(self):
        for name in sorted(registered_model_names()):
            spec, model, _, sample = _build(name)
            if not spec.uses_masks:
                continue
            with self.subTest(model=name):
                with self.assertRaises(TypeError):
                    _generate(model, sample["img"], "prior")
                with self.assertRaises(KeyError):
                    _generate(model, {"img": sample["img"]}, "prior")
                with self.assertRaises(ValueError):
                    _generate(model, {"img": sample["img"]}, "posterior")

    def test_unconditioned_models_accept_the_supported_input_forms(self):
        for name in sorted(registered_model_names()):
            spec, model, shape, sample = _build(name)
            if spec.uses_masks:
                continue
            with self.subTest(model=name):
                image = sample["img"]
                reference, _ = _generate(model, sample, "posterior", variation_strength=0.0, clamp_01=False)
                for form in (
                    image,
                    torch.as_tensor(image),
                    (image, "name"),
                    [torch.as_tensor(image), 0],
                    {"x": image},
                    {"image": torch.as_tensor(image)},
                    {"inputs": image},
                ):
                    result, _ = _generate(model, form, "posterior", variation_strength=0.0, clamp_01=False)
                    np.testing.assert_allclose(result, reference, atol=1e-6)

                # Prior sampling needs no sample when the output shape is given.
                generated, mask = model.generate(
                    None, mode="prior", output_shape=shape[1:], device="cpu",
                )
                self.assertEqual(generated.shape, shape)
                self.assertEqual(tuple(mask.shape), shape[1:])
                with self.assertRaises(ValueError):
                    model.generate(None, mode="prior", device="cpu")
                with self.assertRaises(ValueError):
                    model.generate(None, mode="prior", output_shape=shape, device="cpu")

    def test_unconditioned_models_derive_the_mask_from_the_generated_image(self):
        for name in sorted(registered_model_names()):
            spec, model, shape, sample = _build(name)
            if spec.uses_masks:
                continue
            with self.subTest(model=name):
                generator = TransformGenerator(use_mask_transform=False, background_threshold=0.5)
                image, mask = _generate(
                    model, sample, "posterior", variation_strength=0.0, clamp_01=False,
                    target_mask_generator=generator,
                )
                projection = image.max(axis=0)
                threshold = image.min() + 0.5 * (image.max() - image.min())
                np.testing.assert_array_equal(mask, (projection > threshold).astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
