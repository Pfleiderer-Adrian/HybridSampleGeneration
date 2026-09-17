"""Real checkpoint and multichannel architecture regressions."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.randomness import seeded_random
from hybrid_sample_generator.configuration.training import TrainingConfiguration
from hybrid_sample_generator.generation.training.loop import run_epoch
from tests.generation.test_models import MODEL_CASES


class ModelRegressionTests(unittest.TestCase):
    def test_multichannel_models_and_multiple_conditional_classes(self):
        for name, params, shape, conditional in MODEL_CASES:
            with self.subTest(model=name):
                model = MODEL_REGISTRY[name].build(params, in_channels=2, num_anomaly_classes=2 if conditional else None).eval()
                image = torch.rand((1, 2, *shape[2:]))
                mask = torch.ones((1, 1, *shape[2:]), dtype=torch.long)
                mask[..., :2] = 2
                with torch.no_grad():
                    output = model(image, mask) if conditional else model(image)
                self.assertEqual(output['recon'].shape, image.shape)
                self.assertTrue(torch.isfinite(output['recon']).all())

    def test_real_checkpoints_reload_and_generate_same_seeded_output(self):
        for name, params, shape, conditional in MODEL_CASES:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as root:
                spec = MODEL_REGISTRY[name]
                kwargs = dict(in_channels=shape[1], num_anomaly_classes=1 if conditional else None)
                original = spec.build(params, **kwargs).eval()
                original.warmup(shape[1:])
                sample = {'img': np.ones(shape[1:], dtype=np.float32), 'ori_mask': np.ones(shape[1:], dtype=np.uint8)}
                with seeded_random(41):
                    expected, _ = original.generate(sample, mode='posterior', clamp_01=False)
                path = Path(root) / 'model.pth'
                original.save_checkpoint(path)
                restored = spec.build(params, **kwargs).eval()
                restored.warmup(shape[1:])
                restored.load_checkpoint(path)
                with seeded_random(41):
                    actual, _ = restored.generate(sample, mode='posterior', clamp_01=False)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_registered_models_train_and_validate_through_common_loop(self):
        config = TrainingConfiguration(learning_rate=.001)
        for name, params, shape, conditional in MODEL_CASES:
            with self.subTest(model=name):
                model = MODEL_REGISTRY[name].build(
                    params,
                    in_channels=shape[1],
                    num_anomaly_classes=1 if conditional else None,
                )
                model.warmup(shape[1:])
                optimizer, _ = model.configure_optimizers(config)
                batch = {"img": torch.rand((2, *shape[1:]))}
                if conditional:
                    batch["ori_mask"] = torch.ones_like(batch["img"], dtype=torch.long)
                before = [parameter.detach().clone() for parameter in model.parameters()]
                training = run_epoch(model, [batch], optimizer, config, "cpu", training=True)
                self.assertTrue(np.isfinite(training["total"]))
                self.assertTrue(any(not torch.equal(old, current) for old, current in zip(before, model.parameters())))
                trained = [parameter.detach().clone() for parameter in model.parameters()]
                validation = run_epoch(model, [batch], optimizer, config, "cpu", training=False)
                self.assertTrue(np.isfinite(validation["total"]))
                self.assertTrue(all(torch.equal(old, current) for old, current in zip(trained, model.parameters())))

    def test_odd_anisotropic_shapes_are_preserved_by_forward_and_generation(self):
        for name, params, shape, conditional in MODEL_CASES:
            with self.subTest(model=name):
                spatial = (7, 9) if len(shape) == 4 else (3, 5, 7)
                model = MODEL_REGISTRY[name].build(params, in_channels=1, num_anomaly_classes=1 if conditional else None).eval()
                image = torch.rand((1, 1, *spatial))
                mask = torch.ones_like(image, dtype=torch.long)
                with torch.no_grad():
                    output = model(image, mask) if conditional else model(image)
                self.assertEqual(tuple(output["recon"].shape), tuple(image.shape))
                model.warmup((1, *spatial))
                sample = {"img": image[0].numpy(), "ori_mask": mask[0].numpy()}
                for mode in ("posterior", "prior"):
                    generated, _ = model.generate(sample, mode=mode, clamp_01=False)
                    self.assertEqual(generated.shape, (1, *spatial))
                    self.assertTrue(np.isfinite(generated).all())
