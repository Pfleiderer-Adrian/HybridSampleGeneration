"""Behavioral tests for the actual generator training loop."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from hybrid_sample_generator.configuration.training import TrainingConfiguration
from hybrid_sample_generator.generation.interfaces import StepOutput
from hybrid_sample_generator.generation.training.loop import EarlyStoppingTracker, run_epoch, train
from hybrid_sample_generator.generation.training.metrics import step_scheduler


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(10.))
        self.saved = []
        self.epochs = []
        self.validation_grad_enabled = []

    def warmup(self, *args, **kwargs):
        pass

    def configure_optimizers(self, config):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config.learning_rate)
        return self.optimizer, None

    def on_epoch_start(self, epoch, **kwargs):
        self.epochs.append(epoch)

    def training_step(self, batch, index, config):
        return StepOutput(loss=(self.weight * batch).square().mean(), metrics={})

    def validation_step(self, batch, index, config):
        self.validation_grad_enabled.append(torch.is_grad_enabled())
        loss = torch.tensor([3., 2., 4., 5.][self.epochs[-1]])
        return StepOutput(loss=loss, metrics={'selection': loss})

    def save_checkpoint(self, path, **state):
        self.saved.append(state)
        torch.save(self.state_dict(), path)


class TrainingLoopTests(unittest.TestCase):
    def test_training_updates_parameters_and_clips_gradient(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        config = TrainingConfiguration(gradient_clip_norm=.5)
        metrics = run_epoch(model, [torch.ones(1)], optimizer, config, 'cpu', training=True)
        self.assertAlmostEqual(model.weight.item(), 9.95, places=5)
        self.assertLessEqual(model.weight.grad.norm().item(), .50001)
        self.assertEqual(metrics['loss'], 100.)

    def test_validation_does_not_update_parameters_or_enable_gradients(self):
        model = TinyModel()
        model.epochs = [0]
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        run_epoch(model, [torch.ones(1)], optimizer, TrainingConfiguration(), 'cpu', training=False)
        self.assertEqual(model.weight.item(), 10.)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(model.validation_grad_enabled, [False])
        self.assertFalse(model.training)

    def test_early_stopping_delta_and_reset(self):
        tracker = EarlyStoppingTracker(patience=2, delta=.1)
        self.assertFalse(tracker.step(1.))
        self.assertFalse(tracker.step(.95))
        self.assertFalse(tracker.step(.8))
        self.assertFalse(tracker.step(.75))
        self.assertTrue(tracker.step(.74))

    def test_best_checkpoint_scheduler_and_early_stopping(self):
        model = TinyModel()
        config = TrainingConfiguration(epochs=4, learning_rate=.1, early_stopping={'patience': 1, 'delta': 0.}, lr_scheduler={'patience': 0, 'factor': .5})
        with tempfile.TemporaryDirectory() as root, patch('torch.cuda.is_available', return_value=False):
            path = Path(root) / 'best.pt'
            training, validation, epoch, value = train(model, [torch.ones(1)], [torch.ones(1)], config, anomaly_size=(1, 2, 2), best_model_path=path)
            self.assertTrue(path.is_file())
            self.assertEqual([state['epoch'] for state in model.saved], [1, 2])
            self.assertEqual(validation, [3., 2., 4.])
            self.assertEqual((epoch, value), (2, 2.))
            self.assertEqual(len(training), 3)
            self.assertAlmostEqual(model.optimizer.param_groups[0]['lr'], .05)
            self.assertEqual(model.validation_grad_enabled, [False] * 3)


class SchedulerTests(unittest.TestCase):
    def test_step_scheduler_advances_epoch_scheduler_without_metric(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        optimizer.step()
        step_scheduler(scheduler, 999.)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .05)


class NonFiniteTrainingTests(unittest.TestCase):
    def test_non_finite_losses_fail_before_optimizer_update(self):
        for training in (True, False):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(training=training, value=value):
                    model = TinyModel()
                    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                    output = StepOutput(loss=model.weight * value, metrics={})
                    with patch.object(model, "training_step" if training else "validation_step", return_value=output), self.assertRaisesRegex(ValueError, "non-finite loss"):
                        run_epoch(model, [torch.ones(1)], optimizer, TrainingConfiguration(), "cpu", training=training)
                    self.assertEqual(model.weight.item(), 10.)
                    self.assertIsNone(model.weight.grad)

    def test_non_finite_metric_is_rejected_before_parameter_updates(self):
        for training in (True, False):
            with self.subTest(training=training):
                model = TinyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                output = StepOutput(loss=model.weight.square(), metrics={"selection": float("nan")})
                with patch.object(model, "training_step" if training else "validation_step", return_value=output), self.assertRaisesRegex(ValueError, "non-finite training metrics"):
                    run_epoch(model, [torch.ones(1)], optimizer, TrainingConfiguration(), "cpu", training=training)
                self.assertEqual(model.weight.item(), 10.)
                self.assertIsNone(model.weight.grad)
