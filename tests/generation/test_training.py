"""Tests for training augmentation, metrics, and parameter sampling."""

import unittest

import torch

from hybrid_sample_generator.configuration.augmentation import AugmentationConfiguration
from hybrid_sample_generator.generation.training.augmentation import (
    RandomSpatialOffset,
    apply_training_offset_augmentation,
)
from hybrid_sample_generator.generation.training.metrics import (
    average_metric_dicts,
    metric_value,
)
from hybrid_sample_generator.generation.training.optuna import sample_model_params
from hybrid_sample_generator.generation.model_settings import (
    Choice,
    FloatRange,
    IntRange,
    SearchSpace,
)
from hybrid_sample_generator.generation.vae.resnet.configuration import Config


class _Trial:
    def suggest_int(self, _name, low, _high, **_kwargs):
        return low

    def suggest_float(self, _name, low, _high, **_kwargs):
        return low

    def suggest_categorical(self, _name, choices):
        return choices[0]


class TrainingModuleTests(unittest.TestCase):
    def test_metric_helpers_select_and_average_scalars(self):
        self.assertEqual(metric_value({"loss": torch.tensor(2.0)}), 2.0)
        self.assertEqual(
            average_metric_dicts([{"loss": 1.0}, {"loss": 3.0}]),
            {"loss": 2.0},
        )

    def test_disabled_offset_returns_original_dataset(self):
        dataset = [torch.ones(1, 2, 2)]
        config = AugmentationConfiguration(random_offset_enabled=False)
        self.assertIs(apply_training_offset_augmentation(dataset, config), dataset)

    def test_constant_image_is_not_shifted(self):
        image = torch.ones(1, 3, 3)
        self.assertIs(RandomSpatialOffset()(image), image)

    def test_hyperparameter_sampling_changes_only_searched_values(self):
        parameters = Config(
            n_res_blocks=7,
            z_channels=48,
            recon_weight=25.0,
            recon_loss="mse",
        )
        search = SearchSpace(parameters)
        search.n_res_blocks = IntRange(2, 4)
        search.z_channels = Choice((16, 32))
        search.recon_weight = FloatRange(1.0, 10.0)
        sampled = sample_model_params(_Trial(), parameters, search)
        self.assertIsNot(sampled, parameters)
        self.assertEqual(sampled.n_res_blocks, 2)
        self.assertEqual(sampled.z_channels, 16)
        self.assertEqual(sampled.recon_weight, 1.0)
        self.assertEqual(sampled.recon_loss, "mse")
        self.assertEqual(parameters.n_res_blocks, 7)


if __name__ == "__main__":
    unittest.main()
