"""Tests for generator workflow ownership and deterministic execution."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.service import (
    GenerationService,
    _select_trial,
    _validate_generated_variant,
)
from hybrid_sample_generator.randomness import seeded_random


class _LoadedModel:
    def __init__(self):
        self.device = None
        self.warmup_call = None
        self.checkpoint = None

    def to(self, device):
        self.device = device
        return self

    def warmup(self, shape, **kwargs):
        self.warmup_call = (shape, kwargs)
        return self

    def load_checkpoint(self, path):
        self.checkpoint = path


class GenerationServiceTests(unittest.TestCase):
    def test_generate_requires_a_model_before_reading_records(self):
        config = Configuration(
            "service-test",
            "VAE_ResNet_2D",
            (1, 8, 8),
            study_folder="/tmp/service-test",
        )
        service = GenerationService(
            config,
            Mock(),
            Mock(),
            Mock(),
        )

        with self.assertRaisesRegex(ValueError, "No generator model loaded"):
            service.generate()

    def test_training_prepares_conditional_class_count_and_dataset(self):
        config = Configuration(
            "conditional-service-test",
            "cVAE_ConvNeXt_2D",
            (1, 8, 8),
            study_folder="/tmp/conditional-service-test",
        )
        repository = Mock()
        repository.list_real_anomalies.return_value = [
            SimpleNamespace(metadata={"label": 2}),
            SimpleNamespace(metadata={"label": 5}),
        ]
        dataset = object()
        datasets = Mock()
        datasets.real_anomalies.return_value = dataset
        service = GenerationService(
            config,
            repository,
            Mock(),
            datasets,
        )

        with patch(
            "hybrid_sample_generator.generation.service.optimize"
        ) as optimize:
            service.train(3)

        optimize.assert_called_once_with(3, config, dataset)
        self.assertEqual(
            config.model.parameters.min["num_anomaly_classes"],
            5,
        )
        self.assertEqual(
            config.model.parameters.max["num_anomaly_classes"],
            5,
        )
        datasets.real_anomalies.assert_called_once_with(
            return_artifacts=config.model.parameters.input_artefacts,
            load_to_ram=True,
            dtype=torch.float32,
        )

    def test_load_selects_trial_builds_model_and_owns_it(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "load-service-test",
                "VAE_ResNet_2D",
                (1, 8, 8),
                study_folder=str(Path(root) / "study"),
            )
            trial = SimpleNamespace(
                number=7,
                user_attrs={
                    "model_name": "VAE_ResNet_2D",
                    "params": {"in_channels": 1},
                    "model_path": "model.pth",
                },
            )
            study = SimpleNamespace(
                best_trial=trial,
                get_trials=lambda: [trial],
            )
            model = _LoadedModel()
            spec = Mock()
            spec.build.return_value = model
            service = GenerationService(
                config,
                Mock(),
                Mock(),
                Mock(),
            )
            database = Path(root) / "trials.db"

            with (
                patch(
                    "hybrid_sample_generator.generation.service.optuna.load_study",
                    return_value=study,
                ) as load_study,
                patch(
                    "hybrid_sample_generator.generation.service.get_model_spec",
                    return_value=spec,
                ),
                patch("torch.cuda.is_available", return_value=False),
            ):
                loaded = service.load(database, trial_id=7)

            self.assertIs(loaded, model)
            self.assertIs(service.model, model)
            load_study.assert_called_once_with(
                study_name=config.study.name,
                storage="sqlite:///" + str(database),
            )
            spec.build.assert_called_once_with({"in_channels": 1})
            self.assertEqual(model.device, torch.device("cpu"))
            self.assertEqual(
                model.warmup_call[0],
                config.extraction.anomaly_size,
            )
            self.assertEqual(model.checkpoint, "model.pth")

    def test_trial_selection_supports_best_latest_and_explicit_ids(self):
        first = SimpleNamespace(number=1)
        latest = SimpleNamespace(number=4)
        study = SimpleNamespace(
            best_trial=first,
            get_trials=lambda: [latest, first],
        )

        self.assertIs(_select_trial(study, -1), first)
        self.assertIs(_select_trial(study, -2), latest)
        self.assertIs(_select_trial(study, 1), first)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            _select_trial(study, 99)

    def test_generated_variant_validation_rejects_unaligned_shapes(self):
        image = np.zeros((1, 8, 8), dtype=np.float32)
        valid_mask = np.zeros((1, 8, 8), dtype=np.uint8)

        validated_image, validated_mask = _validate_generated_variant(
            torch.from_numpy(image),
            valid_mask,
            image,
        )
        np.testing.assert_array_equal(validated_image, image)
        np.testing.assert_array_equal(validated_mask, valid_mask)

        with self.assertRaisesRegex(ValueError, "Generated image shape"):
            _validate_generated_variant(
                np.zeros((1, 7, 8)),
                valid_mask,
                image,
            )
        with self.assertRaisesRegex(ValueError, "Generated mask"):
            _validate_generated_variant(
                image,
                np.zeros((2, 7, 8)),
                image,
            )

    def test_seeded_random_is_repeatable_and_restores_global_state(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        with seeded_random(17):
            first = (
                random.random(),
                np.random.random(),
                torch.rand(1),
            )
        with seeded_random(17):
            second = (
                random.random(),
                np.random.random(),
                torch.rand(1),
            )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(
            np.random.get_state()[1],
            numpy_state[1],
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            torch_state,
        )


if __name__ == "__main__":
    unittest.main()
