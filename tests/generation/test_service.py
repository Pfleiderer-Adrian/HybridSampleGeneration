"""Tests for generator workflow ownership and deterministic execution."""

import contextlib
import io
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.generation.service import (
    GenerationService,
    _select_trial,
    _validate_generated_variant,
    _warn_about_oversized_masks,
)
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
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
        config = Configuration("service-test", study_folder="/tmp/service-test")
        config.extraction.anomaly_size = (1, 8, 8)
        config.model.set_model("VAE_ResNet_2D")
        service = GenerationService(
            config,
            Mock(),
            Mock(),
            Mock(),
        )

        with self.assertRaisesRegex(ValueError, "No generator model loaded"):
            service.generate()

    def test_training_prepares_conditional_class_count_and_dataset(self):
        config = Configuration("conditional-service-test", study_folder="/tmp/conditional-service-test")
        config.extraction.anomaly_size = (1, 8, 8)
        config.model.set_model("cVAE_ConvNeXt_2D")
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
        config.training.num_trials = 3

        with patch(
            "hybrid_sample_generator.generation.service.optimize"
        ) as optimize:
            service.train()

        optimize.assert_called_once_with(
            3,
            config,
            dataset,
            num_anomaly_classes=5,
        )
        datasets.real_anomalies.assert_called_once_with(
            return_artifacts=("img", "fname", "ori_mask"),
            load_to_ram=True,
            dtype=torch.float32,
        )

    def test_load_selects_trial_builds_model_and_owns_it(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("load-service-test", study_folder=str(Path(root) / "study"))
            config.extraction.anomaly_size = (1, 8, 8)
            config.model.set_model("VAE_ResNet_2D")
            config.training.trial_selection = 7
            trial = SimpleNamespace(
                number=7,
                user_attrs={
                    "model_name": "VAE_ResNet_2D",
                    "params": {"z_channels": 32},
                    "in_channels": 1,
                    "num_anomaly_classes": None,
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
                loaded = service.load(database)

            self.assertIs(loaded, model)
            self.assertIs(service.model, model)
            load_study.assert_called_once_with(
                study_name=config.study.name,
                storage="sqlite:///" + str(database),
            )
            spec.build.assert_called_once_with(
                {"z_channels": 32},
                in_channels=1,
                num_anomaly_classes=None,
            )
            self.assertEqual(model.device, torch.device("cpu"))
            self.assertEqual(
                model.warmup_call[0],
                config.extraction.anomaly_size,
            )
            self.assertEqual(model.checkpoint, "model.pth")

    def _feedback_service(self, **feedback):
        config = Configuration("feedback-test", study_folder="/tmp/feedback-test")
        config.extraction.anomaly_size = (1, 8, 8)
        config.model.set_model("VAE_ResNet_2D")
        config.generation.feedback.enabled = True
        for name, value in feedback.items():
            setattr(config.generation.feedback, name, value)
        image = np.zeros((1, 8, 8), dtype=np.float32)
        model = Mock()
        model.generate.return_value = (image, np.zeros((1, 8, 8), dtype=np.uint8))
        return GenerationService(config, Mock(), Mock(), Mock(), model=model), model, image

    def test_feedback_relaxes_threshold_after_each_interval(self):
        service, model, image = self._feedback_service(
            similarity_threshold=0.8,
            threshold_relaxation_factor=0.5,
            relaxation_interval=2,
            max_attempts=10,
        )
        # Every attempt scores 0.5. The threshold falls from 0.8 to 0.4 after two
        # rejected attempts, so the third attempt is accepted.
        with patch("hybrid_sample_generator.generation.service.ssim_01", return_value=0.5):
            service._generate_variant({"img": image}, image, Mock())
        self.assertEqual(model.generate.call_count, 3)

    def test_feedback_keeps_threshold_within_an_interval(self):
        service, model, image = self._feedback_service(
            similarity_threshold=0.8,
            threshold_relaxation_factor=0.5,
            relaxation_interval=20,
            max_attempts=10,
        )
        with patch("hybrid_sample_generator.generation.service.ssim_01", return_value=0.5):
            service._generate_variant({"img": image}, image, Mock())
        self.assertEqual(model.generate.call_count, 10)

    def test_default_feedback_relaxes_within_the_default_attempt_limit(self):
        feedback = Configuration("defaults", study_folder="/tmp/defaults").generation.feedback
        self.assertLess(feedback.relaxation_interval, feedback.max_attempts)
        for invalid in (0, -1, True):
            feedback.relaxation_interval = invalid
            with self.assertRaisesRegex(ValueError, "relaxation_interval"):
                feedback.validate()

    def test_oversized_derived_masks_are_reported(self):
        with self.assertWarnsRegex(RuntimeWarning, "2 of 3 synthetic anomalies.*background_threshold = 0.01"):
            _warn_about_oversized_masks([1.0, 4.0, 9.0], 0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_about_oversized_masks([0.8, 1.2, 3.0], 0.01)
            _warn_about_oversized_masks([], 0.01)

    def test_generation_warns_only_for_models_that_derive_masks(self):
        image = np.zeros((1, 16, 16), dtype=np.float32)
        mask = np.zeros((1, 16, 16), dtype=np.uint8)
        image[:, 6:10, 6:10] = 1.0
        mask[:, 6:10, 6:10] = 1

        class WholeCutoutMask:
            def generate(self, sample, **_kwargs):
                cutout = np.asarray(sample["img"], dtype=np.float32)
                return cutout, np.ones_like(cutout, dtype=np.uint8)

        for model_name, expect_warning in (("VAE_ResNet_2D", True), ("cVAE_ConvNeXt_2D", False)):
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as root:
                config = Configuration("mask-warning", study_folder=str(Path(root) / "study"))
                config.extraction.anomaly_size = (1, 8, 8)
                config.extraction.min_coverage_ratio = 0.0
                config.extraction.add_background_noise = False
                config.extraction.normalization = None
                config.model.set_model(model_name)
                config.generation.variants_per_real_anomaly = 1
                generator = HybridDataGenerator(config, generator_model=WholeCutoutMask())
                silent = io.StringIO()
                with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                    generator.ingest_dataset([InputSample(image, mask, "anomaly")])
                    generator.extract_anomalies()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        generator.generate_synthetic_anomalies()
                reported = [w for w in caught if "synthetic anomalies have a mask" in str(w.message)]
                self.assertEqual(len(reported), 1 if expect_warning else 0)

    def _scored_feedback_service(self, scores, **feedback):
        config = Configuration("feedback-test", study_folder="/tmp/feedback-test")
        config.extraction.anomaly_size = (1, 8, 8)
        config.model.set_model("VAE_ResNet_2D")
        config.generation.feedback.enabled = True
        for name, value in feedback.items():
            setattr(config.generation.feedback, name, value)
        real = np.zeros((1, 8, 8), dtype=np.float32)
        # Attempt n returns an image filled with n, so the kept attempt is identifiable.
        outputs = iter(
            (np.full((1, 8, 8), index, dtype=np.float32), np.ones((1, 8, 8), dtype=np.uint8))
            for index in range(len(scores))
        )
        model = Mock()
        model.generate.side_effect = lambda *_args, **_kwargs: next(outputs)
        service = GenerationService(config, Mock(), Mock(), Mock(), model=model)
        return service, model, real, patch(
            "hybrid_sample_generator.generation.service.ssim_01", side_effect=list(scores)
        )

    def test_feedback_stops_at_the_first_variant_above_the_threshold(self):
        service, model, real, similarity = self._scored_feedback_service(
            [0.2, 0.5, 0.9, 0.95], similarity_threshold=0.8, max_attempts=4,
        )
        with similarity:
            image, mask = service._generate_variant({"img": real}, real, Mock())
        self.assertEqual(model.generate.call_count, 3)
        self.assertEqual(float(image[0, 0, 0]), 2.0)
        self.assertEqual(mask.shape, (1, 8, 8))

    def test_feedback_keeps_the_most_similar_variant_when_none_passes(self):
        service, model, real, similarity = self._scored_feedback_service(
            [0.2, 0.6, 0.4], similarity_threshold=0.8, max_attempts=3,
        )
        with similarity:
            image, _ = service._generate_variant({"img": real}, real, Mock())
        self.assertEqual(model.generate.call_count, 3)
        self.assertEqual(float(image[0, 0, 0]), 1.0)

    def test_feedback_rejects_variants_of_another_shape(self):
        service, model, real, similarity = self._scored_feedback_service(
            [0.9], similarity_threshold=0.8, max_attempts=1,
        )
        model.generate.side_effect = None
        model.generate.return_value = (np.zeros((1, 4, 4)), np.zeros((1, 4, 4)))
        with similarity, self.assertRaisesRegex(ValueError, "differs from"):
            service._generate_variant({"img": real}, real, Mock())

    def test_generation_passes_sampling_settings_to_the_model(self):
        config = Configuration("kwargs-test", study_folder="/tmp/kwargs-test")
        config.extraction.anomaly_size = (1, 8, 8)
        image = np.zeros((1, 8, 8), dtype=np.float32)
        for model_name, mode, expects_skip_source in (
            ("cVAE_ConvNeXt_2D", "posterior", True),
            ("cVAE_ConvNeXt_2D", "prior", False),
            ("VAE_ResNet_2D", "posterior", False),
        ):
            with self.subTest(model=model_name, mode=mode):
                config.model.set_model(model_name)
                config.generation.sampling_mode = mode
                config.generation.variation_strength = 0.7
                config.generation.clamp_output = True
                model = Mock()
                model.generate.return_value = (image, image)
                service = GenerationService(config, Mock(), Mock(), Mock(), model=model)
                mask_generator = Mock()

                service._generate_variant({"img": image}, image, mask_generator)

                kwargs = model.generate.call_args.kwargs
                self.assertEqual(kwargs["mode"], mode)
                self.assertEqual(kwargs["variation_strength"], 0.7)
                self.assertTrue(kwargs["clamp_01"])
                self.assertIs(kwargs["target_mask_generator"], mask_generator)
                self.assertEqual("posterior_skip_source" in kwargs, expects_skip_source)

    def test_trial_selection_supports_best_last_and_explicit_ids(self):
        first = SimpleNamespace(number=1)
        latest = SimpleNamespace(number=4)
        study = SimpleNamespace(
            best_trial=first,
            get_trials=lambda: [latest, first],
        )

        self.assertIs(_select_trial(study, "best"), first)
        self.assertIs(_select_trial(study, "last"), latest)
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

    def test_spatial_generated_masks_are_normalized_to_channel_first(self):
        for spatial_shape in ((7, 9), (3, 5, 7)):
            for channels in (1, 3):
                with self.subTest(spatial_shape=spatial_shape, channels=channels):
                    image = np.zeros((channels, *spatial_shape), dtype=np.float32)
                    mask = np.ones(spatial_shape, dtype=np.uint8)
                    _, validated = _validate_generated_variant(image, mask, image)
                    self.assertEqual(validated.shape, (1, *spatial_shape))
                    np.testing.assert_array_equal(validated[0], mask)

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
