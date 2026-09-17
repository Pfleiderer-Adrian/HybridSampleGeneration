"""Real Optuna persistence and training orchestration without worker processes."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import optuna
import torch
from torch.utils.data import DataLoader
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.training.optuna import objective, optimize
from tests.generation.test_loop import TinyModel


class OptunaRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = Configuration('optuna-regression', study_folder=self.temporary.name)
        self.config.training.epochs = 1
        self.config.training.batch_size = 2
        self.config.training.validation_ratio = .25
        self.config.augmentation.random_offset_enabled = False
        self.splits = []

        def loader(dataset, **kwargs):
            self.splits.append(list(dataset.indices))
            kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
            return DataLoader(dataset, **kwargs)

        for target, kwargs in (
            ('hybrid_sample_generator.generation.training.optuna.DataLoader', {'side_effect': loader}),
            ('hybrid_sample_generator.generation.training.optuna.get_model_spec', {'return_value': SimpleNamespace(build=lambda *a, **k: TinyModel())}),
            ('torch.cuda.is_available', {'return_value': False}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_trials_save_metadata_checkpoint_and_resume_existing_study(self):
        dataset = [torch.ones(1) for _ in range(8)]
        optimize(1, self.config, dataset, num_anomaly_classes=None)
        first_splits = self.splits[:]
        optimize(1, self.config, dataset, num_anomaly_classes=None)
        self.assertEqual(first_splits, self.splits[2:])
        self.assertFalse(set(first_splits[0]) & set(first_splits[1]))
        self.assertEqual(set(first_splits[0]) | set(first_splits[1]), set(range(8)))
        study = optuna.load_study(study_name=self.config.study.name, storage=self.config.study.paths.optuna_storage_url)
        self.assertEqual(len(study.trials), 2)
        for trial in study.trials:
            self.assertEqual(trial.state, optuna.trial.TrialState.COMPLETE)
            self.assertEqual(trial.user_attrs['best_epoch'], 1)
            self.assertEqual(trial.user_attrs['model_name'], self.config.model.name)
            self.assertTrue(Path(trial.user_attrs['model_path']).is_file())
            self.assertTrue(trial.user_attrs['params'])
            self.assertEqual(trial.value, trial.user_attrs['best_val_loss'])

    def test_single_sample_without_validation_trains_and_empty_dataset_fails(self):
        study = optuna.create_study()
        trial = study.ask()
        value = objective(trial, self.config, [torch.ones(1)], num_anomaly_classes=None)
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertEqual(self.splits, [[0], []])
        with self.assertRaisesRegex(ValueError, 'num_samples'):
            objective(study.ask(), self.config, [], num_anomaly_classes=None)

    def test_failed_trial_is_persisted_and_study_can_resume(self):
        dataset = [torch.ones(1) for _ in range(8)]
        with patch("hybrid_sample_generator.generation.training.optuna.train", side_effect=ValueError("non-finite loss")), self.assertRaisesRegex(ValueError, "non-finite loss"):
            optimize(1, self.config, dataset, num_anomaly_classes=None)
        study = optuna.load_study(study_name=self.config.study.name, storage=self.config.study.paths.optuna_storage_url)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.FAIL)
        self.assertNotIn("model_path", study.trials[0].user_attrs)
        optimize(1, self.config, dataset, num_anomaly_classes=None)
        resumed = optuna.load_study(study_name=self.config.study.name, storage=self.config.study.paths.optuna_storage_url)
        self.assertEqual(resumed.best_trial.number, 1)
        self.assertEqual(resumed.trials[1].state, optuna.trial.TrialState.COMPLETE)
