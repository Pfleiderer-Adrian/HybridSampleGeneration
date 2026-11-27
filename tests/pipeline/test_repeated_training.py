"""Repeated generator training and the selection of the trial that is loaded."""

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from tests.pipeline.test_trained_generators import SMALL_MODEL, _samples


@contextlib.contextmanager
def _silent():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


class RepeatedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The study is trained once; the tests only change which trial is selected.
        cls.temporary = tempfile.TemporaryDirectory()
        cls.cuda = patch("torch.cuda.is_available", return_value=False)
        cls.cuda.start()
        config = Configuration("training-study", study_folder=str(Path(cls.temporary.name) / "study"))
        config.extraction.anomaly_size = (1, 8, 8)
        config.extraction.add_background_noise = False
        config.model.set_model("cVAE_ConvNeXt_2D")
        for name, value in SMALL_MODEL.items():
            setattr(config.model.parameters, name, value)
        config.model.search.clear()
        config.training.num_trials = 1
        config.training.epochs = 1
        config.training.batch_size = 2
        config.generation.variants_per_real_anomaly = 1
        with _silent():
            generator = HybridDataGenerator(config)
            generator.ingest_dataset(_samples())
            generator.extract_anomalies()
            generator.train_generator()
            # A second call continues the study with another trial.
            config.training.epochs = 3
            HybridDataGenerator(config).train_generator()
        cls.config = config

    @classmethod
    def tearDownClass(cls):
        cls.cuda.stop()
        cls.temporary.cleanup()

    def setUp(self):
        self.config.training.trial_selection = "best"

    def _study(self):
        return optuna.load_study(
            study_name=self.config.study.name, storage=self.config.study.paths.optuna_storage_url
        )

    def _loaded_checkpoint(self, selection, **kwargs):
        """Return the checkpoint file that ``load_generator`` reads for a selection."""
        self.config.training.trial_selection = selection
        loaded = []
        original = HybridVAEBase.load_checkpoint

        def recording(model, path, **options):
            loaded.append(Path(path).name)
            return original(model, path, **options)

        with _silent(), patch.object(HybridVAEBase, "load_checkpoint", recording):
            model = HybridDataGenerator(self.config).load_generator(**kwargs)
        self.assertEqual(len(loaded), 1)
        return loaded[0], model

    def test_training_again_adds_a_trial_and_keeps_the_first_checkpoint(self):
        trials = self._study().trials
        self.assertEqual([trial.number for trial in trials], [0, 1])
        folder = Path(self.config.study.paths.trained_models)
        for number in (0, 1):
            self.assertTrue((folder / f"model_trial_{number}_best.pth").is_file())
            self.assertTrue((folder / f"model_trial_{number}_history.csv").is_file())
        # The history of each trial has one row per trained epoch.
        rows = [
            len((folder / f"model_trial_{number}_history.csv").read_text().strip().splitlines()) - 1
            for number in (0, 1)
        ]
        self.assertEqual(rows, [1, 3])

    def test_trial_selection_decides_which_checkpoint_is_loaded(self):
        study = self._study()
        best = study.best_trial.number

        self.assertEqual(self._loaded_checkpoint("last")[0], "model_trial_1_best.pth")
        self.assertEqual(self._loaded_checkpoint(0)[0], "model_trial_0_best.pth")
        self.assertEqual(self._loaded_checkpoint(1)[0], "model_trial_1_best.pth")
        self.assertEqual(self._loaded_checkpoint("best")[0], f"model_trial_{best}_best.pth")
        self.assertEqual(
            study.best_trial.value, min(trial.value for trial in study.trials)
        )

    def test_loaded_weights_are_those_of_the_selected_trial(self):
        folder = Path(self.config.study.paths.trained_models)
        for number in (0, 1):
            _, model = self._loaded_checkpoint(number)
            stored = torch.load(folder / f"model_trial_{number}_best.pth", map_location="cpu")
            state = model.state_dict()
            self.assertEqual(set(state), set(stored))
            for name, tensor in stored.items():
                torch.testing.assert_close(state[name].cpu(), tensor)

    def test_missing_trial_and_invalid_selection_are_rejected(self):
        self.config.training.trial_selection = 7
        with _silent(), self.assertRaisesRegex(ValueError, "Optuna trial 7 does not exist"):
            HybridDataGenerator(self.config).load_generator()
        for invalid in ("first", -1, True):
            self.config.training.trial_selection = invalid
            with self.assertRaisesRegex(ValueError, "trial_selection"):
                self.config.validate()

    def test_generation_uses_the_selected_trial(self):
        results = {}
        for number in (0, 1):
            self.config.training.trial_selection = number
            with _silent():
                generator = HybridDataGenerator(self.config)
                generator.load_generator()
                records = generator.generate_synthetic_anomalies()
            results[number] = [
                np.asarray(generator.artifact_store.load_array(record.image_path)).copy()
                for record in records
            ]
        self.assertEqual(len(results[0]), 4)
        # Different weights give different variants for the same seeds.
        self.assertFalse(
            all(np.array_equal(first, second) for first, second in zip(results[0], results[1]))
        )

    def test_generator_can_be_loaded_from_a_copied_trial_database(self):
        copied = Path(self.temporary.name) / "elsewhere" / "trials.db"
        copied.parent.mkdir()
        shutil.copyfile(self.config.study.paths.optuna_db_file, copied)

        name, _ = self._loaded_checkpoint("last", path_to_db_file=copied)

        self.assertEqual(name, "model_trial_1_best.pth")


if __name__ == "__main__":
    unittest.main()
