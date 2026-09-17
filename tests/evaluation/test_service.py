"""Empty studies and recorded-mask fallback semantics."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.evaluation.service import evaluate_study, run_feature_calculator
from hybrid_sample_generator.evaluation.pairs import EvaluationPair
from hybrid_sample_generator.evaluation.outliers import find_outliers
from hybrid_sample_generator.evaluation.metrics import compute_glcm, get_volume_feature_diffs
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore


class EvaluationServiceTests(unittest.TestCase):
    def test_empty_study_returns_empty_results_without_csv(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration('empty', study_folder=root)
            results = evaluate_study(config)
            for result in results.values():
                self.assertEqual(result['sample_counter'], 0)
                self.assertEqual(result['mean_real'], {})
                self.assertEqual(result['all_diffs'], {})
            self.assertFalse(Path(config.study.paths.metric_diffs_csv).exists())

    def test_missing_masks_use_threshold_and_disabled_threshold_uses_all_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration('masks', study_folder=root)
            store = ArtifactStore(config.study.paths.study_folder)
            image = np.array([[[0., 0.], [0., 1.]]], dtype=np.float32)
            path = store.save_entity_array('real_anomalies', 'real', 'image', image)
            pair = EvaluationPair('pair', 'real', 'synthetic', path, path)
            Path(config.study.paths.evaluation_results).mkdir(parents=True, exist_ok=True)
            config.evaluation.foreground_threshold = .5
            result = run_feature_calculator([pair], get_volume_feature_diffs, store, config.study.paths, config.evaluation, use_recorded_masks=True)
            self.assertEqual(result['mean_real']['Volume'], 1)
            config.evaluation.foreground_threshold = None
            result = run_feature_calculator([pair], get_volume_feature_diffs, store, config.study.paths, config.evaluation, use_recorded_masks=True)
            self.assertEqual(result['mean_real']['Volume'], 4)

    def test_empty_foreground_has_zero_glcm_and_undefined_volume_centers(self):
        for dims in (2, 3):
            image = np.ones((1, *((3,) * dims)))
            mask = np.zeros_like(image)
            self.assertEqual(compute_glcm(image, mask).sum(), 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                real, synthetic, differences = get_volume_feature_diffs(image, mask, image, mask)
            self.assertEqual(real['Volume'], 0)
            self.assertEqual(differences['Volume'], 0)
            self.assertTrue(all(np.isnan(value) for name, value in real.items() if name != 'Volume'))

    def test_missing_and_corrupt_artifacts_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("invalid-artifact", study_folder=root)
            store = ArtifactStore(config.study.paths.study_folder)
            path = "artifacts/missing.npy"
            pair = EvaluationPair("pair", "real", "synthetic", path, path)
            with self.assertRaises(FileNotFoundError):
                run_feature_calculator([pair], get_volume_feature_diffs, store, config.study.paths, config.evaluation, use_recorded_masks=True)
            store.resolve(path).write_bytes(b"invalid numpy payload")
            with self.assertRaises(ValueError):
                run_feature_calculator([pair], get_volume_feature_diffs, store, config.study.paths, config.evaluation, use_recorded_masks=True)

    def test_different_pair_shapes_have_independent_volume_and_center_measurements(self):
        real = np.zeros((1, 3, 4))
        synthetic = np.zeros((1, 5, 6))
        real[:, 1, 1] = 1
        synthetic[:, 2, 3] = 1
        _, _, differences = get_volume_feature_diffs(real, real, synthetic, synthetic)
        self.assertEqual(differences, {"Volume": 0, "H-center": 1., "W-center": 2.})
        with self.assertRaises((IndexError, ValueError)):
            compute_glcm(real, np.ones((1, 2, 4)))

    def test_undefined_centers_survive_csv_export_and_do_not_become_outliers(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration("undefined-centers", study_folder=root)
            config.evaluation.foreground_threshold = 2
            store = ArtifactStore(config.study.paths.study_folder)
            path = store.save_entity_array("real_anomalies", "real", "image", np.ones((1, 3, 3)))
            pair = EvaluationPair("pair", "real", "synthetic", path, path)
            Path(config.study.paths.evaluation_results).mkdir(parents=True, exist_ok=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                result = run_feature_calculator([pair], get_volume_feature_diffs, store, config.study.paths, config.evaluation, use_recorded_masks=True)
            self.assertTrue(np.isnan(result["all_diffs"]["H-center"][0]))
            self.assertEqual(result["outliers"]["H-center"], [])
            with open(config.study.paths.metric_diffs_csv, newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertTrue(np.isnan(json.loads(row["metric_diffs"])["H-center"]))

    def test_non_finite_values_do_not_hide_finite_outliers(self):
        from hybrid_sample_generator.configuration.evaluation import EvaluationConfiguration
        values = [0.] * 9 + [100., float("nan"), float("inf")]
        entries = [{"value": value, "sample": str(index)} for index, value in enumerate(values)]
        self.assertEqual(find_outliers(values, entries, EvaluationConfiguration(), "H-center"), [entries[9]])
        self.assertEqual(find_outliers([float("nan")], [entries[10]], EvaluationConfiguration(), "H-center"), [])
