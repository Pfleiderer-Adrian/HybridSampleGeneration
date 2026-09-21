"""End-to-end training, persisted checkpoint loading, and real fusion."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from hybrid_sample_generator.evaluation.service import evaluate_study
from tests.generation.test_models import MODEL_CASES


class RealGeneratorPipelineTests(unittest.TestCase):
    def test_cpu_training_with_real_workers_checkpoint_reload_generation_and_fusion(self):
        for dims, case in ((2, MODEL_CASES[0]), (3, MODEL_CASES[1])):
            with self.subTest(dims=dims), tempfile.TemporaryDirectory() as root, patch('torch.cuda.is_available', return_value=False):
                name, parameters, shape, _ = case
                config = Configuration('real-pipeline', study_folder=root)
                config.model.set_model(name)
                for key, value in parameters.items():
                    setattr(config.model.parameters, key, value)
                config.model.search.clear()
                config.extraction.anomaly_size = shape[1:]
                config.extraction.roi.fixed_size = shape[2:]
                config.extraction.min_coverage_ratio = 0
                config.extraction.normalization = None
                config.extraction.add_background_noise = False
                config.training.epochs = 1
                config.training.num_trials = 1
                config.generation.variants_per_real_anomaly = 1
                config.training.batch_size = 2
                config.training.validation_ratio = .25
                config.augmentation.random_offset_enabled = False
                config.matching.routine = 'fixed_from_extraction_control_fusion'
                config.generation.variation_strength = 0
                spatial_shape = (12,) * dims
                samples = []
                for index in range(4):
                    image = np.full((1, *spatial_shape), .2, dtype=np.float32)
                    mask = np.zeros_like(image, dtype=np.uint8)
                    window = (slice(None), *((slice(4, 7),) * dims))
                    image[window] = .7 + .05 * index
                    mask[window] = 1
                    samples.append(InputSample(image, mask, f'anomaly-{index}'))
                samples.append(InputSample(np.full((1, *spatial_shape), .2, dtype=np.float32), None, 'control'))
                pipeline = HybridDataGenerator(config)
                pipeline.ingest_dataset(samples)
                pipeline.extract_anomalies()
                pipeline.train_generator()
                study = optuna.load_study(study_name=config.study.name, storage=config.study.paths.optuna_storage_url)
                trial = study.best_trial
                self.assertEqual(trial.state, optuna.trial.TrialState.COMPLETE)
                self.assertTrue(Path(trial.user_attrs['model_path']).is_file())
                restarted = HybridDataGenerator(config)
                restarted.load_generator()
                synthetic = restarted.generate_synthetic_anomalies()
                self.assertEqual(len(synthetic), 4)
                for record in synthetic:
                    array = restarted.artifact_store.load_array(record.image_path)
                    self.assertEqual(array.shape, shape[1:])
                    self.assertTrue(np.isfinite(array).all())
                restarted.plan_hybrid_samples()
                hybrids = restarted.materialize_hybrid_samples()
                self.assertEqual(len(hybrids), 1)
                self.assertEqual(hybrids[0].status, 'generated')
                self.assertEqual(restarted.artifact_store.load_array(hybrids[0].image_path).shape, (1, *spatial_shape))
                self.assertEqual(evaluate_study(config)['volume_cutout']['sample_counter'], 4)
