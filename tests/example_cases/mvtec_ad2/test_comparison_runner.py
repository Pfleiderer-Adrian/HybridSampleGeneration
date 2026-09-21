"""Paired execution, restart checks and dry-run behavior."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from examples.mvtec_ad2.comparison.configuration import ComparisonConfiguration
from examples.mvtec_ad2.comparison.runner import run_comparison
from tests.example_cases.mvtec_ad2.test_grouped_splits import make_grouped_images

MODULE = 'examples.mvtec_ad2.comparison.runner.'


class ComparisonRunnerTests(unittest.TestCase):
    def test_dry_run_persists_identical_splits_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_grouped_images(root / 'data/can')
            config = ComparisonConfiguration(categories=('can',))
            with patch(MODULE + 'prepare_hybrids') as prepare, patch(MODULE + 'train_downstream') as train:
                first = run_comparison(config, root / 'data', root / 'output', dry_run=True)
                second = run_comparison(config, root / 'data', root / 'output', dry_run=True)
                self.assertEqual(first, second)
                prepare.assert_not_called()
                train.assert_not_called()
            config.seed += 1
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                run_comparison(config, root / 'data', root / 'output', dry_run=True)

    def test_paired_variants_share_initial_weights_and_resume_completed_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_grouped_images(root / 'data/can')
            textures = root / 'textures'
            textures.mkdir()
            config = ComparisonConfiguration(categories=('can',))
            config.downstream.data.texture_root = str(textures)
            config.downstream.training.reconstruction_width = 2
            config.downstream.training.segmentation_width = 2
            calls = []

            def train(generator, manifest, settings, *, output_folder, initial_checkpoint):
                calls.append((manifest['fingerprint'], settings.seed, settings.data.hybrid_fraction,
                              settings.data.normal_fraction, Path(initial_checkpoint).read_bytes()))
                output_folder.mkdir()
                settings.save(output_folder / 'settings.json')

            def evaluate(folder, manifest):
                result = {partition: {'image_auroc': .7} for partition in ('validation', 'test')}
                (folder / 'metrics.json').write_text(json.dumps(result))

            def load(folder, manifest):
                from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
                return None, DownstreamConfiguration.load(folder / 'settings.json'), None

            with patch(MODULE + 'prepare_textures'), patch(MODULE + 'build_generator'), \
                    patch(MODULE + 'prepare_hybrids'), patch(MODULE + 'train_downstream', side_effect=train), \
                    patch(MODULE + 'evaluate_downstream', side_effect=evaluate), patch(MODULE + 'load_run', side_effect=load):
                run_comparison(config, root / 'data', root / 'output')
                run_comparison(config, root / 'data', root / 'output')
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][:2], calls[1][:2])
            self.assertEqual([call[2] for call in calls], [0., .5])
            self.assertEqual([call[3] for call in calls], [.5, .5])
            self.assertEqual(calls[0][4], calls[1][4])
            weights = torch.load(root / 'output/can/draem_initial.pt', weights_only=True)
            self.assertTrue(weights)
            state = json.loads((root / 'output/can/status.json').read_text())
            self.assertEqual(state['baseline'], 'complete')
            self.assertEqual(state['hybrid'], 'complete')
            self.assertTrue((root / 'output/comparison.csv').is_file())

    def test_training_failure_is_recorded_and_interrupted_run_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_grouped_images(root / 'data/can')
            textures = root / 'textures'
            textures.mkdir()
            config = ComparisonConfiguration(categories=('can',))
            config.downstream.data.texture_root = str(textures)
            config.downstream.training.reconstruction_width = 2
            config.downstream.training.segmentation_width = 2
            run_comparison(config, root / 'data', root / 'output', dry_run=True)
            interrupted = root / 'output/can/baseline'
            interrupted.mkdir()
            (interrupted / 'checkpoint.pt').write_text('preserve')
            with patch(MODULE + 'prepare_textures'), patch(MODULE + 'build_generator'), \
                    patch(MODULE + 'prepare_hybrids'), \
                    patch(MODULE + 'train_downstream', side_effect=RuntimeError('training failed')):
                with self.assertRaisesRegex(RuntimeError, 'Failed categories: can'):
                    run_comparison(config, root / 'data', root / 'output')
            state = json.loads((root / 'output/can/status.json').read_text())
            self.assertIn('training failed', state['error'])
            self.assertEqual((root / 'output/can/interrupted/baseline_0/checkpoint.pt').read_text(), 'preserve')


class HybridPreparationTests(unittest.TestCase):
    def config(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        return SimpleNamespace(study=SimpleNamespace(seed=42),
                               to_dict=lambda: {'seed': 42}, save_config_file=Mock())

    def test_preparation_is_ordered_and_completed_steps_are_skipped(self):
        from unittest.mock import Mock
        from examples.mvtec_ad2.comparison.runner import prepare_hybrids
        pipeline = Mock()
        pipeline.repository.list_hybrid_samples.return_value = ['generated']
        manifest = {'fingerprint': 'shared', 'partitions': {'train': []}}
        state = {}
        with tempfile.TemporaryDirectory() as temporary, \
                patch(MODULE + 'HybridDataGenerator', return_value=pipeline), \
                patch(MODULE + 'MVTecAD2Dataloader'), \
                patch(MODULE + 'verify_repository'):
            config = self.config()
            path = Path(temporary) / 'status.json'
            prepare_hybrids(config, manifest, state, path)
            names = [call[0] for call in pipeline.method_calls][:6]
            self.assertEqual(names, ['ingest_dataset', 'extract_anomalies', 'train_generator',
                                    'generate_synthetic_anomalies', 'plan_hybrid_samples',
                                    'materialize_hybrid_samples'])
            pipeline.reset_mock()
            prepare_hybrids(config, manifest, state, path)
            pipeline.train_generator.assert_not_called()
            pipeline.generate_synthetic_anomalies.assert_not_called()
            pipeline.materialize_hybrid_samples.assert_not_called()
            pipeline.load_generator.assert_not_called()

    def test_generation_failure_reloads_trained_generator_on_retry(self):
        from unittest.mock import Mock
        from examples.mvtec_ad2.comparison.runner import prepare_hybrids
        pipeline = Mock()
        pipeline.repository.list_hybrid_samples.return_value = ['generated']
        pipeline.generate_synthetic_anomalies.side_effect = RuntimeError('generation failed')
        manifest = {'fingerprint': 'shared', 'partitions': {'train': []}}
        state = {}
        with tempfile.TemporaryDirectory() as temporary, \
                patch(MODULE + 'HybridDataGenerator', return_value=pipeline), \
                patch(MODULE + 'MVTecAD2Dataloader'), \
                patch(MODULE + 'verify_repository'):
            config = self.config()
            path = Path(temporary) / 'status.json'
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                prepare_hybrids(config, manifest, state, path)
            self.assertEqual(state['generator_training'], 'complete')
            self.assertEqual(state['generation'], 'running')
            pipeline.reset_mock()
            pipeline.generate_synthetic_anomalies.side_effect = None
            prepare_hybrids(config, manifest, state, path)
            pipeline.load_generator.assert_called_once()
            pipeline.train_generator.assert_not_called()
            pipeline.generate_synthetic_anomalies.assert_called_once()
            self.assertEqual(state['fusion'], 'complete')

    def test_different_signature_rejects_reuse_before_pipeline_creation(self):
        from examples.mvtec_ad2.comparison.runner import prepare_hybrids
        with patch(MODULE + 'HybridDataGenerator') as pipeline:
            with self.assertRaisesRegex(ValueError, 'different configuration or split'):
                prepare_hybrids(self.config(), {'fingerprint': 'new'},
                                {'generator_signature': 'old'}, Path('unused.json'))
            pipeline.assert_not_called()
