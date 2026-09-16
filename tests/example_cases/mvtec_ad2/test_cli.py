"""CLI selects the intended workflow without starting model training."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from examples.mvtec_ad2.__main__ import main
from examples.mvtec_ad2.settings import EXPERIMENT


class CliTests(unittest.TestCase):
    def test_generate_passes_paths_categories_and_configured_split(self):
        with patch('examples.mvtec_ad2.__main__.run_new_experiment', return_value=[]) as run:
            main(['generate', '--dataset-root', '/data', '--output-root', '/out',
                  '--categories', 'can', 'fabric', '--steps', 'ingest', 'extract'])
        experiment = run.call_args.args[0]
        self.assertEqual(experiment.dataset_root, Path('/data'))
        self.assertEqual(experiment.output_root, Path('/out'))
        self.assertEqual(experiment.categories, ('can', 'fabric'))
        self.assertEqual(experiment.split, EXPERIMENT.split)
        self.assertEqual(run.call_args.kwargs['steps'], ['ingest', 'extract'])

    def test_continue_keeps_explicit_steps(self):
        with patch('examples.mvtec_ad2.__main__.run_existing_studies', return_value=[]) as run:
            main(['continue', '--study-folder', '/a', '/b', '--steps', 'materialize', 'export'])
        run.assert_called_once_with([Path('/a'), Path('/b')], steps=['materialize', 'export'])

    def test_review_dispatches_outside_pipeline(self):
        with patch('examples.mvtec_ad2.__main__.review_studies') as review:
            main(['review', '--study-folder', '/study', '--actions', 'evaluate_generator'])
        review.assert_called_once_with([Path('/study')], actions=['evaluate_generator'])

    def test_evaluate_selects_one_saved_run(self):
        with patch('examples.mvtec_ad2.__main__.run_existing_studies', return_value=[]) as run:
            main(['evaluate', '--study-folder', '/study', '--run-id', 'saved-run'])
        run.assert_called_once_with([Path('/study')], steps=('evaluate_downstream',),
                                    downstream_run_id='saved-run')

    def test_missing_or_inapplicable_arguments_are_rejected(self):
        for args in ([], ['continue', '--study-folder', '/study'],
                     ['evaluate', '--study-folder', '/study'],
                     ['review', '--study-folder', '/study', '--steps', 'ingest']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
