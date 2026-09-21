"""Complete paired comparisons, missing runs and category macro averages."""
import json
import tempfile
import unittest
from pathlib import Path
from examples.mvtec_ad2.comparison.reporting import METRICS, write_report


class ComparisonReportTests(unittest.TestCase):
    def test_deltas_macro_means_and_incomplete_categories(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for category, baseline, hybrid in (('can', .2, .5), ('rice', .4, .6)):
                folder = root / category
                folder.mkdir()
                manifest = {'partitions': {split: [{'label': 'bad'}, {'label': 'good'}] for split in ('test', 'validation')}}
                (folder / 'split_manifest.json').write_text(json.dumps(manifest))
                for variant, value in (('baseline', baseline), ('hybrid', hybrid)):
                    destination = folder / variant
                    destination.mkdir()
                    metrics = {split: {metric: value for metric in METRICS} for split in ('test', 'validation')}
                    (destination / 'metrics.json').write_text(json.dumps(metrics))
            rows = write_report(root, ('can', 'rice', 'vial'))
            can = next(row for row in rows if row['category'] == 'can' and row['partition'] == 'test')
            self.assertAlmostEqual(can['delta'], .3)
            mean = next(row for row in rows if row['category'].startswith('macro_mean') and row['partition'] == 'test')
            self.assertAlmostEqual(mean['baseline'], .3)
            self.assertAlmostEqual(mean['hybrid'], .55)
            self.assertAlmostEqual(mean['delta'], .25)
            self.assertIn('Incomplete categories: vial', (root / 'comparison.md').read_text())
            self.assertTrue((root / 'comparison.csv').is_file())

    def test_missing_metrics_are_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as root:
            rows = write_report(root, ('can',))
            self.assertEqual(rows, [])
