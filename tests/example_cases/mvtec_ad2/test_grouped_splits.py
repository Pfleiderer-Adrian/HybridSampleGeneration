"""Acquisition variants must never cross the train/validation boundary."""
import unittest
from pathlib import Path
from unittest.mock import patch
from examples.mvtec_ad2.dataset import MVTecAD2Sample
from examples.mvtec_ad2.splits import create_grouped_manifest, capture_group


class GroupedSplitTests(unittest.TestCase):
    def samples(self):
        root = Path('/tmp/grouped_split_fixture')
        samples = []
        for label, captures, conditions in [('good', 20, ['regular']),
                                             ('bad', 15, ['regular', 'overexposed', 'underexposed',
                                                         'shift_1', 'shift_2', 'shift_3'])]:
            for capture in range(captures):
                for condition in conditions:
                    path = root / 'test_public' / label / f'{capture:03d}_{condition}.png'
                    samples.append(MVTecAD2Sample(path, path if label == 'bad' else None,
                                                  f'{label}_{path.name}', 'test_public', label))
        return samples

    def test_groups_disjoint_exact_ratio_and_reproducibility(self):
        samples = self.samples()
        with patch('examples.mvtec_ad2.splits.discover_samples', return_value=samples):
            first = create_grouped_manifest('/tmp/grouped_split_fixture')
            second = create_grouped_manifest('/tmp/grouped_split_fixture')
        self.assertEqual(first, second)
        self.assertEqual(len(first['partitions']['validation']), 44)
        self.assertEqual(len(first['partitions']['train']), 66)
        self.assertFalse(first['partitions']['test'])
        lookup = {str(s.image_path.resolve()): s for s in samples}
        sets = {part: {capture_group(lookup[s['image_path']]) for s in values}
                for part, values in first['partitions'].items()}
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertEqual(first['grouping']['counts']['bad']['validation_groups'], 6)
        self.assertEqual(sum(len(v) for v in first['partitions'].values()), len(samples))

    def test_folder_namespace_and_unrecognized_capture(self):
        sample = self.samples()[0]
        other = MVTecAD2Sample(sample.image_path, None, 'other', 'train', 'good')
        self.assertNotEqual(capture_group(sample), capture_group(other))
        bad = MVTecAD2Sample(Path('/tmp/unknown.png'), None, 'unknown', 'train', 'good')
        with self.assertRaises(ValueError): capture_group(bad)

