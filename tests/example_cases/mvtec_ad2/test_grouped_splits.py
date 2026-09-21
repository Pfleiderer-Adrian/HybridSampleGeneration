"""Grouped split ratios, disjointness and private-data exclusion."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from examples.mvtec_ad2.splits import create_grouped_manifest, validate_grouped_manifest, _fingerprint


def make_grouped_images(root):
    for source, label, groups in (('train', 'good', 18), ('test_public', 'bad', 10), ('test_private', 'bad', 3)):
        folder = root / source / label
        folder.mkdir(parents=True)
        for group in range(groups):
            for variant in ('regular', 'shift_1'):
                name = f'{group:03d}_{variant}'
                Image.fromarray(np.full((64, 64, 3), 80, dtype=np.uint8)).save(folder / f'{name}.png')
                if label != 'good' and source != 'test_private':
                    masks = root / source / 'ground_truth' / label
                    masks.mkdir(parents=True, exist_ok=True)
                    mask = np.zeros((64, 64), dtype=np.uint8)
                    mask[20:24, 20:24] = 255
                    Image.fromarray(mask).save(masks / f'{name}_mask.png')


class GroupedSplitTests(unittest.TestCase):
    def test_reproducible_groups_and_twenty_five_percent_positive_test(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            make_grouped_images(root)
            manifest = create_grouped_manifest(root)
            self.assertEqual(manifest, create_grouped_manifest(root))
            owners = {}
            for partition, items in manifest['partitions'].items():
                for item in items:
                    self.assertNotEqual(item['split'], 'test_private')
                    group = item['acquisition_group']
                    self.assertEqual(owners.setdefault(group, partition), partition)
            test = manifest['partitions']['test']
            self.assertEqual(sum(x['label'] != 'good' for x in test), 4)
            self.assertEqual(len(test), 16)

    def test_unknown_group_names_and_empty_positive_masks_fail(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            make_grouped_images(root)
            original = root / 'train/good/000_regular.png'
            original.rename(original.with_name('unknown.png'))
            with self.assertRaisesRegex(ValueError, 'Unknown acquisition'):
                create_grouped_manifest(root)
            original.with_name('unknown.png').rename(original)
            mask = root / 'test_public/ground_truth/bad/000_regular_mask.png'
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(mask)
            with self.assertRaisesRegex(ValueError, 'usable mask'):
                create_grouped_manifest(root)

    def test_cross_partition_group_is_rejected_even_with_valid_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            make_grouped_images(root)
            manifest = create_grouped_manifest(root)
            train = manifest['partitions']['train']
            item = train.pop(0)
            manifest['partitions']['test'].append(item)
            manifest['fingerprint'] = _fingerprint(manifest)
            with self.assertRaisesRegex(ValueError, 'crosses partitions'):
                validate_grouped_manifest(manifest)
