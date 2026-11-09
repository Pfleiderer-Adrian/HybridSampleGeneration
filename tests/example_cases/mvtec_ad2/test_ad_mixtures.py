"""Equal synthetic budgets and unchanged real/normal draws in new AD mixtures."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.quota_dataset import FixedQuotaDataset
from examples.mvtec_ad2.splits import create_manifest, manifest_samples
from experiments.anomalydiffusion.mixtures import profile
from .test_downstream import create_images


class ADMixtureTests(unittest.TestCase):
    def test_equal_thirds_rotate_with_identical_real_and_normal_draws(self):
        class Pairs:
            def __len__(self):
                return 2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_images(root)
            split = create_manifest(root / 'can')
            healthy = manifest_samples(split, 'train', healthy_only=True)
            real = [s for s in manifest_samples(split, 'train') if s.label != 'good']
            baseline = DownstreamConfiguration(model='dinov3_paired')
            baseline.data.include_real_defects = True
            baseline.data.patch_size = (64, 64)
            baseline.data.anomalydiffusion_manifest = 'validated.json'
            baseline.data.source_quotas, baseline.data.source_quota_rotation = profile('real_anomalydiffusion')
            reference = FixedQuotaDataset(healthy, real, None, baseline, anomalydiffusion=Pairs())
            for variant in ('real_anomalydiffusion_draem', 'real_anomalydiffusion_draem_hybrid'):
                config = deepcopy(baseline)
                config.data.source_quotas, config.data.source_quota_rotation = profile(variant)
                with patch('examples.mvtec_ad2.downstream.quota_dataset.TextureSynthesizer'):
                    dataset = FixedQuotaDataset(healthy, real, Pairs(), config, anomalydiffusion=Pairs())
                totals = Counter()
                for epoch in range(100):
                    dataset.set_epoch(epoch)
                    counts = Counter(dataset.plan)
                    self.assertEqual(len(dataset), 1000)
                    self.assertEqual(counts['normal'], 500)
                    self.assertEqual(counts['real'], 100)
                    self.assertEqual(sum(counts[s] for s in ('draem', 'hybrid', 'anomalydiffusion')), 400)
                    totals.update(counts)
                    if epoch < 3:
                        reference.set_epoch(epoch)
                        self.assertEqual(reference.real_order, dataset.real_order)
                        for source in ('normal', 'real'):
                            i = reference.entries.index((source, 0))
                            j = dataset.entries.index((source, 0))
                            torch.testing.assert_close(reference[i]['image'], dataset[j]['image'])
                if config.data.source_quota_rotation:
                    self.assertEqual([totals[s] for s in ('anomalydiffusion', 'draem', 'hybrid')],
                                     [13334, 13333, 13333])
                    totals3 = Counter()
                    for epoch in range(3):
                        dataset.set_epoch(epoch)
                        counts = Counter(dataset.plan)
                        self.assertEqual(counts[config.data.source_quota_rotation[epoch]], 134)
                        totals3.update({s: counts[s] for s in config.data.source_quota_rotation})
                    self.assertEqual(set(totals3.values()), {400})
                else:
                    self.assertEqual(totals['anomalydiffusion'], 20000)
                    self.assertEqual(totals['draem'], 20000)


if __name__ == '__main__':
    unittest.main()
