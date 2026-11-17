"""Healthy training-only mining, fixed quotas and aligned negative targets."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch import nn
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.datasets import MixedTrainingDataset
from examples.mvtec_ad2.downstream.hard_normals import mine_pool, load_pool, box_iou

class BrightnessModel(nn.Module):
    def forward(self, image):
        score=image.mean(1,keepdim=True)*4
        return None,torch.cat([torch.zeros_like(score),score],dim=1)

class UnusedHybrids:
    def __len__(self):return 1

class HardNormalTests(unittest.TestCase):
    def test_training_records_hard_quota_and_copies_pool(self):
        from .test_downstream import create_images, generator_config, seed_hybrid
        from examples.mvtec_ad2.splits import create_manifest, SplitConfiguration, manifest_samples
        from examples.mvtec_ad2.downstream.runner import train_downstream
        import csv
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);create_images(root)
            manifest=create_manifest(root/'can',SplitConfiguration())
            generator=generator_config(root/'study');seed_hybrid(generator,manifest)
            healthy=manifest_samples(manifest,'train',healthy_only=True)
            pool={'version':1,'source':'healthy_training_only','patch_size':[64,64],
                  'training_sample_ids':sorted(s.sample_id for s in healthy),
                  'records':[{'sample_id':healthy[0].sample_id,'top':0,'left':0,'score':.9}]}
            path=root/'pool.json';path.write_text(json.dumps(pool))
            cfg=DownstreamConfiguration();cfg.data.patch_size=(64,64);cfg.data.samples_per_epoch=8
            cfg.data.normal_fraction=.75;cfg.data.hybrid_fraction=1;cfg.data.hard_normal_fraction=1/3;cfg.data.hard_normal_manifest=str(path)
            cfg.training.epochs=1;cfg.training.batch_size=2;cfg.training.num_workers=0;cfg.training.device='cpu'
            cfg.training.reconstruction_width=2;cfg.training.segmentation_width=2
            output=train_downstream(generator,manifest,cfg)
            with (output/'training_history.csv').open() as file:row=next(csv.DictReader(file))
            self.assertEqual(int(row['hard_normal']),2)
            self.assertEqual(int(row['normal']),6)
            self.assertEqual(json.loads((output/'hard_normal_manifest.json').read_text()),pool)

    def test_mining_selects_bright_regions_and_rejects_nonhealthy_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);array=np.zeros((128,128,3),np.uint8);array[96:,96:]=255
            path=root/'normal.png';Image.fromarray(array).save(path)
            samples=[SimpleNamespace(sample_id=f'train_{i}',image_path=str(path),label='good') for i in range(2)]
            cfg=DownstreamConfiguration();cfg.data.patch_size=(64,64);checkpoint=root/'teacher.pt';checkpoint.write_bytes(b'teacher')
            pool=mine_pool(BrightnessModel(),samples,cfg.data,'cpu',checkpoint,root/'pool.json',per_image_limit=1,capacity=2)
            self.assertEqual(len(pool['records']),2)
            self.assertTrue(all((r['top'],r['left'])==(64,64) for r in pool['records']))
            self.assertEqual(load_pool(root/'pool.json',samples,(64,64)),pool)
            with self.assertRaisesRegex(ValueError,'healthy training split'):
                load_pool(root/'pool.json',[SimpleNamespace(sample_id='validation',label='good')],(64,64))
            samples[0].label='bad'
            with self.assertRaisesRegex(ValueError,'healthy training'):
                mine_pool(BrightnessModel(),samples,cfg.data,'cpu',checkpoint,root/'bad.json')
            self.assertEqual(box_iou((0,0),(0,0),(64,64)),1)
            self.assertEqual(box_iou((0,0),(64,64),(64,64)),0)

    def test_hard_normal_quota_determinism_targets_and_bad_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'normal.png';Image.fromarray(np.full((64,64,3),80,np.uint8)).save(path)
            samples=[SimpleNamespace(sample_id='train_good',image_path=str(path),label='good')]
            pool={'version':1,'source':'healthy_training_only','patch_size':[64,64],
                  'training_sample_ids':['train_good'],'records':[{'sample_id':'train_good','top':0,'left':0,'score':.9}]}
            file=root/'pool.json';file.write_text(json.dumps(pool))
            cfg=DownstreamConfiguration();cfg.data.patch_size=(64,64);cfg.data.hybrid_fraction=1;cfg.data.normal_fraction=.75;cfg.data.samples_per_epoch=24
            cfg.data.hard_normal_fraction=1/3;cfg.data.hard_normal_manifest=str(file);cfg.validate()
            dataset=MixedTrainingDataset(samples,UnusedHybrids(),cfg)
            self.assertEqual(len(dataset.hard_indices),6)
            self.assertEqual(dataset.plan.count('normal'),18)
            for i in dataset.hard_indices:
                item=dataset[i];self.assertEqual(item['source'],'normal');self.assertTrue(item['hard_normal']);self.assertFalse(item['mask'].any());torch.testing.assert_close(item['image'],item['target']);torch.testing.assert_close(item['image'],dataset[i]['image'])
            indices=dataset.hard_indices.copy();dataset.set_epoch(0);self.assertEqual(indices,dataset.hard_indices)
            pool['records'][0]['top']=200;file.write_text(json.dumps(pool));bad=MixedTrainingDataset(samples,UnusedHybrids(),cfg)
            with self.assertRaisesRegex(ValueError,'outside'):
                bad[next(iter(bad.hard_indices))]
            pool['records'][0]['sample_id']='held_out';file.write_text(json.dumps(pool))
            with self.assertRaisesRegex(ValueError,'outside'):
                MixedTrainingDataset(samples,UnusedHybrids(),cfg)
            cfg.data.hard_normal_fraction=0;cfg.validate();normal=MixedTrainingDataset(samples,UnusedHybrids(),cfg);self.assertEqual(len(normal.hard_indices),0)

    def test_invalid_configuration(self):
        cfg=DownstreamConfiguration()
        for value in [-.1,1.1,True,float('nan')]:
            cfg.data.hard_normal_fraction=value
            with self.assertRaises(ValueError):cfg.validate()
        cfg.data.hard_normal_fraction=1/3
        with self.assertRaisesRegex(ValueError,'manifest'):cfg.validate()
