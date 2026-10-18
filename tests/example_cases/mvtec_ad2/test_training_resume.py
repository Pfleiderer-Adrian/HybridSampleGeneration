"""Resuming preserves optimization, epoch history and the previous best."""
import csv,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.training import train

class Data(torch.utils.data.Dataset):
    def set_epoch(self, epoch):self.epoch=epoch
    def __len__(self):return 4
    def __getitem__(self,i):
        a=torch.full((3,64,64),(i+self.epoch+1)/10)
        return {'image':a,'target':a,'mask':torch.ones(1,64,64),'source':'normal'}
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.randn(()))
        self.backbone=type('Backbone',(),{'config':type('Cfg',(),{'to_dict':lambda _: {}})()})()
    def forward(self,image):return None,self.weight*image[:, :1]
    def segmentation_loss(self,logits,mask):return (logits-mask).square().mean()
class ResumeTests(unittest.TestCase):
    def test_resume_equals_uninterrupted_and_preserves_best(self):
        cfg=DownstreamConfiguration(model='dinov3_segmentation');cfg.training.epochs=3;cfg.training.num_workers=0;cfg.training.device='cpu';cfg.training.selection_metric='pixel_aupro_0.05'
        def metrics(score):return {'segmentation_loss':.1,'segmentation_total_loss':.1,'pixel_aupro_0.05':score}
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/'full';resumed=Path(tmp)/'resumed'
            with patch('examples.mvtec_ad2.downstream.training.build_model',side_effect=lambda _:Model()),patch('examples.mvtec_ad2.downstream.training.evaluation_loader',return_value=[]):
                with patch('examples.mvtec_ad2.downstream.training.evaluate',side_effect=[metrics(.9),metrics(.2),metrics(.3)]):train(Data(),None,cfg,full)
                with patch('examples.mvtec_ad2.downstream.training.evaluate',side_effect=[metrics(.9),RuntimeError('interruption')]):
                    with self.assertRaises(RuntimeError):train(Data(),None,cfg,resumed)
                with patch('examples.mvtec_ad2.downstream.training.evaluate',side_effect=[metrics(.2),metrics(.3)]):train(Data(),None,cfg,resumed,resume_checkpoint=resumed/'checkpoints/last.pt')
            first=torch.load(full/'checkpoints/last.pt',weights_only=True);second=torch.load(resumed/'checkpoints/last.pt',weights_only=True)
            torch.testing.assert_close(first['model']['weight'],second['model']['weight'],rtol=0,atol=0)
            self.assertEqual(first['scheduler'],second['scheduler'])
            self.assertEqual(torch.load(resumed/'checkpoints/best.pt',weights_only=True)['epoch'],1)
            with (resumed/'training_history.csv').open() as history:
                self.assertEqual([int(x['epoch']) for x in csv.DictReader(history)],[1,2,3])
