"""The 2x AD generator must preserve native V3 geometry and tiny labels."""
import numpy as np
import unittest

from experiments.anomalydiffusion.scale2x_downstream import stitch, windows


class Scale2xProjectionTests(unittest.TestCase):
    def test_windows_cover_tiny_border_hollow_and_large_defects(self):
        masks=[np.ones((256,256),bool),np.eye(256,dtype=bool)]
        hollow=np.ones((256,256),bool);hollow[3:-3,3:-3]=False;masks.append(hollow)
        for x,y in [(0,0),(255,0),(0,255),(255,255),(127,128)]:
            mask=np.zeros((256,256),bool);mask[y,x]=True;masks.append(mask)
        rng=np.random.default_rng(7)
        masks.extend(rng.random((256,256))<.001 for _ in range(50))
        for mask in masks:
            cover=np.zeros_like(mask)
            for x,y in windows(mask):
                assert 0<=x<=128 and 0<=y<=128
                assert mask[y:y+128,x:x+128].any()
                cover[y:y+128,x:x+128]=True
            assert not (mask & ~cover).any()


    def test_stitch_averages_overlap_and_preserves_outer_context(self):
        healthy=np.full((256,256,3),73,np.uint8)
        result=stitch(healthy,[(0,0,np.full((256,256,3),100,np.uint8)),
            (64,0,np.full((256,256,3),200,np.uint8))])
        assert result.shape==healthy.shape and result.dtype==healthy.dtype
        assert (result[:128,:64]==100).all()
        assert (result[:128,64:128]==150).all()
        assert (result[:128,128:192]==200).all()
        assert (result[128:]==73).all()
        assert (result[:,192:]==73).all()


    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):windows(np.zeros((256,256),bool))
