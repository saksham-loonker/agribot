from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from photometric_replay import lighting_variation


@unittest.skipUnless(READY,'Optional training Torch environment required')
class PhotometricReplayTests(unittest.TestCase):
    def test_identity_factors_preserve_rgb_and_channel_order(self):
        rgb=torch.tensor([[[[.2,.8]],[[.3,.7]],[[.4,.6]]]])
        torch.testing.assert_close(lighting_variation(rgb,torch.ones(1,1,1,1),torch.ones(1,1,1,1),torch.ones(1,1,1,1),torch.ones(1,3)),rgb)

    def test_lighting_stays_in_pixel_range(self):
        rgb=torch.rand(4,3,8,8);factors=torch.full((4,1,1,1),1.2)
        changed=lighting_variation(rgb,factors,factors,factors,torch.tensor([[1.1,1.,.9]]).expand(4,-1))
        self.assertTrue(torch.isfinite(changed).all());self.assertGreaterEqual(float(changed.min()),0.);self.assertLessEqual(float(changed.max()),1.)
