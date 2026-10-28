from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from photo_mixstyle import PhotoMixStyle,mix_photo_statistics


@unittest.skipUnless(READY,'Optional training Torch environment required')
class PhotoMixStyleTests(unittest.TestCase):
    def fixture(self):
        a=torch.arange(8,dtype=torch.float32).reshape(1,2,2,2);return torch.cat((a,a,2*a+10,2*a+10)),torch.tensor([0,0,1,1])

    def test_same_photo_views_share_the_same_style_pairing(self):
        x,owners=self.fixture();y=mix_photo_statistics(x,owners,torch.tensor([1,0]),torch.zeros(2),torch.ones(2,dtype=torch.bool))
        torch.testing.assert_close(y[0],y[1]);torch.testing.assert_close(y[2],y[3])
        torch.testing.assert_close(y[0].mean((1,2)),x[2].mean((1,2)));self.assertFalse(torch.equal(y[0],x[0]))

    def test_statistics_are_detached_from_the_donor_photo(self):
        x,owners=self.fixture();x.requires_grad_();y=mix_photo_statistics(x,owners,torch.tensor([1,0]),torch.zeros(2),torch.ones(2,dtype=torch.bool));y[:2].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad[:2].abs().sum()),0.)
        torch.testing.assert_close(x.grad[2:],torch.zeros_like(x.grad[2:]),rtol=0,atol=0)

    def test_evaluation_is_identity_and_consumes_no_rng(self):
        x,owners=self.fixture();model=PhotoMixStyle().eval();state=torch.get_rng_state().clone()
        self.assertIs(model(x,owners),x);self.assertTrue(torch.equal(torch.get_rng_state(),state))

    def test_constant_channels_remain_finite(self):
        x=torch.ones(4,2,3,3);y=mix_photo_statistics(x,torch.tensor([0,0,1,1]),torch.tensor([1,0]),torch.zeros(2),torch.ones(2,dtype=torch.bool))
        self.assertTrue(torch.isfinite(y).all());torch.testing.assert_close(y,x)
