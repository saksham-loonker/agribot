from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from lora_linear import LoraLinear


@unittest.skipUnless(READY,'Optional training Torch environment required')
class SpatialLoraTests(unittest.TestCase):
    def test_zero_adapter_preserves_spatial_channel_mixing(self):
        base=torch.nn.Linear(8,16);value=torch.randn(2,4,5,8);model=LoraLinear(base,rank=4).eval()
        torch.testing.assert_close(model(value),base(value),atol=0,rtol=0)

    def test_merge_preserves_nonzero_adapter_and_base_stays_frozen(self):
        base=torch.nn.Linear(8,16);value=torch.randn(2,4,5,8);model=LoraLinear(base,rank=4).eval()
        with torch.no_grad():model.up.weight.normal_(0,.1)
        torch.testing.assert_close(model(value),model.merged()(value),atol=2e-6,rtol=2e-6)
        model(value).sum().backward();self.assertIsNone(model.base.weight.grad);self.assertIsNotNone(model.up.weight.grad)
