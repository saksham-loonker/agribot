from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from lora_dino import LoraQkv


@unittest.skipUnless(READY,'Optional training Torch environment required')
class LoraQkvTests(unittest.TestCase):
    def test_zero_initial_update_keeps_original_outputs_exactly(self):
        torch.manual_seed(124);base=torch.nn.Linear(12,36);layer=LoraQkv(base,rank=3,alpha=6.).eval();x=torch.randn(2,7,12)
        torch.testing.assert_close(layer(x),base(x),rtol=0,atol=0)

    def test_merge_preserves_adapted_outputs_and_never_updates_keys(self):
        torch.manual_seed(125);base=torch.nn.Linear(12,36);layer=LoraQkv(base,rank=3,alpha=6.).eval();x=torch.randn(2,7,12)
        with torch.no_grad():layer.bq.weight.normal_(0,.1);layer.bv.weight.normal_(0,.1)
        torch.testing.assert_close(layer(x),layer.merged()(x),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(layer(x)[...,12:24],base(x)[...,12:24],rtol=0,atol=0)

    def test_adaptation_cannot_backpropagate_into_frozen_base(self):
        layer=LoraQkv(torch.nn.Linear(12,36),rank=3);layer(torch.randn(2,7,12)).sum().backward()
        self.assertIsNone(layer.base.weight.grad);self.assertIsNotNone(layer.bq.weight.grad)
