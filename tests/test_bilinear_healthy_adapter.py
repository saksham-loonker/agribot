import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import torch
    from bilinear_healthy_adapter import BilinearHealthyAdapter
    from leaf_feature_expert import LeafFeatureHead
except ImportError:torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable')
class BilinearTests(unittest.TestCase):
    def test_zero_start_preserves_warm_head_exactly(self):
        torch.manual_seed(22);base=LeafFeatureHead('mlp',1408).eval();x=torch.randn(4,1408)
        for frozen in (True,False):
            adapter=BilinearHealthyAdapter(base.state_dict(),freeze_base=frozen).eval()
            with torch.no_grad():self.assertTrue(torch.equal(base(x),adapter(x)))

    def test_interactions_learn_without_changing_frozen_base(self):
        torch.manual_seed(23);base=LeafFeatureHead('mlp',1408);adapter=BilinearHealthyAdapter(base.state_dict(),True,4).train();before={k:v.clone() for k,v in adapter.base.state_dict().items()};x=torch.randn(8,1408);target=torch.arange(8).float()%2;optimizer=torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad],lr=.1)
        for _ in range(2):
            optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(adapter(x),target);loss.backward();optimizer.step()
        self.assertGreater(float(adapter.shape.weight.grad.abs().sum()),0);self.assertGreater(float(adapter.texture.weight.grad.abs().sum()),0)
        self.assertFalse(adapter.base.training);self.assertTrue(all(p.grad is None for p in adapter.base.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in adapter.base.state_dict().items()))

    def test_wrong_feature_contract_rejected(self):
        base=LeafFeatureHead('mlp',1408);adapter=BilinearHealthyAdapter(base.state_dict())
        with self.assertRaises(ValueError):adapter(torch.zeros(2,1024))
