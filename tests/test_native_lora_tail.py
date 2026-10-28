from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from torch import nn
    from native_lora_tail import NativeLoraTail
    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__();self.attn=nn.Module();self.attn.qkv=nn.Linear(16,48)
        def forward(self,x):
            q,k,v=self.attn.qkv(x).chunk(3,-1);return x+.5*(q+v)
    def fixture():
        body=nn.Module();body.blocks=nn.Sequential(FakeBlock(),FakeBlock(),FakeBlock());body.norm=nn.LayerNorm(16);body.fc_norm=nn.Identity();body.num_features=16;body.global_pool='token'
        return body


@unittest.skipUnless(READY,'Optional training Torch environment required')
class NativeTailTests(unittest.TestCase):
    def test_zero_adapters_preserve_original_cls_features(self):
        torch.manual_seed(2174);body=fixture().eval();tail=NativeLoraTail(body,nn.Linear(20,9)).eval();prefix=torch.randn(3,5,16)
        expected=torch.nn.functional.normalize(body.norm(body.blocks[-2:](prefix))[:,0],dim=1)*4
        torch.testing.assert_close(tail.features(prefix),expected,rtol=0,atol=0)

    def test_checkpointed_tail_gets_adapter_gradients_with_frozen_input(self):
        torch.manual_seed(2174);tail=NativeLoraTail(fixture(),nn.Linear(20,9)).train();prefix=torch.randn(3,5,16)
        self.assertFalse(prefix.requires_grad);tail.features(prefix,checkpointed=True)[:,0].sum().backward()
        gradients=[p.grad for name,p in tail.named_parameters() if '.bq.' in name or '.bv.' in name]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients));self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)

    def test_original_backbone_parameters_are_not_mutated(self):
        body=fixture();before={k:v.detach().clone() for k,v in body.state_dict().items()};tail=NativeLoraTail(body,nn.Linear(20,9))
        with torch.no_grad():
            for p in tail.parameters():
                if p.requires_grad:p.add_(1.)
        for k,v in body.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
