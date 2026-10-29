from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from torch import nn
    from native_full_tail import NativeFullTail


@unittest.skipUnless(READY,'Optional training Torch environment required')
class NativeFullTailTests(unittest.TestCase):
    def test_checkpointed_full_tail_updates_without_mutating_backbone(self):
        torch.manual_seed(181)
        backbone=nn.Module();backbone.global_pool='token';backbone.num_features=4
        backbone.blocks=nn.Sequential(nn.Linear(4,4),nn.GELU(),nn.Linear(4,4))
        backbone.norm=nn.LayerNorm(4);backbone.fc_norm=nn.Identity()
        head=nn.Linear(6,2);model=NativeFullTail(backbone,head)
        before={k:v.clone() for k,v in backbone.state_dict().items()}
        prefix=torch.randn(3,5,4);texture=torch.randn(3,2)
        model.eval()
        expected=head(torch.cat((torch.nn.functional.normalize(backbone.norm(backbone.blocks[-2:](prefix))[:,0],dim=1)*2,texture),1))
        self.assertTrue(torch.allclose(model(prefix,texture),expected))
        model.train();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        model(prefix,texture,checkpointed=True).square().mean().backward()
        self.assertGreater(float(model.blocks[-1].weight.grad.abs().sum()),0)
        optimizer.step()
        for k,v in backbone.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertFalse(torch.equal(model.blocks[-1].weight,backbone.blocks[-1].weight))
