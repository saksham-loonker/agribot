import copy,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from auxiliary_suffix_expert import AuxiliarySuffixExpert
except ImportError:torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable')
class AuxiliarySuffixTests(unittest.TestCase):
    def fixture(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.attn=nn.Module();self.attn.qkv=nn.Linear(384,1152)
            def forward(self,x):
                q,k,v=self.attn.qkv(x).chunk(3,-1);return x+.1*(q+k+v)
        torch.manual_seed(23);backbone=nn.Module();backbone.blocks=nn.Sequential(Block(),Block());backbone.norm=nn.LayerNorm(384);tokens=torch.randn(7,3,384)
        with torch.no_grad():dino=F.normalize(backbone.norm(backbone.blocks(tokens))[:,0],dim=1)*math.sqrt(384)
        reference=torch.cat([dino,F.normalize(torch.randn(7,1024),dim=1)*32],1);indices=torch.tensor([[[1,0,0],[2,0,0]],[[2,0,0],[1,0,0]]]);mask=torch.zeros_like(indices,dtype=torch.float32);mask[:,:,0]=1
        return backbone,tokens,reference,indices,mask

    def test_zero_lora_and_copied_suffix_match_frozen_feature_head(self):
        b,t,r,i,m=self.fixture();base=AuxiliarySuffixExpert(b,'frozen').eval()
        for mode in ('lora','full_suffix'):
            model=AuxiliarySuffixExpert(b,mode).eval();model.head.load_state_dict(base.head.state_dict())
            with torch.no_grad():torch.testing.assert_close(model(t,r,i,m),base(t,r,i,m),atol=1e-6,rtol=1e-6)

    def test_padding_never_becomes_a_synthetic_leaf_vote(self):
        b,t,r,i,m=self.fixture();model=AuxiliarySuffixExpert(b,'lora').eval();changed=i.clone();changed[:,:,1:]=6
        with torch.no_grad():torch.testing.assert_close(model(t,r,i,m),model(t,r,changed,m),atol=0,rtol=0)

    def test_encoder_gradients_reach_copies_without_modifying_original(self):
        b,t,r,i,m=self.fixture();original=copy.deepcopy(b.state_dict())
        for mode in ('lora','full_suffix'):
            model=AuxiliarySuffixExpert(b,mode).train();loss=model(t,r,i,m).square().mean();loss.backward();suffix=[p for name,p in model.named_parameters() if name.startswith('blocks.') and p.requires_grad];self.assertGreater(sum(float(p.grad.abs().sum()) for p in suffix if p.grad is not None),0.)
        for key,value in b.state_dict().items():self.assertTrue(torch.equal(value,original[key]))
