"""Zero-start low-rank morphology/texture interaction over frozen encodings."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from leaf_feature_expert import LeafFeatureHead

class BilinearHealthyAdapter(nn.Module):
    def __init__(self,warm_state,freeze_base=True,rank=32):
        super().__init__()
        if rank<1:raise ValueError('Positive interaction rank required')
        self.base=LeafFeatureHead('mlp',1408);self.base.load_state_dict(warm_state,strict=True);self.freeze_base=bool(freeze_base);self.base.requires_grad_(not self.freeze_base)
        self.shape=nn.Linear(384,rank,bias=False);self.texture=nn.Linear(1024,rank,bias=False);self.readout=nn.Linear(rank,1,bias=False);self.rank=rank
        nn.init.normal_(self.shape.weight,std=1/math.sqrt(384));nn.init.normal_(self.texture.weight,std=1/math.sqrt(1024));nn.init.zeros_(self.readout.weight)

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_base:self.base.eval()
        return self

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=1408:raise ValueError('Expected DINO384 plus CNN1024 feature vectors')
        a=self.shape(F.normalize(x[:,:384],dim=1))*math.sqrt(384);b=self.texture(F.normalize(x[:,384:],dim=1))*math.sqrt(1024)
        return self.base(x)+self.readout(a*b/math.sqrt(self.rank)).ravel()
