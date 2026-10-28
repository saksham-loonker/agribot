"""Combine original ROI evidence with the cached whole-image representation."""
import torch
from torch import nn
from leaf_feature_expert import LeafFeatureHead


class ContextRecoveryHead(nn.Module):
    def __init__(self,kind,warm,calibration):
        super().__init__();assert kind in ('additive_context','bilinear_context');self.kind=kind
        self.base=LeafFeatureHead('mlp',1408);self.base.load_state_dict(warm)
        self.context=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.Dropout(.1))
        self.delta=nn.Linear(128 if kind=='additive_context' else 384,1)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        self.register_buffer('calibration',torch.tensor(calibration,dtype=torch.float32))

    def forward(self,leaves,whole):
        # leaves: image x crop-view x real/padded ROI slot x original feature.
        n,views,slots,dimension=leaves.shape;assert dimension==1408 and whole.shape==(n,768)
        hidden=self.base.network[:-1](leaves.reshape(-1,1408)).reshape(n,views,slots,128)
        original=self.base.network[-1](hidden).squeeze(-1)*self.calibration[0]+self.calibration[1]
        context=self.context(whole)[:,None,None,:]
        if self.kind=='additive_context':return original+self.delta(context).squeeze(-1)
        interaction=torch.cat([hidden,context.expand_as(hidden),hidden*context],-1)
        return original+self.delta(interaction).squeeze(-1)
