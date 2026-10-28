"""Shared admission and condition-classification features, supervised per image."""
import torch
from torch import nn
from torch.nn import functional as F
from leaf_feature_expert import LeafFeatureHead


class ClinicalAnchorHead(nn.Module):
    def __init__(self,warm,calibration,classes):
        super().__init__();self.base=LeafFeatureHead('mlp',1408);self.base.load_state_dict(warm)
        self.context=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128,bias=False));nn.init.zeros_(self.context[-1].weight)
        self.conditions=nn.Linear(128,classes);self.register_buffer('calibration',torch.tensor(calibration,dtype=torch.float32))

    def forward(self,paired,whole):
        n,views,dimension=paired.shape;assert dimension==1408 and views==2
        old=self.base.network[1](self.base.network[0](paired));context=self.context(whole)[:,None,:]
        hidden=self.base.network[3](self.base.network[2](old+context))
        healthy=self.base.network[4](hidden).squeeze(-1)*self.calibration[0]+self.calibration[1]
        return healthy,self.conditions(hidden),hidden.mean(1)


def image_condition_loss(logits,targets,healthy_index):
    # A coarse diagnosed-disease label excludes Healthy without inventing a
    # particular named disease or treating supported disease as Other.
    logp=F.log_softmax(logits,dim=-1);exact=targets>=0
    safe_targets=targets.clamp_min(0)
    loss=-logp.gather(-1,safe_targets[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
    nonhealthy=[i for i in range(logits.shape[-1]) if i!=healthy_index]
    partial=-logp[:,:,nonhealthy].logsumexp(-1)
    return torch.where(exact[:,None],loss,partial).mean()
