"""Image-bag supervision over local DINO patches; no patch clinical labels.

Gated attention follows the pooling operator of Ilse et al. (2018):
https://proceedings.mlr.press/v80/ilse18a.html . The residual classifier and
top-k alternative below are task-specific supervised adaptations.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from leaf_feature_expert import LeafFeatureHead


class LocalRecoveryHead(nn.Module):
    def __init__(self,kind,warm,calibration):
        super().__init__();assert kind in ('gated_attention_mil','topk_damage_mil');self.kind=kind
        self.base=LeafFeatureHead('mlp',1408);self.base.load_state_dict(warm)
        self.register_buffer('calibration',torch.tensor(calibration,dtype=torch.float32))
        if kind=='gated_attention_mil':
            self.patch_norm=nn.LayerNorm(384);self.attention_v=nn.Linear(384,64);self.attention_u=nn.Linear(384,64);self.attention_w=nn.Linear(64,1,bias=False)
            self.local_projection=nn.Linear(384,128,bias=False);nn.init.zeros_(self.local_projection.weight)
        else:
            self.damage=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,64),nn.GELU(),nn.Linear(64,1))
            nn.init.zeros_(self.damage[-1].weight);nn.init.zeros_(self.damage[-1].bias)

    def forward(self,tokens,cnn):
        assert tokens.shape[-2:]==(257,384) and cnn.shape[-1]==1024
        original=torch.cat([F.normalize(tokens[:,0],dim=1)*math.sqrt(384),cnn],1)
        patch=tokens[:,1:]
        if self.kind=='gated_attention_mil':
            patch=self.patch_norm(patch)
            weights=self.attention_w(torch.tanh(self.attention_v(patch))*torch.sigmoid(self.attention_u(patch))).softmax(1)
            local=(weights*patch).sum(1)
            pre=self.base.network[1](self.base.network[0](original))+self.local_projection(local)
            z=self.base.network[4](self.base.network[3](self.base.network[2](pre))).ravel()
            return z*self.calibration[0]+self.calibration[1]
        base=self.base(original)*self.calibration[0]+self.calibration[1]
        damage=self.damage(patch).squeeze(-1).topk(8,dim=1).values.mean(1)
        return base-damage
