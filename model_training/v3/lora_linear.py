"""Zero-initialized low-rank residual for spatial channel-mixing linear maps."""
import copy,math
import torch
from torch import nn


class LoraLinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16.,dropout=.05):
        super().__init__()
        if not isinstance(base,nn.Linear) or not 0<rank<=min(base.in_features,base.out_features):raise ValueError('Unsupported low-rank linear shape')
        self.base=copy.deepcopy(base).requires_grad_(False);self.scale=alpha/rank;self.dropout=nn.Dropout(dropout)
        self.down=nn.Linear(base.in_features,rank,bias=False);self.up=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5));nn.init.zeros_(self.up.weight)

    def forward(self,value):return self.base(value)+self.up(self.down(self.dropout(value)))*self.scale

    def merged(self):
        result=copy.deepcopy(self.base)
        with torch.no_grad():result.weight.add_(self.up.weight@self.down.weight,alpha=self.scale)
        return result
