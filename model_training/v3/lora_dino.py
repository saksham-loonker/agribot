"""Zero-initialized low-rank Q/V updates; frozen base weights remain intact.

LoRA reference: https://arxiv.org/abs/2106.09685 . Evaluation can merge the
updates into an ordinary packed QKV linear map after recognition passes.
"""
import copy,math
import torch
from torch import nn


class LoraQkv(nn.Module):
    def __init__(self,base,rank=8,alpha=16.,dropout=.05):
        super().__init__();assert base.out_features==3*base.in_features
        self.base=copy.deepcopy(base).requires_grad_(False);self.scale=alpha/rank;self.dimension=base.in_features
        self.dropout=nn.Dropout(dropout)
        self.aq=nn.Linear(self.dimension,rank,bias=False);self.bq=nn.Linear(rank,self.dimension,bias=False)
        self.av=nn.Linear(self.dimension,rank,bias=False);self.bv=nn.Linear(rank,self.dimension,bias=False)
        nn.init.kaiming_uniform_(self.aq.weight,a=math.sqrt(5));nn.init.kaiming_uniform_(self.av.weight,a=math.sqrt(5))
        nn.init.zeros_(self.bq.weight);nn.init.zeros_(self.bv.weight)

    def forward(self,value):
        x=self.dropout(value);q=self.bq(self.aq(x))*self.scale;v=self.bv(self.av(x))*self.scale
        return self.base(value)+torch.cat([q,torch.zeros_like(q),v],-1)

    def merged(self):
        result=copy.deepcopy(self.base)
        with torch.no_grad():
            result.weight[:self.dimension].add_(self.bq.weight@self.aq.weight,alpha=self.scale)
            result.weight[2*self.dimension:].add_(self.bv.weight@self.av.weight,alpha=self.scale)
        return result


def inject_qv_lora(backbone):
    backbone.requires_grad_(False)
    for block in backbone.blocks:block.attn.qkv=LoraQkv(block.attn.qkv)
    backbone.norm.requires_grad_(True);backbone.head.requires_grad_(True)
    return backbone
