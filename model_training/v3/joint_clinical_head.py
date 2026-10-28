"""Warm large-DINO clinical head with a zero-initialized joint CNN residual."""
import torch
from torch import nn
from torch.nn import functional as F


class JointClinicalHead(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Linear(1024,9)
        self.residual=nn.Sequential(nn.LayerNorm(1792),nn.Linear(1792,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,9))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def load_warm_head(self,state):self.base.load_state_dict(state,strict=True)

    def forward(self,value):
        if value.shape[-1]!=1792:raise ValueError('Joint clinical features require 1024 DINO and 768 CNN dimensions')
        dino=value[...,:1024];cnn=F.normalize(value[...,1024:],dim=-1)*(768**.5)
        return self.base(dino)+self.residual(torch.cat((dino,cnn),-1))
