"""Copied suffix Healthy expert; original deployed classifier stays immutable."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from leaf_feature_expert import LeafFeatureHead
from lora_dino import LoraQkv

class AuxiliarySuffixExpert(nn.Module):
    def __init__(self,backbone,mode):
        super().__init__()
        if mode not in ('frozen','lora','full_suffix'):raise ValueError('Unknown suffix adaptation')
        self.mode=mode;self.blocks=copy.deepcopy(backbone.blocks[-2:]).requires_grad_(mode=='full_suffix');self.norm=copy.deepcopy(backbone.norm).requires_grad_(mode!='frozen');self.head=LeafFeatureHead('mlp',1408)
        if mode=='lora':
            for block in self.blocks:block.attn.qkv=LoraQkv(block.attn.qkv,rank=8,alpha=16.,dropout=.05)

    def forward(self,prefix,reference,indices,mask):
        valid=mask.reshape(-1)>0;ids=indices.reshape(-1)[valid];unique,inverse=torch.unique(ids,return_inverse=True)
        if self.mode=='frozen':dino=reference[unique,:384]
        else:dino=F.normalize(self.norm(self.blocks(prefix[unique]))[:,0],dim=1)*math.sqrt(384)
        features=torch.cat([dino,reference[unique,384:]],1)[inverse];pooled=features.new_zeros((indices.numel(),1408));pooled[valid]=features;pooled=pooled.reshape(*indices.shape,1408).sum(2)/mask.sum(2).clamp_min(1)[...,None];normalized=torch.cat([F.normalize(pooled[:,:,:384],dim=-1),F.normalize(pooled[:,:,384:],dim=-1)],-1)/math.sqrt(2);return self.head(normalized.reshape(-1,1408)).reshape(len(indices),2)
