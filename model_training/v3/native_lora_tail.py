"""Copied final-two-block Q/V adapters with checkpointed native-resolution SFT."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from lora_dino import LoraQkv


class NativeLoraTail(nn.Module):
    def __init__(self,backbone,head):
        super().__init__()
        if backbone.global_pool!='token':raise ValueError('Native tail requires CLS pooling')
        self.dimension=backbone.num_features;self.blocks=copy.deepcopy(backbone.blocks[-2:]).requires_grad_(False)
        self.norm=copy.deepcopy(backbone.norm).requires_grad_(True);self.fc_norm=copy.deepcopy(backbone.fc_norm).requires_grad_(True);self.head=copy.deepcopy(head).requires_grad_(True)
        for block in self.blocks:block.attn.qkv=LoraQkv(block.attn.qkv)

    def features(self,prefix,checkpointed=False):
        def run(value):return self.fc_norm(self.norm(self.blocks(value))[:,0])
        value=checkpoint(run,prefix,use_reentrant=False,preserve_rng_state=True) if checkpointed and self.training else run(prefix)
        return F.normalize(value,dim=1)*math.sqrt(self.dimension)

    def forward(self,prefix,texture,checkpointed=False):return self.head(torch.cat((self.features(prefix,checkpointed),texture),1))
