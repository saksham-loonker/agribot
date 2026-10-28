"""Image-supervised attention over adapted native tokens, without patch labels."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from joint_clinical_head import JointClinicalHead
from native_lora_tail import NativeLoraTail


class NativePatchClinicalHead(nn.Module):
    def __init__(self,warm,dimension=1024,patches=1369):
        super().__init__();self.dimension=dimension;self.patches=patches
        self.global_head=JointClinicalHead();self.global_head.load_state_dict(warm,strict=True)
        self.local_norm=nn.LayerNorm(dimension)
        self.gate_content=nn.Linear(dimension,64);self.gate_select=nn.Linear(dimension,64);self.gate_weight=nn.Linear(64,1,bias=False);self.local_class=nn.Linear(dimension,9)
        nn.init.zeros_(self.local_class.weight);nn.init.zeros_(self.local_class.bias)

    def forward(self,features):
        expected=(self.patches+1)*self.dimension+768
        if features.ndim!=2 or features.shape[1]!=expected:raise ValueError('Native token/texture feature contract differs')
        tokens=features[:,:-768].reshape(len(features),self.patches+1,self.dimension)
        global_features=torch.cat((tokens[:,0],features[:,-768:]),1)
        local=self.local_norm(tokens[:,1:]);attention=self.gate_weight(torch.tanh(self.gate_content(local))*torch.sigmoid(self.gate_select(local))).softmax(1)
        return self.global_head(global_features)+self.local_class((attention*local).sum(1))


class NativeAdaptedPatchTail(NativeLoraTail):
    def features(self,prefix,checkpointed=False):
        def run(value):return self.norm(self.blocks(value))
        raw=checkpoint(run,prefix,use_reentrant=False,preserve_rng_state=True) if checkpointed and self.training else run(prefix)
        if not isinstance(self.fc_norm,nn.Identity):raise ValueError('Adapted native token tail requires identity fc_norm')
        return (F.normalize(raw,dim=-1)*math.sqrt(self.dimension)).flatten(1)
