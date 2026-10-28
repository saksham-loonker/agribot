"""Image-supervised local lesion attention over frozen native DINO features."""
import torch
from torch import nn
from joint_clinical_head import JointClinicalHead


class NativeLocalClinicalHead(nn.Module):
    def __init__(self,warm):
        super().__init__();self.global_head=JointClinicalHead();self.global_head.load_state_dict(warm,strict=True)
        self.local_norm=nn.LayerNorm(1024);self.gate_content=nn.Linear(1024,64);self.gate_select=nn.Linear(1024,64);self.gate_weight=nn.Linear(64,1,bias=False);self.local_class=nn.Linear(1024,9)
        nn.init.zeros_(self.local_class.weight);nn.init.zeros_(self.local_class.bias)

    def forward(self,global_features,local_features):
        if local_features.ndim!=3 or local_features.shape!=(len(global_features),36,1024):raise ValueError('Expected matching frozen 6x6 native local-token grids')
        local=self.local_norm(local_features);weight=self.gate_weight(torch.tanh(self.gate_content(local))*torch.sigmoid(self.gate_select(local))).softmax(1)
        return self.global_head(global_features)+self.local_class((weight*local).sum(1))
