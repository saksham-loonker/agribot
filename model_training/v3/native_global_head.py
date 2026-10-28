"""Original low-resolution expert plus a native-resolution global residual."""
import torch
from torch import nn
from leaf_feature_expert import LeafFeatureHead


class NativeGlobalHead(nn.Module):
    def __init__(self,kind,warm,calibration):
        super().__init__();assert kind in ('linear','mlp');self.base=LeafFeatureHead('mlp',1408);self.base.load_state_dict(warm)
        self.residual=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,1)) if kind=='linear' else nn.Sequential(nn.LayerNorm(384),nn.Linear(384,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias);self.register_buffer('calibration',torch.tensor(calibration,dtype=torch.float32))

    def forward(self,value):
        assert value.shape[-1]==1792
        return self.base(value[:,:1408])*self.calibration[0]+self.calibration[1]+self.residual(value[:,1408:]).squeeze(-1)
