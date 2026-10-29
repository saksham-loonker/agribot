"""Training-only worst-group reweighting with equal binary-class mass."""
import torch
from torch import nn

class ConditionalGroupDRO(nn.Module):
    def __init__(self,group_binary,step_size=.01):
        super().__init__()
        binary=torch.as_tensor(group_binary,dtype=torch.long)
        if binary.ndim!=1 or not all(bool((binary==y).any()) for y in (0,1)) or not bool(((binary==0)|(binary==1)).all()):
            raise ValueError('Both binary classes and valid group labels required')
        if not 0<step_size<=1:raise ValueError('Invalid adversary step size')
        self.register_buffer('group_binary',binary)
        self.register_buffer('log_weights',torch.zeros(len(binary),dtype=torch.float32))
        self.step_size=float(step_size)

    def forward(self,losses,groups):
        if losses.ndim!=1 or groups.shape!=losses.shape or not losses.numel():raise ValueError('Nonempty aligned loss/group vectors required')
        if groups.dtype!=torch.long or bool(((groups<0)|(groups>=len(self.log_weights))).any()):raise ValueError('Unknown training group')
        if not bool(torch.isfinite(losses).all()):raise ValueError('Finite training losses required')
        count=losses.new_zeros(len(self.log_weights)).scatter_add(0,groups,torch.ones_like(losses))
        means=losses.new_zeros(len(self.log_weights)).scatter_add(0,groups,losses)/count.clamp_min(1)
        observed=count>0
        if self.training:
            with torch.no_grad():
                self.log_weights.add_(self.step_size*means.detach()*observed)
                for y in (0,1):
                    mask=self.group_binary==y
                    self.log_weights[mask]-=torch.logsumexp(self.log_weights[mask],0)
        parts=[]
        for y in (0,1):
            mask=observed&(self.group_binary==y)
            if bool(mask.any()):parts.append((self.log_weights[mask].detach().softmax(0)*means[mask]).sum())
        return torch.stack(parts).mean()
