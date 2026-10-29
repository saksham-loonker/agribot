"""Photo-consistent adaptation of MixStyle (Zhou et al., ICLR 2021).

https://arxiv.org/abs/2104.02008
One label-independent donor and coefficient per photo, shared across its real
ROI views. This is an augmentation hypothesis, not certified clinical invariance.
"""
import torch
from torch import nn


def mix_photo_statistics(value,owners,partners,coefficient,enabled,epsilon=1e-6):
    if value.ndim!=4 or owners.shape!=(len(value),) or owners.dtype!=torch.long:raise ValueError('Real ROI features and integer photo owners required')
    n=len(partners)
    if coefficient.shape!=(n,) or enabled.shape!=(n,) or n<1 or not torch.equal(torch.unique(owners),torch.arange(n,device=owners.device)):raise ValueError('Every photo must own a real ROI')
    if not torch.equal(torch.sort(partners).values,torch.arange(n,device=partners.device)) or not bool(((coefficient>=0)&(coefficient<=1)).all()):raise ValueError('Invalid photo-style pairing')
    mean=value.mean((2,3),keepdim=True).detach();scale=(value.var((2,3),correction=0,keepdim=True)+epsilon).sqrt().detach()
    count=torch.bincount(owners,minlength=n).to(value.dtype)[:,None,None,None]
    photo_mean=value.new_zeros((n,value.shape[1],1,1)).index_add(0,owners,mean)/count
    photo_scale=value.new_zeros((n,value.shape[1],1,1)).index_add(0,owners,scale)/count
    amount=coefficient[owners,None,None,None];donor=partners[owners]
    target_mean=amount*mean+(1-amount)*photo_mean[donor];target_scale=amount*scale+(1-amount)*photo_scale[donor]
    mixed=(value-mean)/scale*target_scale+target_mean
    return torch.where(enabled[owners,None,None,None],mixed,value)


class PhotoMixStyle(nn.Module):
    def __init__(self,probability=.5,alpha=.1,epsilon=1e-6):
        super().__init__()
        if not 0<=probability<=1 or alpha<=0 or epsilon<=0:raise ValueError('Invalid style augmentation configuration')
        self.probability=probability;self.alpha=alpha;self.epsilon=epsilon

    def forward(self,value,owners):
        if not self.training or self.probability==0:return value
        if owners.numel()==0:raise ValueError('Every photo must own a real ROI')
        n=int(owners.max())+1;shape=value.new_tensor(self.alpha)
        coefficient=torch.distributions.Beta(shape,shape).sample((n,));partners=torch.randperm(n,device=value.device);enabled=torch.rand(n,device=value.device)<self.probability
        return mix_photo_statistics(value,owners,partners,coefficient,enabled,self.epsilon)
