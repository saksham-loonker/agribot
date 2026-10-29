"""Image-level cross-source contrastive and conditional domain-invariance losses.

Method references: https://arxiv.org/abs/2004.11362 and
https://jmlr.org/papers/v17/15-239.html . These are supervised adaptations
using consumed training sources, not claims of unlabeled target adaptation.
"""
import torch
from torch.nn import functional as F


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value):return value.view_as(value)
    @staticmethod
    def backward(ctx,gradient):return -gradient


def cross_source_contrastive(hidden,labels,sources,photo_ids,temperature=.1):
    hidden=F.normalize(hidden,dim=1)
    logits=hidden@hidden.T/temperature
    distinct=photo_ids[:,None]!=photo_ids[None,:]
    positive=(labels[:,None]==labels[None,:])&(sources[:,None]!=sources[None,:])&distinct
    usable=positive.any(1)
    if not usable.any():return hidden.sum()*0.
    logp=logits-logits.masked_fill(~distinct,-1e9).logsumexp(1,keepdim=True)
    per_anchor=-(logp*positive).sum(1)/positive.sum(1).clamp_min(1)
    return per_anchor[usable].mean()


def conditional_domain_loss(hidden,labels,sources,heads,domain_maps):
    losses=[]
    for label,mapping in domain_maps.items():
        selected=(labels==int(label))&torch.stack([sources==source for source in mapping]).any(0)
        if not selected.any():continue
        targets=torch.full_like(sources[selected],-1)
        for source,target in mapping.items():targets[sources[selected]==source]=target
        losses.append(F.cross_entropy(heads[label](ReverseGradient.apply(hidden[selected])),targets))
    return torch.stack(losses).mean() if losses else hidden.sum()*0.
