"""One clinical label and one loss contribution per original photograph."""
import torch
from torch.nn import functional as F


def balanced_clinical_loss(logits,targets,valid_views,healthy_index):
    if logits.ndim!=3 or valid_views.shape!=logits.shape[:2] or targets.shape!=(logits.shape[0],):raise ValueError('Clinical photo/view shapes do not match')
    valid_views=valid_views.bool();counts=valid_views.sum(1)
    if not bool((counts>0).all()):raise ValueError('Every clinical photo must have a real view')
    if not 0<=healthy_index<logits.shape[-1] or not bool(((targets>=-1)&(targets<logits.shape[-1])).all()):raise ValueError('Unsupported clinical label')
    # Invalid/padded views do not enter normalization, evidence or gradients.
    logp=F.log_softmax(logits.masked_fill(~valid_views[:,:,None],0.),dim=-1)
    exact=-logp.gather(-1,targets.clamp_min(0)[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
    negative=[i for i in range(logits.shape[-1]) if i!=healthy_index]
    partial=-logp[:,:,negative].logsumexp(-1)
    per_view=torch.where((targets>=0)[:,None],exact,partial).masked_fill(~valid_views,0.)
    return (per_view.sum(1)/counts).mean()
