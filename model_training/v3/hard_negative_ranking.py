"""Training-only pairwise separation against the hardest diseased examples."""
import torch
from torch.nn import functional as F


def hard_negative_rank_loss(view_margins,healthy,margin=1.,maximum_negatives=4):
    positive=view_margins[healthy==1].min(1).values
    negative=view_margins[healthy==0].max(1).values
    if not positive.numel() or not negative.numel():return view_margins.sum()*0.
    hard=negative.topk(min(maximum_negatives,negative.numel())).values
    return F.softplus(margin+hard[None,:]-positive[:,None]).mean()
