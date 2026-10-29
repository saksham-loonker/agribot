"""Photo labels: all ROIs for Healthy; one disease witness per real rotation.

Crop views and detected leaves are not independently annotated clinical cases.
This module never relaxes Healthy inference or supplies clinical leaf labels.
"""
import torch
from torch.nn import functional as F


def image_bag_losses(logits,targets,valid,healthy_index=1):
    """Return mean photo clinical/BCE losses and strict all-ROI Healthy margin.

    logits: photo, rotation, crop view, real/padded leaf, clinical class.
    targets: Healthy, a named clinical image class, or -1 for coarse disease.
    Different rotations preserve the photo's disease witness; individual crops
    need not contain that witness. Missing rotations contribute no loss.
    """
    if logits.ndim!=5 or valid.shape!=logits.shape[:-1] or targets.shape!=(len(logits),):raise ValueError('Photo/rotation/ROI clinical shapes do not match')
    if not 0<=healthy_index<logits.shape[-1] or not bool(((targets>=-1)&(targets<logits.shape[-1])).all()):raise ValueError('Invalid image clinical label')
    valid=valid.bool();counts=valid.sum((2,3));rotations=counts>0
    if not bool(rotations.any(1).all()):raise ValueError('Every photo needs a real ROI')
    logits=logits.masked_fill(~valid[...,None],0.)
    logp=F.log_softmax(logits,dim=-1);nonhealthy=[i for i in range(logits.shape[-1]) if i!=healthy_index]
    exact=logp.gather(-1,targets.clamp_min(0)[:,None,None,None,None].expand(*valid.shape,1)).squeeze(-1)
    coarse=logp[...,nonhealthy].logsumexp(-1)
    witness=torch.where((targets>=0)[:,None,None,None],exact,coarse).masked_fill(~valid,-float('inf')).amax((2,3))
    healthy_loss=-logp[...,healthy_index].masked_fill(~valid,0.).sum((2,3))/counts.clamp_min(1)
    clinical=torch.where((targets==healthy_index)[:,None],healthy_loss,-witness).masked_fill(~rotations,0.)
    clinical=(clinical.sum(1)/rotations.sum(1)).mean()
    # Retain the original nine-class margin calibration, including Other bias.
    if logits.shape[-1]!=9 or healthy_index!=1:raise ValueError('Binary margin requires the original nine-class contract')
    z=(logits+logits.new_tensor([0.]*8+[.75]))/.8814700841903687
    margins=z[...,1]-z[...,[0,2,3,4,5,6,7,8]].logsumexp(-1)
    per_rotation=margins.masked_fill(~valid,float('inf')).amin((2,3))
    binary=F.binary_cross_entropy_with_logits(per_rotation.masked_fill(~rotations,0.),(targets==healthy_index)[:,None].expand_as(per_rotation).to(logits.dtype),reduction='none').masked_fill(~rotations,0.)
    binary=(binary.sum(1)/rotations.sum(1)).mean()
    admission=per_rotation.masked_fill(~rotations,float('inf')).min(1).values
    return clinical,binary,admission
