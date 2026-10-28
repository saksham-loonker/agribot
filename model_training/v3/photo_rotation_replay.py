"""Select one real rotation per photo, shared across its crop views and leaves."""
import torch


def select_photo_rotation(index,mask,training,generator=None):
    if index.shape!=mask.shape or index.ndim!=4:raise ValueError('Expected photo,rotation,crop,leaf tables')
    valid=mask.sum((2,3))>0
    if not bool(valid.any(1).all()):raise ValueError('Every photo needs a real rotation')
    if training:rotation=torch.multinomial(valid.float(),1,generator=generator).squeeze(1)
    else:
        if not bool(valid[:,0].all()):raise ValueError('Unaugmented evaluation view is missing')
        rotation=torch.zeros(len(index),dtype=torch.long,device=index.device)
    rows=torch.arange(len(index),device=index.device)
    return index[rows,rotation],mask[rows,rotation],rotation
