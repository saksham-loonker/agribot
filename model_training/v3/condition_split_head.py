"""Split the broad Other logit into supervised condition subclasses.

The summed calibrated non-Healthy mass is conserved before adaptation. These
are internal recovery classes; they do not add unsupported app diagnoses.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

TEMPERATURE=0.8814700841903687


def calibrated_condition_logits(logits):
    assert logits.shape[-1]>=9
    bias=torch.zeros(logits.shape[-1],device=logits.device,dtype=logits.dtype);bias[8:]=.75
    return (logits+bias)/TEMPERATURE


def expanded_healthy_margin(logits):
    calibrated=calibrated_condition_logits(logits);negative=[i for i in range(logits.shape[-1]) if i!=1]
    return calibrated[:,1]-calibrated[:,negative].logsumexp(1)


def split_other_head(original,additional_classes):
    assert original.out_features==9 and additional_classes>0
    head=nn.Linear(original.in_features,9+additional_classes,bias=True).to(original.weight)
    with torch.no_grad():
        head.weight[:9].copy_(original.weight);head.bias[:9].copy_(original.bias)
        head.weight[9:].copy_(original.weight[8]);head.bias[9:].copy_(original.bias[8])
        head.bias[8:].sub_(TEMPERATURE*math.log(additional_classes+1))
    return head


def split_condition_loss(logits,targets):
    # Exact classes >=0; -2 means Other family, -1 means diagnosed disease
    # without a finer class. Neither partial label becomes a made-up subtype.
    logp=F.log_softmax(calibrated_condition_logits(logits),dim=-1)
    exact=-logp.gather(-1,targets.clamp_min(0)[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
    other=-logp[:,:,8:].logsumexp(-1)
    negative=[i for i in range(logits.shape[-1]) if i!=1];disease=-logp[:,:,negative].logsumexp(-1)
    return torch.where((targets>=0)[:,None],exact,torch.where((targets==-2)[:,None],other,disease)).mean()
