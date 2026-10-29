"""Differentiate through one support update before held-source supervision."""
import torch

def meta_domain_loss(model,support_loss,query_loss,inner_lr):
    if inner_lr<0:raise ValueError('Nonnegative inner learning rate required')
    params=dict(model.named_parameters());trainable={name:value for name,value in params.items() if value.requires_grad}
    if not trainable:raise ValueError('At least one trainable parameter required')
    support=support_loss(None)
    gradients=torch.autograd.grad(support,tuple(trainable.values()),create_graph=True)
    fast=dict(params);fast.update({name:value-inner_lr*gradient for (name,value),gradient in zip(trainable.items(),gradients)})
    query=query_loss(fast)
    return .5*(support+query),support.detach(),query.detach()
