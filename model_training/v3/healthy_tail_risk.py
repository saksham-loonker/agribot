"""Weighted upper-tail risk for human-labelled Healthy-versus-disease fitting."""

import torch


def weighted_upper_tail(values, weights=None, fraction=0.1):
    if not 0 < fraction <= 1:
        raise ValueError("Tail mass must be in (0, 1]")
    if values.numel() == 0:
        return values.sum()
    weights = torch.ones_like(values) if weights is None else weights
    assert values.shape == weights.shape and torch.isfinite(values).all()
    assert torch.isfinite(weights).all() and (weights > 0).all()
    order = values.argsort(descending=True)
    ordered = weights[order] / weights.sum()
    previous = ordered.cumsum(0) - ordered
    mass = torch.minimum(ordered, (fraction - previous).clamp_min(0))
    return (mass * values[order]).sum() / fraction
