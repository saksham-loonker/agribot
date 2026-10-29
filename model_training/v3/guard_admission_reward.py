"""Exact expected admit/abstain reward using consumed training labels only."""
import math
import torch


def guard_admission_reward_loss(margin,healthy,eligible,false_healthy_cost=8.,temperature=.5,admission=.95):
    if margin.ndim!=1 or healthy.shape!=margin.shape or eligible.shape!=margin.shape or not 0<admission<1 or false_healthy_cost<=0 or temperature<=0:raise ValueError('Invalid guarded reward contract')
    if not bool((((healthy==0)|(healthy==1))&((eligible==0)|(eligible==1))).all()):raise ValueError('Reward labels and eligibility must be binary')
    probability=torch.sigmoid((margin-math.log(admission/(1-admission)))/temperature)
    reward=eligible*probability*(healthy-false_healthy_cost*(1-healthy))
    # Abstraction: admit receives +1 for Healthy / -cost for disease; abstain
    # receives zero. Enumerate both actions exactly instead of sampling RL.
    return -reward.mean()
