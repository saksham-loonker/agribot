"""Binary image/leaf supervision pooled over frozen local CNN features."""

import math
import torch
from torch import nn


class SpatialHealthyHead(nn.Module):
    def __init__(self, channels=576):
        super().__init__()
        self.local = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, features):
        logits = self.local(features.permute(0, 2, 3, 1)).flatten(1)
        # A binary negative image needs one unhealthy region, not a negative
        # target copied onto every detected leaf or spatial position.
        return -torch.logsumexp(-logits, dim=1) + math.log(logits.shape[1])
