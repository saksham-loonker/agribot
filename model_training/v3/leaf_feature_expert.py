"""Binary Healthy heads over the original recognizer's frozen leaf features."""

import torch
from torch import nn


class LeafFeatureHead(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        assert kind in ("linear", "mlp")
        self.network = (
            nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 1))
            if kind == "linear"
            else nn.Sequential(
                nn.LayerNorm(384),
                nn.Linear(384, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        )

    def forward(self, features):
        return self.network(features).ravel()
