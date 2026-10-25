"""Binary Healthy heads over the original recognizer's frozen leaf features."""

import torch
from torch import nn


class LeafFeatureHead(nn.Module):
    def __init__(self, kind, dimension=384):
        super().__init__()
        self.kind = kind
        assert kind in ("linear", "mlp")
        self.network = (
            nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
            if kind == "linear"
            else nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        )

    def forward(self, features):
        return self.network(features).ravel()
