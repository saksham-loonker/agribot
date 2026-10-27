"""Continue the previously qualified auxiliary expert, never the disease path."""

import copy
import torch
from torch import nn
from partial_dino_healthy import PartialDinoHealthyTail, healthy_margin, prefix_identity


class ContinuedDiagnosticExpert(nn.Module):
    def __init__(self, backbone, checkpoint):
        super().__init__()
        adapted=PartialDinoHealthyTail(backbone)
        adapted.load_state_dict(checkpoint["model"],strict=True)
        assert prefix_identity(backbone)==prefix_identity(adapted.backbone)==checkpoint["prefix_state_sha256"]
        self.blocks=copy.deepcopy(adapted.backbone.blocks[-2:])
        self.norm=copy.deepcopy(adapted.backbone.norm)
        self.head=nn.Sequential(copy.deepcopy(adapted.backbone.fc_norm),copy.deepcopy(adapted.backbone.head))
        self.requires_grad_(True)

    def forward(self,tokens,cnn):
        return healthy_margin(self.head(self.norm(self.blocks(tokens))[:,0]))
