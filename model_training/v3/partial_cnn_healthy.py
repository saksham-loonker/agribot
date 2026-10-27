"""Adapt the retained CNN's final stages without changing its deployed copy."""

import copy
import hashlib
import torch
from torch import nn


def prefix_cut(cnn):
    cut = len(cnn.blocks) - 2
    assert cut > 0
    return cut


def prefix_features(cnn, rgb):
    return cnn.blocks[: prefix_cut(cnn)](cnn.bn1(cnn.conv_stem(rgb)))


def prefix_identity(cnn):
    cut = prefix_cut(cnn)
    digest = hashlib.sha256()
    for name, value in sorted(cnn.state_dict().items()):
        if name.startswith(("conv_stem.", "bn1.")) or any(
            name.startswith(f"blocks.{index}.") for index in range(cut)
        ):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((name, str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class PartialCnnHealthyTail(nn.Module):
    """Original prefix frozen; final two stages/projection/Healthy row trainable.

    Input is the original normalized CNN prefix activation. BatchNorm running
    statistics stay frozen, including during training. No disease or knownness
    output from this separate research expert replaces the deployed recognizer.
    """

    def __init__(self, cnn):
        super().__init__()
        self.cnn = copy.deepcopy(cnn).eval().requires_grad_(False)
        self.cut = prefix_cut(self.cnn)
        original = self.cnn.classifier
        self.cnn.classifier = nn.Linear(
            original.in_features,
            1,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        with torch.no_grad():
            self.cnn.classifier.weight.copy_(original.weight[:1])
            self.cnn.classifier.bias.copy_(original.bias[:1])
        self.cnn.blocks[self.cut :].requires_grad_(True)
        self.cnn.conv_head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, prefix):
        local = self.cnn.blocks[self.cut :](prefix)
        return self.cnn.forward_head(local).squeeze(1)
