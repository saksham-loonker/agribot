"""A copied original DINO final-stage expert; the deployed recognizer is frozen."""

import copy
import hashlib
import torch
from torch import nn


def prefix_cut(backbone):
    return len(backbone.blocks) - 2


def prefix_features(backbone, normalized_rgb):
    x = backbone.patch_embed(normalized_rgb)
    x = backbone._pos_embed(x)
    x = backbone.patch_drop(x)
    x = backbone.norm_pre(x)
    return backbone.blocks[: prefix_cut(backbone)](x)


def prefix_identity(backbone):
    cut = prefix_cut(backbone)
    digest = hashlib.sha256()
    for name, value in sorted(backbone.state_dict().items()):
        if (
            name in ("cls_token", "reg_token", "pos_embed")
            or name.startswith(("patch_embed.", "norm_pre."))
            or any(name.startswith(f"blocks.{i}.") for i in range(cut))
        ):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((name, str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def healthy_margin(logits):
    calibrated = (logits + logits.new_tensor([0.0] * 8 + [0.75])) / 0.8814700841903687
    return calibrated[:, 1] - calibrated[:, [0, 2, 3, 4, 5, 6, 7, 8]].logsumexp(1)


class PartialDinoHealthyTail(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = copy.deepcopy(backbone).eval().requires_grad_(False)
        self.cut = prefix_cut(backbone)
        self.backbone.blocks[self.cut :].requires_grad_(True)
        self.backbone.norm.requires_grad_(True)
        self.backbone.fc_norm.requires_grad_(True)
        self.backbone.head.requires_grad_(True)

    def forward(self, prefix):
        tokens = self.backbone.norm(self.backbone.blocks[self.cut :](prefix))
        return self.backbone.forward_head(tokens)
