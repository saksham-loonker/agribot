"""A copied SigLIP primary-condition tail with an immutable original prefix."""

import copy
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from leaf_feature_expert import LeafFeatureHead


def prefix_features(encoder, nhwc):
    rgb = nhwc.permute(0, 3, 1, 2) * encoder.canonical_std + encoder.canonical_mean
    rgb = F.interpolate(
        rgb, size=(encoder.side, encoder.side), mode="bilinear", align_corners=False
    )
    value = (rgb - encoder.native_mean) / encoder.native_std
    network = encoder.network
    value = network.norm_pre(
        network.patch_drop(network._pos_embed(network.patch_embed(value)))
    )
    return network.blocks[:-2](value)


def prefix_identity(encoder):
    digest = hashlib.sha256()
    cut = len(encoder.network.blocks) - 2
    for name, value in sorted(encoder.state_dict().items()):
        if name.startswith(
            ("network.norm.", "network.fc_norm.", "network.head.", "network.attn_pool.")
        ) or any(name.startswith(f"network.blocks.{i}.") for i in range(cut, cut + 2)):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((name, str(array.dtype), array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PartialPrimarySiglip(nn.Module):
    def __init__(self, encoder, kind="mlp"):
        super().__init__()
        self.encoder = copy.deepcopy(encoder).eval().requires_grad_(False)
        network = self.encoder.network
        network.blocks[-2:].requires_grad_(True)
        network.norm.requires_grad_(True)
        network.fc_norm.requires_grad_(True)
        if network.attn_pool is not None:
            network.attn_pool.requires_grad_(True)
        self.head = LeafFeatureHead(kind, encoder.dimension)

    def features(self, prefix):
        network = self.encoder.network
        tokens = network.norm(network.blocks[-2:](prefix))
        return F.normalize(network.forward_head(tokens), dim=1) * math.sqrt(
            self.encoder.dimension
        )

    def forward(self, prefix):
        return self.head(self.features(prefix))

    def tail_state(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.named_parameters()
            if value.requires_grad
        }

    def load_tail(self, state):
        expected = {
            name for name, value in self.named_parameters() if value.requires_grad
        }
        assert (
            set(state) == expected
        ), "Tail state must include exactly the trainable parameters"
        result = self.load_state_dict(state, strict=False)
        assert (
            not result.unexpected_keys
            and set(result.missing_keys) == set(self.state_dict()) - expected
        )
