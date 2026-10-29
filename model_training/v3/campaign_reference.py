"""Load the immutable installed-model comparator across campaign cache versions."""

from pathlib import Path
import numpy as np


def safe_logits(root):
    root = Path(root)
    path = root / "safe_reference_logits.npy"
    if path.exists():
        return np.load(path)
    import torch

    return torch.load(
        root / "healthy_verifier/features.pt", weights_only=True, map_location="cpu"
    )["safe_logits"].numpy()
