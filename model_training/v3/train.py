"""Fine-tune the shipped architecture (DINOv2 ViT-S/14, 224 px, nine logits).

Use --init-checkpoint to preserve a verified existing nine-class classifier.
Without it, initialization is explicitly pretrained-backbone retraining.

Training uses the grouped split produced by prepare_data.py. Related source images are grouped before splitting. Historical source-test
images can enter the new training split and are not independent test evidence.
PlantVillage diagnostic images are never trainable. Checkpoints, logs and the
uint8 cache stay outside git; compact metrics are retained in the run directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torchvision.ops import roi_align

LABELS = [
    "Early_blight", "Healthy", "Late_blight", "Leaf Miner", "Magnesium Deficiency",
    "Nitrogen Deficiency", "Pottassium Deficiency", "Spotted Wilt Virus", "Other",
]
HEALTHY = LABELS.index("Healthy")
OTHER = LABELS.index("Other")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default=str(Path.home() / "agribot-data/work"))
    p.add_argument("--runs", default=str(Path.home() / "agribot-data/runs"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--unfreeze-blocks", type=int, default=4, help="-1 unfreezes the whole backbone")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.03)
    p.add_argument("--layer-decay", type=float, default=0.80)
    p.add_argument("--drop-path", type=float, default=0.10)
    p.add_argument("--loss", choices=("ce", "weighted", "logit_adjusted"), default="ce")
    p.add_argument("--sampling", choices=("uniform", "class", "class_source"), default="uniform")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--logit-tau", type=float, default=0.5)
    p.add_argument("--max-class-weight", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="debug/smoke cap per epoch")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--init-checkpoint", help="Verified original checkpoint; preserve its existing head")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--max-disease-drop", type=float, default=0.02)
    p.add_argument("--max-false-healthy-increase", type=float, default=0.005)
    p.add_argument("--stop-after-epoch", type=int, help="Clean checkpoint boundary for interruption/resume verification")
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sampling_weights(meta, args):
    train = [i for i, split in enumerate(meta["split"]) if split == "train"]
    labels = np.asarray([LABELS.index(x) for x in meta["label"]])
    sources = np.asarray(meta["source"], dtype=str)
    groups = meta["group"]
    counts = Counter(labels[i] for i in train)
    weights = np.ones(len(train), np.float64)
    if args.sampling in ("class", "class_source"):
        cw = np.array([math.sqrt(len(train) / (len(LABELS) * counts[label])) for label in labels[train]])
        cw = np.clip(cw, 1.0 / args.max_class_weight, args.max_class_weight)
        weights *= cw
    if args.sampling == "class_source":
        source_counts = Counter(sources[i] for i in train)
        sw = np.array([math.sqrt(len(train) / (len(source_counts) * source_counts[sources[i]])) for i in train])
        weights *= np.clip(sw, 0.5, 2.0)
    group_counts = Counter(groups[i] for i in train)
    if args.sampling != "uniform":
        weights /= np.array([group_counts[groups[i]] for i in train], np.float64)
    weights /= weights.sum()
    prior = np.zeros(len(LABELS), np.float64)
    for w, i in zip(weights, train):
        prior[labels[i]] += w
    prior /= prior.sum()
    return torch.from_numpy(np.asarray(train)), torch.tensor(weights, dtype=torch.float64), prior


def gpu_augment(xb, size, seed_offset=0):
    """xb uint8 NHWC [n,256,256,3] on GPU -> normalised float NCHW."""
    n, s = xb.shape[0], xb.shape[1]
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    area = torch.empty(n, device=x.device).uniform_(0.45, 1.0) * s * s
    logr = torch.empty(n, device=x.device).uniform_(math.log(0.80), math.log(1.25))
    w = (area * logr.exp()).sqrt().clamp(max=s)
    h = (area / logr.exp()).sqrt().clamp(max=s)
    x0 = torch.rand(n, device=x.device) * (s - w)
    y0 = torch.rand(n, device=x.device) * (s - h)
    boxes = torch.stack([torch.arange(n, device=x.device, dtype=torch.float), x0, y0, x0 + w, y0 + h], 1)
    x = roi_align(x, boxes, (size, size), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    angle = torch.where(torch.rand(n, device=x.device) < 0.5,
                        torch.empty(n, device=x.device).uniform_(-math.pi / 15, math.pi / 15),
                        torch.zeros(n, device=x.device))
    fx = torch.where(torch.rand(n, device=x.device) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(n, device=x.device) < 0.05, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = angle.cos() * fx
    theta[:, 0, 1] = -angle.sin() * fy
    theta[:, 1, 0] = angle.sin() * fx
    theta[:, 1, 1] = angle.cos() * fy
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                      mode="bilinear", padding_mode="reflection", align_corners=False)
    u = lambda lo, hi: torch.empty(n, 1, 1, 1, device=x.device).uniform_(lo, hi)
    x = x * u(0.80, 1.20)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.80, 1.20) + m
    g = x.mean(1, keepdim=True)
    x = (x - g) * u(0.75, 1.25) + g
    x = x * torch.empty(n, 3, 1, 1, device=x.device).uniform_(0.96, 1.04)
    kernel = torch.tensor([1, 4, 6, 4, 1], device=x.device, dtype=torch.float)
    kernel = (kernel[:, None] * kernel[None]) / 256
    kernel = kernel.expand(3, 1, 5, 5)
    blur = torch.rand(n, device=x.device) < 0.15
    if blur.any():
        blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)
        x = torch.where(blur.view(-1, 1, 1, 1), blurred, x)
    # Mild erasing; small enough to retain the lesion that defines the label.
    erase = torch.rand(n, device=x.device) < 0.10
    if erase.any():
        ew = (torch.rand(n, device=x.device) * 0.15 + 0.05) * size
        eh = (torch.rand(n, device=x.device) * 0.15 + 0.05) * size
        ex = torch.rand(n, device=x.device) * (size - ew)
        ey = torch.rand(n, device=x.device) * (size - eh)
        ar = torch.arange(size, device=x.device, dtype=torch.float)
        mask = erase.view(-1, 1, 1)
        mask = mask & (ar.view(1, 1, -1) >= ex.view(-1, 1, 1)) & (ar.view(1, 1, -1) < (ex + ew).view(-1, 1, 1))
        mask = mask & (ar.view(1, -1, 1) >= ey.view(-1, 1, 1)) & (ar.view(1, -1, 1) < (ey + eh).view(-1, 1, 1))
        x = torch.where(mask.unsqueeze(1), torch.randn_like(x) * 0.25, x)
    x = (x.clamp(0, 1) - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous()


@torch.no_grad()
def predict(model, images, indices, img_size, bs=128, precision="bf16"):
    model.eval()
    outputs = []
    for start in range(0, len(indices), bs):
        idx = indices[start:start + bs]
        xb = images[idx].permute(0, 3, 1, 2).float() / 255.0
        xb = (F.interpolate(xb, size=(img_size, img_size), mode="bilinear", antialias=True, align_corners=False)
              - MEAN.to(xb.device)) / STD.to(xb.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            outputs.append(model(xb).float())
    return torch.cat(outputs) if outputs else torch.zeros(0, len(LABELS), device=images.device)


def metrics_from_logits(logits, y):
    pred = logits.argmax(1).cpu().numpy()
    truth = y.cpu().numpy()
    macro = float(recall_score(truth, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
    mask_h = truth == HEALTHY
    mask_other = truth != HEALTHY
    healthy_recall = float((pred[mask_h] == HEALTHY).mean()) if mask_h.any() else float("nan")
    false_healthy = float((pred[mask_other] == HEALTHY).mean()) if mask_other.any() else 0.0
    per_class = recall_score(truth, pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    return dict(macro_recall=macro, healthy_recall=healthy_recall, false_healthy_rate=false_healthy,
                accuracy=float((pred == truth).mean()), per_class_recall={l: float(v) for l, v in zip(LABELS, per_class)},
                class_counts={l: int((truth == i).sum()) for i, l in enumerate(LABELS)},
                n=int(len(truth)), healthy_n=int(mask_h.sum())), pred


def main():
    from runner import run
    run(parse_args())


if __name__ == "__main__":
    main()
