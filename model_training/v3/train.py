"""Fine-tune the shipped architecture (DINOv2 ViT-S/14, 224 px, nine logits).

The shipped TFLite file is not trainable. This is a documented
"pretrained-backbone retraining" initialized from timm DINOv2 weights.

Training uses the grouped split produced by prepare_data.py. No source-test
image or PlantVillage diagnostic image is trainable. Checkpoints, logs and the
uint8 cache stay outside git; compact metrics are committed by copy_results.py.
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
    p.add_argument("--loss", choices=("ce", "weighted", "logit_adjusted"), default="logit_adjusted")
    p.add_argument("--sampling", choices=("uniform", "class", "class_source"), default="class_source")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--logit-tau", type=float, default=0.5)
    p.add_argument("--max-class-weight", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="debug/smoke cap per epoch")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--compile", action="store_true")
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
def predict(model, images, indices, img_size, bs=128):
    model.eval()
    outputs = []
    for start in range(0, len(indices), bs):
        idx = indices[start:start + bs]
        xb = images[idx].permute(0, 3, 1, 2).float() / 255.0
        xb = (F.interpolate(xb, size=(img_size, img_size), mode="bilinear", antialias=True, align_corners=False)
              - MEAN.to(xb.device)) / STD.to(xb.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
                n=int(len(truth)), healthy_n=int(mask_h.sum())), pred


def make_param_groups(model, args, unfreeze_blocks):
    blocks = getattr(model, "blocks", None)
    n_blocks = len(blocks) if blocks is not None else 0
    groups = []
    head_params = []
    backbone_named = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append((name, param))
        else:
            backbone_named.append((name, param))
    if head_params:
        groups.append(dict(params=[p for _, p in head_params], lr=args.lr_head, weight_decay=args.wd, name="head"))
    if backbone_named:
        if unfreeze_blocks < 0:
            depth = {name: n_blocks + 1 for name, _ in backbone_named}
        else:
            depth = {}
            for name, _ in backbone_named:
                d = n_blocks - 1
                if name.startswith("blocks."):
                    d = int(name.split(".")[1])
                elif name.startswith(("norm.", "patch_embed.", "cls_token", "pos_embed")):
                    d = 0
                depth[name] = d
        buckets = {}
        for name, param in backbone_named:
            if unfreeze_blocks < 0:
                scale = 1.0
            else:
                first = n_blocks - unfreeze_blocks - 1
                scale = args.layer_decay ** max(0, depth[name] - first) if depth[name] >= first else 0.0
            lr = args.lr_backbone * scale
            key = round(lr, 12)
            buckets.setdefault(key, []).append(param)
        for lr, params in sorted(buckets.items(), reverse=True):
            if lr > 0:
                groups.append(dict(params=params, lr=lr, weight_decay=args.wd, name=f"backbone_{lr:.2e}"))
    return groups


def main():
    args = parse_args()
    set_seed(args.seed)
    work = Path(args.work)
    run_dir = Path(args.runs) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
    raw_manifest = json.loads((work / "data_manifest.json").read_text())
    meta_sha = __import__("hashlib").sha256((work / "data_manifest.json").read_bytes()).hexdigest()
    dev = [row for row in raw_manifest if row["split"] in ("train", "val", "calib")]
    assert all(row.get("cache_index") == i for i, row in enumerate(dev)), "cache index mismatch"
    meta = dict(label=[row["label"] for row in dev], split=[row["split"] for row in dev],
                source=[row["source"] for row in dev], group=[row["group"] for row in dev])
    cache = np.load(work / "cache_256.npy", mmap_mode="r")
    images = torch.from_numpy(np.array(cache)).cuda()
    labels = torch.tensor([LABELS.index(x) for x in meta["label"]]).cuda()
    splits = np.asarray(meta["split"])
    train_all, weights, prior = sampling_weights(meta, args)
    train_all = train_all.cuda()
    weights = weights.cuda()
    val_idx = torch.tensor(np.where(splits == "val")[0], device="cuda")
    calib_idx = torch.tensor(np.where(splits == "calib")[0], device="cuda")
    print(f"cache {tuple(images.shape)} train {len(train_all)} val {len(val_idx)} calib {len(calib_idx)}", flush=True)
    print("effective prior", {l: round(float(p), 4) for l, p in zip(LABELS, prior)}, flush=True)

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(LABELS),
                              img_size=args.img, drop_path_rate=args.drop_path)
    model = model.cuda()
    model.set_grad_checkpointing(False)
    if args.compile:
        model = torch.compile(model)

    start_epoch = 0
    best_score = -1e9
    if args.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_score = state["best_score"]
        print(f"resumed epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        phase_warmup = epoch < args.warmup_epochs
        for name, param in model.named_parameters():
            if phase_warmup:
                param.requires_grad = name.startswith("head.")
            else:
                param.requires_grad = True
        if not phase_warmup and args.unfreeze_blocks >= 0:
            blocks = getattr(model, "blocks", [])
            first = max(0, len(blocks) - args.unfreeze_blocks)
            for name, param in model.named_parameters():
                if name.startswith("blocks."):
                    idx = int(name.split(".")[1])
                    if idx < first:
                        param.requires_grad = False
                elif name.startswith(("patch_embed.", "pos_embed", "cls_token")):
                    param.requires_grad = False
        groups = make_param_groups(model, args, -1 if phase_warmup else args.unfreeze_blocks)
        opt = torch.optim.AdamW(groups, weight_decay=args.wd, fused=True)
        steps_per_epoch = len(train_all) // args.bs
        if args.max_steps:
            steps_per_epoch = min(steps_per_epoch, args.max_steps)
        if phase_warmup:
            remaining = args.warmup_epochs - epoch
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps_per_epoch * 0.3)))
        else:
            total = max(1, (args.epochs - args.warmup_epochs) * steps_per_epoch)
            done = (epoch - args.warmup_epochs) * steps_per_epoch
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, (done + s) / total))))
        ema = timm.utils.ModelEmaV3(model, decay=0.999, use_warmup=False)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        if args.loss == "weighted":
            counts = np.bincount(labels[train_all].cpu().numpy(), minlength=len(LABELS))
            cw = np.sqrt(len(train_all) / (len(LABELS) * np.maximum(counts, 1)))
            cw = np.clip(cw / cw.mean(), 1 / args.max_class_weight, args.max_class_weight)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device="cuda"),
                                          label_smoothing=args.label_smoothing)
        log_prior = torch.tensor(np.log(np.clip(prior, 1e-6, None)), dtype=torch.float32, device="cuda")
        model.train()
        t0 = time.time()
        running = 0.0
        order = torch.multinomial(weights, steps_per_epoch * args.bs, replacement=True)
        for step in range(steps_per_epoch):
            idx = train_all[order[step * args.bs:(step + 1) * args.bs]]
            xb = gpu_augment(images[idx], args.img)
            y = labels[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                adjusted = logits.float() + args.logit_tau * log_prior if args.loss == "logit_adjusted" else logits.float()
                loss = loss_fn(adjusted, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            running += float(loss.detach())
        eval_model = ema.module
        val_logits = predict(eval_model, images, val_idx, args.img)
        val_metrics, _ = metrics_from_logits(val_logits, labels[val_idx])
        score = val_metrics["macro_recall"] + 0.5 * val_metrics["healthy_recall"] - 0.5 * val_metrics["false_healthy_rate"]
        row = dict(epoch=epoch, phase="warmup" if phase_warmup else "finetune", seconds=round(time.time() - t0, 1),
                   loss=running / max(1, steps_per_epoch), val_score=score, **val_metrics,
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("epoch", "phase", "seconds", "loss", "val_score", "macro_recall",
                                              "healthy_recall", "false_healthy_rate", "peak_vram_gb")}), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"model": eval_model.state_dict(), "epoch": epoch, "score": score,
                        "args": vars(args), "labels": LABELS, "data_manifest_sha256": meta_sha,
                        "effective_prior": prior.tolist()}, run_dir / "best.pt")
        torch.save({"model": eval_model.state_dict(), "epoch": epoch, "best_score": best_score,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "args": vars(args), "labels": LABELS, "data_manifest_sha256": meta_sha,
                    "effective_prior": prior.tolist()}, run_dir / "last.pt")
        (run_dir / "meta.json").write_text(json.dumps(dict(
            model=args.model, img=args.img, num_classes=len(LABELS), classes=LABELS,
            best_epoch=int(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["epoch"]),
            args=vars(args), data_manifest_sha256=meta_sha,
            environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(0), timm=timm.__version__, platform=platform.platform()),
        ), indent=1))
    print("done", run_dir, "best", best_score, flush=True)


if __name__ == "__main__":
    main()
