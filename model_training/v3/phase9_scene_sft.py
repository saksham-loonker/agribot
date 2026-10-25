"""Supervised whole-image Healthy adaptation of a copied DINO tail.

The original nine-class disease model is never updated. Frozen prefix tokens
are cached for full scenes and horizontal training flips. Validation stays FP32;
FP16 token storage and BF16 computation apply only to training. The consumed
phase-eight reservation is excluded from fitting and is no longer acceptance.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from torch import nn
from torch.nn import functional as F

from campaign_adapter import prefix
from campaign_cnn_fusion import calibrate
from campaign_pixels import open_pixels
from phase7_species_train import balanced
from training_support import atomic_save, capture_rng, load_initial_weights, restore_rng
from train import LABELS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reference(encoder):
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=9,
        img_size=224,
    ).eval()
    load_initial_weights(
        model, encoder, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    return model.requires_grad_(False)


class SceneExpert(nn.Module):
    def __init__(self, backbone, depth, dropout=0.0):
        super().__init__()
        self.tail = copy.deepcopy(backbone.blocks[-depth:]).requires_grad_(True)
        self.norm = copy.deepcopy(backbone.norm).requires_grad_(True)
        self.head = nn.Linear(backbone.num_features, 1)
        self.dropout = dropout
        with torch.no_grad():
            other = [i for i in range(9) if i != 1]
            self.head.weight.copy_(
                backbone.head.weight[1:2]
                - backbone.head.weight[other].mean(0, keepdim=True)
            )
            self.head.bias.copy_(
                backbone.head.bias[1:2] - backbone.head.bias[other].mean() - math.log(8)
            )

    def forward(self, tokens):
        for block in self.tail:
            tokens = block(tokens)
        features = F.dropout(
            self.norm(tokens)[:, 0], p=self.dropout, training=self.training
        )
        return self.head(features).ravel()


class SceneVerifier(nn.Module):
    def __init__(self, state, calibration):
        super().__init__()
        assert digest(state["encoder"]) == state["configuration"]["encoder_sha256"]
        self.backbone = reference(state["encoder"])
        self.depth = state["depth"]
        self.expert = SceneExpert(
            self.backbone, self.depth, state["configuration"].get("dropout", 0.0)
        )
        self.expert.load_state_dict(state["model"], strict=True)
        self.register_buffer(
            "calibration", torch.tensor(calibration, dtype=torch.float32)
        )

    def logits(self, nhwc):
        return (
            self.expert(prefix(self.backbone, nhwc.permute(0, 3, 1, 2), self.depth))
            * self.calibration[0]
            + self.calibration[1]
        )

    def forward(self, nhwc):
        return self.logits(nhwc).sigmoid()


def token_cache(root, backbone, encoder, depth, train, val):
    directory = root / f"scene_tokens_d{depth}"
    directory.mkdir(exist_ok=True)
    identity = dict(
        rows_sha256=digest(root / "scene_inputs/rows.json"),
        encoder_sha256=digest(encoder),
        depth=depth,
        train_rows=train.tolist(),
        val_rows=val.tolist(),
        training_dtype="float16",
        validation_dtype="float32",
        training_views=["full_image_letterbox", "horizontal_flip_after_letterbox"],
    )
    metadata = directory / "identity.json"
    if metadata.exists():
        assert json.loads(metadata.read_text()) == identity
    else:
        metadata.write_text(json.dumps(identity, indent=2))
    progress = directory / "progress.json"
    done = (
        json.loads(progress.read_text()) if progress.exists() else dict(train=0, val=0)
    )
    pixels = open_pixels(root / "scene_inputs")
    caches = {}
    with torch.inference_mode():
        for split, ids, views, dtype in [
            ("train", train, 2, np.float16),
            ("val", val, 1, np.float32),
        ]:
            path = directory / (split + ".npy")
            shape = (len(ids) * views, 257, 384)
            if path.exists():
                cache = np.lib.format.open_memmap(path, mode="r+")
                assert cache.shape == shape and cache.dtype == dtype
            else:
                assert done[split] == 0
                cache = np.lib.format.open_memmap(
                    path, mode="w+", dtype=dtype, shape=shape
                )
            for start in range(done[split], len(ids), 64):
                selected = ids[start : start + 64]
                x = (
                    torch.from_numpy(np.ascontiguousarray(pixels[selected]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                for view in range(views):
                    z = prefix(backbone, x.flip(-1) if view else x, depth)
                    assert torch.isfinite(z).all()
                    cache[
                        start * views + view : (start + len(selected)) * views : views
                    ] = (z.cpu().numpy().astype(dtype))
                cache.flush()
                done[split] = start + len(selected)
                temp = progress.with_suffix(".tmp")
                temp.write_text(json.dumps(done))
                temp.replace(progress)
                if start % 2048 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage="prefix_cache",
                                split=split,
                                complete=done[split],
                                total=len(ids),
                            )
                        ),
                        flush=True,
                    )
            caches[split] = cache
    return caches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--loss", choices=["ce", "focal"], required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--backbone-lr", type=float, default=0.00002)
    ap.add_argument("--head-lr", type=float, default=0.0001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--mixup-alpha", type=float, default=0.0)
    ap.add_argument("--run-name")
    ap.add_argument(
        "--encoder",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    a = ap.parse_args()
    assert 1 <= a.depth <= 12 and a.epochs > 0 and a.batch > 0
    root = Path(a.root)
    out = root / (a.run_name or f"scene_sft_d{a.depth}_{a.loss}")
    out.mkdir(exist_ok=True)
    rows = json.loads((root / "scene_inputs/rows.json").read_text())
    target = np.array([int(r["health_y"]) for r in rows])
    for r, y in zip(rows, target):
        r["species_y"] = int(y)
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    assert set(target[train]) == set(target[val]) == {0, 1}
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    held = json.loads((root / "reserved_manifest.json").read_text())
    assert not ({r["path"] for r in rows} & {r["path"] for r in held})
    assert not ({r["group"] for r in rows} & {r["group"] for r in held})
    config = dict(
        rows_sha256=digest(root / "scene_inputs/rows.json"),
        encoder_sha256=digest(a.encoder),
        depth=a.depth,
        loss=a.loss,
        negative_cost=4.0,
        focal_gamma=2.0 if a.loss == "focal" else 0.0,
        epochs=a.epochs,
        batch=a.batch,
        seed=251,
        backbone_lr=a.backbone_lr,
        head_lr=a.head_lr,
        weight_decay=a.weight_decay,
        scope="Image-level weak Healthy supervision, not verified whole-plant diagnosis; original nine-class branch frozen.",
        reservation="Former phase-eight reservation is consumed regression; excluded from fitting and checkpoint selection, not fresh acceptance.",
    )
    assert (
        a.backbone_lr > 0
        and a.head_lr > 0
        and a.weight_decay >= 0
        and 0 <= a.dropout < 1
    )
    if a.dropout:
        config["dropout"] = a.dropout
    assert a.mixup_alpha >= 0
    if a.mixup_alpha:
        assert (
            a.depth == 12
        ), "Image-equivalent token mixup requires the zero-transformer prefix"
        config["mixup_alpha"] = a.mixup_alpha
    protocol = out / "protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == config
    else:
        protocol.write_text(json.dumps(config, indent=2))
    if (out / "result.json").exists():
        print("Already complete", out)
        return
    torch.set_num_threads(4)
    torch.manual_seed(251)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    backbone = reference(a.encoder).cuda()
    if a.mixup_alpha:
        assert isinstance(backbone.norm_pre, nn.Identity)
    model = SceneExpert(backbone, a.depth, a.dropout).cuda()
    caches = token_cache(root, backbone, a.encoder, a.depth, train, val)
    train_tokens = torch.from_numpy(np.array(caches["train"])).cuda()
    val_tokens = torch.from_numpy(np.array(caches["val"])).cuda()
    del backbone
    weights = torch.tensor(balanced(rows, train), dtype=torch.float64)
    truth = torch.tensor(target, device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=list(model.tail.parameters()) + list(model.norm.parameters()),
                lr=config["backbone_lr"],
            ),
            dict(params=model.head.parameters(), lr=config["head_lr"]),
        ],
        weight_decay=a.weight_decay,
    )
    start_epoch = 0
    best = -float("inf")
    history = []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", weights_only=True, map_location="cpu")
        assert state["configuration"] == config
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start_epoch = state["epoch"] + 1
        best = state["best_score"]
        history = json.loads((out / "history.json").read_text())
    primary = np.array([rows[i]["source"] == "tom2024" for i in val])
    started = time.time()

    def evaluate():
        model.eval()
        with torch.inference_mode():
            return (
                torch.cat([model(x) for x in val_tokens.split(a.batch)]).cpu().numpy()
            )

    for epoch in range(start_epoch, a.epochs):
        model.train()
        order = torch.multinomial(
            weights,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(25100 + epoch),
        )
        total = 0.0
        batches = 0
        for positions in order.split(a.batch):
            flip = torch.randint(0, 2, (len(positions),), device="cuda")
            tokens = train_tokens[positions.cuda() * 2 + flip].float()
            y = truth[train[positions.numpy()]]
            mix = None
            if a.mixup_alpha:
                permutation = torch.randperm(len(tokens), device="cuda")
                mixture = (
                    torch.distributions.Beta(a.mixup_alpha, a.mixup_alpha)
                    .sample((len(tokens),))
                    .cuda()
                )
                tokens = tokens * mixture[:, None, None] + tokens[permutation] * (
                    1 - mixture[:, None, None]
                )
                mix = (mixture, y[permutation])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model(tokens)

            def per_target(target):
                bce = F.binary_cross_entropy_with_logits(
                    z.float(), target, reduction="none"
                )
                return (
                    bce
                    * (1 - torch.exp(-bce)).pow(config["focal_gamma"])
                    * torch.where(target > 0, 1.0, config["negative_cost"])
                )

            terms = per_target(y)
            if mix is not None:
                mixture, other = mix
                terms = mixture * terms + (1 - mixture) * per_target(other)
            loss = terms.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        for group, initial_lr in zip(
            optimizer.param_groups, [config["backbone_lr"], config["head_lr"]]
        ):
            group["lr"] = initial_lr * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
            )
        z = evaluate()
        p = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        y = target[val]
        auc = float(roc_auc_score(y[primary], p[primary]))
        fpr = float((p[y == 0] >= 0.9).mean())
        recall = float((p[y == 1] >= 0.5).mean())
        score = auc + 0.1 * recall - 2 * fpr
        state = dict(
            architecture="DinoSceneExpert",
            encoder=a.encoder,
            depth=a.depth,
            configuration=config,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            epoch=epoch,
            score=score,
        )
        if score > best:
            best = score
            atomic_save(state, out / "best.pt")
        atomic_save(
            dict(
                state,
                optimizer=optimizer.state_dict(),
                rng=capture_rng(),
                best_score=best,
            ),
            out / "last.pt",
        )
        record = dict(
            epoch=epoch,
            loss=total / batches,
            score=score,
            field_auc=auc,
            negative_fpr_at_09=fpr,
            healthy_recall_at_05=recall,
            seconds=time.time() - started,
        )
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    selected = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["model"])
    z = evaluate()
    calibration = calibrate(z, target[val], balanced(rows, val))
    p = 1 / (1 + np.exp(-np.clip(z * calibration[0] + calibration[1], -80, 80)))
    result = dict(
        configuration=config,
        best_checkpoint_sha256=digest(out / "best.pt"),
        calibration=calibration,
        validation_n=len(val),
        field_auc=float(roc_auc_score(target[val][primary], p[primary])),
        thresholds={
            str(t): dict(
                healthy_recall=float((p[target[val] == 1] >= t).mean()),
                negative_false_positive=float((p[target[val] == 0] >= t).mean()),
            )
            for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        },
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        seconds=time.time() - started,
    )
    np.savez(out / "validation_logits.npz", indices=val, logits=z)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
