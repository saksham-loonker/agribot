"""Fit scoped binary heads on frozen original/adapted DINO features.

Genus labels come from source taxonomy; Healthy labels describe whole images.
No disease head/backbone weights are changed. Five independent heads compare
original, adapted and combined features with grouped validation only.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from campaign_pixels import open_pixels
from phase2_export import restore
from phase7_species_train import species_targets, balanced
from campaign_cnn_fusion import calibrate
from training_support import atomic_save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BinaryHead(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        size = 768 if kind == "mlp_both" else 384
        self.network = (
            nn.Sequential(
                nn.LayerNorm(size),
                nn.Linear(size, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            if kind == "mlp_both"
            else nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 1))
        )

    def forward(self, features):
        x = (
            features
            if self.kind == "mlp_both"
            else (
                features[:, :384]
                if self.kind == "linear_original"
                else features[:, 384:]
            )
        )
        return self.network(x).ravel()


def feature_cache(root, work, model, out):
    identity = dict(
        rows_sha256=digest(work / "rows.json"),
        model_sha256=digest(root / "export_initial/pytorch_bundle.pt"),
    )
    path = out / "features.npy"
    meta = out / "features_identity.json"
    if path.exists():
        assert meta.exists() and json.loads(meta.read_text()) == identity
        return np.load(path)
    rows = json.loads((work / "rows.json").read_text())
    pixels = open_pixels(work)
    features = np.empty((len(rows), 768), np.float32)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            x = torch.from_numpy(
                np.ascontiguousarray(pixels[start : start + 128])
            ).cuda()
            *_, original, adapted = model.components(x, return_features=True)
            features[start : start + len(x)] = (
                torch.cat([original, adapted], 1).cpu().numpy()
            )
            if start % 8192 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="features",
                            work=str(work),
                            rows=start,
                            total=len(rows),
                            seconds=time.time() - started,
                        )
                    ),
                    flush=True,
                )
    np.save(path, features)
    meta.write_text(json.dumps(identity, indent=2))
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "feature_heads"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    model = restore(root / "export_initial/pytorch_bundle.pt").cuda().eval()
    mapping_path = Path("/home/genesis/agribot-data/work/data_manifest.json")
    mapping = json.loads(mapping_path.read_text())
    reserved = json.loads((root / "reserved_manifest.json").read_text())
    held = {r["group"] for r in reserved}
    summaries = {}
    for task, work, kinds in [
        ("species", root / "cnn_inputs", ["linear_original", "mlp_both"]),
        (
            "healthy_scene",
            root / "scene_inputs",
            ["linear_original", "linear_adapted", "mlp_both"],
        ),
    ]:
        task_dir = out / task
        task_dir.mkdir(exist_ok=True)
        rows = json.loads((work / "rows.json").read_text())
        labels = (
            species_targets(rows, mapping)
            if task == "species"
            else np.array([int(r["health_y"]) for r in rows])
        )
        for r, y in zip(rows, labels):
            r["species_y"] = int(y)
        train = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["set"] == "training" and r["split"] == "train" and labels[i] >= 0
            ]
        )
        val = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["set"] == "training" and r["split"] == "val" and labels[i] >= 0
            ]
        )
        assert not ({rows[i]["group"] for i in np.concatenate([train, val])} & held)
        assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
        features = torch.from_numpy(feature_cache(root, work, model, task_dir)).cuda()
        truth = torch.tensor(labels, device="cuda", dtype=torch.float32)
        weights = torch.tensor(balanced(rows, train), dtype=torch.float64)
        primary = np.array(
            [
                rows[i]["source"] == ("plantdoc" if task == "species" else "tom2024")
                for i in val
            ]
        )
        task_rank = []
        for kind in kinds:
            run = task_dir / kind
            run.mkdir(exist_ok=True)
            protocol = dict(
                task=task,
                kind=kind,
                epochs=a.epochs,
                seed=191,
                rows_sha256=digest(work / "rows.json"),
                feature_model_sha256=digest(root / "export_initial/pytorch_bundle.pt"),
                source_taxonomy_sha256=digest(mapping_path),
                negative_cost=2.0,
                reserved_used=False,
            )
            if (run / "result.json").exists():
                result = json.loads((run / "result.json").read_text())
                assert result["protocol"] == protocol
                task_rank.append(result)
                continue
            torch.manual_seed(191)
            head = BinaryHead(kind).cuda()
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=0.001, weight_decay=0.01
            )
            best = -float("inf")
            history = []
            started = time.time()
            for epoch in range(a.epochs):
                head.train()
                order = torch.multinomial(
                    weights,
                    len(train),
                    replacement=True,
                    generator=torch.Generator().manual_seed(19100 + epoch),
                )
                total = 0.0
                batches = 0
                for positions in order.split(1024):
                    ids = train[positions.numpy()]
                    z = head(features[ids])
                    y = truth[ids]
                    loss = (
                        F.binary_cross_entropy_with_logits(z, y, reduction="none")
                        * torch.where(y > 0, 1.0, 2.0)
                    ).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        head.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    total += float(loss.detach())
                    batches += 1
                head.eval()
                with torch.inference_mode():
                    logits = head(features[val]).cpu().numpy()
                probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
                y = labels[val]
                auc = float(roc_auc_score(y[primary], probability[primary]))
                fpr = float((probability[y == 0] >= 0.9).mean())
                recall = float((probability[y == 1] >= 0.5).mean())
                score = auc + 0.05 * recall - fpr
                if score > best:
                    best = score
                    atomic_save(
                        dict(
                            kind=kind,
                            task=task,
                            model={
                                k: v.detach().cpu().clone()
                                for k, v in head.state_dict().items()
                            },
                            protocol=protocol,
                            epoch=epoch,
                            score=score,
                        ),
                        run / "best.pt",
                    )
                history.append(
                    dict(
                        epoch=epoch,
                        loss=total / batches,
                        score=score,
                        primary_domain_auc=auc,
                        negative_fpr_at_09=fpr,
                        positive_recall_at_05=recall,
                    )
                )
            state = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
            head.load_state_dict(state["model"])
            head.eval()
            with torch.inference_mode():
                logits = head(features[val]).cpu().numpy()
            calibration = calibrate(logits, labels[val], balanced(rows, val))
            probability = 1 / (
                1 + np.exp(-np.clip(logits * calibration[0] + calibration[1], -80, 80))
            )
            result = dict(
                protocol=protocol,
                checkpoint=str(run / "best.pt"),
                checkpoint_sha256=digest(run / "best.pt"),
                calibration=calibration,
                score=best,
                primary_domain_auc=float(
                    roc_auc_score(labels[val][primary], probability[primary])
                ),
                validation_n=len(val),
                thresholds={
                    str(t): dict(
                        positive_recall=float(
                            (probability[labels[val] == 1] >= t).mean()
                        ),
                        negative_fpr=float((probability[labels[val] == 0] >= t).mean()),
                    )
                    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
                },
                seconds=time.time() - started,
            )
            (run / "history.json").write_text(json.dumps(history, indent=2))
            (run / "result.json").write_text(json.dumps(result, indent=2))
            task_rank.append(result)
            print(
                json.dumps(
                    dict(
                        task=task,
                        kind=kind,
                        auc=result["primary_domain_auc"],
                        thresholds=result["thresholds"],
                        seconds=result["seconds"],
                    )
                ),
                flush=True,
            )
        task_rank.sort(key=lambda r: r["score"], reverse=True)
        summaries[task] = task_rank[0]
    (out / "selected.json").write_text(json.dumps(summaries, indent=2))
    print(
        json.dumps(
            dict(
                state="complete",
                selected={k: v["checkpoint"] for k, v in summaries.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
