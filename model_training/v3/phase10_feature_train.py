"""Compare linear/MLP CE/focal leaf heads without updating the disease model.

All TOM image-level fitting labels stay masked. Human leaf boxes and the
existing leaf corpus supply binary supervision. No fresh AD pixels are used.
"""

from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from torch.nn import functional as F

from campaign_cnn_export import restore
from campaign_cnn_fusion import calibrate
from campaign_pixels import open_pixels
from leaf_feature_expert import LeafFeatureHead
from phase7_species_train import balanced
from training_support import atomic_save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "leaf_feature_heads"
    out.mkdir(exist_ok=True)
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    target = np.array([int(r["health_y"]) for r in rows])
    ev = json.loads((root / "eval_rows.json").read_text())
    for r, y in zip(rows, target):
        r["species_y"] = int(y)
    train = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] == "training" and r["split"] == "train" and target[i] >= 0
        ]
    )
    val = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] == "training" and r["split"] == "val" and target[i] >= 0
        ]
    )
    ids = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] in ev
            or (
                r["set"] == "training"
                and r["split"] in ("train", "val")
                and target[i] >= 0
            )
        ]
    )
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    identity = dict(
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        reference_bundle_sha256=digest(
            root.parent / "export_cnn_safe/pytorch_bundle.pt"
        ),
        ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        feature_scope="Original trained DINO normalized CLS/pre-logit features, strict FP32, no classifier update",
    )
    cache = out / "features.npy"
    meta = out / "feature_identity.json"
    if cache.exists():
        assert json.loads(meta.read_text()) == identity
        features = np.load(cache)
    else:
        reference = (
            restore(root.parent / "export_cnn_safe/pytorch_bundle.pt").cuda().eval()
        )
        reference.requires_grad_(False)
        pixels = open_pixels(root / "cnn_inputs")
        features = np.zeros((len(rows), 384), np.float32)
        original = np.load(root / "reference_logits.npy")
        errors = []
        with torch.inference_mode():
            for start in range(0, len(ids), 128):
                part = ids[start : start + 128]
                x = (
                    torch.from_numpy(np.array(pixels[part], copy=True))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                tokens = reference.backbone.forward_features(x)
                f = reference.backbone.forward_head(tokens, pre_logits=True)
                features[part] = f.cpu().numpy()
                error = float(
                    abs(reference.backbone.head(f).cpu().numpy() - original[part]).max()
                )
                errors.append(error)
                if start % 8192 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage="features",
                                rows=start,
                                total=len(ids),
                                raw_reference_logit_error=error,
                            )
                        ),
                        flush=True,
                    )
        assert max(errors) < 0.002, "Original feature/reference alignment failed"
        np.save(cache, features)
        meta.write_text(json.dumps(identity, indent=2))
        (out / "reference_parity.json").write_text(
            json.dumps(
                dict(passed=True, max_abs_logit_error=max(errors), cases=len(ids)),
                indent=2,
            )
        )
        del reference
    values = torch.from_numpy(features).cuda()
    truth = torch.tensor(target, device="cuda", dtype=torch.float32)
    sampling = torch.tensor(balanced(rows, train), dtype=torch.float64)
    val_weights = balanced(rows, val)
    field = np.array([rows[i]["source"] == "fieldplant" for i in val])
    ranks = []
    for kind, method in itertools.product(["linear", "mlp"], ["bce", "focal"]):
        name = kind + "_" + method
        run = out / name
        run.mkdir(exist_ok=True)
        protocol = dict(
            feature_identity=identity,
            kind=kind,
            method=method,
            epochs=a.epochs,
            negative_cost=4.0,
            seed=379,
            original_classifier_frozen=True,
            fresh_ad_used=False,
        )
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            assert result["protocol"] == protocol
            ranks.append(result)
            continue
        torch.manual_seed(379)
        model = LeafFeatureHead(kind).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        history = []
        best = -float("inf")
        started = time.time()
        for epoch in range(a.epochs):
            model.train()
            order = torch.multinomial(
                sampling,
                len(train),
                replacement=True,
                generator=torch.Generator().manual_seed(37900 + epoch),
            )
            total = 0.0
            batches = 0
            for positions in order.split(1024):
                selected = train[positions.numpy()]
                z = model(values[selected])
                y = truth[selected]
                bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
                loss = (
                    bce
                    * ((1 - torch.exp(-bce)).square() if method == "focal" else 1.0)
                    * torch.where(y > 0, 1.0, 4.0)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            model.eval()
            with torch.inference_mode():
                z = model(values[val]).cpu().numpy()
            prob = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
            y = target[val]
            auc = float(roc_auc_score(y, prob, sample_weight=val_weights))
            field_auc = float(
                roc_auc_score(y[field], prob[field], sample_weight=val_weights[field])
            )
            fpr = float(np.average(prob[y == 0] >= 0.9, weights=val_weights[y == 0]))
            recall = float(np.average(prob[y == 1] >= 0.5, weights=val_weights[y == 1]))
            score = 0.5 * auc + 0.5 * field_auc + 0.1 * recall - 2 * fpr
            if score > best:
                best = score
                atomic_save(
                    dict(
                        kind=kind,
                        model={
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
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
                    weighted_auc=auc,
                    field_auc=field_auc,
                    weighted_false_positive_at_09=fpr,
                    weighted_healthy_recall_at_05=recall,
                )
            )
        state = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()
        with torch.inference_mode():
            z = model(values[val]).cpu().numpy()
        calibration = calibrate(z, target[val], val_weights)
        prob = 1 / (1 + np.exp(-np.clip(z * calibration[0] + calibration[1], -80, 80)))
        result = dict(
            protocol=protocol,
            checkpoint=str(run / "best.pt"),
            checkpoint_sha256=digest(run / "best.pt"),
            calibration=calibration,
            score=best,
            validation_n=len(val),
            field_auc=float(roc_auc_score(target[val][field], prob[field])),
            thresholds={
                str(t): dict(
                    healthy_recall=float((prob[target[val] == 1] >= t).mean()),
                    negative_false_positive=float((prob[target[val] == 0] >= t).mean()),
                    field_healthy_recall=float(
                        (prob[field & (target[val] == 1)] >= t).mean()
                    ),
                    field_negative_false_positive=float(
                        (prob[field & (target[val] == 0)] >= t).mean()
                    ),
                )
                for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
            },
            seconds=time.time() - started,
        )
        (run / "result.json").write_text(json.dumps(result, indent=2))
        (run / "history.json").write_text(json.dumps(history, indent=2))
        ranks.append(result)
        print(
            json.dumps(
                dict(
                    name=name,
                    score=best,
                    field_auc=result["field_auc"],
                    thresholds=result["thresholds"],
                    seconds=result["seconds"],
                )
            ),
            flush=True,
        )
    ranks.sort(key=lambda r: r["score"], reverse=True)
    (out / "ranked.json").write_text(json.dumps(ranks, indent=2))
    print("All leaf feature controls complete", flush=True)


if __name__ == "__main__":
    main()
