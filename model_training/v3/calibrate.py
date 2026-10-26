"""Fit temperature and the Other-logit bias on the calibration split only.

The objective is app-relevant but crop-level: balanced recall over all nine
classes, with explicit Healthy-recall credit and false-Healthy / Other penalties.
The locked test and the historical diagnostic are never read here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score

from train import LABELS, OTHER, HEALTHY, predict


def ece(probs, labels, bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        avg = conf[mask].mean()
        total += mask.mean() * abs(acc - avg)
        rows.append(dict(lo=float(lo), hi=float(hi), n=int(mask.sum()), accuracy=float(acc), confidence=float(avg)))
    return float(total), rows


def brier(probs, labels):
    onehot = np.eye(len(LABELS))[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--work", default=str(Path.home() / "agribot-data/work"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    work = Path(args.work)
    run_dir = Path(args.run_dir)
    raw_manifest = json.loads((work / "data_manifest.json").read_text())
    dev = [row for row in raw_manifest if row["split"] in ("train", "val", "calib")]
    meta = dict(label=[row["label"] for row in dev], split=[row["split"] for row in dev])
    cache = np.load(work / "cache_256.npy", mmap_mode="r")
    images = torch.from_numpy(np.array(cache)).cuda()
    labels = torch.tensor([LABELS.index(x) for x in meta["label"]]).cuda()
    split = np.asarray(meta["split"])
    calib_idx = torch.tensor(np.where(split == "calib")[0], device="cuda")
    val_idx = torch.tensor(np.where(split == "val")[0], device="cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    import timm
    timm.layers.set_fused_attn(False)
    model = timm.create_model(state["args"]["model"], pretrained=False, num_classes=len(LABELS),
                              img_size=state["args"]["img"]).cuda()
    weights = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        calib_logits = predict(model, images, calib_idx, state["args"]["img"])
        val_logits = predict(model, images, val_idx, state["args"]["img"])
    y = labels[calib_idx]
    log_t = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        t = log_t.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(calib_logits / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(log_t.exp().clamp(0.05, 20.0).detach().cpu())
    logp = F.log_softmax(calib_logits, dim=1)
    rows = []
    for bias in np.arange(-2.0, 2.01, 0.10):
        adjusted = logp.clone()
        adjusted[:, OTHER] += float(bias)
        probs = F.softmax(adjusted / temperature, dim=1).cpu().numpy()
        pred = probs.argmax(1)
        truth = y.cpu().numpy()
        macro = float(recall_score(truth, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
        per_class = recall_score(truth, pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
        known_macro = float(np.mean(per_class[:OTHER]))
        healthy_recall = float((pred[truth == HEALTHY] == HEALTHY).mean())
        false_healthy = float((pred[truth != HEALTHY] == HEALTHY).mean())
        other_recall = float((pred[truth == OTHER] == OTHER).mean())
        known_correct = float((pred[truth != OTHER] == truth[truth != OTHER]).mean())
        # Optimise recognition of supported conditions; Other rejection is a
        # guardrail, not a target that can be maximised by rejecting everything.
        score = known_macro + 0.5 * healthy_recall - 1.0 * false_healthy + 0.25 * other_recall
        if other_recall < 0.90:
            score -= 1.0
        rows.append(dict(other_bias=round(float(bias), 2), macro_recall=macro, known_macro_recall=known_macro,
                         healthy_recall=healthy_recall, false_healthy_rate=false_healthy, other_recall=other_recall,
                         known_accuracy=known_correct, score=score))
    chosen = max(rows, key=lambda row: (row["score"], -abs(row["other_bias"])))
    adjusted = logp.clone()
    adjusted[:, OTHER] += chosen["other_bias"]
    probs = F.softmax(adjusted / temperature, dim=1).cpu().numpy()
    truth = y.cpu().numpy()
    calibration_ece, reliability = ece(probs, truth)
    val_logp = F.log_softmax(val_logits, dim=1)
    val_logp[:, OTHER] += chosen["other_bias"]
    val_probs = F.softmax(val_logp / temperature, dim=1).cpu().numpy()
    val_truth = labels[val_idx].cpu().numpy()
    val_ece, val_reliability = ece(val_probs, val_truth)
    result = dict(
        run_dir=str(run_dir), checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=__import__("hashlib").sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        labels=LABELS, num_classes=len(LABELS), temperature=temperature,
        other_bias=float(chosen["other_bias"]), energy_gate=False, energy_reject_below=None,
        calibration_split=dict(n=int(len(y)), by_label={l: int((truth == i).sum()) for i, l in enumerate(LABELS)}),
        selection=dict(objective="known_macro_recall + 0.5*healthy_recall - false_healthy_rate + 0.25*other_recall (other_recall>=0.90 constraint)",
                       chosen=chosen, sweep=rows),
        calibration_metrics=dict(macro_recall=chosen["macro_recall"], healthy_recall=chosen["healthy_recall"],
                                 false_healthy_rate=chosen["false_healthy_rate"], other_recall=chosen["other_recall"],
                                 nll=float(F.nll_loss(torch.from_numpy(np.log(np.clip(probs, 1e-12, 1) + 1e-12)),
                                                      torch.from_numpy(truth))),
                                 brier=brier(probs, truth), ece15=calibration_ece, reliability=reliability),
        validation_metrics=dict(n=int(len(val_truth)), ece15=val_ece, reliability=val_reliability),
        provenance=dict(selection_used_locked_test=False, selection_used_historical_diagnostic=False,
                        data_manifest_sha256=state.get("data_manifest_sha256")),
    )
    target = Path(args.out) if args.out else run_dir / "calibration.json"
    target.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("temperature", "other_bias", "calibration_metrics")}, indent=1))


if __name__ == "__main__":
    main()
