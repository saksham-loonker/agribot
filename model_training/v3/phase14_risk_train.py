"""Separate high-precision-tail, ranking and label-reward head controls.

Uses only pinned Apache SigLIP2 features and human labels. The reward controls
are one-step supervised-label bandits, not human feedback or sequential RL.
No original disease parameters, restricted teachers or cold acceptance data.
"""

from pathlib import Path
import argparse
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from leaf_feature_expert import LeafFeatureHead
from permissive_encoder import digest
from phase7_species_train import balanced
from training_support import atomic_save


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model-root")
    ap.add_argument("--field-sources")
    a = ap.parse_args()
    root = Path(a.model_root) if a.model_root else Path(a.campaign) / "phase13_siglip"
    out = root / "risk_heads"
    out.mkdir(exist_ok=True)
    source = root / "siglip2_feature_heads"
    identity = read(source / "feature_identity.json")
    assert (
        identity["family"] == "siglip2" and identity["restricted_teacher_used"] is False
    )
    assert read(source / "restored_encoder_parity.json")["passed"]
    if not (out / "encoder.pt").exists():
        os.link(source / "encoder.pt", out / "encoder.pt")
    assert digest(out / "encoder.pt") == identity["encoder_sha256"]
    rows = read(root / "cnn_inputs/rows.json")
    assert digest(root / "cnn_inputs/rows.json") == identity["rows_sha256"]
    y = np.array([int(r["health_y"]) for r in rows])
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    for r, target in zip(rows, y):
        r["species_y"] = int(target)
    sampling = torch.tensor(balanced(rows, train), dtype=torch.float64)
    weights = balanced(rows, val)
    field_sources = a.field_sources.split(",") if a.field_sources else ["tom2024"]
    field = np.array([rows[i]["source"] in field_sources for i in val])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = torch.from_numpy(np.load(source / "features.npy")).cuda()
    truth = torch.tensor(y, device="cuda", dtype=torch.float32)
    warm_path = Path(read(source / "ranked.json")[0]["checkpoint"])
    warm = torch.load(warm_path, weights_only=True, map_location="cpu")
    assert warm["kind"] == "mlp"
    reference = LeafFeatureHead("mlp", 768).cuda().eval().requires_grad_(False)
    reference.load_state_dict(warm["model"], strict=True)
    with torch.inference_mode():
        reference_logits = reference(features)
    ranks = []
    for mode in [
        "cvar_bce",
        "partial_auc",
        "reinforce_label_reward",
        "expected_logit_reward",
    ]:
        run = out / mode
        run.mkdir(exist_ok=True)
        protocol = dict(
            feature_identity=identity,
            task_scope="primary_image",
            primary_source="tom2024",
            original_classifier_frozen=True,
            restricted_teacher_used=False,
            method=mode,
            epochs=a.epochs,
            seed=613,
            learning_rate=0.0002,
            warm_head_sha256=digest(warm_path),
            negative_cost=4.0,
            tail_fraction=0.1,
            ranking_margin=1.0,
            reward=dict(
                true_healthy=2.0,
                missed_healthy=-1.0,
                false_healthy=-12.0,
                true_negative=1.0,
            ),
            reward_type="One-step human-label-derived binary reward; no RLHF or sequential environment",
            checkpoint_selection="Grouped internal validation recall at weighted FPR<=0.005; no app or cold-source fitting",
            fresh_ad_used=(root / "AD_ADMISSION.json").exists(),
        )
        if a.field_sources:
            protocol["field_sources"] = field_sources
        if protocol["fresh_ad_used"]:
            protocol["ad_role"] = (
                "Consumed grouped training/internal validation; no independent acceptance"
            )
        if (run / "result.json").exists():
            result = read(run / "result.json")
            assert result["protocol"] == protocol
            ranks.append(result)
            continue
        torch.manual_seed(613)
        model = LeafFeatureHead("mlp", 768).cuda()
        model.load_state_dict(warm["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
        history, best = [], -float("inf")

        def score_logits(z):
            recalls = []
            for mask in [field, np.ones(len(val), bool)]:
                indices = np.where(mask & (y[val] == 0))[0]
                order = indices[np.argsort(-z[indices], kind="stable")]
                cumulative = np.cumsum(weights[order]) / weights[indices].sum()
                allowed = order[cumulative <= 0.005]
                # Exclude the first disallowed score, including all ties.
                # Float32 nextafter avoids NumPy casting a double epsilon away.
                threshold = np.nextafter(z[order[len(allowed)]], np.float32(np.inf))
                positive = mask & (y[val] == 1)
                recalls.append(
                    float(
                        np.average(z[positive] >= threshold, weights=weights[positive])
                    )
                )
            pauc = float(roc_auc_score(y[val], z, sample_weight=weights, max_fpr=0.01))
            return recalls[0] + 0.25 * recalls[1] + 0.1 * pauc

        for epoch in range(-1, a.epochs):
            total, batches = 0.0, 0
            if epoch >= 0:
                model.train()
                order = torch.multinomial(
                    sampling,
                    len(train),
                    replacement=True,
                    generator=torch.Generator().manual_seed(61300 + epoch),
                )
                for positions in order.split(1024):
                    ids = train[positions.numpy()]
                    target = truth[ids]
                    z = model(features[ids])
                    bce = F.binary_cross_entropy_with_logits(
                        z, target, reduction="none"
                    )
                    anchor = (bce * torch.where(target > 0, 1.0, 4.0)).mean()
                    if mode == "cvar_bce":
                        negative = bce[target == 0]
                        hard = negative.topk(
                            max(1, math.ceil(len(negative) * 0.1))
                        ).values
                        loss = bce[target > 0].mean() + 4 * hard.mean() + 0.25 * anchor
                    elif mode == "partial_auc":
                        negative = z[target == 0]
                        hard = negative.topk(min(64, len(negative))).values
                        positive = z[target > 0]
                        loss = (
                            F.softplus(1.0 + hard[None, :] - positive[:, None]).mean()
                            + 0.5 * anchor
                        )
                    else:
                        probability = z.sigmoid().clamp(1e-6, 1 - 1e-6)
                        correct_reward = torch.where(target > 0, 2.0, -12.0)
                        reject_reward = torch.where(target > 0, -1.0, 1.0)
                        expected = (
                            probability * correct_reward
                            + (1 - probability) * reject_reward
                        )
                        if mode == "reinforce_label_reward":
                            action = torch.bernoulli(probability.detach())
                            reward = (
                                action * correct_reward + (1 - action) * reject_reward
                            )
                            logp = action * probability.log() + (
                                1 - action
                            ) * torch.log1p(-probability)
                            reward_loss = -((reward - expected.detach()) * logp).mean()
                        else:
                            reward_loss = -expected.mean()
                        teacher = reference_logits[ids].sigmoid().clamp(1e-6, 1 - 1e-6)
                        kl = (
                            teacher * (teacher.log() - probability.log())
                            + (1 - teacher)
                            * (torch.log1p(-teacher) - torch.log1p(-probability))
                        ).mean()
                        loss = reward_loss + 0.5 * anchor + 0.05 * kl
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
                z = model(features[val]).cpu().numpy()
            score = score_logits(z)
            if score > best:
                best = score
                atomic_save(
                    dict(
                        kind="mlp",
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
            history.append(dict(epoch=epoch, loss=total / max(1, batches), score=score))
        state = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            z = model(features[val]).cpu().numpy()
        calibration = calibrate(z, y[val], weights)
        restored = LeafFeatureHead("mlp", 768).cuda().eval().requires_grad_(False)
        restored.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            actual = restored(features[val]).cpu().numpy()
        error = float(np.max(np.abs(actual - z)))
        assert error < 0.002, "Restored risk head changed validation logits"
        (run / "restored_head_parity.json").write_text(
            json.dumps(
                dict(passed=True, cases=len(val), max_abs_logit_error=error), indent=2
            )
        )
        del restored
        result = dict(
            protocol=protocol,
            checkpoint=str(run / "best.pt"),
            checkpoint_sha256=digest(run / "best.pt"),
            calibration=calibration,
            score=best,
            selected_epoch=state["epoch"],
            field_auc=float(roc_auc_score(y[val][field], z[field])),
            validation_n=len(val),
        )
        (run / "history.json").write_text(json.dumps(history, indent=2))
        (run / "result.json").write_text(json.dumps(result, indent=2))
        ranks.append(result)
        print(
            json.dumps(
                dict(
                    mode=mode,
                    score=best,
                    selected_epoch=state["epoch"],
                    field_auc=result["field_auc"],
                )
            ),
            flush=True,
        )
    ranks.sort(key=lambda r: r["score"], reverse=True)
    (out / "ranked.json").write_text(json.dumps(ranks, indent=2))


if __name__ == "__main__":
    main()
