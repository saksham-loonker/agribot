"""Fit monotone five-input calibration, adding MobileCLIP image evidence."""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from leaf_feature_expert import LeafFeatureHead
from phase7_species_train import balanced


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--model-root")
    ap.add_argument("--include-admitted-training", action="store_true")
    ap.add_argument(
        "--head-family", choices=["mobileclip", "siglip2"], default="mobileclip"
    )
    a = ap.parse_args()
    p = Path(a.campaign)
    root = Path(a.model_root) if a.model_root else p / "phase11_scene"
    if a.include_admitted_training:
        assert (
            a.head_family == "siglip2"
        ), "Admitted training is only supported for the permitted encoder"
    out = root / (
        "logit_fusion_admitted" if a.include_admitted_training else "logit_fusion"
    )
    out.mkdir(exist_ok=True)
    old = p / "phase6_roi/scene_sft_d12_focal_fusion"
    rows_path = root / "cnn_inputs/rows.json"
    rows = json.loads(rows_path.read_text())
    old_identity = json.loads((old / "input_identity.json").read_text())
    old_rows_path = p / "phase6_roi/scene_inputs/cnn_inputs/rows.json"
    if not old_rows_path.exists():
        old_rows_path = p / "phase11_scene/cnn_inputs/rows.json"
    old_rows = json.loads(old_rows_path.read_text())
    assert digest(old_rows_path) == old_identity["scene_rows_sha256"]
    assert rows[: len(old_rows)] == old_rows
    # Calibrate on the original human corpus only. Newly admitted families
    # remain available for the separate internal app validation.
    if not a.include_admitted_training:
        rows = rows[: len(old_rows)]
    selected = json.loads(
        (root / (a.head_family + "_feature_heads") / "ranked.json").read_text()
    )[0]
    directory = Path(selected["checkpoint"]).parent
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    assert state["protocol"]["task_scope"] == "primary_image"
    if a.head_family == "siglip2":
        assert state["protocol"]["feature_identity"]["restricted_teacher_used"] is False
    dimension = 768 if a.head_family == "siglip2" else 512
    model = LeafFeatureHead(state["kind"], dimension).eval()
    model.load_state_dict(state["model"], strict=True)
    features = np.load(root / (a.head_family + "_feature_heads") / "features.npy")[
        : len(rows)
    ]
    torch.set_num_threads(4)
    with torch.inference_mode():
        mobile = model(torch.from_numpy(features)).numpy()
    mobile = mobile * selected["calibration"][0] + selected["calibration"][1]
    legacy = np.load(old / "calibrated_logits.npy")
    if a.include_admitted_training:
        from phase14_admitted_features import extract

        legacy = np.concatenate([legacy, extract(p, root)], 0)
    logits = np.concatenate([legacy, mobile[:, None]], 1)
    identity = dict(
        legacy_input_identity=old_identity,
        legacy_logits_sha256=digest(old / "calibrated_logits.npy"),
        mobile_head_sha256=digest(directory / "best.pt"),
        mobile_calibration_sha256=digest(directory / "result.json"),
        mobile_encoder_sha256=state["protocol"]["feature_identity"]["encoder_sha256"],
        rows_sha256=digest(rows_path),
    )
    labels = np.array([int(r["health_y"]) for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    cal = np.array(
        [
            i
            for i in val
            if int(
                hashlib.sha256(
                    ("calibration211:" + rows[i]["group"]).encode()
                ).hexdigest()[:8],
                16,
            )
            % 100
            < 65
        ]
    )
    chosen = set(cal)
    selection = np.array([i for i in val if i not in chosen])
    if a.include_admitted_training:
        # Never use the new 180 validation photos to fit these coefficients.
        cal = np.concatenate(
            [
                cal[cal < len(old_rows)],
                np.array(
                    [
                        i
                        for i, r in enumerate(rows)
                        if i >= len(old_rows) and r["split"] == "train"
                    ]
                ),
            ]
        )
        selection = np.array(
            [i for i in val if i < len(old_rows) and i not in set(cal)]
        )
        identity["admitted_training_n"] = int(np.sum(cal >= len(old_rows)))
        identity["admitted_validation_fitted"] = False
        identity["admitted_legacy_features_sha256"] = digest(
            root / "admitted_legacy_features/calibrated_logits.npy"
        )
    assert not ({rows[i]["group"] for i in cal} & {rows[i]["group"] for i in selection})
    for r, y in zip(rows, labels):
        r["species_y"] = int(y)
    x = torch.tensor(np.clip(logits[cal], -10, 10), dtype=torch.float64)
    y = torch.tensor(labels[cal], dtype=torch.float64)
    weight = torch.tensor(balanced(rows, cal), dtype=torch.float64)
    results = []
    for negative_cost in [1.0, 2.0, 4.0]:
        raw = torch.zeros(5, dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [raw, bias], max_iter=200, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            w = F.softplus(raw)
            loss = (
                F.binary_cross_entropy_with_logits(x @ w + bias, y, reduction="none")
                * weight
                * torch.where(y > 0, 1.0, negative_cost)
            ).sum() + 0.01 * w.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        weights = F.softplus(raw).detach().numpy()
        b = float(bias.detach())
        z = np.clip(logits, -10, 10) @ weights + b
        probability = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        field = [i for i in selection if rows[i]["source"] == "tom2024"]
        r = dict(
            negative_cost=negative_cost,
            weights=weights.tolist(),
            bias=b,
            input_clip=10.0,
            input_order=[
                "DINO adapted linear",
                "DINO original/adapted MLP",
                "copied DINO12 image Healthy",
                "genus CNN",
                a.head_family + " image Healthy",
            ],
            input_identity=identity,
            calibration_n=len(cal),
            selection_n=len(selection),
            selection_auc=float(
                roc_auc_score(labels[selection], probability[selection])
            ),
            selection_field_auc=float(roc_auc_score(labels[field], probability[field])),
            fresh_ad_used=False,
            note="Internal grouped development; base checkpoints used broader validation. Not independent acceptance.",
        )
        results.append(r)
        print(json.dumps(r), flush=True)
    (out / "candidates.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
