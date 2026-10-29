"""Fit monotone, source-balanced whole-scene logit fusion on calibration groups.

CNN and frozen-feature image-level Healthy predictors are combined without
modifying any disease head. Fusion fitting uses a named subset of development
validation groups, never the prospective reserved photographs.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from campaign_pixels import open_pixels
from phase7_feature_heads import BinaryHead
from phase7_species_train import balanced
from phase7_verifier import restore


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrated_head(directory, features):
    directory = Path(directory)
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    head = BinaryHead(state["kind"]).cuda().eval()
    head.load_state_dict(state["model"])
    result = json.loads((directory / "result.json").read_text())
    with torch.inference_mode():
        return (
            (
                head(torch.from_numpy(features).cuda()) * result["calibration"][0]
                + result["calibration"][1]
            )
            .cpu()
            .numpy()
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene-run", default="scene_healthy")
    ap.add_argument("--run-name", default="logit_fusion")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / a.run_name
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = root / "scene_inputs"
    rows = json.loads((work / "rows.json").read_text())
    features = np.load(root / "feature_heads/healthy_scene/features.npy")
    pixels = open_pixels(work)
    healthy_dirs = [
        root / "feature_heads/healthy_scene/linear_adapted",
        root / "feature_heads/healthy_scene/mlp_both",
    ]
    identity = dict(
        scene_rows_sha256=digest(work / "rows.json"),
        feature_model_sha256=digest(root / "export_initial/pytorch_bundle.pt"),
        head_hashes=[digest(d / "best.pt") for d in healthy_dirs],
        head_calibration_hashes=[digest(d / "result.json") for d in healthy_dirs],
        scene_cnn_sha256=digest(root / a.scene_run / "best.pt"),
        scene_cnn_calibration_sha256=digest(root / a.scene_run / "result.json"),
        species_cnn_sha256=digest(root / "species/best.pt"),
        species_cnn_calibration_sha256=digest(root / "species/result.json"),
    )
    cache = out / "calibrated_logits.npy"
    meta = out / "input_identity.json"
    if cache.exists():
        assert json.loads(meta.read_text()) == identity
        logits = np.load(cache)
    else:
        logits = np.empty((len(rows), 4), np.float32)
        for i, directory in enumerate(healthy_dirs):
            logits[:, i] = calibrated_head(directory, features)
        scene = restore(root / a.scene_run).cuda()
        species = restore(root / "species").cuda()
        with torch.inference_mode():
            for start in range(0, len(rows), 256):
                x = torch.from_numpy(
                    np.ascontiguousarray(pixels[start : start + 256])
                ).cuda()
                logits[start : start + len(x), 2] = (
                    scene.logits(x).ravel().cpu().numpy()
                )
                logits[start : start + len(x), 3] = (
                    species.logits(x).ravel().cpu().numpy()
                )
        np.save(cache, logits)
        meta.write_text(json.dumps(identity, indent=2))
    labels = np.array([int(r["health_y"]) for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    calibration = np.array(
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
    selection = np.array([i for i in val if i not in set(calibration)])
    assert not (
        {rows[i]["group"] for i in calibration} & {rows[i]["group"] for i in selection}
    )
    for r, y in zip(rows, labels):
        r["species_y"] = int(y)
    x = torch.tensor(np.clip(logits[calibration], -10, 10), dtype=torch.float64)
    y = torch.tensor(labels[calibration], dtype=torch.float64)
    weights = torch.tensor(balanced(rows, calibration), dtype=torch.float64)
    results = []
    for negative_cost in [1.0, 2.0, 4.0]:
        raw = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [raw, bias], max_iter=200, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            w = F.softplus(raw)
            loss = (
                F.binary_cross_entropy_with_logits(x @ w + bias, y, reduction="none")
                * weights
                * torch.where(y > 0, 1.0, negative_cost)
            ).sum() + 0.01 * w.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        w = F.softplus(raw).detach().numpy()
        b = float(bias.detach())
        z = np.clip(logits, -10, 10) @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        primary = [i for i in selection if rows[i]["source"] == "tom2024"]
        result = dict(
            negative_cost=negative_cost,
            weights=w.tolist(),
            bias=b,
            input_clip=10.0,
            scene_run=a.scene_run,
            input_order=[
                "healthy_linear_adapted",
                "healthy_mlp_both",
                "healthy_scene_cnn",
                "tomato_species_cnn",
            ],
            input_identity=identity,
            calibration_group_rule="sha256(calibration211:group) modulo 100 below 65, validation groups only",
            calibration_n=len(calibration),
            selection_n=len(selection),
            calibration_used_for_fusion_fit=True,
            selection_group_auc=float(roc_auc_score(labels[selection], p[selection])),
            selection_field_auc=float(roc_auc_score(labels[primary], p[primary])),
            note="Selection groups are internal development, not independent acceptance; base predictor checkpoint selection used broader validation.",
            reserved_used=False,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    (out / "candidates.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
