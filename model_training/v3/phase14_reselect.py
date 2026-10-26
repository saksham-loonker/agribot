"""Compare saved best/final checkpoints at a scale-invariant low-FPR target.

This is checkpoint selection on already-used internal validation, not another
training run or independent evidence. Original artifacts are never overwritten.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch

from campaign_cnn_fusion import calibrate
from campaign_pixels import open_pixels
from permissive_encoder import digest
from permissive_scene import PermissiveScene, load
from phase7_species_train import balanced
from training_support import atomic_save, load_training_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def ranking_score(z, y, weights, field):
    recalls = []
    for mask in [field, np.ones(len(y), bool)]:
        negative = np.where(mask & (y == 0))[0]
        order = negative[np.argsort(-z[negative], kind="stable")]
        cumulative = np.cumsum(weights[order]) / weights[negative].sum()
        allowed = int(np.sum(cumulative <= 0.005))
        threshold = np.nextafter(z[order[allowed]], np.float32(np.inf))
        positive = mask & (y == 1)
        recalls.append(
            float(np.average(z[positive] >= threshold, weights=weights[positive]))
        )
    pauc = float(roc_auc_score(y, z, sample_weight=weights, max_fpr=0.01))
    return dict(
        score=recalls[0] + 0.25 * recalls[1] + 0.1 * pauc,
        field_recall_at_weighted_fpr_005=recalls[0],
        all_recall_at_weighted_fpr_005=recalls[1],
        partial_auc_001=pauc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    root = Path(a.campaign) / "phase13_siglip"
    assert Path(a.run).name == a.run
    source = root / a.run
    out = root / ("reselected_" + a.run)
    out.mkdir(exist_ok=True)
    rows = read(root / "cnn_inputs/rows.json")
    assert read(source / "restored_model_parity.json")["passed"]
    ids = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    y = np.array([int(rows[i]["health_y"]) for i in ids])
    for r in rows:
        r["species_y"] = int(r["health_y"])
    weights = balanced(rows, ids)
    field = np.array([rows[i]["source"] == "tom2024" for i in ids])
    protocol = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
        source_best_sha256=digest(source / "best.pt"),
        source_last_sha256=digest(source / "last.pt"),
        source_rows_sha256=digest(root / "cnn_inputs/rows.json"),
        comparison_set=["best", "last"],
        score="field recall at weighted FPR<=.005 + .25 all-source recall at same FPR + .1 standardized partial AUC at .01",
        new_training_run=False,
        internal_validation_only=True,
        restricted_teacher_used=False,
        cold_ad_used=False,
    )
    meta = out / "selection_protocol.json"
    if meta.exists():
        assert read(meta) == protocol
    else:
        meta.write_text(json.dumps(protocol, indent=2))
    if (out / "restored_model_parity.json").exists():
        assert read(out / "restored_model_parity.json")["passed"]
        print("Already complete", out)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    pixels = open_pixels(root / "cnn_inputs")

    def evaluate(model):
        values = []
        with torch.inference_mode():
            for start in range(0, len(ids), 64):
                x = torch.from_numpy(
                    np.array(pixels[ids[start : start + 64]], copy=True)
                ).cuda()
                values.append(model(x).cpu().numpy())
        return np.concatenate(values)

    candidates = []
    selected_state, selected_z, selected_score = None, None, -float("inf")
    for name in protocol["comparison_set"]:
        state = load_training_checkpoint(source / (name + ".pt"))
        assert (
            state["configuration"]["model_family"] == "siglip2"
            and state["configuration"]["restricted_teacher_used"] is False
        )
        assert state["configuration"]["rows_sha256"] == protocol["source_rows_sha256"]
        model = PermissiveScene(state["spec"]).cuda().eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        z = evaluate(model)
        assert np.isfinite(z).all()
        metric = ranking_score(z, y, weights, field)
        candidates.append(dict(checkpoint=name, epoch=state["epoch"], **metric))
        if metric["score"] > selected_score:
            selected_score, selected_z = metric["score"], z
            selected_state = {
                k: state[k]
                for k in ["architecture", "spec", "configuration", "model", "epoch"]
            }
        del model, state
        torch.cuda.empty_cache()
    cal = calibrate(selected_z, y, weights)
    atomic_save(selected_state, out / "best.pt")
    result = dict(
        configuration=selected_state["configuration"],
        calibration=cal,
        checkpoint_sha256=digest(out / "best.pt"),
        selection_protocol=protocol,
        compared_checkpoints=candidates,
        selected_epoch=selected_state["epoch"],
        selection_score=selected_score,
        validation_n=len(ids),
        field_auc=float(roc_auc_score(y[field], selected_z[field])),
        new_training_run=False,
        cold_ad_used=False,
        consumed_cohort_is_benchmark=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    np.savez(out / "validation_logits.npz", indices=ids, logits=selected_z)
    _, restored = load(out)
    restored = restored.cuda().eval()
    actual = evaluate(restored)
    error = float(np.max(np.abs(actual - (selected_z * cal[0] + cal[1]))))
    assert error < 0.002, "Reselected full model/calibration failed restoration parity"
    (out / "restored_model_parity.json").write_text(
        json.dumps(
            dict(passed=True, cases=len(ids), max_abs_calibrated_logit_error=error),
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(
                run=a.run,
                compared=candidates,
                selected_epoch=result["selected_epoch"],
                restore_error=error,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
