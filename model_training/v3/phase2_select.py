"""Explicit validation-only noninferiority analysis against installed 2.0.2.

The original zero-change gate is still recorded. This separate analysis permits
at most 1 percentage point disease recall loss per set AND per named condition,
and at most 0.5 points additional false Healthy/unknown/unsupported known calls.
Counts are rounded down; small sets/classes receive no exception. External
SLIF photos must remain unopened until this policy and checkpoint are frozen.
"""

from pathlib import Path
import argparse, itertools, json, math, time
import numpy as np
import torch
from campaign_cnn_fusion import calibrate
from campaign_cnn_train import sampling
from campaign_heads import app_evaluate
from campaign_verify import regression_retained
from phase2_policy import DomainPolicy
from app_eval import summarize
from train import LABELS
from campaign_reference import safe_logits


def retain(candidate, baseline):
    for key, m in candidate.items():
        b = baseline[key]
        if m["healthy"]["correct"] < b["healthy"]["correct"]:
            return False
        if m["supported_disease"]["correct"] < b["supported_disease"][
            "correct"
        ] - math.floor(0.01 * b["supported_disease"]["n"]):
            return False
        if m["supported_disease"]["false_healthy"] > b["supported_disease"][
            "false_healthy"
        ] + math.floor(0.005 * b["supported_disease"]["n"]):
            return False
        if m["all"]["false_healthy"] > b["all"]["false_healthy"] + math.floor(
            0.005 * b["supported_disease"]["n"]
        ):
            return False
        if m["supported"]["false_other"] > b["supported"]["false_other"] + math.floor(
            0.005 * b["supported"]["n"]
        ):
            return False
        for scope in ["unsupported_tomato", "non_tomato"]:
            if (m[scope]["false_known"] or 0) > (
                b[scope]["false_known"] or 0
            ) + math.floor(0.005 * b[scope]["n"]):
                return False
        for label, counts in m["disease_classes"].items():
            if counts["correct"] < b["disease_classes"][label]["correct"] - math.floor(
                0.01 * counts["n"]
            ):
                return False
    return True


def evaluate(z, rows, ev, manifest):
    photo = app_evaluate(z, rows, ev, manifest, True)
    selection = photo.pop("selection_app")["rows"]
    groups = {k: v["rows"] for k, v in photo.items()}
    groups["original"] = [
        r for r in selection if r["source"] not in ("tlid", "pakistan")
    ]
    groups["expanded"] = [r for r in selection if r["source"] in ("tlid", "pakistan")]
    result = {}
    for name, data in groups.items():
        m = summarize(data)
        classes = {}
        for label in LABELS:
            if label in ("Healthy", "Other"):
                continue
            subset = [
                r
                for r in data
                if r["label"] == label
                and r.get("analysis_eligible", True)
                and "error" not in r
            ]
            classes[label] = dict(
                n=len(subset),
                correct=sum(
                    r["app_replay"]["kind"] == "DISEASE"
                    and r["app_replay"]["label"] == label
                    for r in subset
                ),
            )
        m["disease_classes"] = classes
        result[name] = m
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--class-conditional", action="store_true")
    p.add_argument("--student-modes", default="lora_d4")
    p.add_argument("--out")
    a = p.parse_args()
    q = Path(a.root)
    out = (
        Path(a.out)
        if a.out
        else q / ("selection_v4" if a.class_conditional else "selection_v3")
    )
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    rows = json.loads((q / "cnn_inputs/rows.json").read_text())
    ev = json.loads((q / "eval_rows.json").read_text())
    manifest = json.loads(
        (q.parent / "original_reference/model_manifest.json").read_text()
    )
    reference = torch.from_numpy(safe_logits(q)).cuda()
    baseline = evaluate(reference.cpu().numpy(), rows, ev, manifest)
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    protocol = dict(
        baseline="installed 2.0.2 d7013580",
        disease_recall_loss_max=0.01,
        per_class_recall_loss_max=0.01,
        false_healthy_increase_max=0.005,
        false_other_increase_max=0.005,
        unsupported_false_known_increase_max=0.005,
        healthy_correct_must_not_decrease=True,
        counts_rounding="floor; zero exceptions for small cohorts",
        independent_cohort="Not asserted by selection. Freeze a separate unopened acceptance cohort before evaluating it.",
        source_history="Input cohorts are opened development/regressions; selection never evaluates a reserved manifest.",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rank = []
    start = time.time()
    for mode in a.student_modes.split(","):
        student_path = q / "multiclass" / mode / "best.pt"
        student = torch.from_numpy(
            np.load(student_path.parent / "best_logits.npy")
        ).cuda()
        raw = np.load(
            q
            / "cnn_inputs/runs/mobilenetv3_small_100.lamb_in1k_center_focal_n2/all_logits.npy"
        )
        cnn = torch.from_numpy(raw).cuda()
        cal = []
        for c, key in enumerate(["health_y", "known_y"]):
            ids = np.array(
                [i for i, r in enumerate(rows) if r["split"] == "val" and r[key] >= 0]
            )
            cal.append(
                calibrate(
                    raw[ids, c],
                    np.array([rows[i][key] for i in ids]),
                    sampling(rows, ids, target=key),
                )
            )
        with torch.inference_mode():
            grid = (
                itertools.product(
                    [0.8, 0.9],
                    [0.8, 0.9, 0.95],
                    [0.5, 0.7],
                    [0.0],
                    [[1, 8], [0, 1, 8], [1, 3, 8], [0, 1, 3, 8]],
                )
                if a.class_conditional
                else (
                    (*v, None)
                    for v in itertools.product(
                        [0.8, 0.9, 0.97], [0.8, 0.9, 0.95], [0.5, 0.7, 0.85], [0.0, 0.7]
                    )
                )
            )
            for smin, hmin, kmin, minimum, allowed in grid:
                spec = dict(
                    alpha=1.0,
                    student_temperature=0.8,
                    student_min=smin,
                    healthy_min=hmin,
                    known_min=kmin,
                    reference_disease_min=minimum,
                    reference_disease_max=1.0,
                    cnn_calibration=cal,
                    allow_unknown=False,
                    healthy_reference_classes=allowed,
                )
                z = DomainPolicy(**spec).cuda()(reference, student, cnn).cpu().numpy()
                metrics = evaluate(z, rows, ev, manifest)
                score = float(
                    np.mean(
                        [
                            (v["healthy"]["correct_rate"] or 0)
                            + (v["supported_disease"]["correct_rate"] or 0)
                            - 3 * (v["supported_disease"]["false_healthy_rate"] or 0)
                            - 2 * (v["unsupported_tomato"]["false_known_rate"] or 0)
                            for v in metrics.values()
                        ]
                    )
                )
                r = dict(
                    id=len(rank),
                    student_checkpoint=str(student_path),
                    cnn_checkpoint=str(
                        q
                        / "cnn_inputs/runs/mobilenetv3_small_100.lamb_in1k_center_focal_n2/best.pt"
                    ),
                    policy=spec,
                    metrics=metrics,
                    retained=retain(metrics, baseline),
                    strict_retained=regression_retained(metrics, baseline),
                    score=score,
                )
                rank.append(r)
                print(
                    json.dumps(
                        dict(
                            id=r["id"],
                            passed=r["retained"],
                            strict=r["strict_retained"],
                            score=score,
                            novel=[
                                metrics["phase2_selection"]["healthy"]["correct"],
                                metrics["phase2_selection"]["supported_disease"][
                                    "correct"
                                ],
                            ],
                            seconds=time.time() - start,
                        )
                    ),
                    flush=True,
                )
    rank.sort(key=lambda r: (r["retained"], r["score"]), reverse=True)
    (out / "ranked.json").write_text(json.dumps(rank, indent=2))
    (out / "candidate.json").write_text(json.dumps(rank[0], indent=2))
    print(
        json.dumps(
            dict(
                best_id=rank[0]["id"],
                passed=rank[0]["retained"],
                passing=sum(r["retained"] for r in rank),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
