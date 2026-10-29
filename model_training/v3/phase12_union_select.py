"""Check complementary retained image experts on opened regression only.

Positive recovery is a union of separately calibrated thresholds; its error
rate is measured jointly. The retired 246-photo cohort is never a benchmark.
"""

from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np

from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from phase2_select import retain
from phase5_scene_ablation import grouped


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase12_scene"
    out = root / "union_selection"
    out.mkdir(exist_ok=True)
    assert (root / "RETIRED_BENCHMARK.json").exists()
    leaf = p / "phase10_leaf"
    rows = read(leaf / "cnn_inputs/rows.json")
    ev = read(leaf / "eval_rows.json")
    manifest = read(p / "original_reference/model_manifest.json")
    base = {
        k: v["rows"]
        for k, v in app_evaluate(safe_logits(leaf), rows, ev, manifest, True).items()
    }
    baseline = grouped(base)
    paths = sorted({r["path"] for values in base.values() for r in values})
    lookup = {path: i for i, path in enumerate(paths)}
    paths_sha = hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    teacher = p / "phase11_scene/prototype_verification"
    assert read(teacher / "development_parity.json")["passed"]
    teacher_pair = np.load(teacher / "development/outputs.npy")
    teacher_h = 1 / (
        1 + np.exp(-np.clip(teacher_pair[:, 0].astype(np.float64), -80, 80))
    )
    genus = 1 / (1 + np.exp(-np.clip(teacher_pair[:, 1].astype(np.float64), -80, 80)))
    assert (
        read(teacher / "development/paths_identity.json")["paths_sha256"] == paths_sha
    )
    identity = dict(
        teacher_bundle=read(teacher / "bundle_identity.json"),
        teacher_outputs_sha256=digest(teacher / "development/outputs.npy"),
        paths_sha256=paths_sha,
        rows_sha256=digest(leaf / "cnn_inputs/rows.json"),
        new_revision_sha256=digest(root / "complete.json"),
        students={},
    )
    students = {}
    for loss in ["ce", "focal", "distill"]:
        source = root / ("sft_" + loss + "_selection")
        assert read(source / "input_identity.json")["paths_sha256"] == paths_sha
        assert read(root / ("sft_" + loss) / "restored_model_parity.json")["passed"]
        students[loss] = np.load(source / "probabilities.npz")["healthy"][:, 0]
        identity["students"][loss] = dict(
            checkpoint_sha256=digest(root / ("sft_" + loss) / "best.pt"),
            calibration_sha256=digest(root / ("sft_" + loss) / "result.json"),
            probabilities_sha256=digest(source / "probabilities.npz"),
        )
    meta = out / "input_identity.json"
    if meta.exists():
        assert read(meta) == identity
    else:
        meta.write_text(json.dumps(identity, indent=2))
    ranks = []
    for loss, tmin, smin, gmin in itertools.product(
        ["ce", "focal", "distill"],
        [0.97, 0.98, 0.99],
        [0.99, 0.995, 0.999, 0.9999],
        [0.0, 0.8, 0.9],
    ):
        positive_teacher = teacher_h >= tmin
        positive_student = (students[loss] >= smin) & (genus >= gmin)
        positive = positive_teacher | positive_student
        probability = np.where(positive_teacher, teacher_h, students[loss])
        revised = {}
        for name, values in base.items():
            revised[name] = []
            for r in values:
                i = lookup[r["path"]]
                new = route_context(
                    r,
                    r,
                    manifest,
                    bool(positive[i]),
                    False,
                    float(probability[i]),
                    False,
                    False,
                )
                if new.get("whole_frame_evidence") is not None:
                    new["context_recovery_source"] = (
                        "teacher" if positive_teacher[i] else "student_" + loss
                    )
                revised[name].append(new)
        metrics = grouped(revised)
        m = metrics["phase5_selection"]
        field = all(
            m["by_source"][s]["healthy_recall"] >= 0.7
            for s in ["tom2024", "swtt5wy2pr"]
        )
        score = float(
            np.mean(
                [
                    (v["healthy"]["correct_rate"] or 0)
                    + (v["supported_disease"]["correct_rate"] or 0)
                    - 3 * (v["all"]["false_healthy_rate"] or 0)
                    - 2 * (v["unsupported_tomato"]["false_known_rate"] or 0)
                    for v in metrics.values()
                ]
            )
        )
        c = dict(
            id=len(ranks),
            policy=dict(
                teacher_healthy_min=tmin,
                student_healthy_min=smin,
                student_genus_min=gmin,
                student_loss=loss,
                rule="Teacher-first positive union; preserve every original aggregate named disease verdict",
            ),
            retained=retain(metrics, baseline),
            field_healthy_passed=field,
            metrics=metrics,
            score=score,
            input_identity=identity,
            fresh_ad_used=False,
            retired_cohort_scored=False,
            native_context_runtime_implemented=False,
        )
        ranks.append(c)
        print(
            json.dumps(
                dict(
                    id=c["id"],
                    loss=loss,
                    retained=c["retained"],
                    field_passed=field,
                    Healthy=m["healthy"]["correct"],
                    false_Healthy=m["all"]["false_healthy"],
                )
            ),
            flush=True,
        )
    ranks.sort(
        key=lambda c: (
            c["retained"] and c["field_healthy_passed"],
            c["retained"],
            c["score"],
        ),
        reverse=True,
    )
    for name, value in [
        ("ranked", ranks),
        ("candidate", ranks[0]),
        ("baseline", baseline),
    ]:
        (out / (name + ".json")).write_text(json.dumps(value, indent=2))
    print(
        json.dumps(
            dict(
                best=ranks[0]["id"],
                retained=ranks[0]["retained"],
                field_passed=ranks[0]["field_healthy_passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
