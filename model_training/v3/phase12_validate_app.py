"""Score baseline plus recovery on new internal family validation.

Standalone recovery-head recall is diagnostic, not the app's Healthy recall.
No retired cohort is presented as independent and no cold AD data is read.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import threading

import numpy as np

from app_eval import Models, image_record
from context_abstention import route_context
from phase2_select import retain
from phase5_scene_ablation import photo_metrics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -80, 80)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase12_scene"
    out = root / "union_app_validation"
    out.mkdir(exist_ok=True)
    candidate_path = root / "union_selection/candidate.json"
    c = read(candidate_path)
    assert c["retained"] and c["field_healthy_passed"]
    policy = c["policy"]
    verified = read(root / "union_verification/regression_parity.json")
    assert verified["passed"]
    rows = read(root / "cnn_inputs/rows.json")
    admitted = read(root / "admission_manifest.json")
    offset = len(rows) - len(admitted)
    ids = [i for i, r in enumerate(admitted) if r["split"] == "val"]
    records = [admitted[i] for i in ids]
    teacher_path = (
        root / "union_verification/admitted_training_development_teacher/outputs.npy"
    )
    teacher = np.load(teacher_path)
    th = sigmoid(teacher[:, 0])
    genus = sigmoid(teacher[:, 1])
    student_dir = root / ("sft_" + policy["student_loss"])
    v = np.load(student_dir / "validation_logits.npz")
    cal = read(student_dir / "result.json")["calibration"]
    sh = sigmoid(v["logits"] * cal[0] + cal[1])
    student_by_row = {int(index): float(sh[k]) for k, index in enumerate(v["indices"])}
    protocol = dict(
        candidate_sha256=digest(candidate_path),
        admission_sha256=digest(root / "admission_manifest.json"),
        teacher_scores_sha256=digest(teacher_path),
        student_validation_logits_sha256=digest(student_dir / "validation_logits.npz"),
        student_calibration_sha256=digest(student_dir / "result.json"),
        baseline_manifest_sha256=digest(p / "export_cnn_safe/model_manifest.json"),
        script_sha256=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
        new_validation_photos=len(records),
        scope="Actual original classifier/detector-byte replay plus recovery on newly admitted grouped internal validation. These families are NOT independent acceptance. Raw recovery-only recall is not an app gate.",
        cold_ad_used=False,
    )
    meta = out / "protocol.json"
    if meta.exists():
        assert read(meta) == protocol
    else:
        meta.write_text(json.dumps(protocol, indent=2))
    path = out / "baseline_predictions.jsonl"
    done = {}
    if path.exists():
        done = {
            r["id"]: r
            for r in [json.loads(line) for line in path.read_text().splitlines()]
        }
    local = threading.local()

    def evaluate(record):
        assert digest(record["path"]) == record["file_sha256"]
        if not hasattr(local, "model"):
            local.model = Models(p / "export_cnn_safe", threads=2)
        return image_record(local.model, record)

    remaining = [r for r in records if r["id"] not in done]
    with path.open("a") as f, concurrent.futures.ThreadPoolExecutor(2) as pool:
        for r in pool.map(evaluate, remaining):
            done[r["id"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if len(done) % 20 == 0:
                print(
                    json.dumps(dict(evaluated=len(done), total=len(records))),
                    flush=True,
                )
    before = []
    after = []
    for i, record in zip(ids, records):
        row = done[record["id"]]
        s = student_by_row[offset + i]
        t = float(th[i])
        teacher_positive = t >= policy["teacher_healthy_min"]
        student_positive = (
            s >= policy["student_healthy_min"]
            and genus[i] >= policy["student_genus_min"]
        )
        candidate = route_context(
            row,
            row,
            (
                local.model.manifest
                if hasattr(local, "model")
                else read(p / "export_cnn_safe/model_manifest.json")
            ),
            bool(teacher_positive or student_positive),
            False,
            t if teacher_positive else s,
            False,
            False,
        )
        before.append(row)
        after.append(candidate)
    sources = sorted({r["source"] for r in records})
    b = {s: photo_metrics([r for r in before if r["source"] == s]) for s in sources}
    observed = {
        s: photo_metrics([r for r in after if r["source"] == s]) for s in sources
    }
    healthy = photo_metrics(after)["healthy"]
    checks = dict(
        original_eight_regressions_passed=verified["retained"],
        original_field_healthy_targets_passed=c["field_healthy_passed"],
        new_family_app_healthy_min_70=healthy["correct"] / healthy["n"] >= 0.7,
        new_family_relative_retention=retain(observed, b),
    )
    result = dict(
        checks=checks,
        passed_preacceptance=all(checks.values()),
        baseline=b,
        candidate=observed,
        healthy=healthy,
        new_family_validation_photos=len(records),
        independent=False,
        cold_ad_used=False,
        retired_cohort_is_benchmark=False,
        standalone_recovery_only_check_is_not_app_recall=True,
        next_step="If passed, freeze an independent-source protocol for the exact candidate before opening AD. Prototype success still requires export and native/device parity.",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passed=result["passed_preacceptance"],
                checks=checks,
                Healthy=healthy["correct"],
                Healthy_n=healthy["n"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
