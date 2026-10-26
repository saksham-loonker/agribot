"""Complementary licensed recovery heads; joint error counts remain gates."""

from pathlib import Path
import argparse
import itertools
import json

import numpy as np

from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import grouped


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase13_siglip"
    out = root / "risk_union_selection"
    out.mkdir(exist_ok=True)
    teacher_candidate = root / "fusion_refined_selection/candidate.json"
    teacher = read(teacher_candidate)
    verification = root / "fusion_refined_selection_validation"
    verified = read(verification / "result.json")
    assert verified["checks"]["development_graph_matches"]
    assert verified["restricted_teacher_used"] is False
    arrays = np.load(verification / "development_probabilities.npz")
    teacher_h, genus = arrays["healthy"][:, 0], arrays["genus"]
    leaf = p / "phase10_leaf"
    manifest = read(p / "original_reference/model_manifest.json")
    base = {
        k: v["rows"]
        for k, v in app_evaluate(
            safe_logits(leaf),
            read(leaf / "cnn_inputs/rows.json"),
            read(leaf / "eval_rows.json"),
            manifest,
            True,
        ).items()
    }
    baseline = grouped(base)
    paths = sorted({r["path"] for values in base.values() for r in values})
    lookup = {path: i for i, path in enumerate(paths)}
    students = {}
    for fit in read(root / "risk_heads/ranked.json"):
        mode = Path(fit["checkpoint"]).parent.name
        directory = root / ("risk_" + mode + "_selection")
        assert (
            read(directory / "input_identity.json")["paths_sha256"]
            == teacher["input_identity"]["paths_sha256"]
        )
        assert (
            fit["protocol"]["feature_identity"]["encoder_sha256"]
            == teacher["input_identity"]["encoder_sha256"]
        )
        students[mode] = (
            fit,
            np.load(directory / "probabilities.npz")["healthy"][:, 0],
        )
    ranks = []
    for mode, tmin, smin, gmin in itertools.product(
        students,
        [0.98, 0.99, 0.995],
        [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999],
        [0.0, 0.8],
    ):
        fit, student = students[mode]
        tpositive = teacher_h >= tmin
        positive = tpositive | ((student >= smin) & (genus >= gmin))
        confidence = np.where(tpositive, teacher_h, student)
        revised = {
            name: [
                route_context(
                    r,
                    r,
                    manifest,
                    bool(positive[lookup[r["path"]]]),
                    False,
                    float(confidence[lookup[r["path"]]]),
                    False,
                    False,
                )
                for r in values
            ]
            for name, values in base.items()
        }
        metrics = grouped(revised)
        field = metrics["phase5_selection"]
        passed = all(
            field["by_source"][s]["healthy_recall"] >= 0.7
            for s in ["tom2024", "swtt5wy2pr"]
        )
        score = float(
            np.mean(
                [
                    (v["healthy"]["correct_rate"] or 0)
                    + (v["supported_disease"]["correct_rate"] or 0)
                    - 3 * (v["all"]["false_healthy_rate"] or 0)
                    for v in metrics.values()
                ]
            )
        )
        directory = Path(fit["checkpoint"]).parent
        candidate = dict(
            id=len(ranks),
            fusion=teacher["fusion"],
            input_identity=teacher["input_identity"],
            policy=dict(
                healthy_min=tmin,
                genus_min=0.0,
                two_view_agreement=False,
                named_leaf_veto=False,
                union_student_healthy_min=smin,
                union_student_genus_min=gmin,
            ),
            union=dict(
                teacher_candidate_sha256=digest(teacher_candidate),
                teacher_bundle_sha256=verified["bundle_sha256"],
                student_directory=str(directory),
                student_checkpoint_sha256=digest(directory / "best.pt"),
                student_calibration_sha256=digest(directory / "result.json"),
                teacher_probabilities_sha256=digest(
                    verification / "development_probabilities.npz"
                ),
                student_probabilities_sha256=digest(
                    root / ("risk_" + mode + "_selection/probabilities.npz")
                ),
                student_mode=mode,
            ),
            metrics=metrics,
            retained=retain(metrics, baseline),
            field_healthy_passed=passed,
            score=score,
            restricted_teacher_used=False,
            fresh_ad_used=False,
            native_context_runtime_implemented=False,
        )
        ranks.append(candidate)
    ranks.sort(
        key=lambda c: (
            c["retained"] and c["field_healthy_passed"],
            c["retained"],
            c["score"],
        ),
        reverse=True,
    )
    (out / "candidate.json").write_text(json.dumps(ranks[0], indent=2))
    # Compact ranking avoids duplicating all cohort metrics for every policy.
    (out / "ranked.json").write_text(
        json.dumps(
            [{k: v for k, v in c.items() if k != "metrics"} for c in ranks], indent=2
        )
    )
    c = ranks[0]
    print(
        json.dumps(
            dict(
                id=c["id"],
                policy=c["policy"],
                mode=c["union"]["student_mode"],
                retained=c["retained"],
                field_passed=c["field_healthy_passed"],
                by_source=c["metrics"]["phase5_selection"]["by_source"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
