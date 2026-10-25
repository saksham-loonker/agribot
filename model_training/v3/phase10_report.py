"""Write compact, inspectable leaf-scope evidence without opening fresh AD data."""

from pathlib import Path
import argparse
import hashlib
import json

from app_eval import posterior
from campaign_reference import safe_logits
from view_guard_policy import decision_key


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    field = root.parent / "phase10_fieldplant"
    evidence = dict(
        schema=1,
        status="Research only; do not deploy a candidate without all retention and fresh acceptance gates",
        source_audit_before_fitting=read(field / "membership_audit.json"),
        preparation=read(root / "complete.json"),
        fieldplant_used_for_binary_training=True,
        cold_ad_diagnostic_predictions_opened=False,
        feature_reference_parity=read(
            root / "leaf_feature_heads/reference_parity.json"
        ),
        trained_controls={},
        policy_replays={},
    )
    for name in [
        "leaf_healthy",
        "leaf_species",
        "leaf_sft_d12_focal",
        "leaf_sft_d12_augmented",
    ]:
        result = root / name / "result.json"
        if result.exists():
            evidence["trained_controls"][name] = read(result)
    for result in sorted((root / "leaf_feature_heads").glob("*/result.json")):
        evidence["trained_controls"]["feature_" + result.parent.name] = read(result)
    names = ["leaf_selection", "leaf_selection_species", "leaf_selection_joint"]
    names += [
        "feature_selection_" + n
        for n in ["linear_bce", "linear_focal", "mlp_bce", "mlp_focal"]
    ]
    names += ["leaf_sft_selection", "leaf_sft_augmented_selection"]
    for name in names:
        path = root / name / "candidate.json"
        if not path.exists():
            continue
        c = read(path)
        evidence["policy_replays"][name] = dict(
            candidate_sha256=digest(path),
            id=c["id"],
            policy=c["policy"],
            full_roi_confirmation=c["full_roi_confirmation"],
            reference_disease_priority=c["reference_disease_priority"],
            retained=c["retained"],
            field_healthy_passed=c["field_healthy_passed"],
            field_leaf_oracle=c["field_leaf_oracle"],
            phase5_selection=c["metrics"]["phase5_selection"],
            input_identity=c["input_identity"],
        )
    rows = read(root / "cnn_inputs/rows.json")
    reference = safe_logits(root)
    manifest = read(root.parent / "original_reference/model_manifest.json")
    baseline = dict(healthy_n=0, healthy_correct=0, negative_n=0, false_healthy=0)
    for i, r in enumerate(rows):
        if not (
            r["source"] == "fieldplant"
            and r["split"] == "val"
            and r["view"] == "fieldplant_center"
        ):
            continue
        healthy = decision_key(posterior([reference[i]], manifest))[0] == "HEALTHY"
        if r["health_y"] == 1:
            baseline["healthy_n"] += 1
            baseline["healthy_correct"] += healthy
        else:
            baseline["negative_n"] += 1
            baseline["false_healthy"] += healthy
    evidence["baseline_distinct_field_leaf_oracle"] = baseline
    if (root / "reference_ceiling_audit.json").exists():
        evidence["reference_disease_priority_ceiling_audit"] = read(
            root / "reference_ceiling_audit.json"
        )
    evidence["completed_trained_controls"] = 262 + len(evidence["trained_controls"])
    evidence["pending"] = [
        n
        for n in [
            "leaf_sft_d12_focal",
            "leaf_sft_selection",
            "leaf_sft_d12_augmented",
            "leaf_sft_augmented_selection",
        ]
        if not (
            root / n / ("candidate.json" if n.endswith("selection") else "result.json")
        ).exists()
    ]
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2))
    print(
        json.dumps(
            dict(
                out=str(path),
                completed_trained_controls=evidence["completed_trained_controls"],
                pending=evidence["pending"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
