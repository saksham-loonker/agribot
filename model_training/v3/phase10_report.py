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
    for family in ["mobileclip", "mobileclip_dino"]:
        for result in sorted(
            (root / (family + "_feature_heads")).glob("*/result.json")
        ):
            evidence["trained_controls"][family + "_" + result.parent.name] = read(
                result
            )
    scene = root.parent / "phase11_scene"
    for result in sorted((scene / "mobileclip_feature_heads").glob("*/result.json")):
        evidence["trained_controls"]["image_mobileclip_" + result.parent.name] = read(
            result
        )
    if (root / "mobileclip_invalid_controls.json").exists():
        evidence["quarantined_invalid_controls"] = read(
            root / "mobileclip_invalid_controls.json"
        )
        evidence["invalid_fits_excluded_from_count"] = sum(
            r["invalid_fits"] for r in evidence["quarantined_invalid_controls"]
        )
    if (root / "mobileclip_source.json").exists():
        source = read(root / "mobileclip_source.json")
        evidence["mobileclip_pretrained_source"] = {
            k: source[k]
            for k in ["repository", "revision", "file_sha256", "model_name"]
        }
        evidence["mobileclip_pretrained_source"]["license"] = source["pretrained_cfg"][
            "license"
        ]
    names = ["leaf_selection", "leaf_selection_species", "leaf_selection_joint"]
    names += [
        "feature_selection_" + n
        for n in ["linear_bce", "linear_focal", "mlp_bce", "mlp_focal"]
    ]
    names += [
        "leaf_sft_selection",
        "leaf_sft_augmented_selection",
        "leaf_sft_full_roi_selection",
        "mobileclip_selection",
        "mobileclip_dino_selection",
        "mobileclip_aggregate_selection",
        "mobileclip_dino_aggregate_selection",
    ]
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
            input_policy=c.get(
                "input_policy", "center_with_optional_full_roi_confirmation"
            ),
            reference_disease_priority=c["reference_disease_priority"],
            retained=c["retained"],
            field_healthy_passed=c["field_healthy_passed"],
            field_leaf_oracle=c["field_leaf_oracle"],
            phase5_selection=c["metrics"]["phase5_selection"],
            input_identity=c["input_identity"],
        )
        parity = root / name / "component_parity.json"
        if parity.exists():
            evidence["policy_replays"][name]["component_parity"] = read(parity)
    if (scene / "input_protocol.json").exists():
        evidence["primary_image_training_inputs"] = read(scene / "input_protocol.json")
    if (scene / "selection/candidate.json").exists():
        c = read(scene / "selection/candidate.json")
        evidence["primary_image_policy"] = c
    if (scene / "context_restoration_parity.json").exists():
        evidence["primary_image_model_parity"] = read(
            scene / "context_restoration_parity.json"
        )
    if (scene / "logit_fusion/candidates.json").exists():
        for fit in read(scene / "logit_fusion/candidates.json"):
            evidence["trained_controls"][
                "image_logit_fusion_negative_cost_" + str(fit["negative_cost"])
            ] = fit
    if (scene / "fusion_selection/candidate.json").exists():
        evidence["primary_image_fused_policy"] = read(
            scene / "fusion_selection/candidate.json"
        )
    verification = scene / "prototype_verification"
    for name in [
        "protocol",
        "bundle_identity",
        "development_parity",
        "opened_regression_result",
    ]:
        if (verification / (name + ".json")).exists():
            evidence["context_prototype_" + name] = read(
                verification / (name + ".json")
            )
    if (root.parent / "phase12_scene/complete.json").exists():
        evidence["new_data_revision"] = read(
            root.parent / "phase12_scene/complete.json"
        )
        revision = root.parent / "phase12_scene"
        for result in sorted(
            (revision / "mobileclip_feature_heads").glob("*/result.json")
        ):
            evidence["trained_controls"]["phase12_feature_" + result.parent.name] = (
                read(result)
            )
        evidence["phase12_pending"] = []
        evidence["phase12_policies"] = {}
        for loss in ["ce", "focal", "distill"]:
            directory = revision / ("sft_" + loss)
            result, parity = (
                directory / "result.json",
                directory / "restored_model_parity.json",
            )
            if result.exists() and parity.exists() and read(parity)["passed"]:
                evidence["trained_controls"]["phase12_sft_" + loss] = dict(
                    result=read(result), restored_parity=read(parity)
                )
            else:
                evidence["phase12_pending"].append(
                    loss + " training/restore validation"
                )
        for name in [
            "selection",
            "sft_ce_selection",
            "sft_focal_selection",
            "sft_distill_selection",
        ]:
            path = revision / name / "candidate.json"
            if path.exists():
                evidence["phase12_policies"][name] = read(path)
            else:
                evidence["phase12_pending"].append(name + " app replay")
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
