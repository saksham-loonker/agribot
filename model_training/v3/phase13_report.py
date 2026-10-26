"""Collect the licensed continuation without reading cold acceptance images."""

from pathlib import Path
import argparse
import json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase13_siglip"
    trained = {}
    for result in sorted((root / "siglip2_feature_heads").glob("*/result.json")):
        trained["siglip2_head_" + result.parent.name] = read(result)
    fits = root / "logit_fusion/candidates.json"
    if fits.exists():
        for fit in read(fits):
            trained["siglip2_fusion_negative_cost_" + str(fit["negative_cost"])] = fit
    admitted_fits = root / "logit_fusion_admitted/candidates.json"
    if admitted_fits.exists():
        for fit in read(admitted_fits):
            trained[
                "siglip2_admitted_fusion_negative_cost_" + str(fit["negative_cost"])
            ] = fit
    for result in sorted((root / "risk_heads").glob("*/result.json")):
        parity = result.parent / "restored_head_parity.json"
        if parity.exists() and read(parity)["passed"]:
            trained["siglip2_risk_" + result.parent.name] = dict(
                result=read(result), restored_parity=read(parity)
            )
    for family, directory in [
        ("dinov2_large", p / "phase15_dino_large"),
        ("hybrid_field", p / "phase16_hybrid"),
    ]:
        for path in sorted(
            (directory / (family + "_feature_heads")).glob("*/result.json")
        ):
            trained[family + "_" + path.parent.name] = read(path)
    pending = []
    multisource_validations = {}
    for directory, family in [
        ("phase23_siglip", "siglip2"),
        ("phase23_dino", "dinov2_large"),
        ("phase23_hybrid", "hybrid_field"),
    ]:
        fresh = p / directory
        for path in sorted((fresh / (family + "_feature_heads")).glob("*/result.json")):
            trained[directory + "_" + path.parent.name] = read(path)
        for path in sorted((fresh / "risk_heads").glob("*/result.json")):
            parity = path.parent / "restored_head_parity.json"
            if parity.exists() and read(parity)["passed"]:
                trained[directory + "_risk_" + path.parent.name] = read(path)
        fits = fresh / "logit_fusion_admitted/candidates.json"
        if fits.exists():
            for fit in read(fits):
                trained[directory + "_fusion_" + str(fit["negative_cost"])] = fit
        for path in sorted(fresh.glob("*_validation/result.json")):
            value = read(path)
            value.pop("old_development", None)
            multisource_validations[directory + "/" + path.parent.name] = value
    for loss in ["focal", "ce", "pool_focal", "tail2_focal"]:
        directory = root / ("sft_" + loss)
        result, parity = (
            directory / "result.json",
            directory / "restored_model_parity.json",
        )
        if result.exists() and parity.exists() and read(parity)["passed"]:
            trained["siglip2_sft_" + loss] = dict(
                result=read(result), restored_parity=read(parity)
            )
        else:
            pending.append(loss + " supervised training/restore validation")
    new_sft = p / "phase23_siglip/sft_ad_tail2_focal_lowfpr"
    if (new_sft / "restored_model_parity.json").exists() and read(
        new_sft / "restored_model_parity.json"
    )["passed"]:
        trained["consumed_ad_sft_tail2_focal_lowfpr"] = dict(
            result=read(new_sft / "result.json"),
            restored_parity=read(new_sft / "restored_model_parity.json"),
        )
    elif new_sft.exists():
        pending.append(
            "Consumed-AD small-step supervised SigLIP2 tail adaptation/restore"
        )
    screening_path = p / "phase27_screen_diagnostics/result.json"
    screening = read(screening_path) if screening_path.exists() else None
    if (
        new_sft.exists()
        and not screening
        and not (
            p
            / "phase23_siglip/sft_ad_tail2_focal_lowfpr_selection_validation/result.json"
        ).exists()
    ):
        pending.append("Consumed-AD supervised model fresh app replay")
    if not (p / "phase24_olid_source/membership_audit.json").exists():
        pending.append(
            "New OLID source overlap/label/family audit; no diagnostic predictions"
        )
    policies, validations = {}, {}
    reselections = {}
    for path in sorted(root.glob("reselected_sft_*/result.json")):
        result = read(path)
        if "selection_protocol" in result:
            reselections[path.parent.name] = result
    selection_names = [
        "selection",
        "fusion_selection",
        "fusion_refined_selection",
        "sft_focal_selection",
        "sft_ce_selection",
        "admitted_fusion_selection",
        "sft_pool_focal_selection",
        "sft_tail2_focal_selection",
        "risk_union_selection",
        "low_fpr_head_selection",
        "sft_pool_low_fpr_selection",
    ] + [
        "risk_" + mode + "_selection"
        for mode in [
            "cvar_bce",
            "partial_auc",
            "reinforce_label_reward",
            "expected_logit_reward",
        ]
    ]
    selection_names += [name + "_selection" for name in reselections]
    for name in selection_names:
        path = root / name / "candidate.json"
        if path.exists():
            policies[name] = read(path)
        path = root / (name + "_validation") / "result.json"
        if path.exists():
            value = read(path)
            # Fresh graph metrics were required to equal the selected metrics;
            # retain one copy and point to its candidate identity.
            value.pop("old_development", None)
            validations[name] = value
    legacy_internal = p / "phase12_scene/union_app_validation/result.json"
    if not (root / "sft_pool_low_fpr/restored_model_parity.json").exists():
        pending.append("Corrected low-FPR pool rerun training/restore")
    elif not (root / "sft_pool_low_fpr_selection_validation/result.json").exists():
        pending.append("Corrected low-FPR pool full app validation")
    evidence = dict(
        schema=1,
        status="Incomplete recognition repair; no replacement model qualified or installed",
        model_source=read(root / "encoder_source.json"),
        source_protocol=read(root / "input_protocol.json"),
        canonical_input_parity=read(
            root / "siglip2_feature_heads/input_contract_parity.json"
        ),
        restored_encoder_parity=read(
            root / "siglip2_feature_heads/restored_encoder_parity.json"
        ),
        restricted_model_decision=dict(
            product_use_blocked=True,
            blocked_completed_controls=22,
            license="apple-amlr",
            source="https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS",
            affected="MobileCLIP leaf/image heads, fusions, all phase12 MobileCLIP SFT/union/distillation",
            restricted_weights_logits_or_teachers_used_in_siglip2=False,
            human_dataset_reuse_allowed=True,
        ),
        phase12_apple_research_internal_app_validation=(
            read(legacy_internal) if legacy_internal.exists() else None
        ),
        previous_valid_completed_research_controls=292,
        trained_controls=trained,
        completed_trained_research_controls=292 + len(trained),
        quarantined_invalid_controls_excluded=8,
        policy_replays=policies,
        exact_graph_internal_validations=validations,
        pending=pending,
        checkpoint_reselection_not_new_training=reselections,
        low_fpr_head_reruns_not_new_independent_controls={
            path.parent.name: read(path)
            for path in sorted((root / "low_fpr_heads").glob("*/result.json"))
        },
        low_fpr_pool_rerun_not_new_independent_control=(
            read(root / "sft_pool_low_fpr/result.json")
            if (root / "sft_pool_low_fpr/result.json").exists()
            else None
        ),
        canonical_source_file_pixel_audit=(
            read(root / "canonical_pixel_cache_audit_verified.json")
            if (root / "canonical_pixel_cache_audit_verified.json").exists()
            else None
        ),
        cold_ad_diagnostic_predictions_opened=(
            p / "phase22_external_acceptance/opened_identity.json"
        ).exists(),
        frozen_external_opening=(
            read(p / "phase22_external_acceptance/result.json")
            if (p / "phase22_external_acceptance/result.json").exists()
            else None
        ),
        ad_training_admission=(
            read(p / "phase23_data/complete.json")
            if (p / "phase23_data/complete.json").exists()
            else None
        ),
        multisource_internal_validations=multisource_validations,
        multisource_bounded_context=(
            read(p / "phase25_verified/result.json")
            if (p / "phase25_verified/result.json").exists()
            else None
        ),
        new_cold_olid_source=dict(
            download=(
                read(p / "phase24_olid_source/download_audit.json")
                if (p / "phase24_olid_source/download_audit.json").exists()
                else None
            ),
            membership=(
                read(p / "phase24_olid_source/membership_audit.json")
                if (p / "phase24_olid_source/membership_audit.json").exists()
                else None
            ),
            diagnostic_predictions_opened=False,
            training_used=False,
        ),
        additional_scene_cache_reclamation=(
            read(p / "scene_cache_reclamation_20261026.json")
            if (p / "scene_cache_reclamation_20261026.json").exists()
            else None
        ),
        cache_first_app_screening=screening,
        stopped_failed_checkpoint_after_cheap_checks=bool(
            screening and screening["passing_cheap_checks"] == 0
        ),
        incremental_frozen_feature_reuse=(
            read(p / "phase27_feature_reuse_audit_v2.json")
            if (p / "phase27_feature_reuse_audit_v2.json").exists()
            else None
        ),
        retired_246_and_related_families_are_benchmarks=False,
        native_context_runtime_implemented=False,
        new_model_installed=False,
        additional_representations={
            directory: dict(
                source=(
                    read(p / directory / "encoder_source.json")
                    if (p / directory / "encoder_source.json").exists()
                    else read(p / directory / "source_links.json")
                ),
                validation=read(p / directory / "selection_validation/result.json"),
            )
            for directory in ["phase15_dino_large", "phase16_hybrid"]
            if (p / directory / "selection_validation/result.json").exists()
        },
        whole_reference_refresh=(
            read(p / "phase20_whole_reference/result.json")
            if (p / "phase20_whole_reference/result.json").exists()
            else None
        ),
        bounded_context_recovery=(
            read(p / "phase21_verified/result.json")
            if (p / "phase21_verified/result.json").exists()
            else None
        ),
        coarse_priority_context_recovery=(
            read(p / "phase22_verified/result.json")
            if (p / "phase22_verified/result.json").exists()
            else None
        ),
        derived_cache_reclamation=(
            read(p / "cache_reclamation_20261026.json")
            if (p / "cache_reclamation_20261026.json").exists()
            else None
        ),
        android_ci=dict(
            commit="07146ba8e194e02ffbaa927384373f0b7ece130f",
            run=40272535806,
            conclusion="success",
        ),
    )
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2))
    print(
        json.dumps(
            dict(
                completed=evidence["completed_trained_research_controls"],
                pending=pending,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
