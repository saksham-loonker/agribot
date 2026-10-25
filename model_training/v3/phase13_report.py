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
    pending = []
    for loss in ["focal", "ce"]:
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
    policies, validations = {}, {}
    for name in [
        "selection",
        "fusion_selection",
        "fusion_refined_selection",
        "sft_focal_selection",
        "sft_ce_selection",
    ]:
        path = root / name / "candidate.json"
        if path.exists():
            policies[name] = read(path)
        path = root / (name + "_validation") / "result.json"
        if path.exists():
            validations[name] = read(path)
    legacy_internal = p / "phase12_scene/union_app_validation/result.json"
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
        cold_ad_diagnostic_predictions_opened=False,
        retired_246_and_related_families_are_benchmarks=False,
        native_context_runtime_implemented=False,
        new_model_installed=False,
        android_ci=dict(
            commit="c955fba9664753663ec016bebbaa8f210a128bf1",
            run=40219608964,
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
