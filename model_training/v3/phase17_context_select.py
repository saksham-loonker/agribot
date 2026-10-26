"""Bidirectional image evidence on development only; no relaxed error budgets."""

from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math

import numpy as np

from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_guard import route_guard
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import grouped


def read(path):
    return json.loads(Path(path).read_text())


def coarse_counts(photos):
    result = {}
    for cohort, values in photos.items():
        for row in values:
            if (
                row["label"] != "Diseased_unspecified"
                or not row.get("analysis_eligible", True)
                or "error" in row
            ):
                continue
            key = "/".join(
                [cohort, row["source"], row.get("raw_label") or "unspecified"]
            )
            n, false = result.get(key, (0, 0))
            result[key] = (n + 1, false + int(row["app_replay"]["kind"] == "HEALTHY"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument(
        "--negative-source",
        choices=[
            "siglip2",
            "cvar_bce",
            "partial_auc",
            "reinforce_label_reward",
            "expected_logit_reward",
            "dinov2_large",
            "hybrid_field",
        ],
        default="siglip2",
    )
    ap.add_argument("--out-name", default="phase17_context_guard")
    ap.add_argument("--protect-suspected-disease", action="store_true")
    ap.add_argument("--preserve-all-uncertainty", action="store_true")
    ap.add_argument("--original-gates-only", action="store_true")
    ap.add_argument(
        "--positive-source",
        choices=[
            "fusion",
            "siglip2",
            "low_fpr_heads",
            "pool_low_fpr",
            "dinov2_large",
            "hybrid_field",
        ],
        default="fusion",
    )
    ap.add_argument(
        "--selection-objective", choices=["legacy", "field_recall"], default="legacy"
    )
    ap.add_argument("--closedset-guard", action="store_true")
    ap.add_argument(
        "--healthy-thresholds", default=".5,.6,.7,.8,.9,.93,.95,.96,.97,.98,.985,.99"
    )
    ap.add_argument(
        "--negative-thresholds", default="0,.00001,.0001,.001,.005,.01,.02,.05,.1,.2"
    )
    ap.add_argument("--genus-thresholds", default="0,.8")
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase13_siglip"
    assert Path(a.out_name).name == a.out_name and a.out_name not in (".", "..")
    out = p / a.out_name
    out.mkdir(exist_ok=True)
    teacher_path = root / "fusion_refined_selection/candidate.json"
    teacher = read(teacher_path)
    verified = read(root / "fusion_refined_selection_validation/result.json")
    assert (
        verified["checks"]["development_graph_matches"]
        and verified["restricted_teacher_used"] is False
    )
    teacher_arrays = np.load(
        root / "fusion_refined_selection_validation/development_probabilities.npz"
    )
    healthy, genus = teacher_arrays["healthy"][:, 0], teacher_arrays["genus"]
    positive_meta = teacher["input_identity"]
    positive_directory = None
    positive_path = (
        root / "fusion_refined_selection_validation/development_probabilities.npz"
    )
    if a.positive_source != "fusion":
        if a.positive_source in ("dinov2_large", "hybrid_field"):
            extra = p / (
                "phase15_dino_large"
                if a.positive_source == "dinov2_large"
                else "phase16_hybrid"
            )
            positive_selection = extra / "selection"
            positive_directory = Path(
                read(extra / (a.positive_source + "_feature_heads/ranked.json"))[0][
                    "checkpoint"
                ]
            ).parent
        else:
            positive_selection = (
                root
                / {
                    "siglip2": "selection",
                    "low_fpr_heads": "low_fpr_head_selection",
                    "pool_low_fpr": "sft_pool_low_fpr_selection",
                }[a.positive_source]
            )
            if a.positive_source == "pool_low_fpr":
                positive_directory = root / "sft_pool_low_fpr"
            else:
                positive_directory = Path(
                    read(
                        root
                        / (
                            a.positive_source.replace(
                                "siglip2", "siglip2_feature_heads"
                            )
                            + "/ranked.json"
                        )
                    )[0]["checkpoint"]
                ).parent
        positive_path = positive_selection / "probabilities.npz"
        positive_meta = read(positive_selection / "input_identity.json")
        assert (
            positive_meta["paths_sha256"] == teacher["input_identity"]["paths_sha256"]
        )
        assert digest(positive_directory / "best.pt") == positive_meta["head_sha256"]
        assert (
            digest(positive_directory / "result.json")
            == positive_meta["calibration_sha256"]
        )
        positive_arrays = np.load(positive_path)
        healthy, genus = positive_arrays["healthy"][:, 0], positive_arrays["genus"]
    if a.negative_source == "siglip2":
        negative_selection = root / "selection"
        negative_directory = Path(
            read(root / "siglip2_feature_heads/ranked.json")[0]["checkpoint"]
        ).parent
    elif a.negative_source in ("dinov2_large", "hybrid_field"):
        extra = p / (
            "phase15_dino_large"
            if a.negative_source == "dinov2_large"
            else "phase16_hybrid"
        )
        negative_selection = extra / "selection"
        negative_directory = Path(
            read(extra / (a.negative_source + "_feature_heads/ranked.json"))[0][
                "checkpoint"
            ]
        ).parent
    else:
        negative_selection = root / ("risk_" + a.negative_source + "_selection")
        negative_directory = root / "risk_heads" / a.negative_source
    negative_meta = read(negative_selection / "input_identity.json")
    assert negative_meta["paths_sha256"] == teacher["input_identity"]["paths_sha256"]
    assert digest(negative_directory / "best.pt") == negative_meta["head_sha256"]
    assert (
        digest(negative_directory / "result.json")
        == negative_meta["calibration_sha256"]
    )
    negative_path = negative_selection / "probabilities.npz"
    negative = np.load(negative_path)["healthy"][:, 0]
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
    whole_path = p / "phase20_whole_reference/predictions.jsonl"
    if a.closedset_guard:
        whole = {
            r["path"]: r
            for r in (json.loads(line) for line in whole_path.read_text().splitlines())
        }
        for values in base.values():
            for row in values:
                if row["app_replay"].get("probabilities") is None:
                    row["context_whole_reference_probabilities"] = whole[row["path"]][
                        "whole_probabilities"
                    ]
    baseline = grouped(base)
    original_coarse = coarse_counts(base)
    paths = sorted({r["path"] for values in base.values() for r in values})
    assert (
        hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        == teacher["input_identity"]["paths_sha256"]
    )
    lookup = {path: i for i, path in enumerate(paths)}
    identity = dict(
        teacher_candidate_sha256=digest(teacher_path),
        teacher_bundle_sha256=verified["bundle_sha256"],
        teacher_probabilities_sha256=digest(
            root / "fusion_refined_selection_validation/development_probabilities.npz"
        ),
        negative_probabilities_sha256=digest(negative_path),
        negative_probabilities_path=str(negative_path),
        baseline_rows_sha256=digest(leaf / "cnn_inputs/rows.json"),
        paths_sha256=teacher["input_identity"]["paths_sha256"],
        routing_sha256=hashlib.sha256(
            Path(__file__).with_name("context_guard.py").read_text().encode()
        ).hexdigest(),
        negative_source=a.negative_source,
        negative_directory=str(negative_directory),
        negative_checkpoint_sha256=negative_meta["head_sha256"],
        negative_calibration_sha256=negative_meta["calibration_sha256"],
        negative_encoder_sha256=negative_meta["encoder_sha256"],
        positive_source=a.positive_source,
        positive_directory=str(positive_directory) if positive_directory else None,
        positive_input_identity=positive_meta,
        positive_probabilities_sha256=digest(positive_path),
        positive_probabilities_path=str(positive_path),
        additional_coarse_and_paired_checks_are_diagnostic=a.original_gates_only,
        original_retention_rules_unchanged=True,
        cold_ad_used=False,
        whole_reference_sha256=digest(whole_path) if a.closedset_guard else None,
        threshold_grid=dict(
            healthy=a.healthy_thresholds,
            negative=a.negative_thresholds,
            genus=a.genus_thresholds,
        ),
    )
    (out / "protocol.json").write_text(json.dumps(identity, indent=2))
    ranks = []
    for high, low, gmin in itertools.product(
        [float(v) for v in a.healthy_thresholds.split(",")],
        [float(v) for v in a.negative_thresholds.split(",")],
        [float(v) for v in a.genus_thresholds.split(",")],
    ):
        policy = dict(healthy_min=high, negative_healthy_max=low, genus_min=gmin)
        if a.closedset_guard:
            policy.update(in_domain_other_max=0.1, in_domain_disease_mass_min=0.6)
        if a.protect_suspected_disease:
            policy["protect_suspected_disease"] = True
        if a.preserve_all_uncertainty:
            policy["preserve_all_uncertainty"] = True
        revised = {
            name: [
                route_guard(
                    r,
                    manifest,
                    float(healthy[lookup[r["path"]]]),
                    float(negative[lookup[r["path"]]]),
                    float(genus[lookup[r["path"]]]),
                    policy,
                )
                for r in values
            ]
            for name, values in base.items()
        }
        observed = grouped(revised)
        raw = coarse_counts(revised)
        coarse_pass = all(
            v[1] <= original_coarse[k][1] + math.floor(0.005 * v[0])
            for k, v in raw.items()
        )
        lost_healthy = sum(
            r["label"] == "Healthy"
            and r["app_replay"]["kind"] == "HEALTHY"
            and s["app_replay"]["kind"] != "HEALTHY"
            for name in base
            for r, s in zip(base[name], revised[name])
        )
        disease_changes = sum(
            r["app_replay"]["kind"] == "DISEASE" and r["app_replay"] != s["app_replay"]
            for name in base
            for r, s in zip(base[name], revised[name])
        )
        assert disease_changes == 0
        field = observed["phase5_selection"]
        passed = all(
            field["by_source"][source]["healthy_recall"] >= 0.7
            for source in ["tom2024", "swtt5wy2pr"]
        )
        retained = retain(observed, baseline)
        score = float(
            np.mean(
                [
                    (v["healthy"]["correct_rate"] or 0)
                    + (v["supported_disease"]["correct_rate"] or 0)
                    - 3 * (v["all"]["false_healthy_rate"] or 0)
                    for v in observed.values()
                ]
            )
        )
        candidate = dict(
            id=len(ranks),
            policy=policy,
            retained=retained,
            field_healthy_passed=passed,
            coarse_condition_retained=coarse_pass,
            lost_original_correct_healthy=lost_healthy,
            changed_original_named_disease=disease_changes,
            score=score,
            metrics=observed,
            fusion=teacher["fusion"] if a.positive_source == "fusion" else None,
            input_identity=positive_meta,
            guard_identity=identity,
            negative_scope="Uncertain evidence conflict; original leaf probabilities and observations retained",
            restricted_teacher_used=False,
            cold_ad_used=False,
            native_context_runtime_implemented=False,
        )
        ranks.append(candidate)
    if a.original_gates_only:
        ranks.sort(
            key=lambda c: (
                c["retained"] and c["field_healthy_passed"],
                c["retained"],
                (
                    sum(
                        c["metrics"]["phase5_selection"]["by_source"][s][
                            "healthy_recall"
                        ]
                        for s in ["tom2024", "swtt5wy2pr"]
                    )
                    / 2
                    if a.selection_objective == "field_recall"
                    else c["score"]
                ),
                c["score"],
            ),
            reverse=True,
        )
    else:
        ranks.sort(
            key=lambda c: (
                c["retained"]
                and c["field_healthy_passed"]
                and c["coarse_condition_retained"]
                and c["lost_original_correct_healthy"] == 0,
                c["retained"]
                and c["coarse_condition_retained"]
                and c["lost_original_correct_healthy"] == 0,
                c["retained"],
                c["score"],
            ),
            reverse=True,
        )
    best = ranks[0]
    (out / "candidate.json").write_text(json.dumps(best, indent=2))
    (out / "ranked.json").write_text(
        json.dumps(
            [
                {
                    k: v
                    for k, v in c.items()
                    if k
                    not in ("metrics", "fusion", "input_identity", "guard_identity")
                }
                for c in ranks
            ],
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(
                id=best["id"],
                policy=best["policy"],
                retained=best["retained"],
                field_healthy_passed=best["field_healthy_passed"],
                coarse_retained=best["coarse_condition_retained"],
                lost_healthy=best["lost_original_correct_healthy"],
                field_sources=best["metrics"]["phase5_selection"]["by_source"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
