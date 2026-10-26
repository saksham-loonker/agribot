"""Screen the completed supervised model and one cached hybrid alternative.

Never reads sealed OLID images. Reuses exact-checkpoint validation logits and
completed, identity-checked frozen-head predictions. Only missing supervised
full-frame inputs are inferred. No center crops, retraining, export or native
integration is scheduled here. Passing still requires fresh graph verification.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import time
from collections import Counter

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from campaign_pixels import open_pixels
from context_guard import route_guard
from evaluation_inputs import input_identity
from feature_cache_reuse import validate_completed_contract
from leaf_feature_expert import LeafFeatureHead
from permissive_encoder import digest
from permissive_scene import load
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics
from phase17_context_select import coarse_counts


def read(path):
    return json.loads(Path(path).read_text())


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -80.0, 80.0)))


def retention_failures(candidate, baseline):
    failures = []
    for name, m in candidate.items():
        b = baseline[name]
        tests = [
            (
                "healthy_count",
                m["healthy"]["correct"],
                b["healthy"]["correct"],
                "minimum",
            ),
            (
                "named_disease_correct",
                m["supported_disease"]["correct"],
                b["supported_disease"]["correct"]
                - math.floor(0.01 * b["supported_disease"]["n"]),
                "minimum",
            ),
            (
                "named_disease_false_healthy",
                m["supported_disease"]["false_healthy"],
                b["supported_disease"]["false_healthy"]
                + math.floor(0.005 * b["supported_disease"]["n"]),
                "maximum",
            ),
            (
                "all_false_healthy",
                m["all"]["false_healthy"],
                b["all"]["false_healthy"]
                + math.floor(0.005 * b["supported_disease"]["n"]),
                "maximum",
            ),
            (
                "supported_false_other",
                m["supported"]["false_other"],
                b["supported"]["false_other"] + math.floor(0.005 * b["supported"]["n"]),
                "maximum",
            ),
        ]
        tests += [
            (
                scope + "_false_known",
                m[scope]["false_known"] or 0,
                (b[scope]["false_known"] or 0) + math.floor(0.005 * b[scope]["n"]),
                "maximum",
            )
            for scope in ["unsupported_tomato", "non_tomato"]
        ]
        tests += [
            (
                label + "_correct",
                c["correct"],
                b["disease_classes"][label]["correct"] - math.floor(0.01 * c["n"]),
                "minimum",
            )
            for label, c in m["disease_classes"].items()
        ]
        for metric, observed, limit, direction in tests:
            if observed < limit if direction == "minimum" else observed > limit:
                failures.append(
                    dict(
                        cohort=name,
                        metric=metric,
                        observed=observed,
                        required=limit,
                        direction=direction,
                    )
                )
    assert (not failures) == retain(
        candidate, baseline
    ), "Diagnostic must match unchanged retention implementation"
    return failures


def cached_score_identity(directory, selection, paths):
    identity = read(selection / "input_identity.json")
    assert identity["head_sha256"] == digest(directory / "best.pt")
    assert identity["calibration_sha256"] == digest(directory / "result.json")
    assert identity["encoder_sha256"] == digest(directory.parent / "encoder.pt")
    assert (
        identity["paths_sha256"]
        == hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    )
    return identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out-name", default="phase27_cache_screen")
    ap.add_argument("--scores-from")
    ap.add_argument("--run-name", default="sft_ad_tail2_focal_lowfpr")
    ap.add_argument(
        "--bounded",
        action="store_true",
        help="Eight supervised/CVaR policies, preserve all original Healthy verdicts; no hybrid/reward sweep",
    )
    a = ap.parse_args()
    started = time.monotonic()
    p = Path(a.campaign)
    root = p / "phase23_siglip"
    assert Path(a.run_name).name == a.run_name and a.run_name not in (".", "..")
    run = root / a.run_name
    assert Path(a.out_name).name == a.out_name
    out = p / a.out_name
    out.mkdir(exist_ok=True)
    assert read(run / "restored_model_parity.json")["passed"]
    fit = read(run / "result.json")
    assert digest(run / "best.pt") == fit["checkpoint_sha256"]
    assert fit["configuration"]["consumed_ad_used_for_training"] is True
    rows = read(root / "cnn_inputs/rows.json")
    assert digest(root / "cnn_inputs/rows.json") == fit["configuration"]["rows_sha256"]
    leaf = p / "phase10_leaf"
    manifest = read(p / "export_cnn_safe/model_manifest.json")
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
    paths = sorted({r["path"] for values in base.values() for r in values})
    internal_file = p / "phase23_data/internal_baseline_predictions.jsonl"
    internal = [json.loads(line) for line in internal_file.read_text().splitlines()]
    inputs = sorted(set(paths) | {r["path"] for r in internal})
    assert not any("phase24_olid_source" in path for path in inputs)
    photos = {
        r["path"]: r for values in list(base.values()) + [internal] for r in values
    }
    source_inputs = input_identity(
        [
            dict(
                id=path,
                path=path,
                **(
                    {"file_sha256": photos[path]["file_sha256"]}
                    if photos[path].get("file_sha256")
                    else {}
                )
            )
            for path in inputs
        ]
    )
    whole_file = p / "phase23_data/whole_reference.jsonl"
    whole = {r["path"]: r for r in map(json.loads, whole_file.read_text().splitlines())}
    for values in list(base.values()) + [internal]:
        for r in values:
            if r["app_replay"].get("probabilities") is None:
                r["context_whole_reference_probabilities"] = whole[r["path"]][
                    "whole_probabilities"
                ]
    validation = np.load(run / "validation_logits.npz")
    val_ids, raw_logits = validation["indices"], validation["logits"]
    assert np.array_equal(
        val_ids, np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    )
    assert raw_logits.shape == (len(val_ids),) and np.isfinite(raw_logits).all()
    scores = {}
    for index, z in zip(val_ids, raw_logits):
        path = rows[int(index)]["path"]
        if path in scores:
            assert abs(scores[path] - float(z)) < 1e-6
        scores[path] = float(z)
    reused = sum(path in scores for path in inputs)
    missing = [path for path in inputs if path not in scores]
    protocol = dict(
        checkpoint_sha256=fit["checkpoint_sha256"],
        training_result_sha256=digest(run / "result.json"),
        validation_logits_sha256=digest(run / "validation_logits.npz"),
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        full_input_paths_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
        source_inputs_sha256=hashlib.sha256(
            json.dumps(source_inputs).encode()
        ).hexdigest(),
        input_scope="Existing development and consumed grouped internal only, full_roi NHWC224 canonical ImageNet",
        preprocessing_sha256=digest(Path(__file__).with_name("app_eval.py")),
        internal_baseline_sha256=digest(internal_file),
        whole_reference_sha256=digest(whole_file),
        script_sha256=digest(__file__),
        routing_sha256=digest(Path(__file__).with_name("context_guard.py")),
        frozen_feature_array_sha256=digest(root / "siglip2_feature_heads/features.npy"),
        sealed_olid_used=False,
        consumed_ad_is_independent=False,
        unique_inputs=len(inputs),
        reused_exact_checkpoint_validation_logits=reused,
        missing_supervised_inputs=len(missing),
        optional_center_views_computed=0,
        original_recognition_and_safety_gates_unchanged=True,
        fixed_policy_grid=dict(
            healthy=(
                [0.8, 0.9, 0.95, 0.99]
                if a.bounded
                else [0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]
            ),
            rejection=[0.001, 0.01] if a.bounded else [0.0, 0.001, 0.005, 0.01, 0.02],
        ),
        supervised_run=a.run_name,
        bounded_supervised_only=a.bounded,
        prior_missing_logits_sha256=(
            digest(Path(a.scores_from) / "supervised_missing_logits.json")
            if a.scores_from
            else None
        ),
    )
    metadata = out / "protocol.json"
    if metadata.exists():
        assert read(metadata) == protocol, "Cache identity changed; version output"
    else:
        metadata.write_text(json.dumps(protocol, indent=2))
    cache = out / "supervised_missing_logits.json"
    if a.scores_from and not cache.exists():
        previous = Path(a.scores_from)
        prior = read(previous / "protocol.json")
        for key in [
            "checkpoint_sha256",
            "training_result_sha256",
            "validation_logits_sha256",
            "rows_sha256",
            "full_input_paths_sha256",
            "source_inputs_sha256",
            "preprocessing_sha256",
            "internal_baseline_sha256",
        ]:
            assert prior[key] == protocol[key], ("Score cache mismatch", key)
        cache.write_text((previous / "supervised_missing_logits.json").read_text())
    done = read(cache) if cache.exists() else {}
    assert set(done).issubset(set(missing))
    pending = [path for path in missing if path not in done]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    inference_started = time.monotonic()
    if pending:
        _, model = load(run)
        model = model.cuda().eval()

        def prepare(path):
            with Image.open(path) as im:
                return classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)

        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(pending), 64):
                part = pending[start : start + 64]
                tensor = torch.from_numpy(
                    np.concatenate(list(pool.map(prepare, part)), 0)
                ).cuda()
                # load() includes the final checkpoint's recorded calibration.
                z = model(tensor).cpu().numpy()
                assert np.isfinite(z).all()
                done.update({path: float(value) for path, value in zip(part, z)})
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps(done))
                tmp.replace(cache)
        del model
        torch.cuda.empty_cache()
    inference_seconds = time.monotonic() - inference_started
    assert set(done) == set(missing)
    calibration = fit["calibration"]
    supervised = {
        path: (
            float(sigmoid(scores[path] * calibration[0] + calibration[1]))
            if path in scores
            else float(sigmoid(done[path]))
        )
        for path in inputs
    }
    paths_index = {path: i for i, path in enumerate(paths)}
    internal_index = {r["path"]: i for i, r in enumerate(internal)}
    positives = {"supervised_tail2": supervised}
    if not a.bounded:
        hybrid = p / "phase23_hybrid"
        hybrid_directory = Path(
            read(hybrid / "hybrid_field_feature_heads/ranked.json")[0]["checkpoint"]
        ).parent
        cached_score_identity(hybrid_directory, hybrid / "selection", paths)
        hp = np.load(hybrid / "selection/probabilities.npz")["healthy"][:, 0]
        hi = np.load(hybrid / "selection_validation/internal_probabilities.npz")[
            "healthy"
        ][:, 0]
        hybrid_verification = read(hybrid / "selection_validation/result.json")
        assert hybrid_verification["candidate_sha256"] == digest(
            hybrid / "selection/candidate.json"
        )
        assert hybrid_verification["internal_baseline_sha256"] == digest(internal_file)
        positives["cached_hybrid"] = {
            path: (
                float(hp[paths_index[path]])
                if path in paths_index
                else float(hi[internal_index[path]])
            )
            for path in inputs
        }
    negatives = {}
    feature_directory = root / "siglip2_feature_heads"
    validate_completed_contract(root, feature_directory, "siglip2")
    feature_identity = read(feature_directory / "feature_identity.json")
    assert feature_identity["encoder_sha256"] == digest(
        feature_directory / "encoder.pt"
    )
    assert feature_identity["rows_sha256"] == digest(root / "cnn_inputs/rows.json")
    assert read(feature_directory / "restored_encoder_parity.json")["passed"]
    features = np.load(feature_directory / "features.npy", mmap_mode="r")
    row_index = {r["path"]: i for i, r in enumerate(rows)}
    # All admitted internal examples are exact rows in the frozen feature cache.
    assert all(r["path"] in row_index for r in internal)
    pixels = open_pixels(root / "cnn_inputs")

    def verify_internal_input(r):
        index = row_index[r["path"]]
        assert r["group"] == rows[index]["group"]
        assert digest(r["path"]) == r["file_sha256"]
        with Image.open(r["path"]) as im:
            fresh = classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)[0]
        assert np.array_equal(
            fresh, pixels[index]
        ), "Cached feature input differs from canonical source pixels"

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(verify_internal_input, internal))
    if hasattr(pixels, "close"):
        pixels.close()
    iv = torch.from_numpy(
        np.array(features[[row_index[r["path"]] for r in internal]], copy=True)
    )
    genus_identity = read(root / "selection/input_identity.json")
    assert genus_identity["genus_checkpoint_sha256"] == digest(
        p / "phase6_roi/species/best.pt"
    )
    assert genus_identity["genus_calibration_sha256"] == digest(
        p / "phase6_roi/species/result.json"
    )
    assert read(root / "selection_validation/result.json")[
        "internal_baseline_sha256"
    ] == digest(internal_file)
    old_genus = np.load(root / "selection/probabilities.npz")["genus"]
    internal_genus = np.load(root / "selection_validation/internal_probabilities.npz")[
        "genus"
    ]
    genus = {
        path: (
            float(old_genus[paths_index[path]])
            if path in paths_index
            else float(internal_genus[internal_index[path]])
        )
        for path in inputs
    }
    modes = (
        ["cvar_bce"]
        if a.bounded
        else [
            "cvar_bce",
            "partial_auc",
            "reinforce_label_reward",
            "expected_logit_reward",
        ]
    )
    for mode in modes:
        directory = root / "risk_heads" / mode
        cached_score_identity(directory, root / ("risk_" + mode + "_selection"), paths)
        state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
        assert state["protocol"]["feature_identity"] == feature_identity
        head = LeafFeatureHead(state["kind"], 768).eval()
        head.load_state_dict(state["model"], strict=True)
        cal = read(directory / "result.json")["calibration"]
        with torch.inference_mode():
            ih = (head(iv) * cal[0] + cal[1]).sigmoid().numpy()
        old = np.load(root / ("risk_" + mode + "_selection/probabilities.npz"))[
            "healthy"
        ][:, 0]
        negatives[mode] = {
            path: (
                float(old[paths_index[path]])
                if path in paths_index
                else float(ih[internal_index[path]])
            )
            for path in inputs
        }
    baseline = grouped(base)
    sources = sorted({r["source"] for r in internal})
    internal_baseline = {
        s: photo_metrics([r for r in internal if r["source"] == s]) for s in sources
    }
    coarse_before = coarse_counts(base)
    ranks, detailed, evaluated = [], {}, 0
    failures_count = Counter()
    screening_started = time.monotonic()
    for positive_name, negative_name, high, low in itertools.product(
        positives,
        negatives,
        protocol["fixed_policy_grid"]["healthy"],
        protocol["fixed_policy_grid"]["rejection"],
    ):
        evaluated += 1
        h, n = positives[positive_name], negatives[negative_name]
        policy = dict(
            healthy_min=high,
            negative_healthy_max=low,
            genus_min=0.0,
            protect_suspected_disease=True,
            in_domain_other_max=0.1,
            in_domain_disease_mass_min=0.6,
        )
        if a.bounded:
            policy["preserve_original_healthy"] = True

        def route(r):
            return route_guard(
                r, manifest, h[r["path"]], n[r["path"]], genus[r["path"]], policy
            )

        observed_rows = {k: [route(r) for r in values] for k, values in base.items()}
        observed = grouped(observed_rows)
        old_failures = retention_failures(observed, baseline)
        old_retained = not old_failures
        field_passed = all(
            observed["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
            for s in ["tom2024", "swtt5wy2pr"]
        )
        coarse_after = coarse_counts(observed_rows)
        coarse_passed = all(
            v[1] <= coarse_before[k][1] + int(0.005 * v[0])
            for k, v in coarse_after.items()
        )
        after = [route(r) for r in internal]
        metrics = {
            s: photo_metrics([r for r in after if r["source"] == s]) for s in sources
        }
        internal_retained = retain(metrics, internal_baseline)
        source_healthy_passed = all(
            m["healthy"]["correct_rate"] >= 0.7
            for m in metrics.values()
            if m["healthy"]["n"]
        )
        summary = dict(
            id=evaluated,
            positive=positive_name,
            negative=negative_name,
            policy=policy,
            old_development_retained=old_retained,
            old_field_healthy_targets=field_passed,
            coarse_condition_retained=coarse_passed,
            internal_retained=internal_retained,
            internal_healthy_each_source=source_healthy_passed,
            passed_cheap_checks=old_retained
            and field_passed
            and coarse_passed
            and internal_retained
            and source_healthy_passed,
            old_retention_failures=old_failures,
            internal_healthy={s: m["healthy"] for s, m in metrics.items()},
            internal_false_healthy={
                s: m["all"]["false_healthy"] for s, m in metrics.items()
            },
        )
        ranks.append(summary)
        for key, passed in [
            ("old_retention", old_retained),
            ("old_field_healthy", field_passed),
            ("coarse_condition", coarse_passed),
            ("internal_retention", internal_retained),
            ("internal_healthy_each_source", source_healthy_passed),
        ]:
            if not passed:
                failures_count[key] += 1
        detailed[evaluated] = dict(
            summary, old_metrics=observed, internal_metrics=metrics
        )
    # Selection remains development-only. Rank by passing all app gates, then
    # weakest-source Healthy recall and total false Healthy, never benchmark AUC.
    ranks.sort(
        key=lambda x: (
            x["passed_cheap_checks"],
            x["old_development_retained"]
            and x["coarse_condition_retained"]
            and x["internal_retained"],
            x["internal_retained"],
            min(m["correct_rate"] for m in x["internal_healthy"].values() if m["n"]),
            -sum(x["internal_false_healthy"].values()),
        ),
        reverse=True,
    )
    result = dict(
        protocol=protocol,
        evaluated_policies=evaluated,
        eligible_old_development=sum(
            r["old_development_retained"]
            and r["old_field_healthy_targets"]
            and r["coarse_condition_retained"]
            for r in ranks
        ),
        failed_gate_counts=dict(failures_count),
        best_by_positive={
            name: detailed[next(r for r in ranks if r["positive"] == name)["id"]]
            for name in positives
        },
        passing_cheap_checks=sum(r["passed_cheap_checks"] for r in ranks),
        best=detailed[ranks[0]["id"]] if ranks else None,
        timings=dict(
            inference_seconds=inference_seconds,
            screening_seconds=time.monotonic() - screening_started,
            total_seconds=time.monotonic() - started,
        ),
        inference_new_images=len(pending),
        cached_validation_inputs=reused,
        frozen_encoders_executed_for_heads=0,
        sealed_olid_used=False,
        consumed_ad_is_independent=False,
        full_verification_required=bool(ranks and ranks[0]["passed_cheap_checks"]),
        stop_rule=(
            "If none of the eight fixed policies passes every unchanged app gate, stop this checkpoint; no threshold expansion, full replay, export or acceptance opening."
            if a.bounded
            else "If no policy passes all original, coarse-condition and per-source internal app gates, stop this checkpoint and the cached hybrid; no new reward/head sweeps, export or acceptance opening."
        ),
    )
    (out / "ranked.json").write_text(json.dumps(ranks, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passing=result["passing_cheap_checks"],
                best=result["best"],
                timings=result["timings"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
