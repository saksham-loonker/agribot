"""Freeze and freshly replay bounded recovery without reading cold AD photos."""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_guard import route_guard
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics
from phase17_context_select import coarse_counts
from phase21_guard_model import build, restore


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--selection", default="phase21_bounded_cvar_bce")
    ap.add_argument("--out-name", default="phase21_verified")
    ap.add_argument("--coarse-priority", action="store_true")
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / a.out_name
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(641)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    ranked = read(p / a.selection / "ranked.json")
    # Selection uses the original development cohorts only. Prevent individual
    # correct Healthy losses as an additional conservative selection condition.
    chosen = next(
        c
        for c in ranked
        if c["retained"]
        and c["field_healthy_passed"]
        and (
            c["coarse_condition_retained"]
            if a.coarse_priority
            else c["lost_original_correct_healthy"] == 0
        )
    )
    selected = read(p / a.selection / "candidate.json")
    selected.update(chosen)
    manifest = read(p / "export_cnn_safe/model_manifest.json")
    leaf = p / "phase10_leaf"
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
    identity = selected["guard_identity"]
    assert (
        hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        == identity["paths_sha256"]
    )
    whole_path = p / "phase20_whole_reference/predictions.jsonl"
    assert digest(whole_path) == identity["whole_reference_sha256"]
    whole = {r["path"]: r for r in map(json.loads, whole_path.read_text().splitlines())}

    def attach(row):
        if row["app_replay"].get("probabilities") is None:
            row["context_whole_reference_probabilities"] = whole[row["path"]][
                "whole_probabilities"
            ]
        return row

    for values in base.values():
        for row in values:
            attach(row)
    for key in ["positive", "negative"]:
        assert (
            digest(identity[key + "_probabilities_path"])
            == identity[key + "_probabilities_sha256"]
        )
    positive = np.load(identity["positive_probabilities_path"])
    negative = np.load(identity["negative_probabilities_path"])
    expected = np.stack(
        [positive["healthy"][:, 0], positive["genus"], negative["healthy"][:, 0]], 1
    )
    index = {path: i for i, path in enumerate(paths)}
    policy = selected["policy"]

    def route(row, probabilities):
        return route_guard(
            row,
            manifest,
            float(probabilities[0]),
            float(probabilities[2]),
            float(probabilities[1]),
            policy,
        )

    cached_rows = {
        k: [route(r, expected[index[r["path"]]]) for r in values]
        for k, values in base.items()
    }
    selected["metrics"] = grouped(cached_rows)
    selected["selection_rule"] = (
        "Original-cohort ranked eligible policy with per-raw-condition retention; individual Healthy transitions diagnostic, original per-cohort Healthy nondecrease mandatory"
        if a.coarse_priority
        else "Original-cohort ranked eligible policy with zero lost individually correct Healthy; internal subset not used for selection"
    )
    selected_path = out / "candidate.json"
    if selected_path.exists():
        assert read(selected_path) == selected, "Candidate changed; version output"
    else:
        selected_path.write_text(json.dumps(selected, indent=2))
    internal = p / "phase12_scene/union_app_validation/baseline_predictions.jsonl"
    protocol = dict(
        candidate_sha256=digest(selected_path),
        selection_rank_sha256=digest(p / a.selection / "ranked.json"),
        internal_baseline_sha256=digest(internal),
        whole_reference_sha256=digest(whole_path),
        script_sha256=digest(__file__),
        model_source_sha256=digest(Path(__file__).with_name("phase21_guard_model.py")),
        routing_sha256=digest(Path(__file__).with_name("context_guard.py")),
        cold_ad_used=False,
        restricted_teacher_used=False,
        original_gates_unchanged=True,
    )
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        assert read(protocol_path) == protocol, "Protocol changed; version output"
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    spec, model = build(p, selected)
    bundle = out / "pytorch_bundle.pt"
    if not bundle.exists():
        torch.save(dict(spec, model=model.state_dict()), bundle)
    fresh = restore(bundle).cuda().eval()
    model = model.cuda().eval()
    with torch.inference_mode():
        probe = torch.rand(3, 224, 224, 3, device="cuda")
        restore_error = float((model(probe) - fresh(probe)).abs().max())
        assert restore_error == 0.0
        # Also verify shared-feature execution against the existing unfused
        # implementation and independently computed guard head on real input.
        unshared = model.base(probe)
        guard = (
            model.negative_head(model.base.permissive.encoder(probe))
            * model.negative_calibration[0]
            + model.negative_calibration[1]
        )
        shared_error = float(
            (model(probe) - torch.cat([unshared, guard[:, None]], 1)).abs().max()
        )
        assert shared_error < 0.00001
    del model
    model = fresh

    def infer(paths, destination):
        def prepare(path):
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
            return classifier_letterbox_input(rgb, 224)

        result = np.empty((len(paths), 3), np.float32)
        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(paths), 32):
                part = paths[start : start + 32]
                inputs = torch.from_numpy(
                    np.concatenate(list(pool.map(prepare, part)), 0)
                ).cuda()
                result[start : start + len(part)] = (
                    model(inputs).sigmoid().cpu().numpy()
                )
                if start % 512 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage=destination.stem,
                                complete=min(start + 32, len(paths)),
                                total=len(paths),
                            )
                        ),
                        flush=True,
                    )
        assert np.isfinite(result).all()
        np.save(destination, result)
        return result

    actual = infer(paths, out / "development_probabilities.npy")
    probability_error = float(np.max(np.abs(actual - expected)))
    assert probability_error < 0.002, probability_error
    observed_rows = {
        k: [route(r, actual[index[r["path"]]]) for r in values]
        for k, values in base.items()
    }
    cases = sum(map(len, base.values()))
    verdict_matches = sum(
        r["app_replay"] == s["app_replay"]
        for k in base
        for r, s in zip(cached_rows[k], observed_rows[k])
    )

    # Healthy evidence probability may differ by rounding; compare diagnosis
    # keys separately, and require aggregate metrics to be exactly identical.
    def key(r):
        v = r["app_replay"]
        return v["kind"], v.get("label"), v.get("scope")

    verdict_matches = sum(
        key(r) == key(s) for k in base for r, s in zip(cached_rows[k], observed_rows[k])
    )
    assert verdict_matches == cases
    observed = grouped(observed_rows)
    assert observed == selected["metrics"]
    records = [attach(json.loads(line)) for line in internal.read_text().splitlines()]
    admitted = {
        r["id"]: r
        for r in read(p / "phase13_siglip/admission_manifest.json")
        if r["split"] == "val"
    }
    assert set(admitted) == {r["id"] for r in records}
    for r in records:
        truth = admitted[r["id"]]
        assert (r["label"], r["group"], r["file_sha256"]) == (
            truth["label"],
            truth["group"],
            truth["file_sha256"],
        )
        assert digest(r["path"]) == r["file_sha256"]
    internal_prob = infer(
        [r["path"] for r in records], out / "internal_probabilities.npy"
    )
    after = [route(r, internal_prob[i]) for i, r in enumerate(records)]
    sources = sorted({r["source"] for r in records})
    before_metrics = {
        s: photo_metrics([r for r in records if r["source"] == s]) for s in sources
    }
    after_metrics = {
        s: photo_metrics([r for r in after if r["source"] == s]) for s in sources
    }
    healthy = photo_metrics(after)["healthy"]
    lost = sum(
        r["label"] == "Healthy"
        and r["app_replay"]["kind"] == "HEALTHY"
        and s["app_replay"]["kind"] != "HEALTHY"
        for k in base
        for r, s in zip(base[k], observed_rows[k])
    )
    disease_changes = sum(
        r["app_replay"]["kind"] == "DISEASE" and r["app_replay"] != s["app_replay"]
        for k in base
        for r, s in zip(base[k], observed_rows[k])
    )
    checks = dict(
        development_graph_matches=verdict_matches == cases,
        development_retained=retain(observed, grouped(base)),
        development_healthy_targets=selected["field_healthy_passed"],
        zero_changed_original_named_disease=disease_changes == 0,
        internal_retained=retain(after_metrics, before_metrics),
        internal_healthy_target=healthy["correct"] / healthy["n"] >= 0.7,
    )
    if a.coarse_priority:
        checks["additional_coarse_condition_retained"] = selected[
            "coarse_condition_retained"
        ]
    else:
        checks["zero_lost_original_correct_healthy"] = lost == 0
    result = dict(
        checks=checks,
        passed_preacceptance=all(checks.values()),
        protocol=protocol,
        bundle_sha256=digest(bundle),
        restore_max_abs_error=restore_error,
        shared_graph_max_abs_error=shared_error,
        max_abs_probability_error=probability_error,
        development_verdict_cases=cases,
        development_verdict_matches=verdict_matches,
        old_development=observed,
        coarse_condition_before=coarse_counts(base),
        coarse_condition_after=coarse_counts(observed_rows),
        coarse_condition_retained=selected["coarse_condition_retained"],
        lost_original_correct_healthy=lost,
        changed_original_named_disease=disease_changes,
        internal_baseline=before_metrics,
        internal_candidate=after_metrics,
        internal_healthy=healthy,
        internal_all=photo_metrics(after)["all"],
        internal_photos=len(records),
        independent=False,
        cold_ad_used=False,
        restricted_teacher_used=False,
        native_context_runtime_implemented=False,
        retired_cohort_is_benchmark=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    for name, rows in [
        ("internal_candidate_predictions", after),
        (
            "development_candidate_predictions",
            [r for values in observed_rows.values() for r in values],
        ),
    ]:
        (out / (name + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    print(
        json.dumps(
            dict(checks=checks, Healthy=healthy, passed=result["passed_preacceptance"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
