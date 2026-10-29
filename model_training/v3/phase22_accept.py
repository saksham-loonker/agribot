"""One frozen, prospective external opening; failure consumes independence."""

from pathlib import Path
import argparse
import concurrent.futures
import json
import threading

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import Models, classifier_letterbox_input, image_record
from context_guard import route_guard
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase8_accept import wilson
from phase21_guard_model import restore


def read(path):
    return json.loads(Path(path).read_text())


def paired_interval(records, before, after, predicate, outcome, seed=661):
    selected = [i for i, r in enumerate(records) if predicate(r)]
    if not selected:
        return None
    groups = sorted({records[i]["group"] for i in selected})
    gains = np.array(
        [outcome(after[i]) - outcome(before[i]) for i in selected], np.float64
    )
    group_values = [
        gains[[j for j, i in enumerate(selected) if records[i]["group"] == g]]
        for g in groups
    ]
    rng = np.random.default_rng(seed)
    values = np.array(
        [
            np.concatenate(
                [group_values[i] for i in rng.integers(0, len(groups), len(groups))]
            ).mean()
            for _ in range(5000)
        ]
    )
    return dict(
        n=len(selected),
        conservative_families=len(groups),
        point=float(gains.mean()),
        bootstrap_two_sided_95=np.quantile(values, [0.025, 0.975]).tolist(),
        bootstrap_one_sided_upper_95=float(np.quantile(values, 0.95)),
        interpretation="Empirical paired family bootstrap; inferred visual families are not verified plant identities",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--freeze-only", action="store_true")
    a = ap.parse_args()
    p = Path(a.campaign)
    verified = p / "phase22_verified"
    validation = read(verified / "result.json")
    assert validation["passed_preacceptance"] and all(validation["checks"].values())
    selected = read(verified / "candidate.json")
    bundle = verified / "pytorch_bundle.pt"
    assert digest(bundle) == validation["bundle_sha256"]
    manifest_path = p / "phase9_ad_source/sealed_geometric_manifest.json"
    assert (
        digest(manifest_path)
        == "5cf37861dfe835c918643b2753a05c229e9327cd750524435006384bdcfae6b6"
    )
    records = read(manifest_path)
    assert len(records) == 502 and len({r["id"] for r in records}) == 502
    assets = p / "export_cnn_safe"
    manifest = read(assets / "model_manifest.json")
    assert (
        manifest["classifier"]["members"][0]["sha256"]
        == "d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39"
    )
    out = p / "phase22_external_acceptance"
    out.mkdir(exist_ok=True)
    protocol = dict(
        stage="PyTorch image-context prototype; actual unchanged LiteRT detector/classifier baseline; not a mobile release",
        candidate_sha256=digest(verified / "candidate.json"),
        bundle_sha256=digest(bundle),
        preacceptance_sha256=digest(verified / "result.json"),
        source_manifest_sha256=digest(manifest_path),
        n=502,
        baseline_manifest_sha256=digest(assets / "model_manifest.json"),
        policy=selected["policy"],
        script_sha256=digest(__file__),
        model_source_sha256=digest(Path(__file__).with_name("phase21_guard_model.py")),
        routing_sha256=digest(Path(__file__).with_name("context_guard.py")),
        app_eval_sha256=digest(Path(__file__).with_name("app_eval.py")),
        healthy_recall_min=0.70,
        healthy_gain_min=0.10,
        healthy_gain_family_bootstrap_lower_exclusive=0.0,
        disease_false_healthy_point_increase_max=0.0,
        disease_false_healthy_one_sided_upper_max=0.02,
        sufficient_raw_condition_n=50,
        bootstrap_samples=5000,
        bootstrap_seed=661,
        original_retention="Unmodified phase2_select.retain, integer floors and known-disease denominator",
        group_scope="68 conservative visual families including 20 Healthy families; not verified plant IDs",
        source_scope="Publisher image-level primary visible-leaf category, not every-leaf or whole-plant health",
        failure_action="Block promotion; this source is consumed development afterward. Do not fit or choose another policy and call it independent.",
        export_and_native_parity_required_after_prototype=True,
        restricted_teacher_used=False,
        no_ceiling_exception=True,
    )
    frozen = out / "frozen_protocol.json"
    if frozen.exists():
        assert (
            read(frozen) == protocol
        ), "Frozen protocol changed; do not open a different candidate"
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if a.freeze_only:
        print(
            json.dumps(
                dict(frozen=True, protocol_sha256=digest(frozen), protocol=protocol)
            ),
            flush=True,
        )
        return
    opened = out / "opened_identity.json"
    identity = dict(
        protocol_sha256=digest(frozen),
        source_manifest_sha256=digest(manifest_path),
        bundle_sha256=digest(bundle),
    )
    if opened.exists():
        assert read(opened) == identity
    else:
        opened.write_text(json.dumps(identity, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    model = restore(bundle).cuda().eval()
    local = threading.local()

    def evaluate_baseline(record):
        assert digest(record["path"]) == record["file_sha256"]
        if not hasattr(local, "model"):
            local.model = Models(assets, threads=2)
        row = image_record(local.model, record)
        assert "error" not in row
        if row["app_replay"].get("probabilities") is None:
            row["context_whole_reference_probabilities"] = row["whole_probabilities"]
        return row

    predictions = out / "predictions.jsonl"
    done = (
        {r["id"]: r for r in map(json.loads, predictions.read_text().splitlines())}
        if predictions.exists()
        else {}
    )
    assert set(done).issubset({r["id"] for r in records})
    remaining = [r for r in records if r["id"] not in done]
    with torch.inference_mode(), predictions.open(
        "a"
    ) as stream, concurrent.futures.ThreadPoolExecutor(2) as pool:
        for start in range(0, len(remaining), 16):
            part = remaining[start : start + 16]
            baselines = list(pool.map(evaluate_baseline, part))
            pixels = []
            for record in part:
                with Image.open(record["path"]) as image:
                    pixels.append(
                        classifier_letterbox_input(
                            np.asarray(image.convert("RGB")), 224
                        )
                    )
            inputs = torch.from_numpy(np.concatenate(pixels, 0)).cuda()
            probabilities = model(inputs).sigmoid().cpu().numpy()
            assert np.isfinite(probabilities).all()
            for r, baseline, prob in zip(part, baselines, probabilities):
                candidate = route_guard(
                    baseline,
                    manifest,
                    float(prob[0]),
                    float(prob[2]),
                    float(prob[1]),
                    selected["policy"],
                )
                result = dict(
                    id=r["id"],
                    baseline=baseline,
                    candidate=candidate,
                    context_probabilities=prob.tolist(),
                )
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                done[r["id"]] = result
            print(json.dumps(dict(evaluated=len(done), total=len(records))), flush=True)
    before = [done[r["id"]]["baseline"] for r in records]
    after = [done[r["id"]]["candidate"] for r in records]
    b = {"ad_greenhouse": photo_metrics(before)}
    c = {"ad_greenhouse": photo_metrics(after)}
    healthy_gain = paired_interval(
        records,
        before,
        after,
        lambda r: r["label"] == "Healthy",
        lambda r: int(r["app_replay"]["kind"] == "HEALTHY"),
    )
    disease_audits = {}
    raw_labels = sorted({r["raw_label"] for r in records if r["label"] != "Healthy"})
    for raw in [None] + raw_labels:
        predicate = lambda r: r["label"] != "Healthy" and (
            raw is None or r["raw_label"] == raw
        )
        audit = paired_interval(
            records,
            before,
            after,
            predicate,
            lambda r: int(r["app_replay"]["kind"] == "HEALTHY"),
        )
        disease_audits[raw or "all_diseased"] = audit
    safety = all(
        v["point"] <= 0.0 and v["bootstrap_one_sided_upper_95"] <= 0.02
        for v in disease_audits.values()
        if v["n"] >= 50
    )
    changed_disease = sum(
        r["app_replay"]["kind"] == "DISEASE" and r["app_replay"] != s["app_replay"]
        for r, s in zip(before, after)
    )
    healthy = c["ad_greenhouse"]["healthy"]
    checks = dict(
        original_retention=retain(c, b),
        healthy_min_70_percent=healthy["correct"] / healthy["n"] >= 0.7,
        healthy_gain_min_10pp=healthy_gain["point"] >= 0.1,
        paired_family_gain_lower_positive=healthy_gain["bootstrap_two_sided_95"][0]
        > 0.0,
        disease_false_healthy_nonincrease=safety,
        original_named_disease_unchanged=changed_disease == 0,
    )
    result = dict(
        protocol=protocol,
        passed=all(checks.values()),
        checks=checks,
        baseline=b,
        candidate=c,
        paired_healthy_gain=healthy_gain,
        paired_disease_false_healthy=disease_audits,
        healthy_wilson_95=wilson(healthy["correct"], healthy["n"]),
        changed_original_named_disease=changed_disease,
        now_opened=True,
        source_independence_consumed=True,
        candidate_sha256=protocol["candidate_sha256"],
        predictions_sha256=digest(predictions),
        native_verified=False,
        installed=False,
        next_step="If passed, exported-byte full regression and native/device parity remain mandatory. If failed, reject promotion and retire this source from independent acceptance.",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                checks=checks,
                healthy=healthy,
                gain=healthy_gain,
                baseline_false_h=b["ad_greenhouse"]["all"]["false_healthy"],
                candidate_false_h=c["ad_greenhouse"]["all"]["false_healthy"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
