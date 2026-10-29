"""One prospective OLID opening of frozen mobile bytes and production Kotlin routing.

No training, threshold selection or source-specific routing is permitted here.
Every model invocation is cached by model bytes and exact float input bytes.
Failure consumes this source's independence and blocks promotion.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import time

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

from app_eval import canon, classifier_letterbox_input, posterior, quality
from permissive_encoder import digest
from phase22_accept import paired_interval
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase8_accept import wilson


def read(path):
    return json.loads(Path(path).read_text())


class ExactMobileCache:
    def __init__(self, directory, contracts):
        directory.mkdir(exist_ok=True)
        self.db = sqlite3.connect(directory / "predictions.sqlite3")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outputs (identity TEXT PRIMARY KEY, values_blob BLOB NOT NULL)"
        )
        self.models = {}
        self.hits = self.computed = 0
        for name, path, sha in contracts:
            assert digest(path) == sha
            model = Interpreter(model_path=str(path), num_threads=4)
            model.allocate_tensors()
            self.models[name] = (model, sha)

    def run(self, name, pixels):
        model, sha = self.models[name]
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        assert np.isfinite(pixels).all()
        identity = hashlib.sha256(
            sha.encode() + str(pixels.shape).encode() + pixels.tobytes()
        ).hexdigest()
        saved = self.db.execute(
            "SELECT values_blob FROM outputs WHERE identity=?", (identity,)
        ).fetchone()
        if saved:
            self.hits += 1
            return np.frombuffer(saved[0], dtype=np.float32).copy()
        model.set_tensor(model.get_input_details()[0]["index"], pixels)
        model.invoke()
        values = model.get_tensor(model.get_output_details()[0]["index"]).ravel()
        assert np.isfinite(values).all()
        self.db.execute(
            "INSERT INTO outputs VALUES (?,?)",
            (identity, values.astype(np.float32).tobytes()),
        )
        self.db.commit()
        self.computed += 1
        return values


def pixel_case(record, cache, manifest, baseline_manifest):
    assert digest(record["path"]) == record["file_sha256"]
    with Image.open(record["path"]) as image:
        image = np.asarray(image.convert("RGB"))
    q = quality(image)
    usable = not (q["contrast"] < 0.015 and q["sharpness"] < 0.012)
    detection, letterbox = canon.detector_input(image, 320)
    raw = cache.run("detector", detection).reshape(1, 5, -1)
    d = manifest["detector"]
    boxes = canon.decode_detections(
        raw,
        letterbox,
        image.shape[1],
        image.shape[0],
        d["score_threshold"],
        d["nms_iou"],
        d["max_detections"],
    )[:6]
    old_observations, new_observations, views = [], [], []
    for box in boxes:
        center = canon.classifier_input(image, 224, box[:4])
        old = cache.run("baseline", center)
        new = cache.run("leaf", center)
        full = cache.run("leaf", classifier_letterbox_input(image, 224, box[:4]))
        assert len(old) == 9 and len(new) == len(full) == 12
        old_observations.append([box[4], posterior([old], baseline_manifest).tolist()])
        new_observations.append([box[4], posterior([new[:9]], manifest).tolist()])
        views.append([new[9:].tolist(), full[9:].tolist()])
    whole = cache.run("whole", classifier_letterbox_input(image, 224))
    fallback = None
    if not boxes:
        center = canon.classifier_input(image, 224)
        fallback = posterior([cache.run("leaf", center)[:9]], manifest).tolist()
    corroboration = None
    if (
        whole[0] >= 0.85
        and whole[2] >= 0.5
        and whole[3] >= 0.5
        and sum(v[0][2] < 0.5 for v in views) >= 2
    ):
        corroboration = float(
            cache.run("leaf", classifier_letterbox_input(image, 224))[11]
        )
    common = dict(usable=usable, quality_weight=q["quality_weight"])
    return dict(
        id=record["id"],
        baseline=dict(common, observations=old_observations),
        candidate=dict(
            common,
            observations=new_observations,
            whole=whole.tolist(),
            leaves=views,
            fallback=fallback,
            whole_adapted=corroboration,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--android", required=True)
    ap.add_argument("--native-evidence", required=True)
    ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args()
    p, android = Path(args.campaign), Path(args.android)
    out = p / "phase76_fresh_acceptance"
    out.mkdir(exist_ok=True)
    source = p / "phase24_olid_source/sealed_manifest.json"
    membership = read(p / "phase24_olid_source/membership_audit.json")
    assert (
        membership["accepted_n"] == 730
        and membership["accepted_counts"]["tomato__healthy"] == 223
    )
    assert (
        not membership["training_used"]
        and digest(source) == membership["sealed_manifest_sha256"]
    )
    assert read(p / "phase73_mobile_verified/result.json")["passed"]
    assert read(p / "phase78_native_gate_audit/result.json")["passed"]
    native = read(args.native_evidence)
    assert native["passed"] and native["failed"] == native["skipped"] == 0
    assert native["tests"] >= 22
    assets = android / "app/src/main/assets"
    assert native["model_manifest_sha256"] == digest(assets / "model_manifest.json")
    assert (
        native["build_source_sha256"]
        == read(p / "field_native_final_build_result.json")["source_sha256"]
    )
    manifest = read(assets / "model_manifest.json")
    baseline_assets = p / "export_cnn_safe"
    baseline_manifest = read(baseline_assets / "model_manifest.json")
    assert (
        manifest["schema"] == 5
        and manifest["primary_recovery"]["policy"] == "phase69_whole_local"
    )
    contracts = [
        (
            "leaf",
            assets / "models" / manifest["classifier"]["members"][0]["file"],
            manifest["classifier"]["members"][0]["sha256"],
        ),
        (
            "whole",
            assets / "models" / manifest["primary_recovery"]["model"]["file"],
            manifest["primary_recovery"]["model"]["sha256"],
        ),
        (
            "baseline",
            baseline_assets
            / "models"
            / baseline_manifest["classifier"]["members"][0]["file"],
            baseline_manifest["classifier"]["members"][0]["sha256"],
        ),
        (
            "detector",
            assets / "models" / manifest["detector"]["file"],
            manifest["detector"]["sha256"],
        ),
    ]
    for _, path, sha in contracts:
        assert digest(path) == sha
    domain = android / "domain/src"
    native_sources = {
        str(path.relative_to(android)): digest(path)
        for path in sorted(domain.rglob("*.kt"))
        if "/main/" in str(path) or path.name == "RecognitionAcceptanceReplay.kt"
    }
    for path in [
        android
        / "featurescan/src/main/java/com/sakshyam/agribot/featurescan/check/CheckPlantViewModel.kt",
        android
        / "ml/src/main/java/com/sakshyam/agribot/ml/vision/AgribotLeafVision.kt",
        android / "ml/src/main/java/com/sakshyam/agribot/ml/vision/VisionManifest.kt",
    ]:
        native_sources[str(path.relative_to(android))] = digest(path)
    protocol = dict(
        schema=1,
        stage="Final mobile bytes with production Kotlin accumulation and quick-check completion",
        source_manifest_sha256=digest(source),
        source_expected_n=730,
        source_expected_healthy=223,
        sealing_audit_sha256=digest(p / "phase24_olid_source/membership_audit.json"),
        manifest_sha256=digest(assets / "model_manifest.json"),
        models={name: sha for name, _, sha in contracts},
        development_mobile_verification_sha256=digest(
            p / "phase73_mobile_verified/result.json"
        ),
        development_native_gate_sha256=digest(
            p / "phase78_native_gate_audit/result.json"
        ),
        native_evidence_sha256=digest(args.native_evidence),
        native_sources=native_sources,
        script_sha256=digest(__file__),
        preprocessing_sha256=digest(Path(__file__).with_name("app_eval.py")),
        canon_sha256=digest(Path(__file__).parents[1] / "v2/final/canon.py"),
        sparse_resize_sha256=digest(
            Path(__file__).parents[2] / "tests/model_evaluation/canonical_runtime.py"
        ),
        paired_interval_sha256=digest(Path(__file__).with_name("phase22_accept.py")),
        original_retention_sha256=digest(Path(__file__).with_name("phase2_select.py")),
        healthy_recall_min=0.70,
        healthy_gain_min=0.10,
        healthy_gain_family_bootstrap_lower_exclusive=0.0,
        disease_false_healthy_point_increase_max=0.0,
        disease_false_healthy_one_sided_upper_max=0.02,
        sufficient_raw_condition_n=50,
        bootstrap_samples=5000,
        bootstrap_seed=661,
        original_retention="Unmodified phase2_select.retain; integer floors and known-disease denominator",
        unsupported_scope="OLID publisher MIT and JAS_MIT map explicitly to unsupported tomato; N_K stays coarse diseased",
        named_disease_retention="Exact kind, named label and partial status; original graph numeric confidence tolerance .002",
        source_scope="Publisher primary visible-leaf condition; no whole-plant or every-leaf guarantee",
        group_scope="Previously sealed conservative inferred capture families; not verified plant identities",
        failure_action="Block promotion and consume OLID independence permanently; no retuning and rebranding as fresh",
        ad_independent=False,
        restricted_teacher_used=False,
        no_ceiling_exception=True,
    )
    frozen = out / "frozen_protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol, "The frozen candidate or protocol changed"
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    identity = digest(frozen)
    if args.freeze_only:
        print(json.dumps(dict(frozen=True, protocol_sha256=identity)), flush=True)
        return
    marker = out / "opened_identity.json"
    opening = dict(
        protocol_sha256=identity,
        source_manifest_sha256=protocol["source_manifest_sha256"],
    )
    if marker.exists():
        assert read(marker) == opening
    else:
        marker.write_text(json.dumps(opening, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    records = read(source)
    assert len(records) == 730 and sum(r["label"] == "Healthy" for r in records) == 223
    assert len({r["id"] for r in records}) == len(records)
    cache = ExactMobileCache(out / "cache", contracts)
    predictions = out / "mobile_scores.jsonl"
    done = (
        {r["id"]: r for r in map(json.loads, predictions.read_text().splitlines())}
        if predictions.exists()
        else {}
    )
    assert set(done) <= {r["id"] for r in records}
    started = time.monotonic()
    with predictions.open("a") as stream:
        for record in records:
            if record["id"] in done:
                continue
            case = pixel_case(record, cache, manifest, baseline_manifest)
            stream.write(json.dumps(case) + "\n")
            stream.flush()
            done[record["id"]] = case
            if len(done) % 25 == 0:
                print(
                    json.dumps(
                        dict(
                            evaluated=len(done),
                            n=len(records),
                            seconds=time.monotonic() - started,
                            cache_hits=cache.hits,
                            new_inputs=cache.computed,
                        )
                    ),
                    flush=True,
                )
    native_input = out / "native_inputs.json"
    native_input.write_text(
        json.dumps(
            dict(protocol_sha256=identity, cases=[done[r["id"]] for r in records])
        )
    )
    native_output = out / "native_predictions.json"
    env = dict(
        os.environ,
        ANDROID_HOME="/home/genesis/Android/Sdk",
        ANDROID_SDK_ROOT="/home/genesis/Android/Sdk",
    )
    with (out / "native_replay.log").open("w") as stream:
        subprocess.run(
            [
                "bash",
                "gradlew",
                ":domain:replayRecognition",
                "-PrecognitionReplayInput=" + str(native_input),
                "-PrecognitionReplayOutput=" + str(native_output),
                "--no-daemon",
                "--max-workers=1",
            ],
            cwd=android,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    native_rows = read(native_output)
    assert native_rows["protocol_sha256"] == identity
    native_lookup = {r["id"]: r for r in native_rows["cases"]}
    labels = manifest["classifier"]["labels"]
    before, after = [], []
    for record in records:
        for name, target in [("baseline", before), ("candidate", after)]:
            v = dict(native_lookup[record["id"]][name])
            index = v.pop("label_index")
            v["label"] = labels[index] if index is not None else None
            target.append(dict(record, app_replay=v))
    baseline_metrics, candidate_metrics = photo_metrics(before), photo_metrics(after)
    # These are publisher primary labels, not invented bounding-box annotations.
    # The generic detector evaluator cannot infer species from an empty objects
    # list, so assign this known tomato-only unsupported denominator explicitly.
    for rows, metrics in [(before, baseline_metrics), (after, candidate_metrics)]:
        metrics["unsupported_tomato"] = photo_metrics(
            [r for r in rows if r["label"] == "Other"]
        )["all"]
    healthy_gain = paired_interval(
        records,
        before,
        after,
        lambda r: r["label"] == "Healthy",
        lambda r: int(r["app_replay"]["kind"] == "HEALTHY"),
    )
    raw_labels = sorted({r["raw_label"] for r in records if r["label"] != "Healthy"})
    safety = {}
    for raw in [None] + raw_labels:
        safety[raw or "all_diseased"] = paired_interval(
            records,
            before,
            after,
            lambda r: r["label"] != "Healthy"
            and (raw is None or r["raw_label"] == raw),
            lambda r: int(r["app_replay"]["kind"] == "HEALTHY"),
        )
    changed_disease = [
        r["id"]
        for r, b, c in zip(records, before, after)
        if b["app_replay"]["kind"] == "DISEASE"
        and (
            any(
                b["app_replay"][key] != c["app_replay"][key]
                for key in ("kind", "label", "partial")
            )
            or abs(b["app_replay"]["confidence"] - c["app_replay"]["confidence"])
            > 0.002
        )
    ]
    healthy = candidate_metrics["healthy"]
    checks = dict(
        original_retention=retain(
            {"olid": candidate_metrics}, {"olid": baseline_metrics}
        ),
        healthy_min_70_percent=healthy["correct"] / healthy["n"] >= 0.70,
        healthy_gain_min_10pp=healthy_gain["point"] >= 0.10,
        paired_family_gain_lower_positive=healthy_gain["bootstrap_two_sided_95"][0]
        > 0.0,
        disease_false_healthy_nonincrease=all(
            v["point"] <= 0.0 and v["bootstrap_one_sided_upper_95"] <= 0.02
            for v in safety.values()
            if v["n"] >= 50
        ),
        original_named_disease_unchanged=not changed_disease,
    )
    result = dict(
        passed=all(checks.values()),
        checks=checks,
        protocol_sha256=identity,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        paired_healthy_gain=healthy_gain,
        paired_disease_false_healthy=safety,
        changed_original_named_disease=changed_disease,
        healthy_wilson_95=wilson(healthy["correct"], healthy["n"]),
        now_opened=True,
        source_independence_consumed=True,
        ad_independent=False,
        mobile_scores_sha256=digest(predictions),
        native_predictions_sha256=digest(native_output),
        elapsed_seconds=time.monotonic() - started,
        cache_hits=cache.hits,
        new_inputs=cache.computed,
        physical_phone_verified=False,
        installed=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                checks=checks,
                healthy=healthy,
                healthy_gain=healthy_gain,
                baseline_false_healthy=baseline_metrics["all"]["false_healthy"],
                candidate_false_healthy=candidate_metrics["all"]["false_healthy"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
