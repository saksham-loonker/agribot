"""Verify the exact graph, then replay the consumed 246-photo regression.

This never opens the cold Abu Dhabi source and never authorizes installation.
The original disease classifier/detector outputs are immutable baseline records.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input, posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics
from phase8_accept import wilson
from phase11_context_model import build, restore


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists():
        assert read(path) == value, ("Frozen identity changed", str(path))
    else:
        path.write_text(json.dumps(value, indent=2))


def route(row, manifest, policy, h, genus):
    blocked = policy["named_leaf_veto"] and any(
        max(posterior(l["member_logits"], manifest)[j] for j in [0, 2, 3, 4, 5, 6, 7])
        >= 0.7
        for l in row["leaves"]
    )
    return route_context(
        row,
        row,
        manifest,
        bool(h >= policy["healthy_min"] and genus >= policy["genus_min"]),
        False,
        float(h),
        blocked,
        blocked,
    )


def infer(model, paths, directory):
    directory.mkdir(exist_ok=True)
    meta = directory / "paths_identity.json"
    immutable(
        meta,
        dict(
            paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
            count=len(paths),
        ),
    )
    z_path, pair_path = directory / "logits.npy", directory / "outputs.npy"
    progress = directory / "progress.json"
    done = read(progress)["complete"] if progress.exists() else 0
    arrays = []
    for path, width in [(z_path, 5), (pair_path, 2)]:
        arr = np.lib.format.open_memmap(
            path,
            mode="r+" if path.exists() else "w+",
            dtype=np.float32,
            shape=(len(paths), width),
        )
        assert arr.shape == (len(paths), width) and arr.dtype == np.float32
        arrays.append(arr)
    logits, outputs = arrays

    def prepare(path):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return classifier_letterbox_input(rgb, 224)

    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(done, len(paths), 32):
            part = paths[start : start + 32]
            x = torch.from_numpy(
                np.concatenate(list(pool.map(prepare, part)), 0)
            ).cuda()
            z, pair = model.components(x)
            assert torch.isfinite(z).all() and torch.isfinite(pair).all()
            logits[start : start + len(part)] = z.cpu().numpy()
            outputs[start : start + len(part)] = pair.cpu().numpy()
            logits.flush()
            outputs.flush()
            temp = progress.with_suffix(".tmp")
            temp.write_text(json.dumps(dict(complete=start + len(part))))
            temp.replace(progress)
            if start % 512 == 0:
                print(
                    json.dumps(
                        dict(
                            stage=directory.name,
                            complete=start + len(part),
                            total=len(paths),
                        )
                    ),
                    flush=True,
                )
    return np.array(logits), np.array(outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase11_scene"
    legacy = p / "phase6_roi"
    out = root / "prototype_verification"
    out.mkdir(exist_ok=True)
    selected = root / "fusion_selection/candidate.json"
    candidate = read(selected)
    assert candidate["retained"] and candidate["field_healthy_passed"]
    policy = candidate["policy"]
    assert not policy[
        "two_view_agreement"
    ], "Implement exact second-view graph routing before using this policy"
    fit = candidate["fusion"]
    identity = fit["input_identity"]
    old = identity["legacy_input_identity"]
    checks = [
        (legacy / "export_initial/pytorch_bundle.pt", old["feature_model_sha256"]),
        (legacy / "scene_sft_d12_focal/best.pt", old["scene_cnn_sha256"]),
        (
            legacy / "scene_sft_d12_focal/result.json",
            old["scene_cnn_calibration_sha256"],
        ),
        (legacy / "species/best.pt", old["species_cnn_sha256"]),
        (legacy / "species/result.json", old["species_cnn_calibration_sha256"]),
    ]
    for i, kind in enumerate(["linear_adapted", "mlp_both"]):
        checks.extend(
            [
                (
                    legacy / "feature_heads/healthy_scene" / kind / "best.pt",
                    old["head_hashes"][i],
                ),
                (
                    legacy / "feature_heads/healthy_scene" / kind / "result.json",
                    old["head_calibration_hashes"][i],
                ),
            ]
        )
    mobile = read(root / "mobileclip_feature_heads/ranked.json")[0]
    directory = Path(mobile["checkpoint"]).parent
    checks.extend(
        [
            (directory / "best.pt", identity["mobile_head_sha256"]),
            (directory / "result.json", identity["mobile_calibration_sha256"]),
            (directory.parent / "encoder.pt", identity["mobile_encoder_sha256"]),
        ]
    )
    for path, sha in checks:
        assert digest(path) == sha, (str(path), "Model identity changed")
    protocol = dict(
        candidate_sha256=digest(selected),
        candidate_id=candidate["id"],
        input_identity=identity,
        scope="Exact graph then consumed regression only; no cold AD diagnostic inference",
        baseline_classifier_sha256="d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39",
        model_code_sha256=digest(Path(__file__).with_name("phase11_context_model.py")),
        script_sha256=digest(__file__),
        routing_sha256=digest(Path(__file__).with_name("context_abstention.py")),
        app_eval_sha256=digest(Path(__file__).with_name("app_eval.py")),
    )
    immutable(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(411)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    bundle = out / "pytorch_bundle.pt"
    if not bundle.exists():
        spec, model = build(p, candidate)
        torch.save(dict(spec, model=model.state_dict()), bundle)
        model = model.cuda().eval()
        restored = restore(bundle).cuda().eval()
        with torch.inference_mode():
            x = (
                torch.rand(1, 224, 224, 3, device="cuda")
                - torch.tensor([0.485, 0.456, 0.406], device="cuda")
            ) / torch.tensor([0.229, 0.224, 0.225], device="cuda")
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        del model, restored
    immutable(out / "bundle_identity.json", dict(sha256=digest(bundle)))
    model = restore(bundle).cuda().eval()
    leaf = p / "phase10_leaf"
    rows = read(leaf / "cnn_inputs/rows.json")
    ev = read(leaf / "eval_rows.json")
    manifest = read(p / "original_reference/model_manifest.json")
    base = {
        k: v["rows"]
        for k, v in app_evaluate(safe_logits(leaf), rows, ev, manifest, True).items()
    }
    baseline = grouped(base)
    paths = sorted({r["path"] for values in base.values() for r in values})
    lookup = {path: i for i, path in enumerate(paths)}
    assert (
        hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        == candidate["input_identity"]["paths_sha256"]
    )
    z, pair = infer(model, paths, out / "development")
    old_z = np.load(legacy / "scene_sft_d12_focal_selection/evaluation_logits.npy")
    mobile_p = np.load(root / "selection/probabilities.npz")["healthy"][:, 0].astype(
        np.float64
    )
    mobile_p = np.clip(mobile_p, 1e-12, 1 - 1e-12)
    cached = np.concatenate(
        [old_z, (np.log(mobile_p) - np.log1p(-mobile_p))[:, None]], 1
    )
    error = float(np.max(np.abs(np.clip(z, -10, 10) - np.clip(cached, -10, 10))))
    cached_h = 1 / (
        1
        + np.exp(
            -np.clip(
                np.clip(cached, -10, 10) @ np.array(fit["weights"]) + fit["bias"],
                -80,
                80,
            )
        )
    )
    h = 1 / (1 + np.exp(-np.clip(pair[:, 0].astype(np.float64), -80, 80)))
    genus = 1 / (1 + np.exp(-np.clip(pair[:, 1].astype(np.float64), -80, 80)))
    expected_genus = 1 / (1 + np.exp(-np.clip(cached[:, 3], -80, 80)))
    revised = {}
    matches = []
    for name, values in base.items():
        revised[name] = []
        for r in values:
            i = lookup[r["path"]]
            actual = route(r, manifest, policy, h[i], genus[i])
            expected = route(r, manifest, policy, cached_h[i], expected_genus[i])
            revised[name].append(actual)

            def key(row):
                v = row["app_replay"]
                return v["kind"], v["label"], v.get("scope")

            matches.append(key(actual) == key(expected))
    observed = grouped(revised)
    m = observed["phase5_selection"]
    development = dict(
        passed=error < 0.002
        and all(matches)
        and retain(observed, baseline)
        and all(
            m["by_source"][s]["healthy_recall"] >= 0.7
            for s in ["tom2024", "swtt5wy2pr"]
        ),
        cases=len(matches),
        max_abs_clipped_context_logit_error=error,
        verdict_matches=sum(matches),
        retained=retain(observed, baseline),
        metrics=observed,
        cold_ad_used=False,
    )
    (out / "development_parity.json").write_text(json.dumps(development, indent=2))
    print(
        json.dumps({k: v for k, v in development.items() if k != "metrics"}), flush=True
    )
    assert development[
        "passed"
    ], "Exact executable graph failed development; do not open a fresh source"
    records = read(legacy / "reserved_manifest.json")
    assert (
        len(records) == 246
        and digest(legacy / "reserved_manifest.json")
        == "ca91cf80b05eab1c733a54746542e8fe37bee503c7c0fe03a97022bcdd0f5dd1"
    )
    previous = legacy / "phase8_acceptance/predictions.jsonl"
    before_by_id = {
        r["id"]: r["baseline"]
        for r in [json.loads(line) for line in previous.read_text().splitlines()]
    }
    assert len(before_by_id) == 246
    immutable(
        out / "opened_regression_identity.json",
        dict(
            manifest_sha256=digest(legacy / "reserved_manifest.json"),
            baseline_predictions_sha256=digest(previous),
            protocol_sha256=digest(out / "protocol.json"),
            scope="Already consumed phase-eight cohort; never fresh acceptance",
        ),
    )
    for r in records:
        assert digest(r["path"]) == r["file_sha256"]
    _, pair = infer(model, [r["path"] for r in records], out / "opened_regression")
    h = 1 / (1 + np.exp(-np.clip(pair[:, 0].astype(np.float64), -80, 80)))
    genus = 1 / (1 + np.exp(-np.clip(pair[:, 1].astype(np.float64), -80, 80)))
    before = [before_by_id[r["id"]] for r in records]
    after = [route(r, manifest, policy, h[i], genus[i]) for i, r in enumerate(before)]
    with (out / "opened_regression_predictions.jsonl").open("w") as f:
        for i, r in enumerate(records):
            f.write(
                json.dumps(
                    dict(
                        id=r["id"],
                        baseline=before[i],
                        candidate=after[i],
                        healthy_probability=float(h[i]),
                        genus_probability=float(genus[i]),
                    )
                )
                + "\n"
            )
    sources = sorted({r["source"] for r in records})
    b = {s: photo_metrics([r for r in before if r["source"] == s]) for s in sources}
    c = {s: photo_metrics([r for r in after if r["source"] == s]) for s in sources}
    healthy = [i for i, r in enumerate(records) if r["label"] == "Healthy"]
    gains = np.array(
        [
            int(after[i]["app_replay"]["kind"] == "HEALTHY")
            - int(before[i]["app_replay"]["kind"] == "HEALTHY")
            for i in healthy
        ]
    )
    groups = sorted({records[i]["group"] for i in healthy})
    group_values = [
        gains[[j for j, i in enumerate(healthy) if records[i]["group"] == g]]
        for g in groups
    ]
    rng = np.random.default_rng(239)
    boot = [
        float(
            np.concatenate(
                [group_values[i] for i in rng.integers(0, len(groups), len(groups))]
            ).mean()
        )
        for _ in range(5000)
    ]
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    gates = dict(
        retained_per_source=retain(c, b),
        healthy_min_70_percent=all(
            m["healthy"]["correct"] / m["healthy"]["n"] >= 0.7
            for m in c.values()
            if m["healthy"]["n"]
        ),
        healthy_gain_min_5pp=float(gains.mean()) >= 0.05,
        paired_group_gain_lower_positive=ci[0] > 0,
    )
    result = dict(
        checks=gates,
        passed_opened_regression=all(gates.values()),
        baseline=b,
        candidate=c,
        paired_healthy_gain=float(gains.mean()),
        paired_capture_group_bootstrap_95=ci,
        healthy_wilson_95={
            s: wilson(m["healthy"]["correct"], m["healthy"]["n"]) for s, m in c.items()
        },
        fresh_ad_used=False,
        promotable=False,
        next_step="If passed: freeze an independent protocol and verify exported bytes/native runtime. No installation is authorized by this research result.",
    )
    (out / "opened_regression_result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passed=result["passed_opened_regression"],
                checks=gates,
                counts={
                    s: dict(
                        Healthy=m["healthy"]["correct"],
                        Healthy_n=m["healthy"]["n"],
                        false_Healthy=m["all"]["false_healthy"],
                    )
                    for s, m in c.items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
