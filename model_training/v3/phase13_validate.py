"""Fresh Apache-only graph replay on development and human-label internal data.

No independent AD files or restricted-model scores are read. Internal baseline
records contain only unchanged DINO/MobileNet/YOLO LiteRT predictions.
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

from app_eval import classifier_letterbox_input, canon, posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from permissive_encoder import digest
from permissive_scene import load as load_head, restore as restore_head
from phase13_context_model import build, restore
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics
from phase7_verifier import restore as restore_genus


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--model-root")
    ap.add_argument(
        "--head-family",
        choices=["siglip2", "dinov2_large", "hybrid_field"],
        default="siglip2",
    )
    ap.add_argument("--selection", default="fusion_selection")
    ap.add_argument("--healthy-directory")
    a = ap.parse_args()
    p = Path(a.campaign)
    root = Path(a.model_root) if a.model_root else p / "phase13_siglip"
    selected = root / a.selection / "candidate.json"
    c = read(selected)
    policy = c["policy"]
    out = root / (a.selection + "_validation")
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(619)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    directory = (
        Path(a.healthy_directory)
        if a.healthy_directory
        else Path(
            read(root / (a.head_family + "_feature_heads/ranked.json"))[0]["checkpoint"]
        ).parent
    )
    identity = c["input_identity"]
    assert digest(directory / "best.pt") == identity["head_sha256"]
    assert digest(directory / "result.json") == identity["calibration_sha256"]
    bundle = out / "pytorch_bundle.pt"
    protocol = dict(
        candidate_sha256=digest(selected),
        baseline_manifest_sha256=digest(p / "export_cnn_safe/model_manifest.json"),
        script_sha256=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
        restricted_teacher_used=False,
        cold_ad_used=False,
    )
    meta = out / "protocol.json"
    if meta.exists():
        assert (
            read(meta) == protocol
        ), "Validation identity changed; use a versioned directory"
    else:
        meta.write_text(json.dumps(protocol, indent=2))
    if c.get("union"):
        from phase14_union_model import build as build_union, restore as restore_union

        spec, model = build_union(p, c, root)
        restoring = restore_union
    elif c.get("fusion"):
        spec, model = build(p, c, root)
        restoring = restore
    else:
        spec, model = load_head(directory)
        restoring = restore_head
    torch.save(dict(spec, model=model.state_dict()), bundle)
    model = model.cuda().eval()
    fresh = restoring(bundle).cuda().eval()
    with torch.inference_mode():
        x = torch.rand(3, 224, 224, 3, device="cuda")
        error = float((model(x) - fresh(x)).abs().max())
    assert error == 0.0
    del model
    model = fresh
    species = restore_genus(p / "phase6_roi/species").cuda().eval()

    def infer(paths, destination):
        def prepare(path):
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
            return np.concatenate(
                [
                    classifier_letterbox_input(rgb, 224),
                    canon.classifier_input(rgb, 224),
                ],
                0,
            )

        h = np.zeros((len(paths), 2), np.float32)
        genus = np.zeros(len(paths), np.float32)
        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(paths), 32):
                part = paths[start : start + 32]
                x = torch.from_numpy(
                    np.concatenate(list(pool.map(prepare, part)), 0)
                ).cuda()
                if c.get("union"):
                    values = model(x[::2]).sigmoid().cpu().numpy()
                    h[start : start + len(part)] = values[:, [0, 2]]
                    genus[start : start + len(part)] = values[:, 1]
                    continue
                elif c.get("fusion"):
                    components, pair = model.components(x)
                    values = pair[:, 0].sigmoid().cpu().numpy().reshape(-1, 2)
                    # Selector's optional center agreement is the standalone
                    # verifier, while its full-frame primary score is fused.
                    values[:, 1] = components[1::2, 4].sigmoid().cpu().numpy()
                else:
                    values = model(x).sigmoid().cpu().numpy().reshape(-1, 2)
                h[start : start + len(part)] = values
                genus[start : start + len(part)] = species(x[::2]).ravel().cpu().numpy()
        assert np.isfinite(h).all() and np.isfinite(genus).all()
        np.savez(destination, healthy=h, genus=genus)
        return h, genus

    manifest = read(p / "export_cnn_safe/model_manifest.json")

    def route(row, h, genus):
        blocked = policy["named_leaf_veto"] and any(
            max(
                posterior(leaf["member_logits"], manifest)[j]
                for j in [0, 2, 3, 4, 5, 6, 7]
            )
            >= 0.7
            for leaf in row["leaves"]
        )
        positive = h[0] >= policy["healthy_min"] and genus >= policy["genus_min"]
        if c.get("union"):
            assert not policy["two_view_agreement"] and not policy["named_leaf_veto"]
            student_positive = (
                h[1] >= policy["union_student_healthy_min"]
                and genus >= policy["union_student_genus_min"]
            )
            probability = float(h[0] if positive else h[1])
            return route_context(
                row,
                row,
                manifest,
                bool(positive or student_positive),
                False,
                probability,
                False,
                False,
            )
        if policy["two_view_agreement"]:
            positive = positive and h[1] >= policy["healthy_min"]
        probability = float(h.min() if policy["two_view_agreement"] else h[0])
        return route_context(
            row, row, manifest, bool(positive), False, probability, blocked, blocked
        )

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
    assert (
        hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        == identity["paths_sha256"]
    )
    lookup = {path: i for i, path in enumerate(paths)}
    h, genus = infer(paths, out / "development_probabilities.npz")
    cached = np.load(
        root / ("selection" if c.get("fusion") else a.selection) / "probabilities.npz"
    )
    expected_h = cached["healthy"].astype(np.float64)
    if c.get("fusion"):
        legacy = np.load(
            p / "phase6_roi/scene_sft_d12_focal_selection/evaluation_logits.npy"
        )
        raw = np.clip(expected_h[:, 0], 1e-12, 1 - 1e-12)
        joined = np.concatenate([legacy, (np.log(raw) - np.log1p(-raw))[:, None]], 1)
        fit = c["fusion"]
        z = np.clip(joined, -10, 10) @ np.asarray(fit["weights"]) + fit["bias"]
        expected_h[:, 0] = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
    if c.get("union"):
        source = (
            root / "fusion_refined_selection_validation/development_probabilities.npz"
        )
        assert digest(source) == c["union"]["teacher_probabilities_sha256"]
        expected_h[:, 0] = np.load(source)["healthy"][:, 0]
        source = root / (
            "risk_" + c["union"]["student_mode"] + "_selection/probabilities.npz"
        )
        assert digest(source) == c["union"]["student_probabilities_sha256"]
        expected_h[:, 1] = np.load(source)["healthy"][:, 0]
    probability_error = float(np.max(np.abs(h - expected_h)))
    genus_error = float(np.max(np.abs(genus - cached["genus"])))
    assert (
        max(probability_error, genus_error) < 0.002
    ), "Executable graph differs from score cache"

    def key(row):
        v = row["app_replay"]
        return v["kind"], v["label"], v.get("scope")

    cases = sum(len(values) for values in base.values())
    verdict_matches = sum(
        key(route(r, h[lookup[r["path"]]], genus[lookup[r["path"]]]))
        == key(
            route(r, expected_h[lookup[r["path"]]], cached["genus"][lookup[r["path"]]])
        )
        for values in base.values()
        for r in values
    )
    assert verdict_matches == cases, "Fresh graph changed individual app verdicts"
    observed = grouped(
        {
            name: [
                route(r, h[lookup[r["path"]]], genus[lookup[r["path"]]]) for r in values
            ]
            for name, values in base.items()
        }
    )
    assert (
        observed == c["metrics"]
    ), "Fresh standalone graph changed selected app metrics"
    internal = p / "phase12_scene/union_app_validation/baseline_predictions.jsonl"
    records = [json.loads(line) for line in internal.read_text().splitlines()]
    admitted = {
        r["id"]: r
        for r in read(root / "admission_manifest.json")
        if r["split"] == "val"
    }
    assert set(admitted) == {r["id"] for r in records}
    for r in records:
        expected = admitted[r["id"]]
        assert (r["label"], r["group"], r["file_sha256"]) == (
            expected["label"],
            expected["group"],
            expected["file_sha256"],
        )
        assert digest(r["path"]) == r["file_sha256"]
    h, genus = infer([r["path"] for r in records], out / "internal_probabilities.npz")
    after = [route(r, h[i], genus[i]) for i, r in enumerate(records)]
    sources = sorted({r["source"] for r in records})
    before_metrics = {
        s: photo_metrics([r for r in records if r["source"] == s]) for s in sources
    }
    after_metrics = {
        s: photo_metrics([r for r in after if r["source"] == s]) for s in sources
    }
    healthy = photo_metrics(after)["healthy"]
    checks = dict(
        development_graph_matches=True,
        development_retained=retain(observed, grouped(base)),
        development_healthy_targets=c["field_healthy_passed"],
        internal_retained=retain(after_metrics, before_metrics),
        internal_healthy_target=healthy["correct"] / healthy["n"] >= 0.7,
    )
    result = dict(
        checks=checks,
        passed_preacceptance=all(checks.values()),
        candidate_sha256=digest(selected),
        bundle_sha256=digest(bundle),
        restore_max_abs_error=error,
        max_abs_probability_error=probability_error,
        max_abs_genus_probability_error=genus_error,
        development_verdict_cases=cases,
        development_verdict_matches=verdict_matches,
        old_development=observed,
        internal_baseline_sha256=digest(internal),
        internal_baseline=before_metrics,
        internal_candidate=after_metrics,
        internal_healthy=healthy,
        internal_photos=len(records),
        independent=False,
        cold_ad_used=False,
        restricted_teacher_used=False,
        native_context_runtime_implemented=False,
        retired_cohort_is_benchmark=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(checks=checks, Healthy=healthy, passed=result["passed_preacceptance"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
