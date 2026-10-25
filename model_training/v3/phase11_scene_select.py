"""Opened-only image-scoped verifier ablation; no fresh AD or promotion."""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import itertools
import json

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import canon, classifier_letterbox_input, posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from phase2_select import retain
from phase5_scene_ablation import grouped
from phase7_verifier import restore as restore_genus


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument(
        "--head-family", choices=["mobileclip", "siglip2"], default="mobileclip"
    )
    ap.add_argument(
        "--healthy-directory", help="Use a separately trained image verifier"
    )
    ap.add_argument(
        "--model-root", help="Use a separately versioned image-scoped model workspace"
    )
    ap.add_argument("--fusion-run")
    ap.add_argument("--out-name", default="selection")
    ap.add_argument("--healthy-thresholds", default="0.6,0.7,0.8,0.9,0.95,0.97,0.99")
    ap.add_argument(
        "--probabilities-from", help="Reuse identity-checked whole-image probabilities"
    )
    a = ap.parse_args()
    thresholds = [float(x) for x in a.healthy_thresholds.split(",")]
    assert thresholds and all(0.0 < x < 1.0 for x in thresholds)
    p = Path(a.campaign)
    root = Path(a.model_root) if a.model_root else p / "phase11_scene"
    leaf = p / "phase10_leaf"
    out = root / a.out_name
    out.mkdir(exist_ok=True)
    if a.healthy_directory:
        directory = Path(a.healthy_directory)
    else:
        fits = json.loads(
            (root / (a.head_family + "_feature_heads") / "ranked.json").read_text()
        )
        directory = Path(fits[0]["checkpoint"]).parent
    rows = json.loads((leaf / "cnn_inputs/rows.json").read_text())
    ev = json.loads((leaf / "eval_rows.json").read_text())
    manifest = json.loads((p / "original_reference/model_manifest.json").read_text())
    base = {
        k: v["rows"]
        for k, v in app_evaluate(safe_logits(leaf), rows, ev, manifest, True).items()
    }
    baseline = grouped(base)
    paths = sorted({r["path"] for values in base.values() for r in values})
    lookup = {path: i for i, path in enumerate(paths)}
    identity = dict(
        head_sha256=digest(directory / "best.pt"),
        calibration_sha256=digest(directory / "result.json"),
        encoder_sha256=digest(
            directory.parent / "encoder.pt"
            if (directory.parent / "encoder.pt").exists()
            else directory / "best.pt"
        ),
        paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
        rows_sha256=digest(leaf / "cnn_inputs/rows.json"),
        evaluation_rows_sha256=digest(leaf / "eval_rows.json"),
        genus_checkpoint_sha256=digest(p / "phase6_roi/species/best.pt"),
        genus_calibration_sha256=digest(p / "phase6_roi/species/result.json"),
        selection_rule="Top training-validation score selected before app replay",
        fresh_ad_used=False,
    )
    base_identity = dict(identity)
    if a.fusion_run:
        identity["fusion_candidates_sha256"] = digest(
            root / a.fusion_run / "candidates.json"
        )
    meta = out / "input_identity.json"
    if meta.exists():
        assert json.loads(meta.read_text()) == identity
    else:
        meta.write_text(json.dumps(identity, indent=2))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    cache_root = (
        Path(a.probabilities_from)
        if a.probabilities_from
        else (root / "selection" if a.fusion_run else out)
    )
    cache = cache_root / "probabilities.npz"
    if a.fusion_run or a.probabilities_from:
        assert (
            json.loads((cache_root / "input_identity.json").read_text())
            == base_identity
        )
        assert cache.exists()
    if cache.exists():
        z = np.load(cache)
        healthy, genus = z["healthy"], z["genus"]
    else:
        if a.head_family == "siglip2":
            from permissive_scene import load
        else:
            from mobileclip_scene import load
        _, model = load(directory)
        model = model.cuda().eval()
        species = restore_genus(p / "phase6_roi/species").cuda().eval()
        healthy = np.zeros((len(paths), 2), np.float32)
        genus = np.zeros(len(paths), np.float32)

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

        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(paths), 128):
                part = paths[start : start + 128]
                x = torch.from_numpy(
                    np.concatenate(list(pool.map(prepare, part)), 0)
                ).cuda()
                healthy[start : start + len(part)] = (
                    model(x).sigmoid().cpu().numpy().reshape(-1, 2)
                )
                genus[start : start + len(part)] = species(x[::2]).ravel().cpu().numpy()
        np.savez(cache, healthy=healthy, genus=genus)
    blocks = {}
    for name, values in base.items():
        blocks[name] = [
            any(
                max(
                    posterior(l["member_logits"], manifest)[j]
                    for j in [0, 2, 3, 4, 5, 6, 7]
                )
                >= 0.7
                for l in r["leaves"]
            )
            for r in values
        ]
    variants = [(None, healthy)]
    if a.fusion_run:
        fusion_fits = json.loads((root / a.fusion_run / "candidates.json").read_text())
        old = p / "phase6_roi/scene_sft_d12_focal_selection"
        old_meta = json.loads((old / "input_identity.json").read_text())
        assert old_meta["paths_sha256"] == identity["paths_sha256"]
        assert (
            old_meta["input_identity"]
            == fusion_fits[0]["input_identity"]["legacy_input_identity"]
        )
        legacy_logits = np.load(old / "evaluation_logits.npy")
        h = np.clip(healthy[:, 0].astype(np.float64), 1e-12, 1 - 1e-12)
        mobile_logits = np.log(h) - np.log1p(-h)
        x = np.clip(np.concatenate([legacy_logits, mobile_logits[:, None]], 1), -10, 10)
        variants = []
        for fit in fusion_fits:
            z = x @ np.asarray(fit["weights"]) + fit["bias"]
            fused = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
            variants.append((fit, np.stack([fused, healthy[:, 1]], 1)))
    ranks = []
    for variant, hmin, gmin, agreement, veto in itertools.product(
        range(len(variants)),
        thresholds,
        [0.0, 0.8, 0.9],
        [False, True],
        [False, True],
    ):
        fit, active = variants[variant]
        positive = (active[:, 0] >= hmin) & (genus >= gmin)
        if agreement:
            positive &= active[:, 1] >= hmin
        revised = {}
        for name, values in base.items():
            revised[name] = []
            for k, r in enumerate(values):
                i = lookup[r["path"]]
                probability = float(active[i].min() if agreement else active[i, 0])
                blocked = veto and blocks[name][k]
                revised[name].append(
                    route_context(
                        r,
                        r,
                        manifest,
                        bool(positive[i]),
                        False,
                        probability,
                        blocked,
                        blocked,
                    )
                )
        metrics = grouped(revised)
        m = metrics["phase5_selection"]
        passed = all(
            m["by_source"][source]["healthy_recall"] >= 0.7
            for source in ["tom2024", "swtt5wy2pr"]
        )
        score = float(
            np.mean(
                [
                    (v["healthy"]["correct_rate"] or 0)
                    + (v["supported_disease"]["correct_rate"] or 0)
                    - 3 * (v["all"]["false_healthy_rate"] or 0)
                    - 2 * (v["unsupported_tomato"]["false_known_rate"] or 0)
                    for v in metrics.values()
                ]
            )
        )
        c = dict(
            id=len(ranks),
            policy=dict(
                healthy_min=hmin,
                genus_min=gmin,
                two_view_agreement=agreement,
                named_leaf_veto=veto,
            ),
            metrics=metrics,
            retained=retain(metrics, baseline),
            field_healthy_passed=passed,
            score=score,
            input_identity=identity,
            native_context_runtime_implemented=False,
            fresh_ad_used=False,
            scope="Image-level Healthy primary view; no fake leaf geometry",
        )
        if fit is not None:
            c["fusion"] = fit
            c["policy"]["center_agreement_model"] = (
                a.head_family + "-only; primary score is fused"
            )
            c["evaluation_note"] = (
                "Mobile logit reconstructed from cached float32 probability and clipped to +/-10; exact graph replay remains mandatory before acceptance."
            )
        ranks.append(c)
        print(
            json.dumps(
                dict(
                    id=c["id"],
                    retained=c["retained"],
                    field_passed=passed,
                    Healthy=m["healthy"]["correct"],
                    false_Healthy=m["all"]["false_healthy"],
                )
            ),
            flush=True,
        )
    ranks.sort(
        key=lambda c: (
            c["retained"] and c["field_healthy_passed"],
            c["retained"],
            c["score"],
        ),
        reverse=True,
    )
    for name, value in [
        ("ranked", ranks),
        ("candidate", ranks[0]),
        ("baseline", baseline),
    ]:
        (out / (name + ".json")).write_text(json.dumps(value, indent=2))
    print(
        json.dumps(
            dict(
                best=ranks[0]["id"],
                retained=ranks[0]["retained"],
                field_passed=ranks[0]["field_healthy_passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
