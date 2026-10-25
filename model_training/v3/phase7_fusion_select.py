"""Replay monotone scene fusion on opened app cohorts before any acceptance."""

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

from app_eval import classifier_letterbox_input, posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from phase2_export import restore as restore_classifier
from phase2_select import retain
from phase5_scene_ablation import grouped
from phase7_feature_heads import BinaryHead
from phase7_verifier import restore as restore_verifier
from context_abstention import route_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inspect-candidate", type=int)
    ap.add_argument("--leaf-corroboration", action="store_true")
    ap.add_argument("--fusion-run", default="logit_fusion")
    ap.add_argument("--run-name")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / (
        a.run_name
        or ("fusion_leaf_selection" if a.leaf_corroboration else "fusion_selection")
    )
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    ev = json.loads((root / "eval_rows.json").read_text())
    manifest = json.loads(
        (root.parent / "original_reference/model_manifest.json").read_text()
    )
    base = {
        k: v["rows"]
        for k, v in app_evaluate(safe_logits(root), rows, ev, manifest, True).items()
    }
    photos = {
        k: v["rows"]
        for k, v in app_evaluate(
            np.load(root / "guard_selection/guarded_logits.npy"),
            rows,
            ev,
            manifest,
            True,
        ).items()
    }
    baseline = grouped(base)
    paths = sorted({r["path"] for values in photos.values() for r in values})
    lookup = {p: i for i, p in enumerate(paths)}
    fits = json.loads((root / a.fusion_run / "candidates.json").read_text())
    identity = dict(
        input_identity=fits[0]["input_identity"],
        paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
    )
    cache_dir = root / "fusion_selection" if a.fusion_run == "logit_fusion" else out
    cache = cache_dir / "evaluation_logits.npy"
    meta = cache_dir / "input_identity.json"
    if cache.exists():
        assert json.loads(meta.read_text()) == identity
        logits = np.load(cache)
    else:
        model = (
            restore_classifier(root / "export_initial/pytorch_bundle.pt").cuda().eval()
        )
        scene = restore_verifier(
            root / fits[0].get("scene_run", "scene_healthy")
        ).cuda()
        species = restore_verifier(root / "species").cuda()
        heads = []
        for kind in ["linear_adapted", "mlp_both"]:
            directory = root / "feature_heads/healthy_scene" / kind
            state = torch.load(
                directory / "best.pt", weights_only=True, map_location="cpu"
            )
            head = BinaryHead(kind).cuda().eval()
            head.load_state_dict(state["model"])
            cal = json.loads((directory / "result.json").read_text())["calibration"]
            heads.append((head, cal))
        logits = np.empty((len(paths), 4), np.float32)

        def prepare(path):
            return classifier_letterbox_input(
                np.asarray(Image.open(path).convert("RGB")), 224
            )

        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(paths), 128):
                part = paths[start : start + 128]
                x = torch.from_numpy(
                    np.concatenate(list(pool.map(prepare, part)), 0)
                ).cuda()
                *_, original, adapted = model.components(x, return_features=True)
                features = torch.cat([original, adapted], 1)
                for i, (head, cal) in enumerate(heads):
                    logits[start : start + len(part), i] = (
                        (head(features) * cal[0] + cal[1]).cpu().numpy()
                    )
                logits[start : start + len(part), 2] = (
                    scene.logits(x).ravel().cpu().numpy()
                )
                logits[start : start + len(part), 3] = (
                    species.logits(x).ravel().cpu().numpy()
                )
        np.save(cache, logits)
        meta.write_text(json.dumps(identity, indent=2))

    def named(row):
        return any(
            max(
                posterior(leaf["member_logits"], manifest)[j]
                for j in range(9)
                if j not in (1, 8)
            )
            >= 0.7
            for leaf in row["leaves"]
        )

    strong = {k: [named(r) for r in v] for k, v in photos.items()}
    base_strong = {k: [named(r) for r in v] for k, v in base.items()}
    species_p = 1 / (1 + np.exp(-np.clip(logits[:, 3], -80, 80)))
    rank = []
    leaf_support = np.zeros(len(paths), np.float32)
    if a.leaf_corroboration:
        leaf_candidate = json.loads(
            (root / "guard_selection/candidate.json").read_text()
        )
        center = np.load(
            Path(leaf_candidate["student_checkpoint"]).parent / "best_logits.npy"
        )
        full = np.load(root / "guard_selection/full_roi_components.npz")["student"]
        for i, r in enumerate(rows):
            if r["set"] not in ev or r.get("leaf", -1) < 0:
                continue
            score = min(
                float(posterior([center[i]], manifest)[1]),
                float(posterior([full[i]], manifest)[1]),
            )
            j = lookup[r["path"]]
            leaf_support[j] = max(leaf_support[j], score)
        np.save(out / "leaf_support.npy", leaf_support)
    # Strong-contradiction abstention also applies to legacy Healthy calls.
    grid = itertools.product(
        fits,
        (
            [0.9, 0.92, 0.95]
            if a.leaf_corroboration
            else [0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
        ),
        [0.5, 0.8] if a.leaf_corroboration else [0.5, 0.65, 0.8, 0.9, 0.95],
        (
            [0.001, 0.05, 0.1]
            if a.leaf_corroboration
            else [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        ),
        [0.5, 0.6, 0.7] if a.leaf_corroboration else [0.0],
    )
    if a.inspect_candidate is not None:
        chosen = next(
            c
            for c in json.loads((out / "ranked.json").read_text())
            if c["id"] == a.inspect_candidate
        )
        grid = [
            (
                chosen["fusion"],
                chosen["healthy_min"],
                chosen["species_min"],
                chosen["negative_healthy_max"],
                chosen.get("leaf_healthy_min", 0.0),
            )
        ]
    for fit, pmin, tmin, negative_max, leaf_min in grid:
        z = (
            np.clip(logits, -fit["input_clip"], fit["input_clip"])
            @ np.array(fit["weights"])
            + fit["bias"]
        )
        healthy_p = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        positive = (
            (healthy_p >= pmin) & (species_p >= tmin) & (leaf_support >= leaf_min)
        )
        revised = {}
        for name, values in photos.items():
            revised[name] = []
            for n, item in enumerate(values):
                i = lookup[item["path"]]
                revised[name].append(
                    route_context(
                        item,
                        base[name][n],
                        manifest,
                        bool(positive[i]),
                        bool(negative_max and healthy_p[i] <= negative_max),
                        float(healthy_p[i]),
                        strong[name][n],
                        base_strong[name][n],
                    )
                )
        if a.inspect_candidate is not None:
            audit = []
            for name, values in revised.items():
                for n, row in enumerate(values):
                    old = base[name][n]
                    if (
                        row["app_replay"]["kind"] == old["app_replay"]["kind"]
                        and row["app_replay"]["label"] == old["app_replay"]["label"]
                    ):
                        continue
                    i = lookup[row["path"]]
                    audit.append(
                        dict(
                            cohort=name,
                            path=row["path"],
                            source=row["source"],
                            label=row["label"],
                            scope=row.get("scope"),
                            original=old["app_replay"],
                            candidate=row["app_replay"],
                            healthy_p=float(healthy_p[i]),
                            species_p=float(species_p[i]),
                            logits=logits[i].tolist(),
                        )
                    )
            (out / f"inspect_{a.inspect_candidate}.json").write_text(
                json.dumps(audit, indent=2)
            )
            return
        observed = grouped(revised)
        new = observed["phase5_selection"]
        field_pass = all(
            new["by_source"][source]["healthy_recall"] >= 0.7
            for source in ("tom2024", "swtt5wy2pr")
        )
        score = float(
            np.mean(
                [
                    (m["healthy"]["correct_rate"] or 0)
                    + (m["supported_disease"]["correct_rate"] or 0)
                    - 3 * (m["all"]["false_healthy_rate"] or 0)
                    - 2 * (m["unsupported_tomato"]["false_known_rate"] or 0)
                    for m in observed.values()
                ]
            )
        )
        record = dict(
            id=len(rank),
            fusion=fit,
            healthy_min=pmin,
            species_min=tmin,
            negative_healthy_max=negative_max,
            leaf_healthy_min=leaf_min,
            metrics=observed,
            retained=retain(observed, baseline),
            field_healthy_passed=field_pass,
            score=score,
            leaf_candidate=json.loads(
                (root / "guard_selection/candidate.json").read_text()
            ),
        )
        rank.append(record)
        print(
            json.dumps(
                dict(
                    id=record["id"],
                    retained=record["retained"],
                    field_passed=field_pass,
                    H=new["healthy"]["correct"],
                    FH=new["all"]["false_healthy"],
                    field_H=new["by_source"]["tom2024"]["healthy_recall"],
                )
            ),
            flush=True,
        )
    rank.sort(
        key=lambda c: (
            c["retained"] and c["field_healthy_passed"],
            c["retained"],
            c["score"],
        ),
        reverse=True,
    )
    (out / "ranked.json").write_text(json.dumps(rank, indent=2))
    (out / "candidate.json").write_text(json.dumps(rank[0], indent=2))
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print(
        json.dumps(
            dict(
                best=rank[0]["id"],
                retained=rank[0]["retained"],
                field_passed=rank[0]["field_healthy_passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
