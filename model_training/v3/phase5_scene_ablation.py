"""Research-only image-context ablation. Does not export or modify app assets.

Tests whether whole-frame Healthy agreement resolves averaging of inconsistent
leaf crops. Requires a detected leaf, preserves named diagnoses and uses only
opened development/regression photos. Never reads the prospective reservation.
"""

from pathlib import Path
import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import time

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import canon, classifier_letterbox_input, posterior, replay, summarize
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from phase2_accept import metrics as file_metrics
from phase2_export import restore
from phase2_policy import DomainPolicy
from phase2_select import retain
from phase5_guard_select import keys
from train import LABELS


def photo_metrics(rows):
    result = summarize(rows)
    result["disease_classes"] = {}
    for name in LABELS:
        if name in ("Healthy", "Other"):
            continue
        subset = [
            r
            for r in rows
            if r["label"] == name
            and r.get("analysis_eligible", True)
            and "error" not in r
        ]
        result["disease_classes"][name] = dict(
            n=len(subset),
            correct=sum(
                r["app_replay"]["kind"] == "DISEASE"
                and r["app_replay"]["label"] == name
                for r in subset
            ),
        )
    return result


def grouped(photos):
    selection = photos["selection_app"]
    groups = {k: v for k, v in photos.items() if k != "selection_app"}
    groups["original"] = [
        r for r in selection if r["source"] not in ("tlid", "pakistan")
    ]
    groups["expanded"] = [r for r in selection if r["source"] in ("tlid", "pakistan")]
    return {k: photo_metrics(v) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-candidate", type=int)
    ap.add_argument("--guard-directory", help="Defaults to root/guard_selection")
    ap.add_argument("--scene-on-no-leaves", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    rows_path = root / "cnn_inputs/rows.json"
    rows = json.loads(rows_path.read_text())
    ev = json.loads((root / "eval_rows.json").read_text())
    manifest = json.loads(
        (root.parent / "original_reference/model_manifest.json").read_text()
    )
    base = app_evaluate(safe_logits(root), rows, ev, manifest, True)
    guard_dir = (
        Path(a.guard_directory) if a.guard_directory else root / "guard_selection"
    )
    leaf_logits = np.load(guard_dir / "guarded_logits.npy")
    frozen = json.loads((guard_dir / "candidate.json").read_text())
    if a.leaf_candidate is not None:
        frozen = next(
            c
            for c in json.loads((guard_dir / "ranked.json").read_text())
            if c["id"] == a.leaf_candidate
        )
        policy = DomainPolicy(**frozen["policy"])
        center = [
            safe_logits(root),
            np.load(Path(frozen["student_checkpoint"]).parent / "best_logits.npy"),
            np.load(Path(frozen["cnn_checkpoint"]).parent / "all_logits.npy"),
        ]
        cache = np.load(guard_dir / "full_roi_components.npz")
        full = [cache[k] for k in ("reference", "student", "cnn")]
        with torch.inference_mode():
            leaf_logits = policy(*[torch.from_numpy(z) for z in center]).numpy()
            alternative = policy(*[torch.from_numpy(z) for z in full]).numpy()
        old, _ = keys(center[0])
        new, positive = keys(leaf_logits)
        alt, _ = keys(alternative)
        fallback = (
            np.array([r["set"] in ev for r in rows])
            & (positive | ((new // 9 == 3) & frozen.get("confirm_unknown", False)))
            & (new != old)
            & (new != alt)
        )
        leaf_logits[fallback] = center[0][fallback]
    candidate = app_evaluate(leaf_logits, rows, ev, manifest, True)
    baseline = grouped({k: v["rows"] for k, v in base.items()})
    photos = {k: v["rows"] for k, v in candidate.items()}
    paths = sorted({r["path"] for values in photos.values() for r in values})
    lookup = {p: i for i, p in enumerate(paths)}
    calibration = np.asarray(frozen["policy"]["cnn_calibration"], np.float32)
    identity = dict(
        bundle_sha256=hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest(),
        rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
    )
    cache = out / "scene_components.npz"
    identity_path = out / "components_identity.json"
    started = time.time()
    if cache.exists():
        assert json.loads(identity_path.read_text()) == identity
        saved = np.load(cache)
        student = saved["student"]
        cnn = saved["cnn"]
    else:
        model = restore(a.bundle).cuda().eval()
        student = np.zeros((len(paths), 2, 9), np.float32)
        cnn = np.zeros((len(paths), 2, 2), np.float32)

        def prepare(path):
            image = np.asarray(Image.open(path).convert("RGB"))
            return np.concatenate(
                [
                    canon.classifier_input(image, 224),
                    classifier_letterbox_input(image, 224),
                ],
                0,
            )

        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(paths), 32):
                part = paths[start : start + 32]
                pixels = np.concatenate(list(pool.map(prepare, part)), 0)
                _, z, c = model.components(torch.from_numpy(pixels).cuda())
                student[start : start + len(part)] = z.cpu().numpy().reshape(-1, 2, 9)
                cnn[start : start + len(part)] = c.cpu().numpy().reshape(-1, 2, 2)
                if start % 256 == 0:
                    print(
                        json.dumps(
                            dict(
                                scene_photos=start,
                                total=len(paths),
                                seconds=time.time() - started,
                            )
                        ),
                        flush=True,
                    )
        np.savez(cache, student=student, cnn=cnn)
        identity_path.write_text(json.dumps(identity, indent=2))
    # Exact app posterior, including the softmax/log floor and Other bias.
    probabilities = np.stack(
        [posterior([z], manifest) for z in student.reshape(-1, 9)]
    ).reshape(-1, 2, 9)
    calibrated = 1 / (
        1 + np.exp(-np.clip(cnn * calibration[:, 0] + calibration[:, 1], -80, 80))
    )
    rank = []
    for smin, hmin, kmin, veto in itertools.product(
        [0.9, 0.95, 0.99], [0.9, 0.95, 0.99], [0.5, 0.7], [False, True]
    ):
        strong = (
            (probabilities[:, :, 1] >= smin).all(1)
            & (calibrated[:, :, 0] >= hmin).all(1)
            & (calibrated[:, :, 1] >= kmin).all(1)
        )
        weak = (probabilities[:, 1, 1] >= 0.6) & (calibrated[:, 1, 0] >= 0.5)
        revised = copy.deepcopy(photos)
        changed = 0
        for group, values in revised.items():
            for n, row in enumerate(values):
                original = base[group]["rows"][n]
                assert row["id"] == original["id"]
                i = lookup[row["path"]]
                if (not row["leaves"] and not a.scene_on_no_leaves) or (
                    row["contrast"] < 0.015 and row["sharpness"] < 0.012
                ):
                    continue
                if (
                    veto
                    and row["app_replay"]["kind"] == "HEALTHY"
                    and original["app_replay"]["kind"] != "HEALTHY"
                    and not weak[i]
                ):
                    row["leaves"] = copy.deepcopy(original["leaves"])
                    row["app_replay"] = original["app_replay"]
                    changed += 1
                named = False
                for leaf in row["leaves"]:
                    p = posterior(leaf["member_logits"], manifest)
                    named |= bool(max(p[j] for j in range(9) if j not in (1, 8)) >= 0.7)
                if (
                    row["app_replay"]["kind"] in ("HEALTHY", "DISEASE")
                    or named
                    or not strong[i]
                ):
                    continue
                probs = probabilities[i, 1]
                row["whole_frame_evidence"] = dict(
                    probabilities=probs.tolist(),
                    scope="whole_frame",
                    healthy_agreement=True,
                )
                row["app_replay"] = dict(
                    kind="HEALTHY",
                    label="Healthy",
                    confidence=float(probs[1]),
                    probabilities=probs.tolist(),
                    margin=float(probs[1] - np.max(np.delete(probs, 1))),
                    frames=5,
                    leaves=0,
                    partial=False,
                    scope="whole_frame",
                    detected_leaves=len(row["leaves"]),
                )
                changed += 1
        observed = grouped(revised)
        score = float(
            np.mean(
                [
                    (m["healthy"]["correct_rate"] or 0)
                    + (m["supported_disease"]["correct_rate"] or 0)
                    - 3 * (m["all"]["false_healthy_rate"] or 0)
                    for m in observed.values()
                ]
            )
        )
        record = dict(
            id=len(rank),
            leaf_candidate=frozen,
            student_min=smin,
            healthy_min=hmin,
            known_min=kmin,
            veto=veto,
            metrics=observed,
            retained=retain(observed, baseline),
            score=score,
            changed_photos=changed,
            scene_on_no_leaves=a.scene_on_no_leaves,
            field_healthy_passed=all(
                observed["phase5_selection"]["by_source"][source]["healthy_recall"]
                >= 0.7
                for source in ("tom2024", "swtt5wy2pr")
            ),
        )
        rank.append(record)
        print(
            json.dumps(
                dict(
                    id=record["id"],
                    retained=record["retained"],
                    score=score,
                    H=observed["phase5_selection"]["healthy"]["correct"],
                    FH=observed["phase5_selection"]["all"]["false_healthy"],
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
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    (out / "ranked.json").write_text(json.dumps(rank, indent=2))
    (out / "candidate.json").write_text(json.dumps(rank[0], indent=2))
    print(
        json.dumps(
            dict(
                best=rank[0]["id"],
                passed=rank[0]["retained"],
                seconds=time.time() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
