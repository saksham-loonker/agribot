"""Validate plant-view Healthy evidence with separate species and scene tasks.

Only opened development/regression photographs are read. This is a research
policy selection, never a promotion or an independent acceptance result.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import time

import numpy as np
from PIL import Image
import torch

from app_eval import classifier_letterbox_input, posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from phase2_select import retain
from phase5_scene_ablation import grouped
from phase7_verifier import restore


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "context_selection"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    ev = json.loads((root / "eval_rows.json").read_text())
    manifest = json.loads(
        (root.parent / "original_reference/model_manifest.json").read_text()
    )
    original = app_evaluate(safe_logits(root), rows, ev, manifest, True)
    adapted = app_evaluate(
        np.load(root / "guard_selection/guarded_logits.npy"), rows, ev, manifest, True
    )
    base = {k: v["rows"] for k, v in original.items()}
    photos = {k: v["rows"] for k, v in adapted.items()}
    baseline = grouped(base)
    paths = sorted({r["path"] for values in photos.values() for r in values})
    lookup = {p: i for i, p in enumerate(paths)}
    scene_cache = root / "scene_selection/scene_components.npz"
    identity = json.loads(
        (root / "scene_selection/components_identity.json").read_text()
    )
    assert (
        identity["paths_sha256"]
        == hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    )
    student = np.load(scene_cache)["student"]
    student_h = np.array(
        [posterior([z], manifest)[1] for z in student.reshape(-1, 9)], np.float32
    ).reshape(-1, 2)
    aux_identity = dict(
        species_checkpoint=digest(root / "species/best.pt"),
        species_calibration=digest(root / "species/result.json"),
        scene_checkpoint=digest(root / "scene_healthy/best.pt"),
        scene_calibration=digest(root / "scene_healthy/result.json"),
        paths_sha256=identity["paths_sha256"],
    )
    cache = out / "auxiliary_probabilities.npy"
    identity_path = out / "auxiliary_identity.json"
    started = time.time()
    if cache.exists():
        assert json.loads(identity_path.read_text()) == aux_identity
        aux = np.load(cache)
    else:
        species = restore(root / "species").cuda()
        healthy = restore(root / "scene_healthy").cuda()
        aux = np.empty((len(paths), 2), np.float32)

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
                aux[start : start + len(part), 0] = species(x).ravel().cpu().numpy()
                aux[start : start + len(part), 1] = healthy(x).ravel().cpu().numpy()
        np.save(cache, aux)
        identity_path.write_text(json.dumps(aux_identity, indent=2))
    strong = {}
    original_strong = {}

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

    for name, values in photos.items():
        strong[name] = [named(r) for r in values]
        original_strong[name] = [named(r) for r in base[name]]
    rank = []
    for smin, hmin, tmin, veto in itertools.product(
        [0.9, 0.95, 0.99], [0.5, 0.7, 0.8, 0.9], [0.5, 0.7, 0.8, 0.9], [False, True]
    ):
        positive = (
            (student_h >= smin).all(1) & (aux[:, 0] >= tmin) & (aux[:, 1] >= hmin)
        )
        # A newly recovered Healthy verdict needs the same image-level Healthy
        # threshold whether it originated from leaf crops or the plant view.
        weak = (student_h[:, 1] >= 0.6) & (aux[:, 1] >= hmin)
        revised = {}
        changed = 0
        for name, values in photos.items():
            revised[name] = []
            for n, item in enumerate(values):
                row = dict(item)
                old = base[name][n]
                i = lookup[row["path"]]
                blocked = strong[name][n]
                if row["contrast"] < 0.015 and row["sharpness"] < 0.012:
                    revised[name].append(row)
                    continue
                if (
                    row["app_replay"]["kind"] == "HEALTHY"
                    and old["app_replay"]["kind"] != "HEALTHY"
                    and ((veto and not weak[i]) or aux[i, 0] < tmin)
                ):
                    row["leaves"] = old["leaves"]
                    row["app_replay"] = old["app_replay"]
                    blocked = original_strong[name][n]
                    changed += 1
                if (
                    row["app_replay"]["kind"] not in ("HEALTHY", "DISEASE")
                    and not blocked
                    and positive[i]
                ):
                    h = float(student_h[i, 1])
                    probs = np.full(9, (1 - h) / 8, np.float32)
                    probs[1] = h
                    row["whole_frame_evidence"] = dict(
                        probabilities=probs.tolist(), scope="whole_frame"
                    )
                    row["app_replay"] = dict(
                        kind="HEALTHY",
                        label="Healthy",
                        confidence=h,
                        probabilities=probs.tolist(),
                        margin=float(h - (1 - h) / 8),
                        frames=7,
                        leaves=0,
                        partial=False,
                        scope="whole_frame",
                    )
                    changed += 1
                revised[name].append(row)
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
            student_min=smin,
            healthy_min=hmin,
            species_min=tmin,
            veto=veto,
            weak_student_min=0.6,
            weak_healthy_min=hmin,
            metrics=observed,
            retained=retain(observed, baseline),
            field_healthy_passed=field_pass,
            score=score,
            changed_photos=changed,
            leaf_candidate=json.loads(
                (root / "guard_selection/candidate.json").read_text()
            ),
            auxiliary_identity=aux_identity,
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
                seconds=time.time() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
