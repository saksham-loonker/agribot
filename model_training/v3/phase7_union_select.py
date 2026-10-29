"""Combine individually retained context policies on opened cohorts only.

Positive image evidence requires one constituent's complete agreement policy;
contradiction requires all constituents. No disease classifier is retrained.
This is a development ablation, never an acceptance or deployment command.
"""

from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np

from app_eval import posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from phase2_select import retain
from phase5_scene_ablation import grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--selection-runs", nargs="+", required=True)
    ap.add_argument("--run-name", default="union_selection")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / a.run_name
    out.mkdir(exist_ok=True)
    assert len(a.selection_runs) >= 2 and a.top > 0
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
    path_sha = hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    options = []
    for run in a.selection_runs:
        directory = root / run
        identity = json.loads((directory / "input_identity.json").read_text())
        assert identity["paths_sha256"] == path_sha
        logits = np.load(directory / "evaluation_logits.npy")
        selected = []
        seen = set()
        for c in json.loads((directory / "ranked.json").read_text()):
            if not c["retained"]:
                continue
            assert c["fusion"]["input_identity"] == identity["input_identity"]
            f = c["fusion"]
            z = (
                np.clip(logits, -f["input_clip"], f["input_clip"])
                @ np.array(f["weights"])
                + f["bias"]
            )
            p = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
            species = 1 / (1 + np.exp(-np.clip(logits[:, -1], -80, 80)))
            positive = (p >= c["healthy_min"]) & (species >= c["species_min"])
            negative = (
                (p <= c["negative_healthy_max"])
                if c["negative_healthy_max"]
                else np.zeros(len(p), bool)
            )
            key = (positive.tobytes(), negative.tobytes())
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                (
                    dict(
                        selection_run=run,
                        id=c["id"],
                        fusion=f,
                        healthy_min=c["healthy_min"],
                        species_min=c["species_min"],
                        negative_healthy_max=c["negative_healthy_max"],
                    ),
                    p,
                    positive,
                    negative,
                )
            )
            if len(selected) >= a.top:
                break
        assert selected, "No individually retained candidates in " + run
        options.append(selected)

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
    rank = []
    for combination in itertools.product(*options):
        positive = np.logical_or.reduce([c[2] for c in combination])
        negative = np.logical_and.reduce([c[3] for c in combination])
        probability = np.max([np.where(c[2], c[1], 0.0) for c in combination], axis=0)
        revised = {}
        for name, values in photos.items():
            revised[name] = []
            for n, row in enumerate(values):
                i = lookup[row["path"]]
                revised[name].append(
                    route_context(
                        row,
                        base[name][n],
                        manifest,
                        bool(positive[i]),
                        bool(negative[i]),
                        float(probability[i]),
                        strong[name][n],
                        base_strong[name][n],
                    )
                )
        metrics = grouped(revised)
        m = metrics["phase5_selection"]
        field_pass = all(
            m["by_source"][source]["healthy_recall"] >= 0.7
            for source in ("tom2024", "swtt5wy2pr")
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
            id=len(rank),
            constituents=[c[0] for c in combination],
            positive_rule="any_complete_constituent",
            negative_rule="all_constituents",
            metrics=metrics,
            retained=retain(metrics, baseline),
            field_healthy_passed=field_pass,
            score=score,
            reserved_used=False,
        )
        rank.append(c)
        print(
            json.dumps(
                dict(
                    id=c["id"],
                    retained=c["retained"],
                    field_passed=field_pass,
                    H=m["healthy"]["correct"],
                    FH=m["all"]["false_healthy"],
                    sources=m["by_source"],
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
    for name, value in [
        ("candidate", rank[0]),
        ("ranked", rank),
        ("baseline", baseline),
    ]:
        (out / (name + ".json")).write_text(json.dumps(value, indent=2))
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
