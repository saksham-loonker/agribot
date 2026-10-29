"""Diagnose completed compact controls using cached scores, never cold data."""

from pathlib import Path
import argparse
import collections
import json
import numpy as np

from opened_context_data import read
from phase29_compact_prepare import region_key
from phase7_species_train import balanced
from source_balance import source_class_group_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    root = p / "phase29_compact"
    rows = read(root / "fitting_rows.json")
    for row in rows:
        row["species_y"] = row["y"]
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    masses = {}
    for name, fn in [
        ("historical_domain", balanced),
        ("publisher_source", source_class_group_weights),
    ]:
        mass = collections.defaultdict(float)
        for i, w in zip(train, fn(rows, train)):
            mass[rows[i]["source"] + ":" + str(rows[i]["y"])] += float(w)
        masses[name] = dict(mass)
    counts = collections.Counter(
        (r["source"], r["split"], r["y"], r["role"]) for r in rows
    )
    ranked = read(root / "head_bce/ranked.json")
    selected = max(
        ranked, key=lambda r: (r["worst_source_healthy"], r["mean_source_healthy"])
    )
    internal = [
        json.loads(x)
        for x in (p / "phase23_data/internal_baseline_predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    center, full = (
        read(root / (name + "_indices.json")) for name in ("center", "full")
    )
    result = read(root / "head_bce/result.json")
    z = np.load(root / "head_bce/all_logits.npy")
    a, b = result["calibration"]
    probability = 1 / (1 + np.exp(-np.clip(z * a + b, -80, 80)))
    distribution = collections.defaultdict(list)
    for r in internal:
        if not r["leaves"]:
            continue
        label = r.get("label", r.get("expected_label"))
        key = (r["source"], str(label))
        for leaf in r["leaves"]:
            k = region_key(r["path"], leaf["box"])
            distribution[key].append(
                [float(probability[center[k]]), float(probability[full[k]])]
            )
    report = dict(
        training_sampling_mass=masses,
        fitting_counts=[
            dict(source=s, split=sp, y=y, role=role, n=n)
            for (s, sp, y, role), n in sorted(counts.items())
        ],
        maximum_recall_policy={
            k: selected[k]
            for k in (
                "epoch",
                "healthy_min",
                "checks",
                "internal_metrics",
                "worst_source_healthy",
                "mean_source_healthy",
            )
        },
        selected_safe_epoch=result["selected_epoch"],
        leaf_scores=[
            dict(
                source=s,
                label=l,
                n=len(v),
                center_quantiles=np.quantile(
                    np.array(v)[:, 0], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                ).tolist(),
                full_quantiles=np.quantile(
                    np.array(v)[:, 1], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                ).tolist(),
            )
            for (s, l), v in sorted(distribution.items())
        ],
        cold_olid_used=False,
    )
    (root / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
