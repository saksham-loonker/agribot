"""Paired baseline/candidate app-replay comparison clustered by source group."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = [
    "Early_blight", "Healthy", "Late_blight", "Leaf Miner", "Magnesium Deficiency",
    "Nitrogen Deficiency", "Pottassium Deficiency", "Spotted Wilt Virus", "Other",
]


def read(path):
    return {row["id"]: row for row in (json.loads(line) for line in Path(path).read_text().splitlines())}


def scope(row):
    if row["label"] in LABELS[:-1]:
        return "supported_tomato"
    if row["label"] == "Other":
        names = [o.get("source_label", "") for o in row.get("objects", [])]
        if row["source"] == "plantvillage" or (names and all(n.startswith("Tomato") for n in names)):
            return "unsupported_tomato"
        if names:
            return "non_tomato"
    return "mixed"


def healthy(row):
    return row["label"] == "Healthy"


def diseased_or_unsupported(row):
    return row["label"] not in ("Healthy", "Mixed")


def correct(row):
    replay = row["app_replay"]
    return (replay["kind"] == "HEALTHY" and row["label"] == "Healthy") or (
        replay["kind"] == "DISEASE" and replay["label"] == row["label"])


def is_healthy(row):
    return row["app_replay"]["kind"] == "HEALTHY"


def false_known(row):
    return row["app_replay"]["kind"] in ("HEALTHY", "DISEASE")


def rates(ids, rows):
    out = {}
    healthy_ids = [i for i in ids if healthy(rows[i])]
    bad_ids = [i for i in ids if diseased_or_unsupported(rows[i])]
    supported = [i for i in ids if scope(rows[i]) == "supported_tomato"]
    unsupported = [i for i in ids if scope(rows[i]) in ("unsupported_tomato", "non_tomato")]
    out["healthy_recall"] = (sum(is_healthy(rows[i]) for i in healthy_ids) / len(healthy_ids)) if healthy_ids else None
    out["false_healthy_rate"] = (sum(is_healthy(rows[i]) for i in bad_ids) / len(bad_ids)) if bad_ids else None
    out["supported_correct_rate"] = (sum(correct(rows[i]) for i in supported) / len(supported)) if supported else None
    out["unsupported_false_known_rate"] = (sum(false_known(rows[i]) for i in unsupported) / len(unsupported)) if unsupported else None
    out["unsupported_other_rate"] = (sum(rows[i]["app_replay"]["kind"] == "UNKNOWN_CONDITION" for i in unsupported) / len(unsupported)) if unsupported else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstraps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--include-ineligible", action="store_true")
    args = parser.parse_args()
    baseline, candidate = read(args.baseline), read(args.candidate)
    ids = sorted(i for i in set(baseline) & set(candidate)
                 if args.include_ineligible or baseline[i].get("analysis_eligible", True))
    assert ids, "no overlapping predictions"
    groups = defaultdict(list)
    for i in ids:
        groups[baseline[i]["group"]].append(i)
    group_keys = sorted(groups)
    rng = np.random.default_rng(args.seed)
    metrics = ("healthy_recall", "false_healthy_rate", "supported_correct_rate",
               "unsupported_false_known_rate", "unsupported_other_rate")
    base_rates = rates(ids, baseline)
    cand_rates = rates(ids, candidate)
    result = dict(
        n=len(ids),
        groups=len(group_keys),
        baseline=base_rates,
        candidate=cand_rates,
        delta={k: (None if base_rates[k] is None or cand_rates[k] is None else cand_rates[k] - base_rates[k])
               for k in metrics},
        bootstraps=[],
    )
    samples = {k: [] for k in metrics}
    for _ in range(args.bootstraps):
        chosen = rng.choice(group_keys, size=len(group_keys), replace=True)
        sample_ids = [i for key in chosen for i in groups[key]]
        b = rates(sample_ids, baseline)
        c = rates(sample_ids, candidate)
        for key in metrics:
            if b[key] is not None and c[key] is not None:
                samples[key].append(c[key] - b[key])
    for key in metrics:
        values = np.asarray(samples[key], dtype=np.float64)
        if len(values):
            result["bootstraps"].append(dict(
                metric=key, n=int(len(values)),
                mean=float(values.mean()),
                ci95=[float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
                one_sided_upper95=float(np.percentile(values, 95)),
                probability_gain=float((values > 0).mean()),
            ))
    (Path(args.out)).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("n", "groups", "baseline", "candidate", "delta")}, indent=1))
    print(json.dumps(result["bootstraps"], indent=1))


if __name__ == "__main__":
    main()
