"""Build real-observation Kotlin replay inputs from verified mobile caches only."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from app_eval import posterior
from opened_context_data import load, read
from permissive_encoder import digest
from phase29_compact_prepare import region_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    data = load(p)
    table = read(p / "phase29_compact/center_indices.json")
    full = read(p / "phase29_compact/full_indices.json")
    keys = {
        path: [region_key(path, leaf["box"]) for leaf in row["leaves"]]
        for path, row in data["photos"].items()
    }
    indices = sorted(
        {t[key] for row in keys.values() for key in row for t in (table, full)}
    )
    verified = read(p / "phase73_mobile_verified/protocol.json")
    assert (
        hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        == verified["leaf_indices_sha256"]
    )
    outputs = np.load(p / "phase73_mobile_verified/leaf_outputs.npy", mmap_mode="r")
    lookup = {
        index: posterior([outputs[i, :9]], data["manifest"]).tolist()
        for i, index in enumerate(indices)
    }
    fixture = read(p / "phase74_native_contract/policy_cases.json")
    after = read(p / "phase73_mobile_verified/reference.json")
    for case in fixture["cases"]:
        name, index = case["id"].rsplit(":", 1)
        index = int(index)
        row = (
            data["internal"][index] if name == "internal" else data["base"][name][index]
        )
        path = row["path"]
        case["observations"] = [
            [leaf["score"], lookup[table[key]]]
            for leaf, key in zip(row["leaves"], keys[path])
        ]
        case["quality_weight"] = row.get("quality_weight", 1.0)
        case["expected_kind"] = after[name][index]["kind"]
        case["expected_label"] = after[name][index].get("label")
    fixture["note"] = (
        "All inputs are opened development. Kotlin replays actual leaf counts and distinct frames; primary scope never invents leaf votes."
    )
    fixture["source_sha256"] = digest(__file__)
    target = p / "phase77_native_temporal"
    target.mkdir(exist_ok=True)
    output = target / "temporal_cases.json"
    encoded = json.dumps(fixture, separators=(",", ":"))
    if output.exists():
        assert output.read_text() == encoded
    else:
        output.write_text(encoded)
    print(
        json.dumps(
            dict(
                cases=len(fixture["cases"]),
                bytes=output.stat().st_size,
                sha256=digest(output),
            )
        )
    )


if __name__ == "__main__":
    main()
