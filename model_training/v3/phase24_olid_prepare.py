"""Seal OLID tomato membership and label scope without diagnostic predictions."""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import hashlib
import json
import re

import numpy as np

from campaign_data import profile
from capture_families import geometric_edges
from permissive_encoder import digest

MAPPING = {
    "tomato__healthy": "Healthy",
    "tomato__LM": "Leaf Miner",
    "tomato__MIT": "Other",
    "tomato__JAS_MIT": "Other",
    "tomato__K": "Pottassium Deficiency",
    "tomato__N": "Nitrogen Deficiency",
    "tomato__N_K": "Diseased_unspecified",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase24_olid_source"
    audit = json.loads((root / "download_audit.json").read_text())
    assert (
        audit["license"] == "CC BY 4.0"
        and not audit["diagnostic_predictions_opened"]
        and not audit["training_used"]
    )
    rows = []
    for path in sorted((root / "files").rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        raw = path.parent.name
        assert raw in MAPPING
        number = int(re.search(r"\((\d+)\)", path.stem).group(1))
        rows.append(
            dict(
                id="olid:" + str(path.relative_to(root / "files")),
                path=str(path),
                source="olid_tomato",
                species="tomato",
                label=MAPPING[raw],
                raw_label=raw,
                box=None,
                split="locked_external",
                group="olid:" + raw + ":block" + str((number - 1) // 5),
                licence="CC BY 4.0",
                source_version=audit["source_doi"],
                source_url="https://zenodo.org/records/8105154",
                label_scope="Publisher image-level leaf condition. N_K is coarse diseased, never nine-class Other; Jassid+Mite are explicitly unsupported. Not a whole-plant health guarantee.",
            )
        )
    assert len(rows) == 798
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for row, (metadata, _) in zip(rows, pool.map(profile, rows)):
            row.update(metadata)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    # Exact/near copies cannot inflate sample size. Contradictory exact/near
    # labels are quarantined, not relabelled according to a model prediction.
    duplicate_edges = []
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            if (
                row["file_sha256"] == other["file_sha256"]
                or row["model_input_sha256"] == other["model_input_sha256"]
                or (int(row["phash"], 16) ^ int(other["phash"], 16)).bit_count() <= 2
            ):
                union(i, j)
                duplicate_edges.append([row["id"], other["id"]])
    duplicates = collections.defaultdict(list)
    for i in range(len(rows)):
        duplicates[find(i)].append(i)
    excluded = {}
    for ids in duplicates.values():
        if len({rows[i]["label"] for i in ids}) > 1:
            for i in ids:
                excluded[i] = "Contradictory exact/near-copy publisher labels"
        else:
            for i in sorted(ids, key=lambda i: rows[i]["id"])[1:]:
                excluded[i] = (
                    "Repeated exact/near-copy image; one representative retained"
                )
    filename_groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        filename_groups[row["group"]].append(i)
    for ids in filename_groups.values():
        for i in ids[1:]:
            union(ids[0], i)
    edges, candidates = geometric_edges(rows)
    for edge in edges:
        union(edge["first"], edge["second"])
    components = collections.defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    for ids in components.values():
        group = (
            "olid:conservative_family:"
            + hashlib.sha256(
                "|".join(sorted(rows[i]["id"] for i in ids)).encode()
            ).hexdigest()[:20]
        )
        for i in ids:
            rows[i]["group"] = group
    old, prior_manifests = [], {}
    for relative in [
        "original_quality_profile.json",
        "new_development_manifest.json",
        "locked_new_source_manifest.json",
        "locked_binary_manifest.json",
        "mixed_stress_manifest.json",
        "phase2_source_manifest.json",
        "phase6_roi/cnn_inputs/rows.json",
        "phase6_roi/reserved_manifest.json",
        "phase13_siglip/admission_manifest.json",
        "phase9_ad_source/sealed_geometric_manifest.json",
    ]:
        file = p / relative
        if file.exists():
            old.extend(json.loads(file.read_text()))
            prior_manifests[relative] = digest(file)
    old_file = {r["file_sha256"] for r in old if r.get("file_sha256")}
    old_rgb = {r["rgb32_sha256"] for r in old if r.get("rgb32_sha256")}
    old_input = {r["model_input_sha256"] for r in old if r.get("model_input_sha256")}
    old_phash = np.array(
        sorted({int(r["phash"], 16) for r in old if r.get("phash")}), np.uint64
    )
    assert len(old_file) > 30000
    overlap = set()
    for row in rows:
        distance = int(
            np.bitwise_count(old_phash ^ np.uint64(int(row["phash"], 16))).min()
        )
        row["nearest_prior_phash_distance"] = distance
        if (
            row["file_sha256"] in old_file
            or row["rgb32_sha256"] in old_rgb
            or row["model_input_sha256"] in old_input
            or distance <= 2
        ):
            overlap.add(row["group"])
    for i, row in enumerate(rows):
        if row["group"] in overlap:
            excluded[i] = "Entire inferred family overlaps an opened/training source"
    accepted = [r for i, r in enumerate(rows) if i not in excluded]
    output = root / "sealed_manifest.json"
    text = json.dumps(accepted, indent=2) + "\n"
    if output.exists():
        assert output.read_text() == text, "Frozen source membership changed"
    else:
        output.write_text(text)
    report = dict(
        source=audit,
        raw_n=len(rows),
        accepted_n=len(accepted),
        accepted_counts=dict(collections.Counter(r["raw_label"] for r in accepted)),
        conservative_families=len({r["group"] for r in accepted}),
        healthy_families=len({r["group"] for r in accepted if r["label"] == "Healthy"}),
        duplicate_edges=duplicate_edges,
        geometric_candidate_pairs=candidates,
        geometric_edges=edges,
        exclusions=[
            dict(id=rows[i]["id"], reason=reason) for i, reason in excluded.items()
        ],
        prior_manifests=prior_manifests,
        prior_file_hashes=len(old_file),
        prior_phashes=len(old_phash),
        sealed_manifest_sha256=digest(output),
        script_sha256=digest(__file__),
        diagnostic_predictions_opened=False,
        training_used=False,
        limitations="Inferred filename/overlap families are not verified plant IDs. No exhaustive cross-source SIFT crop matching. Publisher leaf categories are not independent clinical adjudication.",
    )
    (root / "membership_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "source",
                    "duplicate_edges",
                    "geometric_edges",
                    "prior_manifests",
                    "exclusions",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
