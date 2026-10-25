"""Audit downloaded AD labels/overlap and seal evaluation membership, no inference."""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import hashlib
import json
import re

import numpy as np

from campaign_data import profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase9_ad_source"
    download = json.loads((root / "download_audit.json").read_text())
    assert (
        download["images"] == 502
        and not download["predictions_opened"]
        and not download["training_used"]
    )
    old = json.loads((p / "phase6_roi/cnn_inputs/rows.json").read_text()) + json.loads(
        (p / "phase6_roi/reserved_manifest.json").read_text()
    )
    slif = p / "phase2/slif_evaluation/baseline/predictions.jsonl"
    if slif.exists():
        old.extend(json.loads(line) for line in slif.read_text().splitlines())
    old_file = {r["file_sha256"] for r in old if r.get("file_sha256")}
    old_rgb = {r["rgb32_sha256"] for r in old if r.get("rgb32_sha256")}
    old_phash = np.array(
        sorted({int(r["phash"], 16) for r in old if r.get("phash")}), np.uint64
    )
    mapping = {
        "Healthy": "Healthy",
        "Early Blight": "Early_blight",
        "Bacterial Spot": "Other",
        "Mosaic Virus": "Other",
    }
    records = []
    for f in sorted((root / "files/AD").rglob("*.jpg")):
        category = f.parent.name
        assert category in mapping
        records.append(
            dict(
                id="ad:" + str(f.relative_to(root / "files")),
                path=str(f),
                source="ad_greenhouse",
                species="tomato",
                label=mapping[category],
                raw_label=category,
                box=None,
                split="locked_external",
                licence="CC BY 4.0",
                source_version=download["source_doi"],
                source_url=download["source_url"],
                label_scope="Publisher image-level primary-leaf category; not verified every-leaf or whole-plant disease-free truth.",
            )
        )
    assert len(records) == 502
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for row, (metadata, _) in zip(records, pool.map(profile, records)):
            row.update(metadata)
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    duplicate_pairs = []
    for i, row in enumerate(records):
        for j in range(i):
            other = records[j]
            if (
                row["file_sha256"] == other["file_sha256"]
                or row["rgb32_sha256"] == other["rgb32_sha256"]
                or (int(row["phash"], 16) ^ int(other["phash"], 16)).bit_count() <= 2
            ):
                union(i, j)
                duplicate_pairs.append(
                    dict(
                        first=row["id"],
                        second=other["id"],
                        same_raw_label=row["raw_label"] == other["raw_label"],
                    )
                )
    components = collections.defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    excluded = {}
    representatives = []
    for ids in components.values():
        if len({records[i]["raw_label"] for i in ids}) > 1:
            for i in ids:
                excluded[i] = (
                    "Cross-category exact/near duplicate component; publisher category is ambiguous"
                )
            continue
        keep = min(ids, key=lambda i: records[i]["id"])
        representatives.append(keep)
        for i in ids:
            if i != keep:
                excluded[i] = "Repeated exact/near duplicate; retain one representative"
    accepted = []
    for i in representatives:
        r = records[i]
        distance = (
            int(np.bitwise_count(old_phash ^ np.uint64(int(r["phash"], 16))).min())
            if len(old_phash)
            else None
        )
        if (
            r["file_sha256"] in old_file
            or r["rgb32_sha256"] in old_rgb
            or (distance is not None and distance <= 2)
        ):
            excluded[i] = (
                "Exact/near overlap with prior training, validation or opened regression"
            )
        else:
            number = int(re.search(r"(\d+)$", Path(r["path"]).stem).group(1))
            r["group"] = (
                "ad:inferred_adjacent_5:"
                + r["raw_label"]
                + ":"
                + str((number - 1) // 5)
            )
            r["nearest_prior_phash_distance"] = distance
            accepted.append(r)
    manifest = root / "sealed_manifest.json"
    text = json.dumps(accepted, indent=2) + "\n"
    if manifest.exists():
        assert manifest.read_text() == text, "Sealed AD membership changed"
    else:
        manifest.write_text(text)
    report = dict(
        source=download,
        raw_images=len(records),
        accepted=len(accepted),
        accepted_counts=dict(collections.Counter(r["raw_label"] for r in accepted)),
        inferred_groups=len({r["group"] for r in accepted}),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        prior_file_hashes=len(old_file),
        prior_rgb_hashes=len(old_rgb),
        prior_phashes=len(old_phash),
        excluded=[
            dict(record=records[i], reason=reason)
            for i, reason in sorted(excluded.items())
        ],
        duplicate_pairs=duplicate_pairs,
        predictions_opened=False,
        training_used=False,
        limitations="Publisher labels, not expert-adjudicated clinical ground truth. No plant IDs or EXIF capture dates verified; five-adjacent filenames only infer capture families. More distant overlapping crops may remain. The scope contact sheet is for annotation-role review, not model selection.",
        status="Membership sealed before inference; visual scope review and a frozen acceptance protocol still required.",
    )
    (root / "membership_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "raw_images",
                    "accepted",
                    "accepted_counts",
                    "inferred_groups",
                    "manifest_sha256",
                    "prior_file_hashes",
                    "prior_phashes",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
