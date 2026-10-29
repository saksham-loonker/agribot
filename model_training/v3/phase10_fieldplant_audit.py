"""Audit FieldPlant leaf geometry, prior overlap and grouped fitting membership.

Healthy refers to the annotated leaf, never every leaf in its photo. Broad
disease names supply binary negatives only, and no new genus supervision is
inferred from negative class strings. No diagnostic model is used here.
"""

from pathlib import Path
import argparse
import collections
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import math
import re

import numpy as np

from campaign_data import profile


def capture_family(filename):
    stem = filename.split(".rf.")[0]
    match = re.search(r"IMG_?(\d{8})_?(\d{6})", stem)
    if match:
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        return "fieldplant:inferred_3min:" + str(int(stamp.timestamp()) // 180)
    stem = re.sub(r"(?i)(?:_jpg|_jpeg|_png)$", "", stem).rstrip("-_ ")
    match = re.match(r"^(.*?)[-_( ]+(\d+)[)_\- ]*$", stem)
    if match:
        return (
            "fieldplant:inferred_adjacent5:"
            + match[1].lower()
            + ":"
            + str((int(match[2]) - 1) // 5)
        )
    return "fieldplant:filename:" + stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase10_fieldplant"
    complete = json.loads((root / "download_complete.json").read_text())
    assert complete["all_member_crcs_verified"]
    annotation = json.loads((root / "subset_annotations.json").read_text())
    by = collections.defaultdict(list)
    for row in annotation:
        by[row["filename"]].append(row)
    photos = [
        dict(
            id="fieldplant:" + name,
            path=str(root / "images" / name),
            source="fieldplant",
            species="unverified",
            box=None,
            group=capture_family(name),
            source_version="Roboflow FieldPlant v11 export 2023-08-31, public Kaggle mirror version 1",
            source_url=complete["original_source"],
            licence="CC BY 4.0",
        )
        for name in sorted(by)
    ]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for row, (metadata, _) in zip(photos, pool.map(profile, photos)):
            row.update(metadata)
    old = []
    for name in [
        "original_quality_profile.json",
        "new_development_manifest.json",
        "locked_new_source_manifest.json",
        "locked_binary_manifest.json",
        "mixed_stress_manifest.json",
        "phase2_source_manifest.json",
    ]:
        file = p / name
        if file.exists():
            old.extend(json.loads(file.read_text()))
    old.extend(json.loads((p / "phase6_roi/cnn_inputs/rows.json").read_text()))
    old.extend(json.loads((p / "phase6_roi/reserved_manifest.json").read_text()))
    old_file = {r["file_sha256"] for r in old if r.get("file_sha256")}
    old_rgb = {r["rgb32_sha256"] for r in old if r.get("rgb32_sha256")}
    old_phash = np.array(
        sorted({int(r["phash"], 16) for r in old if r.get("phash")}), np.uint64
    )
    parent = list(range(len(photos)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    families = collections.defaultdict(list)
    for i, r in enumerate(photos):
        families[r["group"]].append(i)
    for ids in families.values():
        for i in ids[1:]:
            union(ids[0], i)
    ph = np.array([int(r["phash"], 16) for r in photos], np.uint64)
    for i, r in enumerate(photos):
        for j in np.flatnonzero(np.bitwise_count(ph[:i] ^ ph[i]) <= 2):
            union(i, int(j))
    groups = collections.defaultdict(list)
    for i in range(len(photos)):
        groups[find(i)].append(i)
    overlap = set()
    excluded = []
    for ids in groups.values():
        key = (
            "fieldplant:family:"
            + hashlib.sha256(
                "|".join(sorted(photos[i]["id"] for i in ids)).encode()
            ).hexdigest()[:20]
        )
        for i in ids:
            r = photos[i]
            r["group"] = key
            r["nearest_prior_phash_distance"] = int(
                np.bitwise_count(old_phash ^ ph[i]).min()
            )
            if (
                r["file_sha256"] in old_file
                or r["rgb32_sha256"] in old_rgb
                or r["nearest_prior_phash_distance"] <= 2
            ):
                overlap.add(key)
    healthy_groups = {
        r["group"]
        for r in photos
        if r["group"] not in overlap
        and any(x["class"] == "Tomato healthy" for x in by[Path(r["path"]).name])
    }
    all_groups = {r["group"] for r in photos if r["group"] not in overlap}
    validation = set()
    for values in [healthy_groups, all_groups - healthy_groups]:
        ordered = sorted(
            values,
            key=lambda g: hashlib.sha256(("fieldplant317:" + g).encode()).hexdigest(),
        )
        validation.update(
            ordered[: max(1, math.ceil(0.2 * len(ordered)))] if ordered else []
        )
    leaves = []
    scenes = []
    invalid = []
    minimum_short_side = 48
    for photo in photos:
        name = Path(photo["path"]).name
        if photo["group"] in overlap:
            excluded.append(
                dict(
                    photo=photo,
                    reason="Prior overlap; entire inferred capture family quarantined",
                )
            )
            continue
        split = "val" if photo["group"] in validation else "train"
        photo["split"] = split
        all_healthy = all(r["class"] == "Tomato healthy" for r in by[name])
        scenes.append(
            dict(
                photo,
                label="Healthy" if all_healthy else "Diseased_unspecified",
                raw_label=(
                    "all annotated leaves Healthy"
                    if all_healthy
                    else "at least one annotated non-Healthy leaf"
                ),
                label_scope="Annotated visible leaves only; unannotated background and whole-plant disease freedom are not certified.",
                annotations=by[name],
            )
        )
        for index, row in enumerate(by[name]):
            box = [float(row[k]) for k in ["xmin", "ymin", "xmax", "ymax"]]
            w, h = photo["width"], photo["height"]
            valid = (
                int(row["width"]) == w
                and int(row["height"]) == h
                and all(math.isfinite(v) for v in box)
                and 0 <= box[0] < box[2] <= w
                and 0 <= box[1] < box[3] <= h
            )
            if not valid or min(box[2] - box[0], box[3] - box[1]) < minimum_short_side:
                invalid.append(
                    dict(
                        photo=photo["id"],
                        annotation=index,
                        box=box,
                        reason="Invalid dimensions/geometry or short side below 48 pixels",
                    )
                )
                continue
            healthy = row["class"] == "Tomato healthy"
            leaves.append(
                dict(
                    photo,
                    id=photo["id"] + ":annotation:" + str(index),
                    box=box,
                    label="Healthy" if healthy else "Diseased_unspecified",
                    raw_label=row["class"],
                    health_y=float(healthy),
                    known_y=1.0 if healthy else -1.0,
                    label_scope="Publisher-annotated visible leaf; binary supervision only. No named disease or negative-ROI genus target inferred.",
                )
            )
    assert {r["split"] for r in leaves if r["label"] == "Healthy"} == {"train", "val"}
    assert not (
        {r["group"] for r in leaves if r["split"] == "train"}
        & {r["group"] for r in leaves if r["split"] == "val"}
    )
    assert not (
        {r["path"] for r in leaves}
        & {
            r["path"]
            for r in json.loads(
                (p / "phase9_ad_source/sealed_geometric_manifest.json").read_text()
            )
        }
    )
    for name, values in [("leaf_manifest", leaves), ("scene_manifest", scenes)]:
        file = root / (name + ".json")
        text = json.dumps(values, indent=2) + "\n"
        if file.exists():
            assert file.read_text() == text, "Audited membership changed"
        else:
            file.write_text(text)
    report = dict(
        source=complete,
        photos=len(photos),
        leaf_instances=len(leaves),
        leaf_counts=dict(
            collections.Counter(r["split"] + ":" + r["label"] for r in leaves)
        ),
        scene_counts=dict(
            collections.Counter(r["split"] + ":" + r["label"] for r in scenes)
        ),
        groups=len(all_groups),
        healthy_groups=len(healthy_groups),
        excluded_prior_overlap=excluded,
        invalid_or_small_annotations=invalid,
        leaf_manifest_sha256=hashlib.sha256(
            (root / "leaf_manifest.json").read_bytes()
        ).hexdigest(),
        scene_manifest_sha256=hashlib.sha256(
            (root / "scene_manifest.json").read_bytes()
        ).hexdigest(),
        minimum_leaf_short_side_pixels=48,
        predictions_opened=False,
        training_used=False,
        review="Sixteen contact-sheet examples show leaf-scale boxes, including truncated leaves. Suspected non-tomato foliage occurs among negative annotations, so negative genus supervision stays masked. Healthy labels apply only to annotated leaf ROIs; 128/141 photos containing Healthy annotations also contain non-Healthy annotations.",
        allowed_scope="Limited binary Healthy/knownness leaf training; broad disease labels are masked for nine-class CE and knownness negatives. Never label every leaf or a mixed whole photo Healthy. No new detector training is authorized by this audit alone.",
        limitations="Inferred timestamp/adjacent-file/perceptual families are not verified plant IDs. Clinical and species labels are publisher annotations, not independently adjudicated by this audit.",
    )
    (root / "membership_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "photos",
                    "leaf_instances",
                    "leaf_counts",
                    "scene_counts",
                    "groups",
                    "healthy_groups",
                    "leaf_manifest_sha256",
                    "scene_manifest_sha256",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
