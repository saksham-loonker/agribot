"""Look for overlapping source crops before claiming independent AD examples.

SIFT/homography supplies conservative capture-family links, not disease labels.
Partial overlaps can show different leaves, so class disagreement across these
links is not automatically a label conflict. No diagnostic model is inferred.
"""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import hashlib
import json

import cv2
import numpy as np


def features(row):
    image = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
    assert image is not None
    h, w = image.shape
    scale = 600 / max(h, w)
    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    keypoints, descriptors = cv2.SIFT_create(nfeatures=512).detectAndCompute(
        image, None
    )
    return np.array([k.pt for k in keypoints], np.float32), descriptors, image.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase9_ad_source"
    rows = json.loads((root / "sealed_manifest.json").read_text())
    cv2.setNumThreads(1)
    cv2.setRNGSeed(269)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        cached = list(pool.map(features, rows))
    owners = []
    parts = []
    for i, (_, d, _) in enumerate(cached):
        if d is not None:
            parts.append(d)
            owners.extend([i] * len(d))
    owners = np.array(owners)
    descriptors = np.concatenate(parts).astype(np.float32)
    index = cv2.flann_Index(descriptors, dict(algorithm=1, trees=4))
    votes = collections.Counter()
    for i, (_, d, _) in enumerate(cached):
        if d is None:
            continue
        neighbors, distances = index.knnSearch(d, 10, params=dict(checks=64))
        for ids, dist in zip(neighbors, distances):
            possible = {
                int(owners[j])
                for j, v in zip(ids, dist)
                if owners[j] != i and v < 400**2
            }
            for j in possible:
                votes[tuple(sorted((i, j)))] += 1
    candidates = [pair for pair, count in votes.items() if count >= 12]

    def verify(pair):
        i, j = pair
        pa, da, sa = cached[i]
        pb, db, sb = cached[j]
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return None
        good = [
            m
            for m, n in cv2.BFMatcher().knnMatch(da, db, k=2)
            if m.distance < 0.75 * n.distance
        ]
        if len(good) < 16:
            return None
        first = np.array([pa[m.queryIdx] for m in good])
        second = np.array([pb[m.trainIdx] for m in good])
        cv2.setRNGSeed(269 + i * 503 + j)
        h, mask = cv2.findHomography(first, second, cv2.RANSAC, 3.0)
        if h is None or mask is None:
            return None
        inliers = mask.ravel().astype(bool)
        n = int(inliers.sum())
        ratio = float(inliers.mean())
        if n < 16 or ratio < 0.7:
            return None
        predicted = cv2.perspectiveTransform(first[:, None, :], h).reshape(-1, 2)
        error = float(
            np.median(np.linalg.norm(predicted[inliers] - second[inliers], axis=1))
        )
        coverage = [
            float(
                cv2.contourArea(cv2.convexHull(points[inliers])) / (shape[0] * shape[1])
            )
            for points, shape in [(first, sa), (second, sb)]
        ]
        if error > 1.5 or min(coverage) < 0.04:
            return None
        return dict(
            first=i,
            second=j,
            first_id=rows[i]["id"],
            second_id=rows[j]["id"],
            inliers=n,
            inlier_fraction=ratio,
            median_reprojection_error=error,
            coverage=coverage,
        )

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        edges = [e for e in pool.map(verify, candidates) if e is not None]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    filename_groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        filename_groups[r["group"]].append(i)
    for ids in filename_groups.values():
        for i in ids[1:]:
            union(ids[0], i)
    for e in edges:
        union(e["first"], e["second"])
    components = collections.defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    for ids in components.values():
        group = (
            "ad:conservative_family:"
            + hashlib.sha256(
                "|".join(sorted(rows[i]["id"] for i in ids)).encode()
            ).hexdigest()[:20]
        )
        for i in ids:
            rows[i]["group"] = group
    # Full prior audits retain fingerprints stripped from compact training rows.
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
    overlap = []
    for r in rows:
        distance = int(
            np.bitwise_count(old_phash ^ np.uint64(int(r["phash"], 16))).min()
        )
        if (
            r["file_sha256"] in old_file
            or r["rgb32_sha256"] in old_rgb
            or distance <= 2
        ):
            overlap.append(r["group"])
        r["nearest_prior_phash_distance"] = distance
    excluded = [r for r in rows if r["group"] in set(overlap)]
    rows = [r for r in rows if r["group"] not in set(overlap)]
    output = root / "sealed_geometric_manifest.json"
    text = json.dumps(rows, indent=2) + "\n"
    if output.exists():
        assert output.read_text() == text, "Geometric membership changed"
    else:
        output.write_text(text)
    healthy_groups = {r["group"] for r in rows if r["label"] == "Healthy"}
    report = dict(
        raw_n=len(cached),
        candidate_pairs=len(candidates),
        geometric_edges=edges,
        accepted=len(rows),
        accepted_counts=dict(collections.Counter(r["raw_label"] for r in rows)),
        capture_families=len({r["group"] for r in rows}),
        healthy_capture_families=len(healthy_groups),
        prior_file_hashes=len(old_file),
        prior_phashes=len(old_phash),
        excluded_prior_overlap=excluded,
        manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        predictions_opened=False,
        training_used=False,
        policy="Filename blocks of five merged with SIFT/RANSAC geometric overlap; retain all primary-leaf crops and use family-aware evaluation. Partial overlap does not prove incompatible primary-leaf labels.",
        limitations="Families are conservative image overlap links, not verified plant identities. Standard full-image fingerprints do not guarantee absence of cropped copies in earlier sources. Publisher image labels are not independently adjudicated whole-plant diagnoses.",
    )
    (root / "geometric_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "raw_n",
                    "candidate_pairs",
                    "accepted",
                    "accepted_counts",
                    "capture_families",
                    "healthy_capture_families",
                    "prior_file_hashes",
                    "prior_phashes",
                    "manifest_sha256",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
