"""Conservative SIFT overlap families; no diagnostic model or label inference."""

import collections
import concurrent.futures

import cv2
import numpy as np

from phase9_ad_geometric import features


def geometric_edges(rows):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(701)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        cached = list(pool.map(features, rows))
    owners, parts = [], []
    for i, (_, descriptors, _) in enumerate(cached):
        if descriptors is not None:
            parts.append(descriptors)
            owners.extend([i] * len(descriptors))
    if not parts:
        return [], 0
    owners = np.array(owners)
    index = cv2.flann_Index(
        np.concatenate(parts).astype(np.float32), dict(algorithm=1, trees=4)
    )
    votes = collections.Counter()
    for i, (_, descriptors, _) in enumerate(cached):
        if descriptors is None:
            continue
        neighbors, distances = index.knnSearch(descriptors, 10, params=dict(checks=64))
        for ids, dist in zip(neighbors, distances):
            for j in {
                int(owners[k])
                for k, v in zip(ids, dist)
                if owners[k] != i and v < 400**2
            }:
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
        cv2.setRNGSeed(701 + i * len(rows) + j)
        homography, mask = cv2.findHomography(first, second, cv2.RANSAC, 3.0)
        if homography is None or mask is None:
            return None
        inliers = mask.ravel().astype(bool)
        count, fraction = int(inliers.sum()), float(inliers.mean())
        if count < 16 or fraction < 0.7:
            return None
        predicted = cv2.perspectiveTransform(first[:, None, :], homography).reshape(
            -1, 2
        )
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
            inliers=count,
            inlier_fraction=fraction,
            median_reprojection_error=error,
            coverage=coverage,
        )

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        return [edge for edge in pool.map(verify, candidates) if edge is not None], len(
            candidates
        )
