"""Retire consumed capture families into a new image-scoped data revision.

Original membership files remain immutable. Contradictory/prior-overlap
quarantines remain excluded. The cold AD source supplies overlap hashes only.
"""

from pathlib import Path
import argparse
import concurrent.futures
import collections
import csv
import hashlib
import json
import re
import shutil

import numpy as np
from PIL import Image

from app_eval import classifier_letterbox_input
from campaign_data import profile
from phase4_leaf_source import MAPPING as PAPER
from phase5_sources import TOM, tom_group


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    old = p / "phase5"
    scene = p / "phase11_scene/cnn_inputs"
    root = p / "phase12_scene"
    root.mkdir(exist_ok=True)
    work = root / "cnn_inputs"
    work.mkdir(exist_ok=True)
    complete = root / "complete.json"
    if complete.exists():
        assert read(complete)["rows_sha256"] == digest(work / "rows.json")
        print("Data revision already complete")
        return
    assert (
        p / "phase11_scene/prototype_verification/opened_regression_result.json"
    ).exists(), "Record the consumed-cohort result before retiring it"
    development = read(old / "development_manifest.json")
    reserved = read(old / "reserved_manifest.json")
    excluded = {r["id"] for r in read(old / "exclusions.json")}
    known = {r["id"]: r for r in development + reserved}
    dev_ids = {r["id"] for r in development}
    jobs = []
    for category, items in read(p / "tom2024_acceptance_files.json").items():
        if category not in TOM:
            continue
        for item in items:
            path = p / "sources/tom2024" / category / item["filename"]
            jobs.append(
                (
                    "tom2024",
                    category,
                    TOM[category],
                    tom_group(path.name),
                    path,
                    "10.17632/3d4yg89rtr.1",
                )
            )
    with (p / "new_field_metadata.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            category = r["Class_Label"]
            if category not in PAPER:
                continue
            group = (
                "paper:"
                + category
                + ":block"
                + str(int(re.search(r"(\d+)$", r["Object_ID"]).group()) // 5)
            )
            path = (
                p
                / "sources/paper_full/images/Resize Dataset"
                / category
                / r["File_Name"]
            )
            jobs.append(
                (
                    "swtt5wy2pr",
                    category,
                    PAPER[category],
                    group,
                    path,
                    "10.17632/swtt5wy2pr.1",
                )
            )

    def recover(job):
        source, category, label, group, path, doi = job
        key = "phase5:" + source + ":" + category + ":" + path.name
        if key in known:
            r = dict(known[key])
            assert r["label"] == label and r["capture_group"] == group
        else:
            r = dict(
                id=key,
                path=str(path),
                source=source,
                species="tomato",
                label=label,
                raw_label=category,
                box=None,
                capture_group=group,
                group=group,
                licence="CC BY 4.0",
                source_version=doi,
                source_url="https://data.mendeley.com/datasets/"
                + doi.split("/")[1].rsplit(".", 1)[0]
                + "/1",
            )
            info, _ = profile(r)
            r.update(info)
        assert digest(path) == r["file_sha256"]
        r["group"] = group
        return r

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        all_rows = list(pool.map(recover, jobs))
    assert len(all_rows) == 5214
    parents = {r["capture_group"]: r["capture_group"] for r in all_rows}

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parents[max(x, y)] = min(x, y)

    hashes = {}
    inputs = {}
    for r in all_rows:
        h = int(r["phash"], 16)
        for before, group in hashes.items():
            if (h ^ before).bit_count() <= 2:
                union(r["capture_group"], group)
        hashes[h] = r["capture_group"]
        key = r["model_input_sha256"]
        if key in inputs:
            union(r["capture_group"], inputs[key])
        inputs[key] = r["capture_group"]
    for r in all_rows:
        r["group"] = find(r["capture_group"])
        if r["id"] in known:
            assert r["group"] == known[r["id"]]["group"], (
                "Recovered family differs from immutable source audit",
                r["id"],
            )
    admitted = [
        r for r in all_rows if r["id"] not in dev_ids and r["id"] not in excluded
    ]
    assert len(admitted) == 787
    assert {r["group"] for r in admitted} == {r["group"] for r in reserved}
    cold_path = p / "phase9_ad_source/sealed_geometric_manifest.json"
    cold = read(cold_path)
    assert len(cold) == 502 and all(
        r.get("file_sha256") and r.get("rgb32_sha256") and r.get("phash") for r in cold
    )
    cold_files = {r["file_sha256"] for r in cold}
    cold_rgb = {r["rgb32_sha256"] for r in cold}
    cold_ph = {int(r["phash"], 16) for r in cold}
    overlap = {
        r["group"]
        for r in admitted
        if r["file_sha256"] in cold_files
        or r["rgb32_sha256"] in cold_rgb
        or any((int(r["phash"], 16) ^ h).bit_count() <= 2 for h in cold_ph)
    }
    admitted = [r for r in admitted if r["group"] not in overlap]
    base = read(scene / "rows.json")
    base_groups = {r["group"] for r in base}
    assert not base_groups & {r["group"] for r in admitted}
    for r in admitted:
        r["split"] = (
            "val"
            if int(
                hashlib.sha256(("phase12-split503:" + r["group"]).encode()).hexdigest()[
                    :8
                ],
                16,
            )
            % 100
            < 20
            else "train"
        )
        r["retired_from_benchmark"] = True
    added = [
        dict(
            r,
            id="phase12:" + r["id"],
            set="training",
            box=None,
            view="full_scene",
            input_variant="full_roi",
            health_y=float(r["label"] == "Healthy"),
            label="Healthy" if r["label"] == "Healthy" else "Not_healthy_tomato",
            source_label=r["label"],
        )
        for r in admitted
    ]
    rows = base + added
    assert not (
        {r["group"] for r in rows if r["split"] == "train"}
        & {r["group"] for r in rows if r["split"] == "val"}
    )
    for split in ["train", "val"]:
        assert any(r["label"] == "Healthy" and r["split"] == split for r in added)
    pixels_path = root / "admitted_full_scene.npy"
    pixels = np.lib.format.open_memmap(
        pixels_path, mode="w+", dtype=np.float32, shape=(len(added), 224, 224, 3)
    )

    def prepare(r):
        with Image.open(r["path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        return classifier_letterbox_input(rgb, 224)[0]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(added), 128):
            pixels[start : start + 128] = np.stack(
                list(pool.map(prepare, added[start : start + 128]))
            )
    pixels.flush()
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    parts = read(scene / "pixels.json")["center"]
    (work / "pixels.json").write_text(
        json.dumps(dict(center=parts + [str(pixels_path)]), indent=2)
    )
    (root / "admission_manifest.json").write_text(json.dumps(admitted, indent=2))
    (root / "eval_rows.json").write_text("{}")
    shutil.copy2(
        p / "phase10_leaf/mobileclip_source.json", root / "mobileclip_source.json"
    )
    report = dict(
        base_scene_rows_sha256=digest(scene / "rows.json"),
        consumed_manifest_sha256=digest(old / "reserved_manifest.json"),
        consumed_photos=246,
        related_family_photos=len(admitted),
        related_groups=len({r["group"] for r in admitted}),
        original_quarantined_ids_preserved=len(excluded),
        cold_ad_manifest_sha256=digest(cold_path),
        cold_ad_overlap_families_excluded=sorted(overlap),
        cold_ad_diagnostic_predictions_opened=False,
        counts=dict(
            collections.Counter(
                r["split"] + ":" + r["source"] + ":" + str(r["health_y"]) for r in added
            )
        ),
        rows=len(rows),
        rows_sha256=digest(work / "rows.json"),
        scope="New image-scoped training/development revision. The consumed 246-photo cohort and all related families are retired as regression/acceptance benchmarks for this revision. No independent claim may use them. Original development split and quarantines preserved; cold AD remains independent and unopened.",
    )
    complete.write_text(json.dumps(report, indent=2))
    (root / "RETIRED_BENCHMARK.json").write_text(
        json.dumps(
            dict(manifest=str(old / "reserved_manifest.json"), reason=report["scope"]),
            indent=2,
        )
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
