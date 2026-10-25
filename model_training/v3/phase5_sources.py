"""Admit opened sources and prospectively reserve never-opened capture groups.

TOM2024 generic virosis remains binary diseased; fruit classes are excluded.
No model predictions are computed here. Held groups and all their other views
are forbidden in fitting, calibration and checkpoint/policy selection.
"""

from pathlib import Path
import argparse, collections, concurrent.futures, csv, hashlib, json, re, time
import requests
from campaign_data import profile
import campaign_download
from phase4_leaf_source import MAPPING as PAPER

TOM = {
    "healthy_leaf": "Healthy",
    "virosis_d": "Diseased_unspecified",
    "Mite_d": "Other",
}


def tom_group(filename):
    times = re.findall(r"(?<!\d)(1\d{12})(?!\d)", filename)
    return "tom2024:burst" + (
        str(int(times[-1]) // 180000) if times else filename.split("_")[0]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / "phase5"
    out.mkdir(exist_ok=True)
    if (out / "sources_complete.json").exists():
        return
    metadata = json.loads((p / "tom2024_acceptance_files.json").read_text())
    csv_rows = list(
        csv.DictReader((p / "new_field_metadata.csv").open(encoding="utf-8-sig"))
    )
    paper_lookup = {
        (r["Class_Label"], r["File_Name"]): r
        for r in csv_rows
        if r["Class_Label"] in PAPER
    }
    seen = {
        tom_group(r["path"].split("/")[-1])
        for r in json.loads((p / "phase4_tom2024/manifest.json").read_text())
    }
    for file in (p / "sources/swtt5wy2pr_v1").glob("*/*.png"):
        row = paper_lookup[(file.parent.name, file.name)]
        seen.add(
            "paper:"
            + row["Class_Label"]
            + ":block"
            + str(int(re.search(r"(\d+)$", row["Object_ID"]).group()) // 5)
        )
    groups = collections.defaultdict(set)
    for category, items in metadata.items():
        if category not in TOM:
            continue
        for item in items:
            groups[tom_group(item["filename"])].add(category)
    reserved = set()
    for category, count in [("healthy_leaf", 96), ("virosis_d", 64), ("Mite_d", 24)]:
        eligible = sorted(
            [
                g
                for g, classes in groups.items()
                if category in classes and g not in seen and g not in reserved
            ],
            key=lambda g: hashlib.sha256(("reserve103:" + g).encode()).hexdigest(),
        )
        assert len(eligible) >= count, (category, len(eligible))
        reserved.update(eligible[:count])
    for category, count in [
        ("Early Blight Leaf", 16),
        ("Late Blight leaf", 16),
        ("Bushy Stunt Virus", 12),
        ("Leaf Curl Virus", 12),
    ]:
        available = {
            int(re.search(r"(\d+)$", r["Object_ID"]).group()) // 5
            for r in csv_rows
            if r["Class_Label"] == category
        }
        keys = ["paper:" + category + ":block" + str(g) for g in available]
        eligible = sorted(
            [g for g in keys if g not in seen],
            key=lambda g: hashlib.sha256(("reserve103:" + g).encode()).hexdigest(),
        )
        assert len(eligible) >= count
        reserved.update(eligible[:count])
    reservation = dict(
        seed=103,
        groups=sorted(reserved),
        previously_opened_groups=sorted(seen),
        source_file_metadata_sha256=hashlib.sha256(
            (p / "tom2024_acceptance_files.json").read_bytes()
        ).hexdigest(),
        paper_csv_sha256=hashlib.sha256(
            (p / "new_field_metadata.csv").read_bytes()
        ).hexdigest(),
        history="Previous phase-four cohorts opened and failed the release gate; now admitted to development. Reserved groups have no overlap with any of those opened capture groups. No predictions computed on this reservation.",
    )
    frozen = out / "capture_reservation.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == reservation
    else:
        frozen.write_text(json.dumps(reservation, indent=2))
    archive = next(
        x
        for x in json.loads((p / "swtt5wy2pr_root_metadata.json").read_text())
        if x["filename"] == "Resize Dataset.zip"
    )
    campaign_download.ROOT = p
    campaign_download.download(("paper_full", archive))
    jobs = []
    for category, items in metadata.items():
        if category not in TOM:
            continue
        for item in items:
            jobs.append(
                ("tom2024", category, TOM[category], tom_group(item["filename"]), item)
            )
    for row in csv_rows:
        category = row["Class_Label"]
        if category not in PAPER:
            continue
        group = (
            "paper:"
            + category
            + ":block"
            + str(int(re.search(r"(\d+)$", row["Object_ID"]).group()) // 5)
        )
        jobs.append(("swtt5wy2pr", category, PAPER[category], group, row))

    def acquire(job):
        source, category, label, group, item = job
        if source == "tom2024":
            path = p / "sources/tom2024" / category / item["filename"]
            assert path.resolve().is_relative_to((p / "sources/tom2024").resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = item["content_details"]["sha256_hash"]
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                for attempt in range(4):
                    try:
                        r = requests.get(
                            item["content_details"]["download_url"], timeout=(20, 60)
                        )
                        r.raise_for_status()
                        assert hashlib.sha256(r.content).hexdigest() == expected
                        path.write_bytes(r.content)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 + attempt)
            doi = "10.17632/3d4yg89rtr.1"
        else:
            path = (
                p
                / "sources/paper_full/images/Resize Dataset"
                / category
                / item["File_Name"]
            )
            doi = "10.17632/swtt5wy2pr.1"
        row = dict(
            id="phase5:" + source + ":" + category + ":" + path.name,
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
        info, _ = profile(row)
        row.update(info)
        return row

    rows = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        for i, row in enumerate(pool.map(acquire, jobs)):
            rows.append(row)
            if i % 250 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="source_profiles",
                            photos=i,
                            total=len(jobs),
                            seconds=time.time() - started,
                        )
                    ),
                    flush=True,
                )
    parents = {r["group"]: r["group"] for r in rows}

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(x, y):
        a, b = find(x), find(y)
        if a != b:
            parents[max(a, b)] = min(a, b)

    hashes = {}
    input_labels = {}
    contradictions = set()
    for row in rows:
        h = int(row["phash"], 16)
        for previous, group in hashes.items():
            if (h ^ previous).bit_count() <= 2:
                union(row["group"], group)
        hashes[h] = row["group"]
        key = row["model_input_sha256"]
        if key in input_labels:
            old_label, old_group = input_labels[key]
            union(row["group"], old_group)
            if old_label != row["label"] and "Diseased_unspecified" not in (
                old_label,
                row["label"],
            ):
                contradictions.update([row["group"], old_group])
            if (old_label == "Healthy") != (row["label"] == "Healthy"):
                contradictions.update([row["group"], old_group])
        else:
            input_labels[key] = (row["label"], row["group"])
    prior = []
    for name in [
        "original_quality_profile.json",
        "new_development_manifest.json",
        "locked_new_source_manifest.json",
        "locked_binary_manifest.json",
        "phase2_source_manifest.json",
    ]:
        prior += json.loads((p / name).read_text())
    prior_sha = {r.get("file_sha256") for r in prior}
    prior_input = {r.get("model_input_sha256") for r in prior}
    prior_phash = {int(r["phash"], 16) for r in prior if r.get("phash")}
    near_prior = set()
    for row in rows:
        if (
            row["file_sha256"] in prior_sha
            or row["model_input_sha256"] in prior_input
            or any((int(row["phash"], 16) ^ h).bit_count() <= 2 for h in prior_phash)
        ):
            near_prior.add(find(row["group"]))
    held = {find(g) for g in reserved}
    opened = {find(g) for g in seen if g in parents}
    bad = near_prior | {find(g) for g in contradictions} | (held & opened)
    eligible = []
    excluded = []
    for row in rows:
        component = find(row["group"])
        row["group"] = component
        if component in bad:
            excluded.append(
                dict(
                    id=row["id"],
                    group=component,
                    reason="Prior overlap, contradictory identical input, or reserved group connected to opened data",
                )
            )
            continue
        row["split"] = (
            "test"
            if component in held
            else (
                "val"
                if int(
                    hashlib.sha256(("split103:" + component).encode()).hexdigest()[:8],
                    16,
                )
                % 100
                < 25
                else "train"
            )
        )
        eligible.append(row)
    trainval = [r for r in eligible if r["split"] != "test"]
    test_groups = collections.defaultdict(list)
    for row in eligible:
        if row["split"] == "test":
            test_groups[(row["group"], row["label"])].append(row)
    test = [
        min(
            items,
            key=lambda r: hashlib.sha256(("test103:" + r["id"]).encode()).hexdigest(),
        )
        for items in test_groups.values()
    ]
    assert (
        sum(r["label"] == "Healthy" for r in test) >= 60
    ), "Insufficient untouched Healthy capture groups"
    assert not ({r["group"] for r in trainval} & {r["group"] for r in test})
    for key in ["file_sha256", "model_input_sha256"]:
        assert not ({r[key] for r in trainval} & {r[key] for r in test})
    (out / "development_manifest.json").write_text(
        json.dumps(trainval, indent=2) + "\n"
    )
    (out / "reserved_manifest.json").write_text(json.dumps(test, indent=2) + "\n")
    (out / "exclusions.json").write_text(json.dumps(excluded, indent=2))
    report = dict(
        schema=1,
        source_photos=len(rows),
        development=len(trainval),
        development_counts=dict(
            collections.Counter(
                r["split"] + ":" + r["source"] + ":" + r["label"] for r in trainval
            )
        ),
        reserved=len(test),
        reserved_counts=dict(
            collections.Counter(r["source"] + ":" + r["label"] for r in test)
        ),
        quarantined=len(excluded),
        reserved_groups=len({r["group"] for r in test}),
        reserved_manifest_sha256=hashlib.sha256(
            (out / "reserved_manifest.json").read_bytes()
        ).hexdigest(),
        development_manifest_sha256=hashlib.sha256(
            (out / "development_manifest.json").read_bytes()
        ).hexdigest(),
        archive_sha256_verified=archive["content_details"]["sha256_hash"],
        reserved_predictions_opened=False,
        scope="Prospectively reserved same-source capture groups, not a claim of new-source independence. All related photographs are excluded from fitting and calibration.",
        annotation_semantics="Image classification only. Generic virosis is binary diseased. Fruit classes excluded. No disease-region boxes used as leaf extents.",
    )
    (out / "sources_complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
