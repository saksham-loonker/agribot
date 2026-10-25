"""Acquire and seal a further external source without model predictions.

Publisher: https://data.mendeley.com/datasets/jttrv2w27r/1 (CC BY 4.0).
Only evaluation membership is created. This source must never be fitted after
its acceptance results are opened without obtaining another untouched cohort.
"""

from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, re, time
import requests
from campaign_data import profile

LABELS = {
    "Bacteiral Spot": "Other",
    "Early Blight": "Early_blight",
    "Healthy": "Healthy",
    "Late Blight": "Late_blight",
    "Leaf Miner Flies": "Leaf Miner",
    "Magnesium Deficiency": "Magnesium Deficiency",
    "Yellow Leaf Curl Virus": "Other",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--full-development",
        action="store_true",
        help="Admit the now-opened source for a later training phase; preserve the former acceptance artifacts",
    )
    a = parser.parse_args()
    root = Path(a.root)
    destination = root / "sources/fresh_classification"
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "fresh_classification_files.json").read_text())
    jobs = []
    for category, files in metadata.items():
        # Deterministic source-balanced sample before any prediction, maximum 120/class.
        files = sorted(
            files,
            key=lambda f: hashlib.sha256(
                ("fresh29:" + f["filename"]).encode()
            ).hexdigest(),
        )
        if not a.full_development:
            files = files[:120]
        jobs.extend((category, item) for item in files)

    def acquire(job):
        category, item = job
        file = destination / category / item["filename"]
        if not file.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unsafe publisher filename")
        file.parent.mkdir(exist_ok=True)
        expected = item["content_details"]["sha256_hash"]
        if (
            not file.exists()
            or hashlib.sha256(file.read_bytes()).hexdigest() != expected
        ):
            for attempt in range(4):
                try:
                    response = requests.get(
                        item["content_details"]["download_url"], timeout=(20, 60)
                    )
                    response.raise_for_status()
                    assert hashlib.sha256(response.content).hexdigest() == expected
                    file.write_bytes(response.content)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1 + attempt)
        number = int(re.search(r"\d+", file.stem).group())
        row = dict(
            id="fresh_jttrv:" + category + "/" + file.name,
            path=str(file),
            source="bangladesh_classification",
            species="tomato",
            label=LABELS[category],
            raw_label=category,
            box=None,
            split="external_final_acceptance",
            group="fresh_jttrv:" + category + ":block" + str(number // 8),
            source_version="10.17632/jttrv2w27r.1",
            source_url="https://data.mendeley.com/datasets/jttrv2w27r/1",
            licence="CC BY 4.0",
        )
        info, _ = profile(row)
        row.update(info)
        assert info["file_sha256"] == expected
        return row

    rows = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        for n, row in enumerate(pool.map(acquire, jobs)):
            rows.append(row)
            if n % 50 == 0:
                print(
                    json.dumps(
                        dict(
                            downloaded=n,
                            total=len(jobs),
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
    prior = []
    for name in [
        "original_quality_profile.json",
        "new_development_manifest.json",
        "locked_new_source_manifest.json",
        "locked_binary_manifest.json",
    ]:
        prior.extend(json.loads((root / name).read_text()))
    files = {r["file_sha256"] for r in prior}
    inputs = {r["model_input_sha256"] for r in prior}
    hashes = {int(r["phash"], 16) for r in prior}
    labels = collections.defaultdict(set)
    for row in rows:
        labels[row["model_input_sha256"]].add(row["label"])
    eligible = []
    excluded = []
    seen = set()
    blocked_groups = set()
    for row in rows:
        reason = None
        if row["file_sha256"] in files or row["model_input_sha256"] in inputs:
            reason = "exact_prior_overlap"
        elif any((int(row["phash"], 16) ^ h).bit_count() <= 2 for h in hashes):
            reason = "near_prior_overlap"
        elif len(labels[row["model_input_sha256"]]) > 1:
            reason = "conflicting_source_labels"
        if reason:
            excluded.append(dict(id=row["id"], group=row["group"], reason=reason))
            blocked_groups.add(row["group"])
    for row in rows:
        if row["group"] in blocked_groups:
            continue
        if row["model_input_sha256"] in seen:
            excluded.append(dict(id=row["id"], reason="duplicate_model_input"))
            continue
        eligible.append(row)
        seen.add(row["model_input_sha256"])
    if a.full_development:
        # Merge inferred capture families and very close internal views before
        # assigning the new source to development fitting and validation.
        parent = list(range(len(eligible)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(i)] = find(j)

        family = {}
        for i, row in enumerate(eligible):
            if row["group"] in family:
                union(i, family[row["group"]])
            family[row["group"]] = i
        phash = [int(r["phash"], 16) for r in eligible]
        for i, value in enumerate(phash):
            for j in range(i):
                if (value ^ phash[j]).bit_count() <= 2:
                    union(i, j)
        clusters = collections.defaultdict(list)
        for i, row in enumerate(eligible):
            clusters[find(i)].append(row)
        for members in clusters.values():
            key = min(r["group"] for r in members)
            group = "phase2:" + hashlib.sha256(key.encode()).hexdigest()[:20]
            split = (
                "train"
                if int(
                    hashlib.sha256(("phase2_29:" + group).encode()).hexdigest()[:8], 16
                )
                % 100
                < 75
                else "val"
            )
            for row in members:
                row.update(group=group, split=split)
    manifest = root / (
        "phase2_source_manifest.json"
        if a.full_development
        else "fresh_acceptance_manifest.json"
    )
    manifest.write_text(json.dumps(eligible, indent=2) + "\n")
    audit = dict(
        source="10.17632/jttrv2w27r.1",
        publisher_contributors=[
            "Nure Alam Siddiki Rachin",
            "Nasim Mahmud Likhon",
            "Md Abdullah Hel Baki",
        ],
        sample_policy=(
            "Full source admitted for phase-two development after the first acceptance opening; all prior acceptance artifacts retained"
            if a.full_development
            else "SHA-256(fresh29:filename) order, first 120 files per original category; no predictions used"
        ),
        downloaded=len(rows),
        eligible=len(eligible),
        excluded=excluded,
        blocked_inferred_capture_groups=len(blocked_groups),
        counts=dict(collections.Counter(r["label"] for r in eligible)),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(
            (root / "fresh_classification_files.json").read_bytes()
        ).hexdigest(),
        limitations=[
            "Publisher labels are not expert-adjudicated in this campaign",
            "Sequential filename groups are inferred; verified plant IDs are unavailable",
            "No detected overlap with campaign data does not prove absence from every historical original-checkpoint source",
        ],
        predictions_opened=a.full_development,
    )
    (
        root
        / (
            "phase2_source_audit.json"
            if a.full_development
            else "fresh_acceptance_audit.json"
        )
    ).write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
