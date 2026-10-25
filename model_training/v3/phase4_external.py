"""Seal a fresh TOM2024 cohort from Category A English files, without predictions.

Excludes Category B augmentation and the duplicate French copy. Timestamp/burst
groups are inferred, not verified plant IDs. Generic virosis and mite remain
unsupported; neither is assigned a more specific disease name.
"""

from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, re, time
import requests
from campaign_data import profile

MAPPING = {
    "healthy_leaf": "Healthy",
    "virosis_d": "Diseased_unspecified",
    "Mite_d": "Other",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / "phase4_tom2024"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Cohort already sealed; do not change its membership")
    meta = json.loads((p / "tom2024_acceptance_files.json").read_text())
    selected = []
    for category, items in meta.items():
        if category not in MAPPING:
            continue
        groups = collections.defaultdict(list)
        for item in items:
            timestamps = re.findall(r"(?<!\d)(1\d{12})(?!\d)", item["filename"])
            # Group photographs in three-minute publication-filename bursts.
            group = (
                str(int(timestamps[-1]) // 180000)
                if timestamps
                else item["filename"].split("_")[0]
            )
            groups[group].append(item)
        jobs = sorted(
            groups.items(),
            key=lambda x: hashlib.sha256(
                ("tom73:" + category + ":" + x[0]).encode()
            ).hexdigest(),
        )[:100]
        for group, files in jobs:
            item = min(
                files, key=lambda f: hashlib.sha256(f["filename"].encode()).hexdigest()
            )
            selected.append((category, group, item))

    def acquire(job):
        category, group, item = job
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
        row = dict(
            id="tom2024:" + category + ":" + item["filename"],
            path=str(path),
            source="tom2024",
            species="tomato",
            label=MAPPING[category],
            raw_label=category,
            box=None,
            group="tom2024:" + category + ":burst" + group,
            split="external_acceptance",
            source_version="10.17632/3d4yg89rtr.1",
            source_url="https://data.mendeley.com/datasets/3d4yg89rtr/1",
            licence="CC BY 4.0",
        )
        info, _ = profile(row)
        row.update(info)
        return row

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(acquire, selected))
    prior = []
    for name in [
        "original_quality_profile.json",
        "new_development_manifest.json",
        "locked_new_source_manifest.json",
        "locked_binary_manifest.json",
        "phase2_source_manifest.json",
        "phase2/slif_acceptance_manifest.json",
        "phase3_detector/source_rows.json",
    ]:
        prior += json.loads((p / name).read_text())
    sha = {r.get("file_sha256") for r in prior}
    inputs = {r.get("model_input_sha256") for r in prior}
    phash = {int(r["phash"], 16) for r in prior if r.get("phash")}
    eligible = []
    excluded = []
    for row in rows:
        reason = None
        if row["file_sha256"] in sha or row["model_input_sha256"] in inputs:
            reason = "exact_prior_overlap"
        elif any((int(row["phash"], 16) ^ h).bit_count() <= 2 for h in phash):
            reason = "close_prior_overlap"
        elif any(
            (int(row["phash"], 16) ^ int(r["phash"], 16)).bit_count() <= 2
            for r in eligible
        ):
            reason = "close_selected_duplicate"
        if reason:
            excluded.append(dict(id=row["id"], reason=reason))
        else:
            eligible.append(row)
    (out / "manifest.json").write_text(json.dumps(eligible, indent=2) + "\n")
    audit = dict(
        doi="10.17632/3d4yg89rtr.1",
        licence="CC BY 4.0",
        population="Category A / CATA-English / tomato_diseases; Category B and French duplicates excluded",
        metadata_sha256=hashlib.sha256(
            (p / "tom2024_acceptance_files.json").read_bytes()
        ).hexdigest(),
        manifest_sha256=hashlib.sha256(
            (out / "manifest.json").read_bytes()
        ).hexdigest(),
        counts=dict(collections.Counter(r["label"] for r in eligible)),
        selected=len(rows),
        eligible=len(eligible),
        excluded=excluded,
        predictions_opened=False,
        training_use=False,
        grouping="One file per inferred three-minute filename burst per class; not verified plant IDs. Close input/perceptual duplicates quarantined.",
        seconds=time.time() - started,
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
