"""Seal a further field source: one published view per SLIF capture family.

Kaggle version 1 supplies augmented/cropped views, not its raw source photos.
These are explicitly reported as such; related views never become independent
acceptance cases. No source image is fitted or used for model selection.
"""

from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, re, time, urllib.parse
import requests
from campaign_data import profile

MAPPING = {
    "Bacterial_Spot": "Other",
    "Early_Blight": "Early_blight",
    "Healthy": "Healthy",
    "Late_Blight": "Late_blight",
    "Mosaic": "Other",
    "Powdery_Mildew": "Other",
    "Septoria": "Other",
    "Wilt": "Other",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    a = p.parse_args()
    campaign = Path(a.campaign)
    root = campaign / "phase2"
    root.mkdir(exist_ok=True)
    destination = campaign / "sources/slif_v1"
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        r
        for r in json.loads((campaign / "slif_all_files.json").read_text())
        if r["name"].startswith("Phase_I-Dataset/")
    ]
    assert len(files) == 2670, "The Phase-I population must be complete"
    groups = collections.defaultdict(list)
    for file in files:
        groups[re.match(r"t\d+", Path(file["name"]).name).group()].append(file)
    conflicted = {
        k: sorted({r["name"].split("/")[1] for r in items})
        for k, items in groups.items()
        if len({r["name"].split("/")[1] for r in items}) > 1
    }
    selected = []
    for category in MAPPING:
        candidates = [
            (key, items)
            for key, items in groups.items()
            if key not in conflicted and items[0]["name"].split("/")[1] == category
        ]
        candidates.sort(
            key=lambda item: hashlib.sha256(("slif29:" + item[0]).encode()).hexdigest()
        )
        for key, items in candidates[:120]:
            # Prefer a full-frame stem over an explicitly numbered leaf crop;
            # choose a deterministic published augmentation, never a prediction.
            items.sort(
                key=lambda r: (
                    (
                        0
                        if re.match(
                            key + r"_(?:jpg|jpeg|png)\.rf\.", Path(r["name"]).name, re.I
                        )
                        else 1
                    ),
                    len(Path(r["name"]).name),
                    hashlib.sha256(r["name"].encode()).hexdigest(),
                )
            )
            selected.append((key, category, items[0]))

    def acquire(job):
        key, category, item = job
        file = destination / item["name"]
        if not file.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unsafe publisher path")
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists() or file.stat().st_size != item["totalBytes"]:
            url = (
                "https://www.kaggle.com/api/v1/datasets/download/romiyalgeorge/slif-tomato-dataset/"
                + urllib.parse.quote(item["name"], safe="")
            )
            for attempt in range(4):
                try:
                    response = requests.get(
                        url, params={"datasetVersionNumber": 1}, timeout=(20, 60)
                    )
                    response.raise_for_status()
                    assert len(response.content) == item["totalBytes"]
                    assert response.content.startswith(
                        b"\xff\xd8"
                    ), "Expected the published JPEG, not an HTML response or archive"
                    file.write_bytes(response.content)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 + attempt)
        row = dict(
            id="slif_v1:" + key,
            path=str(file),
            label=MAPPING[category],
            raw_label=category,
            source="slif_v1",
            species="tomato",
            box=None,
            split="external_acceptance",
            group="slif_v1:" + key,
            source_url="https://www.kaggle.com/datasets/romiyalgeorge/slif-tomato-dataset",
            source_version="Kaggle dataset 8911633 version 1",
            doi="10.34740/kaggle/ds/8911633",
            licence="CC BY 4.0",
            published_file=item["name"],
            published_bytes=item["totalBytes"],
        )
        info, _ = profile(row)
        row.update(info)
        return row

    rows = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for i, row in enumerate(pool.map(acquire, selected)):
            rows.append(row)
            if i % 25 == 0:
                print(
                    json.dumps(
                        dict(
                            downloaded=i,
                            total=len(selected),
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
        "phase2_source_manifest.json",
    ]:
        prior.extend(json.loads((campaign / name).read_text()))
    byte_hashes = {r["file_sha256"] for r in prior}
    input_hashes = {r["model_input_sha256"] for r in prior}
    phashes = {int(r["phash"], 16) for r in prior}
    eligible = []
    excluded = []
    seen = set()
    for row in rows:
        reason = None
        if (
            row["file_sha256"] in byte_hashes
            or row["model_input_sha256"] in input_hashes
        ):
            reason = "exact_prior_overlap"
        elif any((int(row["phash"], 16) ^ h).bit_count() <= 2 for h in phashes):
            reason = "near_prior_overlap"
        elif row["model_input_sha256"] in seen:
            reason = "duplicate_selected_input"
        if reason:
            excluded.append(dict(id=row["id"], reason=reason))
        else:
            eligible.append(row)
            seen.add(row["model_input_sha256"])
    manifest = root / "slif_acceptance_manifest.json"
    manifest.write_text(json.dumps(eligible, indent=2) + "\n")
    audit = dict(
        source="SLIF-Tomato Kaggle version 1",
        doi="10.34740/kaggle/ds/8911633",
        licence="CC BY 4.0",
        metadata_sha256=hashlib.sha256(
            (campaign / "slif_all_files.json").read_bytes()
        ).hexdigest(),
        published_phase1_files=len(files),
        inferred_capture_families=len(groups),
        cross_class_families_quarantined=conflicted,
        selected=len(selected),
        eligible=len(eligible),
        counts=dict(collections.Counter(r["raw_label"] for r in eligible)),
        excluded=excluded,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        predictions_opened=False,
        limitations=[
            "Published augmented/cropped views, not raw camera photos",
            "One fixed published view per capture family; inferred capture IDs are not verified plant IDs",
            "Kaggle supplies file sizes but no publisher SHA-256; HTTPS download size and local content SHA-256 are recorded",
            "No detected overlap does not prove absence from every historical original-model source",
        ],
    )
    (root / "slif_acceptance_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
