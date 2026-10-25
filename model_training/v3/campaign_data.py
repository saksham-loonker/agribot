"""Audit new sources and seal development/acceptance membership before fitting.

Uses raw Pakistan images, never its preaugmented train/test folders. TLID filename
families, exact RGB duplicates and conservative perceptual
duplicates are kept together. These are inferred photo groups, not plant IDs.
This initial audit reserves the Bangladesh binary source. The later prospective
author-training admission is separately recorded by campaign_cnn_data.
"""

from pathlib import Path
import collections, concurrent.futures, hashlib, json, re
import numpy as np
from PIL import Image
from scipy.fft import dctn
from app_eval import canon

ROOT = Path("/home/genesis/agribot-data/recognition_campaign_20261024")
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def profile(record):
    with Image.open(record["path"]) as im:
        image = np.asarray(im.convert("RGB"))
        width, height = im.size
    tensor = canon.classifier_input(image, 32, record.get("box"))[0]
    exact_input = canon.classifier_input(image, 224, record.get("box"))
    rgb = (
        np.clip((tensor * canon.STD + canon.MEAN) * 255, 0, 255)
        .round()
        .astype(np.uint8)
    )
    gray = rgb.astype(np.float32).mean(2)
    options = []
    for flipped in [False, True]:
        for rotation in range(4):
            oriented = np.rot90(np.fliplr(gray) if flipped else gray, rotation)
            low = dctn(oriented, type=2, norm="ortho")[:8, :8].flatten()
            bits = low > np.median(low[1:])
            bits[0] = False
            value = int.from_bytes(np.packbits(bits).tobytes(), "big")
            color = np.rot90(np.fliplr(rgb) if flipped else rgb, rotation)
            options.append((value, color))
    value, color = min(options, key=lambda v: v[0])
    return (
        dict(
            width=width,
            height=height,
            rgb32_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            model_input_sha256=hashlib.sha256(exact_input.tobytes()).hexdigest(),
            file_sha256=hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest(),
            phash=f"{value:016x}",
        ),
        color,
    )


def main():
    originals = json.loads(
        Path("/home/genesis/agribot-data/work/data_manifest.json").read_text()
    )
    new = []
    for file in sorted((ROOT / "sources/tlid/images").rglob("*")):
        if file.suffix.lower() not in IMAGE_TYPES:
            continue
        category = file.parent.name
        label = (
            "Healthy"
            if category.startswith("0-")
            else (
                "Leaf Miner"
                if category.startswith("1-")
                else "Mixed" if category.startswith(("4-", "5-")) else "Other"
            )
        )
        match = re.match(r"IMG_(\d+)_", file.stem)
        if match is None:
            raise ValueError(file)
        new.append(
            dict(
                id="tlid:" + str(file.relative_to(ROOT / "sources/tlid/images")),
                path=str(file),
                label=label,
                raw_label=category,
                source="tlid",
                species="tomato",
                box=None,
                group=f"tlid:{category}:photo_{int(match[1])}",
                source_version="10.17632/kt64b2kh89.2",
                licence="CC BY 4.0",
            )
        )
    augmented_excluded = 0
    for file in sorted((ROOT / "sources/pakistan/images").rglob("*")):
        if file.suffix.lower() not in IMAGE_TYPES:
            continue
        if "Dataset (raw)" not in file.parts:
            augmented_excluded += 1
            continue
        category = file.parent.name
        key = category.lower()
        label = (
            "Healthy"
            if "healthy" in key
            else (
                "Early_blight"
                if "early" in key
                else "Late_blight" if "late" in key else "Other"
            )
        )
        family = re.sub(r"(?i)(?:_?aug\d*)+.*$", "", file.stem)
        new.append(
            dict(
                id="pakistan:"
                + str(file.relative_to(ROOT / "sources/pakistan/images")),
                path=str(file),
                label=label,
                raw_label=category,
                source="pakistan",
                species="tomato",
                box=None,
                group=f"pakistan:{family}",
                source_version="10.17632/3mbnb82mxd.2",
                licence="CC BY 4.0",
            )
        )
    binary = []
    for file in sorted((ROOT / "sources/binary/images").rglob("*")):
        if file.suffix.lower() not in IMAGE_TYPES:
            continue
        if file.stem.startswith("H"):
            label = "Healthy"
        elif file.stem.startswith("D"):
            label = "Diseased_unspecified"
        else:
            raise ValueError(file)
        binary.append(
            dict(
                id="bangladesh:"
                + str(file.relative_to(ROOT / "sources/binary/images")),
                path=str(file),
                label=label,
                source="bangladesh_binary",
                species="tomato",
                box=None,
                group="bangladesh:"
                + str(file.relative_to(ROOT / "sources/binary/images")),
                split="locked_external",
                source_version="10.17632/rnbsw72zb5.1",
                licence="CC BY 4.0",
            )
        )
    all_records = originals + new + binary
    metadata = []
    pixels = []
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for start in range(0, len(all_records), 256):
            for info, color in executor.map(profile, all_records[start : start + 256]):
                metadata.append(info)
                pixels.append(color)
            if start % 2048 == 0:
                print(
                    "profile",
                    min(start + 256, len(all_records)),
                    len(all_records),
                    flush=True,
                )
    parent = list(range(len(all_records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[max(i, j)] = min(i, j)

    groups = {}
    exact = {}
    buckets = collections.defaultdict(list)
    near_edges = []
    for i, (record, meta) in enumerate(zip(all_records, metadata)):
        group = record["group"]
        if group in groups:
            union(i, groups[group])
        else:
            groups[group] = i
        value = int(meta["phash"], 16)
        candidates = set()
        for segment in range(4):
            candidates.update(buckets[(segment, (value >> (16 * segment)) & 65535)])
        for j in candidates:
            if (value ^ int(metadata[j]["phash"], 16)).bit_count() <= 2 and np.abs(
                pixels[i].astype(float) - pixels[j].astype(float)
            ).mean() / 255 < 0.035:
                union(i, j)
                near_edges.append((i, j))
        key = meta["model_input_sha256"]
        if key in exact:
            union(i, exact[key])
        else:
            exact[key] = i
        for segment in range(4):
            buckets[(segment, (value >> (16 * segment)) & 65535)].append(i)
    clusters = collections.defaultdict(list)
    for i in range(len(all_records)):
        clusters[find(i)].append(i)
    exclusions = []
    development = []
    locked = []
    conflicts = []
    mixed = []
    for indices in clusters.values():
        old = [i for i in indices if i < len(originals)]
        reserved = [i for i in indices if i >= len(originals) + len(new)]
        additions = [
            i for i in indices if len(originals) <= i < len(originals) + len(new)
        ]
        if reserved:
            # Keep the external source intact but prevent any near copy entering training.
            for i in additions:
                exclusions.append(
                    dict(
                        id=all_records[i]["id"],
                        reason="overlap_with_reserved_binary_source",
                    )
                )
            continue
        if old:
            for i in additions:
                exclusions.append(
                    dict(id=all_records[i]["id"], reason="overlap_with_prior_source")
                )
            continue
        exact_labels = collections.defaultdict(list)
        for i in additions:
            exact_labels[metadata[i]["model_input_sha256"]].append(i)
        ambiguous = set()
        for same_pixels in exact_labels.values():
            if (
                len(
                    {
                        all_records[i]["label"]
                        for i in same_pixels
                        if all_records[i]["label"] != "Mixed"
                    }
                )
                > 1
            ):
                conflicts.append([all_records[i]["id"] for i in same_pixels])
                ambiguous.update(same_pixels)
        for i in ambiguous:
            exclusions.append(
                dict(
                    id=all_records[i]["id"],
                    reason="identical_model_input_conflicting_labels",
                )
            )
        additions = [i for i in additions if i not in ambiguous]
        if not additions:
            continue
        key = min(all_records[i]["group"] for i in additions)
        percentile = (
            int(hashlib.sha256(("campaign17:" + key).encode()).hexdigest()[:8], 16)
            % 100
        )
        split = (
            "train"
            if percentile < 70
            else "val" if percentile < 85 else "locked_new_source"
        )
        cluster = "campaign:" + hashlib.sha256(key.encode()).hexdigest()[:16]
        seen = set()
        for i in additions:
            r = dict(all_records[i])
            r["original_group"] = r["group"]
            r["group"] = cluster
            r["split"] = split
            r.update(metadata[i])
            if metadata[i]["file_sha256"] in seen:
                exclusions.append(
                    dict(id=r["id"], reason="duplicate_file_within_group")
                )
                continue
            seen.add(metadata[i]["file_sha256"])
            if r["label"] == "Mixed":
                mixed.append(r)
            else:
                (locked if split == "locked_new_source" else development).append(r)
    # Source-photo identity matters even when a prior annotated crop differs.
    prior_files = {m["file_sha256"] for m in metadata[: len(originals)]}
    overlapping_groups = {
        r["group"]
        for r in development + locked + mixed
        if r["file_sha256"] in prior_files
    }
    for r in development + locked + mixed:
        if r["group"] in overlapping_groups:
            exclusions.append(dict(id=r["id"], reason="source_photo_identity_overlap"))
    development = [r for r in development if r["group"] not in overlapping_groups]
    locked = [r for r in locked if r["group"] not in overlapping_groups]
    mixed = [r for r in mixed if r["group"] not in overlapping_groups]
    old_profiles = [
        dict(id=r["id"], split=r["split"], label=r["label"], **m)
        for r, m in zip(originals, metadata)
    ]
    external = [
        dict(r, **metadata[len(originals) + len(new) + i]) for i, r in enumerate(binary)
    ]
    files = {
        "new_development_manifest.json": development,
        "locked_new_source_manifest.json": locked,
        "locked_binary_manifest.json": external,
        "source_exclusions.json": exclusions,
        "original_quality_profile.json": old_profiles,
        "mixed_stress_manifest.json": mixed,
    }
    assert not (
        {r["group"] for r in development if r["split"] == "train"}
        & {r["group"] for r in development if r["split"] == "val"}
    )
    assert not ({r["group"] for r in development} & {r["group"] for r in locked})
    for name, data in files.items():
        (ROOT / name).write_text(json.dumps(data, indent=2) + "\n")
    summary = dict(
        original_records=len(originals),
        new_records=len(new),
        reserved_binary=len(binary),
        augmented_pakistan_excluded=augmented_excluded,
        near_duplicate_edges=len(near_edges),
        mixed_disease_records_excluded_from_single_class_fitting=len(mixed),
        conflicting_label_clusters=conflicts,
        exclusions_by_reason=dict(collections.Counter(x["reason"] for x in exclusions)),
        development_counts={
            str(k): v
            for k, v in collections.Counter(
                (r["source"], r["split"], r["label"]) for r in development
            ).items()
        },
        locked_counts={
            str(k): v
            for k, v in collections.Counter(
                (r["source"], r["label"]) for r in locked
            ).items()
        },
        sealed_manifests={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in files
        },
        limitations=[
            "TLID filename/perceptual groups are inferred photo groups, not verified plant IDs",
            "Pakistan contains augmented and republished imagery; only raw nonoverlapping photos are eligible",
            "Binary external source stays outside fitting and checkpoint selection",
        ],
    )
    (ROOT / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
