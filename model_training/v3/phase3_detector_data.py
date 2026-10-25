"""Archive the rejected region-as-leaf experiment for reproducibility.

SLIF boxes are disease/healthy regions, not whole-leaf extents. The attempted
generic remapping is invalid; its checkpoints must not be resumed or exported.
This helper reproduces acquisition only. Training now enforces annotation role.
"""

from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, re, time, urllib.parse
import numpy as np
import requests
from PIL import Image
from campaign_data import profile


def fetch(item, destination):
    path = destination / item["name"]
    assert path.resolve().is_relative_to(destination.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == item["totalBytes"]:
        return path
    for attempt in range(4):
        try:
            r = requests.get(
                "https://www.kaggle.com/api/v1/datasets/download/romiyalgeorge/slif-tomato-dataset/"
                + urllib.parse.quote(item["name"], safe=""),
                params={"datasetVersionNumber": 1},
                timeout=(20, 60),
            )
            r.raise_for_status()
            assert len(r.content) == item["totalBytes"]
            path.write_bytes(r.content)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 + attempt)


def finalize(out):
    data = out / "dataset"
    rows = json.loads((out / "source_rows.json").read_text())
    # Upgrade an already acquired single-view cache without downloading again.
    for row in rows:
        if row["split"] != "train":
            continue
        first = data / "images/train" / ("slif_" + row["group"] + ".jpg")
        if first.exists():
            for n in range(1, 4):
                target = first.with_name(first.stem + f"_r{n}.jpg")
                label = data / "labels/train" / (target.stem + ".txt")
                if not target.exists():
                    target.symlink_to(first.resolve())
                if not label.exists():
                    label.symlink_to(data / "labels/train" / (first.stem + ".txt"))
    for domain in ["slif", "old"]:
        val_list = out / (domain + "_val.txt")
        val_list.write_text(
            "".join(
                str(f) + "\n"
                for f in sorted((data / "images/val").iterdir())
                if f.name.startswith(domain + "_")
            )
        )
        (out / (domain + "_val.yaml")).write_text(
            f"path: {data}\ntrain: images/train\nval: {val_list}\nnc: 1\nnames: ['leaf']\n"
        )
    summary = json.loads((out / "data_complete.json").read_text())
    summary["training_repeat"] = 4
    summary["actual_training_files"] = len(list((data / "images/train").iterdir()))
    (out / "data_complete.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / "phase3_detector"
    out.mkdir(exist_ok=True)
    if (out / "data_complete.json").exists():
        finalize(out)
        return
    files = json.loads((p / "slif_all_files.json").read_text())
    lookup = {f["name"]: f for f in files}
    phase = [f for f in files if f["name"].startswith("Phase_I-Dataset/")]
    assert len(phase) == 2670
    groups = collections.defaultdict(list)
    for item in phase:
        groups[re.match(r"t\d+", Path(item["name"]).name).group()].append(item)
    selected = []
    for group, items in sorted(groups.items()):
        if len({x["name"].split("/")[1] for x in items}) != 1:
            continue
        items.sort(
            key=lambda x: (
                (
                    0
                    if re.match(
                        group + r"_(?:jpg|jpeg|png)\.rf\.", Path(x["name"]).name, re.I
                    )
                    else 1
                ),
                len(Path(x["name"]).name),
                hashlib.sha256(x["name"].encode()).hexdigest(),
            )
        )
        selected.append((group, items[0]))

    def acquire(pair):
        group, item = pair
        image = fetch(item, p / "sources/slif_v1")
        label_name = (
            "Detection_Dataset/lables/" + Path(item["name"]).with_suffix(".txt").name
        )
        label = fetch(lookup[label_name], p / "sources/slif_v1")
        boxes = []
        raw_classes = []
        for line in label.read_text().splitlines():
            values = list(map(float, line.split()))
            assert len(values) == 5 and values[0].is_integer()
            c, x, y, w, h = values
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            raw_classes.append(int(c))
            boxes.append([x, y, w, h])
        row = dict(id=group, path=str(image), box=None, label="Other", source="slif_v1")
        info, _ = profile(row)
        return dict(
            group=group,
            path=str(image),
            published_file=item["name"],
            file_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            annotation_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
            phash=info["phash"],
            raw_classes=raw_classes,
            boxes=boxes,
        )

    rows = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i, row in enumerate(pool.map(acquire, selected)):
            rows.append(row)
            if i % 40 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="annotations",
                            done=i,
                            total=len(selected),
                            seconds=time.time() - started,
                        )
                    ),
                    flush=True,
                )
    # Merge close perceptual matches before splitting related captures.
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(rows)):
        for j in range(i):
            if (int(rows[i]["phash"], 16) ^ int(rows[j]["phash"], 16)).bit_count() <= 4:
                parents[find(i)] = find(j)
    data = out / "dataset"
    counts = collections.Counter()
    links = []
    for i, row in enumerate(rows):
        group = rows[find(i)]["group"]
        split = (
            "val"
            if int(hashlib.sha256(("detector67:" + group).encode()).hexdigest()[:8], 16)
            % 100
            < 25
            else "train"
        )
        row.update(split=split, split_group=group)
        image_dir = data / "images" / split
        label_dir = data / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for repeat in range(4 if split == "train" else 1):
            image = image_dir / (f'slif_{row["group"]}_r{repeat}.jpg')
            if not image.exists():
                image.symlink_to(row["path"])
            label_dir.joinpath(image.stem + ".txt").write_text(
                "".join(
                    "0 " + " ".join(f"{v:.9f}" for v in box) + "\n"
                    for box in row["boxes"]
                )
            )
        counts["slif_" + split] += 1
    old = Path("/home/genesis/sakshyam-agribot/v2/det_data/leaf_combined")
    for split in ["train", "val"]:
        for image in sorted((old / "images" / split).iterdir()):
            label = old / "labels" / split / (image.stem + ".txt")
            assert label.exists()
            target = data / "images" / split / ("old_" + image.name)
            lt = data / "labels" / split / ("old_" + image.stem + ".txt")
            if not target.exists():
                target.symlink_to(image.resolve())
            if not lt.exists():
                lt.symlink_to(label.resolve())
            counts["original_" + split] += 1
    assert not (
        {r["split_group"] for r in rows if r["split"] == "train"}
        & {r["split_group"] for r in rows if r["split"] == "val"}
    )
    (out / "source_rows.json").write_text(json.dumps(rows, indent=2))
    (data / "data.yaml").write_text(
        f"path: {data}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['leaf']\n"
    )
    for domain in ["slif", "old"]:
        val_list = out / (domain + "_val.txt")
        val_list.write_text(
            "".join(
                str(f) + "\n"
                for f in sorted((data / "images/val").iterdir())
                if f.name.startswith(domain + "_")
            )
        )
        (out / (domain + "_val.yaml")).write_text(
            f"path: {data}\ntrain: images/train\nval: {val_list}\nnc: 1\nnames: ['leaf']\n"
        )
    protocol = dict(
        source="SLIF-Tomato v1, DOI 10.34740/kaggle/ds/8911633, CC BY 4.0",
        source_history="Former external cohort now opened and admitted to detector development. This is not independent acceptance.",
        one_published_view_per_family=True,
        training_repeat=4,
        near_phash_merge_distance=4,
        seed=67,
        counts=dict(counts),
        original_test_used=False,
        source_rows_sha256=hashlib.sha256(
            (out / "source_rows.json").read_bytes()
        ).hexdigest(),
        annotation_mapping="All publisher IDs -> generic leaf only; no crop-level disease inference from image folders",
        seconds=time.time() - started,
    )
    (out / "data_complete.json").write_text(json.dumps(protocol, indent=2))
    finalize(out)
    print(json.dumps(protocol), flush=True)


if __name__ == "__main__":
    main()
