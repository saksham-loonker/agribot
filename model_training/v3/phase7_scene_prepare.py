"""Make image-level Healthy-tomato inputs with image-level supervision only."""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import hashlib
import json
import time

import numpy as np
from PIL import Image
from app_eval import classifier_letterbox_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    work = root / "scene_inputs"
    work.mkdir(exist_ok=True)
    source_path = root / "cnn_inputs/rows.json"
    source = json.loads(source_path.read_text())
    identity = hashlib.sha256(source_path.read_bytes()).hexdigest()
    complete = work / "complete.json"
    if complete.exists():
        assert json.loads(complete.read_text())["source_rows_sha256"] == identity
        return
    path = work / "full_scene.npy"
    assert (
        not path.exists()
    ), "Partial scene pixels require an identity audit before recovery"
    photos = collections.defaultdict(list)
    for row in source:
        if row["set"] == "training" and row["split"] in ("train", "val"):
            photos[row["path"]].append(row)
    rows = []
    excluded = 0
    for filename, items in sorted(photos.items()):
        assert len({r["group"] for r in items}) == len({r["split"] for r in items}) == 1
        labels = {r["health_y"] for r in items}
        if 0.0 in labels:
            target = 0.0
        elif labels == {1.0}:
            target = 1.0
        else:
            excluded += 1
            continue
        first = items[0]
        rows.append(
            dict(
                id="scene:" + hashlib.sha256(filename.encode()).hexdigest()[:20],
                path=filename,
                source=first["source"],
                group=first["group"],
                split=first["split"],
                set="training",
                box=None,
                health_y=target,
                label="Healthy" if target else "Not_healthy_tomato",
                input_variant="full_roi",
                view="full_scene",
            )
        )
    assert not (
        {r["group"] for r in rows if r["split"] == "train"}
        & {r["group"] for r in rows if r["split"] == "val"}
    )
    reserved = json.loads((root / "reserved_manifest.json").read_text())
    assert not ({r["path"] for r in rows} & {r["path"] for r in reserved})
    assert not ({r["group"] for r in rows} & {r["group"] for r in reserved})
    pixels = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(rows), 224, 224, 3)
    )
    started = time.time()

    def prepare(row):
        image = np.asarray(Image.open(row["path"]).convert("RGB"))
        return classifier_letterbox_input(image, 224)[0]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(rows), 128):
            pixels[start : start + 128] = np.stack(
                list(pool.map(prepare, rows[start : start + 128]))
            )
            if start % 2048 == 0:
                print(
                    json.dumps(
                        dict(
                            photos=start, total=len(rows), seconds=time.time() - started
                        )
                    ),
                    flush=True,
                )
    pixels.flush()
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (work / "pixels.json").write_text(json.dumps(dict(center=[str(path)]), indent=2))
    report = dict(
        source_rows_sha256=identity,
        rows_sha256=hashlib.sha256((work / "rows.json").read_bytes()).hexdigest(),
        photos=len(rows),
        counts=dict(
            collections.Counter(
                r["split"] + ":" + r["source"] + ":" + str(r["health_y"]) for r in rows
            )
        ),
        ambiguous_excluded=excluded,
        task="Whole-image Healthy tomato versus all other/unhealthy views. Any annotated unhealthy instance makes the scene negative; ambiguous all-unknown scenes are masked.",
        preprocessing="Full image letterboxed to 224; canonical float32 RGB ImageNet normalization",
        reserved_used=False,
        seconds=time.time() - started,
    )
    complete.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
