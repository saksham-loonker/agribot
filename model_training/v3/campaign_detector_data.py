"""Align fitting and grouped selection with the crops produced by the app.

Annotated multi-leaf photos admit a detected training crop only when it overlaps
the labelled object by IoU >= 0.5. Single-class close-up sources admit their top
leaf crop. Selection photos are built only from the validation split.
"""

from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, threading, time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from app_eval import canon, quality


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--extra", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    assets = repo / "agribot_android_app/android/app/src/main/assets"
    manifest = json.loads((assets / "model_manifest.json").read_text())
    contract = manifest["detector"]
    model_path = assets / "models" / contract["file"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == contract["sha256"]
    original = json.loads(
        Path("/home/genesis/agribot-data/work/data_manifest.json").read_text()
    )
    extra = json.loads(Path(a.extra).read_text())
    assert all(r["split"] in ("train", "val") for r in extra)
    records = original + extra
    photos = collections.defaultdict(list)
    for record in records:
        if record["split"] in ("train", "val"):
            photos[record["path"]].append(record)
    identity = dict(
        detector_sha256=contract["sha256"],
        source_manifest_sha256=hashlib.sha256(Path(a.extra).read_bytes()).hexdigest(),
        original_manifest_sha256=hashlib.sha256(
            Path("/home/genesis/agribot-data/work/data_manifest.json").read_bytes()
        ).hexdigest(),
    )
    local = threading.local()

    def process(item):
        path, annotations = item
        if not hasattr(local, "detector"):
            local.detector = Interpreter(model_path=str(model_path), num_threads=1)
            local.detector.allocate_tensors()
        detector = local.detector
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        tensor, lb = canon.detector_input(image, contract["input"][1])
        detector.set_tensor(detector.get_input_details()[0]["index"], tensor)
        detector.invoke()
        output = detector.get_tensor(detector.get_output_details()[0]["index"])
        assert np.isfinite(output).all()
        boxes = canon.decode_detections(
            output,
            lb,
            image.shape[1],
            image.shape[0],
            contract["score_threshold"],
            contract["nms_iou"],
            contract["max_detections"],
        )[:6]
        first = annotations[0]
        labels = {r["label"] for r in annotations}
        label = next(iter(labels)) if len(labels) == 1 else "Mixed"
        photo = dict(
            id="select:"
            + hashlib.sha256((first["source"] + ":" + path).encode()).hexdigest()[:16],
            path=path,
            source=first["source"],
            species=first.get("species"),
            split=first["split"],
            group=first["group"],
            label=label,
            analysis_eligible=True,
            objects=[
                dict(
                    box=r["box"], label=r["label"], source_label=r.get("raw_label", "")
                )
                for r in annotations
                if r.get("box") is not None
            ],
            detections=boxes,
            leaves=[
                dict(box=b[:4], score=b[4], member_logits=[[0.0] * 9]) for b in boxes
            ],
            **quality(image)
        )
        rows = []
        for record in annotations:
            if record.get("box") is None:
                match = boxes[0] if boxes else None
            else:
                match = (
                    max(boxes, key=lambda b: canon.iou(record["box"], b[:4]))
                    if boxes
                    else None
                )
                if match is not None and canon.iou(record["box"], match[:4]) < 0.5:
                    match = None
            if match is not None:
                r = dict(record)
                r.update(
                    id=record["id"] + ":detector",
                    box=match[:4],
                    view="detector",
                    detector_score=match[4],
                )
                rows.append(r)
        return photo, rows

    selection = []
    detected = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for i, (photo, rows) in enumerate(executor.map(process, photos.items())):
            detected.extend(rows)
            if photo["split"] == "val":
                selection.append(photo)
            if i % 1000 == 0:
                print(
                    json.dumps(
                        dict(
                            photos=i,
                            total=len(photos),
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
    (out / "detector_development_records.json").write_text(
        json.dumps(extra + detected, indent=2) + "\n"
    )
    with (out / "selection_app_photos.jsonl").open("w") as stream:
        for row in selection:
            stream.write(json.dumps(row) + "\n")
    (out / "detector_data_identity.json").write_text(
        json.dumps(
            dict(
                identity,
                photos=len(photos),
                detected_training_views=len(detected),
                selection_photos=len(selection),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                complete=True,
                photos=len(photos),
                detected_views=len(detected),
                selection_photos=len(selection),
                seconds=time.time() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
