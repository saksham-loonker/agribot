"""Bounded symptom-region validation before any expensive app replay.

Publisher-held-out development only, never independent acceptance. The two
confidence cutoffs were fixed by phase 40 before training. This does not export
a detector, change the primary model, or emit disease names.
"""

from pathlib import Path
import argparse
import concurrent.futures
import json
import time

import numpy as np
from PIL import Image
import torch
from libreyolo import LibreYOLO9

from app_eval import canon
from opened_context_data import read
from permissive_encoder import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign)
    source = campaign / "phase40_region_veto"
    out = campaign / "phase45_region_validation"
    out.mkdir(exist_ok=True)
    checkpoint = source / "runs/region_veto_640/weights/best.pt"
    fit = read(source / "training_complete.json")
    assert digest(checkpoint) == fit["checkpoint_sha256"]
    complete = read(source / "data_complete.json")
    assert digest(source / "source_rows.json") == complete["rows_sha256"]
    rows = [r for r in read(source / "source_rows.json") if r["split"] == "val"]
    for row in rows:
        assert digest(row["detection_image"]) == row["file_sha256"]
    protocol = dict(
        task="Held-out publisher symptom-region utility screen, development only",
        checkpoint_sha256=digest(checkpoint),
        source_rows_sha256=digest(source / "source_rows.json"),
        training_result_sha256=digest(source / "training_complete.json"),
        confidence_cutoffs=[0.75, 0.9],
        disease_class=1,
        healthy_class=0,
        require_disease_class_winner=True,
        nms_iou=0.6,
        max_regions=300,
        region_matching_iou=0.5,
        utility_screen="At least 20% diseased-photo veto recall and at most 5% Healthy-photo veto rate; app gates remain mandatory",
        stopping_rule="No app inference if neither fixed cutoff passes this utility screen; no threshold expansion",
        script_sha256=digest(__file__),
        canonical_source_sha256=digest(
            Path(__file__).resolve().parents[2] / "model_training/v2/final/canon.py"
        ),
        cold_olid_used=False,
        ad_independent=False,
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        result = read(out / "result.json")
        assert result["protocol"] == protocol
        assert digest(out / "predictions.json") == result["predictions_sha256"]
        print(json.dumps(result, indent=2))
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LibreYOLO9(str(checkpoint), size="t", nb_classes=2).model
    model = model.cuda().eval().requires_grad_(False)

    def pixels(row):
        with Image.open(row["detection_image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor, lb = canon.detector_input(rgb, 640)
        return tensor, lb, rgb.shape[1], rgb.shape[0]

    results = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(6) as pool, torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            prepared = list(pool.map(pixels, batch))
            tensor = torch.from_numpy(np.concatenate([v[0] for v in prepared], 0))
            raw = model(tensor.permute(0, 3, 1, 2).cuda())["predictions"].cpu().numpy()
            assert raw.ndim == 3 and raw.shape[1] == 6
            assert np.isfinite(raw).all()
            for row, prediction, (_, lb, width, height) in zip(batch, raw, prepared):
                disease = prediction[[0, 1, 2, 3, 5]].copy()
                disease[4, prediction[5] <= prediction[4]] = 0.0
                truth = [
                    [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        (x + w / 2) * width,
                        (y + h / 2) * height,
                    ]
                    for cls, (x, y, w, h) in zip(row["raw_classes"], row["boxes"])
                    if cls != 4
                ]
                decisions = []
                for cutoff in protocol["confidence_cutoffs"]:
                    detected = canon.decode_detections(
                        disease[None],
                        lb,
                        width,
                        height,
                        conf=cutoff,
                        iou_thr=0.6,
                        max_det=300,
                    )
                    remaining = set(range(len(truth)))
                    matched = 0
                    for box in detected:
                        best = max(
                            remaining,
                            key=lambda i: canon.iou(box, truth[i]),
                            default=None,
                        )
                        if best is not None and canon.iou(box, truth[best]) >= 0.5:
                            remaining.remove(best)
                            matched += 1
                    decisions.append(
                        dict(
                            cutoff=cutoff,
                            veto=bool(detected),
                            predicted_regions=len(detected),
                            matched_disease_regions=matched,
                        )
                    )
                results.append(
                    dict(
                        path=row["detection_image"],
                        file_sha256=row["file_sha256"],
                        disease_regions=len(truth),
                        diseased_photo=bool(truth),
                        decisions=decisions,
                    )
                )
    candidates = []
    for index, cutoff in enumerate(protocol["confidence_cutoffs"]):
        diseased = [r for r in results if r["diseased_photo"]]
        healthy = [r for r in results if not r["diseased_photo"]]
        assert diseased and healthy
        recovered = sum(r["decisions"][index]["veto"] for r in diseased)
        false = sum(r["decisions"][index]["veto"] for r in healthy)
        candidates.append(
            dict(
                cutoff=cutoff,
                diseased_photos=len(diseased),
                healthy_photos=len(healthy),
                diseased_photo_veto=recovered,
                healthy_photo_false_veto=false,
                disease_regions=sum(r["disease_regions"] for r in results),
                matched_disease_regions=sum(
                    r["decisions"][index]["matched_disease_regions"] for r in results
                ),
                predicted_regions=sum(
                    r["decisions"][index]["predicted_regions"] for r in results
                ),
                passed_utility_screen=recovered / len(diseased) >= 0.2
                and false / len(healthy) <= 0.05,
            )
        )
    (out / "predictions.json").write_text(json.dumps(results, indent=2))
    report = dict(
        protocol=protocol,
        candidates=candidates,
        passing_utility_screens=sum(c["passed_utility_screen"] for c in candidates),
        predictions_sha256=digest(out / "predictions.json"),
        new_inference_images=len(rows),
        inference_seconds=time.monotonic() - started,
        cold_olid_used=False,
        app_acceptance_claim=False,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
