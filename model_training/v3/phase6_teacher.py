"""Audit text-grounded whole-leaf pseudo boxes before any detector adaptation.

These are model predictions, never human ground truth or acceptance labels.
Only development/train images are eligible. The prospective reservation is not
opened. Nothing in this script modifies the app or an existing checkpoint.
"""

from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms
from huggingface_hub import HfApi
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--full-training", action="store_true")
    ap.add_argument("--healthy-only", action="store_true")
    ap.add_argument("--prompt", default="a tomato leaf.")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Path(a.campaign)
    out = Path(a.out) if a.out else p / "phase6_teacher"
    assert 0 < a.threshold < 1 and a.prompt.endswith(".")
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = json.loads((p / "phase5/development_manifest.json").read_text())
    reserved = json.loads((p / "phase5/reserved_manifest.json").read_text())
    train = [r for r in records if r["split"] == "train"]
    assert not ({r["group"] for r in train} & {r["group"] for r in reserved})
    assert not (
        {r["file_sha256"] for r in train} & {r["file_sha256"] for r in reserved}
    )
    if a.full_training:
        audit = json.loads((out / "annotation_role_audit.json").read_text())
        assert audit[
            "eligible_for_pseudo_training"
        ], "Teacher role audit must pass first"
        assert (
            a.healthy_only
            and audit["allowed_training_scope"] == "tom2024 Healthy train only"
        )
        selected = [
            r for r in train if r["source"] == "tom2024" and r["label"] == "Healthy"
        ]
    else:
        selected = []
        for source, label, n in [
            ("tom2024", "Healthy", 8),
            ("tom2024", "Diseased_unspecified", 4),
            ("swtt5wy2pr", "Early_blight", 2),
            ("swtt5wy2pr", "Late_blight", 2),
        ]:
            eligible = [
                r for r in train if r["source"] == source and r["label"] == label
            ]
            eligible.sort(
                key=lambda r: hashlib.sha256(
                    ("teacher131:" + r["id"]).encode()
                ).hexdigest()
            )
            selected.extend(eligible[:n])
        selected = selected[: a.limit]
    identity_path = out / "model_identity.json"
    name = "IDEA-Research/grounding-dino-tiny"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        assert (
            identity["prompt"] == a.prompt and identity["box_threshold"] == a.threshold
        )
    else:
        revision = HfApi().model_info(name).sha
        identity = dict(
            model=name,
            revision=revision,
            license="Apache-2.0",
            prompt=a.prompt,
            box_threshold=a.threshold,
            text_threshold=0.25,
            ground_truth=False,
            source="https://huggingface.co/IDEA-Research/grounding-dino-tiny",
        )
        identity_path.write_text(json.dumps(identity, indent=2))
    processor = AutoProcessor.from_pretrained(name, revision=identity["revision"])
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            name, revision=identity["revision"]
        )
        .cuda()
        .eval()
    )
    output = out / (
        "training_predictions.jsonl" if a.full_training else "audit_predictions.jsonl"
    )
    existing = {}
    if output.exists():
        existing = {
            r["id"]: r for r in map(json.loads, output.read_text().splitlines())
        }
    started = time.time()
    with output.open("a") as stream, torch.inference_mode():
        for n, row in enumerate(selected):
            if row["id"] in existing:
                continue
            image = Image.open(row["path"]).convert("RGB")
            inputs = processor(
                images=image, text=identity["prompt"], return_tensors="pt"
            ).to("cuda")
            outputs = model(**inputs)
            kwargs = dict(
                text_threshold=identity["text_threshold"],
                target_sizes=[image.size[::-1]],
            )
            method = processor.post_process_grounded_object_detection
            key = (
                "threshold"
                if "threshold" in inspect.signature(method).parameters
                else "box_threshold"
            )
            kwargs[key] = identity["box_threshold"]
            result = method(outputs, inputs.input_ids, **kwargs)[0]
            boxes = result["boxes"]
            scores = result["scores"]
            width, height = image.size
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
            spans = boxes[:, 2:] - boxes[:, :2]
            area = spans.prod(1) / (width * height)
            keep = (spans.min(1).values >= 12) & (area >= 0.005) & (area <= 0.95)
            boxes, scores = boxes[keep], scores[keep]
            chosen = nms(boxes, scores, 0.5)[:20]
            result = dict(
                row,
                teacher_boxes=boxes[chosen].cpu().tolist(),
                teacher_scores=scores[chosen].cpu().tolist(),
                teacher_identity=identity,
            )
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if n % 20 == 0:
                print(
                    json.dumps(
                        dict(
                            photos=n, total=len(selected), seconds=time.time() - started
                        )
                    ),
                    flush=True,
                )
    if not a.full_training:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        predictions = {
            r["id"]: r for r in map(json.loads, output.read_text().splitlines())
        }
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for ax, row in zip(axes.ravel(), selected):
            observed = predictions[row["id"]]
            ax.imshow(Image.open(row["path"]))
            ax.axis("off")
            ax.set_title(row["source"] + " " + row["label"])
            for b, s in zip(observed["teacher_boxes"], observed["teacher_scores"]):
                ax.add_patch(
                    Rectangle(
                        (b[0], b[1]),
                        b[2] - b[0],
                        b[3] - b[1],
                        fill=False,
                        edgecolor="red",
                        linewidth=1,
                    )
                )
                ax.text(b[0], b[1], f"{s:.2f}", color="yellow", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "teacher_audit.png", dpi=110)
        plt.close(fig)
    print(
        json.dumps(
            dict(
                state="complete",
                photos=len(selected),
                predictions=str(output),
                seconds=time.time() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
