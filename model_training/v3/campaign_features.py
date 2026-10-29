"""Cache verified original-backbone features from exact runtime RGB crops.

Training indices use only the train partition; grouped validation selects head
checkpoints. Opened app sets remain regression evidence, never a blind test. The
cache retains identity and exact outputs. Photos and tensors stay outside git.
"""

from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, sys, time
import numpy as np
from PIL import Image
import timm
import torch

from train import LABELS
from training_support import load_initial_weights
from app_eval import canon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/genesis/agribot-data")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--checkpoint",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--extra-manifest")
    p.add_argument("--selection-photos")
    p.add_argument("--dense-pooling", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=9,
        img_size=224,
    ).eval()
    load_initial_weights(
        model, a.checkpoint, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    original_head = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    model = model.cuda().requires_grad_(False)
    records = json.loads((root / "work/data_manifest.json").read_text())
    if a.extra_manifest:
        extra = json.loads(Path(a.extra_manifest).read_text())
        assert all(
            r["split"] in ("train", "val") for r in extra
        ), "locked new sources cannot enter feature fitting cache"
        records += extra
    eval_paths = {
        "development": root
        / "production_20261024/final_development_original/predictions.jsonl",
        "public": root / "production_20261024/final_public_original/predictions.jsonl",
        "prior_test": root
        / "production_20261024/prior_locked_original/predictions.jsonl",
        "taiwan": root / "production_20261024/final_taiwan_original/predictions.jsonl",
    }
    jobs = []
    for i, r in enumerate(records):
        jobs.append(
            dict(
                set="training",
                row=i,
                leaf=-1,
                path=r["path"],
                box=r.get("box"),
                label=r["label"],
                split=r["split"],
                group=r["group"],
                id=r["id"],
                source=r["source"],
                view=r.get("view", "annotation"),
            )
        )
    eval_rows = {}
    if a.selection_photos:
        eval_paths["selection_app"] = Path(a.selection_photos)
    for name, path in eval_paths.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        eval_rows[name] = rows
        for i, r in enumerate(rows):
            for j, leaf in enumerate(r["leaves"]):
                jobs.append(
                    dict(
                        set=name,
                        row=i,
                        leaf=j,
                        path=r["path"],
                        box=leaf["box"],
                        label=r["label"],
                        split="regression",
                        group=r["group"],
                        id=r["id"],
                        source=r["source"],
                    )
                )
    (out / "feature_rows.json").write_text(json.dumps(jobs))
    (out / "eval_rows.json").write_text(json.dumps(eval_rows))

    def prepare(job):
        with Image.open(job["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        return canon.classifier_input(image, 224, job["box"])[0]

    features, logits = [], []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=a.workers
    ) as pool, torch.inference_mode():
        for start in range(0, len(jobs), a.batch):
            images = np.stack(list(pool.map(prepare, jobs[start : start + a.batch])))
            x = torch.from_numpy(images).cuda().permute(0, 3, 1, 2)
            tokens = model.forward_features(x)
            f = model.forward_head(tokens, pre_logits=True)
            if a.dense_pooling:
                patches = tokens[:, model.num_prefix_tokens :]
                mean = patches.mean(1)
                std = (
                    (patches - mean[:, None, :])
                    .square()
                    .mean(1)
                    .clamp_min(1e-12)
                    .sqrt()
                )
                encoded = torch.cat([f, mean, patches.amax(1), std], 1)
            else:
                encoded = f
            features.append(encoded.cpu())
            logits.append(model.head(f).cpu())
            if start % (a.batch * 10) == 0:
                print(
                    json.dumps(
                        dict(
                            stage="features",
                            done=start + len(images),
                            total=len(jobs),
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
    cache = dict(
        features=torch.cat(features),
        logits=torch.cat(logits),
        original_head=original_head,
        labels=LABELS,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(
            (root / "work/data_manifest.json").read_bytes()
        ).hexdigest(),
        rows_sha256=hashlib.sha256(
            (out / "feature_rows.json").read_bytes()
        ).hexdigest(),
        strict_fp32_cuda=True,
        feature_kind="cls_mean_max_std" if a.dense_pooling else "cls",
        extra_manifest_sha256=(
            hashlib.sha256(Path(a.extra_manifest).read_bytes()).hexdigest()
            if a.extra_manifest
            else None
        ),
        preprocessing="exact canon.classifier_input(raw PIL RGB,224,box,pad=.10); no 256 uint8 intermediate",
    )
    # Verify backbone/head reconstruction independently of any optimization arm.
    direct = (
        cache["features"][:, :384] @ original_head["weight"].T + original_head["bias"]
    )
    assert (
        torch.equal(direct, cache["logits"])
        or float((direct - cache["logits"]).abs().max()) < 1e-4
    )
    torch.save(cache, out / "features.pt")
    result = dict(
        rows=len(jobs),
        features=list(cache["features"].shape),
        seconds=round(time.time() - started, 1),
        original_head_reconstruction_max_error=float(
            (direct - cache["logits"]).abs().max()
        ),
        checkpoint_sha256=cache["checkpoint_sha256"],
        preprocessing=cache["preprocessing"],
    )
    (out / "feature_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
