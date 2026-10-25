"""Train a separate tomato-species verifier from explicit source taxonomy.

This does not change any disease label or the nine-class model. PlantDoc crop
taxonomy provides negatives. Mixed-species whole scenes and field pseudo ROIs
without verified genus are masked. No prospective reserved pixels are used.
"""

from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import time

import numpy as np
import timm
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from campaign_cnn_train import sampling
from campaign_cnn_fusion import calibrate
from campaign_pixels import open_pixels
from training_support import atomic_save, capture_rng, restore_rng


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def species_targets(rows, original):
    by_id = {r["id"]: r for r in original if r["source"] == "plantdoc"}
    photo_labels = collections.defaultdict(set)
    for r in by_id.values():
        if r.get("raw_label"):
            photo_labels[r["path"]].add(
                int(r["raw_label"].lower().startswith("tomato"))
            )
    result = []
    tomato_sources = {
        "tomatovillage",
        "plantvillage",
        "tlid",
        "pakistan",
        "bangladesh_binary",
        "bangladesh_classification",
        "swtt5wy2pr",
    }
    for r in rows:
        value = -1
        if r["source"] == "plantdoc":
            if r.get("box") is not None and r["id"] in by_id:
                label = by_id[r["id"]].get("raw_label")
                if label:
                    value = int(label.lower().startswith("tomato"))
            else:
                labels = photo_labels[r["path"]]
                if len(labels) == 1:
                    value = next(iter(labels))
        elif r["source"] in tomato_sources:
            value = 1
        elif r["source"] == "tom2024" and r.get("box") is None:
            # Presence of the publisher's primary tomato plant in a whole scene.
            # A generic teacher's individual leaf proposal does not verify genus.
            value = 1
        result.append(value)
    return np.asarray(result, np.int64)


def balanced(rows, ids):
    weights = sampling(rows, ids, target="species_y")
    y = np.array([rows[i]["species_y"] for i in ids])
    classes = np.unique(y)
    for label in classes:
        mask = y == label
        weights[mask] *= 1 / (len(classes) * weights[mask].sum())
    return weights / weights.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--scene-work")
    ap.add_argument("--run-name")
    ap.add_argument("--warmstart")
    ap.add_argument("--negative-cost", type=float, default=2.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--learning-rate", type=float, default=0.0001)
    a = ap.parse_args()
    assert (
        a.negative_cost > 0
        and a.focal_gamma >= 0
        and a.learning_rate > 0
        and a.epochs > 0
        and a.batch > 0
    )
    scene = bool(a.scene_work)
    root = Path(a.root)
    out = root / (a.run_name or ("scene_healthy" if scene else "species"))
    out.mkdir(exist_ok=True)
    work = Path(a.scene_work) if scene else root / "cnn_inputs"
    rows_path = work / "rows.json"
    rows = json.loads(rows_path.read_text())
    mapping = Path("/home/genesis/agribot-data/work/data_manifest.json")
    original = json.loads(mapping.read_text())
    target = (
        np.array([int(r["health_y"]) for r in rows])
        if scene
        else species_targets(rows, original)
    )
    for r, y in zip(rows, target):
        r["species_y"] = int(y)
    train = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] == "training" and r["split"] == "train" and target[i] >= 0
        ]
    )
    val = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] == "training" and r["split"] == "val" and target[i] >= 0
        ]
    )
    assert set(target[train]) == set(target[val]) == {0, 1}
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    reserved = json.loads((root / "reserved_manifest.json").read_text())
    assert not (
        {rows[i]["group"] for i in np.concatenate([train, val])}
        & {r["group"] for r in reserved}
    )
    warm = (
        Path(a.warmstart)
        if a.warmstart
        else root
        / "cnn_inputs/runs/mobilenetv3_small_100.lamb_in1k_center_focal_n2/best.pt"
    )
    config = dict(
        rows_sha256=digest(rows_path),
        source_taxonomy_sha256=digest(mapping),
        warmstart_sha256=digest(warm),
        epochs=a.epochs,
        batch=a.batch,
        seed=163,
        negative_cost=a.negative_cost,
        lr=a.learning_rate,
        supervision="Explicit PlantDoc leaf taxonomy; unambiguous tomato-only sources; whole TOM2024 tomato presence. Mixed whole scenes and unverified-genus field ROIs masked.",
        training_pixels="float16 cache, training augmentation only; validation/deployment use original canonical float32 pixels",
        reserved_used=False,
    )
    if scene:
        config["task"] = "whole_image_healthy_tomato"
        config["supervision"] = (
            "Image-level labels only; any unhealthy annotated instance is a negative whole scene. No crop label is used as a scene-positive shortcut."
        )
    if a.focal_gamma:
        config["focal_gamma"] = a.focal_gamma
    protocol = out / "protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == config
    else:
        protocol.write_text(json.dumps(config, indent=2))
    if (out / "result.json").exists():
        return
    torch.set_num_threads(4)
    torch.manual_seed(163)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = torch.load(warm, weights_only=True, map_location="cpu")
    model = timm.create_model(
        initial["architecture"], pretrained=False, num_classes=1
    ).cuda()
    state = initial["model"].copy()
    head = 0 if scene else 1
    if state["classifier.weight"].shape[0] != 1:
        state["classifier.weight"] = state["classifier.weight"][head : head + 1].clone()
        state["classifier.bias"] = state["classifier.bias"][head : head + 1].clone()
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.learning_rate, weight_decay=0.001
    )
    pixels = open_pixels(work)
    cached = torch.empty((len(train), 3, 224, 224), dtype=torch.float16, device="cuda")
    for start in range(0, len(train), 128):
        x = torch.from_numpy(pixels[train[start : start + 128]]).permute(0, 3, 1, 2)
        cached[start : start + len(x)] = x.cuda().half()
    weights = torch.tensor(balanced(rows, train), dtype=torch.float64)
    truth = torch.tensor(target, device="cuda", dtype=torch.float32)
    canonical_mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    canonical_std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    mean = torch.tensor(initial["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(initial["std"], device="cuda").view(1, 3, 1, 1)
    history = []
    best = -float("inf")
    start_epoch = 0
    last = out / "last.pt"
    if last.exists():
        saved = torch.load(last, weights_only=True, map_location="cpu")
        assert saved["configuration"] == config
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        best = saved["best_score"]
        start_epoch = saved["epoch"] + 1
        history = json.loads((out / "history.json").read_text())

    def evaluate(ids):
        output = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(ids), 256):
                x = (
                    torch.from_numpy(pixels[ids[start : start + 256]])
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                output.append(
                    model((x * canonical_std + canonical_mean - mean) / std)
                    .ravel()
                    .cpu()
                    .numpy()
                )
        return np.concatenate(output)

    started = time.time()
    print(
        json.dumps(
            dict(
                stage="training",
                train=len(train),
                validation=len(val),
                targets=dict(collections.Counter(target[train].tolist())),
                configuration=config,
            )
        ),
        flush=True,
    )
    for epoch in range(start_epoch, a.epochs):
        model.train()
        order = torch.multinomial(
            weights,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(16300 + epoch),
        )
        total = 0.0
        batches = 0
        for positions in order.split(a.batch):
            rgb = cached[positions.cuda()].float() * canonical_std + canonical_mean
            if torch.rand(()) < 0.5:
                rgb = rgb.flip(-1)
            rgb = torch.rot90(rgb, int(torch.randint(0, 4, ()).item()), (-2, -1))
            rgb = (
                rgb * torch.empty((len(rgb), 1, 1, 1), device="cuda").uniform_(0.9, 1.1)
            ).clamp(0, 1)
            y = truth[train[positions.numpy()]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model((rgb - mean) / std).ravel()
            bce = F.binary_cross_entropy_with_logits(z.float(), y, reduction="none")
            loss = (
                bce
                * (1 - torch.exp(-bce)).pow(a.focal_gamma)
                * torch.where(y > 0, 1.0, a.negative_cost)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        optimizer.param_groups[0]["lr"] = a.learning_rate * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
        )
        logits = evaluate(val)
        prob = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        y = target[val]
        pd = np.array(
            [rows[i]["source"] == ("tom2024" if scene else "plantdoc") for i in val]
        )
        auc = float(roc_auc_score(y[pd], prob[pd]))
        fpr = float((prob[y == 0] >= 0.9).mean())
        recall = float((prob[y == 1] >= 0.5).mean())
        score = auc + 0.1 * recall - 2 * fpr
        state = dict(
            architecture=initial["architecture"],
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            mean=initial["mean"],
            std=initial["std"],
            labels=(
                ["not_healthy_tomato", "healthy_tomato"]
                if scene
                else ["non_tomato", "tomato"]
            ),
            configuration=config,
            epoch=epoch,
            score=score,
        )
        if score > best:
            best = score
            atomic_save(state, out / "best.pt")
        atomic_save(
            dict(
                state,
                optimizer=optimizer.state_dict(),
                rng=capture_rng(),
                best_score=best,
            ),
            last,
        )
        record = dict(
            epoch=epoch,
            loss=total / batches,
            score=score,
            plantdoc_auc=auc,
            non_tomato_fpr_at_09=fpr,
            tomato_recall_at_05=recall,
            seconds=time.time() - started,
        )
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    selected = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["model"])
    logits = evaluate(val)
    calibration = calibrate(logits, target[val], balanced(rows, val))
    probability = 1 / (
        1 + np.exp(-np.clip(logits * calibration[0] + calibration[1], -80, 80))
    )
    result = dict(
        configuration=config,
        best_checkpoint_sha256=digest(out / "best.pt"),
        calibration=calibration,
        calibration_scope="Grouped validation only; balanced class/source/group weights",
        validation_n=len(val),
        positive_label="healthy_tomato" if scene else "tomato",
        thresholds={
            str(t): dict(
                tomato_recall=float((probability[target[val] == 1] >= t).mean()),
                non_tomato_false_positive=float(
                    (probability[target[val] == 0] >= t).mean()
                ),
            )
            for t in [0.8, 0.9, 0.95, 0.99]
        },
        training_targets=dict(collections.Counter(target[train].tolist())),
        seconds=time.time() - started,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
