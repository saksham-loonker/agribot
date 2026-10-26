"""Fine-tune only the separate image verifier on the explicitly revised corpus.

The original disease classifier is never loaded into an optimizer. The consumed
246 and related families are training/internal development now, never test data.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from torch.nn import functional as F
from torchvision import transforms

from campaign_cnn_fusion import calibrate
from campaign_pixels import open_pixels
from leaf_feature_expert import LeafFeatureHead
from phase7_species_train import balanced
from training_support import (
    atomic_save,
    capture_rng,
    restore_rng,
    load_training_checkpoint,
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher_logits(root, rows):
    p = root.parent
    u = p / "phase11_scene"
    s = p / "phase6_roi"
    old_rows = read(u / "cnn_inputs/rows.json")
    assert rows[: len(old_rows)] == old_rows
    fit = read(u / "fusion_selection/candidate.json")["fusion"]
    r = read(u / "mobileclip_feature_heads/ranked.json")[0]
    directory = Path(r["checkpoint"]).parent
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    head = LeafFeatureHead(state["kind"], 512).eval()
    head.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        z = (
            head(
                torch.from_numpy(np.load(u / "mobileclip_feature_heads/features.npy"))
            ).numpy()
            * r["calibration"][0]
            + r["calibration"][1]
        )
    x = np.concatenate(
        [np.load(s / "scene_sft_d12_focal_fusion/calibrated_logits.npy"), z[:, None]], 1
    )
    result = np.full(len(rows), np.nan, np.float32)
    result[: len(old_rows)] = (
        np.clip(x, -10, 10) @ np.asarray(fit["weights"]) + fit["bias"]
    ).astype(np.float32)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", choices=["mobileclip", "siglip2"], default="mobileclip")
    ap.add_argument("--loss", choices=["ce", "focal", "distill"], required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--adaptation", choices=["full", "pool", "tail2"], default="full")
    ap.add_argument("--backbone-lr", type=float)
    ap.add_argument("--run-name")
    ap.add_argument(
        "--selection-metric", choices=["legacy", "low_fpr"], default="legacy"
    )
    a = ap.parse_args()
    if a.family == "siglip2":
        assert (
            a.loss != "distill"
        ), "Restricted MobileCLIP teachers must never train this model"
        from permissive_scene import load
    else:
        from mobileclip_scene import load
    root = Path(a.root)
    if a.adaptation != "full":
        assert a.family == "siglip2" and a.loss != "distill"
        assert a.run_name, "Version partial-adaptation outputs separately"
    if a.run_name:
        assert Path(a.run_name).name == a.run_name and a.run_name not in (".", "..")
    out = root / (a.run_name or "sft_" + a.loss)
    out.mkdir(exist_ok=True)
    assert (root / "RETIRED_BENCHMARK.json").exists()
    rows = read(root / "cnn_inputs/rows.json")
    target = np.array([int(r["health_y"]) for r in rows])
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    assert set(target[train]) == set(target[val]) == {0, 1}
    warm = read(root / (a.family + "_feature_heads") / "ranked.json")[0]
    directory = Path(warm["checkpoint"]).parent
    config = dict(
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        admission_sha256=digest(root / "complete.json"),
        warm_head_sha256=digest(directory / "best.pt"),
        warm_encoder_sha256=digest(directory.parent / "encoder.pt"),
        warm_calibration_sha256=digest(directory / "result.json"),
        loss=a.loss,
        epochs=a.epochs,
        batch=a.batch,
        seed=557,
        backbone_lr=a.backbone_lr if a.backbone_lr is not None else 2e-5,
        head_lr=1e-4,
        weight_decay=0.01,
        negative_cost=4.0,
        focal_gamma=2.0 if a.loss == "focal" else 0.0,
        distillation_weight=0.2 if a.loss == "distill" else 0.0,
        distillation_temperature=2.0,
        supervision="Image/primary-view labels; consumed families are training/internal validation, not benchmarks. No disease model update.",
        augmentation=dict(
            horizontal_flip=True,
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02,
            blur_probability=0.25,
            downsample_sides=[112, 160, 224],
        ),
        training_storage="FP16 canonical tensors; BF16 compute; exact FP32 validation",
        cold_ad_used=False,
    )
    if a.loss == "distill":
        config["teacher_candidate_sha256"] = digest(
            root.parent / "phase11_scene/fusion_selection/candidate.json"
        )
    if a.family == "siglip2":
        config["model_family"] = "siglip2"
        config["restricted_teacher_used"] = False
        assert read(root / "encoder_source.json")["license"] == "apache-2.0"
    if a.adaptation != "full":
        config["adaptation"] = a.adaptation
        config["warm_checkpoint_in_selection"] = True
    if a.selection_metric != "legacy":
        assert a.family == "siglip2"
        config["selection_metric"] = (
            "Scale-invariant recall at weighted FPR<=.005 plus standardized partial AUC; internal validation only"
        )
    protocol = out / "protocol.json"
    if protocol.exists():
        assert read(protocol) == config
    else:
        protocol.write_text(json.dumps(config, indent=2))
    if (out / "result.json").exists() and (out / "restored_model_parity.json").exists():
        assert read(out / "restored_model_parity.json")["passed"]
        print("Already complete", out)
        return
    torch.set_num_threads(4)
    torch.manual_seed(557)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    spec, model = load(directory)
    spec = dict(spec, calibration=[1.0, 0.0])
    model.calibration.copy_(torch.tensor([1.0, 0.0]))
    model = model.cuda().requires_grad_(True)
    if a.adaptation != "full":
        model.encoder.requires_grad_(False)
        network = model.encoder.network
        network.attn_pool.requires_grad_(True)
        network.norm.requires_grad_(True)
        if a.adaptation == "tail2":
            for block in network.blocks[-2:]:
                block.requires_grad_(True)
    for r, y in zip(rows, target):
        r["species_y"] = int(y)
    sampling = torch.tensor(balanced(rows, train), dtype=torch.float64)
    val_weights = balanced(rows, val)
    truth = torch.tensor(target, device="cuda", dtype=torch.float32)
    pixels = open_pixels(root / "cnn_inputs")
    images = {}
    for split, ids, dtype in [
        ("train", train, torch.float16),
        ("val", val, torch.float32),
    ]:
        cached = torch.empty((len(ids), 224, 224, 3), device="cuda", dtype=dtype)
        for start in range(0, len(ids), 256):
            cached[start : start + 256].copy_(
                torch.from_numpy(
                    np.array(pixels[ids[start : start + 256]], copy=True)
                ).cuda()
            )
        images[split] = cached
    teacher = (
        teacher_logits(root, read(root / "cnn_inputs/rows.json"))
        if a.loss == "distill"
        else np.full(len(rows), np.nan, np.float32)
    )
    teacher = torch.tensor(teacher, device="cuda")
    teacher_p = torch.nan_to_num(teacher).sigmoid()
    teacher_mask = torch.isfinite(teacher) & (
        ((truth > 0) & (teacher_p >= 0.9)) | ((truth == 0) & (teacher_p <= 0.1))
    )
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=[p for p in model.encoder.parameters() if p.requires_grad],
                lr=config["backbone_lr"],
            ),
            dict(params=model.head.parameters(), lr=config["head_lr"]),
        ],
        weight_decay=config["weight_decay"],
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    augment = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.25
            ),
        ]
    )
    best = -float("inf")
    start_epoch = 0
    history = []
    if (out / "last.pt").exists():
        state = load_training_checkpoint(out / "last.pt")
        assert state["configuration"] == config
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start_epoch = state["epoch"] + 1
        best = state["best_score"]
        history = read(out / "history.json")
    primary = np.array([rows[i]["source"] == "tom2024" for i in val])
    started = time.time()

    def evaluate():
        model.eval()
        with torch.inference_mode():
            return (
                torch.cat([model(x) for x in images["val"].split(a.batch)])
                .cpu()
                .numpy()
            )

    if a.adaptation != "full" and start_epoch == 0 and not (out / "best.pt").exists():
        warm_logits = evaluate()
        probability = 1 / (1 + np.exp(-np.clip(warm_logits, -80, 80)))
        labels = target[val]
        auc = float(roc_auc_score(labels[primary], probability[primary]))
        recall = float(
            np.sum(val_weights * (labels == 1) * (probability >= 0.5))
            / np.sum(val_weights * (labels == 1))
        )
        fpr = float(
            np.sum(val_weights * (labels == 0) * (probability >= 0.9))
            / np.sum(val_weights * (labels == 0))
        )
        best = auc + 0.1 * recall - 2 * fpr
        if a.selection_metric == "low_fpr":
            from phase14_reselect import ranking_score

            best = ranking_score(warm_logits, labels, val_weights, primary)["score"]
        atomic_save(
            dict(
                architecture="PermissiveSceneSft",
                spec=spec,
                configuration=config,
                model={
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                epoch=-1,
                score=best,
            ),
            out / "best.pt",
        )

    for epoch in range(start_epoch, a.epochs):
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(55700 + epoch),
        )
        total = 0.0
        batches = 0
        for positions in order.split(a.batch):
            x = images["train"][positions.cuda()].float().permute(0, 3, 1, 2)
            rgb = augment((x * std + mean).clamp(0, 1))
            side = [112, 160, 224][int(torch.randint(0, 3, ()).item())]
            if side != 224:
                rgb = F.interpolate(
                    F.interpolate(
                        rgb,
                        size=(side, side),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    ),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            flip = torch.rand(len(rgb), device="cuda") < 0.5
            rgb = torch.where(flip[:, None, None, None], rgb.flip(-1), rgb)
            x = ((rgb - mean) / std).permute(0, 2, 3, 1)
            indices = train[positions.numpy()]
            y = truth[indices]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model(x)
            bce = F.binary_cross_entropy_with_logits(z.float(), y, reduction="none")
            loss = (
                bce
                * (1 - torch.exp(-bce)).pow(config["focal_gamma"])
                * torch.where(y > 0, 1.0, config["negative_cost"])
            ).mean()
            if a.loss == "distill":
                mask = teacher_mask[indices]
                soft = (torch.nan_to_num(teacher[indices]) / 2).sigmoid()
                kd = (
                    F.binary_cross_entropy_with_logits(
                        z.float() / 2, soft, reduction="none"
                    )
                    * 4
                )
                loss = loss + config["distillation_weight"] * (
                    kd * mask
                ).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        for group, lr in zip(
            optimizer.param_groups, [config["backbone_lr"], config["head_lr"]]
        ):
            group["lr"] = lr * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
            )
        z = evaluate()
        prob = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        y = target[val]
        auc = float(roc_auc_score(y[primary], prob[primary]))
        recall = float(
            np.sum(val_weights * (y == 1) * (prob >= 0.5))
            / np.sum(val_weights * (y == 1))
        )
        fpr = float(
            np.sum(val_weights * (y == 0) * (prob >= 0.9))
            / np.sum(val_weights * (y == 0))
        )
        score = auc + 0.1 * recall - 2 * fpr
        if a.selection_metric == "low_fpr":
            from phase14_reselect import ranking_score

            score = ranking_score(z, y, val_weights, primary)["score"]
        state = dict(
            architecture=(
                "PermissiveSceneSft" if a.family == "siglip2" else "MobileClipSceneSft"
            ),
            spec=spec,
            configuration=config,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
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
            out / "last.pt",
        )
        record = dict(
            epoch=epoch,
            loss=total / batches,
            score=score,
            field_auc=auc,
            weighted_false_positive_at_09=fpr,
            weighted_healthy_recall_at_05=recall,
            seconds=time.time() - started,
        )
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    selected = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["model"], strict=True)
    z = evaluate()
    calibration = calibrate(z, target[val], val_weights)
    prob = 1 / (1 + np.exp(-np.clip(z * calibration[0] + calibration[1], -80, 80)))
    result = dict(
        configuration=config,
        checkpoint_sha256=digest(out / "best.pt"),
        calibration=calibration,
        validation_n=len(val),
        field_auc=float(roc_auc_score(target[val][primary], prob[primary])),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        thresholds={
            str(t): dict(
                healthy_recall=float((prob[target[val] == 1] >= t).mean()),
                negative_false_positive=float((prob[target[val] == 0] >= t).mean()),
                field_healthy_recall=float(
                    (prob[primary & (target[val] == 1)] >= t).mean()
                ),
                field_negative_false_positive=float(
                    (prob[primary & (target[val] == 0)] >= t).mean()
                ),
            )
            for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        },
        teacher_agreement_training_cases=int(teacher_mask[train].sum()),
        cold_ad_used=False,
        consumed_cohort_is_benchmark=False,
        seconds=time.time() - started,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    np.savez(out / "validation_logits.npz", indices=val, logits=z)
    _, restored = load(out)
    restored = restored.cuda().eval()
    with torch.inference_mode():
        actual = (
            torch.cat([restored(x) for x in images["val"].split(a.batch)]).cpu().numpy()
        )
    error = float(np.max(np.abs(actual - (z * calibration[0] + calibration[1]))))
    assert error < 0.002, (
        "Calibrated restored verifier differs from validation",
        error,
    )
    (out / "restored_model_parity.json").write_text(
        json.dumps(
            dict(
                passed=True,
                cases=len(val),
                max_abs_calibrated_logit_error=error,
                cold_ad_used=False,
            ),
            indent=2,
        )
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
