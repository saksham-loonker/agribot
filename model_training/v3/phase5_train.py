"""Broader supervised adaptation with binary-only negatives and exact teacher replay.

The original reference stays frozen. Earlier LoRA layers start at zero, so a
deeper learner can warm-start the trained four-block adapter without resetting
its nine-class head. BF16 is training-only; validation/export use strict FP32.
"""

from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from campaign_adapter import prefix
from campaign_cnn_train import sampling
from campaign_pixels import open_pixels
from campaign_heads import classification_metrics
from phase2_multiclass import MulticlassExpert
from training_support import atomic_save, load_initial_weights, capture_rng, restore_rng
from train import LABELS


def extend_lora(model, state):
    old_depth = state.get("depth", 1)
    assert state["mode"] == "lora" and model.depth >= old_depth
    target = model.state_dict()
    offset = model.depth - old_depth
    for key, value in state["model"].items():
        destination = key
        if key.startswith("tail."):
            _, number, suffix = key.split(".", 2)
            destination = f"tail.{int(number)+offset}.{suffix}"
        elif key.startswith("block.") and model.depth != 1:
            destination = f"tail.{model.depth-1}." + key[len("block.") :]
        assert destination in target and target[destination].shape == value.shape, (
            key,
            destination,
        )
        target[destination] = value
    model.load_state_dict(target, strict=True)


def supervised_loss(
    logits, target, health, teacher, binary_weight=0.5, retention_weight=3.0
):
    known = target >= 0
    loss = (
        F.cross_entropy(logits[known], target[known])
        if known.any()
        else logits.sum() * 0
    )
    nonhealthy = torch.cat([logits[:, :1], logits[:, 2:]], 1).logsumexp(1)
    healthy_logit = logits[:, 1] - nonhealthy
    loss = (
        loss
        + binary_weight
        * (
            F.binary_cross_entropy_with_logits(healthy_logit, health, reduction="none")
            * torch.where(health > 0, 1.0, 2.0)
        ).mean()
    )
    reliable = (teacher.argmax(1) == target) & (target != 1) & known
    if reliable.any():
        kl = (
            F.kl_div(
                F.log_softmax(logits[reliable] / 2, 1),
                F.softmax(teacher[reliable] / 2, 1),
                reduction="none",
            ).sum(1)
            * 4
        )
        loss = (
            loss
            + retention_weight
            * torch.stack(
                [kl[target[reliable] == k].mean() for k in target[reliable].unique()]
            ).mean()
        )
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--warmstart", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    root = Path(a.root)
    work = root / "cnn_inputs"
    out = root / "multiclass" / f"lora_d{a.depth}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return
    torch.set_num_threads(4)
    torch.manual_seed(107)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads((work / "rows.json").read_text())
    identity = hashlib.sha256((work / "rows.json").read_bytes()).hexdigest()
    train = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "train" and r["health_y"] >= 0]
    )
    val = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "val" and r["health_y"] >= 0]
    )
    oldval = np.array(
        [
            i
            for i in val
            if rows[i]["source"] in ("plantdoc", "tomatovillage")
            and rows[i]["label"] in LABELS
        ]
    )
    new_sources = ["bangladesh_classification", "tom2024", "swtt5wy2pr"]
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    reserved = json.loads((root / "reserved_manifest.json").read_text())
    assert not (
        {rows[i].get("file_sha256") for i in np.concatenate([train, val])}
        & {r["file_sha256"] for r in reserved}
    )
    assert not (
        {rows[i]["path"] for i in np.concatenate([train, val])}
        & {r["path"] for r in reserved}
    )
    backbone = (
        timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=9,
            img_size=224,
        )
        .cuda()
        .eval()
    )
    encoder = Path("/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt")
    load_initial_weights(
        backbone, str(encoder), LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    backbone.requires_grad_(False)
    initial = torch.load(a.warmstart, weights_only=True, map_location="cpu")
    assert (
        initial["labels"] == LABELS
        and initial["encoder_sha256"]
        == hashlib.sha256(encoder.read_bytes()).hexdigest()
    )
    model = MulticlassExpert(backbone, "lora", initial["rank"], a.depth).cuda()
    extend_lora(model, initial)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.001)
    pixels = open_pixels(work)
    truth = torch.tensor(
        [LABELS.index(r["label"]) if r["label"] in LABELS else -1 for r in rows],
        device="cuda",
    )
    health = torch.tensor([r["health_y"] for r in rows], device="cuda")
    weights = torch.tensor(sampling(rows, train, target="label"))
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    config = dict(
        rows_sha256=identity,
        encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
        warmstart_sha256=hashlib.sha256(Path(a.warmstart).read_bytes()).hexdigest(),
        depth=a.depth,
        rank=initial["rank"],
        epochs=a.epochs,
        batch=a.batch,
        amp="bfloat16 training only",
        seed=107,
        binary_weight=0.5,
        negative_binary_cost=2.0,
        retention_weight=3.0,
    )
    reference = torch.from_numpy(np.load(root / "reference_logits.npy")).cuda()
    baseline = classification_metrics(reference[oldval], truth[oldval])
    history = []
    start_epoch = 0
    best = -float("inf")
    retained_best = -float("inf")
    last = out / "last.pt"
    if last.exists():
        saved = torch.load(last, weights_only=True, map_location="cpu")
        assert saved["training_config"] == config
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start_epoch = saved["epoch"] + 1
        if (out / "history.json").exists():
            history = json.loads((out / "history.json").read_text())[:start_epoch]
        if (out / "best.pt").exists():
            best = torch.load(out / "best.pt", weights_only=True, map_location="cpu")[
                "score"
            ]
        if (out / "retained.pt").exists():
            retained_best = torch.load(
                out / "retained.pt", weights_only=True, map_location="cpu"
            )["score"]

    def evaluate(ids):
        model.eval()
        output = []
        with torch.inference_mode():
            for batch in np.array_split(ids, max(1, math.ceil(len(ids) / 128))):
                x = (
                    torch.from_numpy(np.ascontiguousarray(pixels[batch]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                output.append(model(prefix(backbone, x, a.depth)).float().cpu())
        return torch.cat(output).cuda()

    started = time.time()
    print(
        json.dumps(
            dict(
                stage="training",
                configuration=config,
                training_views=len(train),
                parameters=sum(p.numel() for p in parameters),
                start_epoch=start_epoch,
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
            generator=torch.Generator().manual_seed(10700 + epoch),
        )
        running = 0.0
        batches = 0
        for sample in order.split(a.batch):
            ids = train[sample.numpy()]
            x = (
                torch.from_numpy(np.ascontiguousarray(pixels[ids]))
                .cuda()
                .permute(0, 3, 1, 2)
            )
            rgb = x * std + mean
            if torch.rand(()) < 0.5:
                rgb = rgb.flip(-1)
            rgb = torch.rot90(rgb, int(torch.randint(0, 4, ()).item()), (-2, -1))
            x = (
                (
                    rgb
                    * torch.empty((len(ids), 1, 1, 1), device="cuda").uniform_(0.9, 1.1)
                ).clamp(0, 1)
                - mean
            ) / std
            with torch.no_grad():
                tokens = prefix(backbone, x, a.depth)
                teacher_tokens = tokens
                for block in backbone.blocks[-a.depth :]:
                    teacher_tokens = block(teacher_tokens)
                teacher = backbone.head(backbone.norm(teacher_tokens)[:, 0])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
            loss = supervised_loss(logits.float(), truth[ids], health[ids], teacher)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        optimizer.param_groups[0]["lr"] = 0.0001 * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
        )
        old = classification_metrics(evaluate(oldval), truth[oldval])
        sources = {}
        scores = []
        for source in new_sources:
            ids = np.array([i for i in val if rows[i]["source"] == source])
            z = evaluate(ids)
            known = truth[ids] >= 0
            m = classification_metrics(z[known], truth[ids][known])
            p = (
                (z + torch.tensor([0.0] * 8 + [0.75], device="cuda"))
                / 0.8814700841903687
            ).softmax(1)
            values, classes = p.topk(2, dim=1)
            h = (
                (classes[:, 0] == 1)
                & (values[:, 0] >= 0.6)
                & ((values[:, 0] - values[:, 1]) >= 0.2)
            )
            positive = health[ids] > 0
            m["binary_false_healthy"] = int(h[~positive].sum())
            m["binary_negative_n"] = int((~positive).sum())
            m["binary_healthy_correct"] = int(h[positive].sum())
            m["binary_positive_n"] = int(positive.sum())
            sources[source] = m
            scores.append(
                float(
                    np.mean(
                        [v for v in m["per_class_recall"].values() if v is not None]
                    )
                )
                + 0.5 * m["binary_healthy_correct"] / max(1, m["binary_positive_n"])
                - 3 * m["binary_false_healthy"] / max(1, m["binary_negative_n"])
            )
        score = (
            0.5 * float(np.mean(list(old["per_class_recall"].values())))
            + 0.5 * float(np.mean(scores))
            - old["false_healthy"] / len(oldval)
        )
        retained = all(
            old["per_class_recall"][name] >= baseline["per_class_recall"][name] - 0.02
            for name in LABELS
            if name != "Healthy"
        )
        state = dict(
            architecture="MulticlassExpert",
            mode="lora",
            rank=initial["rank"],
            depth=a.depth,
            labels=LABELS,
            method="supervised_ce_plus_binary_health",
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            training_config=config,
            epoch=epoch,
            score=score,
            old_validation=old,
            source_validation=sources,
            retention_passed=retained,
            rows_sha256=identity,
            encoder_sha256=config["encoder_sha256"],
        )
        if score > best:
            best = score
            atomic_save(state, out / "best.pt")
        if retained and score > retained_best:
            retained_best = score
            atomic_save(state, out / "retained.pt")
        atomic_save(
            dict(state, optimizer=optimizer.state_dict(), rng=capture_rng()), last
        )
        record = dict(
            epoch=epoch,
            loss=running / batches,
            score=score,
            retained=retained,
            old=old,
            sources=sources,
            seconds=time.time() - started,
        )
        history.append(record)
        (out / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False)
        )
        print(json.dumps(record, allow_nan=False), flush=True)
    for name in ["best", "retained"]:
        if (out / (name + ".pt")).exists():
            state = torch.load(
                out / (name + ".pt"), weights_only=True, map_location="cpu"
            )
            model.load_state_dict(state["model"])
            np.save(
                out / (name + "_logits.npy"),
                evaluate(np.arange(len(rows))).cpu().numpy(),
            )
    result = dict(
        best_score=best,
        retained_score=retained_best if math.isfinite(retained_best) else None,
        seconds=time.time() - started,
        parameters=sum(p.numel() for p in parameters),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
