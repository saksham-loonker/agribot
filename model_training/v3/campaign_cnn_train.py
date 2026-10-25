"""Train independent tiny CNN Healthy/knownness experts on exact pixel views."""

from pathlib import Path
import argparse, collections, hashlib, json, math, time
import numpy as np
import timm
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from training_support import atomic_save, capture_rng
from campaign_pixels import open_pixels


def sampling(rows, indices, target="health_y"):
    keys = []
    for i in indices:
        r = rows[i]
        domain = (
            r["source"]
            if r["source"] in ("tom2024", "swtt5wy2pr")
            else (
                "classification"
                if r["source"] == "bangladesh_classification"
                else (
                    "binary"
                    if r["source"] == "bangladesh_binary"
                    else "new" if r["source"] in ("tlid", "pakistan") else "original"
                )
            )
        )
        keys.append(((domain, r[target]), r["group"]))
    sizes = collections.Counter(keys)
    groups = collections.Counter(key for key, group in set(keys))
    weights = np.array(
        [1 / (groups[key] * sizes[(key, group)]) for key, group in keys], np.float64
    )
    return weights / weights.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--policy", choices=["center", "letterbox"], required=True)
    p.add_argument("--loss", choices=["bce", "focal", "reward"], default="bce")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--model", default="mobilenetv3_small_100.lamb_in1k")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--init-checkpoint")
    p.add_argument("--negative-cost", type=float, default=1.0)
    p.add_argument("--domain-selection", action="store_true")
    a = p.parse_args()
    work = Path(a.work)
    rows = json.loads((work / "rows.json").read_text())
    identity = hashlib.sha256((work / "rows.json").read_bytes()).hexdigest()
    name = f"{a.model}_{a.policy}_{a.loss}"
    if a.negative_cost != 1.0:
        name += f"_n{a.negative_cost:g}"
    dest = work / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "result.json").exists():
        return
    torch.set_num_threads(4)
    torch.manual_seed(23)
    np.random.seed(23)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = timm.create_model(
        a.model, pretrained=a.init_checkpoint is None, num_classes=2
    ).cuda()
    if a.init_checkpoint:
        initial = torch.load(a.init_checkpoint, weights_only=True, map_location="cpu")
        assert initial["architecture"] == a.model and initial["policy"] == a.policy
        model.load_state_dict(initial["model"], strict=True)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(
        1, 3, 1, 1
    )
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(
        1, 3, 1, 1
    )
    canonical_mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    canonical_std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    train = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "train" and r["health_y"] >= 0]
    )
    val = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "val" and r["health_y"] >= 0]
    )
    weights = sampling(rows, train)
    labels = torch.tensor(
        [[rows[i]["health_y"], rows[i]["known_y"]] for i in train], device="cuda"
    )
    pixels = open_pixels(work, a.policy)
    print(
        json.dumps(
            dict(
                stage="loading",
                training_images=len(train),
                validation_images=len(val),
                pixel_policy=a.policy,
                bytes=len(train) * 224 * 224 * 3 * 4,
            )
        ),
        flush=True,
    )
    images = torch.empty((len(train), 224, 224, 3), device="cuda", dtype=torch.float32)
    for start in range(0, len(train), 512):
        images[start : start + 512].copy_(
            torch.from_numpy(
                np.ascontiguousarray(pixels[train[start : start + 512]])
            ).cuda()
        )
    classifier = list(model.get_classifier().parameters())
    head_ids = {id(q) for q in classifier}
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=[q for q in model.parameters() if id(q) not in head_ids],
                lr=0.0002,
            ),
            dict(params=classifier, lr=0.001),
        ],
        weight_decay=0.0001,
    )
    best = -float("inf")
    history = []
    started = time.time()

    def evaluate(indices):
        output = []
        model.eval()
        with torch.no_grad():
            for ids in np.array_split(indices, max(1, math.ceil(len(indices) / 128))):
                x = (
                    torch.from_numpy(np.ascontiguousarray(pixels[ids]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                x = (x * canonical_std + canonical_mean - mean) / std
                output.append(model(x).cpu())
        return torch.cat(output).numpy()

    for epoch in range(a.epochs):
        for parameter in model.parameters():
            parameter.requires_grad_(epoch > 0 or id(parameter) in head_ids)
        model.train() if epoch > 0 else model.eval()
        order = torch.multinomial(
            torch.tensor(weights),
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(2300 + epoch),
        ).cuda()
        running = 0.0
        n = 0
        for ids in order.split(a.batch):
            rgb = images[ids].permute(0, 3, 1, 2) * canonical_std + canonical_mean
            if torch.rand(()) < 0.5:
                rgb = rgb.flip(-1)
            turns = int(torch.randint(0, 4, ()).item())
            if turns:
                rgb = torch.rot90(rgb, turns, (-2, -1))
            brightness = torch.empty((len(ids), 1, 1, 1), device="cuda").uniform_(
                0.85, 1.15
            )
            rgb = rgb * brightness
            average = rgb.mean((2, 3), keepdim=True)
            rgb = (rgb - average) * torch.empty(
                (len(ids), 1, 1, 1), device="cuda"
            ).uniform_(0.85, 1.15) + average
            gray = rgb.mean(1, keepdim=True)
            rgb = (rgb - gray) * torch.empty(
                (len(ids), 1, 1, 1), device="cuda"
            ).uniform_(0.9, 1.1) + gray
            z = model((rgb.clamp(0, 1) - mean) / std)
            target = labels[ids]
            health = target[:, 0]
            ce = F.binary_cross_entropy_with_logits(z[:, 0], health, reduction="none")
            if a.loss == "focal":
                ce = (
                    ce
                    * (
                        1
                        - torch.where(
                            health > 0, z[:, 0].sigmoid(), (-z[:, 0]).sigmoid()
                        )
                    )
                    ** 2
                )
            if a.loss == "reward":
                q = z[:, 0].sigmoid()
                ce = -(
                    q * torch.where(health > 0, 2.0, -8.0)
                    + (1 - q) * torch.where(health > 0, -1.0, 1.0)
                )
            valid = target[:, 1] >= 0
            known = (
                F.binary_cross_entropy_with_logits(z[valid, 1], target[valid, 1])
                if valid.any()
                else z.sum() * 0
            )
            ce = ce * torch.where(health > 0, 1.0, a.negative_cost)
            loss = ce.mean() + 0.5 * known
            if not torch.isfinite(loss):
                raise FloatingPointError(name)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 2.0, error_if_nonfinite=True
            )
            optimizer.step()
            running += float(loss.detach())
            n += 1
        for group, initial in zip(optimizer.param_groups, [0.0002, 0.001]):
            group["lr"] = initial * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
            )
        prediction = evaluate(val)
        truth = np.array([[rows[i]["health_y"], rows[i]["known_y"]] for i in val])
        metrics = {}
        for task, index in [("healthy", 0), ("known", 1)]:
            valid = truth[:, index] >= 0
            t = truth[valid, index]
            p = prediction[valid, index]
            guess = p >= 0
            auc = float(roc_auc_score(t, p))
            tpr = float(guess[t > 0].mean())
            tnr = float((~guess[t == 0]).mean())
            metrics[task] = dict(auc=auc, recall=tpr, specificity=tnr, n=len(t))
        score = (
            metrics["healthy"]["auc"]
            + 0.3
            * (metrics["healthy"]["recall"] + metrics["healthy"]["specificity"])
            / 2
            + 0.1 * metrics["known"]["auc"]
        )
        domain_metrics = {}
        if a.domain_selection:
            for source in sorted({rows[i]["source"] for i in val}):
                mask = np.array([rows[i]["source"] == source for i in val])
                t = truth[mask, 0]
                z = prediction[mask, 0]
                if len(set(t.tolist())) < 2:
                    continue
                domain_metrics[source] = dict(
                    auc=float(roc_auc_score(t, z)),
                    recall=float((z[t > 0] >= 0).mean()),
                    specificity=float((z[t == 0] < 0).mean()),
                    n=len(t),
                )
            values = list(domain_metrics.values())
            score = float(
                np.mean([m["auc"] for m in values])
                + 0.25 * min(m["auc"] for m in values)
                + 0.1 * np.mean([m["recall"] + m["specificity"] for m in values])
                + 0.1 * metrics["known"]["auc"]
            )
        record = dict(
            epoch=epoch,
            loss=running / max(1, n),
            score=score,
            metrics=metrics,
            domain_metrics=domain_metrics,
            seconds=time.time() - started,
        )
        history.append(record)
        state = dict(
            schema=1,
            architecture=a.model,
            policy=a.policy,
            loss=a.loss,
            labels=["Healthy", "Known"],
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            epoch=epoch,
            metrics=metrics,
            mean=list(cfg.get("mean", (0.485, 0.456, 0.406))),
            std=list(cfg.get("std", (0.229, 0.224, 0.225))),
            rows_sha256=identity,
            domain_selection=a.domain_selection,
        )
        if score > best:
            best = score
            atomic_save(state, dest / "best.pt")
        atomic_save(
            dict(state, optimizer=optimizer.state_dict(), rng=capture_rng()),
            dest / "last.pt",
        )
        (dest / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    state = torch.load(dest / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state["model"])
    prediction = evaluate(np.arange(len(rows)))
    np.save(dest / "all_logits.npy", prediction)
    result = dict(
        name=name,
        epochs=a.epochs,
        best_epoch=state["epoch"],
        metrics=state["metrics"],
        seconds=time.time() - started,
        pixel_policy=a.policy,
        model_parameters=sum(q.numel() for q in model.parameters()),
        data_rows_sha256=identity,
        checkpoint_sha256=hashlib.sha256((dest / "best.pt").read_bytes()).hexdigest(),
    )
    (dest / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
