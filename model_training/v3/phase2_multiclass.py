"""Fine-tune a copied DINO final block/head with reliable-teacher retention.

The original encoder, final block and head remain frozen for the reference
branch. This learner keeps the original nine-class initialization and adds the
newly admitted domain; it does not reset a random classifier head.
"""

from pathlib import Path
import argparse, copy, hashlib, json, math, time
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from campaign_adapter import prefix, LoRALinear
from campaign_cnn_train import sampling
from campaign_pixels import open_pixels
from campaign_heads import classification_metrics, reward_table, ppo_objective
from training_support import atomic_save, load_initial_weights, capture_rng
from train import LABELS


class MulticlassExpert(nn.Module):
    def __init__(self, backbone, mode="lora", rank=8, depth=1):
        super().__init__()
        self.depth = depth
        self.norm = copy.deepcopy(backbone.norm)
        self.norm.requires_grad_(False)
        if depth == 1:
            self.block = copy.deepcopy(backbone.blocks[-1])
            blocks = [self.block]
        else:
            self.tail = nn.Sequential(
                *[copy.deepcopy(block) for block in backbone.blocks[-depth:]]
            )
            blocks = list(self.tail)
        self.head = copy.deepcopy(backbone.head)
        self.head.requires_grad_(True)
        if mode == "lora":
            for block in blocks:
                block.requires_grad_(False)
                for parent, name in [
                    (block.attn, "qkv"),
                    (block.attn, "proj"),
                    (block.mlp, "fc1"),
                    (block.mlp, "fc2"),
                ]:
                    setattr(parent, name, LoRALinear(getattr(parent, name), rank))
        elif mode == "block":
            for block in blocks:
                block.requires_grad_(True)
        else:
            raise ValueError(mode)

    def forward(self, tokens):
        return self.head(self.features(tokens))

    def features(self, tokens):
        return self.norm(self.block(tokens) if self.depth == 1 else self.tail(tokens))[
            :, 0
        ]


def adaptation_loss(
    method, logits, target, prior=None, old_logits=None, generator=None
):
    """Common supervised objective with separately controlled auxiliary arms.

    PPO is a one-step label-reward bandit. Its old policy is frozen for each
    epoch and its action sampling has a separate RNG from image augmentation.
    """
    if method == "logit":
        assert prior is not None
        return F.cross_entropy(logits + 0.5 * prior.log(), target)
    ce = F.cross_entropy(logits, target)
    if method == "ce":
        return ce
    if method == "reward_ce":
        return ce - 0.1 * (logits.softmax(1) * reward_table(target)).sum(1).mean()
    if method == "ppo":
        assert old_logits is not None
        old_probs = old_logits.detach().softmax(1)
        action = torch.multinomial(old_probs, 1, generator=generator).flatten()
        rewards = reward_table(target)
        baseline = (old_probs * rewards).sum(1)
        advantage = (rewards.gather(1, action[:, None]).flatten() - baseline).detach()
        logp = logits.log_softmax(1).gather(1, action[:, None]).flatten()
        old_logp = (
            old_logits.detach().log_softmax(1).gather(1, action[:, None]).flatten()
        )
        return ce + 0.1 * ppo_objective(logp, old_logp, advantage).mean()
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--mode", choices=["lora", "block"], required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--retention-weight", type=float, default=1.0)
    p.add_argument(
        "--method", choices=["ce", "logit", "reward_ce", "ppo"], default="ce"
    )
    a = p.parse_args()
    root = Path(a.root)
    work = root / "cnn_inputs"
    out = root / "multiclass" / (a.mode if a.depth == 1 else a.mode + f"_d{a.depth}")
    if a.method != "ce":
        out = out.with_name(out.name + "_" + a.method)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return
    torch.set_num_threads(4)
    torch.manual_seed(53)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads((work / "rows.json").read_text())
    train = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["split"] == "train" and r["label"] in LABELS
        ]
    )
    val = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "val" and r["label"] in LABELS]
    )
    oldval = np.array(
        [
            i
            for i in val
            if rows[i]["source"]
            not in (
                "tlid",
                "pakistan",
                "bangladesh_classification",
                "bangladesh_binary",
            )
        ]
    )
    novelval = np.array(
        [i for i in val if rows[i]["source"] == "bangladesh_classification"]
    )
    assert len(oldval) and len(novelval)
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
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
    checkpoint = "/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt"
    load_initial_weights(
        backbone, checkpoint, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    backbone.requires_grad_(False)
    model = MulticlassExpert(backbone, a.mode, depth=a.depth).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=0.0002 if a.mode == "lora" else 0.00003, weight_decay=0.001
    )
    initial_lr = optimizer.param_groups[0]["lr"]
    pixels = open_pixels(work)
    images = torch.empty((len(train), 224, 224, 3), device="cuda")
    for start in range(0, len(train), 512):
        images[start : start + 512].copy_(
            torch.from_numpy(
                np.ascontiguousarray(pixels[train[start : start + 512]])
            ).cuda()
        )
    labels = torch.tensor(
        [LABELS.index(rows[i]["label"]) for i in train], device="cuda"
    )
    prior = torch.bincount(labels, minlength=9).float() + 1
    prior = prior / prior.sum()
    action_generator = torch.Generator(device="cuda").manual_seed(5400)
    weights = torch.tensor(sampling(rows, train, target="label"))
    truth = torch.tensor(
        [LABELS.index(r["label"]) if r["label"] in LABELS else -1 for r in rows],
        device="cuda",
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    reference = torch.from_numpy(np.load(root / "reference_logits.npy")).cuda()
    baseline = classification_metrics(reference[oldval], truth[oldval])
    history = []
    best = -float("inf")
    retained_best = -float("inf")
    started = time.time()

    def evaluate(indices):
        model.eval()
        output = []
        with torch.no_grad():
            for ids in np.array_split(indices, max(1, math.ceil(len(indices) / 128))):
                x = (
                    torch.from_numpy(np.ascontiguousarray(pixels[ids]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                output.append(model(prefix(backbone, x, a.depth)).cpu())
        return torch.cat(output).cuda()

    print(
        json.dumps(
            dict(
                stage="training",
                mode=a.mode,
                train=len(train),
                validation=len(val),
                parameters=sum(p.numel() for p in parameters),
                original_baseline=baseline,
            )
        ),
        flush=True,
    )
    for epoch in range(a.epochs):
        old_policy = (
            copy.deepcopy(model).eval().requires_grad_(False)
            if a.method == "ppo"
            else None
        )
        model.train()
        order = torch.multinomial(
            weights,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(5300 + epoch),
        ).cuda()
        running = 0.0
        batches = 0
        for ids in order.split(a.batch):
            rgb = images[ids].permute(0, 3, 1, 2) * std + mean
            if torch.rand(()) < 0.5:
                rgb = rgb.flip(-1)
            rgb = torch.rot90(rgb, int(torch.randint(0, 4, ()).item()), (-2, -1))
            rgb = (
                rgb * torch.empty((len(ids), 1, 1, 1), device="cuda").uniform_(0.9, 1.1)
            ).clamp(0, 1)
            with torch.no_grad():
                tokens = prefix(backbone, (rgb - mean) / std, a.depth)
                teacher_tokens = tokens
                for block in backbone.blocks[-a.depth :]:
                    teacher_tokens = block(teacher_tokens)
                teacher = backbone.head(backbone.norm(teacher_tokens)[:, 0])
            logits = model(tokens)
            target = labels[ids]
            with torch.no_grad():
                old_logits = old_policy(tokens) if old_policy is not None else None
            loss = adaptation_loss(
                a.method, logits, target, prior, old_logits, action_generator
            )
            reliable = (teacher.argmax(1) == target) & (target != 1)
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
                    + a.retention_weight
                    * torch.stack(
                        [
                            kl[target[reliable] == k].mean()
                            for k in target[reliable].unique()
                        ]
                    ).mean()
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        optimizer.param_groups[0]["lr"] = initial_lr * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
        )
        old = classification_metrics(evaluate(oldval), truth[oldval])
        novel = classification_metrics(evaluate(novelval), truth[novelval])
        old_macro = float(np.mean(list(old["per_class_recall"].values())))
        novel_macro = float(
            np.mean([v for v in novel["per_class_recall"].values() if v is not None])
        )
        score = (
            0.5 * (old_macro + novel_macro)
            + 0.25
            * (
                old["per_class_recall"]["Healthy"]
                + novel["per_class_recall"]["Healthy"]
            )
            - old["false_healthy"] / len(oldval)
            - novel["false_healthy"] / len(novelval)
        )
        retain = all(
            old["per_class_recall"][name] >= baseline["per_class_recall"][name] - 0.02
            for name in LABELS
            if name != "Healthy"
        )
        state = dict(
            architecture="MulticlassExpert",
            method=a.method,
            class_prior=prior.detach().cpu().tolist(),
            mode=a.mode,
            rank=8,
            depth=a.depth,
            retention_weight=a.retention_weight,
            labels=LABELS,
            epoch=epoch,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            old_validation=old,
            novel_validation=novel,
            retention_passed=retain,
            score=score,
            rows_sha256=hashlib.sha256((work / "rows.json").read_bytes()).hexdigest(),
            encoder_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        )
        if score > best:
            best = score
            atomic_save(state, out / "best.pt")
        if retain and score > retained_best:
            retained_best = score
            atomic_save(state, out / "retained.pt")
        atomic_save(
            dict(
                state,
                optimizer=optimizer.state_dict(),
                rng=capture_rng(),
                action_rng=action_generator.get_state(),
            ),
            out / "last.pt",
        )
        record = dict(
            epoch=epoch,
            loss=running / batches,
            score=score,
            retention_passed=retain,
            old=old,
            novel=novel,
            seconds=time.time() - started,
        )
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    for name in ["best", "retained"]:
        file = out / (name + ".pt")
        if file.exists():
            state = torch.load(file, weights_only=True, map_location="cpu")
            model.load_state_dict(state["model"])
            np.save(
                out / (name + "_logits.npy"),
                evaluate(np.arange(len(rows))).cpu().numpy(),
            )
    result = dict(
        mode=a.mode,
        method=a.method,
        best_score=best,
        retained_score=retained_best if math.isfinite(retained_best) else None,
        seconds=time.time() - started,
        trainable_parameters=sum(p.numel() for p in parameters),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
