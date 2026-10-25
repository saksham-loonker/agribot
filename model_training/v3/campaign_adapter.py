"""Train isolated last-block/LoRA binary experts sharing the original DINO prefix.

The original 12 transformer blocks, norm and nine-class head are never updated.
A copy of the last block learns Healthy and knownness evidence. Deployment can
share the first eleven blocks and preserve the original reference logits.
"""

from pathlib import Path
import argparse, copy, hashlib, json, math, time
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from campaign_cnn_train import sampling
from campaign_factorized import T, binary_loss
from training_support import atomic_save, load_initial_weights, capture_rng
from campaign_pixels import open_pixels


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.b.weight)
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class LastBlockExpert(nn.Module):
    def __init__(self, backbone, mode="lora", rank=8):
        super().__init__()
        self.block = copy.deepcopy(backbone.blocks[-1])
        self.norm = copy.deepcopy(backbone.norm)
        self.norm.requires_grad_(False)
        self.original_head = copy.deepcopy(backbone.head)
        self.original_head.requires_grad_(False)
        if mode == "lora":
            self.block.requires_grad_(False)
            for parent, name in [
                (self.block.attn, "qkv"),
                (self.block.attn, "proj"),
                (self.block.mlp, "fc1"),
                (self.block.mlp, "fc2"),
            ]:
                setattr(parent, name, LoRALinear(getattr(parent, name), rank))
        elif mode == "block":
            # deepcopy preserves requires_grad=False from the reference branch.
            self.block.requires_grad_(True)
        else:
            raise ValueError(mode)
        self.correction = nn.Linear(backbone.num_features, 2)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.register_buffer("bias", torch.tensor([0.0] * 8 + [0.75]))

    def forward(self, tokens):
        features = self.norm(self.block(tokens))[:, 0]
        logits = self.original_head(features)
        lp = F.log_softmax((logits + self.bias) / T, 1)
        healthy = lp[:, 1] - torch.cat([lp[:, :1], lp[:, 2:]], 1).logsumexp(1)
        known = lp[:, :8].logsumexp(1) - lp[:, 8]
        return torch.stack([healthy, known], 1) + self.correction(features)


def prefix(backbone, x, tail_blocks=1):
    x = backbone.patch_embed(x)
    x = backbone._pos_embed(x)
    x = backbone.patch_drop(x)
    x = backbone.norm_pre(x)
    for block in backbone.blocks[:-tail_blocks]:
        x = block(x)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--mode", choices=["lora", "block"], required=True)
    p.add_argument(
        "--loss", choices=["bce", "focal", "reward_ce", "reinforce"], required=True
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument(
        "--encoder",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    a = p.parse_args()
    root = Path(a.root)
    work = root / "cnn_inputs"
    name = a.mode + "_" + a.loss
    out = root / "adapters" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return
    torch.set_num_threads(4)
    torch.manual_seed(37)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    load_initial_weights(
        backbone,
        a.encoder,
        [
            "Early_blight",
            "Healthy",
            "Late_blight",
            "Leaf Miner",
            "Magnesium Deficiency",
            "Nitrogen Deficiency",
            "Pottassium Deficiency",
            "Spotted Wilt Virus",
            "Other",
        ],
        "vit_small_patch14_dinov2.lvd142m",
        224,
    )
    backbone.requires_grad_(False)
    expert = LastBlockExpert(backbone, a.mode).cuda()
    rows = json.loads((work / "rows.json").read_text())
    train = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "train" and r["health_y"] >= 0]
    )
    val = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "val" and r["health_y"] >= 0]
    )
    pixels = open_pixels(work)
    images = torch.empty((len(train), 224, 224, 3), device="cuda")
    for start in range(0, len(train), 512):
        images[start : start + 512].copy_(
            torch.from_numpy(
                np.ascontiguousarray(pixels[train[start : start + 512]])
            ).cuda()
        )
    labels = torch.tensor(
        [[rows[i]["health_y"], rows[i]["known_y"]] for i in train], device="cuda"
    )
    weights = torch.tensor(sampling(rows, train))
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    # Prefix extraction must reproduce timm's original forward_features exactly.
    with torch.no_grad():
        x = images[:3].permute(0, 3, 1, 2)
        tokens = prefix(backbone, x)
        original = backbone.head(backbone.norm(backbone.blocks[-1](tokens))[:, 0])
        torch.testing.assert_close(original, backbone(x), atol=0.0001, rtol=0.00001)
    trainable = [q for q in expert.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=0.0002 if a.mode == "lora" else 0.00003, weight_decay=0.001
    )
    base_lr = optimizer.param_groups[0]["lr"]
    best = -float("inf")
    history = []
    started = time.time()
    print(
        json.dumps(
            dict(
                stage="training",
                mode=a.mode,
                loss=a.loss,
                training=len(train),
                validation=len(val),
                trainable_parameters=sum(q.numel() for q in trainable),
            )
        ),
        flush=True,
    )

    def evaluate(indices):
        expert.eval()
        output = []
        with torch.no_grad():
            for ids in np.array_split(indices, max(1, math.ceil(len(indices) / 128))):
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(pixels[ids]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                output.append(expert(prefix(backbone, tensor)).cpu())
        return torch.cat(output).numpy()

    for epoch in range(a.epochs):
        expert.train()
        order = torch.multinomial(
            weights,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(3700 + epoch),
        ).cuda()
        running = 0.0
        count = 0
        for ids in order.split(128):
            rgb = images[ids].permute(0, 3, 1, 2) * std + mean
            if torch.rand(()) < 0.5:
                rgb = rgb.flip(-1)
            rgb = torch.rot90(rgb, int(torch.randint(0, 4, ()).item()), (-2, -1))
            rgb = (
                rgb * torch.empty((len(ids), 1, 1, 1), device="cuda").uniform_(0.9, 1.1)
            ).clamp(0, 1)
            with torch.no_grad():
                tokens = prefix(backbone, (rgb - mean) / std)
            z = expert(tokens)
            target = labels[ids]
            terms = []
            for column in [0, 1]:
                valid = target[:, column] >= 0
                if not valid.any():
                    continue
                method = a.loss if column == 0 else "bce"
                reference = z[valid, column].detach()
                if method == "reinforce":
                    # One-step REINFORCE update; knownness remains supervised.
                    q = reference.sigmoid()
                    action = torch.bernoulli(q)
                    reward = torch.where(
                        action > 0,
                        torch.where(target[valid, column] > 0, 2.0, -8.0),
                        torch.where(target[valid, column] > 0, -1.0, 1.0),
                    )
                    baseline = q * torch.where(target[valid, column] > 0, 2.0, -8.0) + (
                        1 - q
                    ) * torch.where(target[valid, column] > 0, -1.0, 1.0)
                    loss = (
                        (reward - baseline)
                        * F.binary_cross_entropy_with_logits(
                            z[valid, column], action, reduction="none"
                        )
                    ).mean()
                else:
                    loss = binary_loss(
                        z[valid, column], target[valid, column], reference, method
                    )
                terms.append(loss * (1.0 if column == 0 else 0.5))
            loss = sum(terms)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach())
            count += 1
        optimizer.param_groups[0]["lr"] = base_lr * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / a.epochs))
        )
        prediction = evaluate(val)
        truth = np.array([[rows[i]["health_y"], rows[i]["known_y"]] for i in val])
        metrics = {}
        for task, column in [("healthy", 0), ("known", 1)]:
            valid = truth[:, column] >= 0
            y = truth[valid, column]
            z = prediction[valid, column]
            guess = z >= 0
            metrics[task] = dict(
                auc=float(roc_auc_score(y, z)),
                recall=float(guess[y > 0].mean()),
                specificity=float((~guess[y == 0]).mean()),
            )
        score = (
            metrics["healthy"]["auc"]
            + 0.15 * (metrics["healthy"]["recall"] + metrics["healthy"]["specificity"])
            + 0.1 * metrics["known"]["auc"]
        )
        record = dict(
            epoch=epoch,
            loss=running / count,
            metrics=metrics,
            score=score,
            seconds=time.time() - started,
        )
        history.append(record)
        state = dict(
            architecture="LastBlockExpert",
            mode=a.mode,
            loss=a.loss,
            rank=8,
            epoch=epoch,
            model={k: v.detach().cpu().clone() for k, v in expert.state_dict().items()},
            metrics=metrics,
            encoder_sha256=hashlib.sha256(Path(a.encoder).read_bytes()).hexdigest(),
            rows_sha256=hashlib.sha256((work / "rows.json").read_bytes()).hexdigest(),
        )
        if score > best:
            best = score
            atomic_save(state, out / "best.pt")
        atomic_save(
            dict(state, optimizer=optimizer.state_dict(), rng=capture_rng()),
            out / "last.pt",
        )
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    state = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    expert.load_state_dict(state["model"])
    np.save(out / "all_logits.npy", evaluate(np.arange(len(rows))))
    result = dict(
        name=name,
        pixel_policy="center",
        best_epoch=state["epoch"],
        metrics=state["metrics"],
        trainable_parameters=sum(q.numel() for q in trainable),
        seconds=time.time() - started,
        checkpoint_sha256=hashlib.sha256((out / "best.pt").read_bytes()).hexdigest(),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
