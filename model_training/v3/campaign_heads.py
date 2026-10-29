"""Paired loss/architecture ablations over exact-canonical frozen DINO features.

RL arms are one-step contextual bandits with label-derived rewards, not a claim
of human feedback or an interactive farming simulator. Checkpoints are selected
on the existing grouped validation partition. Opened app sets are regressions.
"""

from pathlib import Path
import argparse, copy, hashlib, json, math, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train import LABELS
from app_eval import replay, posterior, summarize as app_metrics

H, O = 1, 8
METHODS = [
    "ce",
    "healthy_ce",
    "weighted",
    "logit",
    "focal",
    "smooth",
    "margin",
    "retention",
    "expected_reward",
    "reward_ce",
    "reinforce",
    "ppo",
    "dpo",
    "mixup",
    "supcon",
]


class FeatureHead(nn.Module):
    def __init__(self, original, architecture):
        super().__init__()
        self.architecture = architecture
        self.base = nn.Linear(384, 9)
        self.base.load_state_dict(original)
        if architecture == "residual_mlp":
            self.base.requires_grad_(False)
            self.hidden = nn.Sequential(nn.Linear(384, 128), nn.GELU())
            self.output = nn.Linear(128, 9)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        z = self.base(x)
        return (
            z + self.output(self.hidden(x))
            if self.architecture == "residual_mlp"
            else z
        )


def reward_table(labels):
    """False Healthy and false rejection of known conditions carry explicit cost."""
    table = torch.full((len(labels), 9), -1.0, device=labels.device)
    table[:, O] = torch.where(labels == O, 1.0, -3.0)
    table[labels == O, :O] = -4.0
    table[:, H] = torch.where(labels == H, 2.0, -8.0)
    table.scatter_(1, labels[:, None], torch.where(labels == H, 2.0, 1.0)[:, None])
    return table


def ppo_objective(logp, old_logp, advantage, clip=0.2):
    ratio = (logp - old_logp).exp()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    )


def supervised_contrastive(hidden, labels):
    hidden = F.normalize(hidden, dim=1)
    scores = hidden @ hidden.T / 0.1
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    logp = scores - scores.masked_fill(diagonal, -1e9).logsumexp(1, keepdim=True)
    counts = positive.sum(1)
    valid = counts > 0
    return (
        -(logp.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1))[valid].mean()
        if valid.any()
        else hidden.sum() * 0
    )


def loss_value(method, z, y, reference, weight, hidden=None):
    ce = F.cross_entropy(z, y, reduction="none")
    rows = torch.arange(len(y), device=y.device)
    if method in ("ce", "healthy_ce", "weighted", "retention", "ppo", "mixup"):
        terms = ce
    elif method == "logit":
        prior = weight["prior"]
        terms = F.cross_entropy(z + 0.5 * prior.log(), y, reduction="none")
    elif method == "focal":
        terms = (1 - z.softmax(1)[rows, y]) ** 2 * ce
    elif method == "smooth":
        terms = F.cross_entropy(z, y, label_smoothing=0.05, reduction="none")
    elif method == "margin":
        adjusted = z - F.one_hot(y, 9) * 0.3
        terms = F.cross_entropy(adjusted, y, reduction="none")
    elif method in ("expected_reward", "reward_ce"):
        terms = -(z.softmax(1) * reward_table(y)).sum(1)
        if method == "reward_ce":
            terms = terms + 0.2 * ce
    elif method == "reinforce":
        probs = z.softmax(1)
        action = torch.multinomial(probs.detach(), 1).squeeze(1)
        rewards = reward_table(y)
        baseline = (probs.detach() * rewards).sum(1)
        terms = -(rewards[rows, action] - baseline) * z.log_softmax(1)[rows, action]
    elif method == "dpo":
        rejected = reference.masked_fill(F.one_hot(y, 9).bool(), -1e9).argmax(1)
        margin = (z[rows, y] - z[rows, rejected]) - (
            reference[rows, y] - reference[rows, rejected]
        )
        terms = -F.logsigmoid(0.5 * margin)
    elif method == "supcon":
        terms = ce
    else:
        raise ValueError(method)
    loss = (terms * weight["sample"]).mean()
    if method == "retention":
        reliable = (y != H) & (reference.argmax(1) == y)
        if reliable.any():
            kl = (
                F.kl_div(
                    F.log_softmax(z[reliable] / 2, 1),
                    F.softmax(reference[reliable] / 2, 1),
                    reduction="none",
                ).sum(1)
                * 4
            )
            present = y[reliable].unique()
            loss = (
                loss
                + torch.stack([kl[y[reliable] == k].mean() for k in present]).mean()
            )
    if method == "supcon":
        loss = loss + 0.1 * supervised_contrastive(hidden, y)
    return loss


def classification_metrics(z, truth):
    bias = torch.tensor([0.0] * 8 + [0.75], device=z.device)
    p = ((z + bias) / 0.8814700841903687).softmax(1)
    values, order = p.topk(2, dim=1)
    pred = order[:, 0]
    h = (pred == H) & (values[:, 0] >= 0.6) & ((values[:, 0] - values[:, 1]) >= 0.2)
    other = (pred == O) & (values[:, 0] >= 0.5)
    disease = (
        (pred != H)
        & (pred != O)
        & (values[:, 0] >= 0.5)
        & ((values[:, 0] - values[:, 1]) >= 0.15)
    )
    correct = (h | (disease)) & (pred == truth)
    per = {
        l: float((pred[truth == i] == i).float().mean()) if (truth == i).any() else None
        for i, l in enumerate(LABELS)
    }
    return dict(
        accuracy=float((pred == truth).float().mean()),
        per_class_recall=per,
        healthy_n=int((truth == H).sum()),
        healthy_correct=int(h[truth == H].sum()),
        false_healthy=int(h[truth != H].sum()),
        false_other=int(other[truth != O].sum()),
        supported_correct=int(correct[truth != O].sum()),
        unsupported_false_known=int((h | disease)[truth == O].sum()),
    )


def val_score(metrics):
    recalls = [v for v in metrics["per_class_recall"].values() if v is not None]
    return (
        sum(recalls) / max(1, len(recalls))
        + 0.5 * metrics["healthy_correct"] / max(1, metrics["healthy_n"])
        - 0.025 * metrics["false_healthy"]
        - 0.005 * metrics["false_other"]
    )


def retention_ok(candidate, base):
    return candidate["false_healthy"] <= base["false_healthy"] and all(
        base["per_class_recall"][name] is None
        or (
            candidate["per_class_recall"][name] is not None
            and candidate["per_class_recall"][name]
            >= base["per_class_recall"][name] - 0.02
        )
        for name in LABELS
        if name != "Healthy"
    )


def fast_static_replay(leaves, manifest, weight):
    if len(leaves) != 1:
        return replay(leaves, manifest, weight)
    probabilities = posterior(leaves[0]["member_logits"], manifest)
    order = np.argsort(-probabilities, kind="stable")
    top, second = map(int, order[:2])
    p1, p2 = probabilities[top], probabilities[second]
    margin = p1 - p2
    if top == O:
        kind = "UNKNOWN_CONDITION" if p1 >= np.float32(0.5) else "UNSURE"
    elif top == H:
        kind = (
            "HEALTHY"
            if p1 >= np.float32(0.6) and margin >= np.float32(0.2)
            else "UNSURE"
        )
    else:
        kind = (
            "DISEASE"
            if p1 >= np.float32(0.5) and margin >= np.float32(0.15)
            else "UNSURE"
        )
    return dict(
        kind=kind,
        label=LABELS[top],
        confidence=float(p1),
        frames=14 if kind == "UNSURE" else 7,
        leaves=1,
        partial=False,
        probabilities=probabilities.astype(np.float64).tolist(),
        margin=float(margin),
    )


def app_evaluate(logits, jobs, eval_rows, manifest, return_rows=False):
    outputs = {}
    for name, original_rows in eval_rows.items():
        rows = copy.deepcopy(original_rows)
        for i, job in enumerate(jobs):
            if job["set"] == name:
                rows[job["row"]]["leaves"][job["leaf"]]["member_logits"] = [
                    logits[i].tolist()
                ]
        for row in rows:
            row.pop(
                "oracle_objects", None
            )  # Oracle object logits are not part of this feature cache.
            row["app_replay"] = fast_static_replay(
                row["leaves"], manifest, row["quality_weight"]
            )
            if row["contrast"] < 0.015 and row["sharpness"] < 0.012:
                row["app_replay"] = dict(
                    kind="RETRY_QUALITY",
                    label=None,
                    confidence=None,
                    frames=0,
                    leaves=0,
                    partial=False,
                )
        outputs[name] = (
            dict(metrics=app_metrics(rows), rows=rows)
            if return_rows
            else app_metrics(rows)
        )
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--architectures", default="linear,residual_mlp")
    p.add_argument("--healthy-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    work = Path(a.work)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(work / "features.pt", weights_only=True, map_location="cpu")
    jobs = json.loads((work / "feature_rows.json").read_text())
    eval_rows = json.loads((work / "eval_rows.json").read_text())
    assert (
        hashlib.sha256((work / "feature_rows.json").read_bytes()).hexdigest()
        == cache["rows_sha256"]
    )
    x = cache["features"].cuda()
    reference = cache["logits"].cuda()
    labels = torch.tensor(
        [LABELS.index(r["label"]) if r["label"] in LABELS else -1 for r in jobs],
        device="cuda",
    )
    train = torch.tensor(
        [
            i
            for i, r in enumerate(jobs)
            if r["set"] == "training" and r["split"] == "train"
        ],
        device="cuda",
    )
    val = torch.tensor(
        [
            i
            for i, r in enumerate(jobs)
            if r["set"] == "training" and r["split"] == "val"
        ],
        device="cuda",
    )
    assert not (
        {jobs[i]["group"] for i in train.cpu().tolist()}
        & {jobs[i]["group"] for i in val.cpu().tolist()}
    )
    app_pairs = [
        (i, r)
        for i, r in enumerate(jobs)
        if r["set"] not in ("training", "selection_app")
    ]
    error = max(
        float(
            (
                reference[i].cpu()
                - torch.tensor(
                    eval_rows[r["set"]][r["row"]]["leaves"][r["leaf"]]["member_logits"][
                        0
                    ]
                )
            )
            .abs()
            .max()
        )
        for i, r in app_pairs
    )
    assert error < 0.002, f"Exact canonical feature/reference parity failed: {error}"
    base = classification_metrics(reference[val], labels[val])
    print(
        json.dumps(dict(stage="baseline", validation=base, app_logit_parity_max=error)),
        flush=True,
    )
    (work / "canonical_baseline.json").write_text(
        json.dumps(dict(validation=base, app_logit_parity_max=error), indent=2)
    )
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )
    counts = torch.bincount(labels[train], minlength=9).float()
    prior = counts / counts.sum()
    for arch in a.architectures.split(","):
        for method in a.methods.split(","):
            if method == "supcon" and arch == "linear":
                continue
            name = f"{arch}_{method}_h{a.healthy_weight:g}_s{a.seed}"
            dest = work / "heads" / name
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "result.json").exists():
                continue
            torch.manual_seed(a.seed)
            model = FeatureHead(cache["original_head"], arch).cuda()
            optimizer = torch.optim.AdamW(
                [q for q in model.parameters() if q.requires_grad],
                lr=3e-4,
                weight_decay=0.01,
            )
            cw = torch.ones(9, device="cuda")
            cw[H] = a.healthy_weight if method != "ce" else 1.0
            if method == "weighted":
                cw *= torch.sqrt(counts.sum() / (9 * counts)).clamp(0.5, 4.0)
            best_score = -float("inf")
            guard_score = val_score(base)
            history = []
            started = time.time()
            for epoch in range(a.epochs):
                gen = torch.Generator(device="cuda").manual_seed(a.seed + 1000 + epoch)
                order = train[torch.randperm(len(train), device="cuda", generator=gen)]
                running = 0.0
                for ids in order.split(512):
                    xb, y, ref = x[ids], labels[ids], reference[ids]
                    if method == "ppo":
                        with torch.no_grad():
                            old = model(xb).log_softmax(1)
                            actions = torch.multinomial(old.exp(), 1).squeeze(1)
                            rows = torch.arange(len(y), device="cuda")
                            rewards = reward_table(y)
                            advantage = rewards[rows, actions] - (
                                old.exp() * rewards
                            ).sum(1)
                            old_action = old[rows, actions]
                        for _ in range(4):
                            z = model(xb)
                            logp = z.log_softmax(1)
                            loss = (
                                ppo_objective(
                                    logp[rows, actions], old_action, advantage
                                )
                                * cw[y]
                            ).mean()
                            loss = loss + 0.03 * F.kl_div(
                                logp, ref.softmax(1), reduction="batchmean"
                            )
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                    else:
                        z = model(xb)
                        if method == "mixup":
                            permutation = torch.randperm(len(y), device="cuda")
                            lam = 0.3 + 0.4 * float(torch.rand(()))
                            mixed = model(lam * xb + (1 - lam) * xb[permutation])
                            loss = (
                                lam * F.cross_entropy(mixed, y, reduction="none")
                                + (1 - lam)
                                * F.cross_entropy(
                                    mixed, y[permutation], reduction="none"
                                )
                            ).mean()
                        else:
                            hidden = model.hidden(xb) if method == "supcon" else None
                            loss = loss_value(
                                method,
                                z,
                                y,
                                ref,
                                dict(prior=prior, sample=cw[y]),
                                hidden,
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(name)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    running += float(loss.detach())
                with torch.no_grad():
                    metrics = classification_metrics(model(x[val]), labels[val])
                score = val_score(metrics)
                eligible = retention_ok(metrics, base)
                row = dict(
                    epoch=epoch,
                    loss=running / math.ceil(len(train) / 512),
                    score=score,
                    retention_passed=eligible,
                    **metrics,
                )
                history.append(row)
                state = dict(
                    architecture=arch,
                    method=method,
                    model={
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    epoch=epoch,
                    validation=metrics,
                    labels=LABELS,
                    healthy_weight=a.healthy_weight,
                    seed=a.seed,
                )
                if score > best_score:
                    best_score = score
                    torch.save(state, dest / "best.pt")
                if eligible and score > guard_score:
                    guard_score = score
                    torch.save(state, dest / "retained.pt")
            state = torch.load(dest / "best.pt", weights_only=True)
            model.load_state_dict(state["model"])
            with torch.no_grad():
                predicted = model(x).cpu().numpy()
            result = dict(
                name=name,
                architecture=arch,
                method=method,
                epochs=a.epochs,
                seconds=round(time.time() - started, 2),
                selected_epoch=state["epoch"],
                validation=state["validation"],
                validation_retention_passed=retention_ok(state["validation"], base),
                guarded_checkpoint=(dest / "retained.pt").exists(),
                app=app_evaluate(predicted, jobs, eval_rows, manifest),
            )
            (dest / "history.json").write_text(json.dumps(history, indent=2))
            (dest / "result.json").write_text(json.dumps(result, indent=2))
            compact = {
                name: dict(
                    healthy=f"{v['healthy']['correct']}/{v['healthy']['n']}",
                    disease=f"{v['supported_disease']['correct']}/{v['supported_disease']['n']}",
                    false_healthy=v["supported_disease"]["false_healthy"],
                    false_other=v["supported"]["false_other"],
                )
                for name, v in result["app"].items()
            }
            print(
                json.dumps(
                    dict(
                        trial=name,
                        seconds=result["seconds"],
                        validation=result["validation"],
                        app=compact,
                    )
                ),
                flush=True,
            )
    print("All requested loss/architecture ablations complete", flush=True)


if __name__ == "__main__":
    main()
