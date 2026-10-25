"""Train isolated Healthy/Miner and known/Other corrections, retaining base scores.

The original backbone and nine-class head are frozen. Original non-Miner disease
winners retain their exact distribution; confident Miner cannot be weakened by
the Healthy correction. The Other
correction applies only when a learned knownness specialist reaches 0.9. These
are prospective constraints, not thresholds fitted to opened test mistakes.
"""

from pathlib import Path
import argparse, copy, hashlib, json, math, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train import LABELS
from campaign_heads import (
    app_evaluate,
    classification_metrics,
    val_score,
    retention_ok,
    ppo_objective,
)
from app_eval import summarize

H, M, O, T = 1, 3, 8, 0.8814700841903687


class FactorizedHead(nn.Module):
    def __init__(self, original, mode="both", architecture="mlp", feature_dim=384):
        super().__init__()
        self.mode = mode
        self.architecture = architecture
        self.base = nn.Linear(384, 9)
        self.base.load_state_dict(original)
        self.base.requires_grad_(False)
        self.register_buffer("bias", torch.tensor([0.0] * 8 + [0.75]))

        def correction():
            net = (
                nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Linear(64, 1))
                if architecture == "mlp"
                else nn.Sequential(nn.Linear(feature_dim, 1))
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.pair = correction()
        self.known = correction()
        self.pair.requires_grad_(mode in ("pair", "both"))
        self.known.requires_grad_(mode in ("known", "both"))

    def raw(self, x):
        z = self.base(x[:, :384])
        lp = F.log_softmax((z + self.bias) / T, 1)
        pair = lp[:, H] - lp[:, M] + self.pair(x).squeeze(1)
        known = lp[:, :O].logsumexp(1) - lp[:, O] + self.known(x).squeeze(1)
        return z, lp, pair, known

    def forward(self, x):
        return self.combine(*self.raw(x))

    def combine(self, z, lp, pair, known):
        winner = lp.argmax(1)
        known_winner = lp[:, :O].argmax(1)
        change_known = (
            (winner == O) & (known.sigmoid() >= 0.9)
            if self.mode in ("known", "both")
            else torch.zeros_like(winner, dtype=torch.bool)
        )
        eligible = ((known_winner == H) | (known_winner == M)) & (
            ((winner == H) | (winner == M)) | change_known
        )
        delta = pair - (lp[:, H] - lp[:, M])
        delta = torch.where(lp[:, M].exp() >= 0.7, delta.clamp_max(0), delta)
        delta = (
            torch.where(eligible, delta, torch.zeros_like(delta))
            if self.mode in ("pair", "both")
            else torch.zeros_like(delta)
        )
        corrected_pair = lp[:, H] - lp[:, M] + delta
        mass = torch.stack([lp[:, H], lp[:, M]], 1).logsumexp(1)
        new_h = mass + F.logsigmoid(corrected_pair)
        new_m = mass + F.logsigmoid(-corrected_pair)
        columns = [new_h if i == H else new_m if i == M else lp[:, i] for i in range(9)]
        adjusted = torch.stack(columns, 1)
        old_mass = lp[:, :O].logsumexp(1)
        known_shift = torch.where(
            change_known, F.logsigmoid(known) - old_mass, torch.zeros_like(known)
        )
        known_columns = adjusted[:, :O] + known_shift[:, None]
        other = torch.where(change_known, F.logsigmoid(-known), lp[:, O])
        modified = z + T * (torch.cat([known_columns, other[:, None]], 1) - lp)
        return torch.where(((delta != 0) | change_known)[:, None], modified, z)


def binary_loss(logit, target, reference, method, negative_cost=1.0):
    weights = torch.where(
        target > 0,
        0.5 / target.mean().clamp_min(0.01),
        0.5 / (1 - target.mean()).clamp_min(0.01),
    )
    weights = weights * torch.where(target > 0, 1.0, negative_cost)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if method in ("bce", "ppo"):
        loss = ce
    elif method == "focal":
        pt = torch.where(target > 0, logit.sigmoid(), (-logit).sigmoid())
        loss = (1 - pt) ** 2 * ce
    elif method == "dpo":
        loss = -F.logsigmoid(0.5 * (logit - reference) * (target * 2 - 1))
    elif method in ("reward", "reward_ce", "reinforce"):
        q = logit.sigmoid()
        r_positive = torch.where(target > 0, 2.0, -8.0)
        r_negative = torch.where(target > 0, -1.0, 1.0)
        if method == "reinforce":
            action = torch.bernoulli(q.detach())
            reward = torch.where(action > 0, r_positive, r_negative)
            baseline = q.detach() * r_positive + (1 - q.detach()) * r_negative
            loss = (reward - baseline) * F.binary_cross_entropy_with_logits(
                logit, action, reduction="none"
            )
        else:
            loss = -(q * r_positive + (1 - q) * r_negative)
            if method == "reward_ce":
                loss = loss + 0.2 * ce
    else:
        raise ValueError(method)
    return (weights * loss).mean()


def selection_evaluate(logits, jobs, eval_rows, manifest):
    rows = app_evaluate(
        logits, jobs, {"selection_app": eval_rows["selection_app"]}, manifest, True
    )["selection_app"]["rows"]
    return dict(
        all=summarize(rows),
        original=summarize(
            [r for r in rows if r["source"] not in ("tlid", "pakistan")]
        ),
        new=summarize([r for r in rows if r["source"] in ("tlid", "pakistan")]),
    )


def selection_score(metrics):
    def score(m):
        return (
            (m["healthy"]["correct_rate"] or 0)
            + (m["supported_disease"]["correct_rate"] or 0)
            - 3 * (m["all"]["false_healthy_rate"] or 0)
            - (m["supported"]["false_other_rate"] or 0)
            - 2 * (m["unsupported_tomato"]["false_known_rate"] or 0)
            - 2 * (m["non_tomato"]["false_known_rate"] or 0)
        )

    return 0.5 * score(metrics["original"]) + 0.5 * score(metrics["new"])


def selection_retained(candidate, base):
    return (
        candidate["original"]["healthy"]["correct"]
        >= base["original"]["healthy"]["correct"]
        and candidate["new"]["healthy"]["correct"] >= base["new"]["healthy"]["correct"]
        and candidate["original"]["all"]["false_healthy"]
        <= base["original"]["all"]["false_healthy"]
        and candidate["original"]["supported_disease"]["correct"]
        >= base["original"]["supported_disease"]["correct"]
        and all(
            (candidate["original"][scope]["false_known"] or 0)
            <= (base["original"][scope]["false_known"] or 0)
            for scope in ("unsupported_tomato", "non_tomato")
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--methods", default="bce,focal,dpo,reward,reward_ce,reinforce,ppo")
    p.add_argument("--modes", default="pair,known,both")
    p.add_argument("--architectures", default="linear,mlp")
    p.add_argument("--negative-cost", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--domain-balanced", action="store_true")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    work = Path(a.work)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(work / "features.pt", weights_only=True)
    assert cache.get("strict_fp32_cuda")
    jobs = json.loads((work / "feature_rows.json").read_text())
    eval_rows = json.loads((work / "eval_rows.json").read_text())
    x = cache["features"].cuda()
    ref = cache["logits"].cuda()
    truth = torch.tensor(
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
    oldval = torch.tensor(
        [
            i
            for i, r in enumerate(jobs)
            if r["set"] == "training"
            and r["split"] == "val"
            and r["source"] not in ("tlid", "pakistan")
        ],
        device="cuda",
    )
    base = classification_metrics(ref[val], truth[val])
    oldbase = classification_metrics(ref[oldval], truth[oldval])
    base_lp = F.log_softmax(
        (ref + torch.tensor([0.0] * 8 + [0.75], device="cuda")) / T, 1
    )
    pair_train = train[(truth[train] == H) | (truth[train] == M)]
    # Add original reference confusables as not-Healthy, without claiming they are Miner labels.
    hard = train[(truth[train] == O) & (base_lp[train, H].exp() > 0.15)]
    pair_train = torch.cat([pair_train, hard])
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )
    selection_base = (
        selection_evaluate(ref.cpu().numpy(), jobs, eval_rows, manifest)
        if "selection_app" in eval_rows
        else None
    )
    if selection_base:
        (work / "selection_baseline.json").write_text(
            json.dumps(selection_base, indent=2)
        )
    for arch in a.architectures.split(","):
        for mode in a.modes.split(","):
            for method in a.methods.split(","):
                name = f"{arch}_{mode}_{method}_n{a.negative_cost:g}_s{a.seed}" + (
                    "_" + a.tag if a.tag else ""
                )
                dest = work / "factorized" / name
                dest.mkdir(parents=True, exist_ok=True)
                if (dest / "result.json").exists():
                    continue
                torch.manual_seed(a.seed)
                model = FactorizedHead(
                    cache["original_head"], mode, arch, x.shape[1]
                ).cuda()
                optimizer = torch.optim.AdamW(
                    [q for q in model.parameters() if q.requires_grad],
                    lr=0.001,
                    weight_decay=0.01,
                )
                best_score = -float("inf")
                guard_score = (
                    selection_score(selection_base) if selection_base else -float("inf")
                )
                history = []
                started = time.time()
                for epoch in range(a.epochs):
                    gen = torch.Generator(device="cuda").manual_seed(
                        a.seed + epoch + 2000
                    )
                    indices = pair_train if mode == "pair" else train
                    if a.domain_balanced:
                        domain = torch.tensor(
                            [
                                int(jobs[i]["source"] in ("tlid", "pakistan"))
                                for i in indices.cpu().tolist()
                            ],
                            device="cuda",
                        )
                        count = torch.bincount(domain, minlength=2).clamp_min(1)
                        weights = 1 / count[domain].float()
                        indices = indices[
                            torch.multinomial(
                                weights, len(indices), replacement=True, generator=gen
                            )
                        ]
                    else:
                        indices = indices[
                            torch.randperm(len(indices), device="cuda", generator=gen)
                        ]
                    running = 0.0
                    for ids in indices.split(512):
                        z, lp, pair, known = model.raw(x[ids])
                        y = truth[ids]
                        pair_mask = (
                            (y == H)
                            | (y == M)
                            | ((y == O) & (base_lp[ids, H].exp() > 0.15))
                        )
                        tasks = []
                        if mode in ("pair", "both") and pair_mask.any():
                            tasks.append(
                                (
                                    pair[pair_mask],
                                    (y[pair_mask] == H).float(),
                                    lp[pair_mask, H] - lp[pair_mask, M],
                                    pair_mask,
                                )
                            )
                        if mode in ("known", "both"):
                            tasks.append(
                                (
                                    known,
                                    (y != O).float(),
                                    lp[:, :O].logsumexp(1) - lp[:, O],
                                    torch.ones_like(y, dtype=torch.bool),
                                )
                            )
                        if method == "ppo":
                            rollout = []
                            for logit, target, reference, mask in tasks:
                                oldq = logit.detach().sigmoid()
                                action = torch.bernoulli(oldq)
                                rpos = torch.where(target > 0, 2.0, -8.0)
                                rneg = torch.where(target > 0, -1.0, 1.0)
                                advantage = torch.where(action > 0, rpos, rneg) - (
                                    oldq * rpos + (1 - oldq) * rneg
                                )
                                oldlogp = -F.binary_cross_entropy_with_logits(
                                    logit.detach(), action, reduction="none"
                                )
                                rollout.append(
                                    (
                                        action,
                                        oldlogp,
                                        advantage,
                                        reference.detach(),
                                        mask,
                                    )
                                )
                            for step in range(4):
                                _, lp, pair, known = model.raw(x[ids])
                                current = []
                                if mode in ("pair", "both") and pair_mask.any():
                                    current.append(pair[pair_mask])
                                if mode in ("known", "both"):
                                    current.append(known)
                                loss = torch.zeros((), device="cuda")
                                for now, (
                                    action,
                                    oldlogp,
                                    advantage,
                                    reference,
                                    mask,
                                ) in zip(current, rollout):
                                    logp = -F.binary_cross_entropy_with_logits(
                                        now, action, reduction="none"
                                    )
                                    oldprob = reference.sigmoid()
                                    kl = F.binary_cross_entropy_with_logits(
                                        now, oldprob, reduction="mean"
                                    ) - F.binary_cross_entropy_with_logits(
                                        reference, oldprob, reduction="mean"
                                    )
                                    loss = (
                                        loss
                                        + ppo_objective(logp, oldlogp, advantage).mean()
                                        + 0.03 * kl
                                    )
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                        else:
                            loss = sum(
                                binary_loss(
                                    logit, target, reference, method, a.negative_cost
                                )
                                for logit, target, reference, mask in tasks
                            )
                            if not torch.isfinite(loss):
                                raise FloatingPointError(name)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        running += float(loss.detach())
                    if epoch % 10 == 9 or epoch == a.epochs - 1:
                        with torch.no_grad():
                            v = classification_metrics(model(x[val]), truth[val])
                            old = classification_metrics(
                                model(x[oldval]), truth[oldval]
                            )
                        score = 0.5 * val_score(v) + 0.5 * val_score(old)
                        eligible = retention_ok(old, oldbase)
                        selection = None
                        if selection_base:
                            with torch.no_grad():
                                predicted = model(x).cpu().numpy()
                            selection = selection_evaluate(
                                predicted, jobs, eval_rows, manifest
                            )
                            score = selection_score(selection)
                            eligible = eligible and selection_retained(
                                selection, selection_base
                            )
                        row = dict(
                            epoch=epoch,
                            validation=v,
                            original_validation=old,
                            selection=selection,
                            score=score,
                            retention_passed=eligible,
                        )
                        history.append(row)
                        state = dict(
                            architecture=arch,
                            mode=mode,
                            method=method,
                            model={
                                k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()
                            },
                            epoch=epoch,
                            validation=v,
                            original_validation=old,
                            selection=selection,
                            labels=LABELS,
                            negative_cost=a.negative_cost,
                            seed=a.seed,
                            feature_kind=cache.get("feature_kind", "cls"),
                            feature_dim=x.shape[1],
                        )
                        if score > best_score:
                            best_score = score
                            torch.save(state, dest / "best.pt")
                        if eligible and score > guard_score:
                            guard_score = score
                            torch.save(state, dest / "retained.pt")
                for selected in ["best", "retained"]:
                    if not (dest / (selected + ".pt")).exists():
                        continue
                    state = torch.load(dest / (selected + ".pt"), weights_only=True)
                    model.load_state_dict(state["model"])
                    with torch.no_grad():
                        predicted = model(x).cpu().numpy()
                    result = dict(
                        name=name,
                        checkpoint=selected,
                        mode=mode,
                        method=method,
                        architecture=arch,
                        seconds=round(time.time() - started, 2),
                        selected_epoch=state["epoch"],
                        validation=state["validation"],
                        original_validation=state["original_validation"],
                        app=app_evaluate(predicted, jobs, eval_rows, manifest),
                    )
                    (dest / (selected + "_result.json")).write_text(
                        json.dumps(result, indent=2)
                    )
                    print(
                        json.dumps(
                            dict(
                                trial=name,
                                checkpoint=selected,
                                app={
                                    n: dict(
                                        healthy=v["healthy"]["correct"],
                                        disease=v["supported_disease"]["correct"],
                                        false_healthy=v["supported_disease"][
                                            "false_healthy"
                                        ],
                                        false_other=v["supported"]["false_other"],
                                    )
                                    for n, v in result["app"].items()
                                },
                            )
                        ),
                        flush=True,
                    )
                (dest / "history.json").write_text(json.dumps(history, indent=2))
                (dest / "result.json").write_text(
                    json.dumps(dict(completed=True, seconds=time.time() - started))
                )
    print("Factorized campaign complete", flush=True)


if __name__ == "__main__":
    main()
