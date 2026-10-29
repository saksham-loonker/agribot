"""Fit a Healthy verifier on frozen original and adapted image representations.

Uses training labels only for optimization; selection uses grouped validation.
Previously opened photo sets are regression gates, never independent evidence.
"""

from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from campaign_adapter import prefix
from campaign_cnn_export import restore
from campaign_cnn_train import sampling
from campaign_pixels import open_pixels
from campaign_heads import app_evaluate
from campaign_verify import regression_retained
from phase2_multiclass import MulticlassExpert
from app_eval import summarize


class HealthyVerifier(nn.Module):
    def __init__(self, dimension=788, hidden=0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("std", torch.ones(dimension))
        self.layers = (
            nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, 1))
            if hidden
            else nn.Linear(dimension, 1)
        )

    def forward(self, x):
        return self.layers((x - self.mean) / self.std).flatten()


def healthy_override(reference, student, verifier, threshold, student_min=0.7):
    bias = torch.tensor([0.0] * 8 + [0.75], device=reference.device)
    temperature = 0.8814700841903687
    p = ((reference + bias) / temperature).softmax(1)
    proposal = ((student + bias) / temperature).softmax(1)
    eligible = (
        (verifier.sigmoid() >= threshold)
        & (proposal.argmax(1) == 1)
        & (proposal[:, 1] >= student_min)
    )
    # The verifier must corroborate an actual nine-class Healthy proposal.
    eligible = eligible & (~((p.argmax(1) == 1) & (p[:, 1] >= 0.6)))
    logits = reference + temperature * (
        proposal.clamp_min(1e-12).log() - p.clamp_min(1e-12).log()
    )
    return torch.where(eligible[:, None], logits, reference)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    q = Path(a.root)
    out = q / "healthy_verifier"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads((q / "cnn_inputs/rows.json").read_text())
    ev = json.loads((q / "eval_rows.json").read_text())
    ref = np.load(q / "reference_logits.npy")
    student_path = q / "multiclass/lora_d4/best.pt"
    student = np.load(student_path.parent / "best_logits.npy")
    cnn = np.load(
        q
        / "cnn_inputs/runs/mobilenetv3_small_100.lamb_in1k_center_focal_n2/all_logits.npy"
    )
    manifest = json.loads(
        (q.parent / "original_reference/model_manifest.json").read_text()
    )
    identity = hashlib.sha256((q / "cnn_inputs/rows.json").read_bytes()).hexdigest()
    cache = out / "features.pt"
    if not cache.exists():
        safe = restore(q.parent / "export_cnn_safe/pytorch_bundle.pt").cuda().eval()
        backbone = safe.backbone
        state = torch.load(student_path, weights_only=True, map_location="cpu")
        assert state["rows_sha256"] == identity
        expert = (
            MulticlassExpert(backbone, state["mode"], state["rank"], state["depth"])
            .cuda()
            .eval()
        )
        expert.load_state_dict(state["model"])
        pixels = open_pixels(q / "cnn_inputs")
        features = []
        safe_logits = []
        start = time.time()
        with torch.inference_mode():
            for first in range(0, len(rows), 128):
                x = (
                    torch.from_numpy(np.ascontiguousarray(pixels[first : first + 128]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                tokens = prefix(backbone, x, state["depth"])
                original = tokens
                for block in backbone.blocks[-state["depth"] :]:
                    original = block(original)
                original = backbone.norm(original)[:, 0]
                adapted = expert.norm(expert.tail(tokens))[:, 0]
                cz = safe.cnn(
                    (x * safe.canonical_std + safe.canonical_mean - safe.cnn_mean)
                    / safe.cnn_std
                )
                rz = backbone.head(original)
                safe_logits.append(safe.fusion(rz, cz).cpu())
                features.append(
                    torch.cat(
                        [
                            original,
                            adapted,
                            torch.from_numpy(ref[first : first + 128]).cuda(),
                            torch.from_numpy(student[first : first + 128]).cuda(),
                            torch.from_numpy(cnn[first : first + 128]).cuda(),
                        ],
                        1,
                    ).cpu()
                )
                if first % 5120 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage="features",
                                rows=first,
                                total=len(rows),
                                seconds=time.time() - start,
                            )
                        ),
                        flush=True,
                    )
        torch.save(
            dict(
                rows_sha256=identity,
                student_sha256=hashlib.sha256(student_path.read_bytes()).hexdigest(),
                features=torch.cat(features),
                safe_logits=torch.cat(safe_logits),
            ),
            cache,
        )
        del safe, expert, backbone
        torch.cuda.empty_cache()
    data = torch.load(cache, weights_only=True)
    assert data["rows_sha256"] == identity
    assert (
        data["student_sha256"] == hashlib.sha256(student_path.read_bytes()).hexdigest()
    ), "Frozen-feature student identity changed"
    x = data["features"].cuda()
    reference = data["safe_logits"].cuda()
    student = torch.from_numpy(student).cuda()
    train = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "train" and r["health_y"] >= 0]
    )
    val = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "val" and r["health_y"] >= 0]
    )
    truth = torch.tensor([r["health_y"] for r in rows], device="cuda")
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    weights = torch.tensor(sampling(rows, train), device="cuda", dtype=torch.float32)
    rank = []

    def evaluate(z):
        results = app_evaluate(z, rows, ev, manifest, True)
        sel = results.pop("selection_app")["rows"]
        novel = results.pop("phase2_selection")["metrics"]
        old = {k: v["metrics"] for k, v in results.items()}
        return dict(
            original=summarize(
                [r for r in sel if r["source"] not in ("tlid", "pakistan")]
            ),
            expanded=summarize([r for r in sel if r["source"] in ("tlid", "pakistan")]),
            novel=novel,
            **old,
        )

    baseline = evaluate(reference.cpu().numpy())
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    for hidden, cost in [(h, c) for h in [0, 128] for c in [1.0, 4.0, 12.0]]:
        torch.manual_seed(71)
        model = HealthyVerifier(x.shape[1], hidden).cuda()
        model.mean.copy_(x[train].mean(0))
        model.std.copy_(x[train].std(0).clamp_min(0.1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        best = -1
        best_state = None
        history = []
        for epoch in range(60):
            model.train()
            order = torch.multinomial(weights, len(train), replacement=True)
            total = 0.0
            for batch in order.split(512):
                ids = torch.as_tensor(train, device="cuda")[batch]
                z = model(x[ids])
                target = truth[ids]
                loss = (
                    F.binary_cross_entropy_with_logits(z, target, reduction="none")
                    * torch.where(target > 0, 1.0, cost)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            model.eval()
            with torch.no_grad():
                vz = model(x[val]).cpu().numpy()
            domains = [
                [j for j, i in enumerate(val) if rows[i]["source"] in s]
                for s in [
                    ("plantdoc", "tomato_village"),
                    ("tlid", "pakistan"),
                    ("bangladesh_classification",),
                    ("bangladesh_binary",),
                ]
            ]
            aucs = []
            for ids in domains:
                if ids and len(set(truth[val[ids]].cpu().tolist())) == 2:
                    aucs.append(roc_auc_score(truth[val[ids]].cpu().numpy(), vz[ids]))
            score = float(np.mean(aucs))
            history.append(dict(epoch=epoch, loss=total, domain_auc=aucs, score=score))
            if score > best:
                best = score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        model.load_state_dict(best_state)
        name = f"healthy_h{hidden}_cost{cost:g}"
        checkpoint = out / (name + ".pt")
        torch.save(
            dict(
                architecture="HealthyVerifier",
                hidden=hidden,
                dimension=x.shape[1],
                model=best_state,
                rows_sha256=identity,
                student_checkpoint=str(student_path),
                cost=cost,
                history=history,
            ),
            checkpoint,
        )
        with torch.inference_mode():
            vz = model(x)
            np.save(out / (name + "_logits.npy"), vz.cpu().numpy())
        for threshold in [0.9, 0.95, 0.975, 0.99, 0.995, 0.999]:
            for student_min in [0.7, 0.9, 0.97]:
                with torch.inference_mode():
                    final = (
                        healthy_override(reference, student, vz, threshold, student_min)
                        .cpu()
                        .numpy()
                    )
                metrics = evaluate(final)
                keep = regression_retained(metrics, baseline)
                score = float(
                    np.mean(
                        [v["healthy"]["correct_rate"] or 0 for v in metrics.values()]
                    )
                )
                r = dict(
                    checkpoint=str(checkpoint),
                    threshold=threshold,
                    student_min=student_min,
                    metrics=metrics,
                    retained=keep,
                    score=score,
                )
                rank.append(r)
        print(
            json.dumps(
                dict(
                    model=name,
                    auc=best,
                    retained=sum(r["retained"] for r in rank),
                    best=sorted(
                        rank, key=lambda r: (r["retained"], r["score"]), reverse=True
                    )[0],
                )
            ),
            flush=True,
        )
    rank.sort(key=lambda r: (r["retained"], r["score"]), reverse=True)
    (out / "ranked.json").write_text(json.dumps(rank, indent=2))
    (out / "candidate.json").write_text(json.dumps(rank[0], indent=2))


if __name__ == "__main__":
    main()
