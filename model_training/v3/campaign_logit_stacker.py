"""Fit small supervised logit calibrators without changing the disease branch.

Fitting uses original calibration records and a prospective hash-half of new
development validation groups. The other half and original validation groups
select the calibrator. Previously opened cohorts remain regressions; the further
external source is untouched. This split revision is saved before fitting.
"""

from pathlib import Path
import argparse, hashlib, itertools, json, math, time
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from app_eval import summarize
from train import LABELS
from training_support import load_initial_weights, atomic_save
from campaign_cnn_train import sampling
from campaign_cnn_fusion import ExpertFusion
from campaign_factorized import T, selection_score
from campaign_consensus import strict_retained
from campaign_heads import app_evaluate
from campaign_verify import regression_retained


def fitting_new_group(group):
    return (
        int(hashlib.sha256(("stack29:" + group).encode()).hexdigest()[:8], 16) % 2 == 0
    )


def stack_features(reference, cnn, adapter):
    lp = F.log_softmax(
        (reference + torch.tensor([0.0] * 8 + [0.75], device=reference.device)) / T, 1
    )
    h = lp[:, 1] - torch.cat([lp[:, :1], lp[:, 2:]], 1).logsumexp(1)
    k = lp[:, :8].logsumexp(1) - lp[:, 8]
    return torch.cat([h[:, None], k[:, None], cnn, adapter, lp], 1).clamp(-30, 30) / 10


class LogitStacker(nn.Module):
    def __init__(self, architecture="linear"):
        super().__init__()
        self.architecture = architecture
        self.residual = (
            nn.Linear(15, 2)
            if architecture == "linear"
            else nn.Sequential(nn.Linear(15, 32), nn.GELU(), nn.Linear(32, 2))
        )
        layer = self.residual if architecture == "linear" else self.residual[-1]
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    def forward(self, features):
        return features[:, :2] * 10 + self.residual(features)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--adapter", required=True)
    a = p.parse_args()
    root = Path(a.root)
    adapter = Path(a.adapter)
    out = adapter / "stackers"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = root / "dense_aligned"
    cache = torch.load(work / "features.pt", weights_only=True)
    jobs = json.loads((work / "feature_rows.json").read_text())
    ev = json.loads((work / "eval_rows.json").read_text())
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    extra_path = root / "cnn_inputs/extra_reference_logits.npy"
    if not extra_path.exists():
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
        backbone.requires_grad_(False)
        load_initial_weights(
            backbone,
            "/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
            LABELS,
            "vit_small_patch14_dinov2.lvd142m",
            224,
        )
        pixels = np.load(root / "cnn_inputs/center.npy", mmap_mode="r")
        parts = []
        with torch.no_grad():
            for start in range(len(jobs), len(rows), 128):
                parts.append(
                    backbone(
                        torch.from_numpy(
                            np.ascontiguousarray(pixels[start : start + 128])
                        )
                        .cuda()
                        .permute(0, 3, 1, 2)
                    ).cpu()
                )
        np.save(extra_path, torch.cat(parts).numpy())
        del backbone
        torch.cuda.empty_cache()
    reference = torch.cat(
        [cache["logits"], torch.from_numpy(np.load(extra_path))]
    ).cuda()
    dino = torch.from_numpy(np.load(adapter / "all_logits.npy")).cuda()
    fit = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] == "training"
            and (
                r["split"] == "calib"
                or (
                    r["split"] == "val"
                    and r["source"] in ("tlid", "pakistan")
                    and fitting_new_group(r["group"])
                )
            )
        ]
    )
    held = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] == "training"
            and r["split"] == "val"
            and r["source"] != "bangladesh_binary"
            and not (
                r["source"] in ("tlid", "pakistan") and fitting_new_group(r["group"])
            )
        ]
    )
    assert not ({rows[i]["group"] for i in fit} & {rows[i]["group"] for i in held})
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                fitting_records=len(fit),
                selection_records=len(held),
                fitting_groups=sorted({rows[i]["group"] for i in fit}),
                selection_groups=sorted({rows[i]["group"] for i in held}),
                revision="Before fitting: original calibration plus stack29 hash-half of new-source development validation; other groups select. No old test or external acceptance labels fitted.",
            ),
            indent=2,
        )
    )
    target = torch.tensor([[r["health_y"], r["known_y"]] for r in rows], device="cuda")
    weights = torch.tensor(sampling(rows, fit), device="cuda", dtype=torch.float32)
    held_weights = torch.tensor(
        sampling(rows, held), device="cuda", dtype=torch.float32
    )
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )

    def select(logits):
        results = app_evaluate(
            logits, jobs, {"selection_app": ev["selection_app"]}, manifest, True
        )["selection_app"]["rows"]
        results = [
            r
            for r in results
            if r["source"] not in ("tlid", "pakistan")
            or not fitting_new_group(r["group"])
        ]
        return dict(
            all=summarize(results),
            original=summarize(
                [r for r in results if r["source"] not in ("tlid", "pakistan")]
            ),
            new=summarize([r for r in results if r["source"] in ("tlid", "pakistan")]),
        )

    base = select(reference[: len(jobs)].cpu().numpy())
    old_ev = {k: v for k, v in ev.items() if k != "selection_app"}
    base_app = app_evaluate(
        reference[: len(jobs)].cpu().numpy(), jobs, old_ev, manifest
    )
    rank = []
    serial = 0
    started = time.time()
    for cnn_loss in ["focal", "reward"]:
        cnn_run = (
            root
            / "cnn_inputs/runs"
            / ("mobilenetv3_small_100.lamb_in1k_center_" + cnn_loss)
        )
        cnn = torch.from_numpy(np.load(cnn_run / "all_logits.npy")).cuda()
        features = stack_features(reference, cnn, dino)
        for architecture, cost in itertools.product(["linear", "mlp"], [1.0, 2.0, 4.0]):
            name = cnn_loss + "_" + architecture + "_n" + str(int(cost))
            dest = out / name
            dest.mkdir(exist_ok=True)
            torch.manual_seed(43)
            model = LogitStacker(architecture).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.01, weight_decay=0.01
            )
            best = float("inf")
            for step in range(600):
                z = model(features[fit])
                truth = target[fit]
                ce = F.binary_cross_entropy_with_logits(z, truth, reduction="none")
                penalty = torch.where(
                    truth > 0, 1.0, torch.tensor([cost, 2.0], device="cuda")
                )
                loss = (ce * penalty * weights[:, None]).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    with torch.no_grad():
                        validation = F.binary_cross_entropy_with_logits(
                            model(features[held]), target[held], reduction="none"
                        )
                        score = float((validation * held_weights[:, None]).sum())
                    if score < best:
                        best = score
                        atomic_save(
                            dict(
                                architecture=architecture,
                                model=model.state_dict(),
                                cost=cost,
                                step=step,
                                validation_loss=score,
                            ),
                            dest / "best.pt",
                        )
            state = torch.load(dest / "best.pt", weights_only=True, map_location="cpu")
            model.load_state_dict(state["model"])
            with torch.no_grad():
                raw = model(features)
                np.save(dest / "all_logits.npy", raw.cpu().numpy())
            for positive, negative, known, hmin in itertools.product(
                [0.5, 1.0], [0.0, 0.25, 0.5], [0.0, 1.0], [0.75, 0.9]
            ):
                spec = dict(
                    positive=positive,
                    negative=negative,
                    known=known,
                    calibration=[[1.0, 0.0], [1.0, 0.0]],
                    positive_scope="confirmed_known",
                    protect_healthy=False,
                    known_scope="healthy",
                    healthy_min=hmin,
                    known_min=0.95,
                    require_known_for_healthy=True,
                )
                with torch.no_grad():
                    final = (
                        ExpertFusion(**spec)
                        .cuda()(reference[: len(jobs)], raw[: len(jobs)])
                        .cpu()
                        .numpy()
                    )
                selection = select(final)
                record = dict(
                    id=serial,
                    adapter_checkpoint=str(adapter / "best.pt"),
                    cnn_checkpoint=str(cnn_run / "best.pt"),
                    stacker_checkpoint=str(dest / "best.pt"),
                    fusion=spec,
                    selection=selection,
                    validation_score=selection_score(selection),
                    retention_passed=strict_retained(selection, base),
                )
                rank.append(record)
                serial += 1
            print(
                json.dumps(
                    dict(
                        adapter=adapter.name,
                        stacker=name,
                        validation_loss=best,
                        trials=serial,
                        seconds=round(time.time() - started, 1),
                    )
                ),
                flush=True,
            )
    rank.sort(
        key=lambda r: (r["retention_passed"], r["validation_score"]), reverse=True
    )
    for r in rank[:24]:
        raw = torch.from_numpy(
            np.load(Path(r["stacker_checkpoint"]).parent / "all_logits.npy")[
                : len(jobs)
            ]
        ).cuda()
        with torch.no_grad():
            final = (
                ExpertFusion(**r["fusion"])
                .cuda()(reference[: len(jobs)], raw)
                .cpu()
                .numpy()
            )
        r["app"] = app_evaluate(final, jobs, old_ev, manifest)
        r["regression_passed"] = regression_retained(r["app"], base_app)
        (out / f"candidate_{r['id']}.json").write_text(json.dumps(r, indent=2))
        print(
            json.dumps(
                dict(
                    id=r["id"],
                    retained=r["retention_passed"],
                    regression_passed=r["regression_passed"],
                    score=r["validation_score"],
                    app={
                        k: [
                            v["healthy"]["correct"],
                            v["supported_disease"]["correct"],
                            v["all"]["false_healthy"],
                            v["supported"]["false_other"],
                        ]
                        for k, v in r["app"].items()
                    },
                )
            ),
            flush=True,
        )
    (out / "ranked.json").write_text(json.dumps(rank, indent=2) + "\n")


if __name__ == "__main__":
    main()
