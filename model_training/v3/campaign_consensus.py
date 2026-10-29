"""Require independent DINO-head and CNN agreement for specialist corrections."""

from pathlib import Path
import argparse, hashlib, itertools, json, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from campaign_cnn_fusion import ExpertFusion
from campaign_heads import FeatureHead, app_evaluate
from campaign_factorized import (
    T,
    selection_evaluate,
    selection_score,
    selection_retained,
)
from campaign_verify import regression_retained


def agreement(reference_head, cnn):
    lp = F.log_softmax(
        (
            reference_head
            + torch.tensor([0.0] * 8 + [0.75], device=reference_head.device)
        )
        / T,
        1,
    )
    healthy = lp[:, 1] - torch.cat([lp[:, :1], lp[:, 2:]], 1).logsumexp(1)
    known = lp[:, :8].logsumexp(1) - lp[:, 8]
    positive = (healthy > 0) & (cnn[:, 0] > 0)
    negative = (healthy < 0) & (cnn[:, 0] < 0)
    h = torch.where(
        positive,
        torch.minimum(healthy, cnn[:, 0]),
        torch.where(
            negative, torch.maximum(healthy, cnn[:, 0]), torch.zeros_like(healthy)
        ),
    )
    # Both experts must regard the input as belonging to a supported class.
    k = torch.minimum(known, cnn[:, 1])
    return torch.stack([h, k], 1)


def strict_retained(candidate, baseline):
    return selection_retained(candidate, baseline) and all(
        candidate[s]["supported_disease"]["correct"]
        >= baseline[s]["supported_disease"]["correct"]
        and candidate[s]["supported_disease"]["false_healthy"]
        <= baseline[s]["supported_disease"]["false_healthy"]
        and candidate[s]["all"]["false_healthy"] <= baseline[s]["all"]["false_healthy"]
        and candidate[s]["supported"]["false_other"]
        <= baseline[s]["supported"]["false_other"]
        and all(
            (candidate[s][k]["false_known"] or 0)
            <= (baseline[s][k]["false_known"] or 0)
            for k in ["unsupported_tomato", "non_tomato"]
        )
        for s in ["original", "new"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--broad", action="store_true")
    a = parser.parse_args()
    root = Path(a.root)
    out = root / ("consensus_broad" if a.broad else "consensus")
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    work = root / "dense_aligned"
    cache = torch.load(work / "features.pt", weights_only=True)
    x = cache["features"][:, :384].cuda()
    reference = cache["logits"].cuda()
    jobs = json.loads((work / "feature_rows.json").read_text())
    ev = json.loads((work / "eval_rows.json").read_text())
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )
    baseline = selection_evaluate(reference.cpu().numpy(), jobs, ev, manifest)
    old_ev = {k: v for k, v in ev.items() if k != "selection_app"}
    base_app = app_evaluate(reference.cpu().numpy(), jobs, old_ev, manifest)
    head_paths = [
        root / phase / "heads" / ("residual_mlp_" + loss + "_h2_s17") / "best.pt"
        for phase, loss in [
            ("", "healthy_ce"),
            ("", "retention"),
            ("aligned", "healthy_ce"),
            ("aligned", "retention"),
            ("aligned", "weighted"),
            ("aligned", "logit"),
            ("aligned", "margin"),
            ("aligned", "ppo"),
        ]
    ]
    if a.broad:
        head_paths = [head_paths[i] for i in [0, 4, 5, 6]]
    rank = []
    serial = 0
    started = time.time()
    with torch.inference_mode(), (out / "search.jsonl").open("w") as stream:
        for head_path in head_paths:
            state = torch.load(head_path, weights_only=True, map_location="cpu")
            head = (
                FeatureHead(cache["original_head"], state["architecture"]).cuda().eval()
            )
            head.load_state_dict(state["model"])
            hz = head(x)
            for loss in (
                ["focal", "reward"] if a.broad else ["bce", "focal", "reward"]
            ):
                run = (
                    root
                    / "cnn_inputs/runs"
                    / ("mobilenetv3_small_100.lamb_in1k_center_" + loss)
                )
                raw = torch.from_numpy(
                    np.load(run / "all_logits.npy")[: len(jobs)]
                ).cuda()
                for kind, cal in [
                    ("identity", [[1.0, 0.0], [1.0, 0.0]]),
                    (
                        "platt",
                        json.loads((run / "fusion/calibration.json").read_text()),
                    ),
                ]:
                    parameters = torch.tensor(cal, device="cuda")
                    cnn = raw * parameters[:, 0] + parameters[:, 1]
                    expert = agreement(hz, cnn)
                    settings = (
                        itertools.product(
                            [0.5, 1.0],
                            [0.25, 0.5, 1.0],
                            [0.0, 1.0],
                            [0.6, 0.75, 0.9],
                            [0.01, 0.05, 0.1],
                        )
                        if a.broad
                        else itertools.product(
                            [0.5, 1.0],
                            [0.0, 0.25, 0.5, 1.0],
                            [0.0, 0.5, 1.0],
                            [0.9, 0.95],
                            [0.01],
                        )
                    )
                    for positive, negative, known, hmin, unhealthy_max in settings:
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
                            unhealthy_max=unhealthy_max,
                        )
                        logits = (
                            ExpertFusion(**spec).cuda()(reference, expert).cpu().numpy()
                        )
                        selection = selection_evaluate(logits, jobs, ev, manifest)
                        record = dict(
                            id=serial,
                            head_checkpoint=str(head_path),
                            cnn_checkpoint=str(run / "best.pt"),
                            cnn_calibration=cal,
                            cnn_calibration_kind=kind,
                            fusion=spec,
                            selection=selection,
                            validation_score=selection_score(selection),
                            retention_passed=strict_retained(selection, baseline),
                        )
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        rank.append(record)
                        serial += 1
                    print(
                        json.dumps(
                            dict(
                                head=str(head_path.parent),
                                cnn=loss,
                                calibration=kind,
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
            state = torch.load(
                r["head_checkpoint"], weights_only=True, map_location="cpu"
            )
            head = (
                FeatureHead(cache["original_head"], state["architecture"]).cuda().eval()
            )
            head.load_state_dict(state["model"])
            run = Path(r["cnn_checkpoint"]).parent
            raw = torch.from_numpy(np.load(run / "all_logits.npy")[: len(jobs)]).cuda()
            cal = torch.tensor(r["cnn_calibration"], device="cuda")
            expert = agreement(head(x), raw * cal[:, 0] + cal[:, 1])
            final = ExpertFusion(**r["fusion"]).cuda()(reference, expert).cpu().numpy()
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
                        head=r["head_checkpoint"],
                        cnn=r["cnn_checkpoint"],
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
