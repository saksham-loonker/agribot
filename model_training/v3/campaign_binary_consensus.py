"""Calibrate agreement between independently trained CNN and DINO binary experts."""

from pathlib import Path
import argparse, itertools, json, time
import numpy as np
import torch
from campaign_cnn_fusion import ExpertFusion, calibrate
from campaign_cnn_train import sampling
from campaign_factorized import selection_evaluate, selection_score
from campaign_consensus import strict_retained
from campaign_heads import app_evaluate
from campaign_verify import regression_retained


def binary_agreement(left, right):
    a, b = left[:, 0], right[:, 0]
    healthy = torch.where(
        (a > 0) & (b > 0),
        torch.minimum(a, b),
        torch.where((a < 0) & (b < 0), torch.maximum(a, b), torch.zeros_like(a)),
    )
    return torch.stack([healthy, torch.minimum(left[:, 1], right[:, 1])], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--allow-healthy-downgrade", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    adapter = Path(a.adapter)
    out = adapter / (
        "binary_consensus_full" if a.allow_healthy_downgrade else "binary_consensus"
    )
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    work = root / "dense_aligned"
    cache = torch.load(work / "features.pt", weights_only=True)
    reference = cache["logits"].cuda()
    jobs = json.loads((work / "feature_rows.json").read_text())
    ev = json.loads((work / "eval_rows.json").read_text())
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    raw = np.load(adapter / "all_logits.npy")
    dino_cal = []
    for column, key in enumerate(["health_y", "known_y"]):
        indices = np.array(
            [i for i, r in enumerate(rows) if r["split"] == "val" and r[key] >= 0]
        )
        truth = np.array([rows[i][key] for i in indices])
        dino_cal.append(
            calibrate(raw[indices, column], truth, sampling(rows, indices, target=key))
        )
    (out / "adapter_calibration.json").write_text(json.dumps(dino_cal, indent=2))
    dino_raw = torch.from_numpy(raw[: len(jobs)]).cuda()
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )
    baseline = selection_evaluate(reference.cpu().numpy(), jobs, ev, manifest)
    old_ev = {k: v for k, v in ev.items() if k != "selection_app"}
    base_app = app_evaluate(reference.cpu().numpy(), jobs, old_ev, manifest)
    rank = []
    serial = 0
    started = time.time()
    with torch.inference_mode(), (out / "search.jsonl").open("w") as stream:
        for loss in ["bce", "focal", "reward"]:
            run = (
                root
                / "cnn_inputs/runs"
                / ("mobilenetv3_small_100.lamb_in1k_center_" + loss)
            )
            cnn_raw = torch.from_numpy(
                np.load(run / "all_logits.npy")[: len(jobs)]
            ).cuda()
            cnn_cal = json.loads((run / "fusion/calibration.json").read_text())
            for dcal, ccal in itertools.product(
                [[[1.0, 0.0], [1.0, 0.0]], dino_cal],
                [[[1.0, 0.0], [1.0, 0.0]], cnn_cal],
            ):
                d = torch.tensor(dcal, device="cuda")
                c = torch.tensor(ccal, device="cuda")
                expert = binary_agreement(
                    dino_raw * d[:, 0] + d[:, 1], cnn_raw * c[:, 0] + c[:, 1]
                )
                for positive, negative, known, hmin in itertools.product(
                    [0.5, 1.0], [0.0, 0.25, 0.5, 1.0], [0.0, 1.0], [0.75, 0.9, 0.95]
                ):
                    spec = dict(
                        positive=positive,
                        negative=negative,
                        known=known,
                        calibration=[[1.0, 0.0], [1.0, 0.0]],
                        positive_scope="confirmed_known",
                        protect_healthy=not a.allow_healthy_downgrade,
                        known_scope="healthy",
                        healthy_min=hmin,
                        known_min=0.95,
                        require_known_for_healthy=True,
                    )
                    logits = (
                        ExpertFusion(**spec).cuda()(reference, expert).cpu().numpy()
                    )
                    selection = selection_evaluate(logits, jobs, ev, manifest)
                    record = dict(
                        id=serial,
                        adapter_checkpoint=str(adapter / "best.pt"),
                        adapter_calibration=dcal,
                        cnn_checkpoint=str(run / "best.pt"),
                        cnn_calibration=ccal,
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
                            adapter=adapter.name,
                            cnn=loss,
                            trials=serial,
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
        rank.sort(
            key=lambda r: (r["retention_passed"], r["validation_score"]), reverse=True
        )
        for r in rank[:20]:
            cnn = torch.from_numpy(
                np.load(Path(r["cnn_checkpoint"]).parent / "all_logits.npy")[
                    : len(jobs)
                ]
            ).cuda()
            d = torch.tensor(r["adapter_calibration"], device="cuda")
            c = torch.tensor(r["cnn_calibration"], device="cuda")
            expert = binary_agreement(
                dino_raw * d[:, 0] + d[:, 1], cnn * c[:, 0] + c[:, 1]
            )
            logits = ExpertFusion(**r["fusion"]).cuda()(reference, expert).cpu().numpy()
            r["app"] = app_evaluate(logits, jobs, old_ev, manifest)
            r["regression_passed"] = regression_retained(r["app"], base_app)
            (out / f"candidate_{r['id']}.json").write_text(json.dumps(r, indent=2))
            print(
                json.dumps(
                    dict(
                        id=r["id"],
                        score=r["validation_score"],
                        retained=r["retention_passed"],
                        regression_passed=r["regression_passed"],
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
