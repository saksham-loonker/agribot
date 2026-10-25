"""GPU boosted-tree experts and validation-only residual-logit calibration."""

from pathlib import Path
import argparse, collections, hashlib, json, time
import numpy as np
import torch
import xgboost as xgb
from torch.nn import functional as F
from campaign_factorized import (
    FactorizedHead,
    selection_evaluate,
    selection_score,
    selection_retained,
    T,
)
from campaign_heads import app_evaluate
from train import LABELS


def sample_weights(indices, target, jobs):
    keys = [
        (int(jobs[i]["source"] in ("tlid", "pakistan")), int(target[k]))
        for k, i in enumerate(indices)
    ]
    group_keys = [(key, jobs[i]["group"]) for key, i in zip(keys, indices)]
    sizes = collections.Counter(group_keys)
    groups = collections.Counter(key for key, group in set(group_keys))
    result = np.array(
        [1 / (groups[key] * sizes[gkey]) for key, gkey in zip(keys, group_keys)],
        np.float32,
    )
    return result / result.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--rounds", type=int, default=800)
    a = p.parse_args()
    work = Path(a.work)
    out = work / "classical"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(work / "features.pt", weights_only=True)
    jobs = json.loads((work / "feature_rows.json").read_text())
    eval_rows = json.loads((work / "eval_rows.json").read_text())
    x = cache["features"].numpy()
    z = cache["logits"].cuda()
    lp = F.log_softmax((z + torch.tensor([0.0] * 8 + [0.75], device="cuda")) / T, 1)
    labels = np.array(
        [LABELS.index(r["label"]) if r["label"] in LABELS else -1 for r in jobs]
    )
    train = np.array(
        [
            i
            for i, r in enumerate(jobs)
            if r["set"] == "training" and r["split"] == "train"
        ]
    )
    val = np.array(
        [
            i
            for i, r in enumerate(jobs)
            if r["set"] == "training" and r["split"] == "val"
        ]
    )
    reference = {
        "pair": (lp[:, 1] - lp[:, 3]).cpu().numpy(),
        "known": (lp[:, :8].logsumexp(1) - lp[:, 8]).cpu().numpy(),
    }
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )
    baseline = selection_evaluate(z.cpu().numpy(), jobs, eval_rows, manifest)
    (out / "selection_baseline.json").write_text(json.dumps(baseline, indent=2))
    banks = {}
    for task in ["pair", "known"]:
        mask = lambda ids: (
            (
                (labels[ids] == 1)
                | (labels[ids] == 3)
                | ((labels[ids] == 8) & (lp[ids, 1].exp().cpu().numpy() > 0.15))
            )
            if task == "pair"
            else np.ones(len(ids), bool)
        )
        ti, vi = train[mask(train)], val[mask(val)]
        target = (
            (labels == 1).astype(np.float32)
            if task == "pair"
            else (labels != 8).astype(np.float32)
        )
        banks[task] = []
        for depth, negative_cost in [(3, 1.0), (5, 1.0), (7, 1.0), (5, 2.0)]:
            name = f"{task}_depth{depth}_n{negative_cost:g}"
            path = out / (name + ".ubj")
            delta_path = out / (name + "_delta.npy")
            meta_path = out / (name + ".json")
            if delta_path.exists():
                banks[task].append(
                    (json.loads(meta_path.read_text()), np.load(delta_path))
                )
                continue
            weights = sample_weights(ti.tolist(), target[ti], jobs) * np.where(
                target[ti] > 0, 1.0, negative_cost
            )
            dtrain = xgb.DMatrix(
                x[ti],
                label=target[ti],
                weight=weights,
                base_margin=reference[task][ti],
                nthread=4,
            )
            dval = xgb.DMatrix(
                x[vi],
                label=target[vi],
                weight=sample_weights(vi.tolist(), target[vi], jobs),
                base_margin=reference[task][vi],
                nthread=4,
            )
            params = dict(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                max_depth=depth,
                eta=0.04,
                min_child_weight=5,
                reg_lambda=10,
                subsample=0.9,
                colsample_bytree=0.85,
                seed=19,
                nthread=4,
            )
            history = {}
            start = time.time()
            booster = xgb.train(
                params,
                dtrain,
                a.rounds,
                [(dval, "validation")],
                early_stopping_rounds=60,
                evals_result=history,
                verbose_eval=False,
            )
            config = json.loads(booster.save_config())
            device = config["learner"]["generic_param"]["device"]
            assert device.startswith("cuda"), device
            count = booster.best_iteration + 1
            booster.save_model(path)
            delta = (
                booster.predict(
                    xgb.DMatrix(x, base_margin=reference[task], nthread=4),
                    output_margin=True,
                    iteration_range=(0, count),
                )
                - reference[task]
            )
            np.save(delta_path, delta)
            meta = dict(
                name=name,
                task=task,
                depth=depth,
                negative_cost=negative_cost,
                iterations=count,
                seconds=time.time() - start,
                validation_logloss=booster.best_score,
                device=device,
                parameters=params,
                model_path=str(path),
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                xgboost=xgb.__version__,
            )
            meta_path.write_text(json.dumps(meta, indent=2))
            (out / (name + "_history.json")).write_text(json.dumps(history))
            banks[task].append((meta, delta))
            print(json.dumps(meta), flush=True)
    # Every calibration candidate is ranked using grouped validation photos only.
    head = FactorizedHead(cache["original_head"], "both", "linear", x.shape[1]).cuda()
    null = dict(name="none")
    zero = np.zeros(len(jobs), np.float32)
    pairs = [(null, zero)] + banks["pair"]
    knowns = [(null, zero)] + banks["known"]
    ranked = []
    serial = 0
    with torch.inference_mode(), (out / "calibration_search.jsonl").open("w") as stream:
        for pair, pair_delta in pairs:
            for known, known_delta in knowns:
                for alpha in (
                    [0.0] if pair["name"] == "none" else [0.25, 0.5, 0.75, 1.0]
                ):
                    for bias in (
                        [0.0] if pair["name"] == "none" else [0.0, 0.25, 0.5, 1.0]
                    ):
                        for known_alpha in (
                            [0.0] if known["name"] == "none" else [0.5, 1.0]
                        ):
                            dp = torch.from_numpy(
                                (alpha * pair_delta + bias).astype(np.float32)
                            ).cuda()
                            dk = torch.from_numpy(
                                (known_alpha * known_delta).astype(np.float32)
                            ).cuda()
                            predicted = (
                                head.combine(
                                    z,
                                    lp,
                                    lp[:, 1] - lp[:, 3] + dp,
                                    lp[:, :8].logsumexp(1) - lp[:, 8] + dk,
                                )
                                .cpu()
                                .numpy()
                            )
                            selection = selection_evaluate(
                                predicted, jobs, eval_rows, manifest
                            )
                            score = selection_score(selection)
                            keep = selection_retained(selection, baseline)
                            record = dict(
                                id=serial,
                                pair=pair["name"],
                                known=known["name"],
                                alpha=alpha,
                                bias=bias,
                                known_alpha=known_alpha,
                                score=score,
                                retention_passed=keep,
                                selection=selection,
                            )
                            stream.write(json.dumps(record) + "\n")
                            stream.flush()
                            ranked.append(record)
                            if serial % 20 == 0:
                                print(
                                    json.dumps(
                                        dict(
                                            calibration=serial,
                                            score=score,
                                            retained=keep,
                                        )
                                    ),
                                    flush=True,
                                )
                            serial += 1
    ranked.sort(key=lambda r: (r["retention_passed"], r["score"]), reverse=True)
    for candidate in ranked[:12]:
        pair_delta = (
            zero
            if candidate["pair"] == "none"
            else next(d for m, d in banks["pair"] if m["name"] == candidate["pair"])
        )
        known_delta = (
            zero
            if candidate["known"] == "none"
            else next(d for m, d in banks["known"] if m["name"] == candidate["known"])
        )
        dp = torch.from_numpy(
            (candidate["alpha"] * pair_delta + candidate["bias"]).astype(np.float32)
        ).cuda()
        dk = torch.from_numpy(
            (candidate["known_alpha"] * known_delta).astype(np.float32)
        ).cuda()
        with torch.no_grad():
            predicted = (
                head.combine(
                    z,
                    lp,
                    lp[:, 1] - lp[:, 3] + dp,
                    lp[:, :8].logsumexp(1) - lp[:, 8] + dk,
                )
                .cpu()
                .numpy()
            )
        candidate["app"] = app_evaluate(predicted, jobs, eval_rows, manifest)
        (out / f"candidate_{candidate['id']}.json").write_text(
            json.dumps(candidate, indent=2)
        )
        print(
            json.dumps(
                dict(
                    candidate=candidate["id"],
                    pair=candidate["pair"],
                    known=candidate["known"],
                    score=candidate["score"],
                    app={
                        n: dict(
                            healthy=v["healthy"]["correct"],
                            disease=v["supported_disease"]["correct"],
                            false_healthy=v["supported_disease"]["false_healthy"],
                            false_other=v["supported"]["false_other"],
                        )
                        for n, v in candidate["app"].items()
                    },
                )
            ),
            flush=True,
        )
    (out / "ranked_calibration.json").write_text(json.dumps(ranked, indent=2))
    print("Classical experts and validation calibration complete", flush=True)


if __name__ == "__main__":
    main()
