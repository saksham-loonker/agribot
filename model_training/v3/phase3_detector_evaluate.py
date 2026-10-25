"""Compare detector checkpoints against separate old and grouped field validation."""

from pathlib import Path
import argparse, hashlib, json, time
import torch
from libreyolo import LibreYOLO9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--trained-sizes", default="320,640")
    a = p.parse_args()
    root = Path(a.root)
    assert not json.loads((root / "data_complete.json").read_text()).get(
        "quarantined", False
    ), "Quarantined annotation semantics; detector comparison is invalid"
    out = root / "candidate_validation"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(6)
    baseline = json.loads((root / "original_320_validation.json").read_text())
    ranking = []
    protocol = dict(
        old_map50_95_loss_max=0.01,
        field_map50_min=0.15,
        field_map50_95_gain_min=0.02,
        baseline="original shipped detector at 320",
        source_history="Both datasets are validation/development. Original test and full app regressions remain required; no independent field acceptance.",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for trained_size in map(int, a.trained_sizes.split(",")):
        assert (
            root / f"y9t_field_{trained_size}_complete.json"
        ).exists(), "Do not evaluate a checkpoint while training is mutating it"
        for name in ["best", "last"]:
            checkpoint = (
                root / "runs" / f"y9t_field_{trained_size}" / "weights" / (name + ".pt")
            )
            if not checkpoint.exists():
                continue
            for size in ([320] if trained_size == 320 else [320, 640]):
                key = f"train{trained_size}_{name}_infer{size}"
                file = out / (key + ".json")
                if file.exists():
                    record = json.loads(file.read_text())
                    assert (
                        record["sha256"]
                        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    ), "Validation checkpoint changed; use a new output directory"
                else:
                    model = LibreYOLO9(str(checkpoint), size="t", nb_classes=1)
                    metrics = {}
                    start = time.time()
                    for domain in ["old", "slif"]:
                        metrics[domain] = model.val(
                            data=str(root / (domain + "_val.yaml")),
                            batch=32,
                            imgsz=size,
                            workers=4,
                            plots=False,
                            project=str(out / "runs"),
                            name=key + "_" + domain,
                            exist_ok=True,
                        )
                    old = float(metrics["old"]["metrics/mAP50-95"])
                    field = float(metrics["slif"]["metrics/mAP50-95"])
                    field50 = float(metrics["slif"]["metrics/mAP50"])
                    passed = (
                        old >= float(baseline["old"]["metrics/mAP50-95"]) - 0.01
                        and field >= float(baseline["slif"]["metrics/mAP50-95"]) + 0.02
                        and field50 >= 0.15
                    )
                    record = dict(
                        id=key,
                        checkpoint=str(checkpoint),
                        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        input_size=size,
                        trained_size=trained_size,
                        metrics=metrics,
                        passed=passed,
                        score=field,
                        seconds=time.time() - start,
                    )
                    file.write_text(json.dumps(record, indent=2, default=str))
                    del model
                    torch.cuda.empty_cache()
                ranking.append(record)
                print(
                    json.dumps(
                        dict(
                            id=key,
                            passed=record["passed"],
                            old=record["metrics"]["old"]["metrics/mAP50-95"],
                            field=record["metrics"]["slif"]["metrics/mAP50-95"],
                        )
                    ),
                    flush=True,
                )
    ranking.sort(key=lambda r: (r["passed"], r["score"]), reverse=True)
    (out / "ranked.json").write_text(json.dumps(ranking, indent=2, default=str))
    (out / "candidate.json").write_text(json.dumps(ranking[0], indent=2, default=str))


if __name__ == "__main__":
    main()
