"""Fine-tune the original generic leaf detector; keep old and field metrics separate."""

from pathlib import Path
import argparse, hashlib, json, time
import torch
from libreyolo import LibreYOLO9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--sizes", default="320,640")
    a = p.parse_args()
    root = Path(a.root)
    checkpoint = Path(
        "/home/genesis/sakshyam-agribot/v2/det_runs/y9t_416_comb/weights/best.pt"
    )
    assert (root / "data_complete.json").exists()
    data_identity = json.loads((root / "data_complete.json").read_text())
    assert (
        not data_identity.get("quarantined", False)
        and data_identity.get("annotation_role") == "whole_leaf_boundary"
    ), "Rejected dataset: region annotations are not whole-leaf boundaries. Do not resume this experiment."
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(
            (root / "source_rows.json").read_bytes()
        ).hexdigest(),
        epochs=a.epochs,
        sizes=a.sizes,
        seed=83,
        lr0=0.0001,
        training_repeat=4,
        source_history="SLIF opened source now development; no independent field acceptance claim",
        gate="Select using old and field validation separately. Original test and app regression replay are required before any promotion.",
    )
    (root / "training_protocol.json").write_text(json.dumps(config, indent=2))
    for size in map(int, a.sizes.split(",")):
        baseline = root / f"original_{size}_validation.json"
        if not baseline.exists():
            m = LibreYOLO9(str(checkpoint), size="t", nb_classes=1)
            result = {}
            for domain in ["old", "slif"]:
                result[domain] = m.val(
                    data=str(root / (domain + "_val.yaml")),
                    imgsz=size,
                    batch=32,
                    workers=4,
                    plots=False,
                    project=str(root / "validation"),
                    name=f"original_{size}_{domain}",
                    exist_ok=True,
                )
            baseline.write_text(json.dumps(result, indent=2, default=str))
            del m
            torch.cuda.empty_cache()
        name = f"y9t_field_{size}"
        done = root / (name + "_complete.json")
        if done.exists():
            continue
        last = root / "runs" / name / "weights/last.pt"
        m = LibreYOLO9(
            str(last if last.exists() else checkpoint), size="t", nb_classes=1
        )
        started = time.time()
        kwargs = dict(
            data=str(root / "dataset/data.yaml"),
            epochs=a.epochs,
            imgsz=size,
            batch=64 if size == 320 else 32,
            optimizer="AdamW",
            lr0=0.0001,
            workers=6,
            project=str(root / "runs"),
            name=name,
            exist_ok=True,
            amp_dtype="bfloat16",
            patience=8,
            eval_interval=2,
            save_period=2,
            seed=83,
            no_aug_epochs=4,
            mosaic_prob=0.2,
            max_labels=300,
        )
        if last.exists():
            kwargs["resume"] = True
        else:
            kwargs["pretrained"] = str(checkpoint)
        result = m.train(**kwargs)
        done.write_text(
            json.dumps(
                dict(result=result, seconds=time.time() - started, protocol=config),
                indent=2,
                default=str,
            )
        )
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
