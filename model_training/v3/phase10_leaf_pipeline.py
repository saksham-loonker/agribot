"""Resume leaf-supervised controls, app replays and evidence; never deploy."""

from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    preparation = json.loads((root / "complete.json").read_text())
    assert preparation["masked_weak_training_rows"] > 0
    assert preparation["cold_ad_predictions_opened"] is False
    scripts = Path(__file__).resolve().parent

    def run(script, args, log_name):
        with (root / log_name).open("a", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(scripts / script), *args],
                cwd=scripts,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for name, augmented, selection in [
        ("leaf_sft_d12_focal", False, "leaf_sft_selection"),
        ("leaf_sft_d12_augmented", True, "leaf_sft_augmented_selection"),
    ]:
        args = [
            "--root",
            str(root),
            "--task-scope",
            "annotated_leaf",
            "--run-name",
            name,
            "--depth",
            "12",
            "--loss",
            "focal",
            "--epochs",
            "16",
            "--batch",
            "64",
            "--backbone-lr",
            "0.00001",
            "--head-lr",
            "0.0001",
            "--weight-decay",
            "0.05",
            "--dropout",
            "0.1",
        ]
        if augmented:
            args.append("--photo-augmentation")
        run("phase9_scene_sft.py", args, name + ".log")
        if not (root / selection / "candidate.json").exists():
            run(
                "phase10_leaf_select.py",
                [
                    "--root",
                    str(root),
                    "--tail-directory",
                    str(root / name),
                    "--out-name",
                    selection,
                ],
                selection + ".log",
            )
        run(
            "phase10_report.py",
            ["--root", str(root), "--out", str(root.parent / "phase10_evidence.json")],
            "phase10_report.log",
        )
        print(
            json.dumps(dict(completed=name, replay=selection, fresh_ad_used=False)),
            flush=True,
        )
    print(
        "Leaf controls complete; promotion still requires every gate and a separately frozen acceptance protocol",
        flush=True,
    )


if __name__ == "__main__":
    main()
