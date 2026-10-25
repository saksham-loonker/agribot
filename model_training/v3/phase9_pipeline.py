"""Calibrate and replay a completed scene control without opening fresh data."""

from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene-run", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    code = Path(__file__).parent
    name = a.scene_run
    assert Path(name).name == name and name not in (".", "..")
    assert (
        root / name / "result.json"
    ).exists(), "Complete training before calibration/replay"

    def run(script, args):
        with (root.parent / (name + "_" + script + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, str(code / (script + ".py")), *args],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

    run(
        "phase7_logit_fusion",
        ["--root", str(root), "--scene-run", name, "--run-name", name + "_fusion"],
    )
    run(
        "phase7_fusion_select",
        [
            "--root",
            str(root),
            "--fusion-run",
            name + "_fusion",
            "--run-name",
            name + "_selection",
        ],
    )
    candidate = json.loads((root / (name + "_selection/candidate.json")).read_text())
    report = dict(
        scene_run=name,
        development_retained=candidate["retained"],
        development_healthy_target=candidate["field_healthy_passed"],
        fresh_acceptance_opened=False,
        promotable=False,
    )
    if candidate["retained"] and candidate["field_healthy_passed"]:
        run(
            "phase9_replay",
            [
                "--root",
                str(root),
                "--selection-run",
                name + "_selection",
                "--run-name",
                name + "_opened_regression",
            ],
        )
        replay = json.loads(
            (root / (name + "_opened_regression/result.json")).read_text()
        )
        report["opened_regression_retained"] = replay["retained"]
        report["opened_tom2024_healthy"] = replay["candidate"]["tom2024"]["healthy"]
        report["opened_tom2024_false_healthy"] = replay["candidate"]["tom2024"]["all"][
            "false_healthy"
        ]
    (root / name / "pipeline_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
