"""Replay each licensed tail-risk/reward control; never touch cold acceptance."""

from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase13_siglip"
    code = Path(__file__).resolve().parent
    ranked = json.loads((root / "risk_heads/ranked.json").read_text())
    for index, fit in enumerate(ranked):
        directory = Path(fit["checkpoint"]).parent
        mode = directory.name
        assert fit["protocol"]["restricted_teacher_used"] is False
        assert json.loads((directory / "restored_head_parity.json").read_text())[
            "passed"
        ]
        name = "risk_" + mode + "_selection"
        commands = [
            [
                sys.executable,
                "-u",
                str(code / "phase11_scene_select.py"),
                "--campaign",
                str(p),
                "--model-root",
                str(root),
                "--head-family",
                "siglip2",
                "--healthy-directory",
                str(directory),
                "--out-name",
                name,
                "--healthy-thresholds",
                "0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.999",
            ]
        ]
        with (root / (name + ".log")).open("a") as log:
            subprocess.run(
                commands[0], stdout=log, stderr=subprocess.STDOUT, check=True
            )
        selected = json.loads((root / name / "candidate.json").read_text())
        # Best by training validation receives an internal diagnostic even if
        # it fails old cohorts. Other heads only advance if development passes.
        if index == 0 or (selected["retained"] and selected["field_healthy_passed"]):
            with (root / (name + "_validation.log")).open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(code / "phase13_validate.py"),
                        "--campaign",
                        str(p),
                        "--selection",
                        name,
                        "--healthy-directory",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print(
            json.dumps(
                dict(
                    mode=mode,
                    replay_complete=True,
                    retained=selected["retained"],
                    field_target=selected["field_healthy_passed"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
