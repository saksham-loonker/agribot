"""Resume expanded image-scoped SFT controls and opened app checks; no AD use."""

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
    root = p / "phase12_scene"
    code = Path(__file__).resolve().parent
    assert (root / "RETIRED_BENCHMARK.json").exists()
    assert (
        json.loads((root / "complete.json").read_text())[
            "cold_ad_diagnostic_predictions_opened"
        ]
        is False
    )

    def run(script, args, log_name):
        with (root / log_name).open("a", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(code / (script + ".py")), *args],
                cwd=code,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for loss in ["ce", "focal", "distill"]:
        run(
            "phase12_scene_sft",
            ["--root", str(root), "--loss", loss],
            "sft_" + loss + ".log",
        )
        name = "sft_" + loss + "_selection"
        if not (root / name / "candidate.json").exists():
            run(
                "phase11_scene_select",
                [
                    "--campaign",
                    str(p),
                    "--model-root",
                    str(root),
                    "--healthy-directory",
                    str(root / ("sft_" + loss)),
                    "--out-name",
                    name,
                ],
                name + ".log",
            )
        c = json.loads((root / name / "candidate.json").read_text())
        print(
            json.dumps(
                dict(
                    control=loss,
                    retained=c["retained"],
                    field_healthy_passed=c["field_healthy_passed"],
                    fresh_ad_used=False,
                )
            ),
            flush=True,
        )
    print(
        "Expanded SFT controls complete. Independent acceptance, export and native validation remain separate required gates.",
        flush=True,
    )


if __name__ == "__main__":
    main()
