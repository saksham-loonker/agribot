"""Resume Apache-only controls and internal validation; never open cold AD.

Use only when no other phase13 tmux worker is active. Source preparation and
human-data admission must already exist, with the pinned provenance records.
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

from permissive_encoder import digest, REPOSITORY, REVISION, SOURCE_SHA256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign).resolve()
    root = p / "phase13_siglip"
    code = Path(__file__).resolve().parent
    source = json.loads((root / "encoder_source.json").read_text())
    assert source["license"] == "apache-2.0"
    assert source["restricted_teacher_used"] is False
    assert (source["repository"], source["revision"], source["file_sha256"]) == (
        REPOSITORY,
        REVISION,
        SOURCE_SHA256,
    )
    assert (root / "RETIRED_BENCHMARK.json").exists()
    log_root = root / "resumable_pipeline"
    log_root.mkdir(exist_ok=True)
    common = [
        "--campaign",
        str(p),
        "--model-root",
        str(root),
        "--head-family",
        "siglip2",
    ]
    stages = [
        (
            "feature_heads",
            "phase10_feature_train.py",
            [
                "--root",
                str(root),
                "--feature-family",
                "siglip2",
                "--task-scope",
                "primary_image",
                "--primary-source",
                "tom2024",
            ],
            [
                "siglip2_feature_heads/ranked.json",
                "siglip2_feature_heads/restored_encoder_parity.json",
            ],
        ),
        (
            "logit_fusion",
            "phase11_logit_fusion.py",
            common,
            ["logit_fusion/candidates.json"],
        ),
        (
            "frozen_replay",
            "phase11_scene_select.py",
            common,
            ["selection/candidate.json"],
        ),
        (
            "fusion_replay",
            "phase11_scene_select.py",
            common
            + [
                "--fusion-run",
                "logit_fusion",
                "--out-name",
                "fusion_refined_selection",
                "--healthy-thresholds",
                "0.951,0.955,0.96,0.965,0.968,0.975,0.98,0.985",
            ],
            ["fusion_refined_selection/candidate.json"],
        ),
        (
            "fusion_validation",
            "phase13_validate.py",
            ["--campaign", str(p), "--selection", "fusion_refined_selection"],
            ["fusion_refined_selection_validation/result.json"],
        ),
    ]
    for loss in ["focal", "ce"]:
        selected = "sft_" + loss + "_selection"
        stages += [
            (
                loss + "_training",
                "phase12_scene_sft.py",
                [
                    "--root",
                    str(root),
                    "--family",
                    "siglip2",
                    "--loss",
                    loss,
                    "--batch",
                    "32",
                ],
                [
                    "sft_" + loss + "/result.json",
                    "sft_" + loss + "/restored_model_parity.json",
                ],
            ),
            (
                loss + "_replay",
                "phase11_scene_select.py",
                common
                + [
                    "--healthy-directory",
                    str(root / ("sft_" + loss)),
                    "--out-name",
                    selected,
                    "--healthy-thresholds",
                    "0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.995,0.999",
                ],
                [selected + "/candidate.json"],
            ),
            (
                loss + "_validation",
                "phase13_validate.py",
                [
                    "--campaign",
                    str(p),
                    "--selection",
                    selected,
                    "--healthy-directory",
                    str(root / ("sft_" + loss)),
                ],
                [selected + "_validation/result.json"],
            ),
        ]
    for name, script, args, outputs in stages:
        marker = log_root / (name + ".json")
        identity = dict(
            script=script,
            script_sha256=digest(code / script),
            arguments=args,
            encoder_source_sha256=digest(root / "encoder_source.json"),
        )
        if marker.exists():
            done = json.loads(marker.read_text())
            assert done["identity"] == identity, (
                "Pipeline stage changed; version its records",
                name,
            )
            assert done["outputs"] == {f: digest(root / f) for f in outputs}
            print(json.dumps(dict(stage=name, already_verified=True)), flush=True)
            continue
        with (log_root / (name + ".log")).open("a") as log:
            subprocess.run(
                [sys.executable, "-u", str(code / script), *args],
                cwd=code,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        marker.write_text(
            json.dumps(
                dict(identity=identity, outputs={f: digest(root / f) for f in outputs}),
                indent=2,
            )
        )
        print(json.dumps(dict(stage=name, completed=True)), flush=True)
    subprocess.run(
        [
            sys.executable,
            str(code / "phase13_report.py"),
            "--campaign",
            str(p),
            "--out",
            str(root / "evidence.json"),
        ],
        check=True,
    )
    print(
        "Development/internal checks complete; inspect qualification results. Cold AD remains unopened; no model installed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
