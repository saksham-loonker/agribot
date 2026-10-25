"""Separate detector/classifier effects with actual model bytes on opened photos.

The same-source prospective reservation is intentionally not read. All variants
must pass development safety and source-specific Healthy checks before it opens.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import sys

from phase2_accept import metrics
from phase2_select import retain


def freeze(path, content):
    if path.exists():
        assert (
            json.loads(path.read_text()) == content
        ), f"Frozen identity changed: {path}"
    else:
        path.write_text(json.dumps(content, indent=2))


def combined_assets(campaign):
    detector = campaign / "phase6_detector/export"
    classifier = campaign / "phase5/export_open_set"
    out = campaign / "phase6_detector/combined"
    (out / "models").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((classifier / "model_manifest.json").read_text())
    det = json.loads((detector / "model_manifest.json").read_text())
    manifest["detector"] = det["detector"]
    manifest["detector_training"] = det["detector_training"]
    manifest["bundle_id"] = (
        "agribot-v3-pseudo-leaf-"
        + manifest["detector"]["sha256"][:8]
        + "-"
        + manifest["classifier"]["members"][0]["sha256"][:8]
    )
    manifest["status"] = (
        "EXPERIMENTAL: app regression and untouched acceptance incomplete"
    )
    freeze(out / "model_manifest.json", manifest)
    for member in manifest["classifier"]["members"]:
        path = out / "models" / member["file"]
        if not path.exists():
            path.symlink_to(classifier / "models" / member["file"])
    path = out / "models" / manifest["detector"]["file"]
    if not path.exists():
        path.symlink_to(detector / "models" / manifest["detector"]["file"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase6_detector"
    out = root / "app_validation"
    out.mkdir(exist_ok=True)
    assert json.loads((root / "export/export_verification.json").read_text())["passed"]
    assert json.loads(
        (
            p / "phase5/export_open_set/regressions/regression_verification.json"
        ).read_text()
    )["passed"]
    assets = combined_assets(p)
    ev = json.loads((p / "phase5/eval_rows.json").read_text())
    selected = copy.deepcopy(ev["phase5_selection"])
    # Remove previous predictions; retain public source identity/label and eligibility.
    for row in selected:
        for key in [
            "leaves",
            "detections",
            "app_replay",
            "ungated_app_replay",
            "whole_member_logits",
            "whole_probabilities",
            "whole_label",
            "oracle_objects",
        ]:
            row.pop(key, None)
    manifest_path = out / "development_manifest.json"
    freeze(manifest_path, selected)
    baseline = json.loads((p / "phase5/open_set_selection/baseline.json").read_text())[
        "phase5_selection"
    ]
    protocol = dict(
        scope="Opened grouped development photos only; not prospective acceptance",
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_healthy_min_recall=0.7,
        healthy_gain_min=0.1,
        retention="Existing phase-two noninferiority rule unchanged; integer count allowances round down",
        variants={"detector_only": str(root / "export"), "combined": str(assets)},
        reserved_used=False,
    )
    freeze(out / "protocol.json", protocol)
    result = {}
    for name, bundle in protocol["variants"].items():
        destination = out / name
        with (out / (name + ".log")).open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("app_eval.py")),
                    "--assets",
                    bundle,
                    "--manifest",
                    str(manifest_path),
                    "--out",
                    str(destination),
                    "--tag",
                    "phase6_" + name,
                    "--workers",
                    "4",
                    "--threads",
                    "2",
                    "--resume",
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        observed = metrics(destination)
        checks = dict(
            noninferiority=retain(dict(new=observed), dict(new=baseline)),
            healthy_gain=(observed["healthy"]["correct_rate"] or 0)
            >= (baseline["healthy"]["correct_rate"] or 0) + 0.1,
        )
        for source in ("tom2024", "swtt5wy2pr"):
            checks[source + "_healthy_min"] = (
                observed["by_source"][source]["healthy_recall"] >= 0.7
            )
        result[name] = dict(
            metrics=observed, checks=checks, passed=all(checks.values())
        )
        print(
            json.dumps(
                dict(
                    variant=name,
                    checks=checks,
                    healthy=observed["healthy"]["correct"],
                    false_healthy=observed["all"]["false_healthy"],
                    sources=observed["by_source"],
                )
            ),
            flush=True,
        )
        (out / "result.json").write_text(
            json.dumps(
                dict(protocol=protocol, baseline=baseline, variants=result), indent=2
            )
        )


if __name__ == "__main__":
    main()
