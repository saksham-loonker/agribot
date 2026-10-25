"""Open the sealed external cohort once, after candidate and protocol freezing."""

from pathlib import Path
import argparse, hashlib, json, subprocess, sys
from phase2_select import retain
from app_eval import summarize
from train import LABELS


def metrics(path):
    rows = [
        json.loads(line)
        for line in (path / "predictions.jsonl").read_text().splitlines()
    ]
    m = summarize(rows)
    m["disease_classes"] = {}
    for label in LABELS:
        if label in ("Healthy", "Other"):
            continue
        subset = [
            r for r in rows if r["label"] == label and r.get("analysis_eligible", True)
        ]
        m["disease_classes"][label] = dict(
            n=len(subset),
            correct=sum(
                r["app_replay"]["kind"] == "DISEASE"
                and r["app_replay"]["label"] == label
                for r in subset
            ),
        )
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    q = Path(a.root)
    protocol = json.loads((q / "slif_frozen_protocol.json").read_text())
    assert (
        hashlib.sha256((q / "frozen_candidate_v4.json").read_bytes()).hexdigest()
        == protocol["candidate_sha256"]
    )
    assert (
        hashlib.sha256((q / "slif_acceptance_manifest.json").read_bytes()).hexdigest()
        == protocol["manifest_sha256"]
    )
    assert json.loads(
        (q / "export_v4/regressions/regression_verification.json").read_text()
    )["passed"]
    assets = [
        ("baseline", q.parent / "export_cnn_safe"),
        ("candidate", q / "export_v4"),
    ]
    opening = q / "slif_opening.json"
    expected = dict(
        protocol=protocol,
        model_hashes={
            name: json.loads((path / "model_manifest.json").read_text())["classifier"][
                "members"
            ][0]["sha256"]
            for name, path in assets
        },
    )
    exported = json.loads((q / "export_v4/model_manifest.json").read_text())
    assert exported["training"]["candidate_sha256"] == protocol["candidate_sha256"]
    if opening.exists():
        assert (
            json.loads(opening.read_text()) == expected
        ), "External evaluation identity changed"
    else:
        opening.write_text(json.dumps(expected, indent=2))
    for name, path in assets:
        out = q / "slif_evaluation" / name
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("app_eval.py")),
                "--assets",
                str(path),
                "--manifest",
                str(q / "slif_acceptance_manifest.json"),
                "--out",
                str(out),
                "--tag",
                "frozen_slif_" + name,
                "--workers",
                "4",
                "--threads",
                "2",
                "--resume",
            ],
            check=True,
        )
    baseline = metrics(q / "slif_evaluation/baseline")
    candidate = metrics(q / "slif_evaluation/candidate")
    report = dict(
        protocol=protocol,
        baseline=baseline,
        candidate=candidate,
        noninferiority_passed=retain(dict(slif=candidate), dict(slif=baseline)),
        healthy_min_passed=(candidate["healthy"]["correct_rate"] or 0)
        >= protocol["healthy_min_recall"],
        note="First opening of this external cohort after freeze. Published augmented views, inferred capture families, 44 Healthy photos; uncertainty remains substantial.",
    )
    report["passed"] = report["noninferiority_passed"] and report["healthy_min_passed"]
    (q / "slif_acceptance_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
