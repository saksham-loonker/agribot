"""Freeze and open two fresh classification cohorts once; never tune on results."""

from pathlib import Path
import argparse, hashlib, json, math, subprocess, sys
import numpy as np
from app_eval import summarize


def read_rows(path):
    return [
        json.loads(l) for l in (path / "predictions.jsonl").read_text().splitlines()
    ]


def assess(before, after, paper=False):
    b = summarize(before)
    c = summarize(after)
    checks = {
        "healthy_no_decrease": c["healthy"]["correct"] >= b["healthy"]["correct"],
        "no_additional_false_healthy": c["all"]["false_healthy"]
        <= b["all"]["false_healthy"],
        "healthy_false_other_no_increase": c["healthy"]["false_other"]
        <= b["healthy"]["false_other"],
        "unsupported_false_known_no_increase": (
            c["unsupported_tomato"]["false_known"] or 0
        )
        <= (b["unsupported_tomato"]["false_known"] or 0),
    }
    if paper:
        checks["healthy_min_70_percent"] = (c["healthy"]["correct_rate"] or 0) >= 0.7
        for label in ["Early_blight", "Late_blight"]:
            base = sum(
                r["app_replay"]["kind"] == "DISEASE"
                and r["app_replay"]["label"] == label
                for r in before
                if r["label"] == label
            )
            final = sum(
                r["app_replay"]["kind"] == "DISEASE"
                and r["app_replay"]["label"] == label
                for r in after
                if r["label"] == label
            )
            checks[label + "_no_decrease"] = final >= base
    return dict(baseline=b, candidate=c, checks=checks, passed=all(checks.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    assets = p / "phase4_confirmed"
    out = assets / "fresh_acceptance"
    out.mkdir(exist_ok=True)
    regression = json.loads(
        (assets / "regressions/regression_verification.json").read_text()
    )
    assert regression["passed"]
    # The opened SLIF regression must also retain the installed model's counts.
    from phase2_accept import metrics
    from phase2_select import retain

    sm = metrics(assets / "opened_slif")
    sb = metrics(p / "phase2/slif_evaluation/baseline")
    assert retain(dict(slif=sm), dict(slif=sb))
    manifests = {
        "tom2024": p / "phase4_tom2024/manifest.json",
        "paper_leaf": p / "phase4_paper_leaf/manifest.json",
    }
    manifest = json.loads((assets / "model_manifest.json").read_text())
    model_sha = manifest["classifier"]["members"][0]["sha256"]
    protocol = dict(
        model_sha256=model_sha,
        manifest_hashes={
            k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in manifests.items()
        },
        healthy_correct_no_decrease_per_source=True,
        false_healthy_no_increase_per_source=True,
        unsupported_false_known_no_increase_per_source=True,
        paper_healthy_min_recall=0.7,
        paper_named_disease_correct_no_decrease_per_class=True,
        aggregate_healthy_gain_min=0.05,
        paired_healthy_bootstrap_lower_bound_min=0.0,
        coarse_disease_scope="TOM2024 virosis is binary diseased only. Named diagnosis and unknown rejection accuracy are not inferred from that label.",
        fresh_sources="No source predictions entered fitting, calibration, policy selection or checkpoint selection.",
        rule="Failure prevents automatic promotion. Do not tune on these cohorts after opening.",
    )
    frozen = out / "frozen_protocol.json"
    if frozen.exists():
        assert (
            json.loads(frozen.read_text()) == protocol
        ), "Frozen acceptance identity changed"
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    for cohort, path in manifests.items():
        for name, bundle in [
            ("baseline", p / "export_cnn_safe"),
            ("candidate", assets),
        ]:
            destination = out / cohort / name
            with (out / (cohort + "_" + name + ".log")).open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("app_eval.py")),
                        "--assets",
                        str(bundle),
                        "--manifest",
                        str(path),
                        "--out",
                        str(destination),
                        "--tag",
                        cohort + "_" + name,
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
            print(
                json.dumps(dict(cohort=cohort, model=name, stage="completed")),
                flush=True,
            )
    results = {}
    differences = []
    healthy_total = 0
    for cohort in manifests:
        before = read_rows(out / cohort / "baseline")
        after = read_rows(out / cohort / "candidate")
        results[cohort] = assess(before, after, cohort == "paper_leaf")
        lookup = {r["id"]: r for r in before}
        for row in after:
            if row["label"] == "Healthy":
                differences.append(
                    int(row["app_replay"]["kind"] == "HEALTHY")
                    - int(lookup[row["id"]]["app_replay"]["kind"] == "HEALTHY")
                )
                healthy_total += 1
    delta = np.array(differences)
    rng = np.random.default_rng(89)
    samples = np.array(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(4000)]
    )
    interval = np.quantile(samples, [0.025, 0.975]).tolist()
    gain = float(delta.mean())
    report = dict(
        protocol=protocol,
        cohorts=results,
        healthy_n=healthy_total,
        paired_healthy_gain=gain,
        paired_bootstrap_95=interval,
        bootstrap_scope="One selected photo per inferred capture group; these are not verified plant IDs. Interval is exploratory.",
        passed=all(r["passed"] for r in results.values())
        and gain >= 0.05
        and interval[0] >= 0.0,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                gain=gain,
                interval=interval,
                cohorts={k: v["checks"] for k, v in results.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
