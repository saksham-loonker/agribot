"""Select and verify completed adaptation controls, without opening new cohorts."""

from pathlib import Path
import argparse, json, subprocess, sys, time, traceback
from phase2_accept import metrics
from phase2_select import retain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    q = p / "phase2"
    out = q / "refinements"
    out.mkdir(exist_ok=True)
    results = []
    scripts = Path(__file__).parent
    baseline = metrics(q / "slif_evaluation/baseline")
    for method in ["logit", "reward_ce", "ppo"]:
        directory = out / method
        directory.mkdir(exist_ok=True)
        decision = directory / "decision.json"
        if decision.exists():
            results.append(json.loads(decision.read_text()))
            continue
        started = time.time()
        run = q / "multiclass" / ("lora_d4_" + method)
        while not (run / "result.json").exists():
            if time.time() - started > 1800:
                raise TimeoutError("Adapter control did not finish: " + method)
            time.sleep(5)

        def stage(name, args):
            print(json.dumps(dict(method=method, stage=name)), flush=True)
            with (directory / (name + ".log")).open("w") as stream:
                subprocess.run(
                    [sys.executable, str(scripts / args[0]), *map(str, args[1:])],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

        result = dict(method=method, passed=False)
        try:
            selection = directory / "selection"
            stage(
                "selection",
                [
                    "phase2_select.py",
                    "--root",
                    q,
                    "--class-conditional",
                    "--student-modes",
                    "lora_d4_" + method,
                    "--out",
                    selection,
                ],
            )
            candidate = json.loads((selection / "candidate.json").read_text())
            result["validation_score"] = candidate["score"]
            if candidate["retained"]:
                assets = directory / "export"
                stage(
                    "export",
                    [
                        "phase2_export.py",
                        "--root",
                        q,
                        "--candidate",
                        selection / "candidate.json",
                        "--out",
                        assets,
                    ],
                )
                stage(
                    "actual_bytes",
                    [
                        "phase2_verify.py",
                        "--root",
                        q,
                        "--assets",
                        assets,
                        "--out",
                        assets / "regressions",
                    ],
                )
                slif = directory / "opened_slif"
                stage(
                    "opened_slif",
                    [
                        "app_eval.py",
                        "--assets",
                        assets,
                        "--manifest",
                        q / "slif_acceptance_manifest.json",
                        "--out",
                        slif,
                        "--tag",
                        "opened_slif_" + method,
                        "--workers",
                        "4",
                        "--threads",
                        "2",
                        "--resume",
                    ],
                )
                observed = metrics(slif)
                passed = (
                    retain(dict(slif=observed), dict(slif=baseline))
                    and observed["healthy"]["correct_rate"] >= 0.7
                )
                result.update(
                    passed=passed,
                    assets=str(assets),
                    candidate=str(selection / "candidate.json"),
                    slif_baseline=baseline,
                    slif_candidate=observed,
                    slif_status="Opened regression; not independent acceptance",
                    model_sha256=json.loads(
                        (assets / "model_manifest.json").read_text()
                    )["classifier"]["members"][0]["sha256"],
                )
            else:
                result["reason"] = "Validation disease-retention gate failed"
        except Exception as error:
            result.update(reason=repr(error), traceback=traceback.format_exc())
        decision.write_text(json.dumps(result, indent=2))
        results.append(result)
        print(
            json.dumps(
                dict(
                    method=method, passed=result["passed"], reason=result.get("reason")
                )
            ),
            flush=True,
        )
    passed = sorted(
        [r for r in results if r["passed"]],
        key=lambda r: r["validation_score"],
        reverse=True,
    )
    (out / "results.json").write_text(json.dumps(results, indent=2))
    selection = dict(
        passed=bool(passed),
        selected=passed[0] if passed else None,
        rule="Highest grouped validation score among candidates passing actual-byte old regressions and opened SLIF regression. New TOM2024 and paper-leaf cohorts remain unopened.",
    )
    (out / "selected.json").write_text(json.dumps(selection, indent=2))
    print(
        json.dumps(dict(selected=passed[0]["method"] if passed else None)), flush=True
    )


if __name__ == "__main__":
    main()
