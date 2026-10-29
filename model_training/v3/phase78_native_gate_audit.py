"""Apply the original development recognition gates to full Kotlin verdicts."""

from pathlib import Path
import argparse
import json

from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    data = load(p)
    records = read(args.predictions)
    lookup = {r["id"]: r for r in records}
    actual = {}
    disease_changes = []
    for name, rows in list(data["base"].items()) + [("internal", data["internal"])]:
        for index, row in enumerate(rows):
            v = lookup[f"{name}:{index}"]
            if row["app_replay"]["kind"] == "DISEASE" and (v["kind"], v["label"]) != (
                row["app_replay"]["kind"],
                row["app_replay"]["label"],
            ):
                disease_changes.append(v)
            revised = dict(row)
            revised["app_replay"] = dict(
                row["app_replay"],
                kind=v["kind"],
                label=v["label"],
                frames=v["frames"],
                leaves=v["leaves"],
            )
            if v["primary_view"]:
                revised["app_replay"]["scope"] = "primary_visible_view"
            actual[id(row)] = revised
    assert len(actual) == len(records) == 6742
    screen = OpenedGuardScreen(data).evaluate(lambda row: actual[id(row)])
    screen["checks"].pop("original_disease_and_healthy_exact")
    screen["checks"]["original_named_disease_unchanged"] = not disease_changes
    screen["passed"] = all(screen["checks"].values())
    result = dict(
        passed=screen["passed"],
        checks=screen["checks"],
        audit=screen,
        temporal_differences=[r for r in records if not r["matches"]],
        named_disease_changes=disease_changes,
        predictions_sha256=digest(args.predictions),
        observations_sha256=digest(p / "phase77_native_temporal/temporal_cases.json"),
        source_sha256=digest(__file__),
        cold_olid_used=False,
        ad_independent=False,
    )
    out = p / "phase78_native_gate_audit"
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                checks=result["checks"],
                temporal_differences=len(result["temporal_differences"]),
            )
        )
    )
    assert result["passed"], "Native gate failure blocks fresh acceptance"


if __name__ == "__main__":
    main()
