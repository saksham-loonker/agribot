"""Join actual-byte no-leaf references; preserve historical baseline records."""

from pathlib import Path
import argparse
import json

from permissive_encoder import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    original = p / "phase20_whole_reference/predictions.jsonl"
    opened = p / "phase22_external_acceptance/predictions.jsonl"
    protocol = json.loads(
        (p / "phase22_external_acceptance/frozen_protocol.json").read_text()
    )
    assert protocol["baseline_manifest_sha256"] == digest(
        p / "export_cnn_safe/model_manifest.json"
    )
    records = {r["path"]: r for r in map(json.loads, original.read_text().splitlines())}
    for x in map(json.loads, opened.read_text().splitlines()):
        row = x["baseline"]
        assert row["source"] == "ad_greenhouse"
        if not row["leaves"]:
            assert digest(row["path"]) == row["file_sha256"]
            records[row["path"]] = {
                k: row[k] for k in ["path", "file_sha256", "whole_probabilities"]
            }
    out = p / "phase23_data"
    target = out / "whole_reference.jsonl"
    text = "".join(json.dumps(records[k]) + "\n" for k in sorted(records))
    if target.exists():
        assert target.read_text() == text
    else:
        target.write_text(text)
    report = dict(
        original_cache_sha256=digest(original),
        opened_ad_predictions_sha256=digest(opened),
        baseline_manifest_sha256=protocol["baseline_manifest_sha256"],
        predictions_sha256=digest(target),
        count=len(records),
        new_inference=False,
        independent=False,
    )
    (out / "whole_reference_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
