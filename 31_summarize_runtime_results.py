"""
31_summarize_runtime_results.py
==============================
Builds a leaderboard CSV from the runtime tracking tables.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from runtime_suite_common import (
    MAX_SELECTED_FRAME_LATENCY_MS,
    MAX_RUNTIME_ACCURACY_DROP_VS_PT,
    MIN_ACCEPTED_FRAME_ACCURACY,
    MIN_PIPELINE_FPS,
    PI_DEPLOYMENT_RUNTIMES,
    RUNTIME_ORDER,
    STRICT_FRAME_ACCURACY,
    suite_csv_path,
    upsert_csv,
    write_json,
)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError, OverflowError):
        return None


def as_bool(row: dict, key: str) -> bool:
    return row.get(key, "").strip().lower() == "true"


def index_rows(rows: list[dict], key_fields: list[str]) -> dict[tuple[str, ...], dict]:
    indexed = {}
    for row in rows:
        indexed[tuple(row.get(field, "") for field in key_fields)] = row
    return indexed


def main():
    export_rows = read_csv_rows(suite_csv_path("export_results.csv"))
    accuracy_rows = read_csv_rows(suite_csv_path("accuracy_results.csv"))
    benchmark_rows = read_csv_rows(suite_csv_path("benchmark_results.csv"))

    exports = index_rows(export_rows, ["runtime", "task"])
    accuracy = index_rows(accuracy_rows, ["runtime"])
    benchmarks = index_rows(benchmark_rows, ["runtime", "with_detector"])

    summary_path = suite_csv_path("runtime_leaderboard.csv")
    rows = []
    for runtime in RUNTIME_ORDER:
        acc = accuracy.get((runtime,), {})
        clf_only = benchmarks.get((runtime, "false"), {})
        full_pipe = benchmarks.get((runtime, "true"), {})
        deployment_accuracy = (
            full_pipe.get("overall_accuracy")
            or clf_only.get("overall_accuracy")
            or acc.get("accuracy", "")
        )

        row = {
            "runtime": runtime,
            "classifier_export": exports.get((runtime, "classify"), {}).get("status", ""),
            "detector_export": exports.get((runtime, "detect"), {}).get("status", ""),
            "accuracy": acc.get("accuracy", ""),
            "pt_agreement": acc.get("pt_agreement", ""),
            "classifier_img_per_sec": acc.get("img_per_sec", ""),
            "benchmark_classifier_only_fps": clf_only.get("avg_fps", ""),
            "benchmark_classifier_only_accuracy": clf_only.get("overall_accuracy", ""),
            "benchmark_full_pipeline_fps": full_pipe.get("avg_fps", ""),
            "benchmark_full_pipeline_accuracy": full_pipe.get("overall_accuracy", ""),
            "deployment_accuracy": deployment_accuracy,
            "accuracy_report": acc.get("report_txt", ""),
            "classifier_only_report": clf_only.get("report_json", ""),
            "full_pipeline_report": full_pipe.get("report_json", ""),
            "full_pipeline_latency_ms": full_pipe.get("avg_latency_ms", ""),
            "full_pipeline_max_latency_ms": full_pipe.get("max_latency_ms", ""),
            "over_200ms_frames": full_pipe.get("over_hard_latency_frames", ""),
        }
        rows.append(row)
        upsert_csv(summary_path, key_fields=["runtime"], row=row)

    pt_row = next((row for row in rows if row["runtime"] == "pt"), {})
    pt_deployment_accuracy = as_float(pt_row, "deployment_accuracy")

    candidates = []
    for row in rows:
        runtime = row["runtime"]
        if runtime not in PI_DEPLOYMENT_RUNTIMES:
            continue

        acc = as_float(row, "deployment_accuracy")
        fps = as_float(row, "benchmark_full_pipeline_fps")
        latency = as_float(row, "full_pipeline_latency_ms")
        max_latency = as_float(row, "full_pipeline_max_latency_ms")
        full_pipe = benchmarks.get((runtime, "true"), {})
        preserves_pt_accuracy = (
            runtime == "pt"
            or pt_deployment_accuracy is None
            or acc is None
            or acc >= pt_deployment_accuracy - MAX_RUNTIME_ACCURACY_DROP_VS_PT
        )

        eligible = (
            acc is not None
            and fps is not None
            and latency is not None
            and max_latency is not None
            and acc >= MIN_ACCEPTED_FRAME_ACCURACY
            and fps >= MIN_PIPELINE_FPS
            and latency <= MAX_SELECTED_FRAME_LATENCY_MS
            and max_latency <= MAX_SELECTED_FRAME_LATENCY_MS
            and preserves_pt_accuracy
            and as_bool(full_pipe, "meets_min_fps")
            and as_bool(full_pipe, "meets_latency_ceiling")
            and as_bool(full_pipe, "complete")
        )
        strict_accuracy = acc is not None and acc >= STRICT_FRAME_ACCURACY
        candidates.append(
            {
                "runtime": runtime,
                "eligible": eligible,
                "strict_accuracy": strict_accuracy,
                "deployment_accuracy": acc,
                "full_pipeline_fps": fps,
                "avg_latency_ms": latency,
                "max_latency_ms": max_latency,
                "preserves_pt_accuracy": preserves_pt_accuracy,
                "classifier_model": full_pipe.get("classifier_model", ""),
                "detector_model": full_pipe.get("detector_model", ""),
                "reason": (
                    "passes strict accuracy target"
                    if eligible and strict_accuracy
                    else "passes accepted accuracy target"
                    if eligible
                    else "fails one or more deployment constraints"
                ),
            }
        )

    eligible_candidates = [row for row in candidates if row["eligible"]]
    strict_candidates = [row for row in eligible_candidates if row["strict_accuracy"]]
    choice_pool = strict_candidates or eligible_candidates
    selected = (
        max(choice_pool, key=lambda row: row["full_pipeline_fps"] or -1.0)
        if choice_pool
        else None
    )

    best_accuracy = max(
        rows,
        key=lambda row: float(row["accuracy"]) if row["accuracy"] else -1.0,
    ) if rows else None
    best_clf_fps = max(
        rows,
        key=lambda row: float(row["benchmark_classifier_only_fps"]) if row["benchmark_classifier_only_fps"] else -1.0,
    ) if rows else None
    best_pipeline_fps = max(
        rows,
        key=lambda row: float(row["benchmark_full_pipeline_fps"]) if row["benchmark_full_pipeline_fps"] else -1.0,
    ) if rows else None

    best_path = suite_csv_path("best_runtime_summary.csv")
    if best_accuracy:
        upsert_csv(
            best_path,
            key_fields=["category"],
            row={
                "category": "best_accuracy",
                "runtime": best_accuracy["runtime"],
                "value": best_accuracy["accuracy"],
            },
        )
    if best_clf_fps:
        upsert_csv(
            best_path,
            key_fields=["category"],
            row={
                "category": "best_classifier_only_fps",
                "runtime": best_clf_fps["runtime"],
                "value": best_clf_fps["benchmark_classifier_only_fps"],
            },
        )
    if best_pipeline_fps:
        upsert_csv(
            best_path,
            key_fields=["category"],
            row={
                "category": "best_full_pipeline_fps",
                "runtime": best_pipeline_fps["runtime"],
                "value": best_pipeline_fps["benchmark_full_pipeline_fps"],
            },
        )

    recommendation_path = suite_csv_path("deployment_recommendation.json")
    write_json(
        recommendation_path,
        {
            "status": "pass" if selected else "fail",
            "selected_runtime": selected["runtime"] if selected else None,
            "selected": selected,
            "targets": {
                "accepted_frame_accuracy": MIN_ACCEPTED_FRAME_ACCURACY,
                "strict_frame_accuracy": STRICT_FRAME_ACCURACY,
                "min_fps": MIN_PIPELINE_FPS,
                "max_selected_frame_latency_ms": MAX_SELECTED_FRAME_LATENCY_MS,
                "max_accuracy_drop_vs_pt": MAX_RUNTIME_ACCURACY_DROP_VS_PT,
                "accuracy_source": "benchmark_full_pipeline_accuracy when available, otherwise classifier-only",
            },
            "candidates": candidates,
        },
    )
    if selected:
        upsert_csv(
            best_path,
            key_fields=["category"],
            row={
                "category": "deployment_choice",
                "runtime": selected["runtime"],
                "value": selected["full_pipeline_fps"],
            },
        )

    print(f"Leaderboard -> {summary_path}")
    print(f"Best summary -> {best_path}")
    print(f"Deployment recommendation -> {recommendation_path}")
    if selected:
        print(
            "Selected runtime -> "
            f"{selected['runtime']} ({selected['full_pipeline_fps']:.2f} FPS, "
            f"{selected['deployment_accuracy'] * 100:.2f}% detector-first accuracy)"
        )
    else:
        print("Selected runtime -> none; no runtime passed the Pi deployment constraints")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
