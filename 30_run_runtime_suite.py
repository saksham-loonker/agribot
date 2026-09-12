"""
30_run_runtime_suite.py
======================
Runs the export + accuracy + benchmark suite in order and writes a suite run log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from runtime_suite_common import suite_csv_path, upsert_csv


ORDERED_COMMANDS = [
    ["10_export_runtime_models.py", "--runtimes", "ncnn", "openvino"],
    ["11_test_runtime_pt.py"],
    ["13_test_runtime_ncnn.py"],
    ["14_test_runtime_openvino.py"],
    ["21_benchmark_runtime_pt.py"],
    ["23_benchmark_runtime_ncnn.py"],
    ["24_benchmark_runtime_openvino.py"],
    ["21_benchmark_runtime_pt.py", "--with-detector"],
    ["23_benchmark_runtime_ncnn.py", "--with-detector"],
    ["24_benchmark_runtime_openvino.py", "--with-detector"],
    ["31_summarize_runtime_results.py"],
]


def render_progress(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]   0%"
    ratio = max(0.0, min(1.0, current / total))
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}%"


def parse_args():
    p = argparse.ArgumentParser(description="Run the runtime suite in order")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run all commands for diagnostics, but still exit nonzero if any command fails",
    )
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def should_skip(command: list[str], args) -> bool:
    return args.skip_export and command[0] == "10_export_runtime_models.py"


def main():
    args = parse_args()
    python_exe = args.python

    runnable_commands = [cmd for cmd in ORDERED_COMMANDS if not should_skip(cmd, args)]
    total_commands = len(runnable_commands)
    completed = 0
    failures = []
    stop_on_error = args.stop_on_error or not args.continue_on_error

    for command in ORDERED_COMMANDS:
        if should_skip(command, args):
            continue

        print(f"\nProgress {completed}/{total_commands} {render_progress(completed, total_commands)}")
        started = time.perf_counter()
        display = " ".join(command)
        print(f"\n>>> {display}\n")
        result = subprocess.run([python_exe, *command], cwd=Path(__file__).parent)
        elapsed = time.perf_counter() - started
        completed += 1
        status_text = "ok" if result.returncode == 0 else "failed"
        print(
            f"\nProgress {completed}/{total_commands} "
            f"{render_progress(completed, total_commands)} "
            f"({status_text})"
        )

        upsert_csv(
            suite_csv_path("suite_runs.csv"),
            key_fields=["command"],
            row={
                "command": display,
                "status": "ok" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "elapsed_sec": f"{elapsed:.6f}",
            },
        )

        if result.returncode != 0 and args.stop_on_error:
            raise SystemExit(result.returncode)

        if result.returncode != 0:
            failures.append((display, result.returncode))
        if result.returncode != 0 and stop_on_error:
            raise SystemExit(result.returncode)

    if failures:
        print("\nRuntime suite failed:")
        for command, return_code in failures:
            print(f"  {command} (exit {return_code})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
