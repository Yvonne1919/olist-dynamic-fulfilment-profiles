"""Run the frozen RQ1 test module and persist an auditable text receipt."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from analysis.rq1_speed_reliability_review_v1.scripts.rq1_io import (
    WORKSPACE,
    WORKING_DIR,
    write_json,
    write_text,
)


TEST_MODULE = WORKSPACE / "scripts/test_rq1_speed_reliability.py"
RUN_STATE_PATH = WORKING_DIR / "RUN_STATE.json"
TEST_RUNNER_MODULE = "analysis.rq1_speed_reliability_review_v1.scripts.run_rq1_tests"


def _counts(output: str) -> dict[str, int]:
    result = {
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
    }
    collected = re.search(r"collected\s+(\d+)\s+items?", output)
    if collected:
        result["collected"] = int(collected.group(1))
    for label in ("passed", "failed", "skipped", "deselected", "errors"):
        matches = re.findall(rf"(\d+)\s+{label}\b", output)
        if matches:
            result[label] = int(matches[-1])
    if not result["collected"]:
        result["collected"] = (
            result["passed"]
            + result["failed"]
            + result["skipped"]
            + result["errors"]
        )
    return result


def run(output: Path) -> int:
    target = output.resolve()
    try:
        target.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise ValueError("Test receipt must remain inside the RQ1 workspace") from exc
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        str(TEST_MODULE),
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    wrapper_command = shlex.join([
        sys.executable,
        "-B",
        "-m",
        TEST_RUNNER_MODULE,
        "--output",
        str(output),
    ])
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(WORKSPACE / "working/matplotlib")
    completed = subprocess.run(
        command,
        cwd=WORKSPACE.parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    counts = _counts(combined)
    receipt = {
        "command": command,
        "wrapper_command": wrapper_command,
        "return_code": int(completed.returncode),
        **counts,
    }
    body = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\n--- pytest output ---\n"
        + combined.rstrip()
    )
    write_text(target, body)
    if RUN_STATE_PATH.is_file():
        state = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
        events = list(state.get("events", []))
        events.append({
            "sequence": len(events) + 1,
            "stage": "tests",
            "status": "passed" if completed.returncode == 0 else "failed",
            "command": wrapper_command,
            "detail": {
                "receipt": str(target),
                "pytest_command": shlex.join(command),
                **counts,
                "return_code": completed.returncode,
            },
        })
        state["events"] = events
        write_json(RUN_STATE_PATH, state)
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.output))


if __name__ == "__main__":
    main()
