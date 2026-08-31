#!/usr/bin/env python3
"""Run and persist the frozen V1 profile-validation pytest suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
TEST_FILE = ROOT / "analysis/dynamic_profile_profile_validation_v1/scripts/test_profile_validation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-artifact", action="store_true")
    return parser.parse_args()


def _count(text: str, label: str) -> int:
    matches = re.findall(rf"(?<!\d)(\d+)\s+{re.escape(label)}\b", text)
    return int(matches[-1]) if matches else 0


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        str(TEST_FILE),
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if args.exclude_artifact:
        command.extend(["-k", "not artifact"])
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    duration = time.monotonic() - started
    output = completed.stdout or ""
    passed = _count(output, "passed")
    failed = _count(output, "failed")
    skipped = _count(output, "skipped")
    deselected = _count(output, "deselected")
    errors = _count(output, "error") + _count(output, "errors")
    collected = passed + failed + skipped + deselected + errors
    log = "\n".join([
        f"COMMAND: {shlex.join(command)}",
        f"CWD: {ROOT}",
        f"RETURN_CODE: {completed.returncode}",
        f"COLLECTED: {collected}",
        f"PASSED: {passed}",
        f"FAILED: {failed}",
        f"SKIPPED: {skipped}",
        f"DESELECTED: {deselected}",
        f"ERRORS: {errors}",
        f"DURATION_SECONDS: {duration:.6f}",
        "",
        "PYTEST_OUTPUT:",
        output.rstrip(),
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(log, encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "return_code": completed.returncode,
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "deselected": deselected,
        "duration_seconds": round(duration, 3),
        "output": str(args.output),
    }, sort_keys=True))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
