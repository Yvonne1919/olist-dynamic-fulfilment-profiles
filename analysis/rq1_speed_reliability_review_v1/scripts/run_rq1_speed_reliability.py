"""Stage runner for the frozen supplementary RQ1 analysis."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.rq1_speed_reliability_review_v1.scripts.rq1_data import (
    build_analysis_frames,
    build_data_audit_tables,
)
from analysis.rq1_speed_reliability_review_v1.scripts.rq1_io import (
    WORKING_DIR,
    WORKSPACE,
    ensure_workspace_dirs,
    write_json,
)
from analysis.rq1_speed_reliability_review_v1.scripts.rq1_preflight import (
    PRESTATE_PATH,
    _local_frozen_inputs,
    preflight,
    verify_protected_unchanged,
)
from analysis.rq1_speed_reliability_review_v1.scripts.rq1_reporting import (
    finalize_manifest,
    write_analysis_artifacts,
)
from analysis.rq1_speed_reliability_review_v1.scripts.rq1_stats import (
    run_statistical_analysis,
)


CONFIG_PATH = WORKSPACE / "RQ1_SPEED_RELIABILITY_FROZEN_CONFIG.json"
RUN_STATE_PATH = WORKING_DIR / "RUN_STATE.json"
RUNNER_MODULE = "analysis.rq1_speed_reliability_review_v1.scripts.run_rq1_speed_reliability"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command() -> str:
    # ``sys.argv[0]`` is the resolved file path even when Python was invoked
    # with ``-m``.  Persist the actual, replayable package-module invocation.
    return shlex.join([sys.executable, "-B", "-m", RUNNER_MODULE, *sys.argv[1:]])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_event(stage: str, status: str, detail: dict[str, Any] | None = None) -> None:
    state = _load_json(RUN_STATE_PATH) if RUN_STATE_PATH.is_file() else {
        "analysis_id": "RQ1_SPEED_RELIABILITY_REVIEW_V1",
        "events": [],
    }
    events = list(state.get("events", []))
    events.append({
        "sequence": len(events) + 1,
        "stage": stage,
        "status": status,
        "at_utc": _utc_now(),
        "command": _command(),
        "detail": detail or {},
    })
    state["events"] = events
    write_json(RUN_STATE_PATH, state)


def _require_prestate() -> dict[str, Any]:
    if not PRESTATE_PATH.is_file():
        raise RuntimeError("Formal preflight receipt is missing")
    state = _load_json(PRESTATE_PATH)
    if state.get("status") != "passed":
        raise RuntimeError("Formal preflight did not pass")
    current = _local_frozen_inputs()
    if current != state.get("local_frozen_input_hashes"):
        raise RuntimeError("Frozen local source/config hashes changed after preflight")
    return state


def run_preflight(data_dir: Path) -> None:
    ensure_workspace_dirs()
    state = preflight(data_dir, _command())
    _append_event(
        "preflight",
        "passed",
        {
            "prestate_sha_recorded": True,
            "protected_file_count": state["protected_baseline"]["file_count"],
            "protected_aggregate_sha256": state["protected_baseline"]["aggregate_sha256"],
        },
    )


def run_analysis(data_dir: Path) -> None:
    ensure_workspace_dirs()
    prestate = _require_prestate()
    config = _load_json(CONFIG_PATH)
    _append_event("analysis", "started")
    all_orders, reviewed, audit = build_analysis_frames(data_dir, config)
    data_tables = build_data_audit_tables(all_orders, reviewed, audit, config)
    statistics = run_statistical_analysis(reviewed, config)
    receipts = write_analysis_artifacts(
        all_orders=all_orders,
        reviewed=reviewed,
        data_tables=data_tables,
        statistics=statistics,
        audit=audit,
        prestate=prestate,
        config=config,
    )
    preservation = verify_protected_unchanged(prestate)
    write_json(WORKING_DIR / "PRE_TEST_PRESERVATION.json", preservation)
    write_json(WORKING_DIR / "ANALYSIS_RECEIPTS.json", receipts)
    _append_event(
        "analysis",
        "passed",
        {
            "canonical_orders": len(all_orders),
            "reviewed_orders": len(reviewed),
            "decision_label": statistics["decision"]["label"],
            "artifact_count": len(receipts),
            "protected_preservation_passed": preservation["passed"],
        },
    )


def run_finalize(test_results: Path) -> None:
    _require_prestate()
    if not test_results.is_file():
        raise FileNotFoundError(test_results)
    _append_event("finalize", "started", {"test_results": str(test_results.resolve())})
    state = _load_json(RUN_STATE_PATH)
    commands = [event["command"] for event in state.get("events", [])]
    result = finalize_manifest(test_results_path=test_results, commands=commands)
    _append_event(
        "finalize",
        "passed" if result["overall_pass"] else "blocked",
        result,
    )
    if not result["overall_pass"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preflight", "analysis", "finalize"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--test-results", type=Path)
    arguments = parser.parse_args()
    if arguments.stage in {"preflight", "analysis"} and arguments.data_dir is None:
        parser.error("--data-dir is required for preflight and analysis")
    if arguments.stage == "finalize" and arguments.test_results is None:
        parser.error("--test-results is required for finalize")
    if arguments.stage == "preflight":
        run_preflight(arguments.data_dir)
    elif arguments.stage == "analysis":
        run_analysis(arguments.data_dir)
    else:
        run_finalize(arguments.test_results)


if __name__ == "__main__":
    main()
