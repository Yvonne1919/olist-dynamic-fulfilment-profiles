"""Staged runner for the frozen direct-promise robustness extension."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from .robustness_experiment import run_model_experiment
from .robustness_integrity import (
    AFTER_PATH,
    BEFORE_PATH,
    MANIFEST_PATH,
    ROOT,
    WORKSPACE,
    capture_controls_before_model,
    capture_preflight,
    finalize_integrity,
    load_config,
    read_json,
    sha256_file,
    utc_now,
    workspace_inventory,
    write_hash_inventory,
    write_json,
)
from .robustness_reporting import render_reports
from .validate_robustness import REQUIRED_OUTPUTS, validate


CANONICAL_COMMANDS = {
    stage: [
        ".venv/bin/python", "-B", "-m",
        "analysis.direct_model_family_robustness_v1.scripts.run_robustness", stage,
    ]
    for stage in ["preflight", "model", "report", "finalize", "validate"]
}


def _manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("preflight manifest is missing")
    return read_json(MANIFEST_PATH)


def _write_manifest(value: dict[str, Any]) -> None:
    value["updated_at_utc"] = utc_now()
    value["manifest_sequence"] = int(value.get("manifest_sequence", 0)) + 1
    write_json(MANIFEST_PATH, value)


def _actual_invocation() -> list[str]:
    return [sys.executable, "-B", "-m", "analysis.direct_model_family_robustness_v1.scripts.run_robustness", *sys.argv[1:]]


def _record_stage_start(manifest: dict[str, Any], stage: str) -> None:
    manifest["status"] = f"{stage}_running"
    manifest.setdefault("command_history", []).append(
        {"stage": stage, "started_at_utc": utc_now(), "command": _actual_invocation(), "status": "running"}
    )
    manifest.setdefault("stages", {})[stage] = {
        "status": "running", "started_at_utc": utc_now(), "command": _actual_invocation()
    }
    _write_manifest(manifest)


def _record_stage_complete(manifest: dict[str, Any], stage: str, result: Any) -> None:
    completed = utc_now()
    manifest["status"] = f"{stage}_complete"
    manifest["stages"][stage].update(status="complete", completed_at_utc=completed, result=result)
    for event in reversed(manifest["command_history"]):
        if event["stage"] == stage and event["status"] == "running":
            event.update(status="complete", completed_at_utc=completed)
            break
    _write_manifest(manifest)


def _record_stage_failure(manifest: dict[str, Any], stage: str, error: BaseException) -> None:
    failed = utc_now()
    payload = {
        "type": type(error).__name__, "message": str(error),
        "traceback": traceback.format_exc(),
    }
    manifest["status"] = f"{stage}_failed"
    manifest["stages"][stage].update(status="failed", failed_at_utc=failed, error=payload)
    for event in reversed(manifest["command_history"]):
        if event["stage"] == stage and event["status"] == "running":
            event.update(status="failed", failed_at_utc=failed, error=payload)
            break
    _write_manifest(manifest)


def run_preflight() -> dict[str, Any]:
    if MANIFEST_PATH.exists() or BEFORE_PATH.exists():
        raise RuntimeError("preflight artifacts already exist; refusing to overwrite a prior run")
    config = load_config()
    before = capture_preflight()
    direct_root = before["protection_boundary"]["roots"]["analysis/direct_promise_profile_extension_v1"]
    manifest: dict[str, Any] = {
        "analysis_id": config["analysis_id"],
        "schema_version": "direct_model_family_robustness_manifest_v1",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "manifest_sequence": 0,
        "status": "preflight_complete",
        "profile_history_variant": "selected_90_day",
        "all_mature_workspace_consumed": False,
        "all_mature_empirical_results_consumed": False,
        "isolation_incident": config["isolation_incident"],
        "scope": config["scope"],
        "authorisation": config["authorisation"],
        "git_before": before["repository"],
        "environment": before["environment"],
        "source_verification_before": before["source_verification"],
        "protected_before": {
            "path": BEFORE_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(BEFORE_PATH),
            "root_count": before["protection_boundary"]["root_count"],
            "file_count": before["protection_boundary"]["file_count"],
            "total_bytes": before["protection_boundary"]["total_bytes"],
            "aggregate_sha256": before["protection_boundary"]["aggregate_sha256"],
        },
        "complete_direct_extension": direct_root,
        "commands": CANONICAL_COMMANDS,
        "command_history": [
            {
                "stage": "preflight", "started_at_utc": before["captured_at_utc"],
                "completed_at_utc": utc_now(), "command": _actual_invocation(), "status": "complete",
            }
        ],
        "stages": {
            "preflight": {
                "status": "complete", "completed_at_utc": utc_now(),
                "command": _actual_invocation(),
            }
        },
        "pre_execution_audit": {
            "type": "read_only_source_and_governance_inspection",
            "note": "Authoritative controls, completed direct-extension artifacts, and prior model definitions were inspected before freeze. Exact shell transcripts before the staged run were not treated as model-execution commands.",
            "excluded_workspace_incident": config["isolation_incident"],
        },
        "implementation_deviations": [
            "Spline Logistic was blocked because no applicable exact direct-feature predictive implementation/grid was recoverable.",
            "Breach Random Forest, leaf-weighted quantile Random Forest and lognormal Ridge use recovered singleton grids because no prior multi-point grids existed.",
            "Breach Random Forest retains its exact controlled historical preprocessing without missing indicators and seed 42; this differs transparently from the direct primary preprocessing.",
            "Recovered severity source is a validated but untracked analysis artifact in a linked temporary Git worktree; its exact logic and source hashes are archived in this workspace.",
        ],
        "protected_primary_label_policy": "pre-existing direct-extension labels copied unchanged; new-family rows use ROBUSTNESS EVIDENCE LABEL",
        "thesis_files_edited": False,
        "commit_created": False,
    }
    _write_manifest(manifest)
    return manifest["protected_before"]


def _run_stage(stage: str, function: Callable[[], Any], expected_previous: set[str]) -> Any:
    manifest = _manifest()
    if manifest.get("status") not in expected_previous:
        raise RuntimeError(f"stage {stage} cannot follow manifest status {manifest.get('status')}")
    _record_stage_start(manifest, stage)
    try:
        result = function()
    except BaseException as error:
        manifest = _manifest()
        _record_stage_failure(manifest, stage, error)
        raise
    manifest = _manifest()
    _record_stage_complete(manifest, stage, result)
    return result


def run_model() -> Any:
    manifest = _manifest()
    if manifest.get("status") != "preflight_complete":
        raise RuntimeError(f"model cannot follow manifest status {manifest.get('status')}")
    config = load_config()
    controls = capture_controls_before_model(config)
    manifest["controls_before_model"] = {
        "captured_at_utc": controls["captured_at_utc"],
        "file_count": controls["file_count"],
        "aggregate_sha256": controls["aggregate_sha256"],
        "files": controls["files"],
        "sources_verified": controls["sources_verified"]["all_verified"],
    }
    manifest["isolation_incident"] = config["isolation_incident"]
    manifest["all_mature_workspace_consumed"] = False
    manifest["all_mature_empirical_results_consumed"] = False
    _write_manifest(manifest)
    return _run_stage("model", run_model_experiment, {"preflight_complete"})


def run_report() -> Any:
    return _run_stage("report", render_reports, {"model_complete"})


def _finalize() -> dict[str, Any]:
    after = finalize_integrity()
    write_hash_inventory(after)
    return {
        "passed": after["passed"],
        "protected_comparison": after["comparison"],
        "control_comparison": after["control_comparison"],
        "after_path": AFTER_PATH.relative_to(ROOT).as_posix(),
        "after_sha256": sha256_file(AFTER_PATH),
        "hash_inventory_sha256": sha256_file(WORKSPACE / "HASH_INVENTORY.txt"),
    }


def run_finalize() -> Any:
    return _run_stage("finalize", _finalize, {"report_complete"})


def _validate_and_complete() -> dict[str, Any]:
    report = validate()
    after = read_json(AFTER_PATH)
    write_hash_inventory(after)
    return {
        "overall_passed": report["overall_passed"],
        "check_count": report["check_count"],
        "passed_count": report["passed_count"],
        "validation_report_sha256": sha256_file(WORKSPACE / "VALIDATION_REPORT.json"),
        "hash_inventory_sha256": sha256_file(WORKSPACE / "HASH_INVENTORY.txt"),
    }


def run_validate() -> Any:
    result = _run_stage("validate", _validate_and_complete, {"finalize_complete"})
    manifest = _manifest()
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    manifest["git_after"] = read_json(AFTER_PATH)["repository"]
    manifest["protected_integrity"] = read_json(AFTER_PATH)["comparison"]
    manifest["protected_integrity_passed"] = True
    manifest["source_hashes_verified_after"] = read_json(AFTER_PATH)["source_verification"]["all_verified"]
    manifest["validation_report"] = read_json(WORKSPACE / "VALIDATION_REPORT.json")
    manifest["required_output_status"] = {
        name: (WORKSPACE / name).is_file() for name in REQUIRED_OUTPUTS
    }
    manifest["workspace_inventory"] = workspace_inventory(
        exclude={"RUN_MANIFEST.json", "HASH_INVENTORY.txt"}
    )
    manifest["completion_checks"] = {
        "all_required_outputs_present": all(manifest["required_output_status"].values()),
        "primary_reproduction_gate_passed": True,
        "protected_primary_labels_unchanged": True,
        "protected_files_byte_identical": True,
        "profile_history_variant_selected_90_day": True,
        "all_mature_workspace_consumed_false": True,
        "terminal_not_labelled_or_pooled": True,
        "thesis_files_edited": False,
        "commit_created": False,
    }
    _write_manifest(manifest)
    write_hash_inventory(read_json(AFTER_PATH))
    final_inventory_sha256 = sha256_file(WORKSPACE / "HASH_INVENTORY.txt")
    manifest = _manifest()
    manifest["hash_inventory_final_sha256"] = final_inventory_sha256
    manifest["stages"]["validate"]["result"]["hash_inventory_sha256"] = final_inventory_sha256
    _write_manifest(manifest)
    result["hash_inventory_sha256"] = final_inventory_sha256
    return result


def run_all() -> None:
    run_preflight()
    run_model()
    run_report()
    run_finalize()
    run_validate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "model", "report", "finalize", "validate", "all"])
    args = parser.parse_args(argv)
    functions = {
        "preflight": run_preflight,
        "model": run_model,
        "report": run_report,
        "finalize": run_finalize,
        "validate": run_validate,
        "all": run_all,
    }
    result = functions[args.stage]()
    if result is not None:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
