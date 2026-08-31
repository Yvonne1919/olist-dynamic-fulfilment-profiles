#!/usr/bin/env python3
"""Isolated stage runner for the direct promise/profile extension.

The runner deliberately contains no modelling or reporting logic.  It orders the
frozen extension APIs, persists one self-excluding run manifest, and refuses to
advance when a predecessor receipt or stage is missing.  Run it with ``python
-B``; bytecode writes are also disabled here before project modules are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "analysis/direct_promise_profile_extension_v1"
SCRIPTS = WORKSPACE / "scripts"
CONFIG_PATH = WORKSPACE / "DIRECT_FROZEN_CONFIG.json"
MANIFEST_PATH = WORKSPACE / "RUN_MANIFEST.json"
BEFORE_PATH = WORKSPACE / "working/INTEGRITY_BEFORE.json"
AFTER_PATH = WORKSPACE / "working/INTEGRITY_AFTER.json"
INVENTORY_PATH = WORKSPACE / "HASH_INVENTORY.txt"
VALIDATION_REPORT_PATH = WORKSPACE / "VALIDATION_REPORT.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.direct_promise_profile_extension_v1.scripts import direct_integrity


ANALYSIS_ID = "direct_promise_profile_extension_v1"
MANIFEST_SCHEMA = "direct_extension_run_manifest_v1"
STAGE_ORDER = ("preflight", "model", "report", "finalize", "validate")
PREDECESSOR_STATUS = {
    "model": "preflight_complete",
    "report": "model_complete",
    "finalize": "report_complete",
    "validate": "finalize_complete",
}

RUNNER_RELATIVE = (
    "analysis/direct_promise_profile_extension_v1/scripts/run_direct_extension.py"
)
SCRIPT_RELATIVES = {
    "runner": RUNNER_RELATIVE,
    "integrity": (
        "analysis/direct_promise_profile_extension_v1/scripts/direct_integrity.py"
    ),
    "experiment": (
        "analysis/direct_promise_profile_extension_v1/scripts/direct_experiment.py"
    ),
    "reporting": (
        "analysis/direct_promise_profile_extension_v1/scripts/direct_reporting.py"
    ),
    "validation": (
        "analysis/direct_promise_profile_extension_v1/scripts/"
        "validate_direct_extension.py"
    ),
}

STAGE_SCRIPTS = {
    "preflight": ("runner", "integrity"),
    "model": ("runner", "integrity", "experiment"),
    "report": ("runner", "reporting"),
    "finalize": ("runner", "integrity"),
    "validate": ("runner", "validation"),
}

STAGE_CALLABLES = {
    "preflight": (
        "direct_integrity.capture_preflight",
    ),
    "model": (
        "direct_integrity.verify_configured_sources",
        "direct_experiment.run_and_write_modeling",
    ),
    "report": (
        "direct_reporting.write_reporting_outputs",
    ),
    "finalize": (
        "direct_integrity.finalize_integrity",
    ),
    "validate": (
        "validate_direct_extension.validate",
    ),
}

IMPLEMENTATION_DEVIATIONS = (
    {
        "deviation_id": "NO_SEPARATE_ORDER_V1_AMENDMENT",
        "status": "recorded_non_substitutive_deviation",
        "description": (
            "No separate Order V1 amendment exists in the repository. The "
            "protected Order V1 authority is ORDER_PROTOCOL.md, "
            "ORDER_FROZEN_CONFIG.json, and ORDER_MODEL_SELECTION_FREEZE.json; "
            "the unrelated Phase 2A amendment is not substituted."
        ),
    },
    {
        "deviation_id": "REPORTING_RETRY_AFTER_STRING_ACCESSOR_ERROR",
        "status": "corrected_before_accepted_preflight_and_rerun",
        "description": (
            "The first reporting attempt failed before summary generation because "
            "two pandas string-accessor expressions used .str.eq. The reporting-only "
            "expressions were corrected to Series.eq, the failure was receipted in "
            "RETRY_LOG.md, and the complete preflight/model/report sequence was rerun "
            "so the corrected script was frozen before the accepted execution."
        ),
    },
    {
        "deviation_id": "PINNED_CHARTER_MIRROR_RESTORED",
        "status": "restored_byte_identically_before_accepted_preflight",
        "description": (
            "A model-stage source gate found the pinned repository charter mirror "
            "missing and halted before fitting. The registered external charter "
            "was verified and the mirror restored at the identical pinned SHA-256; "
            "the accepted run starts from a new preflight receipt."
        ),
    },
    {
        "deviation_id": "CONCURRENT_SENSITIVITY_WORKSPACE_EXCLUDED_FROM_BOUNDARY",
        "status": "recorded_concurrency_exclusion",
        "description": (
            "A separate active task created and continued writing "
            "analysis/all_mature_history_sensitivity_v1 after this extension began. "
            "That new non-input workspace is excluded from the aggregate; every "
            "prior analysis workspace and all other protected artifacts remain "
            "under strict before/after byte-hash comparison."
        ),
    },
    {
        "deviation_id": "VALIDATOR_RETRY_AFTER_GUARD_NAMESPACE_COLLISION",
        "status": "corrected_before_accepted_preflight_and_rerun",
        "description": (
            "The independent validator initially selected an empty severity-only "
            "high_support_guard field on breach rows before the populated breach-"
            "specific guard. It halted completion, was corrected to require a "
            "non-missing task-specific guard, and the full frozen sequence was rerun."
        ),
    },
    {
        "deviation_id": "SUMMARY_RETRY_AFTER_GUARD_NAMESPACE_COLLISION",
        "status": "corrected_before_accepted_preflight_and_rerun",
        "description": (
            "The generated summary initially displayed blank breach high-support "
            "guards because a severity-only generic field masked the populated "
            "breach-specific guard in the wide evidence table. Reporting now "
            "coalesces task-specific guards without changing persisted labels or metrics."
        ),
    },
)

# Mirrored only so a failed validation can still report the required-file state.
# On a successful validation run, the validator's public REQUIRED_OUTPUTS tuple is
# authoritative and replaces this fallback.
REQUIRED_OUTPUTS_FALLBACK = (
    "RUN_MANIFEST.json",
    "README.md",
    "EXACT_FEATURE_MANIFEST.md",
    "MODEL_SELECTION.csv",
    "DIRECT_BREACH_MONTHLY.csv",
    "DIRECT_BREACH_POOLED.csv",
    "DIRECT_BREACH_CALIBRATION.csv",
    "DIRECT_BREACH_SUPPORT_STRATA.csv",
    "DIRECT_SEVERITY_MONTHLY.csv",
    "DIRECT_SEVERITY_POOLED.csv",
    "DIRECT_SEVERITY_COVERAGE.csv",
    "DIRECT_TERMINAL.csv",
    "DIRECT_VS_CURRENT_CONTEXT_ROBUSTNESS.csv",
    "RESULT_SUMMARY.md",
    "RESULT_SUMMARY_ZH.md",
    "FIGURE_DATA.csv",
    "HASH_INVENTORY.txt",
    "DIRECT_MODEL_MANIFESTS.csv",
    "EVIDENCE_LABELS.csv",
)


class StageError(RuntimeError):
    """A stage gate or returned API contract failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _extension_relative(path: Path) -> str:
    try:
        return path.absolute().relative_to(WORKSPACE.absolute()).as_posix()
    except ValueError as exc:
        raise StageError(f"path is outside the extension workspace: {path}") from exc


def _jsonable(value: object) -> object:
    """Convert API results to strict, deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return _repo_relative(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=lambda item: _compact_json(item))
        return items
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _jsonable(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _compact_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_stable_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StageError(f"required JSON file is missing: {_repo_relative(path)}") from exc
    except json.JSONDecodeError as exc:
        raise StageError(f"invalid JSON in {_repo_relative(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError(f"JSON root must be an object: {_repo_relative(path)}")
    return value


def _sha256_file(path: Path) -> tuple[str, int]:
    before = path.stat()
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if signature_before != signature_after:
        raise StageError(f"file changed while being inventoried: {_repo_relative(path)}")
    return digest.hexdigest(), int(after.st_size)


def _file_record(path: Path) -> dict[str, object]:
    if path.is_symlink():
        before = path.lstat()
        target = os.readlink(path)
        payload = b"symlink\0" + os.fsencode(target)
        after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StageError(
                f"symlink changed while being inventoried: {_repo_relative(path)}"
            )
        return {
            "kind": "symlink",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "link_target": target,
            "repo_relative_path": _repo_relative(path),
        }
    if not path.is_file():
        raise StageError(f"expected regular file: {_repo_relative(path)}")
    digest, size = _sha256_file(path)
    return {
        "kind": "file",
        "bytes": size,
        "sha256": digest,
        "repo_relative_path": _repo_relative(path),
    }


def _extension_file_inventory() -> dict[str, dict[str, object]]:
    """Hash every extension file except RUN_MANIFEST.json itself."""

    if not WORKSPACE.is_dir():
        raise StageError(f"extension workspace is missing: {WORKSPACE}")

    leaves: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise StageError(f"could not traverse extension workspace: {error}") from error

    for directory, dirnames, filenames in os.walk(
        WORKSPACE,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        retained: list[str] = []
        for name in dirnames:
            candidate = directory_path / name
            if candidate.is_symlink():
                leaves.append(candidate)
            else:
                retained.append(name)
        dirnames[:] = retained
        leaves.extend(directory_path / name for name in filenames)

    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(leaves, key=_extension_relative):
        if path.absolute() == MANIFEST_PATH.absolute():
            continue
        relative = _extension_relative(path)
        inventory[relative] = _file_record(path)
    return inventory


def _inventory_sha256(inventory: Mapping[str, object]) -> str:
    return hashlib.sha256(_compact_json(inventory).encode("utf-8")).hexdigest()


def _required_output_status(
    required_names: Sequence[str],
    inventory: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for name in sorted(set(map(str, required_names)) | {"VALIDATION_REPORT.json"}):
        if name == MANIFEST_PATH.name:
            exists = MANIFEST_PATH.is_file()
            files[name] = {
                "exists": exists,
                "sha256": None,
                "bytes": None,
                "reason": "self_referential_manifest_excluded_from_own_inventory",
            }
        else:
            row = inventory.get(name)
            exists = row is not None
            files[name] = {
                "exists": exists,
                "sha256": row.get("sha256") if row else None,
                "bytes": row.get("bytes") if row else None,
            }
        if not exists:
            missing.append(name)
    return {
        "required_count": len(files),
        "present_count": len(files) - len(missing),
        "missing": missing,
        "all_present": not missing,
        "files": files,
    }


def _write_manifest(
    manifest: dict[str, Any],
    *,
    required_names: Sequence[str] | None = None,
) -> None:
    inventory = _extension_file_inventory()
    manifest["extension_file_inventory"] = inventory
    manifest["extension_file_inventory_sha256"] = _inventory_sha256(inventory)
    manifest["extension_file_count"] = len(inventory)
    manifest["extension_file_total_bytes"] = sum(
        int(row["bytes"]) for row in inventory.values()
    )
    if required_names is not None:
        manifest["required_output_status"] = _required_output_status(
            required_names, inventory
        )
    manifest["updated_at_utc"] = _utc_now()
    manifest["manifest_sequence"] = int(manifest.get("manifest_sequence", 0)) + 1
    _atomic_write_json(MANIFEST_PATH, manifest)


def _canonical_commands() -> dict[str, list[str]]:
    return {
        stage: [sys.executable, "-B", RUNNER_RELATIVE, stage]
        for stage in STAGE_ORDER
    }


def _actual_invocation() -> list[str]:
    original = getattr(sys, "orig_argv", None)
    if isinstance(original, list) and original:
        return [str(value) for value in original]
    return [sys.executable, *map(str, sys.argv)]


def _script_records(stage: str) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for logical_name in STAGE_SCRIPTS[stage]:
        path = ROOT / SCRIPT_RELATIVES[logical_name]
        if not path.is_file() or path.is_symlink():
            raise StageError(
                f"required stage script is missing or not regular: {_repo_relative(path)}"
            )
        records[logical_name] = {
            "logical_name": logical_name,
            "path": _repo_relative(path),
            **_file_record(path),
        }
    return records


def _stage_event(
    stage: str,
    *,
    started_at: str,
    status: str,
    completed_at: str | None = None,
    error: Mapping[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "stage": stage,
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "actual_invocation": _actual_invocation(),
        "canonical_stage_command": _canonical_commands()[stage],
        "callables": list(STAGE_CALLABLES[stage]),
        "scripts": _script_records(stage),
    }
    if error is not None:
        event["error"] = _jsonable(error)
    return event


def _merge_executed_scripts(
    manifest: dict[str, Any], event: Mapping[str, object]
) -> None:
    executed = manifest.setdefault("executed_scripts", {})
    if not isinstance(executed, dict):
        raise StageError("manifest executed_scripts must be an object")
    scripts = event.get("scripts")
    if not isinstance(scripts, Mapping):
        raise StageError("stage event lacks script records")
    for row in scripts.values():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise StageError("invalid stage script record")
        executed[str(row["path"])] = _jsonable(row)


def _base_manifest(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "manifest_sequence": 0,
        "status": "preflight_running",
        "stage_order": list(STAGE_ORDER),
        "commands": {
            "actual_preflight_invocation": _actual_invocation(),
            "canonical_stage_commands": _canonical_commands(),
        },
        "command_history": [],
        "stages": {},
        "executed_scripts": {},
        "implementation_deviations": [dict(row) for row in IMPLEMENTATION_DEVIATIONS],
    }


def _repository_summary(repository: Mapping[str, object]) -> dict[str, object]:
    porcelain = str(repository.get("status_porcelain_v1", ""))
    full = str(repository.get("status_full", ""))
    return {
        "available": repository.get("available"),
        "branch": repository.get("branch"),
        "detached_head": repository.get("detached_head"),
        "commit": repository.get("commit"),
        "status_clean": repository.get("status_clean"),
        "status_porcelain_v1": porcelain,
        "status_porcelain_v1_sha256": hashlib.sha256(
            porcelain.encode("utf-8")
        ).hexdigest(),
        "status_full": full,
        "status_full_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
        "integrity_receipt": _repo_relative(BEFORE_PATH),
    }


def _environment_summary(environment: Mapping[str, object]) -> dict[str, object]:
    distributions = environment.get("installed_distributions")
    distribution_count = len(distributions) if isinstance(distributions, list) else None
    return {
        "python_version": environment.get("python_version"),
        "python_implementation": environment.get("python_implementation"),
        "python_compiler": environment.get("python_compiler"),
        "executable": environment.get("executable"),
        "platform": environment.get("platform"),
        "machine": environment.get("machine"),
        "primary_distributions": environment.get("primary_distributions"),
        "installed_distribution_count": distribution_count,
        "environment_fingerprint_sha256": environment.get(
            "environment_fingerprint_sha256"
        ),
        "complete_environment_in_receipt": _repo_relative(BEFORE_PATH),
    }


def _preflight_receipt_fields(receipt: Mapping[str, object]) -> dict[str, object]:
    sources = receipt.get("source_verification")
    controls = receipt.get("extension_controls")
    repository = receipt.get("repository")
    environment = receipt.get("environment")
    boundary = receipt.get("protection_boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (sources, controls, repository, environment, boundary)
    ):
        raise StageError("preflight receipt is missing required sections")
    source_entries = sources.get("entries")  # type: ignore[union-attr]
    control_files = controls.get("files")  # type: ignore[union-attr]
    if not isinstance(source_entries, Mapping) or not isinstance(control_files, Mapping):
        raise StageError("preflight receipt lacks exact source/control hashes")
    before_record = _file_record(BEFORE_PATH)
    return {
        "source_hashes": _jsonable(source_entries),
        "source_hashes_verified": bool(sources.get("all_verified")),  # type: ignore[union-attr]
        "control_hashes": _jsonable(control_files),
        "repository_summary": _repository_summary(repository),  # type: ignore[arg-type]
        "environment_summary": _environment_summary(environment),  # type: ignore[arg-type]
        "integrity_before": {
            "path": _repo_relative(BEFORE_PATH),
            **before_record,
            "protected_root_count": boundary.get("root_count"),  # type: ignore[union-attr]
            "protected_file_count": boundary.get("file_count"),  # type: ignore[union-attr]
            "protected_total_bytes": boundary.get("total_bytes"),  # type: ignore[union-attr]
            "protected_aggregate_sha256": boundary.get("aggregate_sha256"),  # type: ignore[union-attr]
        },
        "external_charter": _jsonable(receipt.get("external_charter")),
    }


def _error_record(error: BaseException) -> dict[str, object]:
    record: dict[str, object] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    detail = getattr(error, "detail", None)
    if detail:
        record["detail"] = _jsonable(detail)
    return record


def _load_manifest(expected_status: str | None = None) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise StageError("RUN_MANIFEST.json has an unsupported schema")
    if manifest.get("analysis_id") != ANALYSIS_ID:
        raise StageError("RUN_MANIFEST.json has the wrong analysis_id")
    if expected_status is not None and manifest.get("status") != expected_status:
        raise StageError(
            f"stage requires manifest status {expected_status!r}; found "
            f"{manifest.get('status')!r}"
        )
    return manifest


def _begin_stage(manifest: dict[str, Any], stage: str) -> str:
    started = _utc_now()
    event = _stage_event(stage, started_at=started, status="running")
    stages = manifest.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise StageError("manifest stages must be an object")
    stages[stage] = event
    manifest["status"] = f"{stage}_running"
    history = manifest.setdefault("command_history", [])
    if not isinstance(history, list):
        raise StageError("manifest command_history must be an array")
    history.append(
        {
            "stage": stage,
            "started_at_utc": started,
            "actual_invocation": _actual_invocation(),
            "canonical_stage_command": _canonical_commands()[stage],
        }
    )
    _merge_executed_scripts(manifest, event)
    _write_manifest(manifest)
    return started


def _complete_stage(
    manifest: dict[str, Any],
    stage: str,
    started: str,
    status: str,
    *,
    required_names: Sequence[str] | None = None,
) -> None:
    event = _stage_event(
        stage,
        started_at=started,
        completed_at=_utc_now(),
        status="complete",
    )
    stages = manifest.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise StageError("manifest stages must be an object")
    stages[stage] = event
    manifest["status"] = status
    _merge_executed_scripts(manifest, event)
    _write_manifest(manifest, required_names=required_names)


def _fail_stage(
    manifest: dict[str, Any],
    stage: str,
    started: str,
    error: BaseException,
    *,
    required_names: Sequence[str] | None = None,
) -> None:
    failure = _error_record(error)
    event = _stage_event(
        stage,
        started_at=started,
        completed_at=_utc_now(),
        status="failed",
        error=failure,
    )
    stages = manifest.setdefault("stages", {})
    if isinstance(stages, dict):
        stages[stage] = event
    manifest["status"] = f"{stage}_failed"
    manifest["last_error"] = failure
    _merge_executed_scripts(manifest, event)
    _write_manifest(manifest, required_names=required_names)


def _verify_before_receipt(manifest: Mapping[str, object]) -> dict[str, Any]:
    receipt = _read_json(BEFORE_PATH)
    if receipt.get("schema_version") != direct_integrity.SCHEMA_VERSION:
        raise StageError("INTEGRITY_BEFORE.json has an unsupported schema")
    if receipt.get("analysis_id") != ANALYSIS_ID or receipt.get("stage") != "before":
        raise StageError("INTEGRITY_BEFORE.json is not this analysis's before receipt")
    recorded = manifest.get("integrity_before")
    if not isinstance(recorded, Mapping):
        raise StageError("RUN_MANIFEST.json lacks the preflight receipt fingerprint")
    actual_sha, _ = _sha256_file(BEFORE_PATH)
    if actual_sha != recorded.get("sha256"):
        raise StageError("INTEGRITY_BEFORE.json changed after preflight")
    return receipt


def _path_result_records(paths: Mapping[str, object]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for logical_name, raw_path in sorted(paths.items(), key=lambda pair: str(pair[0])):
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = WORKSPACE / path
        if not path.is_file() or path.is_symlink():
            raise StageError(
                f"modelling API returned a missing/non-regular output: {logical_name}={path}"
            )
        records[str(logical_name)] = {
            "path": _extension_relative(path),
            **_file_record(path),
        }
    return records


def _table_result_summary(tables: Mapping[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name, table in sorted(tables.items(), key=lambda pair: str(pair[0])):
        shape = getattr(table, "shape", None)
        columns = getattr(table, "columns", None)
        summary[str(name)] = {
            "rows": int(shape[0]) if isinstance(shape, tuple) and len(shape) >= 1 else None,
            "columns": (
                [str(column) for column in columns]
                if columns is not None
                else None
            ),
        }
    return summary


def _report_output_records(hashes: Mapping[str, object]) -> dict[str, object]:
    records: dict[str, object] = {}
    for filename, reported_hash in sorted(hashes.items(), key=lambda pair: str(pair[0])):
        path = WORKSPACE / str(filename)
        if not path.is_file() or path.is_symlink():
            raise StageError(f"reporting API returned a missing output: {filename}")
        record = _file_record(path)
        if record["sha256"] != str(reported_hash):
            raise StageError(f"reporting API hash mismatch for {filename}")
        records[str(filename)] = {
            "path": _extension_relative(path),
            **record,
        }
    return records


def run_preflight() -> dict[str, Any]:
    """Verify frozen inputs/protected roots and create the initial manifest."""

    started = _utc_now()
    manifest = _base_manifest(started)
    try:
        receipt = direct_integrity.capture_preflight(
            config_path=CONFIG_PATH,
            before_path=BEFORE_PATH,
        )
        fields = _preflight_receipt_fields(receipt)
        manifest.update(fields)
        event = _stage_event(
            "preflight",
            started_at=started,
            completed_at=_utc_now(),
            status="complete",
        )
        manifest["stages"]["preflight"] = event
        manifest["command_history"].append(
            {
                "stage": "preflight",
                "started_at_utc": started,
                "completed_at_utc": event["completed_at_utc"],
                "actual_invocation": _actual_invocation(),
                "canonical_stage_command": _canonical_commands()["preflight"],
            }
        )
        _merge_executed_scripts(manifest, event)
        manifest["status"] = "preflight_complete"
        _write_manifest(manifest)
        return manifest
    except Exception as error:
        failure = _error_record(error)
        event = _stage_event(
            "preflight",
            started_at=started,
            completed_at=_utc_now(),
            status="failed",
            error=failure,
        )
        manifest["stages"]["preflight"] = event
        _merge_executed_scripts(manifest, event)
        manifest["status"] = "preflight_failed"
        manifest["last_error"] = failure
        _write_manifest(manifest)
        raise


def run_model() -> dict[str, Any]:
    """Verify the before receipt/sources, fit once, and persist selection metadata."""

    manifest = _load_manifest(PREDECESSOR_STATUS["model"])
    _verify_before_receipt(manifest)
    started = _begin_stage(manifest, "model")
    try:
        config = direct_integrity.load_config(CONFIG_PATH)
        source_verification = direct_integrity.verify_configured_sources(
            config,
            strict=True,
        )
        manifest["model_source_verification"] = _jsonable(source_verification)

        from analysis.direct_promise_profile_extension_v1.scripts import (
            direct_experiment,
        )

        runner = getattr(direct_experiment, "run_and_write_modeling", None)
        if not callable(runner):
            raise StageError("direct_experiment.run_and_write_modeling is unavailable")
        result = runner(
            config=config,
            output_dir=WORKSPACE,
            frame=None,
            selection=None,
        )
        if not isinstance(result, Mapping):
            raise StageError("modelling API must return a mapping")
        selection = result.get("selection")
        tables = result.get("tables")
        paths = result.get("paths")
        if not isinstance(selection, Mapping):
            raise StageError("modelling API result lacks selection mapping")
        if not isinstance(tables, Mapping):
            raise StageError("modelling API result lacks tables mapping")
        if not isinstance(paths, Mapping) or not paths:
            raise StageError("modelling API result lacks output paths")

        selection_jsonable = _jsonable(selection)
        manifest["selected_parameters"] = selection_jsonable
        manifest["selected_parameter_json"] = _compact_json(selection_jsonable)
        manifest["model_output_inventory"] = _path_result_records(paths)
        manifest["model_table_summary"] = _table_result_summary(tables)

        freeze = WORKSPACE / "DIRECT_MODEL_SELECTION_FREEZE.json"
        if not freeze.is_file() or freeze.is_symlink():
            raise StageError("model stage did not persist DIRECT_MODEL_SELECTION_FREEZE.json")
        manifest["model_selection_freeze"] = {
            "path": _extension_relative(freeze),
            "content": _jsonable(_read_json(freeze)),
            **_file_record(freeze),
        }
        _complete_stage(manifest, "model", started, "model_complete")
        return manifest
    except Exception as error:
        _fail_stage(manifest, "model", started, error)
        raise


def run_report() -> dict[str, Any]:
    """Create deterministic reporting outputs from persisted model tables."""

    manifest = _load_manifest(PREDECESSOR_STATUS["report"])
    started = _begin_stage(manifest, "report")
    try:
        from analysis.direct_promise_profile_extension_v1.scripts import direct_reporting

        hashes = direct_reporting.write_reporting_outputs(
            extension_dir=WORKSPACE,
            repo_root=ROOT,
        )
        if not isinstance(hashes, Mapping) or not hashes:
            raise StageError("reporting API returned no output hashes")
        manifest["reporting_output_inventory"] = _report_output_records(hashes)
        _complete_stage(manifest, "report", started, "report_complete")
        return manifest
    except Exception as error:
        _fail_stage(manifest, "report", started, error)
        raise


def run_finalize() -> dict[str, Any]:
    """Compare the protected before/after states and persist the hash inventory."""

    manifest = _load_manifest(PREDECESSOR_STATUS["finalize"])
    started = _begin_stage(manifest, "finalize")
    try:
        sections = direct_integrity.finalize_integrity(
            config_path=CONFIG_PATH,
            before_path=BEFORE_PATH,
            after_path=AFTER_PATH,
            inventory_path=INVENTORY_PATH,
            manifest_sections_path=None,
            strict=True,
        )
        if not isinstance(sections, Mapping):
            raise StageError("integrity finalizer returned no manifest sections")
        verdict = sections.get("overall_integrity_verdict")
        if not isinstance(verdict, Mapping) or verdict.get("passed") is not True:
            raise StageError("protected before/after integrity verdict did not pass")
        manifest["integrity_sections"] = _jsonable(sections)
        manifest["integrity_verdict"] = _jsonable(verdict)
        _complete_stage(manifest, "finalize", started, "finalize_complete")
        return manifest
    except Exception as error:
        if AFTER_PATH.is_file():
            try:
                after = _read_json(AFTER_PATH)
                manifest["integrity_verdict"] = _jsonable(
                    after.get("comparison_to_before")
                )
            except Exception:
                pass
        _fail_stage(manifest, "finalize", started, error)
        raise


def run_validate() -> dict[str, Any]:
    """Run independent persisted-output validation and close the manifest."""

    manifest = _load_manifest(PREDECESSOR_STATUS["validate"])
    started = _begin_stage(manifest, "validate")
    required_names: Sequence[str] = REQUIRED_OUTPUTS_FALLBACK
    try:
        from analysis.direct_promise_profile_extension_v1.scripts import (
            validate_direct_extension,
        )

        configured_required = getattr(
            validate_direct_extension, "REQUIRED_OUTPUTS", REQUIRED_OUTPUTS_FALLBACK
        )
        if isinstance(configured_required, Sequence) and not isinstance(
            configured_required, (str, bytes)
        ):
            required_names = tuple(map(str, configured_required))

        report = validate_direct_extension.validate()
        if not isinstance(report, Mapping) or report.get("status") != "PASS":
            raise StageError("independent validation did not return PASS")

        # The validator currently persists this report itself.  Replacing it with
        # the same stable representation makes the runner-owned final write atomic.
        _atomic_write_json(VALIDATION_REPORT_PATH, report)
        persisted_report = _read_json(VALIDATION_REPORT_PATH)
        if _compact_json(persisted_report) != _compact_json(report):
            raise StageError("persisted validation report differs from validator result")
        manifest["validation_report"] = _jsonable(report)

        event = _stage_event(
            "validate",
            started_at=started,
            completed_at=_utc_now(),
            status="complete",
        )
        manifest["stages"]["validate"] = event
        _merge_executed_scripts(manifest, event)
        manifest["status"] = "validation_complete"

        inventory = _extension_file_inventory()
        required = _required_output_status(required_names, inventory)
        stage_passed = all(
            isinstance(manifest.get("stages", {}).get(stage), Mapping)
            and manifest["stages"][stage].get("status") == "complete"
            for stage in STAGE_ORDER
        )
        integrity = manifest.get("integrity_verdict")
        integrity_passed = isinstance(integrity, Mapping) and integrity.get("passed") is True
        source_passed = bool(manifest.get("source_hashes_verified"))
        model_sources = manifest.get("model_source_verification")
        model_sources_passed = (
            isinstance(model_sources, Mapping)
            and model_sources.get("all_verified") is True
        )
        all_passed = bool(
            stage_passed
            and integrity_passed
            and source_passed
            and model_sources_passed
            and required["all_present"]
            and report.get("status") == "PASS"
        )
        manifest["completion_checks"] = {
            "all_stage_events_complete": stage_passed,
            "integrity_passed": integrity_passed,
            "preflight_sources_verified": source_passed,
            "model_sources_reverified": model_sources_passed,
            "all_required_outputs_present": required["all_present"],
            "validation_passed": report.get("status") == "PASS",
            "all_passed": all_passed,
        }
        if not all_passed:
            raise StageError("completion gates did not all pass")

        manifest["status"] = "complete"
        _write_manifest(manifest, required_names=required_names)
        return manifest
    except Exception as error:
        _fail_stage(
            manifest,
            "validate",
            started,
            error,
            required_names=required_names,
        )
        raise


def run_all() -> dict[str, Any]:
    """Execute every stage in its frozen strict order."""

    run_preflight()
    run_model()
    run_report()
    run_finalize()
    return run_validate()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen direct promise/profile extension by stage."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("preflight", help="capture source/protection baseline")
    subparsers.add_parser("model", help="fit and persist the frozen direct ladder")
    subparsers.add_parser("report", help="derive reports from persisted outputs")
    subparsers.add_parser("finalize", help="compare protected before/after state")
    subparsers.add_parser("validate", help="validate outputs and close the manifest")
    subparsers.add_parser("all", help="run all five stages in strict order")
    return parser


def _command_summary(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "analysis_id": manifest.get("analysis_id"),
        "status": manifest.get("status"),
        "manifest": _repo_relative(MANIFEST_PATH),
        "manifest_sequence": manifest.get("manifest_sequence"),
        "extension_file_count": manifest.get("extension_file_count"),
        "integrity_passed": (
            manifest.get("integrity_verdict", {}).get("passed")
            if isinstance(manifest.get("integrity_verdict"), Mapping)
            else None
        ),
        "validation_status": (
            manifest.get("validation_report", {}).get("status")
            if isinstance(manifest.get("validation_report"), Mapping)
            else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runners = {
        "preflight": run_preflight,
        "model": run_model,
        "report": run_report,
        "finalize": run_finalize,
        "validate": run_validate,
        "all": run_all,
    }
    try:
        manifest = runners[args.stage]()
    except Exception as error:
        print(
            _compact_json(
                {
                    "analysis_id": ANALYSIS_ID,
                    "stage": args.stage,
                    "status": "FAILED",
                    "error": _error_record(error),
                    "manifest": _repo_relative(MANIFEST_PATH),
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(_compact_json(_command_summary(manifest)))
    return 0


__all__ = [
    "MANIFEST_PATH",
    "StageError",
    "main",
    "run_all",
    "run_finalize",
    "run_model",
    "run_preflight",
    "run_report",
    "run_validate",
]


if __name__ == "__main__":
    raise SystemExit(main())
