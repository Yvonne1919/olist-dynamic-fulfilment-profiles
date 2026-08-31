from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/all_mature_history_sensitivity_v1"
WORK = OUT / "working"
CONFIG_PATH = OUT / "SENSITIVITY_FROZEN_CONFIG.json"
RAW_DIR = Path("data/olist_data")
EXTERNAL_CHARTER = Path("docs/omitted-private-controls/OLIST_PROFILE_PIVOT_PROJECT_CHARTER_2026-08-21.md")
CONCURRENT_EXCLUDED_ANALYSIS_DIRS = {"direct_model_family_robustness_v1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def repository_state() -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip(),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _protected_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    analysis = ROOT / "analysis"
    for child in sorted(analysis.iterdir(), key=lambda item: item.name):
        if (
            child.resolve() == OUT.resolve()
            or child.name in CONCURRENT_EXCLUDED_ANALYSIS_DIRS
        ):
            continue
        roots.append((f"analysis/{child.name}", child))
    for name in ("docs", "report", "results", "src", "exploratory"):
        path = ROOT / name
        if path.exists():
            roots.append((name, path))
    for name in ("AGENTS.md", "PROJECT_CONTEXT.md", "RESULTS_REGISTRY.md", "DECISION_LOG.md"):
        path = ROOT / name
        if path.exists():
            roots.append((name, path))
    if RAW_DIR.exists():
        roots.append(("external_raw_olist", RAW_DIR))
    if EXTERNAL_CHARTER.exists():
        roots.append(("external_project_charter", EXTERNAL_CHARTER))
    return roots


def _leaf_rows(root_name: str, path: Path) -> Iterable[dict[str, object]]:
    if path.is_file():
        candidates = [path]
        base = path.parent
    else:
        candidates = sorted(
            (item for item in path.rglob("*") if item.is_file() or item.is_symlink()),
            key=lambda item: item.relative_to(path).as_posix(),
        )
        base = path
    for item in candidates:
        relative = item.name if path.is_file() else item.relative_to(base).as_posix()
        if item.is_symlink():
            target = os.readlink(item)
            payload = target.encode("utf-8")
            yield {
                "protected_root": root_name,
                "relative_path": relative,
                "kind": "symlink",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        else:
            yield {
                "protected_root": root_name,
                "relative_path": relative,
                "kind": "file",
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }


def capture_protected_inventory(label: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    for root_name, path in _protected_roots():
        print(f"hashing protected root: {root_name}", flush=True)
        rows = list(_leaf_rows(root_name, path))
        all_rows.extend(rows)
        digest = hashlib.sha256()
        for row in rows:
            digest.update(
                (
                    f"{row['relative_path']}\0{row['kind']}\0{row['bytes']}\0"
                    f"{row['sha256']}\n"
                ).encode("utf-8")
            )
        root_rows.append(
            {
                "protected_root": root_name,
                "path": str(path),
                "leaf_count": len(rows),
                "total_bytes": sum(int(row["bytes"]) for row in rows),
                "aggregate_sha256": digest.hexdigest(),
            }
        )
    _write_csv(WORK / f"PROTECTED_HASHES_{label}.csv", all_rows)
    _write_csv(WORK / f"PROTECTED_ROOT_DIGESTS_{label}.csv", root_rows)
    return all_rows, root_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["protected_root", "relative_path", "kind", "bytes", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_inventory(path: Path) -> dict[tuple[str, str], tuple[str, int, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (str(row["protected_root"]), str(row["relative_path"])):
            (str(row["kind"]), int(row["bytes"]), str(row["sha256"]))
            for row in rows
        }


def verify_frozen_dependencies() -> dict[str, dict[str, object]]:
    cfg = json_read(CONFIG_PATH)
    profile = ROOT / cfg["profile_validation_dir"]
    expected = cfg["frozen_dependencies"]
    paths = {
        "profile_core_sha256": profile / "scripts/profile_core.py",
        "profile_reporting_sha256": profile / "scripts/profile_reporting.py",
        "profile_protocol_sha256": profile / "PROFILE_PROTOCOL.md",
        "profile_frozen_config_sha256": profile / "PROFILE_FROZEN_CONFIG.json",
        "profile_selection_freeze_sha256": profile / "PROFILE_SELECTION_FREEZE.json",
        "profile_selected_candidates_sha256": profile / "PROFILE_SELECTED_CANDIDATES.csv",
        "profile_daily_scores_sha256": profile / "PROFILE_DAILY_SCORES.csv.gz",
        "anchor_schedule_sha256": profile / "ANCHOR_SCHEDULE.csv",
        "profile_run_manifest_sha256": profile / "RUN_MANIFEST.json",
    }
    receipt: dict[str, dict[str, object]] = {}
    for key, path in paths.items():
        actual = sha256_file(path)
        ok = actual == expected[key]
        receipt[key] = {
            "path": path.relative_to(ROOT).as_posix(),
            "expected_sha256": expected[key],
            "actual_sha256": actual,
            "verified": ok,
        }
        if not ok:
            raise RuntimeError(f"frozen dependency mismatch: {key}: {actual}")
    assembler = ROOT / cfg["canonical_assembler"]["path"]
    actual = sha256_file(assembler)
    if actual != cfg["canonical_assembler"]["sha256"]:
        raise RuntimeError(f"canonical assembler mismatch: {actual}")
    receipt["canonical_assembler"] = {
        "path": assembler.relative_to(ROOT).as_posix(),
        "expected_sha256": cfg["canonical_assembler"]["sha256"],
        "actual_sha256": actual,
        "verified": True,
    }
    return receipt


def verify_raw_hashes() -> dict[str, str]:
    from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp

    expected = json_read(CONFIG_PATH)["raw_file_hashes"]
    actual = dp.raw_file_sha256s(RAW_DIR)
    if actual != expected:
        raise RuntimeError(f"raw-data hash mismatch: {actual}")
    return actual


def environment_receipt() -> dict:
    distributions: dict[str, list[str]] = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "unknown").strip()
        distributions.setdefault(name, []).append(dist.version)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "distributions": {key: sorted(values) for key, values in sorted(distributions.items())},
    }


def direct_extension_gate() -> dict:
    direct = ROOT / "analysis/direct_promise_profile_extension_v1"
    manifest_path = direct / "RUN_MANIFEST.json"
    required = [
        direct / "DIRECT_MODEL_SELECTION_FREEZE.json",
        direct / "VALIDATION_REPORT.json",
        direct / "HASH_INVENTORY.txt",
        direct / "working/INTEGRITY_AFTER.json",
    ]
    manifest = json_read(manifest_path) if manifest_path.exists() else {}
    validation_path = direct / "VALIDATION_REPORT.json"
    validation = json_read(validation_path) if validation_path.exists() else {}
    validation_ok = validation.get("status") == "PASS"
    embedded_validation = manifest.get("validation_report")
    embedded_validation_ok = (
        isinstance(embedded_validation, dict)
        and embedded_validation.get("status") == "PASS"
    )
    required_output_status = manifest.get("required_output_status")
    required_output_status_ok = (
        isinstance(required_output_status, dict)
        and required_output_status.get("all_present") is True
    )
    checks = manifest.get("completion_checks")
    checks_ok = isinstance(checks, dict) and bool(checks) and all(bool(value) for value in checks.values())
    integrity = manifest.get("integrity_verdict")
    integrity_ok = (
        integrity == "PASS"
        or (
            isinstance(integrity, dict)
            and (
                integrity.get("passed") is True
                or integrity.get("status") == "PASS"
                or integrity.get("verdict") == "PASS"
            )
        )
    )
    status = manifest.get("status")
    available = (
        status == "complete"
        and checks_ok
        and integrity_ok
        and validation_ok
        and embedded_validation_ok
        and required_output_status_ok
        and all(path.is_file() for path in required)
    )
    return {
        "evaluated_at_utc": utc_now(),
        "available": available,
        "decision": "run_order_level_branch" if available else "skip_order_level_branch",
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_exists": manifest_path.exists(),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "manifest_status": status,
        "manifest_sequence": manifest.get("manifest_sequence"),
        "completion_checks": checks,
        "completion_checks_all_true": checks_ok,
        "integrity_verdict": integrity,
        "integrity_ok": integrity_ok,
        "validation_report_status": validation.get("status"),
        "validation_ok": validation_ok,
        "embedded_validation_report_status": (
            embedded_validation.get("status")
            if isinstance(embedded_validation, dict) else None
        ),
        "embedded_validation_ok": embedded_validation_ok,
        "required_output_status_all_present": (
            required_output_status.get("all_present")
            if isinstance(required_output_status, dict) else None
        ),
        "required_output_status_ok": required_output_status_ok,
        "required_artifacts": {
            path.relative_to(ROOT).as_posix(): path.is_file() for path in required
        },
        "reason": (
            "completed and integrity-validated direct extension available"
            if available else
            "direct extension was not complete and integrity-validated at the frozen preflight gate"
        ),
    }


def sensitivity_control_inventory() -> dict[str, dict[str, object]]:
    paths = sorted((OUT / "scripts").glob("*.py")) + [
        OUT / "SENSITIVITY_FROZEN_CONFIG.json",
        OUT / "EXACT_HISTORY_DEFINITIONS.md",
    ]
    return {
        path.relative_to(OUT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    }


def preflight() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    dependencies = verify_frozen_dependencies()
    raw = verify_raw_hashes()
    gate = direct_extension_gate()
    json_write(WORK / "DIRECT_EXTENSION_GATE.json", gate)
    rows, root_rows = capture_protected_inventory("BEFORE")
    state = {
        "stage": "preflight_complete",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "repository": repository_state(),
        "frozen_dependencies": dependencies,
        "raw_file_hashes": raw,
        "environment": environment_receipt(),
        "direct_extension_gate": gate,
        "concurrent_workspace_exclusions": [
            {
                "path": f"analysis/{name}",
                "reason": (
                    "created concurrently after this sensitivity began; not prior evidence "
                    "and not an input to this analysis"
                ),
            }
            for name in sorted(CONCURRENT_EXCLUDED_ANALYSIS_DIRS)
        ],
        "sensitivity_controls_before": sensitivity_control_inventory(),
        "protected_leaf_count": len(rows),
        "protected_root_count": len(root_rows),
        "protected_total_bytes": sum(int(row["bytes"]) for row in rows),
    }
    json_write(WORK / "PRE_EXECUTION_STATE.json", state)
    return state


def finalize_integrity() -> dict:
    before_path = WORK / "PROTECTED_HASHES_BEFORE.csv"
    if not before_path.exists():
        raise FileNotFoundError("preflight protected inventory is missing")
    before = _read_inventory(before_path)
    _, _ = capture_protected_inventory("AFTER")
    after = _read_inventory(WORK / "PROTECTED_HASHES_AFTER.csv")
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(key for key in before_keys & after_keys if before[key] != after[key])
    pre = json_read(WORK / "PRE_EXECUTION_STATE.json")
    repo_after = repository_state()
    branch_commit_unchanged = (
        repo_after["branch"] == pre["repository"]["branch"]
        and repo_after["commit"] == pre["repository"]["commit"]
    )
    controls_before = pre.get("sensitivity_controls_before")
    controls_after = sensitivity_control_inventory()
    controls_unchanged = controls_before == controls_after
    passed = (
        not added and not removed and not changed
        and branch_commit_unchanged and controls_unchanged
    )
    receipt = {
        "verified_at_utc": utc_now(),
        "status": "PASS" if passed else "FAIL",
        "protected_files_byte_identical": not added and not removed and not changed,
        "branch_commit_unchanged": branch_commit_unchanged,
        "sensitivity_controls_unchanged": controls_unchanged,
        "sensitivity_controls_before": controls_before,
        "sensitivity_controls_after": controls_after,
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added": [f"{root}/{path}" for root, path in added],
        "removed": [f"{root}/{path}" for root, path in removed],
        "changed": [f"{root}/{path}" for root, path in changed],
        "repository_before": pre["repository"],
        "repository_after": repo_after,
    }
    json_write(WORK / "PROTECTED_INTEGRITY_VERDICT.json", receipt)
    if not passed:
        raise RuntimeError(f"protected integrity verification failed: {receipt}")
    return receipt
