"""Read-only provenance and byte-integrity controls for the robustness run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "analysis/direct_model_family_robustness_v1"
CONFIG_PATH = WORKSPACE / "ROBUSTNESS_FROZEN_CONFIG.json"
WORKING = WORKSPACE / "working"
BEFORE_PATH = WORKING / "INTEGRITY_BEFORE.json"
AFTER_PATH = WORKING / "INTEGRITY_AFTER.json"
MANIFEST_PATH = WORKSPACE / "RUN_MANIFEST.json"
CONTROL_BEFORE_PATH = WORKING / "CONTROL_HASHES_BEFORE_MODEL.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config.get("analysis_id") != "direct_model_family_robustness_v1":
        raise AssertionError("unexpected robustness analysis_id")
    scope = config.get("scope", {})
    if scope.get("profile_history_variant") != "selected_90_day":
        raise AssertionError("profile history variant is not selected_90_day")
    if scope.get("all_mature_workspace_consumed") is not False:
        raise AssertionError("excluded workspace consumption flag must be false")
    return config


def resolve_source(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    all_verified = True
    for name, pair in sorted(config["sources"].items()):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"invalid source tuple for {name}")
        path = resolve_source(str(pair[0]))
        expected = str(pair[1])
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        passed = bool(exists and actual == expected)
        all_verified = all_verified and passed
        entries[name] = {
            "path": str(path if path.is_absolute() and ROOT not in path.parents else path.relative_to(ROOT)),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes": path.stat().st_size if exists else None,
            "verified": passed,
        }
    return {"checked_at_utc": utc_now(), "all_verified": all_verified, "entries": entries}


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
        )
        return result.stdout.rstrip("\n")

    return {
        "captured_at_utc": utc_now(),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_tracked_only_porcelain_v1": run("status", "--porcelain=v1", "--untracked-files=no"),
        "untracked_files_enumerated": False,
        "status_scope": "tracked_paths_only_to_preserve_excluded_workspace_isolation",
    }


def environment_state() -> dict[str, Any]:
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "statsmodels",
        "patsy",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def protection_root_paths(config: Mapping[str, Any]) -> list[Path]:
    """Resolve an explicit allowlist without listing ``analysis/`` itself."""

    seed_path = ROOT / str(config["protection"]["allowlist_source"]).split("#", 1)[0]
    seed = read_json(seed_path)
    root_names = sorted(seed["protection_boundary"]["roots"].keys())
    root_names.extend(config["protection"]["additional_complete_roots"])
    target = str(config["protection"]["excluded_target"])
    excluded = str(config["protection"]["excluded_concurrent_workspace"])
    if target in root_names or excluded in root_names:
        raise AssertionError("explicit protection allowlist contains an excluded workspace")
    paths: list[Path] = []
    for name in sorted(set(map(str, root_names))):
        if name.startswith(excluded + "/") or name.startswith(target + "/"):
            raise AssertionError("excluded workspace leaked into protection allowlist")
        path = ROOT / name
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f"protected root is missing: {name}")
        paths.append(path)
    return paths


def _leaf_paths(root: Path) -> list[Path]:
    if root.is_file() or root.is_symlink():
        return [root]
    result: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() or path.is_symlink():
            result.append(path)
    return sorted(result, key=lambda value: value.relative_to(ROOT).as_posix())


def _leaf_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    if path.is_symlink():
        target = os.readlink(path)
        digest = sha256_bytes(("symlink\0" + target).encode("utf-8"))
        return {"path": relative, "kind": "symlink", "target": target, "bytes": 0, "sha256": digest}
    return {
        "path": relative,
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _aggregate(records: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        f"{row['path']}\t{row['kind']}\t{row['bytes']}\t{row['sha256']}\n"
        for row in sorted(records, key=lambda value: str(value["path"]))
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))


def capture_protected(config: Mapping[str, Any]) -> dict[str, Any]:
    roots = protection_root_paths(config)
    all_paths: list[Path] = []
    memberships: dict[str, list[str]] = {}
    for root in roots:
        name = root.relative_to(ROOT).as_posix()
        leaves = _leaf_paths(root)
        memberships[name] = [path.relative_to(ROOT).as_posix() for path in leaves]
        all_paths.extend(leaves)
    unique_paths = sorted(set(all_paths), key=lambda value: value.relative_to(ROOT).as_posix())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(_leaf_record, unique_paths))
    by_path = {row["path"]: row for row in records}
    root_records: dict[str, Any] = {}
    for name, members in memberships.items():
        member_records = [by_path[path] for path in members]
        root_records[name] = {
            "path": name,
            "kind": "file" if len(members) == 1 and members[0] == name else "directory",
            "file_count": len(member_records),
            "total_bytes": sum(int(row["bytes"]) for row in member_records),
            "aggregate_sha256": _aggregate(member_records),
        }
    return {
        "captured_at_utc": utc_now(),
        "coverage_rule": "explicit roots recovered from the prior direct-extension protection receipt plus the complete direct-extension tree; analysis directory itself was not enumerated",
        "analysis_directory_enumerated": False,
        "excluded_concurrent_workspace": config["protection"]["excluded_concurrent_workspace"],
        "excluded_target": config["protection"]["excluded_target"],
        "root_count": len(root_records),
        "file_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "aggregate_sha256": _aggregate(records),
        "roots": root_records,
        "files": by_path,
    }


def compare_protected(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_files = before["files"]
    after_files = after["files"]
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    changed = {
        path: {
            "before": before_files[path]["sha256"],
            "after": after_files[path]["sha256"],
        }
        for path in sorted(set(before_files) & set(after_files))
        if before_files[path]["sha256"] != after_files[path]["sha256"]
        or before_files[path]["bytes"] != after_files[path]["bytes"]
        or before_files[path]["kind"] != after_files[path]["kind"]
    }
    passed = not added and not removed and not changed
    return {
        "checked_at_utc": utc_now(),
        "passed": passed,
        "aggregate_equal": before["aggregate_sha256"] == after["aggregate_sha256"],
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": after["aggregate_sha256"],
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": len(set(before_files) & set(after_files)) - len(changed),
    }


def capture_preflight() -> dict[str, Any]:
    config = load_config()
    source_verification = verify_sources(config)
    if not source_verification["all_verified"]:
        raise RuntimeError("one or more frozen source hashes failed preflight")
    record = {
        "analysis_id": config["analysis_id"],
        "stage": "preflight",
        "captured_at_utc": utc_now(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "repository": git_state(),
        "environment": environment_state(),
        "source_verification": source_verification,
        "protection_boundary": capture_protected(config),
        "isolation": {
            "profile_history_variant": "selected_90_day",
            "all_mature_workspace_consumed": False,
            "analysis_directory_enumerated": False,
            "incident": config["isolation_incident"],
        },
    }
    write_json(BEFORE_PATH, record)
    return record


def control_paths() -> list[Path]:
    paths = [
        WORKSPACE / "ROBUSTNESS_FROZEN_CONFIG.json",
        WORKSPACE / "ROBUSTNESS_PROTOCOL.md",
        WORKSPACE / "MODEL_FAMILY_DEFINITIONS.md",
        WORKSPACE / "RECOVERED_MODEL_SOURCE_RECEIPT.md",
        WORKSPACE / "PREFLIGHT_ATTEMPT1_INCIDENT_RECEIPT.md",
        WORKSPACE / "VALIDATION_ATTEMPT1_INCIDENT_RECEIPT.md",
        WORKSPACE / "FINAL_QA_SUPERSESSION_RECEIPT.md",
    ]
    paths.extend(sorted((WORKSPACE / "scripts").glob("*.py")))
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing robustness control files: {missing}")
    return paths


def capture_controls_before_model(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = verify_sources(config)
    if not sources["all_verified"]:
        raise RuntimeError("frozen upstream source hash failed immediately before model execution")
    files = {
        path.relative_to(WORKSPACE).as_posix(): {
            "path": path.relative_to(WORKSPACE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in control_paths()
    }
    record = {
        "analysis_id": config["analysis_id"],
        "captured_at_utc": utc_now(),
        "stage": "immediately_before_model_execution",
        "sources_verified": sources,
        "file_count": len(files),
        "aggregate_sha256": _aggregate([{**row, "kind": "file"} for row in files.values()]),
        "files": files,
    }
    write_json(CONTROL_BEFORE_PATH, record)
    return record


def verify_controls_after() -> dict[str, Any]:
    before = read_json(CONTROL_BEFORE_PATH)
    after_files = {
        path.relative_to(WORKSPACE).as_posix(): {
            "path": path.relative_to(WORKSPACE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in control_paths()
    }
    before_files = before["files"]
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    changed = {
        path: {"before": before_files[path]["sha256"], "after": after_files[path]["sha256"]}
        for path in sorted(set(before_files) & set(after_files))
        if before_files[path]["sha256"] != after_files[path]["sha256"]
        or before_files[path]["bytes"] != after_files[path]["bytes"]
    }
    aggregate = _aggregate([{**row, "kind": "file"} for row in after_files.values()])
    return {
        "checked_at_utc": utc_now(),
        "passed": not added and not removed and not changed and aggregate == before["aggregate_sha256"],
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": aggregate,
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": len(set(before_files) & set(after_files)) - len(changed),
    }


def finalize_integrity() -> dict[str, Any]:
    config = load_config()
    before = read_json(BEFORE_PATH)
    sources_after = verify_sources(config)
    protected_after = capture_protected(config)
    comparison = compare_protected(before["protection_boundary"], protected_after)
    controls = verify_controls_after()
    record = {
        "analysis_id": config["analysis_id"],
        "stage": "finalize_integrity",
        "captured_at_utc": utc_now(),
        "repository": git_state(),
        "environment": environment_state(),
        "source_verification": sources_after,
        "protection_boundary": protected_after,
        "comparison": comparison,
        "control_comparison": controls,
        "passed": bool(sources_after["all_verified"] and comparison["passed"] and controls["passed"]),
    }
    write_json(AFTER_PATH, record)
    if not record["passed"]:
        raise RuntimeError("protected byte-integrity verification failed")
    return record


def workspace_inventory(*, exclude: Iterable[str] = ()) -> dict[str, Any]:
    exclusions = set(exclude)
    records = []
    for path in sorted(WORKSPACE.rglob("*")):
        if not path.is_file() or path.relative_to(WORKSPACE).as_posix() in exclusions:
            continue
        relative = path.relative_to(WORKSPACE).as_posix()
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "file_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "aggregate_sha256": _aggregate(
            [{**row, "kind": "file"} for row in records]
        ),
        "files": {row["path"]: row for row in records},
    }


def write_hash_inventory(final_record: Mapping[str, Any]) -> Path:
    before = read_json(BEFORE_PATH)
    lines = [
        "# direct_model_family_robustness_v1 hash inventory",
        f"# generated_at_utc\t{utc_now()}",
        f"# protected_integrity_passed\t{final_record['passed']}",
        "section\tpath\tsha256\tbytes\tstatus",
    ]
    for name, row in sorted(before["source_verification"]["entries"].items()):
        lines.append(
            f"SOURCE_BEFORE\t{row['path']}\t{row['actual_sha256']}\t{row['bytes']}\t{'verified' if row['verified'] else 'failed'}"
        )
    for name, row in sorted(before["protection_boundary"]["roots"].items()):
        lines.append(
            f"PROTECTED_ROOT_BEFORE\t{name}\t{row['aggregate_sha256']}\t{row['total_bytes']}\tcaptured"
        )
    for name, row in sorted(final_record["protection_boundary"]["roots"].items()):
        lines.append(
            f"PROTECTED_ROOT_AFTER\t{name}\t{row['aggregate_sha256']}\t{row['total_bytes']}\tverified_unchanged"
        )
    inventory = workspace_inventory(exclude={"HASH_INVENTORY.txt", "RUN_MANIFEST.json"})
    for path, row in sorted(inventory["files"].items()):
        lines.append(f"ROBUSTNESS_OUTPUT\t{path}\t{row['sha256']}\t{row['bytes']}\tcaptured")
    destination = WORKSPACE / "HASH_INVENTORY.txt"
    atomic_write_text(destination, "\n".join(lines) + "\n")
    return destination


__all__ = [
    "AFTER_PATH",
    "BEFORE_PATH",
    "CONTROL_BEFORE_PATH",
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "ROOT",
    "WORKSPACE",
    "atomic_write_text",
    "capture_preflight",
    "capture_controls_before_model",
    "environment_state",
    "finalize_integrity",
    "git_state",
    "load_config",
    "read_json",
    "sha256_file",
    "stable_json",
    "utc_now",
    "workspace_inventory",
    "write_hash_inventory",
    "write_json",
]
