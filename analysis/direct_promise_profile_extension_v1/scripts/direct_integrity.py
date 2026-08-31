"""Read-only provenance and protected-output integrity for the direct extension.

The extension is allowed to mutate only its own workspace.  This module therefore
builds a flat, repository-relative SHA-256 inventory of every other immediate
``analysis`` child except explicitly recorded concurrently active workspaces, the
thesis/code trees, and the project control files.  It also verifies the exact
upstream inputs pinned in ``DIRECT_FROZEN_CONFIG.json``.

Two CLI stages are provided:

``preflight``
    Verify sources and capture repository, environment, extension-control, and
    protected-tree state in ``working/INTEGRITY_BEFORE.json``.

``finalize``
    Repeat the capture, compare every protected path and hash, write
    ``working/INTEGRITY_AFTER.json`` and ``HASH_INVENTORY.txt``, and return
    manifest-ready integrity sections.  All receipts are written before a failed
    comparison raises, so a mutation leaves an auditable failure record.

The mutable extension workspace is never recursively included in its own
protection boundary.  Only its explicit frozen controls and Python sources are
fingerprinted separately, avoiding self-reference through generated outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "analysis/direct_promise_profile_extension_v1"
CONFIG_PATH = WORKSPACE / "DIRECT_FROZEN_CONFIG.json"
WORKING = WORKSPACE / "working"
BEFORE_PATH = WORKING / "INTEGRITY_BEFORE.json"
AFTER_PATH = WORKING / "INTEGRITY_AFTER.json"
HASH_INVENTORY_PATH = WORKSPACE / "HASH_INVENTORY.txt"

ANALYSIS_ID = "direct_promise_profile_extension_v1"
SCHEMA_VERSION = "direct_integrity_v1"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

ROOT_CONTROL_RELATIVES = (
    "AGENTS.md",
    "PROJECT_CONTEXT.md",
    "RESULTS_REGISTRY.md",
    "DECISION_LOG.md",
)
OPTIONAL_ROOT_CONTROL_RELATIVES = ("EVIDENCE_LEDGER.md",)
PROTECTED_TREE_RELATIVES = ("docs", "report", "results", "src")

# This workspace was created by a separate active Codex task after this direct
# extension began.  It is not prior empirical evidence or an input to this run.
# Excluding only this named concurrent write target prevents legitimate parallel
# additions from invalidating the byte-identity proof for every prior artifact.
CONCURRENT_ANALYSIS_EXCLUSION_RELATIVES = (
    "analysis/all_mature_history_sensitivity_v1",
)

EXTENSION_CONTROL_RELATIVES = (
    "analysis/direct_promise_profile_extension_v1/DIRECT_EXTENSION_PROTOCOL.md",
    "analysis/direct_promise_profile_extension_v1/DIRECT_FROZEN_CONFIG.json",
    "analysis/direct_promise_profile_extension_v1/EXACT_FEATURE_MANIFEST.md",
)

EXTERNAL_CHARTER_PATH = Path(
    "docs/omitted-private-controls/"
    "OLIST_PROFILE_PIVOT_PROJECT_CHARTER_2026-08-21.md"
)
EXTERNAL_CHARTER_SHA256 = (
    "4aeb9ec87c14b208902a5dca74a00f7475ac4df0074da764518ea8e44c0bbe42"
)

PRIMARY_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "xgboost",
    "matplotlib",
    "holidays",
    "pyarrow",
)


class IntegrityError(RuntimeError):
    """Hard-stop error with optional JSON-serialisable audit detail."""

    def __init__(self, message: str, detail: Mapping[str, object] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_object_sha256(value: object) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    _atomic_write_text(path, payload + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IntegrityError(f"required integrity receipt is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"invalid JSON receipt: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON receipt must be an object: {path}")
    return value


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    if config.get("analysis_id") != ANALYSIS_ID:
        raise IntegrityError(
            f"wrong analysis_id in {path}: {config.get('analysis_id')!r}"
        )
    if not isinstance(config.get("sources"), Mapping):
        raise IntegrityError("DIRECT_FROZEN_CONFIG.json requires a sources mapping")
    return config


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size, stat.st_mtime_ns)


@dataclass
class HashCache:
    """Reuse hashes when configured sources are revisited by the tree inventory."""

    values: dict[tuple[str, tuple[int, int, int, int, int]], str] = field(
        default_factory=dict
    )

    def sha256_file(self, path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"expected a regular file for SHA-256: {path}")
        before = _stat_signature(path)
        key = (str(path.absolute()), before)
        cached = self.values.get(key)
        if cached is not None:
            if _stat_signature(path) != before:
                raise IntegrityError(f"file changed while consulting hash cache: {path}")
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        after = _stat_signature(path)
        if before != after:
            raise IntegrityError(f"file changed while being hashed: {path}")
        result = digest.hexdigest()
        self.values[key] = result
        return result


def sha256_file(path: Path, cache: HashCache | None = None) -> str:
    return (cache or HashCache()).sha256_file(path)


def _repo_relative(path: Path) -> str:
    try:
        return path.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError as exc:
        raise IntegrityError(f"path is outside the repository: {path}") from exc


def _display_path(path: Path) -> str:
    try:
        return _repo_relative(path)
    except IntegrityError:
        return str(path.absolute())


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _git(command: Sequence[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["git", *command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "command": ["git", *command],
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": ["git", *command],
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def capture_repository_state() -> dict[str, object]:
    branch_result = _git(("branch", "--show-current"))
    commit_result = _git(("rev-parse", "HEAD"))
    porcelain_result = _git(("status", "--porcelain=v1", "--untracked-files=all"))
    full_status_result = _git(("status", "--untracked-files=all"))

    branch_stdout = str(branch_result["stdout"]).strip()
    commit_stdout = str(commit_result["stdout"]).strip()
    available = all(
        result.get("return_code") == 0
        for result in (commit_result, porcelain_result, full_status_result)
    )
    return {
        "captured_at_utc": _utc_now(),
        "repository_root": ".",
        "available": available,
        "branch": branch_stdout or None,
        "detached_head": bool(available and not branch_stdout),
        "commit": commit_stdout or None,
        "status_clean": bool(
            porcelain_result.get("return_code") == 0
            and not str(porcelain_result.get("stdout", "")).strip()
        ),
        "status_porcelain_v1": str(porcelain_result.get("stdout", "")),
        "status_full": str(full_status_result.get("stdout", "")),
        "commands": {
            "branch": branch_result,
            "commit": commit_result,
            "porcelain": porcelain_result,
            "full_status": full_status_result,
        },
    }


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def capture_python_environment() -> dict[str, object]:
    distributions: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name") or ""
        name = str(raw_name).strip() or "<unnamed>"
        distributions.append(
            {
                "name": name,
                "canonical_name": _canonical_distribution_name(name),
                "version": str(distribution.version),
            }
        )
    distributions.sort(
        key=lambda row: (row["canonical_name"], row["name"], row["version"])
    )
    by_canonical: dict[str, list[str]] = {}
    for row in distributions:
        by_canonical.setdefault(row["canonical_name"], []).append(row["version"])
    primary = {
        name: by_canonical.get(_canonical_distribution_name(name), [])
        for name in PRIMARY_DISTRIBUTIONS
    }
    stable = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "primary_distributions": primary,
        "installed_distributions": distributions,
    }
    return {
        "captured_at_utc": _utc_now(),
        **stable,
        "environment_fingerprint_sha256": stable_object_sha256(stable),
    }


def _configured_source_specs(
    config: Mapping[str, object],
) -> list[tuple[str, Path, str, str]]:
    result: list[tuple[str, Path, str, str]] = []
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise IntegrityError("config sources must be a mapping")
    for logical_name, raw_spec in sorted(sources.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_spec, Sequence) or isinstance(raw_spec, (str, bytes)):
            raise IntegrityError(f"source {logical_name!r} must be [path, sha256]")
        if len(raw_spec) != 2:
            raise IntegrityError(f"source {logical_name!r} must contain two values")
        path = _resolve_path(raw_spec[0])
        expected = str(raw_spec[1]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise IntegrityError(f"source {logical_name!r} has invalid SHA-256")
        result.append((str(logical_name), path, expected, "frozen_source"))

    authorisation = config.get("authorisation")
    if isinstance(authorisation, Mapping):
        raw_path = authorisation.get("request_path")
        raw_hash = authorisation.get("request_sha256")
        if raw_path is not None and raw_hash is not None:
            expected = str(raw_hash).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise IntegrityError("authorisation.request_sha256 is invalid")
            result.append(
                (
                    "authorisation_request",
                    _resolve_path(raw_path),
                    expected,
                    "authorisation",
                )
            )
    return result


def verify_configured_sources(
    config: Mapping[str, object],
    cache: HashCache | None = None,
    *,
    strict: bool = True,
) -> dict[str, object]:
    cache = cache or HashCache()
    entries: dict[str, dict[str, object]] = {}
    for logical_name, path, expected, source_type in _configured_source_specs(config):
        exists = path.is_file() and not path.is_symlink()
        actual: str | None = None
        error: str | None = None
        if exists:
            try:
                actual = cache.sha256_file(path)
            except (OSError, IntegrityError) as exc:
                error = str(exc)
        status = (
            "verified"
            if actual == expected
            else "missing"
            if not exists
            else "unreadable"
            if actual is None
            else "hash_mismatch"
        )
        entries[logical_name] = {
            "logical_name": logical_name,
            "source_type": source_type,
            "path": _display_path(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
            "status": status,
            "error": error,
        }
    failures = sorted(name for name, row in entries.items() if row["status"] != "verified")
    result = {
        "checked_at_utc": _utc_now(),
        "entries": entries,
        "all_verified": not failures,
        "failures": failures,
    }
    if failures and strict:
        raise IntegrityError("configured source verification failed", result)
    return result


def inspect_external_charter(cache: HashCache | None = None) -> dict[str, object]:
    """Verify the external charter when readable; absence is non-blocking."""

    cache = cache or HashCache()
    path = EXTERNAL_CHARTER_PATH
    if not path.exists():
        return {
            "path": str(path),
            "expected_sha256": EXTERNAL_CHARTER_SHA256,
            "actual_sha256": None,
            "exists": False,
            "status": "missing_non_blocking",
            "blocking": False,
        }
    if not path.is_file() or path.is_symlink():
        return {
            "path": str(path),
            "expected_sha256": EXTERNAL_CHARTER_SHA256,
            "actual_sha256": None,
            "exists": True,
            "status": "not_readable_regular_file",
            "blocking": True,
        }
    try:
        actual = cache.sha256_file(path)
    except (OSError, IntegrityError) as exc:
        return {
            "path": str(path),
            "expected_sha256": EXTERNAL_CHARTER_SHA256,
            "actual_sha256": None,
            "exists": True,
            "status": "unreadable",
            "blocking": True,
            "error": str(exc),
        }
    verified = actual == EXTERNAL_CHARTER_SHA256
    return {
        "path": str(path),
        "expected_sha256": EXTERNAL_CHARTER_SHA256,
        "actual_sha256": actual,
        "exists": True,
        "bytes": path.stat().st_size,
        "status": "verified" if verified else "hash_mismatch",
        "blocking": not verified,
    }


def protected_targets() -> dict[str, Path]:
    """Return the complete external protection roots with stable repo keys."""

    analysis = ROOT / "analysis"
    if not analysis.is_dir():
        raise IntegrityError(f"analysis directory is missing: {analysis}")
    excluded = {
        (ROOT / relative).absolute()
        for relative in CONCURRENT_ANALYSIS_EXCLUSION_RELATIVES
    }
    targets: dict[str, Path] = {}
    for child in sorted(analysis.iterdir(), key=lambda item: item.name):
        if child.absolute() == WORKSPACE.absolute() or child.absolute() in excluded:
            continue
        targets[_repo_relative(child)] = child
    for relative in PROTECTED_TREE_RELATIVES:
        targets[relative] = ROOT / relative
    for relative in ROOT_CONTROL_RELATIVES:
        targets[relative] = ROOT / relative
    for relative in OPTIONAL_ROOT_CONTROL_RELATIVES:
        path = ROOT / relative
        if path.exists():
            targets[relative] = path
    return dict(sorted(targets.items()))


def _enumerate_leaves(root: Path) -> list[Path]:
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or root.is_file():
        return [root]
    if not root.is_dir():
        raise IntegrityError(f"unsupported protected root type: {root}")

    leaves: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise IntegrityError(
            f"could not completely traverse protected root {root}: {error}"
        ) from error

    for directory, dirnames, filenames in os.walk(
        root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        retained_dirs: list[str] = []
        for name in dirnames:
            candidate = directory_path / name
            if candidate.is_symlink():
                leaves.append(candidate)
            else:
                retained_dirs.append(name)
        dirnames[:] = retained_dirs
        leaves.extend(directory_path / name for name in filenames)
    return sorted(leaves, key=lambda item: _repo_relative(item))


def _leaf_record(path: Path, cache: HashCache) -> dict[str, object]:
    if path.is_symlink():
        before = path.lstat()
        target = os.readlink(path)
        payload = b"symlink\0" + os.fsencode(target)
        after = path.lstat()
        before_signature = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_signature != after_signature:
            raise IntegrityError(f"symlink changed while being inventoried: {path}")
        return {
            "kind": "symlink",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "link_target": target,
        }
    if not path.is_file():
        raise IntegrityError(f"unsupported protected leaf type: {path}")
    return {
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": cache.sha256_file(path),
    }


def capture_protection_boundary(cache: HashCache | None = None) -> dict[str, object]:
    cache = cache or HashCache()
    roots: dict[str, dict[str, object]] = {}
    files: dict[str, dict[str, object]] = {}
    targets_before = protected_targets()

    for root_key, path in targets_before.items():
        before_leaves = _enumerate_leaves(path)
        before_keys = [_repo_relative(item) for item in before_leaves]
        root_records: dict[str, dict[str, object]] = {}
        for leaf, key in zip(before_leaves, before_keys):
            root_records[key] = _leaf_record(leaf, cache)
        after_leaves = _enumerate_leaves(path)
        after_keys = [_repo_relative(item) for item in after_leaves]
        if before_keys != after_keys:
            raise IntegrityError(
                f"protected root changed during baseline capture: {root_key}",
                {
                    "added": sorted(set(after_keys) - set(before_keys)),
                    "removed": sorted(set(before_keys) - set(after_keys)),
                },
            )
        overlap = sorted(set(files) & set(root_records))
        if overlap:
            raise IntegrityError(
                f"protected roots overlap at files: {overlap[:10]}"
            )
        files.update(root_records)
        exists = path.exists() or path.is_symlink()
        kind = (
            "missing"
            if not exists
            else "symlink"
            if path.is_symlink()
            else "file"
            if path.is_file()
            else "directory"
        )
        roots[root_key] = {
            "exists": exists,
            "kind": kind,
            "file_count": len(root_records),
            "total_bytes": sum(int(row["bytes"]) for row in root_records.values()),
        }

    targets_after = protected_targets()
    before_target_map = {
        key: _repo_relative(path) for key, path in targets_before.items()
    }
    after_target_map = {
        key: _repo_relative(path) for key, path in targets_after.items()
    }
    if before_target_map != after_target_map:
        raise IntegrityError(
            "protected root set changed during boundary capture",
            _compare_record_maps(before_target_map, after_target_map),
        )

    stable_payload = {"roots": roots, "files": files}
    return {
        "captured_at_utc": _utc_now(),
        "coverage_rule": (
            "every immediate analysis child recursively except "
            "analysis/direct_promise_profile_extension_v1 and the explicitly "
            "recorded concurrent workspace analysis/all_mature_history_sensitivity_v1; "
            "plus docs, report, results, src, root controls, and "
            "EVIDENCE_LEDGER.md when present"
        ),
        "excluded_workspace": _repo_relative(WORKSPACE),
        "concurrent_workspace_exclusions": list(
            CONCURRENT_ANALYSIS_EXCLUSION_RELATIVES
        ),
        "stable_repo_relative_keys": True,
        "root_count": len(roots),
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files.values()),
        "roots": roots,
        "files": files,
        "aggregate_sha256": stable_object_sha256(stable_payload),
    }


def capture_extension_controls(cache: HashCache | None = None) -> dict[str, object]:
    """Hash only explicit frozen controls and Python source, never run outputs."""

    cache = cache or HashCache()
    paths = [ROOT / relative for relative in EXTENSION_CONTROL_RELATIVES]
    script_root = WORKSPACE / "scripts"
    if script_root.is_dir():
        paths.extend(
            path
            for path in script_root.rglob("*.py")
            if "__pycache__" not in path.parts and path.is_file()
        )
    unique = sorted({path.absolute() for path in paths}, key=_repo_relative)
    missing = [_repo_relative(path) for path in unique if not path.is_file()]
    if missing:
        raise IntegrityError("extension frozen controls are missing", {"missing": missing})
    files = {
        _repo_relative(path): {
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": cache.sha256_file(path),
        }
        for path in unique
    }
    return {
        "captured_at_utc": _utc_now(),
        "coverage_rule": "explicit extension controls plus scripts/**/*.py only",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files.values()),
        "aggregate_sha256": stable_object_sha256(files),
    }


def _compare_record_maps(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    before_keys = set(map(str, before.keys()))
    after_keys = set(map(str, after.keys()))
    common = sorted(before_keys & after_keys)
    changed = {
        key: {"before": before[key], "after": after[key]}
        for key in common
        if before[key] != after[key]
    }
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": changed,
        "unchanged_count": sum(before[key] == after[key] for key in common),
        "passed": not (after_keys - before_keys or before_keys - after_keys or changed),
    }


def compare_protection_boundaries(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    before_files = before.get("files")
    after_files = after.get("files")
    before_roots = before.get("roots")
    after_roots = after.get("roots")
    if not all(
        isinstance(value, Mapping)
        for value in (before_files, after_files, before_roots, after_roots)
    ):
        raise IntegrityError("protected-boundary receipt schema is invalid")
    file_comparison = _compare_record_maps(before_files, after_files)  # type: ignore[arg-type]
    root_comparison = _compare_record_maps(before_roots, after_roots)  # type: ignore[arg-type]
    aggregate_equal = before.get("aggregate_sha256") == after.get("aggregate_sha256")
    return {
        "before_aggregate_sha256": before.get("aggregate_sha256"),
        "after_aggregate_sha256": after.get("aggregate_sha256"),
        "aggregate_equal": aggregate_equal,
        "file_path_and_hash_comparison": file_comparison,
        "root_comparison": root_comparison,
        "passed": bool(
            aggregate_equal
            and file_comparison["passed"]
            and root_comparison["passed"]
        ),
    }


def compare_integrity_records(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    if before.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError("before integrity receipt uses an unsupported schema")
    if after.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError("after integrity receipt uses an unsupported schema")

    protection_before = before.get("protection_boundary")
    protection_after = after.get("protection_boundary")
    controls_before = before.get("extension_controls")
    controls_after = after.get("extension_controls")
    if not all(
        isinstance(value, Mapping)
        for value in (
            protection_before,
            protection_after,
            controls_before,
            controls_after,
        )
    ):
        raise IntegrityError("integrity receipt is missing boundary/control mappings")

    protection = compare_protection_boundaries(  # type: ignore[arg-type]
        protection_before, protection_after
    )
    control_files_before = controls_before.get("files")  # type: ignore[union-attr]
    control_files_after = controls_after.get("files")  # type: ignore[union-attr]
    if not isinstance(control_files_before, Mapping) or not isinstance(
        control_files_after, Mapping
    ):
        raise IntegrityError("extension-control receipt schema is invalid")
    controls = _compare_record_maps(control_files_before, control_files_after)

    sources_before = before.get("source_verification")
    sources_after = after.get("source_verification")
    if not isinstance(sources_before, Mapping) or not isinstance(sources_after, Mapping):
        raise IntegrityError("source-verification receipt schema is invalid")
    source_entries_before = sources_before.get("entries")
    source_entries_after = sources_after.get("entries")
    if not isinstance(source_entries_before, Mapping) or not isinstance(
        source_entries_after, Mapping
    ):
        raise IntegrityError("source-verification entries are missing")
    sources = _compare_record_maps(source_entries_before, source_entries_after)
    sources_verified = bool(
        sources_before.get("all_verified") and sources_after.get("all_verified")
    )

    repository_before = before.get("repository")
    repository_after = after.get("repository")
    if not isinstance(repository_before, Mapping) or not isinstance(
        repository_after, Mapping
    ):
        raise IntegrityError("repository-state receipt schema is invalid")
    repository_identity_unchanged = bool(
        repository_before.get("available")
        and repository_after.get("available")
        and repository_before.get("branch") == repository_after.get("branch")
        and repository_before.get("commit") == repository_after.get("commit")
    )

    environment_before = before.get("environment")
    environment_after = after.get("environment")
    if not isinstance(environment_before, Mapping) or not isinstance(
        environment_after, Mapping
    ):
        raise IntegrityError("environment receipt schema is invalid")
    environment_unchanged = bool(
        environment_before.get("environment_fingerprint_sha256")
        == environment_after.get("environment_fingerprint_sha256")
    )

    charter_before = before.get("external_charter")
    charter_after = after.get("external_charter")
    if not isinstance(charter_before, Mapping) or not isinstance(charter_after, Mapping):
        raise IntegrityError("external-charter receipt schema is invalid")
    charter_nonblocking_or_verified = bool(
        not charter_before.get("blocking") and not charter_after.get("blocking")
    )

    overall = bool(
        protection["passed"]
        and controls["passed"]
        and sources["passed"]
        and sources_verified
        and repository_identity_unchanged
        and environment_unchanged
        and charter_nonblocking_or_verified
    )
    return {
        "checked_at_utc": _utc_now(),
        "protected_before_after": protection,
        "extension_controls_before_after": controls,
        "configured_sources_before_after": sources,
        "configured_sources_verified": sources_verified,
        "repository_identity_unchanged": repository_identity_unchanged,
        "repository_status_is_recorded_but_expected_to_change": True,
        "environment_unchanged": environment_unchanged,
        "external_charter_nonblocking_or_verified": charter_nonblocking_or_verified,
        "passed": overall,
    }


def capture_preflight(
    config_path: Path = CONFIG_PATH,
    before_path: Path = BEFORE_PATH,
) -> dict[str, object]:
    cache = HashCache()
    config = load_config(config_path)
    sources = verify_configured_sources(config, cache, strict=True)
    charter = inspect_external_charter(cache)
    if charter.get("blocking"):
        raise IntegrityError("external charter exists but failed verification", charter)
    controls = capture_extension_controls(cache)
    boundary = capture_protection_boundary(cache)
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "stage": "before",
        "captured_at_utc": _utc_now(),
        "config_path": _display_path(config_path),
        "repository": capture_repository_state(),
        "environment": capture_python_environment(),
        "source_verification": sources,
        "external_charter": charter,
        "extension_controls": controls,
        "protection_boundary": boundary,
    }
    write_json(before_path, record)
    return record


def _source_entries(record: Mapping[str, object]) -> Mapping[str, object]:
    source = record.get("source_verification")
    if not isinstance(source, Mapping) or not isinstance(source.get("entries"), Mapping):
        raise IntegrityError("integrity record lacks source-verification entries")
    return source["entries"]  # type: ignore[return-value]


def _files(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = record.get(key)
    if not isinstance(section, Mapping) or not isinstance(section.get("files"), Mapping):
        raise IntegrityError(f"integrity record lacks {key}.files")
    return section["files"]  # type: ignore[return-value]


def _inventory_field(value: object) -> str:
    return str(value).replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def write_hash_inventory(
    path: Path,
    before: Mapping[str, object],
    after: Mapping[str, object],
    comparison: Mapping[str, object],
) -> None:
    lines = [
        f"# analysis_id\t{ANALYSIS_ID}",
        f"# schema_version\t{SCHEMA_VERSION}",
        f"# generated_at_utc\t{_utc_now()}",
        f"# overall_integrity_passed\t{bool(comparison.get('passed'))}",
        "section\tlogical_name\tsha256\texpected_sha256\tstatus\tpath",
    ]

    for stage, record in (("SOURCE_BEFORE", before), ("SOURCE_AFTER", after)):
        for logical, raw in sorted(_source_entries(record).items()):
            row = raw if isinstance(raw, Mapping) else {}
            lines.append(
                "\t".join(
                    map(
                        _inventory_field,
                        (
                            stage,
                            logical,
                            row.get("actual_sha256", ""),
                            row.get("expected_sha256", ""),
                            row.get("status", ""),
                            row.get("path", ""),
                        ),
                    )
                )
            )

    for stage, record in (("CONTROL_BEFORE", before), ("CONTROL_AFTER", after)):
        for repo_path, raw in sorted(_files(record, "extension_controls").items()):
            row = raw if isinstance(raw, Mapping) else {}
            lines.append(
                "\t".join(
                    map(
                        _inventory_field,
                        (stage, "", row.get("sha256", ""), "", "captured", repo_path),
                    )
                )
            )

    for stage, record in (("PROTECTED_BEFORE", before), ("PROTECTED_AFTER", after)):
        for repo_path, raw in sorted(_files(record, "protection_boundary").items()):
            row = raw if isinstance(raw, Mapping) else {}
            lines.append(
                "\t".join(
                    map(
                        _inventory_field,
                        (stage, "", row.get("sha256", ""), "", "captured", repo_path),
                    )
                )
            )

    for stage, record in (("CHARTER_BEFORE", before), ("CHARTER_AFTER", after)):
        row = record.get("external_charter")
        row = row if isinstance(row, Mapping) else {}
        lines.append(
            "\t".join(
                map(
                    _inventory_field,
                    (
                        stage,
                        "external_charter",
                        row.get("actual_sha256", ""),
                        row.get("expected_sha256", ""),
                        row.get("status", ""),
                        row.get("path", ""),
                    ),
                )
            )
        )
    _atomic_write_text(path, "\n".join(lines) + "\n")


def manifest_integrity_sections(
    before: Mapping[str, object],
    after: Mapping[str, object],
    comparison: Mapping[str, object],
    *,
    before_path: Path = BEFORE_PATH,
    after_path: Path = AFTER_PATH,
    inventory_path: Path = HASH_INVENTORY_PATH,
) -> dict[str, object]:
    before_boundary = before.get("protection_boundary")
    after_boundary = after.get("protection_boundary")
    if not isinstance(before_boundary, Mapping) or not isinstance(
        after_boundary, Mapping
    ):
        raise IntegrityError("integrity records lack protection-boundary summaries")
    artifacts: dict[str, dict[str, object]] = {}
    for name, path in (
        ("integrity_before", before_path),
        ("integrity_after", after_path),
        ("hash_inventory", inventory_path),
    ):
        artifacts[name] = {
            "path": _display_path(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "integrity_schema_version": SCHEMA_VERSION,
        "repository_before": before.get("repository"),
        "repository_after": after.get("repository"),
        "python_environment_before": before.get("environment"),
        "python_environment_after": after.get("environment"),
        "configured_source_verification_before": before.get("source_verification"),
        "configured_source_verification_after": after.get("source_verification"),
        "external_charter_before": before.get("external_charter"),
        "external_charter_after": after.get("external_charter"),
        "extension_controls_before": before.get("extension_controls"),
        "extension_controls_after": after.get("extension_controls"),
        "protected_baseline_summary": {
            "coverage_rule": before_boundary.get("coverage_rule"),
            "excluded_workspace": before_boundary.get("excluded_workspace"),
            "concurrent_workspace_exclusions": before_boundary.get(
                "concurrent_workspace_exclusions"
            ),
            "stable_repo_relative_keys": before_boundary.get(
                "stable_repo_relative_keys"
            ),
            "before_root_count": before_boundary.get("root_count"),
            "before_file_count": before_boundary.get("file_count"),
            "before_total_bytes": before_boundary.get("total_bytes"),
            "before_aggregate_sha256": before_boundary.get("aggregate_sha256"),
            "after_root_count": after_boundary.get("root_count"),
            "after_file_count": after_boundary.get("file_count"),
            "after_total_bytes": after_boundary.get("total_bytes"),
            "after_aggregate_sha256": after_boundary.get("aggregate_sha256"),
        },
        "protected_before_after_verdict": comparison.get(
            "protected_before_after"
        ),
        "extension_controls_before_after_verdict": comparison.get(
            "extension_controls_before_after"
        ),
        "overall_integrity_verdict": comparison,
        "integrity_artifacts": artifacts,
    }


def finalize_integrity(
    config_path: Path = CONFIG_PATH,
    before_path: Path = BEFORE_PATH,
    after_path: Path = AFTER_PATH,
    inventory_path: Path = HASH_INVENTORY_PATH,
    manifest_sections_path: Path | None = None,
    *,
    strict: bool = True,
) -> dict[str, object]:
    before = read_json(before_path)
    cache = HashCache()
    config = load_config(config_path)
    after: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "stage": "after",
        "captured_at_utc": _utc_now(),
        "config_path": _display_path(config_path),
        "repository": capture_repository_state(),
        "environment": capture_python_environment(),
        "source_verification": verify_configured_sources(
            config, cache, strict=False
        ),
        "external_charter": inspect_external_charter(cache),
        "extension_controls": capture_extension_controls(cache),
        "protection_boundary": capture_protection_boundary(cache),
    }
    comparison = compare_integrity_records(before, after)
    after["comparison_to_before"] = comparison
    write_json(after_path, after)
    write_hash_inventory(inventory_path, before, after, comparison)
    sections = manifest_integrity_sections(
        before,
        after,
        comparison,
        before_path=before_path,
        after_path=after_path,
        inventory_path=inventory_path,
    )
    if manifest_sections_path is not None:
        write_json(manifest_sections_path, sections)
    if strict and not comparison.get("passed"):
        raise IntegrityError("final integrity comparison failed", comparison)
    return sections


def _summary(record: Mapping[str, object]) -> dict[str, object]:
    boundary = record.get("protection_boundary")
    boundary = boundary if isinstance(boundary, Mapping) else {}
    sources = record.get("source_verification")
    sources = sources if isinstance(sources, Mapping) else {}
    return {
        "stage": record.get("stage"),
        "source_hashes_verified": sources.get("all_verified"),
        "protected_root_count": boundary.get("root_count"),
        "protected_file_count": boundary.get("file_count"),
        "protected_total_bytes": boundary.get("total_bytes"),
        "protected_aggregate_sha256": boundary.get("aggregate_sha256"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrity preflight/finalize for the direct promise-profile extension."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="verify sources and capture the protected before-state"
    )
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.add_argument("--before", type=Path, default=BEFORE_PATH)

    finalize = subparsers.add_parser(
        "finalize", help="capture after-state, compare, and write integrity inventory"
    )
    finalize.add_argument("--config", type=Path, default=CONFIG_PATH)
    finalize.add_argument("--before", type=Path, default=BEFORE_PATH)
    finalize.add_argument("--after", type=Path, default=AFTER_PATH)
    finalize.add_argument("--inventory", type=Path, default=HASH_INVENTORY_PATH)
    finalize.add_argument("--manifest-sections", type=Path)
    finalize.add_argument(
        "--no-strict",
        action="store_true",
        help="return the failed comparison after writing receipts instead of exiting nonzero",
    )

    verify = subparsers.add_parser(
        "verify-sources", help="verify configured sources without hashing protected trees"
    )
    verify.add_argument("--config", type=Path, default=CONFIG_PATH)

    subparsers.add_parser(
        "repository-state", help="print branch, commit, and full Git status"
    )
    subparsers.add_parser(
        "python-environment", help="print Python and installed-package versions"
    )
    subparsers.add_parser(
        "list-boundary", help="list protected roots without hashing their contents"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            record = capture_preflight(args.config, args.before)
            print(json.dumps(_summary(record), sort_keys=True))
        elif args.command == "finalize":
            sections = finalize_integrity(
                args.config,
                args.before,
                args.after,
                args.inventory,
                args.manifest_sections,
                strict=not args.no_strict,
            )
            verdict = sections.get("overall_integrity_verdict")
            passed = verdict.get("passed") if isinstance(verdict, Mapping) else False
            print(json.dumps({"integrity_passed": passed}, sort_keys=True))
        elif args.command == "verify-sources":
            result = verify_configured_sources(load_config(args.config), strict=True)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "repository-state":
            print(
                json.dumps(
                    capture_repository_state(),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
        elif args.command == "python-environment":
            print(
                json.dumps(
                    capture_python_environment(),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
        elif args.command == "list-boundary":
            print(
                json.dumps(
                    {key: _display_path(path) for key, path in protected_targets().items()},
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
        else:  # pragma: no cover - argparse enforces the subcommand.
            raise IntegrityError(f"unsupported command: {args.command}")
    except IntegrityError as exc:
        error = {"error": str(exc), "detail": exc.detail}
        print(json.dumps(error, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    return 0


__all__ = [
    "AFTER_PATH",
    "ANALYSIS_ID",
    "BEFORE_PATH",
    "CONFIG_PATH",
    "HASH_INVENTORY_PATH",
    "HashCache",
    "IntegrityError",
    "capture_extension_controls",
    "capture_preflight",
    "capture_protection_boundary",
    "capture_python_environment",
    "capture_repository_state",
    "compare_integrity_records",
    "compare_protection_boundaries",
    "finalize_integrity",
    "inspect_external_charter",
    "load_config",
    "manifest_integrity_sections",
    "protected_targets",
    "sha256_file",
    "stable_object_sha256",
    "verify_configured_sources",
    "write_hash_inventory",
]


if __name__ == "__main__":
    raise SystemExit(main())
