"""Fail-closed provenance controls for the supplementary RQ1 study.

This module deliberately writes only two pre-execution receipts inside the
new analysis workspace.  All prior empirical, source, thesis, and governance
content is treated as protected state and is recursively hashed.  The current
repository is already dirty, so preservation is defined as equality to the
captured preflight bytes and outside-workspace Git status, not equality to
``HEAD``.

The repository contains two assembler roles that must not be conflated:

* ``src/data/olist.py`` is the order assembler actually called by the existing
  RQ1 analysis; and
* ``analysis/profile_pivot_phase2a/scripts/data_pipeline.py`` is the canonical
  assembler registered indirectly through the later-study run manifests.

Both are trusted anchors.  The new analysis must use the former when
reproducing the existing RQ1 construction and retain the latter as the current
programme-level provenance check.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/rq1_speed_reliability_review_v1"
WORKING = OUT / "working"
PRESTATE_PATH = WORKING / "PRE_EXECUTION_STATE.json"
SOURCE_AUDIT_PATH = WORKING / "SOURCE_INPUT_AUDIT.csv"

TRUSTED_REGISTRY_SHA256 = (
    "adc62c280777b5e0d43aa1c7995d5f533d19f982a878f372670f648b11f302b4"
)

TRUSTED_RAW_INPUTS: dict[str, dict[str, Any]] = {
    "customers": {
        "filename": "olist_customers_dataset.csv",
        "sha256": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
        "bytes": 9_033_957,
    },
    "geolocation": {
        "filename": "olist_geolocation_dataset.csv",
        "sha256": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
        "bytes": 61_273_883,
    },
    "items": {
        "filename": "olist_order_items_dataset.csv",
        "sha256": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
        "bytes": 15_438_671,
    },
    "payments": {
        "filename": "olist_order_payments_dataset.csv",
        "sha256": "4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0",
        "bytes": 5_777_138,
    },
    "reviews": {
        "filename": "olist_order_reviews_dataset.csv",
        "sha256": "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7",
        "bytes": 14_451_670,
    },
    "orders": {
        "filename": "olist_orders_dataset.csv",
        "sha256": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
        "bytes": 17_654_914,
    },
    "products": {
        "filename": "olist_products_dataset.csv",
        "sha256": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
        "bytes": 2_379_446,
    },
    "sellers": {
        "filename": "olist_sellers_dataset.csv",
        "sha256": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
        "bytes": 174_703,
    },
    "categories": {
        "filename": "product_category_name_translation.csv",
        "sha256": "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
        "bytes": 2_613,
    },
}

TRUSTED_SOURCE_ANCHORS: dict[str, dict[str, str]] = {
    "programme_canonical_assembler": {
        "path": "analysis/profile_pivot_phase2a/scripts/data_pipeline.py",
        "sha256": "0c4cad3c99db268292253abd26e2070ccf2a286337bcfe9fb76fb3768c44ab8d",
    },
    "existing_rq1_order_assembler": {
        "path": "src/data/olist.py",
        "sha256": "5fcc289fe79756bc9d3e08b037d5c23c474ab986f054dfbf4a75ffeade27cab0",
    },
    "existing_rq1_date_normalizer": {
        "path": "src/features/targets.py",
        "sha256": "303913e67144d7721ffa46ca2567ba3cbffef1c901ce0410d64cccee0edaa1ea",
    },
    "existing_rq1_experiment": {
        "path": "src/experiments/rq1_customer_relevance.py",
        "sha256": "647024c67312a6899670e199d43b2982d3b0d06027a42057f11620567f69fbbf",
    },
    "existing_rq1_review_selection": {
        "path": "src/data/reviews.py",
        "sha256": "7d8b5182188ef73f03c200b74b7ebd546a6d98dae4628ec26239fd9296c88520",
    },
    "existing_rq1_tests": {
        "path": "tests/test_rq1_customer_relevance.py",
        "sha256": "a9b3a8181a2819cc2bd8ae64dbc09f5271a8929d13a33e6044591141cf0042bd",
    },
}

TRUSTED_EXISTING_RQ1_OUTPUTS: dict[str, str] = {
    "results/RQ1_AUDIT.md": "234680e7fd1e3872eaaf35593e703a1246d8eeabeed54f022ebf3d4b34a7e1a5",
    "results/figures/rq1_low_review_by_promise_error.pdf": "6a116c15bf39970c68639a8cebb8978ba45ea6bab4712369ad9a6696aef3641f",
    "results/figures/rq1_promise_error_over_time.pdf": "1d2a090c766c22e59a97f6feca8b3b03bcbe690472bc62f151e461023e1cbad3",
    "results/metrics/rq1_adjusted_low_review_full_coefficients.csv": "ee2cfcc5df6ebef28eda482b62209f5aae669575ba376d6f427ab350ca13c6a5",
    "results/metrics/rq1_adjusted_low_review_group_effects.csv": "02571b5fe338741f1d222d3ffe267b9892cd2ab96704e0106fe431c1117cddae",
    "results/metrics/rq1_adjusted_low_review_model_diagnostics.csv": "09b54fbfe87a2e3a1637db89aada31d285ed7599e7d7549971bbe4ec4393005f",
    "results/metrics/rq1_run_manifest.json": "4310d23244b975a853d85c572e309067eb670bf00c69d577c4f24650469ad388",
    "results/tables/rq1_promise_error_distribution.csv": "41431a25edcf047a9e80140539fefcb31709067b1d55a88cc5c9db35caae9d4e",
    "results/tables/rq1_promise_error_extreme_tails.csv": "a5831c4e1009f5fbd2977fb84ca75b3344b69ea5820dd3647182ad9584e3e2bb",
    "results/tables/rq1_promise_error_monthly.csv": "358234cee4542deb87cac767995869739d1c79e4c8be99334abf35790cedb315",
    "results/tables/rq1_review_coverage_by_error_group.csv": "ada6d814ac1a0822a9c68ca0c99b710936fa6f286e58f203a8feec6b9e6df611",
    "results/tables/rq1_review_coverage_by_purchase_month.csv": "775df1c32b2c7e5a43a4649fb7b8fe8acaa9da6a06c9c416412a4a3ea61c9fca",
    "results/tables/rq1_review_join_audit.csv": "b704e9fab4ce15b43bba5e057843f7994976ed650fd7e3f6680b385d394042de",
    "results/tables/rq1_review_multiple_records.csv": "de563c36e1a2540739c532934849fec17c33108a51bdb016d04f16fc4c41fb97",
    "results/tables/rq1_review_outcomes_by_error_group.csv": "a2b685473d559e5bd74f92017f28e27fc716607d8d4ece2ee5f31d6fc09e4d00",
    "results/tables/rq1_review_score_distribution.csv": "2b5eb937d10635bf7da8d2026d72b98f86029fff3a1cd511785ef7120022c703",
    "report/thesis/images/rq1_low_review_by_promise_error.pdf": "6a116c15bf39970c68639a8cebb8978ba45ea6bab4712369ad9a6696aef3641f",
    "report/thesis/images/rq1_promise_error_over_time.pdf": "1d2a090c766c22e59a97f6feca8b3b03bcbe690472bc62f151e461023e1cbad3",
}

TRUSTED_REGISTRY_MANIFESTS: dict[str, dict[str, str]] = {
    "phase2a": {
        "path": "analysis/profile_pivot_phase2a/RUN_MANIFEST.json",
        "sha256": "753540273c0fd74131b4e03f1532879efbc1a242edc8d25f44bcbe8992e1e4a0",
    },
    "dynamic_profile_eda_v1_1": {
        "path": "analysis/dynamic_profile_eda_v1_1/RUN_MANIFEST.json",
        "sha256": "b71ed8ff20fe61ba4364500ab1ddf92c418fa30825f14328406f86ca592e91f6",
    },
    "profile_validation_v1": {
        "path": "analysis/dynamic_profile_profile_validation_v1/RUN_MANIFEST.json",
        "sha256": "1dc505526f95b5476173703585d63385c9cc46a2a13f6047323e70195c7344a7",
    },
    "order_breach_severity_v1": {
        "path": "analysis/order_breach_severity_v1/RUN_MANIFEST.json",
        "sha256": "15faa0e8446e4284c4178e493d41c2c01b4c9caf7679ba9a77ee308231b44c99",
    },
}

_AUDIT_COLUMNS = (
    "input_group",
    "input_name",
    "path",
    "expected_sha256",
    "actual_sha256",
    "expected_bytes",
    "actual_bytes",
    "exists",
    "matched",
    "status",
    "authority_note",
)
_CACHE_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
}
_CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}
_TOP_LEVEL_EXCLUSIONS = {".git", ".venv", ".pytest_cache", "tmp", "analysis"}
_REQUIRED_LOCAL_CONTROLS = (
    "RQ1_SPEED_RELIABILITY_PROTOCOL.md",
    "RQ1_SPEED_RELIABILITY_FROZEN_CONFIG.json",
)
_PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "patsy",
    "matplotlib",
    "scikit-learn",
    "pytest",
)


class PreflightError(RuntimeError):
    """Raised when a required provenance or preservation check fails."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.details = details


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_fingerprint != after_fingerprint:
        raise PreflightError(f"file changed while being hashed: {path}")
    return digest.hexdigest()


def _stable_object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_source_audit(rows: Sequence[Mapping[str, Any]]) -> None:
    SOURCE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=SOURCE_AUDIT_PATH.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=_AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in _AUDIT_COLUMNS})
        temporary = Path(handle.name)
    os.replace(temporary, SOURCE_AUDIT_PATH)


def _audit_row(
    group: str,
    name: str,
    path: Path,
    expected_sha256: str,
    *,
    expected_bytes: int | None = None,
    authority_note: str,
) -> dict[str, Any]:
    exists = path.is_file()
    actual_sha256 = _sha256_file(path) if exists else ""
    actual_bytes = path.stat().st_size if exists else ""
    hash_match = exists and actual_sha256 == expected_sha256
    byte_match = expected_bytes is None or actual_bytes == expected_bytes
    matched = bool(hash_match and byte_match)
    return {
        "input_group": group,
        "input_name": name,
        "path": str(path.resolve(strict=False)),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "expected_bytes": "" if expected_bytes is None else expected_bytes,
        "actual_bytes": actual_bytes,
        "exists": str(exists).lower(),
        "matched": str(matched).lower(),
        "status": "verified" if matched else "mismatch",
        "authority_note": authority_note,
    }


def _audit_trusted_inputs(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in sorted(TRUSTED_RAW_INPUTS.items()):
        rows.append(
            _audit_row(
                "raw_olist",
                name,
                data_dir / str(spec["filename"]),
                str(spec["sha256"]),
                expected_bytes=int(spec["bytes"]),
                authority_note=(
                    "Current byte anchor; the seven non-payment/review files also "
                    "match registered later-study manifests. Payment/review are "
                    "additionally guarded by exact RQ1 reproduction."
                ),
            )
        )
    for name, spec in sorted(TRUSTED_SOURCE_ANCHORS.items()):
        rows.append(
            _audit_row(
                "source_anchor",
                name,
                ROOT / spec["path"],
                spec["sha256"],
                authority_note="Audited current RQ1/programme source anchor.",
            )
        )
    for relative, expected in sorted(TRUSTED_EXISTING_RQ1_OUTPUTS.items()):
        rows.append(
            _audit_row(
                "existing_rq1_output",
                Path(relative).name,
                ROOT / relative,
                expected,
                authority_note="Existing persisted RQ1 artifact; read-only.",
            )
        )
    rows.append(
        _audit_row(
            "governance_control",
            "results_registry",
            ROOT / "RESULTS_REGISTRY.md",
            TRUSTED_REGISTRY_SHA256,
            authority_note="Current Registry governing the indirect manifest chain.",
        )
    )
    for name, spec in sorted(TRUSTED_REGISTRY_MANIFESTS.items()):
        rows.append(
            _audit_row(
                "registered_manifest",
                name,
                ROOT / spec["path"],
                spec["sha256"],
                authority_note="Manifest path/hash explicitly registered in RESULTS_REGISTRY.md.",
            )
        )
    return rows


def _verify_registry_manifest_chain() -> dict[str, Any]:
    registry_path = ROOT / "RESULTS_REGISTRY.md"
    if not registry_path.is_file():
        raise PreflightError("RESULTS_REGISTRY.md is missing")
    registry_hash = _sha256_file(registry_path)
    registry_text = registry_path.read_text(encoding="utf-8")
    assembler = TRUSTED_SOURCE_ANCHORS["programme_canonical_assembler"]
    assembler_path = ROOT / assembler["path"]
    assembler_hash = _sha256_file(assembler_path) if assembler_path.is_file() else ""

    manifest_checks: dict[str, dict[str, Any]] = {}
    for name, spec in sorted(TRUSTED_REGISTRY_MANIFESTS.items()):
        manifest_path = ROOT / spec["path"]
        exists = manifest_path.is_file()
        actual_hash = _sha256_file(manifest_path) if exists else ""
        text = manifest_path.read_text(encoding="utf-8") if exists else ""
        json_valid = False
        if exists:
            try:
                json.loads(text)
                json_valid = True
            except json.JSONDecodeError:
                json_valid = False
        # The Registry writes the EDA/Phase-2A manifest paths in full.  Its
        # declared path convention permits the profile/order rows to use an
        # unprefixed ``RUN_MANIFEST.json`` under the row's experiment root.
        # Require either the full path or that exact root + hash-qualified
        # unprefixed reference; a bare filename by itself is insufficient.
        experiment_root = Path(spec["path"]).parent.as_posix()
        hash_qualified_unprefixed = (
            f"`RUN_MANIFEST.json` (`{spec['sha256']}`)" in registry_text
        )
        registered_path_present = bool(
            spec["path"] in registry_text
            or (experiment_root in registry_text and hash_qualified_unprefixed)
        )
        registered_hash_present = spec["sha256"] in registry_text
        assembler_hash_present = assembler["sha256"] in text
        assembler_path_present = (
            "scripts/data_pipeline.py" in text or "assembler_sha256" in text
        )
        passed = all(
            (
                exists,
                actual_hash == spec["sha256"],
                json_valid,
                registered_path_present,
                registered_hash_present,
                assembler_hash_present,
                assembler_path_present,
            )
        )
        manifest_checks[name] = {
            "path": str(manifest_path.resolve(strict=False)),
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual_hash,
            "json_valid": json_valid,
            "registered_path_present": registered_path_present,
            "registered_hash_present": registered_hash_present,
            "assembler_hash_present": assembler_hash_present,
            "assembler_path_present": assembler_path_present,
            "passed": passed,
        }

    eda_manifest = ROOT / TRUSTED_REGISTRY_MANIFESTS["dynamic_profile_eda_v1_1"]["path"]
    eda_text = eda_manifest.read_text(encoding="utf-8") if eda_manifest.is_file() else ""
    rq1_snapshot_checks = {
        name: spec["path"] in eda_text and spec["sha256"] in eda_text
        for name, spec in sorted(TRUSTED_SOURCE_ANCHORS.items())
        if name != "existing_rq1_tests"
    }
    passed = bool(
        registry_hash == TRUSTED_REGISTRY_SHA256
        and assembler_hash == assembler["sha256"]
        and all(check["passed"] for check in manifest_checks.values())
        and all(rq1_snapshot_checks.values())
    )
    return {
        "passed": passed,
        "verdict": (
            "indirect_registry_manifest_chain_match" if passed else "registry_chain_failed"
        ),
        "registry_path": str(registry_path.resolve()),
        "registry_expected_sha256": TRUSTED_REGISTRY_SHA256,
        "registry_actual_sha256": registry_hash,
        "direct_assembler_hash_in_registry": assembler["sha256"] in registry_text,
        "chain_note": (
            "RESULTS_REGISTRY.md does not directly state the canonical assembler "
            "hash. It registers immutable manifests by path and hash; those exact "
            "manifest bytes record the canonical assembler hash. This two-hop "
            "Registry-to-manifest-to-assembler chain is the verified authority."
        ),
        "programme_canonical_assembler": {
            "path": str(assembler_path.resolve(strict=False)),
            "expected_sha256": assembler["sha256"],
            "actual_sha256": assembler_hash,
        },
        "existing_rq1_assembler_role": {
            "path": str((ROOT / TRUSTED_SOURCE_ANCHORS["existing_rq1_order_assembler"]["path"]).resolve()),
            "sha256": TRUSTED_SOURCE_ANCHORS["existing_rq1_order_assembler"]["sha256"],
            "note": "Actual assembler invoked by the persisted RQ1 implementation.",
        },
        "registered_manifests": manifest_checks,
        "rq1_source_snapshot_in_registered_eda_manifest": rq1_snapshot_checks,
    }


def _normalise_command(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    return shlex.join(str(part) for part in command)


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip("\n")
    except subprocess.CalledProcessError as exc:
        raise PreflightError(
            f"git {' '.join(arguments)} failed", {"output": exc.output}
        ) from exc


def _status_line_inside_workspace(line: str) -> bool:
    if len(line) < 4:
        return False
    payload = line[3:]
    relative = OUT.relative_to(ROOT).as_posix()
    paths = [part.strip().strip('"') for part in payload.split(" -> ")]
    return bool(paths) and all(
        path == relative or path.startswith(relative + "/") for path in paths
    )


def _repository_state() -> dict[str, Any]:
    status_text = _git("status", "--porcelain=v1", "-uall")
    status = status_text.splitlines() if status_text else []
    outside = [line for line in status if not _status_line_inside_workspace(line)]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_outside_workspace": bool(outside),
        "status_porcelain": status,
        "status_porcelain_outside_workspace": outside,
        "workspace_status_lines_excluded": len(status) - len(outside),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _is_cache_file(relative: Path) -> bool:
    return bool(
        any(part in _CACHE_DIRECTORY_NAMES for part in relative.parts)
        or relative.suffix.lower() in _CACHE_FILE_SUFFIXES
    )


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _CACHE_DIRECTORY_NAMES
        )
        current = Path(directory)
        for filename in sorted(filenames):
            candidate = current / filename
            relative = candidate.relative_to(path)
            if not _is_cache_file(relative) and candidate.is_file():
                files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(path).as_posix())


def _protected_targets() -> dict[str, Path]:
    targets: dict[str, Path] = {}
    analysis = ROOT / "analysis"
    if not analysis.is_dir():
        raise PreflightError(f"analysis directory is missing: {analysis}")
    for item in sorted(analysis.iterdir(), key=lambda value: value.name):
        if item.resolve() == OUT.resolve():
            continue
        targets[f"analysis/{item.name}"] = item
    for item in sorted(ROOT.iterdir(), key=lambda value: value.name):
        if item.name in _TOP_LEVEL_EXCLUSIONS:
            continue
        if item.resolve() == OUT.resolve():
            continue
        targets[item.name] = item
    return targets


def _path_inventory(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if not path.exists():
        return {}, {
            "path": str(path.resolve(strict=False)),
            "exists": False,
            "kind": "missing",
            "file_count": 0,
            "total_bytes": 0,
        }
    files = _iter_files(path)
    before = {
        (item.name if path.is_file() else item.relative_to(path).as_posix()): (
            item.stat().st_dev,
            item.stat().st_ino,
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in files
    }
    hashes = {
        (item.name if path.is_file() else item.relative_to(path).as_posix()): _sha256_file(item)
        for item in files
    }
    after_files = _iter_files(path)
    after = {
        (item.name if path.is_file() else item.relative_to(path).as_posix()): (
            item.stat().st_dev,
            item.stat().st_ino,
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in after_files
    }
    if before != after:
        raise PreflightError(
            f"protected path changed during baseline capture: {path}",
            {
                "added": sorted(set(after) - set(before)),
                "removed": sorted(set(before) - set(after)),
                "metadata_changed": sorted(
                    name for name in set(before) & set(after) if before[name] != after[name]
                ),
            },
        )
    return hashes, {
        "path": str(path.resolve()),
        "exists": True,
        "kind": "file" if path.is_file() else "directory",
        "file_count": len(after),
        "total_bytes": sum(value[2] for value in after.values()),
    }


def _capture_protected_baseline() -> dict[str, Any]:
    hashes: dict[str, dict[str, str]] = {}
    roots: dict[str, dict[str, Any]] = {}
    for name, path in _protected_targets().items():
        root_hashes, root_state = _path_inventory(path)
        hashes[name] = root_hashes
        roots[name] = root_state
    payload = {"roots": roots, "hashes": hashes}
    return {
        "coverage_rule": (
            "Every existing immediate child of analysis/ except the new RQ1 "
            "workspace, plus every existing top-level repository item except "
            ".git, .venv, .pytest_cache, tmp and analysis itself. Python/tool "
            "cache directories and .pyc/.pyo files are excluded; empirical "
            "working/intermediate partitions remain protected."
        ),
        "excluded_new_workspace": str(OUT.resolve()),
        "excluded_cache_directory_names": sorted(_CACHE_DIRECTORY_NAMES),
        "root_count": len(roots),
        "file_count": sum(int(state["file_count"]) for state in roots.values()),
        "total_bytes": sum(int(state["total_bytes"]) for state in roots.values()),
        "roots": roots,
        "hashes": hashes,
        "aggregate_sha256": _stable_object_sha256(payload),
    }


def _compare_protected_baselines(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    before_hashes = before.get("hashes")
    after_hashes = after.get("hashes")
    if not isinstance(before_hashes, Mapping) or not isinstance(after_hashes, Mapping):
        raise PreflightError("protected baseline schema is invalid")
    detail: dict[str, Any] = {}
    passed = True
    for root_name in sorted(set(before_hashes) | set(after_hashes)):
        old_raw = before_hashes.get(root_name, {})
        new_raw = after_hashes.get(root_name, {})
        old = dict(old_raw) if isinstance(old_raw, Mapping) else {}
        new = dict(new_raw) if isinstance(new_raw, Mapping) else {}
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(
            name for name in set(old) & set(new) if old[name] != new[name]
        )
        unchanged = not (added or removed or changed)
        detail[root_name] = {
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
        }
        passed = passed and unchanged
    return {
        "passed": passed,
        "before_aggregate_sha256": before.get("aggregate_sha256"),
        "after_aggregate_sha256": after.get("aggregate_sha256"),
        "detail": detail,
        "after": after,
    }


def _local_frozen_inputs() -> dict[str, str]:
    missing = [name for name in _REQUIRED_LOCAL_CONTROLS if not (OUT / name).is_file()]
    if missing:
        raise PreflightError(
            "required local protocol/config must exist before preflight", {"missing": missing}
        )
    paths = [OUT / name for name in _REQUIRED_LOCAL_CONTROLS]
    scripts = OUT / "scripts"
    if scripts.is_dir():
        paths.extend(
            item
            for item in _iter_files(scripts)
            if item.suffix == ".py" and item.is_file()
        )
    if not any(path.name == "rq1_preflight.py" for path in paths):
        raise PreflightError("rq1_preflight.py is absent from the local source freeze")
    return {
        path.relative_to(OUT).as_posix(): _sha256_file(path)
        for path in sorted(set(paths), key=lambda item: item.relative_to(OUT).as_posix())
    }


def preflight(data_dir: str | Path, command: str | Sequence[str]) -> dict[str, Any]:
    """Capture the trusted-input and protected-state baseline.

    The function writes ``working/SOURCE_INPUT_AUDIT.csv`` and
    ``working/PRE_EXECUTION_STATE.json``.  Any mismatch writes a blocked
    prestate where possible and raises :class:`PreflightError`; callers must
    not fit a model after such an exception.
    """

    if PRESTATE_PATH.exists() or SOURCE_AUDIT_PATH.exists():
        raise PreflightError(
            "preflight receipts already exist; refusing to overwrite them",
            {
                "prestate_exists": PRESTATE_PATH.exists(),
                "source_audit_exists": SOURCE_AUDIT_PATH.exists(),
            },
        )
    started = _utc_now()
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise PreflightError(f"raw data directory is missing: {data_path}")

    rows = _audit_trusted_inputs(data_path)
    _write_source_audit(rows)
    mismatches = [row for row in rows if row["status"] != "verified"]
    base_state: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "rq1_speed_reliability_review_v1",
        "started_at_utc": started,
        "captured_at_utc": _utc_now(),
        "command": _normalise_command(command),
        "command_working_directory": str(Path.cwd().resolve()),
        "data_dir": str(data_path),
        "source_input_audit": {
            "path": str(SOURCE_AUDIT_PATH.resolve()),
            "sha256": _sha256_file(SOURCE_AUDIT_PATH),
            "row_count": len(rows),
            "verified_row_count": sum(row["status"] == "verified" for row in rows),
        },
        "trusted_anchors": {
            "raw_inputs": TRUSTED_RAW_INPUTS,
            "source_anchors": TRUSTED_SOURCE_ANCHORS,
            "existing_rq1_outputs": TRUSTED_EXISTING_RQ1_OUTPUTS,
            "registry_sha256": TRUSTED_REGISTRY_SHA256,
            "registered_manifests": TRUSTED_REGISTRY_MANIFESTS,
        },
    }
    if mismatches:
        blocked = {
            **base_state,
            "status": "blocked",
            "failure_stage": "trusted_input_audit",
            "blockers": mismatches,
        }
        _atomic_write_json(PRESTATE_PATH, blocked)
        raise PreflightError("trusted source/input audit failed", mismatches)

    registry_chain = _verify_registry_manifest_chain()
    if not registry_chain["passed"]:
        blocked = {
            **base_state,
            "status": "blocked",
            "failure_stage": "registry_manifest_assembler_chain",
            "registry_manifest_assembler_chain": registry_chain,
        }
        _atomic_write_json(PRESTATE_PATH, blocked)
        raise PreflightError("indirect Registry/manifest/assembler chain failed", registry_chain)

    local_frozen_inputs = _local_frozen_inputs()
    protected_baseline = _capture_protected_baseline()
    repository = _repository_state()
    state = {
        **base_state,
        "status": "passed",
        "repository": repository,
        "environment": _environment(),
        "registry_manifest_assembler_chain": registry_chain,
        "local_frozen_input_hashes": local_frozen_inputs,
        "protected_baseline": protected_baseline,
        "protected_hashes": protected_baseline["hashes"],
        "raw_file_hashes": {
            name: spec["sha256"] for name, spec in sorted(TRUSTED_RAW_INPUTS.items())
        },
        "raw_file_paths": {
            name: str((data_path / spec["filename"]).resolve())
            for name, spec in sorted(TRUSTED_RAW_INPUTS.items())
        },
        "existing_rq1_output_hashes": TRUSTED_EXISTING_RQ1_OUTPUTS,
        "blockers": [],
    }
    _atomic_write_json(PRESTATE_PATH, state)
    return state


def verify_protected_unchanged(
    prestate: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Re-hash all protected state and raise on any pre/post drift."""

    if isinstance(prestate, (str, Path)):
        state = json.loads(Path(prestate).read_text(encoding="utf-8"))
    else:
        state = dict(prestate)
    if state.get("status") != "passed":
        raise PreflightError("cannot verify a prestate that did not pass")
    before = state.get("protected_baseline")
    if not isinstance(before, Mapping):
        raise PreflightError("prestate protected_baseline is missing or invalid")

    after = _capture_protected_baseline()
    protected_audit = _compare_protected_baselines(before, after)
    current_rows = _audit_trusted_inputs(Path(str(state["data_dir"])))
    trusted_inputs_unchanged = all(row["status"] == "verified" for row in current_rows)
    registry_chain = _verify_registry_manifest_chain()
    local_frozen_inputs = _local_frozen_inputs()
    local_frozen_inputs_unchanged = (
        local_frozen_inputs == state.get("local_frozen_input_hashes")
    )
    repository_before = state.get("repository")
    repository_after = _repository_state()
    repository_unchanged = bool(
        isinstance(repository_before, Mapping)
        and repository_after["commit"] == repository_before.get("commit")
        and repository_after["branch"] == repository_before.get("branch")
        and repository_after["status_porcelain_outside_workspace"]
        == repository_before.get("status_porcelain_outside_workspace")
    )
    passed = bool(
        protected_audit["passed"]
        and trusted_inputs_unchanged
        and registry_chain["passed"]
        and local_frozen_inputs_unchanged
        and repository_unchanged
    )
    audit = {
        "verified_at_utc": _utc_now(),
        "passed": passed,
        "preservation_verdict": "unchanged" if passed else "changed_or_unverified",
        "protected_paths": protected_audit,
        "trusted_inputs_unchanged": trusted_inputs_unchanged,
        "trusted_input_mismatches": [
            row for row in current_rows if row["status"] != "verified"
        ],
        "registry_manifest_assembler_chain": registry_chain,
        "local_frozen_inputs_unchanged": local_frozen_inputs_unchanged,
        "local_frozen_inputs_before": state.get("local_frozen_input_hashes"),
        "local_frozen_inputs_after": local_frozen_inputs,
        "repository_unchanged_outside_workspace": repository_unchanged,
        "repository_before": repository_before,
        "repository_after": repository_after,
    }
    if not passed:
        raise PreflightError("protected inputs changed since preflight", audit)
    return audit


__all__ = [
    "OUT",
    "PRESTATE_PATH",
    "SOURCE_AUDIT_PATH",
    "PreflightError",
    "TRUSTED_EXISTING_RQ1_OUTPUTS",
    "TRUSTED_RAW_INPUTS",
    "TRUSTED_REGISTRY_MANIFESTS",
    "TRUSTED_SOURCE_ANCHORS",
    "preflight",
    "verify_protected_unchanged",
]
