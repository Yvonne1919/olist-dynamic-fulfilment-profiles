"""Read-only provenance gate for the frozen order-level experiment.

The preflight deliberately keeps its trust anchors in code rather than taking
expected hashes from the files being checked.  Successful execution writes two
receipts inside the new (unprotected) workspace:

* ``SOURCE_INPUT_AUDIT.csv``; and
* ``working/PRE_EXECUTION_STATE.json``.

The protected baseline covers every pre-existing top-level item under
``analysis`` except this experiment, plus ``docs``, ``report``, ``results``,
``src`` and the four project control files.  The upstream profile manifest's
artifact inventory is retained as useful provenance, but is explicitly not
treated as a complete protection boundary.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "analysis/order_breach_severity_v1"
WORKING = WORKSPACE / "working"
CONFIG_PATH = WORKSPACE / "ORDER_FROZEN_CONFIG.json"
PRESTATE_PATH = WORKING / "PRE_EXECUTION_STATE.json"
SOURCE_AUDIT_PATH = WORKSPACE / "SOURCE_INPUT_AUDIT.csv"

EXPECTED_ORDER_CONFIG_SHA256 = (
    "675fc4079f770b701f67a8aa46247c38caf202bafe51815912884efd5532f0a2"
)
EXPECTED_ORDER_PROTOCOL_SHA256 = (
    "27c6a871a8e67d4349c3f27b16497b3094435a8a7e8ffacdfff6954a8f64a599"
)
EXPECTED_ASSEMBLER_SHA256 = (
    "0c4cad3c99db268292253abd26e2070ccf2a286337bcfe9fb76fb3768c44ab8d"
)

RAW_FILE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "orders",
        "olist_orders_dataset.csv",
        "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    ),
    (
        "customers",
        "olist_customers_dataset.csv",
        "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    ),
    (
        "geolocation",
        "olist_geolocation_dataset.csv",
        "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    ),
    (
        "items",
        "olist_order_items_dataset.csv",
        "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    ),
    (
        "products",
        "olist_products_dataset.csv",
        "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    ),
    (
        "sellers",
        "olist_sellers_dataset.csv",
        "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    ),
    (
        "categories",
        "product_category_name_translation.csv",
        "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
    ),
)

ASSEMBLER_RELATIVE = "analysis/profile_pivot_phase2a/scripts/data_pipeline.py"
PROFILE_VALIDATION_RELATIVE = "analysis/dynamic_profile_profile_validation_v1"
PROFILE_PROTOCOL_SHA256 = (
    "69c6a3167e9434cb0c63809998f89e02428a7ee22dcaf9392dee62e036b2b6df"
)
PROFILE_CONFIG_SHA256 = (
    "7494cd3ddfe12514d96f75d5c765b920ca6b4a39da7561632901ead5b77c5d02"
)

# name, repository-relative path, immutable SHA-256, config path key, config hash key
PROFILE_INPUT_SPECS: tuple[tuple[str, str, str, str | None, str | None], ...] = (
    (
        "profile_protocol",
        f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_PROTOCOL.md",
        PROFILE_PROTOCOL_SHA256,
        None,
        None,
    ),
    (
        "profile_frozen_config",
        f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_FROZEN_CONFIG.json",
        PROFILE_CONFIG_SHA256,
        None,
        None,
    ),
    (
        "profile_selection_freeze",
        f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_SELECTION_FREEZE.json",
        "f2409082543bca174c13a2ba94481d2d03c7413021232678ff139438ece69742",
        "profile_selection_freeze",
        "profile_selection_freeze_sha256",
    ),
    (
        "profile_selected_candidates",
        f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_SELECTED_CANDIDATES.csv",
        "b8a4cb4b71a09493c9db5fa8da5248078e2906ceb0c7faad46a6770433358659",
        "profile_selected_candidates",
        "profile_selected_candidates_sha256",
    ),
    (
        "profile_daily_scores",
        f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_DAILY_SCORES.csv.gz",
        "ff3c3f19982714087b9309e03fb35d99bf9039a574b819afa2b6b544e330b56c",
        "profile_daily_input",
        "profile_daily_input_sha256",
    ),
    (
        "profile_parent_structure",
        f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_PARENT_STRUCTURE.csv",
        "da0ae9431165f8bc635f6805ef89d79ef8bf28802aed53c50020596415593aa4",
        "profile_parent_input",
        "profile_parent_input_sha256",
    ),
    (
        "confirmation_labels",
        f"{PROFILE_VALIDATION_RELATIVE}/working/CONFIRMATION_LABELS.csv",
        "1cdca2914de4a85edec10529ae6ff7c8c134d28b15091f861874095fa9207ee4",
        "confirmation_labels",
        "confirmation_labels_sha256",
    ),
    (
        "confirmation_by_month_for_labels",
        f"{PROFILE_VALIDATION_RELATIVE}/working/CONFIRMATION_BY_MONTH_FOR_LABELS.csv",
        "dbc29ad31b30ada32ec7be6d00698a1ac207c7d6930d4d2a75dbc8bcb518a103",
        "confirmation_month_input",
        "confirmation_month_input_sha256",
    ),
    (
        "hrd_daily_labels",
        f"{PROFILE_VALIDATION_RELATIVE}/working/HRD_DAILY_LABELS.csv",
        "9e63f45823998303a8bd47d0a1b238fb563fc455cad3e7bf65d5c100c6ed66d3",
        "hrd_daily_labels",
        "hrd_daily_labels_sha256",
    ),
)

EXPECTED_PROFILE_BLOCKS: Mapping[str, Mapping[str, object]] = {
    "S1": {
        "name": "seller_handling_level",
        "target": "handling_level",
        "candidate_id": (
            "handling_level|seller_id|C|w90|l14|P1|parent=global|"
            "kappa=na|min_support=5"
        ),
        "base_candidate_id": (
            "handling_level|seller_id|C|w90|l14|P1|parent=global|kappa=na"
        ),
        "profile_spec_id": "ps_18f6d18af885ac9c1930",
        "entity": "seller_id",
        "scheme": "C",
        "window_days": 90,
        "lag_days": 14,
        "estimator": "P1",
        "parent": "global",
        "kappa": None,
        "min_support": 5,
    },
    "S2": {
        "name": "seller_handling_tail",
        "target": "handling_tail",
        "candidate_id": (
            "handling_tail|seller_id|A|w90|l0|P1|parent=global|"
            "kappa=10|min_support=5"
        ),
        "base_candidate_id": (
            "handling_tail|seller_id|A|w90|l0|P1|parent=global|kappa=10"
        ),
        "profile_spec_id": "ps_29c28f8f40eed03c1031",
        "entity": "seller_id",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 10,
        "min_support": 5,
    },
    "R1": {
        "name": "route_transit_level",
        "target": "transit_level",
        "candidate_id": (
            "transit_level|state_od|A|w90|l0|P0|parent=global|"
            "kappa=na|min_support=5"
        ),
        "base_candidate_id": (
            "transit_level|state_od|A|w90|l0|P0|parent=global|kappa=na"
        ),
        "profile_spec_id": "ps_18f16966ac00ff520226",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P0",
        "parent": "global",
        "kappa": None,
        "min_support": 5,
    },
    "R2": {
        "name": "route_transit_tail",
        "target": "transit_tail",
        "candidate_id": (
            "transit_tail|state_od|A|w90|l0|P1|parent=global|"
            "kappa=10|min_support=5"
        ),
        "base_candidate_id": (
            "transit_tail|state_od|A|w90|l0|P1|parent=global|kappa=10"
        ),
        "profile_spec_id": "ps_9799491505b2347220fb",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 10,
        "min_support": 5,
    },
    "M5_ENDPOINT": {
        "name": "route_historical_final_breach",
        "target": "final_breach",
        "candidate_id": (
            "final_breach|state_od|A|w90|l0|P1|parent=global|"
            "kappa=100|min_support=5"
        ),
        "base_candidate_id": (
            "final_breach|state_od|A|w90|l0|P1|parent=global|kappa=100"
        ),
        "profile_spec_id": "ps_ef5d05dc7c0496cca415",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 100,
        "min_support": 5,
        "representative_rule": (
            "lowest_frozen_development_selection_rank_then_lexical_candidate_id"
        ),
        "frozen_selection_rank": 1,
    },
}

SOURCE_AUDIT_COLUMNS = (
    "input_group",
    "input_name",
    "path",
    "expected_sha256",
    "actual_sha256",
    "exists",
    "bytes",
    "status",
)

PACKAGE_NAMES = (
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

CONTROL_RELATIVES = (
    "AGENTS.md",
    "PROJECT_CONTEXT.md",
    "RESULTS_REGISTRY.md",
    "DECISION_LOG.md",
)

HashCache = dict[str, tuple[int, int, int, int, int, str]]


class PreflightError(RuntimeError):
    """A frozen provenance condition failed."""

    def __init__(self, message: str, detail: object | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"JSON input is not an object: {path}")
    return value


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, default=str
    ) + "\n"
    _atomic_write_text(path, payload)


def _write_source_audit(rows: Sequence[Mapping[str, object]]) -> None:
    SOURCE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=SOURCE_AUDIT_PATH.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_AUDIT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SOURCE_AUDIT_COLUMNS})
        temporary = Path(handle.name)
    os.replace(temporary, SOURCE_AUDIT_PATH)


def _stat_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _sha256_file(path: Path, cache: HashCache | None = None) -> str:
    key = str(path.resolve())
    before = _stat_fingerprint(path)
    if cache is not None and key in cache and cache[key][:-1] == before:
        return cache[key][-1]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = _stat_fingerprint(path)
    if before != after:
        raise PreflightError(f"file changed while being hashed: {path}")
    value = digest.hexdigest()
    if cache is not None:
        cache[key] = (*after, value)
    return value


def _stable_object_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_row(
    group: str,
    name: str,
    path: Path,
    expected_sha256: str,
    cache: HashCache,
) -> dict[str, object]:
    exists = path.is_file()
    actual = _sha256_file(path, cache) if exists else ""
    if not exists:
        status = "missing"
    elif actual != expected_sha256:
        status = "hash_mismatch"
    else:
        status = "verified"
    return {
        "input_group": group,
        "input_name": name,
        "path": str(path.resolve(strict=False)),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "exists": str(exists).lower(),
        "bytes": path.stat().st_size if exists else "",
        "status": status,
    }


def _audit_required_sources(
    data_dir: Path,
    config: Mapping[str, object],
    cache: HashCache,
) -> tuple[list[dict[str, object]], list[str]]:
    rows = [
        _source_row("raw_olist", name, data_dir / filename, expected, cache)
        for name, filename, expected in RAW_FILE_SPECS
    ]
    rows.append(
        _source_row(
            "canonical_assembler",
            "data_pipeline",
            ROOT / ASSEMBLER_RELATIVE,
            EXPECTED_ASSEMBLER_SHA256,
            cache,
        )
    )
    rows.extend(
        _source_row("profile_validation", name, ROOT / relative, expected, cache)
        for name, relative, expected, _, _ in PROFILE_INPUT_SPECS
    )

    errors = [
        f"{row['input_name']}:{row['status']}"
        for row in rows
        if row["status"] != "verified"
    ]
    data = config.get("data")
    if not isinstance(data, Mapping):
        errors.append("ORDER_FROZEN_CONFIG data section is missing or invalid")
        return rows, errors

    expected_data_values: dict[str, object] = {
        "canonical_assembler": ASSEMBLER_RELATIVE,
        "canonical_assembler_sha256": EXPECTED_ASSEMBLER_SHA256,
    }
    for _, relative, expected, path_key, hash_key in PROFILE_INPUT_SPECS:
        if path_key is not None:
            expected_data_values[path_key] = relative
        if hash_key is not None:
            expected_data_values[hash_key] = expected
    for key, expected in expected_data_values.items():
        if data.get(key) != expected:
            errors.append(
                f"ORDER_FROZEN_CONFIG data.{key} drifted: "
                f"expected {expected!r}, observed {data.get(key)!r}"
            )
    return rows, errors


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PreflightError(f"CSV has no header: {path}")
            return list(reader)
    except OSError as exc:
        raise PreflightError(f"cannot read CSV input {path}: {exc}") from exc


def _optional_number(value: object) -> int | float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "na", "nan", "none", "null"}:
        return None
    number = float(text)
    return int(number) if number.is_integer() else number


def _integer(value: object, label: str) -> int:
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise PreflightError(f"invalid integer {label}: {value!r}") from exc
    if not number.is_integer():
        raise PreflightError(f"invalid integer {label}: {value!r}")
    return int(number)


def _parse_candidate_id(candidate_id: str) -> dict[str, object]:
    parts = candidate_id.split("|")
    if len(parts) != 9:
        raise PreflightError(f"unexpected candidate_id grammar: {candidate_id}")
    target, entity, scheme, window, lag, estimator = parts[:6]
    attributes: dict[str, str] = {}
    for part in parts[6:]:
        if "=" not in part:
            raise PreflightError(f"unexpected candidate_id token: {part}")
        key, value = part.split("=", 1)
        if key in attributes:
            raise PreflightError(f"duplicate candidate_id token {key}: {candidate_id}")
        attributes[key] = value
    if set(attributes) != {"parent", "kappa", "min_support"}:
        raise PreflightError(f"candidate_id attributes drifted: {candidate_id}")
    if not window.startswith("w") or not lag.startswith("l"):
        raise PreflightError(f"candidate_id window/lag grammar drifted: {candidate_id}")
    return {
        "target": target,
        "entity": entity,
        "scheme": scheme,
        "window_days": _integer(window[1:], "candidate window_days"),
        "lag_days": _integer(lag[1:], "candidate lag_days"),
        "estimator": estimator,
        "parent": attributes["parent"],
        "kappa": _optional_number(attributes["kappa"]),
        "min_support": _integer(attributes["min_support"], "candidate min_support"),
    }


def _candidate_row_projection(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "target": row.get("target"),
        "entity": row.get("granularity"),
        "scheme": row.get("scheme"),
        "window_days": _integer(row.get("window_days"), "selected window_days"),
        "lag_days": _integer(row.get("lag_days"), "selected lag_days"),
        "estimator": row.get("estimator"),
        "parent": row.get("parent_structure"),
        "kappa": _optional_number(row.get("kappa")),
        "min_support": _integer(row.get("min_support"), "selected min_support"),
    }


def _validate_profile_blocks(
    config: Mapping[str, object],
) -> dict[str, object]:
    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping):
        raise PreflightError("ORDER_FROZEN_CONFIG profiles section is missing or invalid")

    selected_path = ROOT / f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_SELECTED_CANDIDATES.csv"
    rows = _read_csv_rows(selected_path)
    if not rows:
        raise PreflightError("PROFILE_SELECTED_CANDIDATES.csv is empty")
    required_columns = {
        "candidate_id",
        "base_candidate_id",
        "profile_spec_id",
        "target",
        "granularity",
        "scheme",
        "window_days",
        "lag_days",
        "estimator",
        "parent_structure",
        "kappa",
        "min_support",
        "selection_rank",
        "selection_decision",
    }
    missing_columns = sorted(required_columns - set(rows[0]))
    if missing_columns:
        raise PreflightError(f"selected-candidate columns missing: {missing_columns}")
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise PreflightError("duplicate candidate_id in PROFILE_SELECTED_CANDIDATES.csv")
    by_candidate = {str(row["candidate_id"]): row for row in rows}

    receipts: dict[str, object] = {}
    for block_name, expected_with_target in EXPECTED_PROFILE_BLOCKS.items():
        expected = dict(expected_with_target)
        target = str(expected.pop("target"))
        observed = profiles.get(block_name)
        if observed != expected:
            raise PreflightError(
                f"frozen profile block {block_name} drifted",
                {"expected": expected, "observed": observed},
            )
        parsed = _parse_candidate_id(str(expected["candidate_id"]))
        semantic_expected = {
            "target": target,
            "entity": expected["entity"],
            "scheme": expected["scheme"],
            "window_days": expected["window_days"],
            "lag_days": expected["lag_days"],
            "estimator": expected["estimator"],
            "parent": expected["parent"],
            "kappa": expected["kappa"],
            "min_support": expected["min_support"],
        }
        if parsed != semantic_expected:
            raise PreflightError(
                f"candidate_id semantics disagree for {block_name}",
                {"parsed": parsed, "expected": semantic_expected},
            )
        row = by_candidate.get(str(expected["candidate_id"]))
        if row is None:
            raise PreflightError(f"frozen candidate absent from selected CSV: {block_name}")
        if str(row.get("selection_decision")) != "selected":
            raise PreflightError(f"frozen candidate is not marked selected: {block_name}")
        if row.get("base_candidate_id") != expected["base_candidate_id"]:
            raise PreflightError(f"base_candidate_id mismatch for {block_name}")
        if row.get("profile_spec_id") != expected["profile_spec_id"]:
            raise PreflightError(f"profile_spec_id mismatch for {block_name}")
        if _candidate_row_projection(row) != semantic_expected:
            raise PreflightError(
                f"selected CSV specification mismatch for {block_name}",
                {
                    "observed": _candidate_row_projection(row),
                    "expected": semantic_expected,
                },
            )
        receipts[block_name] = {
            "candidate_id": expected["candidate_id"],
            "base_candidate_id": expected["base_candidate_id"],
            "profile_spec_id": expected["profile_spec_id"],
            "selection_rank": _integer(
                row.get("selection_rank"), f"{block_name} selection_rank"
            ),
            "specification_verified": True,
        }

    freeze_path = ROOT / f"{PROFILE_VALIDATION_RELATIVE}/PROFILE_SELECTION_FREEZE.json"
    freeze = _load_json(freeze_path)
    if freeze.get("frozen_config_sha256") != PROFILE_CONFIG_SHA256:
        raise PreflightError("selection freeze profile-config hash mismatch")
    if freeze.get("protocol_sha256") != PROFILE_PROTOCOL_SHA256:
        raise PreflightError("selection freeze profile-protocol hash mismatch")
    promoted = freeze.get("promoted_candidates")
    if not isinstance(promoted, list):
        raise PreflightError("selection freeze promoted_candidates is missing or invalid")
    tie_break_choices = freeze.get("tie_break_choices")
    if not isinstance(tie_break_choices, list):
        raise PreflightError("selection freeze tie_break_choices is missing or invalid")
    freeze_by_candidate: dict[str, Mapping[str, object]] = {}
    for item in promoted:
        if not isinstance(item, Mapping) or not isinstance(item.get("candidate_id"), str):
            raise PreflightError("invalid promoted candidate in selection freeze")
        candidate = str(item["candidate_id"])
        if candidate in freeze_by_candidate:
            raise PreflightError(f"duplicate promoted candidate in selection freeze: {candidate}")
        freeze_by_candidate[candidate] = item
    tie_break_by_candidate: dict[str, Mapping[str, object]] = {}
    for item in tie_break_choices:
        if not isinstance(item, Mapping) or not isinstance(item.get("candidate_id"), str):
            raise PreflightError("invalid candidate in selection-freeze tie-break choices")
        candidate = str(item["candidate_id"])
        if candidate in tie_break_by_candidate:
            raise PreflightError(
                f"duplicate candidate in selection-freeze tie-break choices: {candidate}"
            )
        tie_break_by_candidate[candidate] = item
    for block_name, receipt in receipts.items():
        candidate = str(receipt["candidate_id"])
        frozen_row = freeze_by_candidate.get(candidate)
        if frozen_row is None:
            raise PreflightError(f"frozen block absent from selection freeze: {block_name}")
        frozen_rank = _integer(
            frozen_row.get("selection_rank"), f"{block_name} frozen selection_rank"
        )
        if frozen_rank != receipt["selection_rank"]:
            raise PreflightError(f"CSV/freeze selection-rank mismatch for {block_name}")
        tie_break_row = tie_break_by_candidate.get(candidate)
        if tie_break_row is None:
            raise PreflightError(f"frozen block absent from tie-break choices: {block_name}")
        if (
            _integer(
                tie_break_row.get("selection_rank"),
                f"{block_name} tie-break selection_rank",
            )
            != frozen_rank
            or tie_break_row.get("selection_decision") != "selected"
            or tie_break_row.get("selected_for_confirmation") is not True
        ):
            raise PreflightError(f"tie-break receipt mismatch for frozen block {block_name}")

    m5_expected = EXPECTED_PROFILE_BLOCKS["M5_ENDPOINT"]
    m5_semantic = {
        "target": m5_expected["target"],
        "entity": m5_expected["entity"],
        "scheme": m5_expected["scheme"],
        "window_days": m5_expected["window_days"],
        "lag_days": m5_expected["lag_days"],
        "estimator": m5_expected["estimator"],
        "parent": m5_expected["parent"],
        "min_support": m5_expected["min_support"],
    }
    m5_family: list[tuple[int, str, int | float | None]] = []
    for row in rows:
        projection = _candidate_row_projection(row)
        without_kappa = {key: value for key, value in projection.items() if key != "kappa"}
        if without_kappa == m5_semantic and row.get("selection_decision") == "selected":
            tie_break_row = tie_break_by_candidate.get(str(row["candidate_id"]))
            if (
                tie_break_row is None
                or tie_break_row.get("selection_decision") != "selected"
                or tie_break_row.get("selected_for_confirmation") is not True
            ):
                raise PreflightError(
                    f"M5 family candidate lacks a selected development tie-break receipt: "
                    f"{row['candidate_id']}"
                )
            m5_family.append(
                (
                    _integer(row.get("selection_rank"), "M5 family selection_rank"),
                    str(row["candidate_id"]),
                    projection["kappa"],
                )
            )
    if not m5_family:
        raise PreflightError("no frozen development candidates exist for the M5 endpoint family")
    m5_family.sort(key=lambda value: (value[0], value[1]))
    chosen_rank, chosen_candidate, chosen_kappa = m5_family[0]
    if chosen_candidate != m5_expected["candidate_id"]:
        raise PreflightError(
            "M5 endpoint is not the lowest development-rank/lexical representative",
            {"resolved": chosen_candidate, "configured": m5_expected["candidate_id"]},
        )
    if chosen_rank != m5_expected["frozen_selection_rank"]:
        raise PreflightError("M5 frozen selection rank mismatch")
    receipts["M5_ENDPOINT"]["rank_resolution"] = {
        "rule": m5_expected["representative_rule"],
        "eligible_candidates": [
            {"selection_rank": rank, "candidate_id": candidate, "kappa": kappa}
            for rank, candidate, kappa in m5_family
        ],
        "resolved_candidate_id": chosen_candidate,
        "resolved_selection_rank": chosen_rank,
        "resolved_kappa": chosen_kappa,
        "confirmation_metrics_consulted": False,
    }
    return {
        "block_count": len(EXPECTED_PROFILE_BLOCKS),
        "selected_csv_row_count": len(rows),
        "selection_freeze_promoted_count": len(promoted),
        "selection_freeze_tie_break_choice_count": len(tie_break_choices),
        "blocks": receipts,
    }


def _validate_upstream_manifest(
    raw_hashes: Mapping[str, str],
) -> dict[str, object]:
    profile_root = ROOT / PROFILE_VALIDATION_RELATIVE
    manifest_path = profile_root / "RUN_MANIFEST.json"
    manifest = _load_json(manifest_path)
    expected_links = {
        "assembler_sha256": EXPECTED_ASSEMBLER_SHA256,
        "config_sha256": PROFILE_CONFIG_SHA256,
        "protocol_sha256": PROFILE_PROTOCOL_SHA256,
        "selection_freeze_sha256": next(
            expected
            for name, _, expected, _, _ in PROFILE_INPUT_SPECS
            if name == "profile_selection_freeze"
        ),
    }
    for key, expected in expected_links.items():
        if manifest.get(key) != expected:
            raise PreflightError(f"upstream RUN_MANIFEST {key} mismatch")
    if manifest.get("raw_file_hashes") != dict(raw_hashes):
        raise PreflightError("upstream RUN_MANIFEST raw_file_hashes mismatch")

    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, Mapping):
        raise PreflightError("upstream RUN_MANIFEST artifact_inventory is missing or invalid")
    tree_files = {
        path.relative_to(profile_root).as_posix()
        for path in profile_root.rglob("*")
        if path.is_file()
    }
    listed = {str(path) for path in inventory}
    listed_missing = sorted(listed - tree_files)
    unlisted = tree_files - listed
    if not unlisted:
        raise PreflightError(
            "expected selective upstream manifest inventory is unexpectedly complete; "
            "review the governance classification"
        )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_artifact_inventory_entries": len(listed),
        "current_profile_tree_file_count": len(tree_files),
        "unlisted_current_tree_file_count": len(unlisted),
        "listed_but_missing_count": len(listed_missing),
        "listed_but_missing": listed_missing,
        "classification": "selective_not_complete_tree_inventory",
        "used_as_protection_boundary": False,
        "governance_note": (
            "The upstream RUN_MANIFEST.json artifact_inventory is a curated, "
            "selective artifact inventory rather than a complete recursive file-tree "
            "inventory. It is retained for upstream deliverable provenance only. "
            "The order experiment's protection boundary is the independent complete "
            "recursive protected_hashes baseline captured by this preflight."
        ),
    }


def _protected_targets() -> dict[str, Path]:
    analysis = ROOT / "analysis"
    if not analysis.is_dir():
        raise PreflightError(f"analysis directory is missing: {analysis}")
    targets: dict[str, Path] = {}
    for item in sorted(analysis.iterdir(), key=lambda path: path.name):
        if item.name == WORKSPACE.name:
            continue
        targets[f"analysis/{item.name}"] = item
    for relative in ("docs", "report", "results", "src", *CONTROL_RELATIVES):
        targets[relative] = ROOT / relative
    return targets


def _path_inventory(
    path: Path,
    cache: HashCache | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    if not path.exists():
        return {}, {
            "path": str(path.resolve(strict=False)),
            "exists": False,
            "kind": "missing",
            "file_count": 0,
            "total_bytes": 0,
        }
    if path.is_file():
        before = _stat_fingerprint(path)
        hashes = {path.name: _sha256_file(path, cache)}
        after = _stat_fingerprint(path)
        if before != after:
            raise PreflightError(f"protected file changed during baseline capture: {path}")
        return hashes, {
            "path": str(path.resolve()),
            "exists": True,
            "kind": "file",
            "file_count": 1,
            "total_bytes": path.stat().st_size,
        }
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    before_fingerprints = {
        item.relative_to(path).as_posix(): _stat_fingerprint(item) for item in files
    }
    hashes = {
        item.relative_to(path).as_posix(): _sha256_file(item, cache) for item in files
    }
    after_files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    after_fingerprints = {
        item.relative_to(path).as_posix(): _stat_fingerprint(item) for item in after_files
    }
    if before_fingerprints != after_fingerprints:
        before_names = set(before_fingerprints)
        after_names = set(after_fingerprints)
        raise PreflightError(
            f"protected directory changed during baseline capture: {path}",
            {
                "added": sorted(after_names - before_names),
                "removed": sorted(before_names - after_names),
                "metadata_changed": sorted(
                    name
                    for name in before_names & after_names
                    if before_fingerprints[name] != after_fingerprints[name]
                ),
            },
        )
    return hashes, {
        "path": str(path.resolve()),
        "exists": True,
        "kind": "directory",
        "file_count": len(files),
        "total_bytes": sum(fingerprint[2] for fingerprint in after_fingerprints.values()),
    }


def _capture_protected_baseline(
    cache: HashCache | None = None,
) -> dict[str, object]:
    hashes: dict[str, dict[str, str]] = {}
    roots: dict[str, dict[str, object]] = {}
    for name, path in _protected_targets().items():
        root_hashes, root_state = _path_inventory(path, cache)
        hashes[name] = root_hashes
        roots[name] = root_state
    return {
        "coverage_rule": (
            "all immediate analysis items recursively except "
            "analysis/order_breach_severity_v1; plus docs, report, results, src "
            "and AGENTS.md, PROJECT_CONTEXT.md, RESULTS_REGISTRY.md, DECISION_LOG.md"
        ),
        "excluded_new_workspace": str(WORKSPACE.resolve()),
        "root_count": len(roots),
        "file_count": sum(int(root["file_count"]) for root in roots.values()),
        "total_bytes": sum(int(root["total_bytes"]) for root in roots.values()),
        "roots": roots,
        "hashes": hashes,
        "aggregate_sha256": _stable_object_sha256({"roots": roots, "hashes": hashes}),
    }


def _compare_protected_baselines(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    before_hashes = before.get("hashes")
    after_hashes = after.get("hashes")
    before_roots = before.get("roots")
    after_roots = after.get("roots")
    if not all(
        isinstance(value, Mapping)
        for value in (before_hashes, after_hashes, before_roots, after_roots)
    ):
        raise PreflightError("protected baseline schema is invalid")
    assert isinstance(before_hashes, Mapping)
    assert isinstance(after_hashes, Mapping)
    assert isinstance(before_roots, Mapping)
    assert isinstance(after_roots, Mapping)

    detail: dict[str, object] = {}
    passed = True
    all_roots = sorted(set(before_hashes) | set(after_hashes))
    for root in all_roots:
        old_raw = before_hashes.get(root, {})
        new_raw = after_hashes.get(root, {})
        old = dict(old_raw) if isinstance(old_raw, Mapping) else {}
        new = dict(new_raw) if isinstance(new_raw, Mapping) else {}
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(key for key in set(old) & set(new) if old[key] != new[key])
        old_state = before_roots.get(root)
        new_state = after_roots.get(root)
        state_changed = old_state != new_state
        unchanged = not (added or removed or changed or state_changed)
        passed = passed and unchanged
        detail[root] = {
            "added": added,
            "removed": removed,
            "changed": changed,
            "root_state_changed": state_changed,
            "before_root_state": old_state,
            "after_root_state": new_state,
            "unchanged": unchanged,
        }
    return {
        "passed": passed,
        "checked_at_utc": _utc_now(),
        "before_aggregate_sha256": before.get("aggregate_sha256"),
        "after_aggregate_sha256": after.get("aggregate_sha256"),
        "root_detail": detail,
    }


def _repository_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).strip("\n")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PreflightError(f"git {' '.join(arguments)} failed: {exc}") from exc

    status = run("status", "--porcelain=v1", "-uall")
    return {
        "commit": run("rev-parse", "HEAD").strip(),
        "branch": run("branch", "--show-current").strip(),
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines() if status else [],
    }


def _environment() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _local_run_source_hashes(cache: HashCache | None = None) -> dict[str, str]:
    paths: list[Path] = [
        WORKSPACE / "ORDER_PROTOCOL.md",
        WORKSPACE / "ORDER_FROZEN_CONFIG.json",
        WORKSPACE / "ORDER_FEATURE_DICTIONARY.md",
        WORKSPACE / "EVIDENCE_STATUS.md",
    ]
    paths.extend(
        path for path in sorted((WORKSPACE / "scripts").glob("*.py")) if path.is_file()
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PreflightError(f"local frozen source files missing: {missing}")
    return {
        path.relative_to(WORKSPACE).as_posix(): _sha256_file(path, cache) for path in paths
    }


def _validate_local_frozen_controls(cache: HashCache) -> dict[str, str]:
    observed_config = _sha256_file(CONFIG_PATH, cache)
    if observed_config != EXPECTED_ORDER_CONFIG_SHA256:
        raise PreflightError(
            f"ORDER_FROZEN_CONFIG.json hash mismatch: {observed_config}"
        )
    protocol = WORKSPACE / "ORDER_PROTOCOL.md"
    observed_protocol = _sha256_file(protocol, cache)
    if observed_protocol != EXPECTED_ORDER_PROTOCOL_SHA256:
        raise PreflightError(f"ORDER_PROTOCOL.md hash mismatch: {observed_protocol}")
    return {
        "ORDER_FROZEN_CONFIG.json": observed_config,
        "ORDER_PROTOCOL.md": observed_protocol,
    }


def _invocation_command() -> str:
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    return shlex.join(argv)


def preflight(data_dir: str | Path) -> dict[str, object]:
    """Validate all immutable inputs and write the complete pre-execution receipt.

    Hash mismatches and frozen profile-specification drift are hard failures.
    The expensive protected-tree hash is captured only after the smaller source
    and specification gates pass.  Hashes already calculated for source inputs
    are cached and reused when those files also lie inside a protected tree.
    """

    started = _utc_now()
    cache: HashCache = {}
    data_path = Path(data_dir).expanduser().resolve()
    _write_json(
        PRESTATE_PATH,
        {
            "schema_version": 1,
            "analysis_id": "order_breach_severity_v1",
            "status": "in_progress",
            "started_at_utc": started,
            "command": _invocation_command(),
            "command_working_directory": str(Path.cwd().resolve()),
            "data_dir": str(data_path),
        },
    )
    config = _load_json(CONFIG_PATH)
    if config.get("analysis_id") != "order_breach_severity_v1":
        raise PreflightError("ORDER_FROZEN_CONFIG analysis_id mismatch")
    local_controls = _validate_local_frozen_controls(cache)
    audit_rows, source_errors = _audit_required_sources(data_path, config, cache)
    _write_source_audit(audit_rows)
    if source_errors:
        failure = {
            "analysis_id": "order_breach_severity_v1",
            "status": "failed",
            "started_at_utc": started,
            "failed_at_utc": _utc_now(),
            "failure_stage": "source_input_audit",
            "errors": source_errors,
            "source_input_audit": str(SOURCE_AUDIT_PATH.resolve()),
        }
        _write_json(PRESTATE_PATH, failure)
        raise PreflightError("source input audit failed", source_errors)

    profile_block_audit = _validate_profile_blocks(config)
    raw_hashes = {
        str(row["input_name"]): str(row["actual_sha256"])
        for row in audit_rows
        if row["input_group"] == "raw_olist"
    }
    raw_paths = {
        str(row["input_name"]): str(row["path"])
        for row in audit_rows
        if row["input_group"] == "raw_olist"
    }
    profile_input_hashes = {
        str(row["input_name"]): str(row["actual_sha256"])
        for row in audit_rows
        if row["input_group"] == "profile_validation"
    }
    profile_input_paths = {
        str(row["input_name"]): str(row["path"])
        for row in audit_rows
        if row["input_group"] == "profile_validation"
    }
    assembler_row = next(
        row for row in audit_rows if row["input_group"] == "canonical_assembler"
    )
    manifest_governance = _validate_upstream_manifest(raw_hashes)
    protected_baseline = _capture_protected_baseline(cache)
    source_code_hashes = _local_run_source_hashes(cache)
    protected_hashes = protected_baseline["hashes"]
    control_file_hashes: dict[str, str] = {}
    for relative in CONTROL_RELATIVES:
        root_hashes = protected_hashes.get(relative, {})
        filename = Path(relative).name
        if not isinstance(root_hashes, Mapping) or filename not in root_hashes:
            raise PreflightError(f"required protected control file is missing: {relative}")
        control_file_hashes[relative] = str(root_hashes[filename])
    state: dict[str, object] = {
        "schema_version": 1,
        "analysis_id": "order_breach_severity_v1",
        "status": "passed",
        "started_at_utc": started,
        "captured_at_utc": _utc_now(),
        "command": _invocation_command(),
        "command_working_directory": str(Path.cwd().resolve()),
        "data_dir": str(data_path),
        "assembler_sha256": str(assembler_row["actual_sha256"]),
        "canonical_assembler": {
            "path": str(assembler_row["path"]),
            "sha256": str(assembler_row["actual_sha256"]),
        },
        "raw_file_hashes": raw_hashes,
        "raw_file_paths": raw_paths,
        "profile_input_hashes": profile_input_hashes,
        "profile_input_paths": profile_input_paths,
        "source_input_audit": {
            "path": str(SOURCE_AUDIT_PATH.resolve()),
            "sha256": _sha256_file(SOURCE_AUDIT_PATH, cache),
            "row_count": len(audit_rows),
            "verified_row_count": sum(
                row["status"] == "verified" for row in audit_rows
            ),
        },
        "local_frozen_controls": local_controls,
        "control_file_hashes": control_file_hashes,
        "source_code_hashes": source_code_hashes,
        "repository": _repository_state(),
        "environment": _environment(),
        "profile_block_audit": profile_block_audit,
        "manifest_inventory_governance": manifest_governance,
        "protected_baseline": protected_baseline,
        # Compatibility alias for simple consumers; this is the same complete map.
        "protected_hashes": protected_hashes,
    }
    _write_json(PRESTATE_PATH, state)
    return state


def verify_protected_unchanged(
    prestate: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Re-hash the complete protected scope and hard-stop on any drift.

    ``prestate`` may be the dictionary returned by :func:`preflight` or the
    path to its JSON receipt.  Local frozen controls/source code are checked in
    addition to the external protected scope; mutable run outputs remain
    excluded because the complete order workspace is intentionally excluded.
    """

    if isinstance(prestate, (str, Path)):
        state = _load_json(Path(prestate))
    else:
        state = dict(prestate)
    if state.get("status") != "passed":
        raise PreflightError("cannot verify a prestate that did not pass")
    before = state.get("protected_baseline")
    if not isinstance(before, Mapping):
        raise PreflightError("prestate protected_baseline is missing or invalid")
    after = _capture_protected_baseline()
    audit = _compare_protected_baselines(before, after)

    expected_sources = state.get("source_code_hashes")
    expected_controls = state.get("local_frozen_controls")
    current_sources = _local_run_source_hashes()
    current_controls = _validate_local_frozen_controls({})
    audit["local_source_code_unchanged"] = expected_sources == current_sources
    audit["local_frozen_controls_unchanged"] = expected_controls == current_controls
    audit["passed"] = bool(audit["passed"])
    audit["passed"] = (
        audit["passed"]
        and audit["local_source_code_unchanged"]
        and audit["local_frozen_controls_unchanged"]
    )
    if not audit["passed"]:
        raise PreflightError("protected paths changed since preflight", audit)
    return audit


__all__ = [
    "PRESTATE_PATH",
    "SOURCE_AUDIT_PATH",
    "PreflightError",
    "preflight",
    "verify_protected_unchanged",
]
