"""Frozen historical-profile joins for the order-level experiment.

This module consumes the already executed public profile artifacts.  It does
not rebuild, tune, rank, or select profiles.  Every order is joined to the
snapshot at its normalised purchase date.  Valid entities that are absent from
that snapshot receive the frozen global-parent score; an invalid entity mapping
is retained and audited separately.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

PROFILE_BLOCKS = ("S1", "S2", "R1", "R2", "M5_ENDPOINT")
PROFILE_PREFIXES = {
    "S1": "S1",
    "S2": "S2",
    "R1": "R1",
    "R2": "R2",
    "M5_ENDPOINT": "M5",
}
SELLER_PROFILE_BLOCKS = ("S1", "S2")
ROUTE_PROFILE_BLOCKS = ("R1", "R2", "M5_ENDPOINT")

PROFILE_PAYLOAD_FIELDS = (
    "score",
    "log1p_support",
    "cold_start",
    "posterior_se",
    "freshness_days",
)
PROFILE_AUDIT_FIELDS = ("mapping_status", "last_mature_outcome_date")
PROFILE_OPTIONAL_SUPPORT_FIELD = "support"

PROFILE_FEATURE_COLUMNS = tuple(
    f"{PROFILE_PREFIXES[block]}_{field}"
    for block in PROFILE_BLOCKS
    for field in PROFILE_PAYLOAD_FIELDS
)
PROFILE_BLOCK_FEATURE_COLUMNS = {
    block: tuple(f"{PROFILE_PREFIXES[block]}_{field}" for field in PROFILE_PAYLOAD_FIELDS)
    for block in PROFILE_BLOCKS
}
SELLER_PROFILE_FEATURE_COLUMNS = tuple(
    column for block in SELLER_PROFILE_BLOCKS for column in PROFILE_BLOCK_FEATURE_COLUMNS[block]
)
ROUTE_PROFILE_FEATURE_COLUMNS = tuple(
    column for block in ("R1", "R2") for column in PROFILE_BLOCK_FEATURE_COLUMNS[block]
)
M5_ENDPOINT_FEATURE_COLUMNS = PROFILE_BLOCK_FEATURE_COLUMNS["M5_ENDPOINT"]
PROFILE_JOIN_COLUMNS = tuple(
    column
    for block in PROFILE_BLOCKS
    for column in (
        *(f"{PROFILE_PREFIXES[block]}_{field}" for field in PROFILE_PAYLOAD_FIELDS),
        *(f"{PROFILE_PREFIXES[block]}_{field}" for field in PROFILE_AUDIT_FIELDS),
        f"{PROFILE_PREFIXES[block]}_{PROFILE_OPTIONAL_SUPPORT_FIELD}",
    )
)

PROFILE_SNAPSHOT_DATE_MIN = "2016-12-03"
PROFILE_SNAPSHOT_DATE_MAX = "2018-08-30"
PROFILE_SNAPSHOT_DATE_COUNT = 636
PROFILE_DAILY_TOTAL_ROWS = 4_545_295
PROFILE_PARENT_TOTAL_ROWS = 10_176

PROFILE_DAILY_SHA256 = "ff3c3f19982714087b9309e03fb35d99bf9039a574b819afa2b6b544e330b56c"
PROFILE_PARENT_SHA256 = "da0ae9431165f8bc635f6805ef89d79ef8bf28802aed53c50020596415593aa4"
PROFILE_SELECTION_FREEZE_SHA256 = (
    "f2409082543bca174c13a2ba94481d2d03c7413021232678ff139438ece69742"
)
PROFILE_SELECTED_CANDIDATES_SHA256 = (
    "b8a4cb4b71a09493c9db5fa8da5248078e2906ceb0c7faad46a6770433358659"
)

EXPECTED_PROFILE_HASHES = {
    "PROFILE_DAILY_SCORES.csv.gz": PROFILE_DAILY_SHA256,
    "PROFILE_PARENT_STRUCTURE.csv": PROFILE_PARENT_SHA256,
    "PROFILE_SELECTION_FREEZE.json": PROFILE_SELECTION_FREEZE_SHA256,
    "PROFILE_SELECTED_CANDIDATES.csv": PROFILE_SELECTED_CANDIDATES_SHA256,
}

# These are the five fixed V1.1 choices.  ``selection_rank`` is the persisted
# development rank within the relevant target/granularity comparison.  In
# particular, the M5 representative is rank 1 without consulting confirmation.
FROZEN_PROFILE_SPECS: dict[str, dict[str, object]] = {
    "S1": {
        "name": "seller_handling_level",
        "prefix": "S1",
        "candidate_id": (
            "handling_level|seller_id|C|w90|l14|P1|parent=global|"
            "kappa=na|min_support=5"
        ),
        "base_candidate_id": (
            "handling_level|seller_id|C|w90|l14|P1|parent=global|kappa=na"
        ),
        "profile_spec_id": "ps_18f6d18af885ac9c1930",
        "target": "handling_level",
        "entity": "seller_id",
        "scheme": "C",
        "window_days": 90,
        "lag_days": 14,
        "estimator": "P1",
        "parent": "global",
        "kappa": None,
        "min_support": 5,
        "selection_rank": 2,
        "expected_daily_rows": 611_625,
    },
    "S2": {
        "name": "seller_handling_tail",
        "prefix": "S2",
        "candidate_id": (
            "handling_tail|seller_id|A|w90|l0|P1|parent=global|"
            "kappa=10|min_support=5"
        ),
        "base_candidate_id": (
            "handling_tail|seller_id|A|w90|l0|P1|parent=global|kappa=10"
        ),
        "profile_spec_id": "ps_29c28f8f40eed03c1031",
        "target": "handling_tail",
        "entity": "seller_id",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 10,
        "min_support": 5,
        "selection_rank": 2,
        "expected_daily_rows": 633_032,
    },
    "R1": {
        "name": "route_transit_level",
        "prefix": "R1",
        "candidate_id": (
            "transit_level|state_od|A|w90|l0|P0|parent=global|"
            "kappa=na|min_support=5"
        ),
        "base_candidate_id": (
            "transit_level|state_od|A|w90|l0|P0|parent=global|kappa=na"
        ),
        "profile_spec_id": "ps_18f16966ac00ff520226",
        "target": "transit_level",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P0",
        "parent": "global",
        "kappa": None,
        "min_support": 5,
        "selection_rank": 1,
        "expected_daily_rows": 143_593,
    },
    "R2": {
        "name": "route_transit_tail",
        "prefix": "R2",
        "candidate_id": (
            "transit_tail|state_od|A|w90|l0|P1|parent=global|"
            "kappa=10|min_support=5"
        ),
        "base_candidate_id": (
            "transit_tail|state_od|A|w90|l0|P1|parent=global|kappa=10"
        ),
        "profile_spec_id": "ps_9799491505b2347220fb",
        "target": "transit_tail",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 10,
        "min_support": 5,
        "selection_rank": 2,
        "expected_daily_rows": 143_593,
    },
    "M5_ENDPOINT": {
        "name": "route_historical_final_breach",
        "prefix": "M5",
        "candidate_id": (
            "final_breach|state_od|A|w90|l0|P1|parent=global|"
            "kappa=100|min_support=5"
        ),
        "base_candidate_id": (
            "final_breach|state_od|A|w90|l0|P1|parent=global|kappa=100"
        ),
        "profile_spec_id": "ps_ef5d05dc7c0496cca415",
        "target": "final_breach",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 100,
        "min_support": 5,
        "selection_rank": 1,
        "expected_daily_rows": 143_593,
    },
}

DAILY_PRIMARY_KEY = ("candidate_id", "snapshot_date", "entity_id")
PARENT_PRIMARY_KEY = ("base_candidate_id", "snapshot_date", "parent_id")
DAILY_REQUIRED_COLUMNS = (
    "entity_id",
    "snapshot_date",
    "target",
    "granularity",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent_structure",
    "kappa",
    "base_candidate_id",
    "parent_id",
    "score",
    "support",
    "posterior_se",
    "cold_start",
    "profile_freshness_days",
    "last_mature_outcome_date",
    "candidate_id",
    "profile_spec_id",
    "min_support",
)
PARENT_REQUIRED_COLUMNS = (
    "base_candidate_id",
    "snapshot_date",
    "target",
    "granularity",
    "parent_structure",
    "parent_id",
    "parent_score",
    "global_score",
    "parent_supported",
    "valid",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _coerce_config(config: Mapping[str, object] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _parse_iso_dates(values: pd.Series, label: str, *, allow_missing: bool = False) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
    if not allow_missing and parsed.isna().any():
        raise AssertionError(f"{label} contains missing dates")
    present = parsed.notna()
    canonical = parsed.loc[present].dt.strftime("%Y-%m-%d")
    raw = values.loc[present].astype(str)
    if not canonical.eq(raw).all():
        raise AssertionError(f"{label} contains non-canonical ISO calendar dates")
    return parsed


def _bool_values(values: pd.Series, label: str) -> pd.Series:
    if values.isna().any():
        raise AssertionError(f"{label} contains missing booleans")
    normalised = values.astype(str).str.strip().str.lower()
    if not normalised.isin({"true", "false", "1", "0"}).all():
        raise AssertionError(f"{label} contains invalid boolean values")
    return normalised.isin({"true", "1"})


def _assert_string_constant(frame: pd.DataFrame, column: str, expected: object, label: str) -> None:
    values = frame[column]
    if values.isna().any() or not values.astype(str).eq(str(expected)).all():
        raise AssertionError(f"{label}: {column} differs from frozen value {expected!r}")


def _assert_numeric_constant(frame: pd.DataFrame, column: str, expected: object, label: str) -> None:
    values = pd.to_numeric(frame[column], errors="raise")
    if expected is None:
        if values.notna().any():
            raise AssertionError(f"{label}: {column} must be missing for the frozen specification")
    elif values.isna().any() or not values.eq(float(expected)).all():
        raise AssertionError(f"{label}: {column} differs from frozen value {expected!r}")


def _validate_config(config: Mapping[str, Any]) -> dict[str, Path]:
    if config.get("analysis_id") != "order_breach_severity_v1":
        raise AssertionError("wrong or missing order analysis_id")
    data = config.get("data")
    profiles = config.get("profiles")
    population = config.get("population")
    if not isinstance(data, Mapping) or not isinstance(profiles, Mapping):
        raise KeyError("frozen config requires data and profiles mappings")
    if not isinstance(population, Mapping):
        raise KeyError("frozen config requires a population mapping")

    if tuple(profiles.get("feature_payload", ())) != PROFILE_PAYLOAD_FIELDS:
        raise AssertionError("profile feature payload differs from the frozen five-field payload")
    if profiles.get("primary_level_feature_allowed") is not False:
        raise AssertionError("primary level features must remain disabled")
    if profiles.get("raw_and_eb_together_allowed") is not False:
        raise AssertionError("raw and EB profile scores must not be combined")

    config_fields = (
        "name",
        "candidate_id",
        "base_candidate_id",
        "profile_spec_id",
        "entity",
        "scheme",
        "window_days",
        "lag_days",
        "estimator",
        "parent",
        "kappa",
        "min_support",
    )
    for block in PROFILE_BLOCKS:
        actual = profiles.get(block)
        if not isinstance(actual, Mapping):
            raise KeyError(f"frozen config is missing profile block {block}")
        expected = FROZEN_PROFILE_SPECS[block]
        for field in config_fields:
            if actual.get(field) != expected[field]:
                raise AssertionError(
                    f"config {block}.{field}={actual.get(field)!r}; expected {expected[field]!r}"
                )
    endpoint = profiles["M5_ENDPOINT"]
    if endpoint.get("representative_rule") != (
        "lowest_frozen_development_selection_rank_then_lexical_candidate_id"
    ):
        raise AssertionError("M5 endpoint representative rule is not the frozen development rule")
    if endpoint.get("frozen_selection_rank") != 1:
        raise AssertionError("M5 endpoint is not frozen development rank 1")

    expected_rules = {
        "profile_snapshot_time": "purchase_calendar_date_midnight",
        "profile_snapshot_rule": "snapshot_date_equals_normalized_order_purchase_date",
        "profile_history_rule": "label_available_at_strictly_less_than_snapshot_midnight",
    }
    for key, expected in expected_rules.items():
        if data.get(key) != expected:
            raise AssertionError(f"config data.{key} differs from the frozen rule")
    if population.get("multi_seller_rule") != (
        "deterministic_modal_main_seller_then_lexical_tie_break"
    ):
        raise AssertionError("multi-seller entity rule differs from the frozen deterministic rule")

    paths = {
        "daily": _resolve_path(data["profile_daily_input"]),
        "parent": _resolve_path(data["profile_parent_input"]),
        "freeze": _resolve_path(data["profile_selection_freeze"]),
        "selected": _resolve_path(data["profile_selected_candidates"]),
    }
    expected_paths = {
        "daily": ROOT / "analysis/dynamic_profile_profile_validation_v1/PROFILE_DAILY_SCORES.csv.gz",
        "parent": ROOT / "analysis/dynamic_profile_profile_validation_v1/PROFILE_PARENT_STRUCTURE.csv",
        "freeze": ROOT / "analysis/dynamic_profile_profile_validation_v1/PROFILE_SELECTION_FREEZE.json",
        "selected": ROOT / "analysis/dynamic_profile_profile_validation_v1/PROFILE_SELECTED_CANDIDATES.csv",
    }
    for name, path in paths.items():
        if path != expected_paths[name].resolve():
            raise AssertionError(f"{name} must use the frozen public artifact: {expected_paths[name]}")
        if not path.is_file():
            raise FileNotFoundError(path)
    if data.get("profile_selection_freeze_sha256") != PROFILE_SELECTION_FREEZE_SHA256:
        raise AssertionError("config selection-freeze SHA-256 differs from the frozen hash")
    return paths


def _validate_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    expected = {
        "daily": PROFILE_DAILY_SHA256,
        "parent": PROFILE_PARENT_SHA256,
        "freeze": PROFILE_SELECTION_FREEZE_SHA256,
        "selected": PROFILE_SELECTED_CANDIDATES_SHA256,
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        digest = _sha256_file(path)
        actual[name] = digest
        if digest != expected[name]:
            raise RuntimeError(f"frozen {name} artifact hash mismatch: {digest} != {expected[name]}")
    return actual


def _validate_selection_controls(paths: Mapping[str, Path]) -> None:
    selected = pd.read_csv(paths["selected"])
    required = (
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
    )
    _require_columns(selected, required, "PROFILE_SELECTED_CANDIDATES.csv")
    if selected["candidate_id"].isna().any() or selected["candidate_id"].duplicated().any():
        raise AssertionError("selected-candidate candidate_id is not a primary key")

    by_candidate = selected.set_index("candidate_id", drop=False)
    for block, spec in FROZEN_PROFILE_SPECS.items():
        candidate_id = str(spec["candidate_id"])
        if candidate_id not in by_candidate.index:
            raise AssertionError(f"{block} is absent from the frozen selected-candidate artifact")
        row = by_candidate.loc[candidate_id]
        if isinstance(row, pd.DataFrame):
            raise AssertionError(f"duplicate frozen selected-candidate row for {block}")
        string_expectations = {
            "base_candidate_id": spec["base_candidate_id"],
            "profile_spec_id": spec["profile_spec_id"],
            "target": spec["target"],
            "granularity": spec["entity"],
            "scheme": spec["scheme"],
            "estimator": spec["estimator"],
            "parent_structure": spec["parent"],
            "selection_decision": "selected",
        }
        for field, expected in string_expectations.items():
            if pd.isna(row[field]) or str(row[field]) != str(expected):
                raise AssertionError(f"selected-candidate {block}.{field} differs from freeze")
        numeric_expectations = {
            "window_days": spec["window_days"],
            "lag_days": spec["lag_days"],
            "min_support": spec["min_support"],
            "selection_rank": spec["selection_rank"],
        }
        for field, expected in numeric_expectations.items():
            if pd.isna(row[field]) or float(row[field]) != float(expected):
                raise AssertionError(f"selected-candidate {block}.{field} differs from freeze")
        if spec["kappa"] is None:
            if pd.notna(row["kappa"]):
                raise AssertionError(f"selected-candidate {block}.kappa must be missing")
        elif pd.isna(row["kappa"]) or float(row["kappa"]) != float(spec["kappa"]):
            raise AssertionError(f"selected-candidate {block}.kappa differs from freeze")

    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    if freeze.get("confirmation_outcomes_accessed") is not False:
        raise AssertionError("selection freeze is not the pre-confirmation development freeze")
    if freeze.get("development_purchase_end_exclusive") != "2018-01-01":
        raise AssertionError("selection freeze has the wrong development endpoint")
    promoted = freeze.get("promoted_candidates")
    if not isinstance(promoted, list):
        raise AssertionError("selection freeze has no promoted-candidate list")
    promoted_by_id = {str(row.get("candidate_id")): row for row in promoted}
    if len(promoted_by_id) != len(promoted):
        raise AssertionError("selection freeze contains duplicate promoted candidate IDs")
    for block, spec in FROZEN_PROFILE_SPECS.items():
        row = promoted_by_id.get(str(spec["candidate_id"]))
        if row is None:
            raise AssertionError(f"{block} is absent from the pre-confirmation selection freeze")
        for field in ("base_candidate_id", "profile_spec_id"):
            if row.get(field) != spec[field]:
                raise AssertionError(f"selection-freeze {block}.{field} differs from config")
        if int(row.get("selection_rank", -1)) != int(spec["selection_rank"]):
            raise AssertionError(f"selection-freeze {block}.selection_rank differs from config")


def _deterministic_entities(frame: pd.DataFrame) -> dict[str, pd.Series]:
    _require_columns(
        frame,
        ("main_seller_id", "main_seller_state", "customer_state"),
        "canonical/order frame",
    )
    seller = frame["main_seller_id"].astype("string")
    state_od = (
        frame["main_seller_state"].astype("string")
        + " -> "
        + frame["customer_state"].astype("string")
    )
    seller = seller.mask(seller.str.contains("<NA>", na=True), pd.NA)
    state_od = state_od.mask(state_od.str.contains("<NA>", na=True), pd.NA)

    # If aliases from the V1.1 all-placed frame are present, they must agree
    # with the deterministic canonical construction; they are never trusted as
    # a second mapping source.
    for column, derived in (("seller_id", seller), ("state_od", state_od)):
        if column not in frame.columns:
            continue
        supplied = frame[column].astype("string")
        supplied = supplied.mask(supplied.str.contains("<NA>", na=True), pd.NA)
        equal = supplied.fillna("__MISSING_ENTITY__").eq(
            derived.fillna("__MISSING_ENTITY__")
        )
        if not equal.all():
            raise AssertionError(f"supplied {column} differs from the deterministic entity key")
    return {"seller_id": seller, "state_od": state_od}


def _key_index(snapshot: pd.Series, entity: pd.Series) -> pd.MultiIndex:
    valid = entity.notna()
    if not valid.any():
        return pd.MultiIndex.from_arrays([[], []], names=["snapshot_date", "entity_id"])
    return pd.MultiIndex.from_arrays(
        [snapshot.loc[valid].to_numpy(), entity.loc[valid].astype(str).to_numpy()],
        names=["snapshot_date", "entity_id"],
    ).unique()


def _validate_daily_static(frame: pd.DataFrame, block: str) -> tuple[pd.Series, pd.Series]:
    spec = FROZEN_PROFILE_SPECS[block]
    label = f"PROFILE_DAILY_SCORES[{block}]"
    string_expectations = {
        "candidate_id": spec["candidate_id"],
        "base_candidate_id": spec["base_candidate_id"],
        "profile_spec_id": spec["profile_spec_id"],
        "target": spec["target"],
        "granularity": spec["entity"],
        "scheme": spec["scheme"],
        "estimator": spec["estimator"],
        "parent_structure": spec["parent"],
        "parent_id": "__GLOBAL__",
    }
    for field, expected in string_expectations.items():
        _assert_string_constant(frame, field, expected, label)
    for field, expected in (
        ("window_days", spec["window_days"]),
        ("lag_days", spec["lag_days"]),
        ("kappa", spec["kappa"]),
        ("min_support", spec["min_support"]),
    ):
        _assert_numeric_constant(frame, field, expected, label)

    snapshot = _parse_iso_dates(frame["snapshot_date"], f"{label}.snapshot_date")
    last_mature = _parse_iso_dates(
        frame["last_mature_outcome_date"], f"{label}.last_mature_outcome_date"
    )
    if not last_mature.lt(snapshot).all():
        raise AssertionError(f"{label} contains last_mature_outcome_date >= snapshot_date")

    score = pd.to_numeric(frame["score"], errors="raise")
    support = pd.to_numeric(frame["support"], errors="raise")
    freshness = pd.to_numeric(frame["profile_freshness_days"], errors="raise")
    if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
        raise AssertionError(f"{label} contains non-finite scores")
    if support.isna().any() or support.lt(0).any() or not support.mod(1).eq(0).all():
        raise AssertionError(f"{label} contains invalid support")
    expected_freshness = (snapshot - last_mature).dt.days
    if freshness.isna().any() or not freshness.eq(expected_freshness).all():
        raise AssertionError(f"{label} freshness differs from snapshot minus last maturity")
    if _bool_values(frame["cold_start"], f"{label}.cold_start").any():
        raise AssertionError(f"{label} persisted entity rows must not be cold-start rows")
    posterior = pd.to_numeric(frame["posterior_se"], errors="raise")
    finite_posterior = posterior.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_posterior).all() or (finite_posterior < 0).any():
        raise AssertionError(f"{label} contains invalid posterior standard errors")
    return snapshot, last_mature


def _load_daily_profiles(
    path: Path,
    order_keys: Mapping[str, pd.MultiIndex],
    *,
    chunksize: int = 250_000,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    candidate_to_block = {
        str(spec["candidate_id"]): block for block, spec in FROZEN_PROFILE_SPECS.items()
    }
    selected_ids = set(candidate_to_block)
    retained: dict[str, list[pd.DataFrame]] = {block: [] for block in PROFILE_BLOCKS}
    row_counts = {block: 0 for block in PROFILE_BLOCKS}
    dates: dict[str, set[pd.Timestamp]] = {block: set() for block in PROFILE_BLOCKS}
    previous_key: tuple[str, str, str] | None = None
    source_rows = 0

    reader = pd.read_csv(
        path,
        usecols=list(DAILY_REQUIRED_COLUMNS),
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        source_rows += len(chunk)
        if chunk[list(DAILY_PRIMARY_KEY)].isna().any().any():
            raise AssertionError("daily profile primary key contains nulls")
        all_keys = pd.MultiIndex.from_frame(chunk.loc[:, list(DAILY_PRIMARY_KEY)].astype(str))
        if all_keys.has_duplicates:
            raise AssertionError("daily profile primary key is duplicated within a chunk")
        if not all_keys.is_monotonic_increasing:
            raise AssertionError("daily profile primary key is not in frozen sort order")
        first_key = tuple(map(str, chunk.iloc[0][list(DAILY_PRIMARY_KEY)]))
        last_key = tuple(map(str, chunk.iloc[-1][list(DAILY_PRIMARY_KEY)]))
        if previous_key is not None and first_key <= previous_key:
            raise AssertionError("daily profile primary key is duplicated or out of order")
        previous_key = last_key

        selected = chunk.loc[chunk["candidate_id"].isin(selected_ids)].copy()
        if selected.empty:
            continue

        for candidate_id, candidate_rows in selected.groupby("candidate_id", sort=False):
            block = candidate_to_block[str(candidate_id)]
            spec = FROZEN_PROFILE_SPECS[block]
            candidate_rows = candidate_rows.copy()
            snapshot, last_mature = _validate_daily_static(candidate_rows, block)
            candidate_rows["snapshot_date"] = snapshot
            candidate_rows["last_mature_outcome_date"] = last_mature
            row_counts[block] += len(candidate_rows)
            dates[block].update(snapshot.unique())

            candidate_index = pd.MultiIndex.from_arrays(
                [snapshot.to_numpy(), candidate_rows["entity_id"].astype(str).to_numpy()],
                names=["snapshot_date", "entity_id"],
            )
            relevant = candidate_index.isin(order_keys[str(spec["entity"])])
            if relevant.any():
                keep = candidate_rows.loc[
                    relevant,
                    [
                        "snapshot_date",
                        "entity_id",
                        "score",
                        "support",
                        "posterior_se",
                        "profile_freshness_days",
                        "last_mature_outcome_date",
                    ],
                ].copy()
                retained[block].append(keep)

    expected_date_index = pd.date_range(
        PROFILE_SNAPSHOT_DATE_MIN, PROFILE_SNAPSHOT_DATE_MAX, freq="D"
    )
    if len(expected_date_index) != PROFILE_SNAPSHOT_DATE_COUNT:
        raise AssertionError("internal frozen profile date-count constant is inconsistent")
    if source_rows != PROFILE_DAILY_TOTAL_ROWS:
        raise AssertionError(
            f"public daily profile has {source_rows} rows; expected {PROFILE_DAILY_TOTAL_ROWS}"
        )

    tables: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict[str, object]] = {}
    for block, spec in FROZEN_PROFILE_SPECS.items():
        if row_counts[block] != int(spec["expected_daily_rows"]):
            raise AssertionError(
                f"{block} daily rows={row_counts[block]}; expected={spec['expected_daily_rows']}"
            )
        actual_dates = pd.DatetimeIndex(sorted(dates[block]))
        if not actual_dates.equals(expected_date_index):
            raise AssertionError(f"{block} does not contain the exact frozen 636 snapshot dates")
        if retained[block]:
            table = pd.concat(retained[block], ignore_index=True)
            table["entity_id"] = table["entity_id"].astype(str)
            if table.duplicated(["snapshot_date", "entity_id"]).any():
                raise AssertionError(f"{block} retained join keys are duplicated")
            table = table.sort_values(
                ["snapshot_date", "entity_id"], kind="mergesort"
            ).reset_index(drop=True)
        else:
            table = pd.DataFrame(
                columns=(
                    "snapshot_date",
                    "entity_id",
                    "score",
                    "support",
                    "posterior_se",
                    "profile_freshness_days",
                    "last_mature_outcome_date",
                )
            )
        tables[block] = table
        audit[block] = {
            "candidate_id": spec["candidate_id"],
            "base_candidate_id": spec["base_candidate_id"],
            "profile_spec_id": spec["profile_spec_id"],
            "source_rows": row_counts[block],
            "source_snapshot_date_count": len(actual_dates),
            "source_snapshot_date_min": actual_dates.min().strftime("%Y-%m-%d"),
            "source_snapshot_date_max": actual_dates.max().strftime("%Y-%m-%d"),
            "retained_distinct_order_join_keys": len(table),
        }
    return tables, audit


def _load_parent_scores(path: Path) -> tuple[dict[str, pd.Series], dict[str, object]]:
    parent = pd.read_csv(path)
    _require_columns(parent, PARENT_REQUIRED_COLUMNS, "PROFILE_PARENT_STRUCTURE.csv")
    if len(parent) != PROFILE_PARENT_TOTAL_ROWS:
        raise AssertionError(
            f"public parent structure has {len(parent)} rows; expected {PROFILE_PARENT_TOTAL_ROWS}"
        )
    if parent[list(PARENT_PRIMARY_KEY)].isna().any().any():
        raise AssertionError("parent-structure primary key contains nulls")
    if parent.duplicated(list(PARENT_PRIMARY_KEY)).any():
        raise AssertionError("parent-structure primary key is duplicated")
    parent["snapshot_date"] = _parse_iso_dates(
        parent["snapshot_date"], "PROFILE_PARENT_STRUCTURE.snapshot_date"
    )
    expected_dates = pd.date_range(PROFILE_SNAPSHOT_DATE_MIN, PROFILE_SNAPSHOT_DATE_MAX)
    scores: dict[str, pd.Series] = {}
    for block, spec in FROZEN_PROFILE_SPECS.items():
        rows = parent.loc[parent["base_candidate_id"].eq(spec["base_candidate_id"])].copy()
        if len(rows) != PROFILE_SNAPSHOT_DATE_COUNT:
            raise AssertionError(f"{block} parent rows={len(rows)}; expected 636")
        for field, expected in (
            ("base_candidate_id", spec["base_candidate_id"]),
            ("target", spec["target"]),
            ("granularity", spec["entity"]),
            ("parent_structure", "global"),
            ("parent_id", "__GLOBAL__"),
        ):
            _assert_string_constant(rows, field, expected, f"parent[{block}]")
        if not pd.DatetimeIndex(rows["snapshot_date"].sort_values().unique()).equals(
            expected_dates
        ):
            raise AssertionError(f"{block} parent rows do not contain the exact 636 dates")
        if not _bool_values(rows["valid"], f"parent[{block}].valid").all():
            raise AssertionError(f"{block} has an invalid parent row")
        if not _bool_values(
            rows["parent_supported"], f"parent[{block}].parent_supported"
        ).all():
            raise AssertionError(f"{block} has an unsupported frozen global parent row")
        parent_score = pd.to_numeric(rows["parent_score"], errors="raise")
        global_score = pd.to_numeric(rows["global_score"], errors="raise")
        if (
            parent_score.isna().any()
            or global_score.isna().any()
            or not np.isfinite(parent_score.to_numpy(dtype=float)).all()
            or not np.isfinite(global_score.to_numpy(dtype=float)).all()
            or not np.allclose(
                parent_score.to_numpy(dtype=float),
                global_score.to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            )
        ):
            raise AssertionError(f"{block} has invalid or inconsistent global-parent scores")
        score_series = pd.Series(
            parent_score.to_numpy(dtype=float),
            index=rows["snapshot_date"].to_numpy(),
            name="parent_score",
        )
        if score_series.index.has_duplicates:
            raise AssertionError(f"{block} parent snapshot dates are duplicated")
        scores[block] = score_series.sort_index()
    return scores, {
        "source_rows": len(parent),
        "primary_key_columns": list(PARENT_PRIMARY_KEY),
        "primary_key_valid": True,
        "selected_parent_rows": PROFILE_SNAPSHOT_DATE_COUNT * len(PROFILE_BLOCKS),
        "selected_parent_id": "__GLOBAL__",
    }


def _join_one_block(
    result: pd.DataFrame,
    snapshot: pd.Series,
    entity: pd.Series,
    profile: pd.DataFrame,
    parent_score: pd.Series,
    block: str,
) -> dict[str, object]:
    prefix = PROFILE_PREFIXES[block]
    mapping_valid = entity.notna().to_numpy(dtype=bool)
    safe_entity = entity.astype("string").fillna("__MISSING_ENTITY__")
    keys = pd.MultiIndex.from_arrays(
        [snapshot.to_numpy(), safe_entity.astype(str).to_numpy()],
        names=["snapshot_date", "entity_id"],
    )
    lookup = profile.copy()
    lookup["_source_present"] = True
    lookup = lookup.set_index(["snapshot_date", "entity_id"], verify_integrity=True)
    matched = lookup.reindex(keys)
    seen = mapping_valid & matched["_source_present"].notna().to_numpy(dtype=bool)

    fallback = snapshot.map(parent_score)
    if fallback.isna().any() or not np.isfinite(fallback.to_numpy(dtype=float)).all():
        raise AssertionError(f"{block} has no valid global-parent fallback for an order date")
    entity_score = pd.to_numeric(matched["score"], errors="coerce").to_numpy(dtype=float)
    score = np.where(seen, entity_score, fallback.to_numpy(dtype=float))
    if not np.isfinite(score).all():
        raise AssertionError(f"{block} joined score is not finite for every retained order")

    matched_support = pd.to_numeric(matched["support"], errors="coerce").fillna(0)
    support = np.where(seen, matched_support.to_numpy(dtype=float), 0).astype("int64")
    posterior = pd.to_numeric(matched["posterior_se"], errors="coerce").to_numpy(dtype=float)
    posterior = np.where(seen, posterior, np.nan)
    freshness = pd.to_numeric(
        matched["profile_freshness_days"], errors="coerce"
    ).to_numpy(dtype=float)
    freshness = np.where(seen, freshness, np.nan)
    last_mature = pd.to_datetime(matched["last_mature_outcome_date"], errors="coerce")
    last_mature = pd.Series(last_mature.to_numpy(), index=result.index).where(seen)

    mapping_status = np.select(
        [~mapping_valid, seen],
        ["missing_mapping", "seen"],
        default="mapped_cold_start",
    )
    cold_start = mapping_status == "mapped_cold_start"
    if cold_start[~mapping_valid].any():
        raise AssertionError(f"{block} missing mappings were conflated with cold starts")
    if seen.any():
        seen_last = last_mature.loc[seen]
        seen_snapshot = snapshot.loc[seen]
        if seen_last.isna().any() or not seen_last.lt(seen_snapshot).all():
            raise AssertionError(f"{block} seen rows violate strict pre-snapshot maturity")

    result[f"{prefix}_score"] = score
    result[f"{prefix}_log1p_support"] = np.log1p(support.astype(float))
    result[f"{prefix}_cold_start"] = cold_start
    result[f"{prefix}_posterior_se"] = posterior
    result[f"{prefix}_freshness_days"] = freshness
    result[f"{prefix}_mapping_status"] = mapping_status
    result[f"{prefix}_last_mature_outcome_date"] = last_mature.to_numpy()
    result[f"{prefix}_support"] = support

    counts = pd.Series(mapping_status).value_counts()
    return {
        "orders_seen": int(counts.get("seen", 0)),
        "orders_mapped_cold_start": int(counts.get("mapped_cold_start", 0)),
        "orders_missing_mapping": int(counts.get("missing_mapping", 0)),
        "orders_global_parent_fallback": int((mapping_status != "seen").sum()),
        "seen_fraction": float(seen.mean()) if len(seen) else np.nan,
        "mapping_valid_fraction": float(mapping_valid.mean()) if len(mapping_valid) else np.nan,
        "joined_score_missing": int(pd.isna(score).sum()),
        "seen_history_time_violations": 0,
    }


def _order_id_hash(order_ids: pd.Series) -> str:
    values = sorted(order_ids.astype(str))
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def join_profiles(
    frame: pd.DataFrame,
    config: Mapping[str, object] | str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach the five frozen purchase-date profile blocks to an order frame.

    The input may be any non-empty subset of the frozen 2017-04-01 through
    2018-08-30 canonical delivered population, but it must remain one row per
    non-missing ``order_id``.  The returned frame preserves row order, index,
    and every input column.  The audit dictionary is JSON-serialisable.
    """

    frozen = _coerce_config(config)
    paths = _validate_config(frozen)
    hashes = _validate_hashes(paths)
    _validate_selection_controls(paths)

    _require_columns(frame, ("order_id", "order_purchase_timestamp"), "canonical/order frame")
    if frame.empty:
        raise ValueError("profile join requires at least one order")
    if frame["order_id"].isna().any() or frame["order_id"].duplicated().any():
        raise AssertionError("profile join input must be one row per non-missing order_id")
    collisions = sorted(set(PROFILE_JOIN_COLUMNS) & set(frame.columns))
    if collisions:
        raise ValueError(f"profile output columns already exist: {collisions}")

    purchase = pd.to_datetime(frame["order_purchase_timestamp"], errors="raise")
    if purchase.isna().any():
        raise AssertionError("profile join input contains missing purchase timestamps")
    if getattr(purchase.dt, "tz", None) is not None:
        raise AssertionError("profile snapshots require timezone-naive canonical purchase timestamps")
    snapshot = purchase.dt.normalize()
    if not snapshot.le(purchase).all():
        raise AssertionError("normalised snapshot is after its purchase timestamp")

    population = frozen["population"]
    lower = pd.Timestamp(population["start_inclusive"])
    upper = pd.Timestamp(population["end_inclusive"])
    if snapshot.lt(lower).any() or snapshot.gt(upper).any():
        raise AssertionError(
            f"order snapshot dates must lie within frozen population [{lower.date()},{upper.date()}]"
        )

    entities = _deterministic_entities(frame)
    order_keys = {
        entity_name: _key_index(snapshot, entity)
        for entity_name, entity in entities.items()
    }
    profiles, profile_audit = _load_daily_profiles(paths["daily"], order_keys)
    parent_scores, parent_audit = _load_parent_scores(paths["parent"])

    result = frame.copy()
    input_index = frame.index.copy()
    input_order_ids = frame["order_id"].astype(str).to_numpy(copy=True)
    for block in PROFILE_BLOCKS:
        block_audit = _join_one_block(
            result,
            snapshot,
            entities[str(FROZEN_PROFILE_SPECS[block]["entity"])],
            profiles[block],
            parent_scores[block],
            block,
        )
        profile_audit[block].update(block_audit)

    if len(result) != len(frame) or not result.index.equals(input_index):
        raise AssertionError("profile join changed input row count, order, or index")
    if not np.array_equal(result["order_id"].astype(str).to_numpy(), input_order_ids):
        raise AssertionError("profile join changed order_id order")
    if tuple(column for column in PROFILE_JOIN_COLUMNS if column not in result.columns):
        raise AssertionError("profile join did not emit the complete frozen join schema")

    audit: dict[str, object] = {
        "input_orders": len(frame),
        "output_orders": len(result),
        "order_id_sha256": _order_id_hash(frame["order_id"]),
        "snapshot_rule": "snapshot_date_equals_normalized_order_purchase_date",
        "snapshot_date_min": snapshot.min().strftime("%Y-%m-%d"),
        "snapshot_date_max": snapshot.max().strftime("%Y-%m-%d"),
        "snapshot_after_purchase_violations": 0,
        "profile_daily_path": str(paths["daily"]),
        "profile_daily_sha256": hashes["daily"],
        "profile_parent_path": str(paths["parent"]),
        "profile_parent_sha256": hashes["parent"],
        "profile_selection_freeze_sha256": hashes["freeze"],
        "profile_selected_candidates_sha256": hashes["selected"],
        "profile_snapshot_source_date_count": PROFILE_SNAPSHOT_DATE_COUNT,
        "profile_daily_source_rows": PROFILE_DAILY_TOTAL_ROWS,
        "daily_primary_key_columns": list(DAILY_PRIMARY_KEY),
        "daily_primary_key_valid": True,
        "parent": parent_audit,
        "blocks": profile_audit,
        "row_preservation_valid": True,
    }
    return result, audit


__all__ = [
    "EXPECTED_PROFILE_HASHES",
    "FROZEN_PROFILE_SPECS",
    "M5_ENDPOINT_FEATURE_COLUMNS",
    "PROFILE_AUDIT_FIELDS",
    "PROFILE_BLOCK_FEATURE_COLUMNS",
    "PROFILE_BLOCKS",
    "PROFILE_DAILY_SHA256",
    "PROFILE_DAILY_TOTAL_ROWS",
    "PROFILE_FEATURE_COLUMNS",
    "PROFILE_JOIN_COLUMNS",
    "PROFILE_OPTIONAL_SUPPORT_FIELD",
    "PROFILE_PARENT_SHA256",
    "PROFILE_PARENT_TOTAL_ROWS",
    "PROFILE_PAYLOAD_FIELDS",
    "PROFILE_PREFIXES",
    "PROFILE_SELECTED_CANDIDATES_SHA256",
    "PROFILE_SELECTION_FREEZE_SHA256",
    "PROFILE_SNAPSHOT_DATE_COUNT",
    "PROFILE_SNAPSHOT_DATE_MAX",
    "PROFILE_SNAPSHOT_DATE_MIN",
    "ROUTE_PROFILE_BLOCKS",
    "ROUTE_PROFILE_FEATURE_COLUMNS",
    "SELLER_PROFILE_BLOCKS",
    "SELLER_PROFILE_FEATURE_COLUMNS",
    "join_profiles",
]
