"""Efficient, leakage-safe daily profile-score stability.

The reference implementation in :mod:`profile_core` rebuilds every history
slice with pandas ``groupby`` operations.  That path is intentionally simple
and remains the semantic authority.  This module implements the same P0/P1
binary and continuous score formulae with interval sufficient-statistic cubes:
each eligible order contributes to one contiguous interval of daily snapshots
under Scheme A or Scheme C, so counts and sums can be accumulated with range
updates followed by a cumulative sum.

Continuous P2 is also exactly reducible to sufficient statistics once the
row-origin expected values have been generated.  Binary P2 is not: its
penalised logistic offset depends on every individual nuisance logit.  Binary
P2 therefore uses a deliberately bounded hybrid that calls the shared exact
``profile_core.build_profiles`` implementation.  It never substitutes an
aggregate approximation.  If the configured call budget is insufficient the
run raises before doing P2 work.

The public entry point is :func:`compute_daily_score_stability`.  The parity
helpers at the bottom of the file compare this engine with ``build_profiles``
and include a deterministic synthetic fixture covering binary P0/P1/P2 and
continuous P0/P1/P2.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:  # Package import in production and pytest.
    from . import profile_core as core
except ImportError:  # Direct import during a local script/debug invocation.
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_core as core


STABILITY_COLUMNS = [
    "base_candidate_id",
    "target",
    "granularity",
    "previous_snapshot_date",
    "snapshot_date",
    "n_common_entities",
    "day_to_day_spearman",
    "median_absolute_score_change",
    "p90_absolute_score_change",
    "top20_jaccard",
    "valid",
    "invalid_reason",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent_structure",
    "kappa",
    "engine_mode",
]

_METRIC_COLUMNS = (
    "day_to_day_spearman",
    "median_absolute_score_change",
    "p90_absolute_score_change",
    "top20_jaccard",
)


@dataclass(frozen=True)
class _IntervalCube:
    """Daily sufficient statistics for one source and one model value."""

    dates: pd.DatetimeIndex
    entity_levels: np.ndarray
    parent_levels: np.ndarray
    entity_parent_codes: np.ndarray
    entity_n: np.ndarray
    entity_sum: np.ndarray
    entity_sum_sq: np.ndarray
    parent_n: np.ndarray
    parent_sum: np.ndarray
    global_n: np.ndarray
    global_sum: np.ndarray
    global_extra_sum: np.ndarray | None


def _normalise_dates(
    dates: Iterable[str | pd.Timestamp] | None,
    config: Mapping[str, object],
) -> pd.DatetimeIndex:
    if dates is None:
        start = pd.Timestamp(config["time"]["development"]["start"])
        end = pd.Timestamp(config["time"]["development"]["end_exclusive"])
        result = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    else:
        result = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise")).normalize()
        result = result.sort_values()
    if result.has_duplicates:
        raise ValueError("stability dates must be unique")
    if len(result) >= 2:
        gaps = np.diff(result.asi8)
        if not np.all(gaps == pd.Timedelta(days=1).value):
            raise ValueError("daily stability requires consecutive calendar dates")
    return result


def _normalise_sources(
    sources: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    result = [dict(source) for source in (core.candidate_sources() if sources is None else sources)]
    required = {"target", "granularity", "scheme", "window_days", "lag_days"}
    seen: set[tuple[object, ...]] = set()
    for source in result:
        missing = required - set(source)
        if missing:
            raise KeyError(f"candidate source is missing fields: {sorted(missing)}")
        key = tuple(source[name] for name in ("target", "granularity", "scheme", "window_days", "lag_days"))
        if key in seen:
            raise ValueError(f"duplicate candidate source: {key}")
        seen.add(key)
    return result


def _selected_variants(
    source: Mapping[str, object],
    allowed_base_ids: set[str] | None,
    estimators: set[str] | None = None,
) -> list[dict[str, object]]:
    variants = [dict(value) for value in core.candidate_variants(source)]
    if allowed_base_ids is not None:
        variants = [v for v in variants if str(v["base_candidate_id"]) in allowed_base_ids]
    if estimators is not None:
        variants = [v for v in variants if str(v["estimator"]) in estimators]
    return variants


def _interval_bounds(
    work: pd.DataFrame,
    source: Mapping[str, object],
    dates: pd.DatetimeIndex,
    available_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    if work.empty:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    snapshots = dates.asi8
    available = pd.to_datetime(work[available_col], errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    window_ns = pd.Timedelta(days=int(source["window_days"])).value
    if str(source["scheme"]) == "A":
        start = np.searchsorted(snapshots, available, side="right")
        stop = np.searchsorted(snapshots, available + window_ns, side="right")
    elif str(source["scheme"]) == "C":
        purchase = pd.to_datetime(work["order_purchase_timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64)
        lag_ns = pd.Timedelta(days=int(source["lag_days"])).value
        start = np.searchsorted(snapshots, np.maximum(available, purchase + lag_ns), side="right")
        stop = np.searchsorted(snapshots, purchase + lag_ns + window_ns, side="right")
    else:
        raise ValueError(f"unsupported scheme {source['scheme']}")
    return start.astype(np.int32, copy=False), stop.astype(np.int32, copy=False)


def _range_count_sum(
    start: np.ndarray,
    stop: np.ndarray,
    codes: np.ndarray,
    values: np.ndarray,
    n_days: int,
    n_groups: int,
    *,
    with_sum_sq: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_groups == 0:
        shape = (n_days, 0)
        return (
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
        )
    valid = (codes >= 0) & (start < stop)
    starts = start[valid]
    stops = stop[valid]
    group_codes = codes[valid].astype(np.intp, copy=False)
    vals = values[valid].astype(np.float64, copy=False)

    count_diff = np.zeros((n_days + 1, n_groups), dtype=np.int32)
    sum_diff = np.zeros((n_days + 1, n_groups), dtype=np.float64)
    np.add.at(count_diff, (starts, group_codes), 1)
    np.add.at(count_diff, (stops, group_codes), -1)
    np.add.at(sum_diff, (starts, group_codes), vals)
    np.add.at(sum_diff, (stops, group_codes), -vals)
    count = np.cumsum(count_diff[:-1], axis=0, dtype=np.int32)
    total = np.cumsum(sum_diff[:-1], axis=0, dtype=np.float64)

    if with_sum_sq:
        square_diff = np.zeros((n_days + 1, n_groups), dtype=np.float64)
        squares = np.square(vals)
        np.add.at(square_diff, (starts, group_codes), squares)
        np.add.at(square_diff, (stops, group_codes), -squares)
        total_sq = np.cumsum(square_diff[:-1], axis=0, dtype=np.float64)
    else:
        total_sq = np.zeros_like(total)
    return count, total, total_sq


def _range_sum_1d(
    start: np.ndarray,
    stop: np.ndarray,
    values: np.ndarray,
    n_days: int,
) -> np.ndarray:
    valid = start < stop
    diff = np.zeros(n_days + 1, dtype=np.float64)
    np.add.at(diff, start[valid], values[valid])
    np.add.at(diff, stop[valid], -values[valid])
    return np.cumsum(diff[:-1], dtype=np.float64)


def _factorize_strings(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    codes, levels = pd.factorize(values.astype("string"), sort=True)
    return codes.astype(np.int32, copy=False), np.asarray(levels.astype(str), dtype=object)


def _entity_parent_codes(
    entity_codes: np.ndarray,
    parent_codes: np.ndarray,
    n_entities: int,
    n_parents: int,
) -> np.ndarray:
    result = np.full(n_entities, -1, dtype=np.int32)
    mapped = entity_codes >= 0
    if not mapped.any() or n_entities == 0:
        return result
    packed = entity_codes[mapped].astype(np.int64) * max(n_parents, 1) + parent_codes[mapped].astype(np.int64)
    pairs = np.unique(packed)
    entities = (pairs // max(n_parents, 1)).astype(np.int32)
    parents = (pairs % max(n_parents, 1)).astype(np.int32)
    multiplicity = np.bincount(entities, minlength=n_entities)
    if np.any(multiplicity > 1):
        bad = np.flatnonzero(multiplicity > 1)[:5].tolist()
        raise RuntimeError(f"nondeterministic parent mapping for entity codes: {bad}")
    result[entities] = parents
    return result


def _build_interval_cube(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    dates: pd.DatetimeIndex,
    *,
    residual_expected_col: str | None = None,
) -> _IntervalCube:
    target = str(source["target"])
    granularity = str(source["granularity"])
    spec = core.TARGET_SPECS[target]
    value_col = str(spec["value"])
    available_col = str(spec["available"])
    entity_col = core.ENTITY_COLUMNS[granularity]
    structural_parent, structural_parent_col = core.STRUCTURAL_PARENT[granularity]

    valid = frame["in_canonical"].fillna(False).astype(bool) & core.target_valid_mask(frame, target)
    columns = ["order_purchase_timestamp", available_col, value_col, entity_col]
    if structural_parent == "global":
        parent_series = pd.Series("__GLOBAL__", index=frame.index, dtype="string")
    else:
        if structural_parent_col is None:
            raise AssertionError(f"missing structural parent column for {granularity}")
        columns.append(structural_parent_col)
        parent_series = frame[structural_parent_col].astype("string").fillna("__MISSING_PARENT__")
    if residual_expected_col is not None:
        if residual_expected_col not in frame:
            raise KeyError(residual_expected_col)
        valid &= frame[residual_expected_col].notna()
        columns.append(residual_expected_col)

    columns = list(dict.fromkeys(columns))
    work = frame.loc[valid, columns].copy()
    work["_parent_id"] = parent_series.loc[valid].to_numpy()
    raw_value = pd.to_numeric(work[value_col], errors="coerce").to_numpy(dtype=float)
    extra_values: np.ndarray | None = None
    if residual_expected_col is None:
        model_value = raw_value
    else:
        extra_values = pd.to_numeric(work[residual_expected_col], errors="coerce").to_numpy(dtype=float)
        model_value = raw_value - extra_values

    start, stop = _interval_bounds(work, source, dates, available_col)
    overlaps = start < stop
    work = work.loc[overlaps].reset_index(drop=True)
    model_value = model_value[overlaps]
    if extra_values is not None:
        extra_values = extra_values[overlaps]
    start = start[overlaps]
    stop = stop[overlaps]

    entity_codes, entity_levels = _factorize_strings(work[entity_col])
    parent_codes, parent_levels = _factorize_strings(work["_parent_id"])
    if (parent_codes < 0).any():
        raise AssertionError("parent IDs must be explicit, including missing-parent sentinel")
    entity_parent = _entity_parent_codes(
        entity_codes, parent_codes, len(entity_levels), len(parent_levels),
    )
    n_days = len(dates)
    entity_n, entity_sum, entity_sum_sq = _range_count_sum(
        start, stop, entity_codes, model_value, n_days, len(entity_levels), with_sum_sq=True,
    )
    parent_n, parent_sum, _ = _range_count_sum(
        start, stop, parent_codes, model_value, n_days, len(parent_levels), with_sum_sq=False,
    )
    global_n = parent_n.sum(axis=1, dtype=np.int64).astype(np.int32, copy=False)
    global_sum = parent_sum.sum(axis=1, dtype=np.float64)
    global_extra_sum = (
        _range_sum_1d(start, stop, extra_values, n_days) if extra_values is not None else None
    )
    return _IntervalCube(
        dates=dates,
        entity_levels=entity_levels,
        parent_levels=parent_levels,
        entity_parent_codes=entity_parent,
        entity_n=entity_n,
        entity_sum=entity_sum,
        entity_sum_sq=entity_sum_sq,
        parent_n=parent_n,
        parent_sum=parent_sum,
        global_n=global_n,
        global_sum=global_sum,
        global_extra_sum=global_extra_sum,
    )


def _binary_scores_for_day(
    cube: _IntervalCube,
    day_index: int,
    variants: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, np.ndarray]:
    n = cube.entity_n[day_index].astype(float)
    events = cube.entity_sum[day_index]
    active = n > 0
    raw = np.divide(events, n, out=np.full(len(n), np.nan), where=active)
    global_n = float(cube.global_n[day_index])
    global_raw = float(cube.global_sum[day_index] / global_n) if global_n > 0 else np.nan
    parent_n = cube.parent_n[day_index].astype(float)
    parent_events = cube.parent_sum[day_index]
    parent_min = int(config["binary_eb"]["parent_min_support"])
    result: dict[str, np.ndarray] = {}

    for variant in variants:
        estimator = str(variant["estimator"])
        candidate_id = str(variant["base_candidate_id"])
        if estimator == "P0":
            result[candidate_id] = raw.copy()
            continue
        if estimator != "P1":
            raise ValueError(f"binary interval cube does not support {estimator}")
        kappa = float(variant["kappa"])
        if str(variant["parent_structure"]) == "global":
            parent_for_entity = np.full(len(n), global_raw, dtype=float)
        else:
            parent_score = np.divide(
                parent_events + kappa * global_raw,
                parent_n + kappa,
                out=np.full(len(parent_n), global_raw, dtype=float),
                where=(parent_n + kappa) > 0,
            )
            unsupported = parent_n < parent_min
            parent_score[unsupported] = global_raw
            parent_for_entity = np.full(len(n), global_raw, dtype=float)
            mapped_parent = cube.entity_parent_codes >= 0
            parent_for_entity[mapped_parent] = parent_score[cube.entity_parent_codes[mapped_parent]]
        score = np.full(len(n), np.nan, dtype=float)
        score[active] = (
            events[active] + kappa * parent_for_entity[active]
        ) / (n[active] + kappa)
        result[candidate_id] = score
    return result


def _continuous_scores_for_day(
    cube: _IntervalCube,
    day_index: int,
    variants: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, np.ndarray]:
    n = cube.entity_n[day_index].astype(float)
    total = cube.entity_sum[day_index]
    total_sq = cube.entity_sum_sq[day_index]
    active = n > 0
    entity_mean = np.divide(total, n, out=np.full(len(n), np.nan), where=active)
    global_n = float(cube.global_n[day_index])
    global_model_mean = float(cube.global_sum[day_index] / global_n) if global_n > 0 else np.nan
    global_expected = (
        float(cube.global_extra_sum[day_index] / global_n)
        if cube.global_extra_sum is not None and global_n > 0
        else 0.0
    )
    parent_n = cube.parent_n[day_index].astype(float)
    parent_sum = cube.parent_sum[day_index]
    parent_min = int(config["binary_eb"]["parent_min_support"])
    floor = float(config["continuous_eb"]["variance_floor"])
    result: dict[str, np.ndarray] = {}

    within_df = float(np.maximum(n - 1.0, 0.0).sum())
    entity_ss = np.zeros(len(n), dtype=float)
    entity_ss[active] = total_sq[active] - np.square(total[active]) / n[active]
    # Direct residual squares in profile_core cannot be negative.  Range-sum
    # cancellation can create tiny negative values, so clip only that numerical
    # artefact and retain the exact algebra otherwise.
    entity_ss = np.maximum(entity_ss, 0.0)
    within = float(entity_ss.sum() / within_df) if within_df > 0 else np.nan

    for variant in variants:
        estimator = str(variant["estimator"])
        candidate_id = str(variant["base_candidate_id"])
        if estimator == "P0":
            result[candidate_id] = entity_mean.copy()
            continue
        if estimator not in {"P1", "P2"}:
            raise ValueError(f"continuous interval cube does not support {estimator}")
        if str(variant["parent_structure"]) == "global":
            parent_model = np.full(len(n), global_model_mean, dtype=float)
        else:
            parent_means = np.divide(
                parent_sum,
                parent_n,
                out=np.full(len(parent_n), global_model_mean, dtype=float),
                where=parent_n > 0,
            )
            parent_means[parent_n < parent_min] = global_model_mean
            parent_model = np.full(len(n), global_model_mean, dtype=float)
            mapped_parent = cube.entity_parent_codes >= 0
            parent_model[mapped_parent] = parent_means[cube.entity_parent_codes[mapped_parent]]

        weights = n[active]
        if weights.sum() > 0:
            deviations = entity_mean[active] - parent_model[active]
            weighted_var = float(np.average(np.square(deviations), weights=weights))
            noise = float(within * np.average(1.0 / weights, weights=weights)) if np.isfinite(within) else np.nan
        else:
            weighted_var = np.nan
            noise = np.nan
        between = max(weighted_var - noise, 0.0) if np.isfinite(weighted_var) and np.isfinite(noise) else np.nan
        score = np.full(len(n), np.nan, dtype=float)
        parent_score = global_expected + parent_model
        if not np.isfinite(within) or not np.isfinite(between) or within <= floor or between <= floor:
            score[active] = parent_score[active]
        else:
            precision = n[active] / within + 1.0 / between
            posterior_model = (
                n[active] * entity_mean[active] / within
                + parent_model[active] / between
            ) / precision
            score[active] = global_expected + posterior_model
        result[candidate_id] = score
    return result


def _variant_metadata(
    source: Mapping[str, object],
    variant: Mapping[str, object],
    engine_mode: str,
) -> dict[str, object]:
    return {
        "base_candidate_id": str(variant["base_candidate_id"]),
        "target": str(source["target"]),
        "granularity": str(source["granularity"]),
        "scheme": str(source["scheme"]),
        "window_days": int(source["window_days"]),
        "lag_days": int(source["lag_days"]),
        "estimator": str(variant["estimator"]),
        "parent_structure": str(variant["parent_structure"]),
        "kappa": np.nan if variant.get("kappa") is None else float(variant["kappa"]),
        "engine_mode": engine_mode,
    }


def _metric_row_from_arrays(
    previous: np.ndarray,
    current: np.ndarray,
    entity_levels: np.ndarray,
    previous_date: pd.Timestamp,
    current_date: pd.Timestamp,
    source: Mapping[str, object],
    variant: Mapping[str, object],
    engine_mode: str,
) -> dict[str, object]:
    previous_finite = np.isfinite(previous)
    current_finite = np.isfinite(current)
    common_codes = np.flatnonzero(previous_finite & current_finite)
    n_common = int(len(common_codes))
    if n_common:
        previous_values = previous[common_codes]
        current_values = current[common_codes]
        valid = (
            n_common >= 10
            and np.unique(previous_values).size > 1
            and np.unique(current_values).size > 1
        )
        change = np.abs(current_values - previous_values)
        median_change = float(np.median(change))
        p90_change = float(np.quantile(change, 0.90, method="linear"))
        top_n = max(1, int(math.ceil(n_common * 0.20)))
        previous_order = np.lexsort((common_codes, -previous_values))
        current_order = np.lexsort((common_codes, -current_values))
        top_previous = set(common_codes[previous_order[:top_n]].tolist())
        top_current = set(common_codes[current_order[:top_n]].tolist())
        union = top_previous | top_current
        jaccard = float(len(top_previous & top_current) / len(union)) if union else np.nan
        rho = float(spearmanr(previous_values, current_values).statistic) if valid else np.nan
        reason = "" if valid else "fewer_than_10_or_constant_common_entities"
    else:
        valid = False
        median_change = np.nan
        p90_change = np.nan
        jaccard = np.nan
        rho = np.nan
        reason = (
            "profile_unavailable_previous_or_current"
            if not previous_finite.any() or not current_finite.any()
            else "fewer_than_10_or_constant_common_entities"
        )
    row = {
        **_variant_metadata(source, variant, engine_mode),
        "previous_snapshot_date": pd.Timestamp(previous_date).normalize(),
        "snapshot_date": pd.Timestamp(current_date).normalize(),
        "n_common_entities": n_common,
        "day_to_day_spearman": rho,
        "median_absolute_score_change": median_change,
        "p90_absolute_score_change": p90_change,
        "top20_jaccard": jaccard,
        "valid": bool(valid),
        "invalid_reason": reason,
    }
    return row


def _metric_row_from_series(
    previous: pd.Series | None,
    current: pd.Series | None,
    previous_date: pd.Timestamp,
    current_date: pd.Timestamp,
    source: Mapping[str, object],
    variant: Mapping[str, object],
    engine_mode: str,
) -> dict[str, object]:
    if previous is None or current is None or previous.empty or current.empty:
        empty = np.array([], dtype=float)
        row = _metric_row_from_arrays(
            empty, empty, np.array([], dtype=object), previous_date, current_date,
            source, variant, engine_mode,
        )
        row["invalid_reason"] = "profile_unavailable_previous_or_current"
        return row
    common = previous.index.intersection(current.index).sort_values()
    if len(common) == 0:
        row = _metric_row_from_arrays(
            np.array([np.nan]), np.array([np.nan]), np.array(["__NONE__"], dtype=object),
            previous_date, current_date, source, variant, engine_mode,
        )
        # Both exact profiles exist, but their entity sets are disjoint.  The
        # reference builder emits a row and classifies this as an insufficient
        # common-entity comparison rather than an unavailable profile.
        row["invalid_reason"] = "fewer_than_10_or_constant_common_entities"
        return row
    previous_values = previous.reindex(common).to_numpy(dtype=float)
    current_values = current.reindex(common).to_numpy(dtype=float)
    # The arrays already contain only common entities, so the stable lexical
    # order of ``common`` is also the exact entity-ID tie-break order.
    return _metric_row_from_arrays(
        previous_values, current_values, common.to_numpy(dtype=object),
        previous_date, current_date, source, variant, engine_mode,
    )


def _interval_rows_for_source(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    dates: pd.DatetimeIndex,
    config: Mapping[str, object],
    variants: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not variants or len(dates) < 2:
        return []
    target = str(source["target"])
    kind = str(core.TARGET_SPECS[target]["kind"])
    estimators = {str(v["estimator"]) for v in variants}
    if kind == "binary":
        if not estimators <= {"P0", "P1"}:
            raise ValueError("binary P2 must use the exact bounded hybrid")
        cube = _build_interval_cube(frame, source, dates)
        score_function = _binary_scores_for_day
    else:
        p2_only = estimators == {"P2"}
        if "P2" in estimators and not p2_only:
            raise ValueError("continuous P2 must be cubed separately from outcome P0/P1")
        expected_col = f"expected_{target}" if p2_only else None
        cube = _build_interval_cube(frame, source, dates, residual_expected_col=expected_col)
        score_function = _continuous_scores_for_day

    rows: list[dict[str, object]] = []
    previous_scores: dict[str, np.ndarray] | None = None
    for day_index, snapshot in enumerate(dates):
        current_scores = score_function(cube, day_index, variants, config)
        if previous_scores is not None:
            for variant in variants:
                candidate_id = str(variant["base_candidate_id"])
                rows.append(_metric_row_from_arrays(
                    previous_scores[candidate_id], current_scores[candidate_id], cube.entity_levels,
                    dates[day_index - 1], snapshot, source, variant, "interval_cube",
                ))
        previous_scores = current_scores
    return rows


def _exact_binary_p2_rows(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    dates: pd.DatetimeIndex,
    config: Mapping[str, object],
    variants: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], int]:
    if not variants or len(dates) < 2:
        return [], 0
    expected_col = f"expected_{source['target']}"
    if expected_col not in frame:
        rows = []
        for previous_date, current_date in zip(dates[:-1], dates[1:]):
            for variant in variants:
                row = _metric_row_from_arrays(
                    np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=object),
                    previous_date, current_date, source, variant, "exact_shared_p2_hybrid",
                )
                row["invalid_reason"] = "p2_expected_column_missing"
                rows.append(row)
        return rows, 0

    allowed = {str(v["base_candidate_id"]) for v in variants}
    rows: list[dict[str, object]] = []
    previous: dict[str, pd.Series] | None = None
    calls = 0
    for day_index, snapshot in enumerate(dates):
        profiles, _ = core.build_profiles(
            frame, source, snapshot, config, allowed_base_ids=allowed,
        )
        calls += 1
        current: dict[str, pd.Series] = {}
        if not profiles.empty:
            for candidate_id, part in profiles.groupby("base_candidate_id", sort=False):
                series = part.set_index("entity_id")["score"].astype(float)
                if series.index.has_duplicates:
                    raise AssertionError(f"duplicate exact P2 profile entities for {candidate_id}")
                current[str(candidate_id)] = series.sort_index()
        if previous is not None:
            for variant in variants:
                candidate_id = str(variant["base_candidate_id"])
                rows.append(_metric_row_from_series(
                    previous.get(candidate_id), current.get(candidate_id),
                    dates[day_index - 1], snapshot, source, variant,
                    "exact_shared_p2_hybrid",
                ))
        previous = current
    return rows, calls


def compute_daily_score_stability(
    frame: pd.DataFrame,
    config: Mapping[str, object] | None = None,
    *,
    dates: Iterable[str | pd.Timestamp] | None = None,
    sources: Sequence[Mapping[str, object]] | None = None,
    allowed_base_ids: set[str] | None = None,
    include_binary_p2: bool = True,
    include_continuous_p2: bool = True,
    max_exact_binary_p2_calls: int = 20_000,
) -> pd.DataFrame:
    """Compute exact consecutive-day score stability for frozen candidates.

    Parameters
    ----------
    frame:
        Analysis frame produced by ``profile_core.build_analysis_frame`` with
        tail targets attached.  P2 additionally requires the frozen
        ``expected_<target>`` row-origin columns.
    config:
        Frozen config mapping.  ``profile_core.load_config()`` is used when
        omitted.
    dates:
        Consecutive daily snapshots.  The frozen development interval is used
        by default.
    sources:
        Optional bounded subset of ``profile_core.candidate_sources()``.
    allowed_base_ids:
        Optional candidate-ID subset, useful after the development freeze or
        for parity checks.
    include_binary_p2:
        Binary P2 uses the exact shared reference builder.  Disabling it is an
        explicit scope restriction; no approximate P2 rows are generated.
    include_continuous_p2:
        Continuous P2 is exactly cubed from row-origin residuals.
    max_exact_binary_p2_calls:
        Hard budget on source-by-day calls to the exact binary P2 builder.  The
        function raises before P2 work if the requested grid exceeds it.
    """

    cfg = core.load_config() if config is None else config
    daily_dates = _normalise_dates(dates, cfg)
    selected_sources = _normalise_sources(sources)
    allowed = None if allowed_base_ids is None else set(map(str, allowed_base_ids))

    binary_p2_call_requirement = 0
    if include_binary_p2:
        for source in selected_sources:
            target = str(source["target"])
            if str(core.TARGET_SPECS[target]["kind"]) != "binary":
                continue
            variants = _selected_variants(source, allowed, {"P2"})
            if variants and f"expected_{target}" in frame:
                binary_p2_call_requirement += len(daily_dates)
    if binary_p2_call_requirement > int(max_exact_binary_p2_calls):
        raise RuntimeError(
            "exact binary P2 stability requires "
            f"{binary_p2_call_requirement} source-day calls, exceeding the frozen "
            f"budget {int(max_exact_binary_p2_calls)}; no approximation was used"
        )

    rows: list[dict[str, object]] = []
    p2_calls = 0
    for source in selected_sources:
        target = str(source["target"])
        kind = str(core.TARGET_SPECS[target]["kind"])
        if kind == "binary":
            p01 = _selected_variants(source, allowed, {"P0", "P1"})
            rows.extend(_interval_rows_for_source(frame, source, daily_dates, cfg, p01))
            if include_binary_p2:
                p2 = _selected_variants(source, allowed, {"P2"})
                exact_rows, calls = _exact_binary_p2_rows(frame, source, daily_dates, cfg, p2)
                rows.extend(exact_rows)
                p2_calls += calls
        else:
            p01 = _selected_variants(source, allowed, {"P0", "P1"})
            rows.extend(_interval_rows_for_source(frame, source, daily_dates, cfg, p01))
            if include_continuous_p2:
                p2 = _selected_variants(source, allowed, {"P2"})
                expected_col = f"expected_{target}"
                if p2 and expected_col in frame:
                    rows.extend(_interval_rows_for_source(frame, source, daily_dates, cfg, p2))
                elif p2:
                    for previous_date, current_date in zip(daily_dates[:-1], daily_dates[1:]):
                        for variant in p2:
                            row = _metric_row_from_arrays(
                                np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=object),
                                previous_date, current_date, source, variant, "interval_cube",
                            )
                            row["invalid_reason"] = "p2_expected_column_missing"
                            rows.append(row)

    result = pd.DataFrame(rows, columns=STABILITY_COLUMNS)
    if not result.empty:
        result = result.sort_values(
            ["base_candidate_id", "snapshot_date"], kind="mergesort",
        ).reset_index(drop=True)
    result.attrs.update({
        "engine": "interval_sufficient_statistics_with_exact_binary_p2_hybrid",
        "n_dates": int(len(daily_dates)),
        "n_date_pairs": int(max(len(daily_dates) - 1, 0)),
        "n_sources": int(len(selected_sources)),
        "n_base_candidates": int(result["base_candidate_id"].nunique()) if not result.empty else 0,
        "exact_binary_p2_calls": int(p2_calls),
        "binary_p2_included": bool(include_binary_p2),
        "continuous_p2_included": bool(include_continuous_p2),
        "binary_p2_approximation_used": False,
    })
    return result


def parity_against_build_profiles(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, object] | None = None,
    dates: Iterable[str | pd.Timestamp],
    sources: Sequence[Mapping[str, object]],
    allowed_base_ids: set[str] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-11,
) -> pd.DataFrame:
    """Compare fast stability rows with the authoritative profile builder."""

    cfg = core.load_config() if config is None else config
    daily_dates = _normalise_dates(dates, cfg)
    selected_sources = _normalise_sources(sources)
    allowed = None if allowed_base_ids is None else set(map(str, allowed_base_ids))
    fast = compute_daily_score_stability(
        frame,
        cfg,
        dates=daily_dates,
        sources=selected_sources,
        allowed_base_ids=allowed,
        include_binary_p2=True,
        include_continuous_p2=True,
        max_exact_binary_p2_calls=max(1, len(daily_dates) * len(selected_sources)),
    )

    reference_parts: list[pd.DataFrame] = []
    for source in selected_sources:
        previous: pd.DataFrame | None = None
        for day_index, snapshot in enumerate(daily_dates):
            profiles, _ = core.build_profiles(
                frame, source, snapshot, cfg, allowed_base_ids=allowed,
            )
            if previous is not None:
                reference_parts.append(core.stability_between_profiles(
                    previous, profiles, daily_dates[day_index - 1], snapshot,
                ))
            previous = profiles
    reference_columns = [
        "base_candidate_id", "target", "granularity",
        "previous_snapshot_date", "snapshot_date", "n_common_entities",
        *_METRIC_COLUMNS, "valid", "invalid_reason",
    ]
    nonempty_reference = [part for part in reference_parts if not part.empty]
    reference = (
        pd.concat(nonempty_reference, ignore_index=True)
        if nonempty_reference else pd.DataFrame(columns=reference_columns)
    )
    keys = ["base_candidate_id", "previous_snapshot_date", "snapshot_date"]
    comparison = fast.merge(
        reference[keys + ["n_common_entities", *_METRIC_COLUMNS, "valid", "invalid_reason"]],
        on=keys,
        how="outer",
        suffixes=("_fast", "_reference"),
        indicator=True,
        validate="1:1",
    )

    rows: list[dict[str, object]] = []
    for record in comparison.to_dict("records"):
        present = record["_merge"] == "both"
        differences: list[float] = []
        metrics_ok = True
        if present:
            for name in _METRIC_COLUMNS:
                left = record[f"{name}_fast"]
                right = record[f"{name}_reference"]
                ok = bool(np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
                metrics_ok &= ok
                if np.isfinite(left) and np.isfinite(right):
                    differences.append(abs(float(left) - float(right)))
            structural_ok = (
                int(record["n_common_entities_fast"]) == int(record["n_common_entities_reference"])
                and bool(record["valid_fast"]) == bool(record["valid_reference"])
                and str(record["invalid_reason_fast"]) == str(record["invalid_reason_reference"])
            )
            parity_ok = metrics_ok and structural_ok
            reason = "" if parity_ok else "metric_or_validity_mismatch"
        elif record["_merge"] == "left_only":
            fast_reason = str(record.get("invalid_reason_fast", ""))
            parity_ok = (
                not bool(record.get("valid_fast", False))
                and fast_reason in {
                    "profile_unavailable_previous_or_current",
                    "p2_expected_column_missing",
                }
            )
            reason = "reference_profile_row_absent" if parity_ok else "unexpected_fast_only_row"
        else:
            parity_ok = False
            reason = "reference_row_missing_from_fast_output"
        rows.append({
            "base_candidate_id": record["base_candidate_id"],
            "previous_snapshot_date": record["previous_snapshot_date"],
            "snapshot_date": record["snapshot_date"],
            "parity_ok": bool(parity_ok),
            "comparison_status": record["_merge"],
            "max_absolute_metric_difference": max(differences) if differences else 0.0,
            "reason": reason,
        })
    return pd.DataFrame(rows).sort_values(
        ["base_candidate_id", "snapshot_date"], kind="mergesort",
    ).reset_index(drop=True)


def synthetic_parity_fixture() -> tuple[pd.DataFrame, list[dict[str, object]], pd.DatetimeIndex]:
    """Return a deterministic fixture exercising all estimator families."""

    rows: list[dict[str, object]] = []
    for entity_index in range(12):
        seller = f"seller_{entity_index:02d}"
        seller_state = f"S{entity_index % 3}"
        for order_index in range(12):
            purchase = (
                pd.Timestamp("2017-02-26 06:00:00")
                + pd.Timedelta(days=2 * order_index)
                + pd.Timedelta(hours=entity_index % 4)
            )
            available = (
                pd.Timestamp("2017-04-07 00:00:00")
                + pd.Timedelta(days=order_index % 6)
                + pd.Timedelta(hours=entity_index % 4)
            )
            late = float((entity_index + 2 * order_index) % 5 == 0)
            handling_days = 0.5 + ((3 * entity_index + order_index) % 15) / 2.0
            rows.append({
                "order_id": f"o_{entity_index:02d}_{order_index:02d}",
                "order_purchase_timestamp": purchase,
                "in_canonical": True,
                "seller_id": seller,
                "main_seller_state": seller_state,
                "state_od": f"state_route_{entity_index:02d}",
                "region_od": f"R{entity_index % 2}",
                "late_delivery": late,
                "promise_error_days": 2.0 if late else -2.0,
                "final_breach_available_at": available,
                "expected_final_breach": 0.05 + 0.01 * (entity_index % 5) + 0.002 * order_index,
                "handling_duration": handling_days,
                "handling_level_value": math.log1p(handling_days),
                "handling_available_at": available,
                "expected_handling_level": math.log1p(1.5 + 0.2 * (entity_index % 4) + 0.1 * order_index),
            })
    frame = pd.DataFrame(rows).sort_values("order_id", kind="mergesort").reset_index(drop=True)
    sources = [
        {
            "target": "final_breach", "granularity": "seller_id",
            "scheme": "A", "window_days": 30, "lag_days": 0,
        },
        {
            "target": "handling_level", "granularity": "seller_id",
            "scheme": "C", "window_days": 30, "lag_days": 14,
        },
        {
            "target": "final_breach", "granularity": "state_od",
            "scheme": "C", "window_days": 60, "lag_days": 30,
        },
    ]
    dates = pd.date_range("2017-04-10", "2017-04-14", freq="D")
    return frame, sources, dates


def run_synthetic_parity_check(
    config: Mapping[str, object] | None = None,
    *,
    assert_on_failure: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-11,
) -> pd.DataFrame:
    """Run the built-in exact-reference parity fixture."""

    frame, sources, dates = synthetic_parity_fixture()
    result = parity_against_build_profiles(
        frame,
        config=core.load_config() if config is None else config,
        dates=dates,
        sources=sources,
        rtol=rtol,
        atol=atol,
    )
    if assert_on_failure and (result.empty or not result["parity_ok"].all()):
        failures = result.loc[~result["parity_ok"]].head(20).to_dict("records")
        raise AssertionError(f"fast stability synthetic parity failed: {failures}")
    return result


# Concise integration alias; the longer name states the artifact semantics.
compute_fast_stability = compute_daily_score_stability


__all__ = [
    "STABILITY_COLUMNS",
    "compute_daily_score_stability",
    "compute_fast_stability",
    "parity_against_build_profiles",
    "synthetic_parity_fixture",
    "run_synthetic_parity_check",
]
