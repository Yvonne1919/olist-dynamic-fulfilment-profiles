"""High-performance daily snapshot engine for dynamic-profile EDA V1.1.

This module is deliberately side-effect free.  It accepts the already assembled
all-placed order frame and returns the four snapshot matrices normally produced
by :func:`run_eda_v1_1.process_snapshot` plus ``combine_parts``.

The implementation preserves the reference semantics while avoiding repeated
DataFrame filtering and groupby calls:

* timestamp windows are resolved with stable sorted integer-nanosecond indexes;
* target-level summaries are calculated once per source and broadcast over the
  seven entity granularities;
* entity identifiers are factorised once and aggregated with NumPy bincount,
  unique and maximum-at operations;
* exact entity medians and p90 values are calculated only on frozen rank days;
* the final-breach and positive-severity branches share structural support
  calculations because they have the same availability and validity masks.

No predictive model, profile estimator, or output file is created here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.dynamic_profile_eda_v1_1.scripts.core import (
    ENTITIES,
    HORIZONS,
    LAGS,
    SUPPORTS,
    TARGETS,
    WINDOWS,
)


_DAY_NS = 86_400_000_000_000
_NAT_INT = np.iinfo(np.int64).min
_SCHEME_DEFS = (("A", 0), ("B", 0)) + tuple(("C", lag) for lag in LAGS)
_RANK_PAIRS = (
    (("A", 0), ("B", 0)),
    *((("A", 0), ("C", lag)) for lag in LAGS),
    *((("B", 0), ("C", lag)) for lag in LAGS),
    *((("C", a), ("C", b)) for a, b in zip(LAGS[:-1], LAGS[1:])),
)


def _datetime_ns(values: pd.Series | pd.Index | Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Return naive datetime64[ns] integers plus an explicit non-NaT mask."""

    converted = pd.to_datetime(values, errors="coerce")
    if isinstance(converted, pd.Series):
        array = converted.to_numpy(dtype="datetime64[ns]")
    else:
        array = np.asarray(converted, dtype="datetime64[ns]")
    integers = array.view("i8")
    return integers, integers != _NAT_INT


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _safe_quantile(values: np.ndarray, probability: float) -> float:
    finite = values[~np.isnan(values)]
    return float(np.quantile(finite, probability, method="linear")) if finite.size else np.nan


def _safe_mean(values: np.ndarray) -> float:
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else np.nan


@dataclass(frozen=True)
class _TimeIndex:
    order: np.ndarray
    values: np.ndarray

    def between(self, lower_inclusive: int, upper_exclusive: int) -> np.ndarray:
        left = np.searchsorted(self.values, lower_inclusive, side="left")
        right = np.searchsorted(self.values, upper_exclusive, side="left")
        return self.order[left:right]


@dataclass(frozen=True)
class _TargetArrays:
    value: np.ndarray
    available_ns: np.ndarray
    available_observed: np.ndarray
    valid: np.ndarray
    available_index: _TimeIndex
    kind: str
    structural_group: str


@dataclass(frozen=True)
class _Selection:
    full: np.ndarray
    mature: np.ndarray
    eventual: np.ndarray
    valid_mature: np.ndarray


@dataclass(frozen=True)
class _EntityBase:
    support: np.ndarray
    summary: dict | None


@dataclass(frozen=True)
class _RankEntity:
    support: np.ndarray
    values: dict[str, np.ndarray]


class _PreparedFrame:
    def __init__(self, frame: pd.DataFrame, audit_endpoint: pd.Timestamp):
        missing = {
            "order_id",
            "order_purchase_timestamp",
            "purchase_date",
            "is_multi_seller",
            *ENTITIES.values(),
            *(spec["value"] for spec in TARGETS.values()),
            *(spec["available"] for spec in TARGETS.values()),
        } - set(frame.columns)
        if missing:
            raise KeyError(f"fast snapshot engine missing columns: {sorted(missing)}")

        self.frame = frame
        self.n = len(frame)
        self.audit_ns = pd.Timestamp(audit_endpoint).value
        self.purchase_ns, self.purchase_observed = _datetime_ns(frame["order_purchase_timestamp"])
        purchase_order = np.flatnonzero(self.purchase_observed)
        purchase_order = purchase_order[
            np.argsort(self.purchase_ns[purchase_order], kind="mergesort")
        ]
        self.purchase_index = _TimeIndex(purchase_order, self.purchase_ns[purchase_order])

        purchase_day_ns, purchase_day_observed = _datetime_ns(frame["purchase_date"])
        self.purchase_day_observed = purchase_day_observed
        self.purchase_day = np.full(self.n, -1, dtype=np.int32)
        if purchase_day_observed.any():
            day_number = purchase_day_ns[purchase_day_observed] // _DAY_NS
            self.day_origin = int(day_number.min())
            self.purchase_day[purchase_day_observed] = (
                day_number - self.day_origin
            ).astype(np.int32)
            self.day_span = int(day_number.max() - self.day_origin + 1)
        else:
            self.day_origin = 0
            self.day_span = 1

        self.multi_seller = frame["is_multi_seller"].fillna(False).to_numpy(dtype=bool)

        self.entity_codes: dict[str, np.ndarray] = {}
        self.entity_levels: dict[str, int] = {}
        for granularity, column in ENTITIES.items():
            codes, levels = pd.factorize(frame[column], sort=True, use_na_sentinel=True)
            self.entity_codes[granularity] = codes.astype(np.int32, copy=False)
            self.entity_levels[granularity] = len(levels)

        self.targets: dict[str, _TargetArrays] = {}
        for target, spec in TARGETS.items():
            available_ns, available_observed = _datetime_ns(frame[spec["available"]])
            available_order = np.flatnonzero(available_observed)
            available_order = available_order[
                np.argsort(available_ns[available_order], kind="mergesort")
            ]
            value = _numeric(frame[spec["value"]])
            valid = available_observed & ~np.isnan(value)
            if spec["kind"] == "process":
                valid &= value >= 0
            structural_group = (
                "delivery_error" if target in {"final_breach", "positive_late_days"} else target
            )
            self.targets[target] = _TargetArrays(
                value=value,
                available_ns=available_ns,
                available_observed=available_observed,
                valid=valid,
                available_index=_TimeIndex(
                    available_order, available_ns[available_order]
                ),
                kind=spec["kind"],
                structural_group=structural_group,
            )

    def future(self, t_ns: int, horizon_days: int) -> np.ndarray:
        return self.purchase_index.between(t_ns, t_ns + horizon_days * _DAY_NS)

    def purchase_cohort(
        self, t_ns: int, window_days: int, scheme: str, lag_days: int
    ) -> np.ndarray:
        if scheme == "B":
            lower = t_ns - window_days * _DAY_NS
            upper = t_ns
        elif scheme == "C":
            lower = t_ns - (lag_days + window_days) * _DAY_NS
            upper = t_ns - lag_days * _DAY_NS
        else:
            raise ValueError("purchase_cohort is defined only for Schemes B and C")
        return self.purchase_index.between(lower, upper)

    def selection(
        self,
        target: str,
        t_ns: int,
        window_days: int,
        scheme: str,
        lag_days: int,
        purchase_cache: dict[tuple[int, str, int], np.ndarray],
    ) -> _Selection:
        arrays = self.targets[target]
        if scheme == "A":
            full = arrays.available_index.between(
                t_ns - window_days * _DAY_NS, t_ns
            )
            mature = full
            eventual = full
        else:
            key = (window_days, scheme, lag_days)
            full = purchase_cache.get(key)
            if full is None:
                full = self.purchase_cohort(t_ns, window_days, scheme, lag_days)
                purchase_cache[key] = full
            mature_mask = (
                arrays.available_observed[full]
                & (arrays.available_ns[full] < t_ns)
            )
            eventual_mask = (
                arrays.available_observed[full]
                & (arrays.available_ns[full] <= self.audit_ns)
            )
            mature = full[mature_mask]
            eventual = full[eventual_mask]
        return _Selection(
            full=full,
            mature=mature,
            eventual=eventual,
            valid_mature=mature[arrays.valid[mature]],
        )


def _target_summary(values: np.ndarray, indices: np.ndarray, kind: str, prefix: str) -> dict:
    x = values[indices]
    observed = x[~np.isnan(x)]
    result = {f"{prefix}_target_value_count": int(observed.size)}
    if kind == "binary":
        result.update(
            {
                f"{prefix}_breach_count": int(np.count_nonzero(observed == 1)),
                f"{prefix}_breach_rate": float(observed.mean()) if observed.size else np.nan,
            }
        )
    elif kind == "severity":
        positive = observed[observed > 0]
        zero_share = (
            float(np.count_nonzero(observed == 0) / observed.size)
            if observed.size
            else np.nan
        )
        result.update(
            {
                f"{prefix}_zero_severity_share": zero_share,
                f"{prefix}_mean": _safe_mean(observed),
                f"{prefix}_median": _safe_quantile(observed, 0.5),
                f"{prefix}_p75": _safe_quantile(observed, 0.75),
                f"{prefix}_p90": _safe_quantile(observed, 0.90),
                f"{prefix}_p95": _safe_quantile(observed, 0.95),
                f"{prefix}_positive_only_count": int(positive.size),
                f"{prefix}_positive_only_mean": _safe_mean(positive),
                f"{prefix}_positive_only_median": _safe_quantile(positive, 0.5),
                f"{prefix}_positive_only_p90": _safe_quantile(positive, 0.90),
                f"{prefix}_positive_only_p95": _safe_quantile(positive, 0.95),
            }
        )
    else:
        nonnegative = observed[observed >= 0]
        result.update(
            {
                f"{prefix}_raw_duration_count": int(observed.size),
                f"{prefix}_negative_duration_count": int(np.count_nonzero(observed < 0)),
                f"{prefix}_nonnegative_duration_count": int(nonnegative.size),
                f"{prefix}_raw_mean": _safe_mean(observed),
                f"{prefix}_raw_median": _safe_quantile(observed, 0.5),
                f"{prefix}_raw_p75": _safe_quantile(observed, 0.75),
                f"{prefix}_raw_p90": _safe_quantile(observed, 0.90),
                f"{prefix}_raw_p95": _safe_quantile(observed, 0.95),
                f"{prefix}_nonnegative_mean": _safe_mean(nonnegative),
                f"{prefix}_nonnegative_median": _safe_quantile(nonnegative, 0.5),
                f"{prefix}_nonnegative_p75": _safe_quantile(nonnegative, 0.75),
                f"{prefix}_nonnegative_p90": _safe_quantile(nonnegative, 0.90),
                f"{prefix}_nonnegative_p95": _safe_quantile(nonnegative, 0.95),
            }
        )
    return result


def _add_selection_differences(row: dict) -> None:
    for key in list(row):
        if not key.startswith("asof_"):
            continue
        suffix = key[len("asof_") :]
        eventual_key = "eventual_" + suffix
        if eventual_key not in row:
            continue
        a = row[key]
        b = row[eventual_key]
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
            b, (int, float, np.integer, np.floating)
        ):
            row["eventual_minus_asof_" + suffix] = (
                b - a if pd.notna(a) and pd.notna(b) else np.nan
            )


def _entity_base(
    prepared: _PreparedFrame,
    valid_indices: np.ndarray,
    available_ns: np.ndarray,
    granularity: str,
    snapshot_ns: int,
) -> _EntityBase:
    codes_all = prepared.entity_codes[granularity]
    level_count = prepared.entity_levels[granularity]
    local_codes = codes_all[valid_indices]
    mapped = local_codes >= 0
    support = np.zeros(level_count, dtype=np.int32)
    if not mapped.any():
        return _EntityBase(support=support, summary=None)

    rows = valid_indices[mapped]
    codes = local_codes[mapped]
    support = np.bincount(codes, minlength=level_count).astype(np.int32, copy=False)
    active = support > 0
    active_support = support[active].astype(float)

    last_available = np.full(level_count, _NAT_INT, dtype=np.int64)
    np.maximum.at(last_available, codes, available_ns[rows])

    active_days = np.zeros(level_count, dtype=np.int32)
    has_day = prepared.purchase_day_observed[rows]
    if has_day.any():
        pair_keys = (
            codes[has_day].astype(np.int64) * prepared.day_span
            + prepared.purchase_day[rows[has_day]].astype(np.int64)
        )
        unique_pairs = np.unique(pair_keys)
        pair_entities = (unique_pairs // prepared.day_span).astype(np.int32)
        active_days = np.bincount(
            pair_entities, minlength=level_count
        ).astype(np.int32, copy=False)

    descending = np.sort(active_support)[::-1]
    total = float(descending.sum())
    summary = {
        "active_entities": int(active.sum()),
        "support_p10": _safe_quantile(active_support, 0.10),
        "support_p25": _safe_quantile(active_support, 0.25),
        "support_median": _safe_quantile(active_support, 0.50),
        "support_p75": _safe_quantile(active_support, 0.75),
        "support_p90": _safe_quantile(active_support, 0.90),
        "pct_entities_active_one_day": float((active_days[active] == 1).mean()),
        "median_active_days": _safe_quantile(active_days[active].astype(float), 0.50),
        "profile_freshness_median_days": _safe_quantile(
            (snapshot_ns - last_available[active]).astype(float) / _DAY_NS, 0.50
        ),
    }
    for threshold in SUPPORTS:
        qualified = active_support >= threshold
        summary[f"entities_support_ge_{threshold}"] = int(qualified.sum())
        summary[f"pct_entities_support_ge_{threshold}"] = float(qualified.mean())
    for percent in (1, 5, 10):
        take = max(1, int(np.ceil(descending.size * percent / 100)))
        summary[f"top_{percent}pct_order_concentration"] = (
            float(descending[:take].sum() / total) if total else np.nan
        )
    return _EntityBase(support=support, summary=summary)


def _group_quantiles(
    codes: np.ndarray,
    values: np.ndarray,
    level_count: int,
    probabilities: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Exact pandas-compatible linear quantiles for integer-coded groups."""

    outputs = tuple(np.full(level_count, np.nan, dtype=float) for _ in probabilities)
    if codes.size == 0:
        return outputs
    order = np.lexsort((values, codes))
    sorted_codes = codes[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    counts = np.diff(np.r_[starts, sorted_codes.size])
    groups = sorted_codes[starts]
    for probability, output in zip(probabilities, outputs):
        positions = (counts - 1) * probability
        lower = np.floor(positions).astype(np.int64)
        upper = np.ceil(positions).astype(np.int64)
        fraction = positions - lower
        low_values = sorted_values[starts + lower]
        high_values = sorted_values[starts + upper]
        difference = high_values - low_values
        if probability == 0.5:
            # pandas GroupBy.median uses the conventional endpoint average,
            # whereas Series.quantile(.5) uses NumPy's two-sided interpolation.
            interpolated = (low_values + high_values) / 2.0
        else:
            # Series.quantile uses NumPy's numerically stable two-sided _lerp:
            # weights >= .5 subtract from the upper value.
            interpolated = low_values + difference * fraction
            upper_side = fraction >= 0.5
            interpolated[upper_side] = high_values[upper_side] - difference[
                upper_side
            ] * (1.0 - fraction[upper_side])
        output[groups] = interpolated
    return outputs


def _group_kahan_means(
    codes: np.ndarray, values: np.ndarray, level_count: int, support: np.ndarray
) -> np.ndarray:
    """Match pandas groupby.mean's compensated summation in input-row order."""

    sums = np.zeros(level_count, dtype=float)
    compensation = np.zeros(level_count, dtype=float)
    # This loop is restricted to month-start rank calculations.  It prevents
    # near-tied entity means from receiving different ranks merely because
    # np.bincount and pandas use different floating-point accumulation rules.
    for code, value in zip(codes, values):
        adjusted = value - compensation[code]
        updated = sums[code] + adjusted
        compensation[code] = (updated - sums[code]) - adjusted
        sums[code] = updated
    means = np.full(level_count, np.nan, dtype=float)
    active = support > 0
    means[active] = sums[active] / support[active]
    return means


def _rank_entity(
    prepared: _PreparedFrame,
    valid_indices: np.ndarray,
    target: str,
    granularity: str,
    base: _EntityBase,
) -> _RankEntity:
    level_count = prepared.entity_levels[granularity]
    # Reference entity_table receives a boolean-indexed frame and therefore
    # aggregates rows in the all-placed frame's original order, not time-sort
    # order.  Preserve that order for bit-level mean/rank parity.
    ordered_indices = np.sort(valid_indices, kind="mergesort")
    codes = prepared.entity_codes[granularity][ordered_indices]
    values = prepared.targets[target].value[ordered_indices]
    mapped = (codes >= 0) & ~np.isnan(values)
    codes = codes[mapped]
    values = values[mapped]

    means = _group_kahan_means(codes, values, level_count, base.support)

    if prepared.targets[target].kind == "binary":
        result = {"rate": means}
    else:
        median, p90 = _group_quantiles(codes, values, level_count, (0.5, 0.9))
        result = {"mean": means, "median": median, "p90": p90}
    return _RankEntity(support=base.support, values=result)


def _spearman_row_fast(
    a: _RankEntity, b: _RankEntity, statistic: str, threshold: int
) -> dict:
    common = (a.support >= threshold) & (b.support >= threshold)
    indices = np.flatnonzero(common)
    value_a = a.values[statistic][indices]
    value_b = b.values[statistic][indices]
    constant_a = bool(np.unique(value_a[~np.isnan(value_a)]).size <= 1)
    constant_b = bool(np.unique(value_b[~np.isnan(value_b)]).size <= 1)
    valid = True
    reason = ""
    rho = np.nan
    p_value = np.nan
    if indices.size < 10:
        valid = False
        reason = "fewer_than_10_common_entities"
    elif constant_a or constant_b:
        valid = False
        reason = "constant_vector"
    else:
        complete = ~np.isnan(value_a) & ~np.isnan(value_b)
        if np.count_nonzero(complete) < 10:
            valid = False
            reason = "fewer_than_10_complete_pairs"
        else:
            result = spearmanr(value_a[complete], value_b[complete])
            rho = float(result.statistic)
            p_value = float(result.pvalue)
    return {
        "n_entities_source_a": int(np.count_nonzero(a.support)),
        "n_entities_source_b": int(np.count_nonzero(b.support)),
        "n_common_entities": int(indices.size),
        "support_threshold": threshold,
        "spearman_correlation": rho,
        "p_value": p_value,
        "constant_vector_a": constant_a,
        "constant_vector_b": constant_b,
        "valid": valid,
        "invalid_reason": reason,
    }


def _coverage_row(
    prepared: _PreparedFrame,
    future_indices: np.ndarray,
    entity_base: _EntityBase,
    granularity: str,
    snapshot: pd.Timestamp,
    target: str,
    window_days: int,
    scheme: str,
    lag_days: int,
    horizon_days: int,
) -> tuple[dict, dict]:
    codes = prepared.entity_codes[granularity][future_indices]
    mapped = codes >= 0
    historical_support = np.zeros(future_indices.size, dtype=float)
    if mapped.any():
        historical_support[mapped] = entity_base.support[codes[mapped]]
    seen = mapped & (historical_support >= 1)
    cold = mapped & (historical_support == 0)
    mapped_support = historical_support[mapped]
    total = future_indices.size
    mapped_count = int(mapped.sum())
    row = {
        "snapshot_date": snapshot,
        "sample": "all_placed",
        "target": target,
        "window_days": window_days,
        "scheme": scheme,
        "lag_days": lag_days,
        "granularity": granularity,
        "future_horizon_days": horizon_days,
        "total_future_placed_orders": total,
        "orders_with_valid_entity_mapping": mapped_count,
        "entity_id_nonmissing_rate": float(mapped.mean()) if total else np.nan,
        "historical_seen_orders": int(seen.sum()),
        "historical_seen_rate": float(seen.mean()) if total else np.nan,
        "mapped_cold_start_orders": int(cold.sum()),
        "cold_start_rate": float(cold.mean()) if total else np.nan,
        "cold_start_rate_among_mapped": (
            float(cold.sum() / mapped_count) if mapped_count else np.nan
        ),
        "missing_mapping_count": int((~mapped).sum()),
        "multi_seller_count": int(prepared.multi_seller[future_indices].sum()),
        "support_quantile_denominator": "mapped_future_orders_including_seen_and_unseen",
        "median_historical_support": _safe_quantile(mapped_support, 0.50),
        "support_p10": _safe_quantile(mapped_support, 0.10),
        "support_p25": _safe_quantile(mapped_support, 0.25),
        "support_p75": _safe_quantile(mapped_support, 0.75),
        "support_p90": _safe_quantile(mapped_support, 0.90),
    }
    support_fields = {
        "seen_rate": row["historical_seen_rate"],
        "entity_id_nonmissing_rate": row["entity_id_nonmissing_rate"],
    }
    for threshold in SUPPORTS:
        qualified = mapped & (historical_support >= threshold)
        count = int(qualified.sum())
        rate = float(qualified.mean()) if total else np.nan
        row[f"orders_support_ge_{threshold}"] = count
        row[f"order_weighted_support_ge_{threshold}_rate"] = rate
        support_fields[f"support_ge_{threshold}_rate"] = rate
    return row, support_fields


def compute_snapshot_outputs(
    frame: pd.DataFrame,
    audit_endpoint: pd.Timestamp | str,
    snapshots: Iterable[pd.Timestamp | str],
    rank_days: Iterable[pd.Timestamp | str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return corrected slicing, support, coverage and rank snapshot matrices.

    The returned schemas and row semantics match the existing
    ``process_snapshot``/``combine_parts`` path.  Snapshots are evaluated at the
    exact supplied timestamp; the normal production call supplies daily
    midnights and month-start ``rank_days``.
    """

    snapshot_index = pd.DatetimeIndex(pd.to_datetime(list(snapshots), errors="raise"))
    if snapshot_index.has_duplicates:
        raise ValueError("snapshots must be unique")
    snapshot_index = snapshot_index.sort_values()
    rank_set = {pd.Timestamp(value) for value in rank_days}
    prepared = _PreparedFrame(frame, pd.Timestamp(audit_endpoint))

    slicing_rows: list[dict] = []
    support_rows: list[dict] = []
    coverage_rows: list[dict] = []
    rank_rows: list[dict] = []

    for snapshot in snapshot_index:
        snapshot_ns = snapshot.value
        future_cache = {
            horizon: prepared.future(snapshot_ns, horizon) for horizon in HORIZONS
        }
        purchase_cache: dict[tuple[int, str, int], np.ndarray] = {}
        selection_cache: dict[tuple[str, int, str, int], _Selection] = {}
        structural_cache: dict[tuple[str, int, str, int, str], _EntityBase] = {}

        for target, spec in TARGETS.items():
            target_arrays = prepared.targets[target]
            for window_days in WINDOWS:
                rank_cache: dict[tuple[str, int, str], _RankEntity] = {}
                for scheme, lag_days in _SCHEME_DEFS:
                    selection_key = (
                        target_arrays.structural_group,
                        window_days,
                        scheme,
                        lag_days,
                    )
                    selection = selection_cache.get(selection_key)
                    if selection is None:
                        selection = prepared.selection(
                            target,
                            snapshot_ns,
                            window_days,
                            scheme,
                            lag_days,
                            purchase_cache,
                        )
                        selection_cache[selection_key] = selection

                    full = selection.full
                    mature = selection.mature
                    eventual = selection.eventual
                    valid_mature = selection.valid_mature
                    asof_summary = _target_summary(
                        target_arrays.value, mature, spec["kind"], "asof"
                    )
                    eventual_summary = _target_summary(
                        target_arrays.value, eventual, spec["kind"], "eventual"
                    )

                    for granularity in ENTITIES:
                        entity_codes = prepared.entity_codes[granularity]
                        mapped_full_count = int(np.count_nonzero(entity_codes[full] >= 0))
                        full_count = int(full.size)
                        structural_key = (
                            target_arrays.structural_group,
                            window_days,
                            scheme,
                            lag_days,
                            granularity,
                        )
                        entity_base = structural_cache.get(structural_key)
                        if entity_base is None:
                            entity_base = _entity_base(
                                prepared,
                                valid_mature,
                                target_arrays.available_ns,
                                granularity,
                                snapshot_ns,
                            )
                            structural_cache[structural_key] = entity_base

                        slicing = {
                            "snapshot_date": snapshot,
                            "sample": "all_placed",
                            "target": target,
                            "target_kind": spec["kind"],
                            "window_days": window_days,
                            "scheme": scheme,
                            "lag_days": lag_days,
                            "granularity": granularity,
                            "source_records": full_count,
                            "entity_id_available_orders": mapped_full_count,
                            "entity_id_nonmissing_rate": (
                                mapped_full_count / full_count if full_count else np.nan
                            ),
                            "timestamp_observed_asof_orders": int(mature.size),
                            "valid_outcomes_asof": int(valid_mature.size),
                            "invalid_or_anomalous_outcomes_asof": int(
                                mature.size - valid_mature.size
                            ),
                        }
                        if scheme == "A":
                            slicing.update(
                                {
                                    "cohort_total_orders_all_placed": np.nan,
                                    "eventually_available_orders_by_audit_end": np.nan,
                                    "mature_asof_orders": np.nan,
                                    "unresolved_asof_orders": np.nan,
                                    "never_observed_by_audit_end_orders": np.nan,
                                    "unconditional_maturity_fraction": np.nan,
                                    "conditional_maturity_fraction": np.nan,
                                    "eventual_observed_fraction": np.nan,
                                    "purchase_cohort_maturity_reason": "not_applicable_for_completion_window",
                                }
                            )
                        else:
                            denominator = full_count
                            eventual_count = int(eventual.size)
                            mature_count = int(mature.size)
                            slicing.update(
                                {
                                    "cohort_total_orders_all_placed": denominator,
                                    "eventually_available_orders_by_audit_end": eventual_count,
                                    "mature_asof_orders": mature_count,
                                    "unresolved_asof_orders": denominator - mature_count,
                                    "never_observed_by_audit_end_orders": denominator
                                    - eventual_count,
                                    "unconditional_maturity_fraction": (
                                        mature_count / denominator if denominator else np.nan
                                    ),
                                    "conditional_maturity_fraction": (
                                        mature_count / eventual_count
                                        if eventual_count
                                        else np.nan
                                    ),
                                    "eventual_observed_fraction": (
                                        eventual_count / denominator
                                        if denominator
                                        else np.nan
                                    ),
                                    "purchase_cohort_maturity_reason": "full_purchase_cohort_denominator",
                                }
                            )
                        slicing.update(asof_summary)
                        slicing.update(eventual_summary)
                        _add_selection_differences(slicing)
                        slicing_rows.append(slicing)

                        support = {
                            "snapshot_date": snapshot,
                            "sample": "all_placed",
                            "target": target,
                            "window_days": window_days,
                            "scheme": scheme,
                            "lag_days": lag_days,
                            "granularity": granularity,
                        }
                        if entity_base.summary is None:
                            support.update(
                                {
                                    "active_entities": 0,
                                    "entity_id_nonmissing_rate": (
                                        mapped_full_count / full_count
                                        if full_count
                                        else np.nan
                                    ),
                                }
                            )
                        else:
                            support.update(entity_base.summary)
                            support["entity_id_nonmissing_rate"] = (
                                mapped_full_count / full_count if full_count else np.nan
                            )

                        for horizon, future_indices in future_cache.items():
                            coverage, support_fields = _coverage_row(
                                prepared,
                                future_indices,
                                entity_base,
                                granularity,
                                snapshot,
                                target,
                                window_days,
                                scheme,
                                lag_days,
                                horizon,
                            )
                            coverage_rows.append(coverage)
                            support[f"future_{horizon}d_seen_rate"] = support_fields[
                                "seen_rate"
                            ]
                            support[
                                f"future_{horizon}d_entity_id_nonmissing_rate"
                            ] = support_fields["entity_id_nonmissing_rate"]
                            for threshold in SUPPORTS:
                                support[
                                    f"future_{horizon}d_support_ge_{threshold}_rate"
                                ] = support_fields[f"support_ge_{threshold}_rate"]
                        support_rows.append(support)

                        if snapshot in rank_set:
                            rank_cache[(scheme, lag_days, granularity)] = _rank_entity(
                                prepared,
                                valid_mature,
                                target,
                                granularity,
                                entity_base,
                            )

                if snapshot in rank_set:
                    statistics = (
                        ("rate",)
                        if spec["kind"] == "binary"
                        else ("mean", "median", "p90")
                    )
                    for granularity in ENTITIES:
                        for source_a, source_b in _RANK_PAIRS:
                            entity_a = rank_cache[
                                (source_a[0], source_a[1], granularity)
                            ]
                            entity_b = rank_cache[
                                (source_b[0], source_b[1], granularity)
                            ]
                            for statistic in statistics:
                                for threshold in (1, 5, 10, 20):
                                    rank = _spearman_row_fast(
                                        entity_a, entity_b, statistic, threshold
                                    )
                                    rank.update(
                                        {
                                            "snapshot_date": snapshot,
                                            "target": target,
                                            "window_days": window_days,
                                            "granularity": granularity,
                                            "entity_statistic": statistic,
                                            "scheme_a": source_a[0],
                                            "lag_a_days": source_a[1],
                                            "scheme_b": source_b[0],
                                            "lag_b_days": source_b[1],
                                            "interpretation": "descriptive_historical_rank_agreement_not_predictive_validation",
                                        }
                                    )
                                    rank_rows.append(rank)

    return (
        pd.DataFrame(slicing_rows),
        pd.DataFrame(support_rows),
        pd.DataFrame(coverage_rows),
        pd.DataFrame(rank_rows),
    )


def iter_snapshot_output_chunks(
    frame: pd.DataFrame,
    audit_endpoint: pd.Timestamp | str,
    snapshots: Iterable[pd.Timestamp | str],
    rank_days: Iterable[pd.Timestamp | str],
    *,
    chunk_days: int = 14,
):
    """Yield bounded snapshot blocks for deterministic streaming persistence.

    ``compute_snapshot_outputs`` intentionally returns materialised DataFrames,
    as required by its public contract.  A complete 625-day run nevertheless
    contains roughly two million wide output rows, so production integration
    should normally use this generator and persist each yielded tuple as four
    numbered part files before invoking the existing ``combine_parts`` helper.

    Yields ``(chunk_number, chunk_snapshots, outputs)``.  Re-preparing the
    99,441-row factorised frame once per small block is a bounded, low-cost
    trade-off that keeps this module stateless and prevents any full-run row
    dictionary accumulation.  Snapshot and final combine ordering remain
    deterministic because both are sorted with stable keys.
    """

    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    ordered = pd.DatetimeIndex(pd.to_datetime(list(snapshots), errors="raise"))
    if ordered.has_duplicates:
        raise ValueError("snapshots must be unique")
    ordered = ordered.sort_values()
    frozen_rank_days = tuple(pd.to_datetime(list(rank_days), errors="raise"))
    for chunk_number, start in enumerate(range(0, len(ordered), chunk_days)):
        chunk = ordered[start : start + chunk_days]
        yield (
            chunk_number,
            chunk,
            compute_snapshot_outputs(
                frame,
                audit_endpoint,
                chunk,
                frozen_rank_days,
            ),
        )
