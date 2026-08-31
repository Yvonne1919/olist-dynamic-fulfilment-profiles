"""Deterministic selection and uncertainty helpers for profile validation V1.

This module implements the frozen development-only selection rules without
reading project files or mutating process state.  The public functions operate
on explicit pandas data frames so that the development process can persist all
inputs and the fresh confirmation process can reproduce labels from the frozen
selection artifact.

Conventions
-----------
The protocol's binary Pareto deltas are ``candidate - reference``; consequently
smaller values are better and a positive improvement is ``max(-delta, 0)``.
``profile_core.score_anchor`` currently emits ``delta_log_loss`` and
``delta_brier`` as ``reference - candidate``.  :func:`aggregate_anchor_metrics`
normalises those source columns to the protocol convention when
``source_delta_convention='reference_minus_candidate'`` (the default).

The bootstrap helpers resample whole entity or calendar-month clusters.  They
never resample individual orders independently.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


BINARY_TARGETS = frozenset({"handling_tail", "transit_tail", "final_breach"})
CONTINUOUS_TARGETS = frozenset(
    {"handling_level", "transit_level", "positive_late_severity"}
)

CATALOG_COLUMNS = (
    "candidate_id",
    "base_candidate_id",
    "profile_spec_id",
    "target",
    "target_family",
    "granularity",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent_structure",
    "kappa",
    "support_threshold",
    "min_support",
)

_DESIGN_COLUMNS = (
    "base_candidate_id",
    "profile_spec_id",
    "target",
    "target_family",
    "granularity",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent_structure",
    "kappa",
    "support_threshold",
)

_SIMPLICITY_RANK = {"P0": 0, "P1": 1, "P2": 2}
_DEFAULT_TOLERANCE = 1e-12
_DEFAULT_SEED = 20260823
_DEVELOPMENT_7D_ANCHORS = tuple(
    pd.date_range("2017-04-01", periods=39, freq="7D").to_pydatetime()
)


def target_family(target: str) -> str:
    """Return the frozen target family, raising for a non-candidate target."""

    if target in BINARY_TARGETS:
        return "binary"
    if target in CONTINUOUS_TARGETS:
        return "continuous"
    raise ValueError(f"unsupported frozen target: {target!r}")


def stable_base_candidate_id(
    *,
    target: str,
    granularity: str,
    scheme: str,
    window_days: int,
    lag_days: int,
    estimator: str,
    parent_structure: str,
    kappa: float | int | None,
) -> str:
    """Build the exact stable base ID used by ``profile_core``.

    The fields are all from frozen finite catalogs; no locale-dependent or
    floating-point formatting enters the ID.  ``kappa`` must be integral when
    present.
    """

    if estimator not in _SIMPLICITY_RANK:
        raise ValueError(f"unsupported estimator: {estimator!r}")
    if scheme not in {"A", "C"}:
        raise ValueError(f"unsupported scheme: {scheme!r}")
    if kappa is None or pd.isna(kappa):
        kappa_text = "na"
    else:
        kappa_float = float(kappa)
        if not np.isfinite(kappa_float) or not kappa_float.is_integer():
            raise ValueError("kappa must be a finite integer or None")
        kappa_text = str(int(kappa_float))
    return "|".join(
        [
            str(target),
            str(granularity),
            str(scheme),
            f"w{int(window_days)}",
            f"l{int(lag_days)}",
            str(estimator),
            f"parent={parent_structure}",
            f"kappa={kappa_text}",
        ]
    )


def stable_candidate_id(base_candidate_id: str, support_threshold: int) -> str:
    """Attach a frozen communication-level support rule to a stable base ID."""

    support = int(support_threshold)
    if support <= 0:
        raise ValueError("support_threshold must be positive")
    return f"{base_candidate_id}|min_support={support}"


def stable_profile_spec_id(base_candidate_id: str) -> str:
    """Return the stable opaque profile-spec ID used by the V1 runner."""

    return "ps_" + hashlib.sha256(str(base_candidate_id).encode("utf-8")).hexdigest()[:20]


def build_candidate_catalog(config: Mapping[str, Any]) -> pd.DataFrame:
    """Enumerate the complete frozen candidate/support catalog.

    P0 has only the global parent, matching ``profile_core.candidate_variants``.
    P1/P2 use every allowed structural parent.  Binary P1/P2 candidates cross
    the applicable frozen kappa grid; continuous P1/P2 have no kappa.  Scheme A
    uses lag zero, and Scheme C uses the target-specific lag catalog.

    Returns
    -------
    pandas.DataFrame
        One row per candidate/support rule, sorted by analytical keys.  Both
        ``base_candidate_id`` and ``candidate_id`` are guaranteed unique at
        their intended granularities (the base ID repeats only across support
        thresholds).
    """

    required = {"targets", "schemes", "parents", "binary_eb", "p2", "levels"}
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"frozen config missing sections: {missing}")

    support_values = tuple(int(x) for x in config["levels"]["support_candidates"])
    if not support_values or any(x <= 0 for x in support_values):
        raise ValueError("support candidate catalog must contain positive integers")

    rows: list[dict[str, Any]] = []
    targets_by_granularity = config["targets"]
    for granularity in sorted(targets_by_granularity):
        parent_options = tuple(str(x) for x in config["parents"][granularity])
        if "global" not in parent_options:
            raise ValueError(f"{granularity!r} parent catalog must include global")
        for target in sorted(str(x) for x in targets_by_granularity[granularity]):
            family = target_family(target)
            source_specs: list[tuple[str, int, int]] = []
            for window in config["schemes"]["A"]["windows_days"]:
                source_specs.append(("A", int(window), 0))
            c_lags = config["schemes"]["C"]["lags_by_target"][target]
            for window in config["schemes"]["C"]["windows_days"]:
                for lag in c_lags:
                    source_specs.append(("C", int(window), int(lag)))

            for scheme, window_days, lag_days in source_specs:
                variants: list[tuple[str, str, float | int | None]] = [
                    ("P0", "global", None)
                ]
                for estimator in ("P1", "P2"):
                    if family == "binary":
                        if estimator == "P1":
                            kappas = config["binary_eb"]["kappa_candidates"]
                        else:
                            kappas = config["p2"]["binary_offset_kappa_candidates"]
                        variants.extend(
                            (estimator, parent, int(kappa))
                            for parent in parent_options
                            for kappa in kappas
                        )
                    else:
                        variants.extend(
                            (estimator, parent, None) for parent in parent_options
                        )

                for estimator, parent, kappa in variants:
                    base_id = stable_base_candidate_id(
                        target=target,
                        granularity=granularity,
                        scheme=scheme,
                        window_days=window_days,
                        lag_days=lag_days,
                        estimator=estimator,
                        parent_structure=parent,
                        kappa=kappa,
                    )
                    for support in support_values:
                        rows.append(
                            {
                                "candidate_id": stable_candidate_id(base_id, support),
                                "base_candidate_id": base_id,
                                "profile_spec_id": stable_profile_spec_id(base_id),
                                "target": target,
                                "target_family": family,
                                "granularity": granularity,
                                "scheme": scheme,
                                "window_days": window_days,
                                "lag_days": lag_days,
                                "estimator": estimator,
                                "parent_structure": parent,
                                "kappa": np.nan if kappa is None else int(kappa),
                                "support_threshold": support,
                                "min_support": support,
                            }
                        )

    out = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    if out.empty:
        raise ValueError("frozen candidate catalog is empty")
    if out["candidate_id"].duplicated().any():
        duplicate = out.loc[out["candidate_id"].duplicated(), "candidate_id"].iloc[0]
        raise ValueError(f"duplicate candidate ID generated: {duplicate}")
    if not out["support_threshold"].eq(out["min_support"]).all():
        raise AssertionError("support aliases diverged while building the catalog")
    base_design = [
        c
        for c in CATALOG_COLUMNS
        if c not in {"candidate_id", "support_threshold", "min_support"}
    ]
    base_rows = out[list(base_design)].drop_duplicates()
    if base_rows["base_candidate_id"].duplicated().any():
        duplicate = base_rows.loc[
            base_rows["base_candidate_id"].duplicated(), "base_candidate_id"
        ].iloc[0]
        raise ValueError(f"base candidate ID collision: {duplicate}")
    sort_columns = [
        "target",
        "granularity",
        "scheme",
        "window_days",
        "lag_days",
        "estimator",
        "parent_structure",
        "kappa",
        "support_threshold",
        "candidate_id",
    ]
    return out.sort_values(sort_columns, kind="mergesort", na_position="first").reset_index(
        drop=True
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def _normalise_anchor_frame(
    anchor_metrics: pd.DataFrame,
    *,
    source_delta_convention: str,
) -> pd.DataFrame:
    """Return an anchor frame with protocol-canonical metric names and signs."""

    _require_columns(
        anchor_metrics,
        ["candidate_id", "target", "granularity", "anchor_date", "valid"],
        "anchor metrics",
    )
    if source_delta_convention not in {
        "reference_minus_candidate",
        "candidate_minus_reference",
    }:
        raise ValueError(
            "source_delta_convention must be 'reference_minus_candidate' or "
            "'candidate_minus_reference'"
        )
    out = anchor_metrics.copy()
    out["anchor_date"] = pd.to_datetime(out["anchor_date"], errors="raise")
    out["anchor_month"] = out["anchor_date"].dt.to_period("M").astype(str)
    if "target_family" not in out:
        out["target_family"] = out["target"].map(target_family)
    else:
        expected = out["target"].map(target_family)
        mismatch = out["target_family"].astype(str).ne(expected)
        if mismatch.any():
            bad = out.loc[mismatch, ["target", "target_family"]].iloc[0].to_dict()
            raise ValueError(f"target family mismatch: {bad}")

    binary = out["target_family"].eq("binary")
    continuous = out["target_family"].eq("continuous")
    out["protocol_delta_logloss"] = np.nan
    out["protocol_delta_brier"] = np.nan

    log_source = None
    for name in ("delta_logloss", "delta_log_loss"):
        if name in out:
            log_source = name
            break
    if binary.any() and log_source is None:
        raise KeyError("binary anchors require delta_logloss or delta_log_loss")
    if binary.any() and "delta_brier" not in out:
        raise KeyError("binary anchors require delta_brier")
    sign = -1.0 if source_delta_convention == "reference_minus_candidate" else 1.0
    if log_source is not None:
        out.loc[binary, "protocol_delta_logloss"] = (
            sign * pd.to_numeric(out.loc[binary, log_source], errors="coerce")
        )
    if "delta_brier" in out:
        out.loc[binary, "protocol_delta_brier"] = (
            sign * pd.to_numeric(out.loc[binary, "delta_brier"], errors="coerce")
        )

    out["primary_improvement"] = np.nan
    out.loc[binary, "primary_improvement"] = -out.loc[
        binary, "protocol_delta_logloss"
    ]
    if continuous.any():
        if "parent_minus_candidate_mae" in out:
            improvement = pd.to_numeric(
                out.loc[continuous, "parent_minus_candidate_mae"], errors="coerce"
            )
        elif "log_mae_improvement" in out:
            improvement = pd.to_numeric(
                out.loc[continuous, "log_mae_improvement"], errors="coerce"
            )
        elif {"parent_log_mae", "log_mae"}.issubset(out.columns):
            improvement = pd.to_numeric(
                out.loc[continuous, "parent_log_mae"], errors="coerce"
            ) - pd.to_numeric(out.loc[continuous, "log_mae"], errors="coerce")
        else:
            raise KeyError(
                "continuous anchors require parent_minus_candidate_mae, "
                "log_mae_improvement, or parent_log_mae and log_mae"
            )
        out.loc[continuous, "primary_improvement"] = improvement
    out["positive_primary_improvement"] = out["primary_improvement"].clip(lower=0)
    out["valid"] = out["valid"].fillna(False).astype(bool)
    return out


def _median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _validate_development_anchor_grid(frame: pd.DataFrame) -> None:
    """Hard-stop unless every candidate has the exact frozen 39-anchor grid."""

    expected = pd.DatetimeIndex(_DEVELOPMENT_7D_ANCHORS)
    for candidate_id, group in frame.groupby("candidate_id", sort=True, dropna=False):
        observed = pd.DatetimeIndex(group["anchor_date"].drop_duplicates()).sort_values()
        if not observed.equals(expected):
            missing = expected.difference(observed).strftime("%Y-%m-%d").tolist()
            extra = observed.difference(expected).strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"{candidate_id}: development 7d anchor grid mismatch; "
                f"missing={missing}, extra={extra}"
            )


def aggregate_anchor_metrics(
    anchor_metrics: pd.DataFrame,
    *,
    scheduled_anchors: int = 39,
    source_delta_convention: str = "reference_minus_candidate",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate development seven-day anchor metrics overall and by month.

    The denominator is the frozen count of 39 scheduled development anchors;
    it is never inferred from observed rows.  Input must already be restricted
    to the development phase and seven-day horizon.  Duplicate
    ``candidate_id, anchor_date`` rows hard-stop because they would otherwise
    silently overweight an anchor.

    Parameters
    ----------
    source_delta_convention:
        ``reference_minus_candidate`` matches current ``profile_core`` output.
        ``candidate_minus_reference`` is accepted for already-canonical input.

    Returns
    -------
    (aggregate, by_month): tuple[pandas.DataFrame, pandas.DataFrame]
        Both tables contain protocol-canonical binary deltas.  Medians only use
        rows marked valid; counts retain the fixed scheduled denominator.
    """

    if int(scheduled_anchors) != 39:
        raise ValueError("development 7d selection denominator is frozen at 39")
    normal = _normalise_anchor_frame(
        anchor_metrics, source_delta_convention=source_delta_convention
    )
    duplicate = normal.duplicated(["candidate_id", "anchor_date"], keep=False)
    if duplicate.any():
        sample = normal.loc[duplicate, ["candidate_id", "anchor_date"]].iloc[0]
        raise ValueError(
            "duplicate candidate/anchor row: "
            f"{sample['candidate_id']} @ {sample['anchor_date']}"
        )
    _validate_development_anchor_grid(normal)

    metric_map = {
        "protocol_delta_logloss": "median_delta_logloss",
        "protocol_delta_brier": "median_delta_brier",
        "log_loss": "median_log_loss",
        "brier": "median_brier",
        "log_mae": "median_candidate_mae",
        "parent_log_mae": "median_parent_mae",
        "primary_improvement": "median_primary_improvement",
        "weighted_spearman": "median_weighted_spearman",
        "top_quintile_lift": "median_top_quintile_lift",
        "support_qualified_coverage": "median_support_qualified_coverage",
        "future_seen_coverage": "median_seen_coverage",
        "daily_stability_spearman": "median_daily_stability_spearman",
    }

    def summarise(group: pd.DataFrame, expected: int | None) -> dict[str, Any]:
        valid = group.loc[group["valid"]]
        result: dict[str, Any] = {
            "n_anchor_rows": int(len(group)),
            "n_valid_anchors": int(group.loc[group["valid"], "anchor_date"].nunique()),
        }
        if expected is not None:
            result["n_scheduled_anchors"] = int(expected)
            result["valid_anchor_fraction"] = result["n_valid_anchors"] / float(expected)
        for source, destination in metric_map.items():
            result[destination] = (
                _median_or_nan(valid[source]) if source in valid.columns else np.nan
            )
        result["median_parent_minus_candidate_mae"] = (
            result["median_primary_improvement"]
            if str(group["target_family"].iloc[0]) == "continuous"
            else np.nan
        )
        return result

    aggregate_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    for candidate_id, group in normal.groupby("candidate_id", sort=True, dropna=False):
        metadata = {"candidate_id": candidate_id}
        for column in _DESIGN_COLUMNS:
            if column in group:
                values = group[column].drop_duplicates()
                if len(values) > 1:
                    raise ValueError(f"{candidate_id}: inconsistent {column}")
                metadata[column] = values.iloc[0] if len(values) else np.nan
        aggregate_rows.append({**metadata, **summarise(group, scheduled_anchors)})
        for month, month_group in group.groupby("anchor_month", sort=True):
            month_rows.append(
                {
                    **metadata,
                    "anchor_month": str(month),
                    **summarise(month_group, None),
                    "positive_primary_improvement_sum": float(
                        pd.to_numeric(
                            month_group.loc[
                                month_group["valid"], "positive_primary_improvement"
                            ],
                            errors="coerce",
                        ).fillna(0.0).sum()
                    ),
                }
            )

    aggregate = pd.DataFrame(aggregate_rows)
    by_month = pd.DataFrame(month_rows)
    return (
        aggregate.sort_values("candidate_id", kind="mergesort").reset_index(drop=True),
        by_month.sort_values(
            ["candidate_id", "anchor_month"], kind="mergesort"
        ).reset_index(drop=True),
    )


def _support_at_least_five(stratum: pd.Series) -> pd.Series:
    text = stratum.astype(str)
    return text.isin({"support_5_9", "support_10_19", "support_20_plus"}) | text.str.match(
        r"^(?:support_)?(?:[5-9]|[1-9][0-9]+)(?:_|$)", na=False
    )


def _support_below_five(stratum: pd.Series) -> pd.Series:
    """Recognise the frozen cold/1--4 support strata, not missing mapping."""

    text = stratum.astype(str)
    return text.isin({"support_0_cold_start", "support_1_4"}) | text.str.match(
        r"^(?:support_)?[0-4](?:_|$)", na=False
    )


def minimum_evidence_gates(
    anchor_metrics: pd.DataFrame,
    *,
    support_strata: pd.DataFrame | None = None,
    scheduled_anchors: int = 39,
    minimum_valid_fraction: float = 0.75,
    maximum_month_share: float = 0.50,
    source_delta_convention: str = "reference_minus_candidate",
    tolerance: float = _DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Evaluate every frozen minimum-evidence safeguard by candidate.

    Required anchor audit fields are ``information_time_violations`` and either
    explicit ``required_fields_complete``, or complete candidate/date/valid and
    primary metric fields.  Seen and support-qualified coverage must be finite
    for every valid anchor.  Support strata must be supplied and contain
    ``candidate_id``, ``support_stratum`` and ``primary_improvement``.

    Month concentration is based on sums of positive anchor-level primary
    improvement.  When no anchor has positive improvement, the concentration
    and support-location safeguards are recorded as vacuously true rather than
    converted into an efficacy/significance gate; ``no_positive_improvement``
    remains explicit for Pareto/confirmation interpretation.
    """

    if int(scheduled_anchors) != 39:
        raise ValueError("development 7d selection denominator is frozen at 39")
    minimum_valid = int(math.ceil(float(minimum_valid_fraction) * scheduled_anchors))
    if minimum_valid != 30:
        raise ValueError("frozen 75% of 39 gate must equal 30 anchors")
    normal = _normalise_anchor_frame(
        anchor_metrics, source_delta_convention=source_delta_convention
    )
    duplicate = normal.duplicated(["candidate_id", "anchor_date"], keep=False)
    if duplicate.any():
        raise ValueError("minimum-evidence input has duplicate candidate/anchor rows")
    _validate_development_anchor_grid(normal)

    if support_strata is not None:
        _require_columns(
            support_strata,
            [
                "candidate_id",
                "anchor_date",
                "support_stratum",
                "primary_improvement",
            ],
            "support strata",
        )
        strata = support_strata.copy()
        strata["anchor_date"] = pd.to_datetime(strata["anchor_date"], errors="raise")
        strata["primary_improvement"] = pd.to_numeric(
            strata["primary_improvement"], errors="coerce"
        )
    else:
        strata = None

    results: list[dict[str, Any]] = []
    for candidate_id, group in normal.groupby("candidate_id", sort=True, dropna=False):
        valid = group.loc[group["valid"]]
        n_valid = int(valid["anchor_date"].nunique())
        valid_count_pass = n_valid >= minimum_valid

        if "information_time_violations" in group:
            violation_values = pd.to_numeric(
                group["information_time_violations"], errors="coerce"
            )
            time_audit_present = bool(violation_values.notna().all())
            time_leakage_pass = bool(
                time_audit_present and (violation_values.fillna(np.inf) == 0).all()
            )
            n_violations = (
                int(violation_values.fillna(0).sum()) if time_audit_present else np.nan
            )
        else:
            time_audit_present = False
            time_leakage_pass = False
            n_violations = np.nan

        coverage_columns = ["future_seen_coverage", "support_qualified_coverage"]
        coverage_reported = all(column in valid for column in coverage_columns)
        if coverage_reported and len(valid):
            coverage_reported = all(
                np.isfinite(pd.to_numeric(valid[column], errors="coerce")).all()
                for column in coverage_columns
            )
        elif not len(valid):
            coverage_reported = False

        primary_complete = np.isfinite(
            pd.to_numeric(valid["primary_improvement"], errors="coerce")
        ).all()
        if "required_fields_complete" in group:
            required_fields_complete = bool(
                len(valid)
                and valid["required_fields_complete"].fillna(False).astype(bool).all()
                and primary_complete
            )
        else:
            required_fields_complete = bool(primary_complete and len(valid) > 0)

        positive = pd.to_numeric(
            valid["positive_primary_improvement"], errors="coerce"
        ).fillna(0.0)
        total_positive = float(positive.sum())
        no_positive = total_positive <= tolerance
        if no_positive:
            maximum_observed_month_share = 0.0
            concentration_pass = True
        else:
            per_month = positive.groupby(valid["anchor_month"], sort=True).sum()
            maximum_observed_month_share = float(per_month.max() / total_positive)
            concentration_pass = maximum_observed_month_share <= maximum_month_share + tolerance

        candidate_strata = (
            strata.loc[strata["candidate_id"].eq(candidate_id)].copy()
            if strata is not None
            else pd.DataFrame()
        )
        valid_anchor_dates = pd.DatetimeIndex(valid["anchor_date"].drop_duplicates())
        reported_anchor_dates = pd.DatetimeIndex(
            candidate_strata["anchor_date"].drop_duplicates()
        )
        low_support_stratum_reported = bool(
            len(candidate_strata)
            and _support_below_five(candidate_strata["support_stratum"]).any()
        )
        high_support_stratum_reported = bool(
            len(candidate_strata)
            and _support_at_least_five(candidate_strata["support_stratum"]).any()
        )
        support_strata_reported = bool(
            len(candidate_strata)
            and valid_anchor_dates.difference(reported_anchor_dates).empty
            and low_support_stratum_reported
            and high_support_stratum_reported
        )
        high_support_positive = 0.0
        if support_strata_reported:
            high = candidate_strata.loc[
                _support_at_least_five(candidate_strata["support_stratum"])
            ]
            high_support_positive = float(
                high["primary_improvement"].clip(lower=0).fillna(0.0).sum()
            )
        support_location_pass = bool(
            no_positive or (support_strata_reported and high_support_positive > tolerance)
        )

        passes = {
            "valid_anchor_count_pass": valid_count_pass,
            "coverage_reported_pass": coverage_reported,
            "support_strata_reported_pass": support_strata_reported,
            "time_leakage_pass": time_leakage_pass,
            "required_fields_complete_pass": required_fields_complete,
            "month_concentration_pass": concentration_pass,
            "support_at_least_five_pass": support_location_pass,
        }
        failed = [name.removesuffix("_pass") for name, value in passes.items() if not value]
        results.append(
            {
                "candidate_id": candidate_id,
                "n_scheduled_anchors": scheduled_anchors,
                "minimum_valid_anchors": minimum_valid,
                "n_valid_anchors": n_valid,
                "valid_anchor_fraction": n_valid / float(scheduled_anchors),
                "information_time_audit_present": time_audit_present,
                "information_time_violations": n_violations,
                "total_positive_primary_improvement": total_positive,
                "no_positive_improvement": no_positive,
                "maximum_single_month_positive_share": maximum_observed_month_share,
                "high_support_positive_improvement": high_support_positive,
                "low_support_stratum_reported": low_support_stratum_reported,
                "high_support_stratum_reported": high_support_stratum_reported,
                **passes,
                "minimum_evidence_pass": bool(all(passes.values())),
                "minimum_evidence_failure_reasons": ";".join(failed),
            }
        )
    return pd.DataFrame(results).sort_values("candidate_id", kind="mergesort").reset_index(
        drop=True
    )


def _pareto_dimensions(family: str) -> tuple[tuple[str, str], ...]:
    if family == "binary":
        return (
            ("median_delta_logloss", "min"),
            ("median_delta_brier", "min"),
            ("median_top_quintile_lift", "max"),
            ("median_support_qualified_coverage", "max"),
            ("median_daily_stability_spearman", "max"),
        )
    if family == "continuous":
        return (
            ("median_parent_minus_candidate_mae", "max"),
            ("median_weighted_spearman", "max"),
            ("median_top_quintile_lift", "max"),
            ("median_support_qualified_coverage", "max"),
            ("median_daily_stability_spearman", "max"),
        )
    raise ValueError(f"unknown target family: {family!r}")


def pareto_frontier(
    candidate_summary: pd.DataFrame,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Mark candidates dominated under the frozen multi-objective directions.

    There is no composite score.  Only candidates passing
    ``minimum_evidence_pass`` and finite on every family-specific Pareto
    dimension are eligible.  Dominance means no worse on every dimension and
    better by more than ``tolerance`` on at least one.
    """

    _require_columns(
        candidate_summary,
        [
            "candidate_id",
            "target",
            "target_family",
            "granularity",
            "minimum_evidence_pass",
        ],
        "candidate summary",
    )
    out = candidate_summary.copy()
    out["pareto_eligible"] = False
    out["pareto_nondominated"] = False
    out["dominated_by"] = ""
    out["pareto_ineligible_reason"] = ""

    for (_, _), indices in out.groupby(["target", "granularity"], sort=True).groups.items():
        group_indices = list(indices)
        families = out.loc[group_indices, "target_family"].astype(str).unique()
        if len(families) != 1:
            raise ValueError("target/granularity group has mixed target families")
        dimensions = _pareto_dimensions(str(families[0]))
        _require_columns(out, [name for name, _ in dimensions], "candidate summary")

        eligible_indices: list[int] = []
        for index in group_indices:
            if not bool(out.at[index, "minimum_evidence_pass"]):
                out.at[index, "pareto_ineligible_reason"] = "minimum_evidence_failed"
                continue
            values = pd.to_numeric(
                out.loc[index, [name for name, _ in dimensions]], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                out.at[index, "pareto_ineligible_reason"] = "nonfinite_pareto_dimension"
                continue
            out.at[index, "pareto_eligible"] = True
            eligible_indices.append(index)

        for b_index in eligible_indices:
            dominators: list[str] = []
            for a_index in eligible_indices:
                if a_index == b_index:
                    continue
                no_worse = True
                strictly_better = False
                for column, direction in dimensions:
                    a = float(out.at[a_index, column])
                    b = float(out.at[b_index, column])
                    if direction == "min":
                        if a > b + tolerance:
                            no_worse = False
                            break
                        strictly_better |= a < b - tolerance
                    else:
                        if a < b - tolerance:
                            no_worse = False
                            break
                        strictly_better |= a > b + tolerance
                if no_worse and strictly_better:
                    dominators.append(str(out.at[a_index, "candidate_id"]))
            if dominators:
                out.at[b_index, "dominated_by"] = ";".join(sorted(dominators))
            else:
                out.at[b_index, "pareto_nondominated"] = True
    return out.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def _practically_equivalent(a: pd.Series, b: pd.Series, family: str) -> bool:
    def finite_pair(column: str) -> bool:
        values = pd.to_numeric(pd.Series([a.get(column), b.get(column)]), errors="coerce")
        return bool(np.isfinite(values.to_numpy(dtype=float)).all())

    if family == "binary":
        log_column = (
            "median_log_loss"
            if finite_pair("median_log_loss")
            else "median_delta_logloss"
        )
        brier_column = (
            "median_brier"
            if finite_pair("median_brier")
            else "median_delta_brier"
        )
        if not finite_pair(log_column) or not finite_pair(brier_column):
            return False
        return bool(
            abs(float(a[log_column]) - float(b[log_column])) < 0.001
            and abs(float(a[brier_column]) - float(b[brier_column])) < 0.0005
        )
    if family == "continuous":
        if not finite_pair("median_weighted_spearman") or not finite_pair(
            "median_candidate_mae"
        ):
            return False
        rho_close = (
            abs(
                float(a["median_weighted_spearman"])
                - float(b["median_weighted_spearman"])
            )
            < 0.02
        )
        mae_a = float(a["median_candidate_mae"])
        mae_b = float(b["median_candidate_mae"])
        denominator = max(abs(mae_a), abs(mae_b), np.finfo(float).eps)
        return bool(rho_close and abs(mae_a - mae_b) / denominator < 0.01)
    raise ValueError(f"unknown target family: {family!r}")


def select_confirmation_candidates(
    pareto_table: pd.DataFrame,
    *,
    max_candidates: int = 2,
) -> pd.DataFrame:
    """Apply frozen practical tie-breaks and promote at most two candidates.

    Complexity is used only inside a practically-equivalent pair; a simpler
    materially worse candidate is never promoted merely because it is P0/P1.
    Equivalent survivors then prefer longer windows, higher
    support-qualified coverage, and lexical candidate ID.  If more than two
    genuinely unresolved Pareto candidates remain, those same deterministic
    non-composite ordering keys choose the two frozen confirmation candidates.
    """

    if int(max_candidates) != 2:
        raise ValueError("frozen maximum is exactly two candidates per target/granularity")
    required = [
        "candidate_id",
        "target",
        "target_family",
        "granularity",
        "estimator",
        "window_days",
        "median_support_qualified_coverage",
        "pareto_nondominated",
    ]
    _require_columns(pareto_table, required, "Pareto table")
    out = pareto_table.copy()
    out["selected_for_confirmation"] = False
    out["selection_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["selection_decision"] = np.where(
        out["pareto_nondominated"], "unresolved_pareto", "not_pareto_nondominated"
    )

    for (_, _), indices in out.groupby(["target", "granularity"], sort=True).groups.items():
        active = {
            int(index)
            for index in indices
            if bool(out.at[index, "pareto_nondominated"])
        }
        if not active:
            continue
        family_values = out.loc[list(active), "target_family"].astype(str).unique()
        if len(family_values) != 1:
            raise ValueError("Pareto group has mixed target families")
        family = str(family_values[0])

        # Frozen simplicity preference, but only for practically equal results.
        for a_index in sorted(active, key=lambda i: str(out.at[i, "candidate_id"])):
            if a_index not in active:
                continue
            for b_index in sorted(active, key=lambda i: str(out.at[i, "candidate_id"])):
                if a_index == b_index or b_index not in active:
                    continue
                a_rank = _SIMPLICITY_RANK.get(str(out.at[a_index, "estimator"]))
                b_rank = _SIMPLICITY_RANK.get(str(out.at[b_index, "estimator"]))
                if a_rank is None or b_rank is None:
                    raise ValueError("unknown estimator in Pareto table")
                if a_rank < b_rank and _practically_equivalent(
                    out.loc[a_index], out.loc[b_index], family
                ):
                    active.remove(b_index)
                    out.at[b_index, "selection_decision"] = (
                        "practically_equivalent_simpler_family"
                    )

        # Longer window and then coverage operate only within equivalent pairs.
        for criterion in ("window", "coverage"):
            for a_index in sorted(active, key=lambda i: str(out.at[i, "candidate_id"])):
                if a_index not in active:
                    continue
                for b_index in sorted(active, key=lambda i: str(out.at[i, "candidate_id"])):
                    if a_index == b_index or b_index not in active:
                        continue
                    if not _practically_equivalent(
                        out.loc[a_index], out.loc[b_index], family
                    ):
                        continue
                    if criterion == "window":
                        preferred = float(out.at[a_index, "window_days"]) > float(
                            out.at[b_index, "window_days"]
                        )
                        reason = "practically_equivalent_longer_window"
                    else:
                        preferred = float(
                            out.at[a_index, "median_support_qualified_coverage"]
                        ) > float(
                            out.at[b_index, "median_support_qualified_coverage"]
                        )
                        reason = "practically_equivalent_higher_coverage"
                    if preferred:
                        active.remove(b_index)
                        out.at[b_index, "selection_decision"] = reason

        ordered = sorted(
            active,
            key=lambda i: (
                -float(out.at[i, "window_days"]),
                -float(out.at[i, "median_support_qualified_coverage"]),
                str(out.at[i, "candidate_id"]),
            ),
        )
        for rank, index in enumerate(ordered[:max_candidates], start=1):
            out.at[index, "selected_for_confirmation"] = True
            out.at[index, "selection_rank"] = rank
            out.at[index, "selection_decision"] = "selected"
        for index in ordered[max_candidates:]:
            out.at[index, "selection_decision"] = "max_two_lexical_tiebreak"

    return out.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def weighted_inverse_ecdf(
    values: Sequence[float] | pd.Series | np.ndarray,
    weights: Sequence[float] | pd.Series | np.ndarray,
    probabilities: Sequence[float] = (0.33, 0.67),
) -> tuple[float, ...]:
    """Compute weighted quantiles by the frozen inverse-ECDF definition.

    Nonfinite value/weight pairs and zero-weight rows are ignored.  Negative
    weights hard-stop.  For each probability, the result is the first sorted
    score whose cumulative nonnegative weight reaches ``p * total_weight``.
    Stable sorting makes tied-score handling deterministic.
    """

    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape != w.shape:
        raise ValueError("values and weights must have identical shapes")
    if any(not 0 <= float(p) <= 1 for p in probabilities):
        raise ValueError("weighted quantile probabilities must lie in [0, 1]")
    finite_weight = np.isfinite(w)
    if (w[finite_weight] < 0).any():
        raise ValueError("weighted inverse ECDF forbids negative weights")
    keep = np.isfinite(x) & finite_weight & (w > 0)
    if not keep.any():
        raise ValueError("weighted inverse ECDF has no positive finite weight")
    x = x[keep]
    w = w[keep]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    answers: list[float] = []
    for probability in probabilities:
        threshold = float(probability) * total
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        index = min(index, len(x) - 1)
        answers.append(float(x[index]))
    return tuple(answers)


def derive_weighted_level_thresholds(
    development_scores: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("candidate_id", "support_threshold"),
    score_column: str = "score",
    weight_column: str = "future_7d_all_placed_exposure",
    support_column: str = "support",
    cold_start_column: str = "cold_start",
    missing_mapping_column: str = "missing_mapping",
    lower_column: str = "lower_interval",
    upper_column: str = "upper_interval",
) -> pd.DataFrame:
    """Freeze weighted development q33/q67 for each promoted support rule.

    The pre-cutoff eligible set removes missing mappings, cold starts, rows
    below the selected support, and nonfinite scores/intervals.  The later
    "interval spans both cutoffs" Unknown rule is intentionally not iterated
    into cutoff estimation because it cannot be known before the fixed cutoffs
    exist.  This one-pass rule avoids a circular, data-dependent fixed point.
    """

    required = [
        *group_columns,
        score_column,
        weight_column,
        support_column,
        cold_start_column,
        lower_column,
        upper_column,
    ]
    _require_columns(development_scores, required, "development level scores")
    scores = development_scores.copy()
    if missing_mapping_column not in scores:
        scores[missing_mapping_column] = False
    result: list[dict[str, Any]] = []
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in scores.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        identity = dict(zip(group_columns, keys))
        support_rule = int(group["support_threshold"].iloc[0])
        finite = (
            np.isfinite(pd.to_numeric(group[score_column], errors="coerce"))
            & np.isfinite(pd.to_numeric(group[lower_column], errors="coerce"))
            & np.isfinite(pd.to_numeric(group[upper_column], errors="coerce"))
        )
        eligible = (
            ~group[missing_mapping_column].fillna(True).astype(bool)
            & ~group[cold_start_column].fillna(True).astype(bool)
            & pd.to_numeric(group[support_column], errors="coerce").ge(support_rule)
            & finite
        )
        selected = group.loc[eligible]
        try:
            q33, q67 = weighted_inverse_ecdf(
                selected[score_column], selected[weight_column], (0.33, 0.67)
            )
            valid = bool(np.isfinite(q33) and np.isfinite(q67) and q33 <= q67)
            reason = "" if valid else "reversed_level_cutoffs"
        except ValueError as exc:
            q33 = np.nan
            q67 = np.nan
            valid = False
            reason = str(exc)
        result.append(
            {
                **identity,
                "q33": q33,
                "q67": q67,
                "threshold_rows": int(len(selected)),
                "threshold_total_weight": float(
                    pd.to_numeric(selected[weight_column], errors="coerce")
                    .clip(lower=0)
                    .fillna(0)
                    .sum()
                ),
                "thresholds_valid": valid,
                "threshold_invalid_reason": reason,
            }
        )
    return pd.DataFrame(result).sort_values(list(group_columns), kind="mergesort").reset_index(
        drop=True
    )


def assign_risk_levels(
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("candidate_id", "support_threshold"),
    score_column: str = "score",
    support_column: str = "support",
    cold_start_column: str = "cold_start",
    missing_mapping_column: str = "missing_mapping",
    lower_column: str = "lower_interval",
    upper_column: str = "upper_interval",
) -> pd.DataFrame:
    """Assign frozen Unknown/Low/Medium/High communication levels.

    Low is ``score <= q33``, Medium is ``q33 < score <= q67``, and High is
    ``score > q67``.  Unknown takes precedence for missing mapping, cold start,
    support below the selected threshold, nonfinite score/interval, invalid
    cutoffs, or a 95% interval spanning both cutoffs.  Missing mapping is
    explicitly not reclassified as cold start.
    """

    _require_columns(
        scores,
        [
            *group_columns,
            score_column,
            support_column,
            cold_start_column,
            lower_column,
            upper_column,
        ],
        "level score rows",
    )
    _require_columns(
        thresholds, [*group_columns, "q33", "q67", "thresholds_valid"], "thresholds"
    )
    if thresholds.duplicated(list(group_columns)).any():
        raise ValueError("threshold table has duplicate group keys")
    out = scores.copy()
    if missing_mapping_column not in out:
        out[missing_mapping_column] = False
    out = out.merge(
        thresholds[[*group_columns, "q33", "q67", "thresholds_valid"]],
        on=list(group_columns),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    out["level"] = "Unknown"
    out["unknown_reason"] = ""

    score = pd.to_numeric(out[score_column], errors="coerce")
    lower = pd.to_numeric(out[lower_column], errors="coerce")
    upper = pd.to_numeric(out[upper_column], errors="coerce")
    q33 = pd.to_numeric(out["q33"], errors="coerce")
    q67 = pd.to_numeric(out["q67"], errors="coerce")
    support_rule = pd.to_numeric(out["support_threshold"], errors="coerce")
    support = pd.to_numeric(out[support_column], errors="coerce")

    reasons = [
        (out[missing_mapping_column].fillna(True).astype(bool), "missing_mapping"),
        (
            ~out[missing_mapping_column].fillna(True).astype(bool)
            & out[cold_start_column].fillna(True).astype(bool),
            "cold_start",
        ),
        (support.lt(support_rule) | support.isna(), "support_below_threshold"),
        (~np.isfinite(score), "nonfinite_score"),
        (~np.isfinite(lower) | ~np.isfinite(upper), "nonfinite_interval"),
        (
            ~out["thresholds_valid"].fillna(False).astype(bool)
            | ~np.isfinite(q33)
            | ~np.isfinite(q67)
            | q33.gt(q67),
            "invalid_fixed_thresholds",
        ),
        (lower.le(q33) & upper.ge(q67), "interval_spans_both_cutoffs"),
    ]
    unresolved = pd.Series(True, index=out.index)
    for mask, reason in reasons:
        assign = unresolved & mask.fillna(True)
        out.loc[assign, "unknown_reason"] = reason
        unresolved &= ~assign
    out.loc[unresolved & score.le(q33), "level"] = "Low"
    out.loc[unresolved & score.gt(q33) & score.le(q67), "level"] = "Medium"
    out.loc[unresolved & score.gt(q67), "level"] = "High"
    if (unresolved & out["level"].eq("Unknown")).any():
        raise RuntimeError("finite eligible level row did not receive Low/Medium/High")
    return out


def _confirmation_primary_improvement(frame: pd.DataFrame, family: str) -> pd.Series:
    if "primary_improvement" in frame:
        return pd.to_numeric(frame["primary_improvement"], errors="coerce")
    if family == "binary":
        if "delta_logloss" in frame:
            return -pd.to_numeric(frame["delta_logloss"], errors="coerce")
        if "delta_log_loss" in frame:
            # profile_core source convention is reference minus candidate.
            return pd.to_numeric(frame["delta_log_loss"], errors="coerce")
    else:
        for column in ("parent_minus_candidate_mae", "log_mae_improvement"):
            if column in frame:
                return pd.to_numeric(frame[column], errors="coerce")
    raise KeyError("confirmation rows lack a primary-improvement field")


def confirmation_label_rubric(
    *,
    development_primary_improvement: float,
    confirmation_months: pd.DataFrame,
    target_family: str,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Apply the frozen descriptive confirmation-label rubric.

    Strong confirmation requires: favourable direction in a strict majority of
    valid confirmation months; aggregate confirmation magnitude within 50% of
    the positive development magnitude; and no material reversal in support
    ``>=5`` rows.  Partial confirmation requires favourable aggregate direction
    while at least one strong condition fails.  Non-favourable aggregate
    direction, no valid months, or advantage confined to support below five or
    cold start is Not Confirmed.  These labels are descriptive and do not form a
    thesis-pivot gate.
    """

    if target_family not in {"binary", "continuous"}:
        raise ValueError("target_family must be binary or continuous")
    months = confirmation_months.copy()
    month_column = next(
        (name for name in ("anchor_month", "calendar_month") if name in months), None
    )
    if month_column is not None:
        if months[month_column].isna().any():
            raise ValueError("confirmation month labels must be complete")
        if months[month_column].astype(str).duplicated().any():
            raise ValueError(
                "confirmation rubric requires exactly one primary row per calendar month"
            )
    if "valid" in months:
        months = months.loc[months["valid"].fillna(False).astype(bool)]
    improvement = _confirmation_primary_improvement(months, target_family)
    valid_improvement = improvement[np.isfinite(improvement)]
    n_valid = int(len(valid_improvement))
    n_favourable = int((valid_improvement > tolerance).sum())
    strict_majority = bool(n_valid > 0 and n_favourable > n_valid / 2)
    aggregate = _median_or_nan(valid_improvement)
    aggregate_favourable = bool(np.isfinite(aggregate) and aggregate > tolerance)

    development = float(development_primary_improvement)
    within_50_percent = bool(
        np.isfinite(development)
        and development > tolerance
        and aggregate_favourable
        and abs(aggregate - development) <= 0.50 * abs(development) + tolerance
    )

    if "high_support_material_reversal" in months:
        high_support_reversal = bool(
            months["high_support_material_reversal"].fillna(False).astype(bool).any()
        )
    else:
        high_support_reversal = False
    if "support_ge5_primary_improvement" in months:
        high_values = pd.to_numeric(
            months["support_ge5_primary_improvement"], errors="coerce"
        )
        finite_high = high_values[np.isfinite(high_values)]
        high_support_reversal = bool(
            high_support_reversal or (finite_high < -tolerance).any()
        )
        high_support_positive = bool((finite_high > tolerance).any())
    else:
        # Absence of the frozen high-support audit cannot earn strong/partial status.
        high_support_positive = False

    if "advantage_only_low_support_or_cold" in months:
        advantage_only_low_or_cold = bool(
            months["advantage_only_low_support_or_cold"]
            .fillna(False)
            .astype(bool)
            .any()
        )
    else:
        advantage_only_low_or_cold = bool(
            aggregate_favourable and not high_support_positive
        )

    strong = bool(
        aggregate_favourable
        and strict_majority
        and within_50_percent
        and not high_support_reversal
        and not advantage_only_low_or_cold
    )
    if strong:
        label = "Strongly confirmed"
    elif aggregate_favourable and not advantage_only_low_or_cold:
        label = "Partially confirmed"
    else:
        label = "Not confirmed"

    failed_strong_conditions = []
    for name, passed in (
        ("strict_majority", strict_majority),
        ("magnitude_within_50_percent", within_50_percent),
        ("no_high_support_material_reversal", not high_support_reversal),
        ("not_confined_below_support5_or_cold", not advantage_only_low_or_cold),
    ):
        if not passed:
            failed_strong_conditions.append(name)
    return {
        "confirmation_label": label,
        "n_valid_confirmation_months": n_valid,
        "n_favourable_confirmation_months": n_favourable,
        "strict_majority_favourable": strict_majority,
        "development_primary_improvement": development,
        "confirmation_aggregate_primary_improvement": aggregate,
        "aggregate_direction_favourable": aggregate_favourable,
        "aggregate_magnitude_within_50_percent": within_50_percent,
        "high_support_material_reversal": high_support_reversal,
        "advantage_only_below_support5_or_cold": advantage_only_low_or_cold,
        "failed_strong_conditions": ";".join(failed_strong_conditions),
        "label_is_descriptive_not_stage_gate": True,
    }


def confirmation_labels(
    development_summary: pd.DataFrame,
    confirmation_by_month: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorise :func:`confirmation_label_rubric` over frozen candidates."""

    _require_columns(
        development_summary,
        ["candidate_id", "target_family", "median_primary_improvement"],
        "development summary",
    )
    _require_columns(confirmation_by_month, ["candidate_id"], "confirmation months")
    if development_summary["candidate_id"].duplicated().any():
        raise ValueError("development summary has duplicate candidate IDs")
    rows: list[dict[str, Any]] = []
    for record in development_summary.sort_values("candidate_id", kind="mergesort").to_dict(
        orient="records"
    ):
        candidate_id = str(record["candidate_id"])
        months = confirmation_by_month.loc[
            confirmation_by_month["candidate_id"].eq(candidate_id)
        ]
        rubric = confirmation_label_rubric(
            development_primary_improvement=float(record["median_primary_improvement"]),
            confirmation_months=months,
            target_family=str(record["target_family"]),
        )
        rows.append({"candidate_id": candidate_id, **rubric})
    return pd.DataFrame(rows)


def _bootstrap_summary(
    original: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    samples: Iterable[pd.DataFrame],
    *,
    confidence: float,
    requested_replicates: int,
) -> dict[str, Any]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    estimate = float(statistic(original.copy()))
    values: list[float] = []
    for sample in samples:
        value = float(statistic(sample))
        if np.isfinite(value):
            values.append(value)
    draws = np.asarray(values, dtype=float)
    alpha = 1.0 - confidence
    if draws.size:
        lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        lower = np.nan
        upper = np.nan
    return {
        "estimate": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "requested_replicates": int(requested_replicates),
        "valid_replicates": int(draws.size),
        "bootstrap_values": draws,
        "interval_method": "percentile_linear_quantile",
    }


def entity_cluster_bootstrap(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    entity_column: str = "entity_id",
    replicates: int = 500,
    seed: int = _DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return a percentile interval from whole-entity cluster resampling.

    Each replicate draws the observed unique entities with replacement and
    retains every row belonging to a selected entity.  Repeated cluster copies
    receive ``__bootstrap_entity_draw`` so entity-level statistics can keep
    copies distinct.  The original estimate receives deterministic draw IDs as
    well.  Individual orders are never sampled independently.
    """

    _require_columns(data, [entity_column], "entity bootstrap data")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    usable = data.loc[data[entity_column].notna()].copy()
    entities = sorted(pd.unique(usable[entity_column]), key=lambda value: str(value))
    if not entities:
        raise ValueError("entity-cluster bootstrap has no nonmissing entities")
    clusters = {entity: usable.loc[usable[entity_column].eq(entity)] for entity in entities}
    original = usable.copy()
    original["__bootstrap_entity_draw"] = pd.factorize(
        original[entity_column], sort=True
    )[0]
    rng = np.random.default_rng(int(seed))

    def sample_iterator() -> Iterable[pd.DataFrame]:
        for _ in range(int(replicates)):
            draw_indices = rng.integers(0, len(entities), size=len(entities))
            pieces: list[pd.DataFrame] = []
            for draw_id, entity_index in enumerate(draw_indices):
                piece = clusters[entities[int(entity_index)]].copy()
                piece["__bootstrap_entity_draw"] = draw_id
                pieces.append(piece)
            yield pd.concat(pieces, ignore_index=True, sort=False)

    result = _bootstrap_summary(
        original,
        statistic,
        sample_iterator(),
        confidence=confidence,
        requested_replicates=int(replicates),
    )
    result.update(
        {
            "resampling_unit": "entity_cluster",
            "n_clusters": len(entities),
            "seed": int(seed),
        }
    )
    return result


def month_block_bootstrap_summary(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    month_column: str = "anchor_month",
    replicates: int = 500,
    seed: int = _DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarise calendar-month variation with whole-month block bootstrap.

    At least three distinct months are required by the frozen protocol.  Each
    replicate draws that many calendar months with replacement and carries all
    rows in each selected month.  Repeated copies receive
    ``__bootstrap_month_draw``.  This is a time-block summary, not an iid order
    bootstrap.
    """

    _require_columns(data, [month_column], "month-block bootstrap data")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    usable = data.loc[data[month_column].notna()].copy()
    months = sorted(pd.unique(usable[month_column]), key=lambda value: str(value))
    if len(months) < 3:
        raise ValueError("month-block bootstrap requires at least three months")
    blocks = {month: usable.loc[usable[month_column].eq(month)] for month in months}
    original = usable.copy()
    original["__bootstrap_month_draw"] = pd.factorize(
        original[month_column], sort=True
    )[0]
    rng = np.random.default_rng(int(seed))

    def sample_iterator() -> Iterable[pd.DataFrame]:
        for _ in range(int(replicates)):
            draw_indices = rng.integers(0, len(months), size=len(months))
            pieces: list[pd.DataFrame] = []
            for draw_id, month_index in enumerate(draw_indices):
                piece = blocks[months[int(month_index)]].copy()
                piece["__bootstrap_month_draw"] = draw_id
                pieces.append(piece)
            yield pd.concat(pieces, ignore_index=True, sort=False)

    result = _bootstrap_summary(
        original,
        statistic,
        sample_iterator(),
        confidence=confidence,
        requested_replicates=int(replicates),
    )
    result.update(
        {
            "resampling_unit": "calendar_month_block",
            "n_months": len(months),
            "month_labels": [str(month) for month in months],
            "seed": int(seed),
        }
    )
    return result


__all__ = [
    "BINARY_TARGETS",
    "CONTINUOUS_TARGETS",
    "aggregate_anchor_metrics",
    "assign_risk_levels",
    "build_candidate_catalog",
    "confirmation_label_rubric",
    "confirmation_labels",
    "derive_weighted_level_thresholds",
    "entity_cluster_bootstrap",
    "minimum_evidence_gates",
    "month_block_bootstrap_summary",
    "pareto_frontier",
    "select_confirmation_candidates",
    "stable_base_candidate_id",
    "stable_candidate_id",
    "stable_profile_spec_id",
    "target_family",
    "weighted_inverse_ecdf",
]
