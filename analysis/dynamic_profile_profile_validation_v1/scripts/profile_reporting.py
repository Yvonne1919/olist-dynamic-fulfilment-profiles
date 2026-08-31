"""Deterministic reporting for dynamic-profile statistical validation V1.

The module is deliberately a pure reporting layer.  It reads persisted profile,
anchor, entity, order and selection tables, writes the frozen output contract,
and never fits an order-level prediction model or a business policy.  The
runner can call :func:`finalize_reporting`; the smaller public helpers are kept
independent so synthetic tests can exercise aggregation, plotting and artifact
validation without Olist data.

Every PNG is rendered by reopening its paired source CSV.  Empty or wholly
invalid sources produce an explicit ``No valid data`` panel rather than a
misleading blank chart.  CSV ordering, float formatting, gzip headers, random
seeds and PNG metadata are fixed for reproducibility.
"""

from __future__ import annotations

import gzip
import gc
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.dynamic_profile_profile_validation_v1.scripts import profile_selection, selected_daily

from analysis.dynamic_profile_profile_validation_v1.scripts.profile_core import (
    CONFIG_PATH,
    OUT,
    PROFILE_BASE_COLUMNS,
    PROTECTED,
    TARGET_SPECS,
    V11,
    control_file_hashes,
    load_config,
    recursive_hashes,
    sha256_file,
    weighted_spearman,
)


FLOAT_FORMAT = "%.12g"
DATE_FORMAT = "%Y-%m-%d"
NA_REP = ""
RANDOM_SEED = 20260823
SCHEMA_VERSION = "profile_validation_v1"
# The persisted rich/daily tables are loaded with categorical, nullable-integer
# and nullable-boolean dtypes below.  Empirical storage is roughly 170--300
# bytes per row; 900 bytes per row is the deliberately conservative reporting
# peak allowance for groupby temporaries and copies.  The hard stop is a
# *combined* memory budget, so a valid full promoted grid is not rejected by an
# arbitrary candidate-count cap.
DEFAULT_MAX_IN_MEMORY_RICH_SCORING_ROWS = 30_000_000
DEFAULT_MAX_IN_MEMORY_DAILY_PROFILE_ROWS = 30_000_000
ESTIMATED_PEAK_BYTES_PER_RICH_ROW = 900
ESTIMATED_PEAK_BYTES_PER_DAILY_ROW = 900
DEFAULT_REPORTING_MEMORY_FRACTION = 0.70
DEFAULT_REPORTING_MEMORY_BUDGET_BYTES = 22_000_000_000
DAILY_TREND_SERIALIZATION_ATOL = 1e-10

REQUIRED_ARTIFACTS = (
    "PROFILE_PROTOCOL.md",
    "PROFILE_FROZEN_CONFIG.json",
    "PROFILE_SELECTION_FREEZE.json",
    "PROFILE_DATA_DICTIONARY.md",
    "PROFILE_CONSTRUCTION_AUDIT.csv",
    "PROFILE_DAILY_SCORES.csv",
    "PROFILE_SUPPORT_UNCERTAINTY.csv",
    "PROFILE_PARENT_STRUCTURE.csv",
    "PROFILE_DEVELOPMENT_RESULTS.csv",
    "PROFILE_DEVELOPMENT_BY_MONTH.csv",
    "PROFILE_PARETO_FRONTIER.csv",
    "PROFILE_SELECTED_CANDIDATES.csv",
    "PROFILE_CONFIRMATION_RESULTS.csv",
    "PROFILE_CONFIRMATION_BY_MONTH.csv",
    "PROFILE_TERMINAL_STRESS.csv",
    "PROFILE_LEVEL_RESULTS.csv",
    "PROFILE_LEVEL_TRANSITIONS.csv",
    "PROFILE_DAILY_STABILITY.csv",
    "PROFILE_FUTURE_ENTITY_TRANSFER.csv",
    "PROFILE_FUTURE_ORDER_SCORING.csv",
    "PROFILE_SUPPORT_STRATA.csv",
    "PROFILE_COLD_START_RESULTS.csv",
    "PROFILE_HRD_DIAGNOSTICS.csv",
    "PROFILE_ABLATIONS.csv",
    "PROFILE_RESULTS_SUMMARY.md",
    "PROFILE_RESULTS_SUMMARY_ZH.md",
    "BLOCKERS.md",
    "RUN_MANIFEST.json",
    "TEST_RESULTS.txt",
    "ARTIFACT_VALIDATION_REPORT.md",
)

FIGURE_STEMS = (
    "01_seller_support_vs_uncertainty",
    "02_route_support_vs_uncertainty",
    "03_raw_vs_eb_seller_scores",
    "04_raw_vs_eb_route_scores",
    "05_adjusted_vs_unadjusted_scores",
    "06_window_30_60_90_comparison",
    "07_scheme_a_vs_c_comparison",
    "08_development_future_rank_transfer",
    "09_confirmation_future_rank_transfer",
    "10_top_quintile_future_lift",
    "11_future_outcome_by_level",
    "12_daily_profile_stability",
    "13_level_transition_heatmap",
    "14_coverage_by_support_threshold",
    "15_seller_cold_start",
    "16_state_od_vs_region_od",
    "17_development_vs_confirmation",
    "18_terminal_stress",
)

FIGURE_TITLES = {
    "01_seller_support_vs_uncertainty": "Seller support versus uncertainty",
    "02_route_support_vs_uncertainty": "Route support versus uncertainty",
    "03_raw_vs_eb_seller_scores": "Raw versus EB seller scores",
    "04_raw_vs_eb_route_scores": "Raw versus EB route scores",
    "05_adjusted_vs_unadjusted_scores": "Adjusted versus unadjusted profile scores",
    "06_window_30_60_90_comparison": "30/60/90-day window comparison",
    "07_scheme_a_vs_c_comparison": "Scheme A versus Scheme C",
    "08_development_future_rank_transfer": "Development future rank transfer",
    "09_confirmation_future_rank_transfer": "Confirmation future rank transfer",
    "10_top_quintile_future_lift": "Top-quintile future lift",
    "11_future_outcome_by_level": "Future outcome by profile level",
    "12_daily_profile_stability": "Daily profile score stability",
    "13_level_transition_heatmap": "Profile-level transitions",
    "14_coverage_by_support_threshold": "Future coverage by support threshold",
    "15_seller_cold_start": "Seller cold-start analysis",
    "16_state_od_vs_region_od": "State-OD versus region-OD performance",
    "17_development_vs_confirmation": "Development versus confirmation",
    "18_terminal_stress": "Terminal stress diagnostic",
}

FIGURE_SOURCE_SCHEMA = (
    "figure_id",
    "panel",
    "x_value",
    "x_label",
    "y_value",
    "series",
    "weight",
    "valid",
    "invalid_reason",
)

METRIC_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "anchor_date", "calendar_month", "horizon_days", "stratum_type",
    "stratum_value", "metric_name", "reference_id", "aggregation",
    "n_scheduled_anchors", "n_valid_anchors", "n_orders", "n_events",
    "n_entities", "n_common_entities", "estimate", "ci_lower", "ci_upper",
    "valid", "invalid_reason",
)

CONSTRUCTION_AUDIT_SCHEMA = (
    "base_candidate_id", "snapshot_date", "period", "target", "granularity",
    "scheme", "window_days", "lag_days", "estimator", "parent_structure",
    "kappa", "history_sample", "future_denominator_sample",
    "source_interval_axis", "source_interval_start", "source_interval_end",
    "availability_cutoff", "entity_domain_count", "source_orders_observed",
    "source_orders_valid", "source_orders_excluded_negative",
    "affected_entities_negative", "profile_rows", "parent_rows",
    "cold_start_rows", "coverage_before_negative_exclusion",
    "coverage_after_negative_exclusion", "max_source_purchase_at",
    "max_source_label_available_at", "last_mature_outcome_date",
    "strict_asof_pass", "window_pass", "valid", "invalid_reason",
)

DAILY_INDEX_SCHEMA = (
    "schema_version", "relative_path", "target", "granularity", "scheme",
    "window_days", "lag_days", "estimator", "snapshot_date_min",
    "snapshot_date_max", "row_count", "sha256", "primary_key_columns",
    "sort_columns",
)

SUPPORT_UNCERTAINTY_SCHEMA = (
    "base_candidate_id", "snapshot_date", "period", "target", "granularity",
    "scheme", "window_days", "lag_days", "estimator", "parent_structure",
    "kappa", "support_stratum", "entity_count", "order_exposure",
    "median_support", "median_score", "median_posterior_se",
    "p90_posterior_se", "median_interval_width", "p90_interval_width",
    "cold_start_count", "valid", "invalid_reason",
)

PARENT_STRUCTURE_SCHEMA = (
    "base_candidate_id", "snapshot_date", "target", "granularity",
    "parent_structure", "parent_id", "parent_support", "parent_event_count",
    "parent_score", "global_score", "parent_within_variance",
    "parent_between_variance", "parent_posterior_se", "parent_interval_lower",
    "parent_interval_upper", "fallback_child_count", "parent_supported",
    "valid", "invalid_reason",
)

TERMINAL_SCHEMA = METRIC_SCHEMA + (
    "outcome_observation_rate", "followup_available_days",
    "maturity_censoring_flag", "distribution_shift_reference",
)

LEVEL_RESULTS_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "horizon_days", "level", "metric_name", "n_orders", "n_entities",
    "future_support", "estimate", "ci_lower", "ci_upper", "monotone_lmh",
    "percent_unknown", "valid", "invalid_reason",
)

LEVEL_TRANSITIONS_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "from_level", "to_level", "transition_count", "eligible_from_count",
    "transition_probability", "median_persistence_days", "ci_lower",
    "ci_upper", "valid", "invalid_reason",
)

DAILY_STABILITY_SCHEMA = (
    "base_candidate_id", "target", "granularity", "previous_snapshot_date",
    "snapshot_date", "period", "regime", "n_common_entities",
    "newly_matured_support", "day_to_day_spearman",
    "median_absolute_score_change", "p90_absolute_score_change",
    "top20_jaccard", "score_change_per_new_label",
    "entities_changing_level", "pct_entities_changing_level",
    "cold_start_entries", "cold_start_exits",
    "transition_entity_union_count", "cold_start_entry_rate", "cold_start_exit_rate",
    "valid", "invalid_reason",
)

ENTITY_TRANSFER_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "anchor_date", "horizon_days", "stratum_type", "stratum_value",
    "n_common_entities", "future_support", "unweighted_spearman",
    "weighted_spearman", "top_quintile_lift", "high_low_risk_ratio",
    "spearman_ci_lower", "spearman_ci_upper", "lift_ci_lower",
    "lift_ci_upper", "bootstrap_unit", "bootstrap_resamples", "valid",
    "invalid_reason",
)

ORDER_SCORING_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "anchor_date", "horizon_days", "order_id", "purchase_timestamp",
    "entity_id", "mapping_status", "history_support", "cold_start",
    "profile_score", "parent_score", "level", "target_observed",
    "target_value", "label_available_at", "eligible_for_metric",
)

RICH_ORDER_SCORING_SCHEMA = (
    "order_id", "purchase_timestamp", "entity_id", "mapping_status",
    "history_support", "cold_start", "profile_score", "shrinkage_score",
    "raw_score", "parent_score", "global_score", "level", "unknown_reason",
    "target_observed", "target_value", "raw_target_value",
    "label_available_at", "eligible_for_metric", "posterior_se",
    "lower_interval", "upper_interval", "candidate_id", "profile_spec_id",
    "target", "granularity", "period", "anchor_date", "horizon_days",
)

RICH_SCORING_DTYPES: dict[str, str] = {
    column: "category"
    for column in (
        "order_id", "purchase_timestamp", "entity_id", "mapping_status",
        "candidate_id", "profile_spec_id", "target", "granularity", "period",
        "anchor_date", "level", "unknown_reason", "label_available_at",
    )
}
RICH_SCORING_DTYPES.update(
    {
        "history_support": "Int32",
        "horizon_days": "Int16",
        "cold_start": "boolean",
        "target_observed": "boolean",
        "eligible_for_metric": "boolean",
    }
)

DAILY_PROFILE_DTYPES: dict[str, str] = {
    column: "category"
    for column in (
        "entity_id", "snapshot_date", "target", "granularity", "scheme",
        "estimator", "parent_structure", "base_candidate_id", "parent_id",
        "last_mature_outcome_date", "candidate_id", "profile_spec_id", "level",
        "unknown_reason", "period",
    )
}
DAILY_PROFILE_DTYPES.update(
    {
        "window_days": "Int16", "lag_days": "Int16", "support": "Int32",
        "event_count": "Float64", "active_days": "Int32",
        "profile_freshness_days": "Float64", "min_support": "Int16",
        "support_30d": "Float64", "support_90d": "Float64",
        "cold_start": "boolean",
    }
)

SUPPORT_STRATA_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "horizon_days", "support_stratum", "metric_name", "reference_id",
    "n_orders", "n_events", "n_entities", "estimate", "ci_lower",
    "ci_upper", "valid", "invalid_reason",
)

COLD_START_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "horizon_days", "mapping_status", "n_orders", "order_share",
    "metric_name", "estimate", "ci_lower", "ci_upper", "valid",
    "invalid_reason",
)

HRD_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "hrd_definition", "regime", "phase", "horizon_days", "n_days",
    "n_orders", "n_entities", "historical_support", "metric_name",
    "estimate", "ci_lower", "ci_upper", "valid", "invalid_reason",
)

ABLATION_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "horizon_days", "ablation_id", "stratum_type", "stratum_value",
    "metric_name", "reference_id", "n_orders", "n_events", "n_entities",
    "estimate", "ci_lower", "ci_upper", "valid", "invalid_reason",
)

PARETO_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "target_family",
    "granularity", "scheme", "window_days", "lag_days", "estimator",
    "parent_structure", "kappa", "support_threshold",
    "n_scheduled_anchors", "n_valid_anchors", "valid_anchor_fraction",
    "median_delta_logloss", "median_delta_brier",
    "median_parent_minus_candidate_mae", "median_weighted_spearman",
    "median_top_quintile_lift", "median_support_qualified_coverage",
    "median_daily_stability_spearman", "minimum_evidence_pass",
    "minimum_evidence_failure_reasons", "pareto_eligible",
    "pareto_nondominated", "dominated_by", "pareto_ineligible_reason",
    "selected_for_confirmation", "selection_rank", "selection_decision",
)

SELECTED_SCHEMA = (
    "candidate_id", "base_candidate_id", "profile_spec_id", "target",
    "target_family", "granularity", "scheme", "window_days", "lag_days",
    "estimator", "parent_structure", "kappa", "min_support",
    "low_medium_cutoff", "medium_high_cutoff", "selection_rank",
    "selection_decision", "confirmation_label", "confirmation_label_reason",
)

CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "PROFILE_CONSTRUCTION_AUDIT.csv": CONSTRUCTION_AUDIT_SCHEMA,
    "PROFILE_DAILY_SCORES.csv": DAILY_INDEX_SCHEMA,
    "PROFILE_SUPPORT_UNCERTAINTY.csv": SUPPORT_UNCERTAINTY_SCHEMA,
    "PROFILE_PARENT_STRUCTURE.csv": PARENT_STRUCTURE_SCHEMA,
    "PROFILE_DEVELOPMENT_RESULTS.csv": METRIC_SCHEMA,
    "PROFILE_DEVELOPMENT_BY_MONTH.csv": METRIC_SCHEMA,
    "PROFILE_PARETO_FRONTIER.csv": PARETO_SCHEMA,
    "PROFILE_SELECTED_CANDIDATES.csv": SELECTED_SCHEMA,
    "PROFILE_CONFIRMATION_RESULTS.csv": METRIC_SCHEMA,
    "PROFILE_CONFIRMATION_BY_MONTH.csv": METRIC_SCHEMA,
    "PROFILE_TERMINAL_STRESS.csv": TERMINAL_SCHEMA,
    "PROFILE_LEVEL_RESULTS.csv": LEVEL_RESULTS_SCHEMA,
    "PROFILE_LEVEL_TRANSITIONS.csv": LEVEL_TRANSITIONS_SCHEMA,
    "PROFILE_DAILY_STABILITY.csv": DAILY_STABILITY_SCHEMA,
    "PROFILE_FUTURE_ENTITY_TRANSFER.csv": ENTITY_TRANSFER_SCHEMA,
    "PROFILE_FUTURE_ORDER_SCORING.csv": ORDER_SCORING_SCHEMA,
    "PROFILE_SUPPORT_STRATA.csv": SUPPORT_STRATA_SCHEMA,
    "PROFILE_COLD_START_RESULTS.csv": COLD_START_SCHEMA,
    "PROFILE_HRD_DIAGNOSTICS.csv": HRD_SCHEMA,
    "PROFILE_ABLATIONS.csv": ABLATION_SCHEMA,
}

# These two development-selection tables are byte-frozen before confirmation.
# Their wide schemas are produced by the selection module and therefore the
# persisted header (plus its freeze hash) is authoritative at reporting time.
FROZEN_WIDE_SELECTION_TABLES = {
    "PROFILE_PARETO_FRONTIER.csv",
    "PROFILE_SELECTED_CANDIDATES.csv",
}

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "PROFILE_CONSTRUCTION_AUDIT.csv": ("base_candidate_id", "snapshot_date"),
    "PROFILE_DAILY_SCORES.csv": (
        "relative_path", "target", "granularity", "scheme", "window_days",
        "lag_days", "estimator",
    ),
    "PROFILE_SUPPORT_UNCERTAINTY.csv": (
        "base_candidate_id", "snapshot_date", "support_stratum",
    ),
    "PROFILE_PARENT_STRUCTURE.csv": (
        "base_candidate_id", "snapshot_date", "parent_id",
    ),
    "PROFILE_DEVELOPMENT_RESULTS.csv": (
        "candidate_id", "horizon_days", "stratum_type", "stratum_value",
        "metric_name", "reference_id", "aggregation",
    ),
    "PROFILE_DEVELOPMENT_BY_MONTH.csv": (
        "candidate_id", "calendar_month", "horizon_days", "stratum_type",
        "stratum_value", "metric_name", "reference_id", "aggregation",
    ),
    "PROFILE_PARETO_FRONTIER.csv": ("candidate_id",),
    "PROFILE_SELECTED_CANDIDATES.csv": ("candidate_id",),
    "PROFILE_CONFIRMATION_RESULTS.csv": (
        "candidate_id", "horizon_days", "stratum_type", "stratum_value",
        "metric_name", "reference_id", "aggregation",
    ),
    "PROFILE_CONFIRMATION_BY_MONTH.csv": (
        "candidate_id", "calendar_month", "horizon_days", "stratum_type",
        "stratum_value", "metric_name", "reference_id", "aggregation",
    ),
    "PROFILE_TERMINAL_STRESS.csv": (
        "candidate_id", "horizon_days", "stratum_type", "stratum_value",
        "metric_name", "reference_id", "aggregation",
    ),
    "PROFILE_LEVEL_RESULTS.csv": (
        "candidate_id", "period", "horizon_days", "level", "metric_name",
    ),
    "PROFILE_LEVEL_TRANSITIONS.csv": (
        "candidate_id", "period", "from_level", "to_level",
    ),
    "PROFILE_DAILY_STABILITY.csv": (
        "base_candidate_id", "previous_snapshot_date", "snapshot_date",
    ),
    "PROFILE_FUTURE_ENTITY_TRANSFER.csv": (
        "candidate_id", "period", "anchor_date", "horizon_days",
        "stratum_type", "stratum_value",
    ),
    "PROFILE_FUTURE_ORDER_SCORING.csv": (
        "candidate_id", "period", "anchor_date", "horizon_days", "order_id",
    ),
    "PROFILE_SUPPORT_STRATA.csv": (
        "candidate_id", "period", "horizon_days", "support_stratum",
        "metric_name", "reference_id",
    ),
    "PROFILE_COLD_START_RESULTS.csv": (
        "candidate_id", "period", "horizon_days", "mapping_status",
        "metric_name",
    ),
    "PROFILE_HRD_DIAGNOSTICS.csv": (
        "candidate_id", "period", "hrd_definition", "regime", "phase",
        "horizon_days", "metric_name",
    ),
    "PROFILE_ABLATIONS.csv": (
        "candidate_id", "period", "horizon_days", "ablation_id",
        "stratum_type", "stratum_value", "metric_name", "reference_id",
    ),
}

SORT_KEYS: dict[str, tuple[str, ...]] = {
    name: PRIMARY_KEYS[name] for name in PRIMARY_KEYS
}

DAILY_EXTRA_COLUMNS = (
    "candidate_id", "profile_spec_id", "min_support", "low_medium_cutoff",
    "medium_high_cutoff", "level", "unknown_reason", "period",
    "score_30d", "support_30d", "score_90d", "support_90d",
    "short_long_trend",
)
DAILY_ROW_SCHEMA = tuple(PROFILE_BASE_COLUMNS) + DAILY_EXTRA_COLUMNS
RAW_SELECTED_DAILY_ROW_SCHEMA = tuple(selected_daily.SELECTED_DAILY_COLUMNS)


@dataclass
class ReportingInputs:
    """Explicit reporting inputs; every frame may also be discovered on disk."""

    daily_profiles: pd.DataFrame | None = None
    parent_profiles: pd.DataFrame | None = None
    construction_audit: pd.DataFrame | None = None
    anchor_metrics: pd.DataFrame | None = None
    entity_rows: pd.DataFrame | None = None
    order_scoring: pd.DataFrame | None = None
    support_strata: pd.DataFrame | None = None
    pareto: pd.DataFrame | None = None
    selected_candidates: pd.DataFrame | None = None
    hrd_days: pd.DataFrame | None = None
    hrd_phases: pd.DataFrame | None = None
    confirmation_labels: pd.DataFrame | None = None


def _empty(schema: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(schema))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_token(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:20]


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _read_csv(
    path: Path,
    *,
    dtype: Mapping[str, str] | None = None,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            low_memory=False,
            dtype=dtype,
            usecols=list(usecols) if usecols is not None else None,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_first(paths: Iterable[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return _read_csv(path)
    return pd.DataFrame()


def _read_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [_read_csv(path) for path in sorted(paths) if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalise_schema(frame: pd.DataFrame | None, schema: Sequence[str]) -> pd.DataFrame:
    result = pd.DataFrame() if frame is None else frame.copy()
    for column in schema:
        if column not in result:
            result[column] = np.nan
    return result.loc[:, list(schema)]


def _sort_frame(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    actual = [key for key in keys if key in frame]
    if not actual or frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(actual, kind="mergesort", na_position="last").reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path, schema: Sequence[str], keys: Sequence[str] = ()) -> None:
    """Write an exact-schema deterministic analytical CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _sort_frame(_normalise_schema(frame, schema), keys)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data.to_csv(
        temporary,
        index=False,
        float_format=FLOAT_FORMAT,
        date_format=DATE_FORMAT,
        na_rep=NA_REP,
        lineterminator="\n",
    )
    os.replace(temporary, path)


def write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        data = (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False, default=str,
            )
            + "\n"
        ).encode("utf-8")
    else:
        data = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            + "\n"
        ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> str:
    """Write fixed-header gzip CSV and return its SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _sort_frame(
        _normalise_schema(frame, DAILY_ROW_SCHEMA),
        ("candidate_id", "snapshot_date", "entity_id"),
    )
    plain_handle = tempfile.NamedTemporaryFile(
        prefix="profile_daily_", suffix=".csv", dir=path.parent, delete=False
    )
    plain_path = Path(plain_handle.name)
    plain_handle.close()
    compressed_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        data.to_csv(
            plain_path, index=False, float_format=FLOAT_FORMAT,
            date_format=DATE_FORMAT, na_rep=NA_REP, lineterminator="\n",
        )
        with plain_path.open("rb") as source, compressed_path.open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as zipper:
                shutil.copyfileobj(source, zipper, length=8 * 1024 * 1024)
        os.replace(compressed_path, path)
    finally:
        plain_path.unlink(missing_ok=True)
        compressed_path.unlink(missing_ok=True)
    return sha256_file(path)


def _period_for_date(value: object) -> str:
    date = pd.Timestamp(value)
    if pd.isna(date):
        return ""
    if pd.Timestamp("2017-04-01") <= date < pd.Timestamp("2018-01-01"):
        return "development"
    if pd.Timestamp("2018-01-01") <= date < pd.Timestamp("2018-07-01"):
        return "confirmation"
    if pd.Timestamp("2018-07-01") <= date <= pd.Timestamp("2018-08-30"):
        return "terminal"
    return "warmup_or_outside_evaluation"


def _target_family(target: object) -> str:
    spec = TARGET_SPECS.get(str(target), {})
    return str(spec.get("kind", ""))


def _support_stratum(support: object, mapping_status: object = "seen") -> str:
    if str(mapping_status) == "missing_mapping":
        return "missing_mapping"
    try:
        value = int(float(support))
    except (TypeError, ValueError):
        return "missing_support"
    if value <= 0:
        return "support_0_cold_start"
    if value < 5:
        return "support_1_4"
    if value < 10:
        return "support_5_9"
    if value < 20:
        return "support_10_19"
    return "support_20_plus"


def _profile_spec_id(base_candidate_id: object) -> str:
    return "ps_" + _stable_token(base_candidate_id)


def _coerce_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    text = series.astype("string").str.lower()
    return text.map({"true": True, "1": True, "false": False, "0": False}).fillna(default).astype(bool)


def discover_reporting_inputs(
    output_dir: str | Path = OUT,
    work_dir: str | Path | None = None,
) -> ReportingInputs:
    """Discover current runner/selection tables without mutating them."""

    output = Path(output_dir)
    work = Path(work_dir) if work_dir is not None else output / "working"
    freeze_candidates = pd.DataFrame()
    freeze_path = output / "PROFILE_SELECTION_FREEZE.json"
    if freeze_path.exists():
        try:
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze_candidates = pd.DataFrame(freeze.get("promoted_candidates", []))
        except (json.JSONDecodeError, OSError, TypeError):
            freeze_candidates = pd.DataFrame()
    selected = _read_first(
        (
            output / "PROFILE_SELECTED_CANDIDATES.csv",
            work / "PROFILE_SELECTED_CANDIDATES.csv",
            work / "SELECTED_CANDIDATES.csv",
        )
    )
    if selected.empty and not freeze_candidates.empty:
        selected = freeze_candidates
    daily_paths = (
        work / "SELECTED_DAILY_PROFILE_ROWS.csv.gz",
        work / "SELECTED_DAILY_PROFILES.csv.gz",
        work / "PROFILE_DAILY_PROFILE_ROWS.csv.gz",
        work / "SELECTED_DAILY_PROFILE_ROWS.csv",
        work / "SELECTED_DAILY_PROFILES.csv",
    )
    daily_path = next((path for path in daily_paths if path.exists()), None)
    daily = (
        _read_csv(daily_path, dtype=DAILY_PROFILE_DTYPES)
        if daily_path is not None else pd.DataFrame()
    )
    exact_rich_path = work / "SELECTED_ORDER_SCORING_RICH.csv.gz"
    rich_scoring = (
        _read_csv(exact_rich_path, dtype=RICH_SCORING_DTYPES)
        if exact_rich_path.exists() else pd.DataFrame()
    )
    if exact_rich_path.exists():
        if tuple(rich_scoring.columns) != RICH_ORDER_SCORING_SCHEMA:
            raise ValueError(
                "working rich scoring schema mismatch: "
                f"{tuple(rich_scoring.columns)} != {RICH_ORDER_SCORING_SCHEMA}"
            )
        rich_key = [
            "candidate_id", "period", "anchor_date", "horizon_days", "order_id"
        ]
        if not rich_scoring.empty and (
            rich_scoring[rich_key].isna().any().any()
            or rich_scoring.duplicated(rich_key).any()
        ):
            raise ValueError("working rich scoring primary key is missing or duplicated")
    if rich_scoring.empty and not exact_rich_path.exists():
        rich_scoring = _read_first(
            (
                work / "SELECTED_ORDER_SCORING.csv",
                work / "SELECTED_FUTURE_ORDER_SCORING.csv",
                work / "PROFILE_FUTURE_ORDER_SCORING_RICH.csv",
            )
        )
    if rich_scoring.empty and not exact_rich_path.exists():
        rich_scoring = _read_many(
            (work / "selected_evaluation_parts").glob("*/scoring.csv")
        )
    if rich_scoring.empty and not exact_rich_path.exists():
        rich_scoring = _read_csv(output / "PROFILE_FUTURE_ORDER_SCORING.csv")
    return ReportingInputs(
        daily_profiles=daily,
        parent_profiles=_read_first(
            (
                work / "SELECTED_PARENT_ROWS.csv",
                work / "SELECTED_DAILY_PARENT_ROWS.csv",
                work / "PARENT_PROFILE_ROWS.csv",
            )
        ),
        construction_audit=_read_first(
            (
                work / "SELECTED_CONSTRUCTION_AUDIT.csv",
                work / "DEVELOPMENT_CONSTRUCTION_AUDIT.csv",
            )
        ),
        anchor_metrics=_read_first(
            (
                work / "SELECTED_ANCHOR_METRICS.csv",
                work / "DEVELOPMENT_ANCHOR_METRICS.csv",
            )
        ),
        entity_rows=_read_first((work / "SELECTED_ENTITY_ROWS.csv",)),
        order_scoring=rich_scoring,
        support_strata=_read_first(
            (
                work / "SELECTED_ANCHOR_STRATA.csv",
                work / "DEVELOPMENT_ANCHOR_STRATA.csv",
            )
        ),
        pareto=_read_first(
            (
                output / "PROFILE_PARETO_FRONTIER.csv",
                work / "PROFILE_PARETO_FRONTIER.csv",
                work / "DEVELOPMENT_PARETO_FRONTIER.csv",
            )
        ),
        selected_candidates=selected,
        hrd_days=_read_first(
            (
                work / "HRD_DAILY_LABELS.csv",
                work / "SELECTED_HRD_DAYS.csv",
                V11 / "DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv",
            )
        ),
        hrd_phases=_read_first(
            (
                work / "HRD_EVENT_PHASES.csv",
                V11 / "HRD_EVENT_PHASES.csv",
            )
        ),
        confirmation_labels=_read_first((work / "CONFIRMATION_LABELS.csv",)),
    )


def normalize_selected_candidates(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise the immutable promoted-candidate table.

    An empty table is a valid negative selection outcome and remains empty with
    the exact schema.  No fallback candidate is invented.
    """

    if frame is None or frame.empty:
        return _empty(SELECTED_SCHEMA)
    out = frame.copy()
    aliases = {
        "support_threshold": "min_support",
        "q33": "low_medium_cutoff",
        "q67": "medium_high_cutoff",
    }
    for source, destination in aliases.items():
        if destination not in out and source in out:
            out[destination] = out[source]
    if "base_candidate_id" not in out and "candidate_id" in out:
        out["base_candidate_id"] = out["candidate_id"].astype(str).str.replace(
            r"\|min_support=\d+$", "", regex=True
        )
    if "candidate_id" not in out and "base_candidate_id" in out:
        support = pd.to_numeric(out.get("min_support", 5), errors="coerce").fillna(5).astype(int)
        out["candidate_id"] = [
            f"{base}|min_support={threshold}"
            for base, threshold in zip(out["base_candidate_id"], support)
        ]
    if "profile_spec_id" not in out:
        out["profile_spec_id"] = out["base_candidate_id"].map(_profile_spec_id)
    if "target_family" not in out:
        out["target_family"] = out.get("target", pd.Series(index=out.index, dtype=object)).map(_target_family)
    defaults: dict[str, object] = {
        "min_support": np.nan,
        "low_medium_cutoff": np.nan,
        "medium_high_cutoff": np.nan,
        "selection_rank": np.nan,
        "selection_decision": "selected",
        "confirmation_label": "",
        "confirmation_label_reason": "",
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    out = _normalise_schema(out, SELECTED_SCHEMA)
    if out["candidate_id"].isna().any() or out["candidate_id"].duplicated().any():
        raise ValueError("selected candidates require unique nonmissing candidate_id")
    return _sort_frame(out, ("target", "granularity", "selection_rank", "candidate_id"))


def normalize_pareto(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty(PARETO_SCHEMA)
    out = frame.copy()
    if "support_threshold" not in out and "min_support" in out:
        out["support_threshold"] = out["min_support"]
    if "target_family" not in out:
        out["target_family"] = out.get("target", pd.Series(index=out.index, dtype=object)).map(_target_family)
    if "base_candidate_id" not in out and "candidate_id" in out:
        out["base_candidate_id"] = out["candidate_id"].astype(str).str.replace(
            r"\|min_support=\d+$", "", regex=True
        )
    result = _normalise_schema(out, PARETO_SCHEMA)
    if not result.empty and result["candidate_id"].duplicated().any():
        raise ValueError("Pareto table has duplicate candidate_id")
    return _sort_frame(result, ("target", "granularity", "candidate_id"))


def _candidate_metadata(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "base_candidate_id", "target", "granularity", "scheme",
        "window_days", "lag_days", "estimator", "parent_structure", "kappa",
        "min_support", "low_medium_cutoff", "medium_high_cutoff", "profile_spec_id",
    ]
    return selected.loc[:, columns].drop_duplicates("candidate_id") if not selected.empty else pd.DataFrame(columns=columns)


def _assign_daily_levels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    score = pd.to_numeric(out["score"], errors="coerce")
    support = pd.to_numeric(out["support"], errors="coerce")
    lower = pd.to_numeric(out["lower_interval"], errors="coerce")
    upper = pd.to_numeric(out["upper_interval"], errors="coerce")
    q33 = pd.to_numeric(out["low_medium_cutoff"], errors="coerce")
    q67 = pd.to_numeric(out["medium_high_cutoff"], errors="coerce")
    minimum = pd.to_numeric(out["min_support"], errors="coerce")
    cold = _coerce_bool(out["cold_start"], default=True)
    reason = pd.Series("", index=out.index, dtype="object")
    unknown = pd.Series(False, index=out.index)
    rules = (
        (cold, "cold_start"),
        (support.isna() | support.lt(minimum), "below_min_support"),
        (~np.isfinite(score), "nonfinite_score"),
        (~np.isfinite(lower) | ~np.isfinite(upper), "nonfinite_interval"),
        (~np.isfinite(q33) | ~np.isfinite(q67) | q33.gt(q67), "invalid_frozen_cutoffs"),
        (lower.le(q33) & upper.ge(q67), "interval_spans_both_cutoffs"),
    )
    for mask, text in rules:
        assign = ~unknown & mask.fillna(True)
        reason.loc[assign] = text
        unknown |= assign
    out["level"] = np.select(
        [unknown, score.le(q33), score.le(q67)],
        ["Unknown", "Low", "Medium"],
        default="High",
    )
    out["unknown_reason"] = reason
    return out


def prepare_daily_profiles(
    daily_profiles: pd.DataFrame | None,
    selected_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Accept either base-only or ``selected_daily``-expanded daily rows.

    Base-only input is expanded by immutable candidate/support metadata.
    Expanded input is validated against that metadata without a second
    many-to-many expansion.  Levels are always recomputed and checked, while
    the selected-daily 30d/90d component scores and trend are preserved.
    """

    if daily_profiles is None or daily_profiles.empty:
        return _empty(DAILY_ROW_SCHEMA)
    if selected_candidates.empty:
        raise ValueError("daily profile rows exist but the immutable promotion set is empty")
    profiles = daily_profiles.copy()
    missing = [column for column in PROFILE_BASE_COLUMNS if column not in profiles]
    if missing:
        raise KeyError(f"daily profile rows missing required columns: {missing}")
    profiles["snapshot_date"] = pd.to_datetime(
        profiles["snapshot_date"].astype("string"), errors="coerce"
    )
    if "last_mature_outcome_date" in profiles:
        profiles["last_mature_outcome_date"] = pd.to_datetime(
            profiles["last_mature_outcome_date"].astype("string"),
            errors="coerce",
        )
    metadata = _candidate_metadata(selected_candidates)
    expanded = "candidate_id" in profiles and profiles["candidate_id"].notna().any()
    input_level = profiles["level"].astype(str).copy() if expanded and "level" in profiles else None
    input_period = profiles["period"].astype(str).copy() if expanded and "period" in profiles else None
    for source, destination in (("q33", "low_medium_cutoff"), ("q67", "medium_high_cutoff")):
        if destination not in profiles and source in profiles:
            profiles[destination] = profiles[source]
    if expanded:
        if profiles.duplicated(["candidate_id", "snapshot_date", "entity_id"]).any():
            raise ValueError("expanded selected-daily input has duplicate primary keys")
        expected = metadata[
            [
                "candidate_id", "base_candidate_id", "profile_spec_id", "min_support",
                "low_medium_cutoff", "medium_high_cutoff",
            ]
        ].rename(
            columns={column: f"_expected_{column}" for column in (
                "base_candidate_id", "profile_spec_id", "min_support",
                "low_medium_cutoff", "medium_high_cutoff",
            )}
        )
        profiles = profiles.merge(
            expected, on="candidate_id", how="left", validate="many_to_one", sort=False
        )
        if profiles["_expected_base_candidate_id"].isna().any():
            unknown = sorted(
                set(profiles.loc[profiles["_expected_base_candidate_id"].isna(), "candidate_id"].astype(str))
            )
            raise ValueError(f"expanded daily rows contain non-promoted candidates: {unknown[:3]}")
        for column in ("base_candidate_id", "profile_spec_id"):
            expected_column = f"_expected_{column}"
            if column in profiles:
                matches = profiles[column].astype(str).eq(profiles[expected_column].astype(str))
                if not matches.all():
                    raise ValueError(f"expanded daily {column} disagrees with immutable selection")
            profiles[column] = profiles[expected_column]
        for column in ("min_support", "low_medium_cutoff", "medium_high_cutoff"):
            expected_column = f"_expected_{column}"
            if column in profiles:
                observed = pd.to_numeric(profiles[column], errors="coerce")
                wanted = pd.to_numeric(profiles[expected_column], errors="coerce")
                matches = np.isclose(observed, wanted, rtol=0, atol=1e-12, equal_nan=True)
                if not bool(np.all(matches)):
                    raise ValueError(f"expanded daily {column} disagrees with immutable selection")
            profiles[column] = profiles[expected_column]
        profiles = profiles.drop(columns=[column for column in profiles if column.startswith("_expected_")])
    else:
        base = profiles.loc[:, PROFILE_BASE_COLUMNS].copy()
        if base.duplicated(["base_candidate_id", "snapshot_date", "entity_id"]).any():
            raise ValueError("base-only daily input has duplicate profile primary keys")
        profiles = base.merge(
            metadata[
                [
                    "candidate_id", "base_candidate_id", "profile_spec_id", "min_support",
                    "low_medium_cutoff", "medium_high_cutoff",
                ]
            ],
            on="base_candidate_id",
            how="inner",
            validate="many_to_many",
            sort=False,
        )
    for column in ("score_30d", "support_30d", "score_90d", "support_90d", "short_long_trend"):
        if column not in profiles:
            profiles[column] = np.nan
    score_30 = pd.to_numeric(profiles["score_30d"], errors="coerce")
    score_90 = pd.to_numeric(profiles["score_90d"], errors="coerce")
    trend = pd.to_numeric(profiles["short_long_trend"], errors="coerce")
    both = np.isfinite(score_30) & np.isfinite(score_90)
    if not np.isclose(
        trend[both], (score_30 - score_90)[both], rtol=0,
        atol=DAILY_TREND_SERIALIZATION_ATOL,
        equal_nan=False,
    ).all():
        raise ValueError("expanded daily short_long_trend is not score_30d - score_90d")
    if (trend[~both].notna()).any():
        raise ValueError("daily trend must be missing when either component score is missing")
    profiles.loc[both, "short_long_trend"] = (score_30 - score_90)[both]
    profiles.loc[~both, "short_long_trend"] = np.nan
    profiles["cold_start"] = _coerce_bool(profiles["cold_start"]) | pd.to_numeric(
        profiles["support"], errors="coerce"
    ).fillna(0).eq(0)
    profiles = _assign_daily_levels(profiles)
    profiles["period"] = profiles["snapshot_date"].map(_period_for_date)
    if input_level is not None:
        recomputed = profiles["level"].astype(str)
        if len(input_level) != len(recomputed) or not input_level.reset_index(drop=True).eq(recomputed.reset_index(drop=True)).all():
            raise ValueError("expanded daily level disagrees with frozen reporting rule")
    if input_period is not None:
        recomputed_period = profiles["period"].astype(str)
        if len(input_period) != len(recomputed_period) or not input_period.reset_index(drop=True).eq(recomputed_period.reset_index(drop=True)).all():
            raise ValueError("expanded daily period disagrees with frozen date boundaries")
    profiles = _normalise_schema(profiles, DAILY_ROW_SCHEMA)
    key = ["candidate_id", "snapshot_date", "entity_id"]
    if profiles[key].isna().any().any() or profiles.duplicated(key).any():
        raise ValueError("daily selected profile primary key is missing or duplicated")
    return _sort_frame(profiles, key)


def write_daily_artifacts(
    daily_rows: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, Path]:
    """Persist deterministic full rows plus the compact exact-schema index."""

    output = Path(output_dir)
    row_path = output / "PROFILE_DAILY_SCORES.csv.gz"
    digest = write_deterministic_gzip_csv(daily_rows, row_path)
    rows: list[dict[str, object]] = []
    group_columns = [
        "target", "granularity", "scheme", "window_days", "lag_days", "estimator",
    ]
    if not daily_rows.empty:
        for key, group in daily_rows.groupby(
            group_columns, sort=True, dropna=False, observed=True
        ):
            record = dict(zip(group_columns, key))
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "relative_path": row_path.name,
                    **record,
                    "snapshot_date_min": pd.to_datetime(group["snapshot_date"]).min(),
                    "snapshot_date_max": pd.to_datetime(group["snapshot_date"]).max(),
                    "row_count": int(len(group)),
                    "sha256": digest,
                    "primary_key_columns": "candidate_id|snapshot_date|entity_id",
                    "sort_columns": "candidate_id|snapshot_date|entity_id",
                }
            )
    index = _normalise_schema(pd.DataFrame(rows), DAILY_INDEX_SCHEMA)
    write_csv(index, output / "PROFILE_DAILY_SCORES.csv", DAILY_INDEX_SCHEMA, SORT_KEYS["PROFILE_DAILY_SCORES.csv"])
    return index, row_path


def normalize_order_scoring(
    frame: pd.DataFrame | None,
    selected_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact public scoring rows and a richer internal copy for ablations."""

    if frame is None or frame.empty:
        return _empty(ORDER_SCORING_SCHEMA), pd.DataFrame()
    rich = frame.copy()
    for column in ("purchase_timestamp", "anchor_date", "label_available_at"):
        if column in rich:
            rich[column] = pd.to_datetime(
                rich[column].astype("string"), errors="coerce"
            )
    if "candidate_id" not in rich and "base_candidate_id" in rich:
        lookup = selected_candidates[["base_candidate_id", "candidate_id"]]
        rich = rich.merge(lookup, on="base_candidate_id", how="left", validate="many_to_one")
    if "base_candidate_id" not in rich and "candidate_id" in rich:
        rich["base_candidate_id"] = rich["candidate_id"].astype(str).str.replace(
            r"\|min_support=\d+$", "", regex=True
        )
    metadata = _candidate_metadata(selected_candidates)
    if not metadata.empty:
        add = metadata[["candidate_id", "target", "granularity"]].rename(
            columns={"target": "_selected_target", "granularity": "_selected_granularity"}
        )
        rich = rich.merge(add, on="candidate_id", how="left", validate="many_to_one")
        if "target" not in rich:
            rich["target"] = rich["_selected_target"]
        else:
            rich["target"] = rich["target"].fillna(rich["_selected_target"])
        if "granularity" not in rich:
            rich["granularity"] = rich["_selected_granularity"]
        else:
            rich["granularity"] = rich["granularity"].fillna(rich["_selected_granularity"])
        rich = rich.drop(columns=["_selected_target", "_selected_granularity"])
    aliases = {
        "history_support": ("support",),
        "profile_score": ("score",),
        "purchase_timestamp": ("order_purchase_timestamp",),
    }
    for destination, choices in aliases.items():
        if destination not in rich:
            for choice in choices:
                if choice in rich:
                    rich[destination] = rich[choice]
                    break
    if "period" not in rich and "anchor_date" in rich:
        rich["period"] = pd.to_datetime(rich["anchor_date"], errors="coerce").map(_period_for_date)
    if "mapping_status" not in rich:
        missing = rich.get("entity_id", pd.Series(index=rich.index, dtype=object)).isna()
        cold = _coerce_bool(rich.get("cold_start", pd.Series(False, index=rich.index)))
        rich["mapping_status"] = np.select(
            [missing, cold], ["missing_mapping", "mapped_cold_start"], default="seen"
        )
    if "cold_start" not in rich:
        rich["cold_start"] = rich["mapping_status"].eq("mapped_cold_start")
    if "target_observed" not in rich:
        rich["target_observed"] = pd.to_numeric(
            rich.get("target_value", pd.Series(index=rich.index, dtype=float)), errors="coerce"
        ).notna()
    if "eligible_for_metric" not in rich:
        rich["eligible_for_metric"] = (
            _coerce_bool(rich["target_observed"])
            & rich["mapping_status"].ne("missing_mapping")
            & np.isfinite(pd.to_numeric(rich.get("profile_score"), errors="coerce"))
        )
    if "level" not in rich:
        rich["level"] = "Unknown"
    key = ["candidate_id", "period", "anchor_date", "horizon_days", "order_id"]
    if rich[key].isna().any().any() or rich.duplicated(key).any():
        raise ValueError("future order scoring primary key is missing or duplicated")
    rich = _sort_frame(rich, key)
    public = _normalise_schema(rich, ORDER_SCORING_SCHEMA)
    public["cold_start"] = _coerce_bool(public["cold_start"])
    public["target_observed"] = _coerce_bool(public["target_observed"])
    public["eligible_for_metric"] = _coerce_bool(public["eligible_for_metric"])
    if not public.empty and (public[key].isna().any().any() or public.duplicated(key).any()):
        raise ValueError("future order scoring primary key is missing or duplicated")
    if not public[key].reset_index(drop=True).equals(
        rich[key].reset_index(drop=True)
    ):
        raise ValueError("public and rich order-scoring row order diverged")
    return public, rich


def _metric_reference(metric: str) -> tuple[str, str]:
    public = {
        "delta_log_loss": (
            "delta_log_loss_candidate_minus_reference", "best_parent_or_global"
        ),
        "delta_brier": (
            "delta_brier_candidate_minus_reference", "best_parent_or_global"
        ),
        "log_mae_improvement": (
            "parent_minus_candidate_log_mae", "best_parent_or_global"
        ),
        "weighted_spearman": (
            "weighted_future_spearman", "future_entity_outcome"
        ),
        "unweighted_spearman": (
            "unweighted_future_spearman", "future_entity_outcome"
        ),
        "top_quintile_lift": (
            "top_quintile_future_lift", "all_future_entities"
        ),
    }
    if metric in public:
        return public[metric]
    for prefix, reference in (("parent_", "parent"), ("global_", "global"), ("raw_", "P0_raw")):
        if metric.startswith(prefix):
            return metric[len(prefix):], reference
    return metric, "candidate"


ANCHOR_METRIC_COLUMNS = (
    "log_loss", "brier", "citl", "calibration_slope", "average_precision",
    "roc_auc", "top10_order_lift", "log_mae", "log_rmse",
    "future_mean_days", "future_median_days", "unweighted_spearman",
    "weighted_spearman", "top_quintile_lift", "delta_log_loss",
    "delta_brier", "log_mae_improvement", "future_seen_coverage",
    "support_qualified_coverage",
    "parent_log_loss", "parent_brier", "parent_citl",
    "parent_calibration_slope", "parent_average_precision", "parent_roc_auc",
    "parent_log_mae", "parent_log_rmse", "global_log_loss", "global_brier",
    "global_citl", "global_calibration_slope", "global_average_precision",
    "global_roc_auc", "global_log_mae", "global_log_rmse", "raw_log_loss",
    "raw_brier", "raw_citl", "raw_calibration_slope", "raw_average_precision",
    "raw_roc_auc", "raw_log_mae", "raw_log_rmse",
)


def _month_block_ci(values: pd.Series, months: pd.Series, identity: object) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    usable = pd.DataFrame({"value": numeric, "month": months.astype(str)}).dropna()
    unique = sorted(usable["month"].unique())
    if len(unique) < 3:
        return np.nan, np.nan
    groups = {month: usable.loc[usable["month"].eq(month), "value"].to_numpy(dtype=float) for month in unique}
    rng = np.random.default_rng(_stable_seed(RANDOM_SEED, "month_block", identity))
    replicates = np.empty(500, dtype=float)
    for index in range(500):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        block = np.concatenate([groups[str(month)] for month in sampled])
        replicates[index] = float(np.median(block))
    return tuple(float(x) for x in np.quantile(replicates, [0.025, 0.975]))


def _scheduled_anchor_count(config: Mapping[str, object], period: str, horizon: int) -> int:
    key = f"{period}_{int(horizon)}d"
    return int(config["time"]["expected_valid_anchor_counts"].get(key, 0))


def aggregate_anchor_results(
    anchor_metrics: pd.DataFrame | None,
    selected_candidates: pd.DataFrame,
    period: str,
    *,
    by_month: bool = False,
    config: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Convert wide selected-anchor metrics into the frozen long result schema."""

    schema = TERMINAL_SCHEMA if period == "terminal" and not by_month else METRIC_SCHEMA
    if anchor_metrics is None or anchor_metrics.empty:
        return _empty(schema)
    cfg = load_config() if config is None else config
    frame = anchor_metrics.copy()
    if "period" not in frame and "anchor_date" in frame:
        frame["period"] = pd.to_datetime(frame["anchor_date"], errors="coerce").map(_period_for_date)
    frame = frame.loc[frame["period"].astype(str).eq(period)].copy()
    if frame.empty:
        return _empty(schema)
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"], errors="coerce")
    if "calendar_month" not in frame:
        frame["calendar_month"] = frame["anchor_date"].dt.strftime("%Y-%m")
    metadata = _candidate_metadata(selected_candidates)
    if "candidate_id" not in frame and "base_candidate_id" in frame and not metadata.empty:
        frame = frame.merge(metadata[["base_candidate_id", "candidate_id"]], on="base_candidate_id", how="left")
    if "base_candidate_id" not in frame and "candidate_id" in frame:
        frame["base_candidate_id"] = frame["candidate_id"].astype(str).str.replace(
            r"\|min_support=\d+$", "", regex=True
        )
    if not metadata.empty:
        additions = metadata[["candidate_id", "target", "granularity"]].rename(
            columns={"target": "_target", "granularity": "_granularity"}
        )
        frame = frame.merge(additions, on="candidate_id", how="left", validate="many_to_one")
        for column in ("target", "granularity"):
            if column not in frame:
                frame[column] = frame[f"_{column}"]
            else:
                frame[column] = frame[column].fillna(frame[f"_{column}"])
        frame = frame.drop(columns=["_target", "_granularity"])
    if "valid" not in frame:
        frame["valid"] = True
    frame["valid"] = _coerce_bool(frame["valid"])
    group_columns = ["candidate_id", "base_candidate_id", "target", "granularity", "horizon_days"]
    if by_month:
        group_columns.append("calendar_month")
    rows: list[dict[str, object]] = []
    metric_columns = [column for column in ANCHOR_METRIC_COLUMNS if column in frame]
    for keys, group in frame.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        valid_rows = group.loc[group["valid"]]
        for source_metric in metric_columns:
            values = pd.to_numeric(valid_rows[source_metric], errors="coerce")
            # profile_core emits reference-candidate for binary deltas; the
            # public frozen convention is candidate-reference (smaller wins).
            if source_metric in {"delta_log_loss", "delta_brier"}:
                values = -values
            finite = np.isfinite(values)
            metric, reference = _metric_reference(source_metric)
            estimate = float(np.median(values[finite])) if finite.any() else np.nan
            if by_month:
                lower, upper = np.nan, np.nan
                aggregation = "median_within_calendar_month"
            else:
                lower, upper = _month_block_ci(
                    values[finite], valid_rows.loc[finite, "calendar_month"],
                    (identity.get("candidate_id"), identity.get("horizon_days"), source_metric, period),
                ) if finite.any() else (np.nan, np.nan)
                aggregation = "median_across_scheduled_anchors"
            n_orders_column = "future_target_valid_orders" if "future_target_valid_orders" in valid_rows else "n_orders"
            n_events_column = "future_events" if "future_events" in valid_rows else "n_events"
            common_column = "n_common_entities"
            record: dict[str, object] = {
                **identity,
                "period": period,
                "anchor_date": np.nan,
                "calendar_month": identity.get("calendar_month", ""),
                "stratum_type": "overall",
                "stratum_value": "all",
                "metric_name": metric,
                "reference_id": reference,
                "aggregation": aggregation,
                "n_scheduled_anchors": _scheduled_anchor_count(cfg, period, int(identity["horizon_days"])),
                "n_valid_anchors": int(finite.sum()),
                "n_orders": int(pd.to_numeric(_series(valid_rows, n_orders_column, 0), errors="coerce").fillna(0).sum()),
                "n_events": float(pd.to_numeric(_series(valid_rows, n_events_column), errors="coerce").sum(min_count=1)),
                "n_entities": int(pd.to_numeric(_series(valid_rows, common_column, 0), errors="coerce").fillna(0).sum()),
                "n_common_entities": int(pd.to_numeric(_series(valid_rows, common_column, 0), errors="coerce").fillna(0).sum()),
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid": bool(finite.any()),
                "invalid_reason": "" if finite.any() else "no_finite_valid_anchor_metric",
            }
            if period == "terminal" and not by_month:
                scoring_obs = pd.to_numeric(_series(group, "outcome_observation_rate"), errors="coerce")
                record.update(
                    {
                        "outcome_observation_rate": float(scoring_obs.median()) if np.isfinite(scoring_obs).any() else np.nan,
                        "followup_available_days": float(pd.to_numeric(_series(group, "followup_available_days"), errors="coerce").median()),
                        "maturity_censoring_flag": bool(
                            _coerce_bool(group.get("maturity_censoring_flag", pd.Series(False, index=group.index))).any()
                        ),
                        "distribution_shift_reference": "locked_confirmation",
                    }
                )
            rows.append(record)
    return _sort_frame(_normalise_schema(pd.DataFrame(rows), schema), PRIMARY_KEYS[
        "PROFILE_TERMINAL_STRESS.csv" if schema == TERMINAL_SCHEMA else (
            f"PROFILE_{period.upper()}_BY_MONTH.csv" if by_month else f"PROFILE_{period.upper()}_RESULTS.csv"
        )
    ] if period in {"development", "confirmation"} else PRIMARY_KEYS["PROFILE_TERMINAL_STRESS.csv"])


def _series(frame: pd.DataFrame, column: str, default: object = np.nan) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index)


def append_confirmation_label_rows(
    confirmation_results: pd.DataFrame,
    anchor_metrics: pd.DataFrame | None,
    raw_support_strata: pd.DataFrame | None,
    selected_candidates: pd.DataFrame,
    development_results: pd.DataFrame,
) -> pd.DataFrame:
    """Persist the frozen descriptive confirmation rubric as metric rows.

    The immutable selected-candidate file is never modified.  Label evidence is
    added to the non-frozen confirmation results using the existing long
    schema, including the label string in ``stratum_value`` and each rubric
    condition as a separate numeric metric.
    """

    if selected_candidates.empty:
        return confirmation_results
    anchors = pd.DataFrame() if anchor_metrics is None else anchor_metrics.copy()
    if anchors.empty:
        return confirmation_results
    if "period" not in anchors and "anchor_date" in anchors:
        anchors["period"] = pd.to_datetime(anchors["anchor_date"], errors="coerce").map(_period_for_date)
    anchors = anchors.loc[
        anchors["period"].astype(str).eq("confirmation")
        & pd.to_numeric(anchors["horizon_days"], errors="coerce").eq(7)
    ].copy()
    if anchors.empty:
        return confirmation_results
    anchors["anchor_date"] = pd.to_datetime(anchors["anchor_date"], errors="coerce")
    anchors["calendar_month"] = anchors["anchor_date"].dt.strftime("%Y-%m")
    if "valid" not in anchors:
        anchors["valid"] = True
    anchors["valid"] = _coerce_bool(anchors["valid"])

    high_support = pd.DataFrame()
    strata = pd.DataFrame() if raw_support_strata is None else raw_support_strata.copy()
    required_strata = {"candidate_id", "support_stratum", "primary_improvement"}
    if required_strata.issubset(strata.columns):
        if "period" not in strata and "anchor_date" in strata:
            strata["period"] = pd.to_datetime(strata["anchor_date"], errors="coerce").map(_period_for_date)
        if "horizon_days" not in strata:
            strata["horizon_days"] = 7
        strata = strata.loc[
            strata["period"].astype(str).eq("confirmation")
            & pd.to_numeric(strata["horizon_days"], errors="coerce").eq(7)
            & strata["support_stratum"].astype(str).isin(
                ["support_5_9", "support_10_19", "support_20_plus"]
            )
        ].copy()
        if not strata.empty:
            strata["calendar_month"] = pd.to_datetime(strata["anchor_date"], errors="coerce").dt.strftime("%Y-%m")
            high_support = strata.groupby(
                ["candidate_id", "calendar_month"], sort=True, observed=True
            )["primary_improvement"].median().rename("support_ge5_primary_improvement").reset_index()

    development = development_results.copy()
    existing = confirmation_results.loc[
        ~confirmation_results["stratum_type"].astype(str).eq("confirmation_label")
    ].copy() if not confirmation_results.empty else confirmation_results.copy()
    rows: list[dict[str, object]] = []
    label_codes = {"Not confirmed": 0.0, "Partially confirmed": 1.0, "Strongly confirmed": 2.0}
    for candidate in selected_candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        family = str(candidate["target_family"])
        candidate_anchors = anchors.loc[anchors["candidate_id"].astype(str).eq(candidate_id)].copy()
        if family == "binary":
            primary = pd.to_numeric(_series(candidate_anchors, "delta_log_loss"), errors="coerce")
        else:
            primary = pd.to_numeric(_series(candidate_anchors, "log_mae_improvement"), errors="coerce")
        candidate_anchors["primary_improvement"] = primary
        months = candidate_anchors.groupby(
            "calendar_month", sort=True, observed=True
        ).agg(
            primary_improvement=("primary_improvement", "median"),
            valid=("valid", "max"),
        ).reset_index()
        if not high_support.empty:
            months = months.merge(
                high_support.loc[high_support["candidate_id"].astype(str).eq(candidate_id), ["calendar_month", "support_ge5_primary_improvement"]],
                on="calendar_month", how="left", validate="one_to_one",
            )
        dev = development.loc[development["candidate_id"].astype(str).eq(candidate_id)] if not development.empty else development
        development_improvement = np.nan
        preferred = dev.loc[dev["metric_name"].astype(str).eq("primary_improvement")] if not dev.empty else dev
        if not preferred.empty:
            development_improvement = float(pd.to_numeric(preferred["estimate"], errors="coerce").median())
        elif family == "binary" and not dev.empty:
            delta = dev.loc[dev["metric_name"].astype(str).isin(["delta_log_loss_candidate_minus_reference", "delta_logloss_candidate_minus_reference"])]
            if not delta.empty:
                development_improvement = -float(pd.to_numeric(delta["estimate"], errors="coerce").median())
        elif family == "continuous" and not dev.empty:
            improvement = dev.loc[dev["metric_name"].astype(str).isin(["parent_minus_candidate_log_mae", "log_mae_improvement"])]
            if not improvement.empty:
                development_improvement = float(pd.to_numeric(improvement["estimate"], errors="coerce").median())
        rubric = profile_selection.confirmation_label_rubric(
            development_primary_improvement=development_improvement,
            confirmation_months=months,
            target_family=family,
        )
        common = {
            "candidate_id": candidate_id,
            "base_candidate_id": candidate["base_candidate_id"],
            "target": candidate["target"],
            "granularity": candidate["granularity"],
            "period": "confirmation",
            "anchor_date": np.nan,
            "calendar_month": "",
            "horizon_days": 7,
            "stratum_type": "confirmation_label",
            "reference_id": "frozen_descriptive_confirmation_rubric",
            "aggregation": "confirmation_month_rubric",
            "n_scheduled_anchors": 25,
            "n_valid_anchors": int(rubric["n_valid_confirmation_months"]),
            "n_orders": int(pd.to_numeric(_series(candidate_anchors, "future_target_valid_orders", 0), errors="coerce").fillna(0).sum()),
            "n_events": float(pd.to_numeric(_series(candidate_anchors, "future_events"), errors="coerce").sum(min_count=1)),
            "n_entities": int(pd.to_numeric(_series(candidate_anchors, "n_common_entities", 0), errors="coerce").fillna(0).sum()),
            "n_common_entities": int(pd.to_numeric(_series(candidate_anchors, "n_common_entities", 0), errors="coerce").fillna(0).sum()),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "valid": True,
            "invalid_reason": str(rubric["failed_strong_conditions"]),
        }
        rows.append(
            {
                **common,
                "stratum_value": rubric["confirmation_label"],
                "metric_name": "confirmation_label_code",
                "estimate": label_codes[str(rubric["confirmation_label"])],
            }
        )
        conditions = {
            "strict_majority_favourable": rubric["strict_majority_favourable"],
            "aggregate_magnitude_within_50_percent": rubric["aggregate_magnitude_within_50_percent"],
            "high_support_material_reversal": rubric["high_support_material_reversal"],
            "advantage_only_below_support5_or_cold": rubric["advantage_only_below_support5_or_cold"],
            "confirmation_aggregate_primary_improvement": rubric["confirmation_aggregate_primary_improvement"],
        }
        for metric_name, value in conditions.items():
            rows.append(
                {
                    **common,
                    "stratum_value": rubric["confirmation_label"],
                    "metric_name": metric_name,
                    "estimate": float(value) if pd.notna(value) else np.nan,
                }
            )
    combined = pd.concat([existing, _normalise_schema(pd.DataFrame(rows), METRIC_SCHEMA)], ignore_index=True)
    return _sort_frame(_normalise_schema(combined, METRIC_SCHEMA), PRIMARY_KEYS["PROFILE_CONFIRMATION_RESULTS.csv"])


def append_persisted_confirmation_labels(
    confirmation_results: pd.DataFrame,
    confirmation_labels: pd.DataFrame,
    selected_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Expose runner-persisted confirmation labels without touching selection."""

    if confirmation_labels.empty:
        return confirmation_results
    required = {"candidate_id", "confirmation_label"}
    missing = sorted(required - set(confirmation_labels.columns))
    if missing:
        raise KeyError(f"CONFIRMATION_LABELS.csv missing columns: {missing}")
    if confirmation_labels["candidate_id"].duplicated().any():
        raise ValueError("CONFIRMATION_LABELS.csv has duplicate candidate_id")
    metadata = selected_candidates.set_index("candidate_id", drop=False)
    unknown = sorted(set(confirmation_labels["candidate_id"].astype(str)) - set(metadata.index.astype(str)))
    if unknown:
        raise ValueError(f"confirmation labels reference non-promoted candidates: {unknown[:3]}")
    existing = confirmation_results.loc[
        ~confirmation_results["stratum_type"].astype(str).eq("confirmation_label")
    ].copy() if not confirmation_results.empty else confirmation_results.copy()
    codes = {"Not confirmed": 0.0, "Partially confirmed": 1.0, "Strongly confirmed": 2.0}
    rows: list[dict[str, object]] = []
    condition_columns = (
        "strict_majority_favourable",
        "aggregate_direction_favourable",
        "aggregate_magnitude_within_50_percent",
        "high_support_material_reversal",
        "advantage_only_below_support5_or_cold",
        "confirmation_aggregate_primary_improvement",
        "development_primary_improvement",
    )
    for record in confirmation_labels.to_dict("records"):
        candidate_id = str(record["candidate_id"])
        candidate = metadata.loc[candidate_id]
        label = str(record["confirmation_label"])
        if label not in codes:
            raise ValueError(f"unknown frozen confirmation label: {label}")
        n_valid_value = pd.to_numeric(
            pd.Series([record.get("n_valid_confirmation_months", 0)]),
            errors="coerce",
        ).fillna(0).iloc[0]
        failed_conditions = record.get("failed_strong_conditions", "")
        failed_conditions = "" if pd.isna(failed_conditions) else str(failed_conditions)
        common = {
            "candidate_id": candidate_id,
            "base_candidate_id": candidate["base_candidate_id"],
            "target": candidate["target"],
            "granularity": candidate["granularity"],
            "period": "confirmation",
            "anchor_date": np.nan,
            "calendar_month": "",
            "horizon_days": 7,
            "stratum_type": "confirmation_label",
            "stratum_value": label,
            "reference_id": "working/CONFIRMATION_LABELS.csv",
            "aggregation": "frozen_descriptive_confirmation_rubric",
            "n_scheduled_anchors": 25,
            "n_valid_anchors": int(n_valid_value),
            "n_orders": 0,
            "n_events": np.nan,
            "n_entities": 0,
            "n_common_entities": 0,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "valid": True,
            "invalid_reason": failed_conditions,
        }
        rows.append({**common, "metric_name": "confirmation_label_code", "estimate": codes[label]})
        for column in condition_columns:
            if column not in record:
                continue
            value = record.get(column)
            if isinstance(value, (bool, np.bool_)):
                estimate = float(value)
            else:
                estimate = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            rows.append({**common, "metric_name": column, "estimate": estimate})
    combined = pd.concat(
        [existing, _normalise_schema(pd.DataFrame(rows), METRIC_SCHEMA)], ignore_index=True
    )
    return _sort_frame(
        _normalise_schema(combined, METRIC_SCHEMA),
        PRIMARY_KEYS["PROFILE_CONFIRMATION_RESULTS.csv"],
    )


def aggregate_construction_audit(
    audit: pd.DataFrame | None,
    daily_rows: pd.DataFrame,
    selected_candidates: pd.DataFrame,
) -> pd.DataFrame:
    if audit is None or audit.empty:
        return _empty(CONSTRUCTION_AUDIT_SCHEMA)
    source = audit.copy()
    source["snapshot_date"] = pd.to_datetime(source["snapshot_date"], errors="coerce")
    if "period" not in source:
        source["period"] = source["snapshot_date"].map(_period_for_date)
    design = selected_candidates[
        [
            "base_candidate_id", "target", "granularity", "scheme", "window_days",
            "lag_days", "estimator", "parent_structure", "kappa",
        ]
    ].drop_duplicates("base_candidate_id")
    join_keys = ["target", "granularity", "scheme", "window_days", "lag_days"]
    if "base_candidate_id" not in source:
        source = source.merge(design, on=join_keys, how="inner", validate="many_to_many")
    else:
        source = source.merge(
            design,
            on="base_candidate_id",
            how="inner",
            suffixes=("", "_selected"),
            validate="many_to_one",
        )
        for column in join_keys + ["estimator", "parent_structure", "kappa"]:
            chosen = f"{column}_selected"
            if chosen in source:
                source[column] = source.get(column, source[chosen]).fillna(source[chosen])
    if source.empty:
        return _empty(CONSTRUCTION_AUDIT_SCHEMA)
    snapshot = source["snapshot_date"]
    window = pd.to_numeric(source["window_days"], errors="coerce")
    lag = pd.to_numeric(source["lag_days"], errors="coerce").fillna(0)
    scheme_c = source["scheme"].astype(str).eq("C")
    source["source_interval_axis"] = np.where(scheme_c, "purchase_timestamp", "label_available_at")
    source["source_interval_end"] = snapshot - pd.to_timedelta(np.where(scheme_c, lag, 0), unit="D")
    source["source_interval_start"] = source["source_interval_end"] - pd.to_timedelta(window, unit="D")
    source["availability_cutoff"] = snapshot
    source["history_sample"] = "canonical_delivered_96470_target_valid"
    # Preserve the runner's more precise frozen audit label.  Reporting may
    # fill a missing legacy value, but must never replace persisted denominator
    # provenance with a less specific shorthand.
    denominator_label = "all_placed_99441_next_7d_purchase_cohort"
    if "future_denominator_sample" not in source:
        source["future_denominator_sample"] = denominator_label
    else:
        source["future_denominator_sample"] = source[
            "future_denominator_sample"
        ].astype("string")
        denominator = source["future_denominator_sample"].str.strip()
        missing_denominator = denominator.isna() | denominator.eq("")
        source.loc[missing_denominator, "future_denominator_sample"] = denominator_label
    if "strict_asof_pass" not in source:
        maximum = pd.to_datetime(_series(source, "max_source_label_available_at"), errors="coerce")
        source["strict_asof_pass"] = maximum.lt(snapshot) | maximum.isna()
    source["window_pass"] = source["strict_asof_pass"]
    if "valid" not in source:
        source["valid"] = _coerce_bool(source["strict_asof_pass"]) & _coerce_bool(source["window_pass"])
    if "invalid_reason" not in source:
        source["invalid_reason"] = np.where(_coerce_bool(source["valid"]), "", "construction_audit_failed")
    if not daily_rows.empty:
        daily_summary = daily_rows.groupby(
            ["base_candidate_id", "snapshot_date"], sort=True, dropna=False,
            observed=True,
        ).agg(
            entity_domain_count=("entity_id", "nunique"),
            profile_rows=("entity_id", "size"),
            cold_start_rows=("cold_start", "sum"),
            last_mature_outcome_date=("last_mature_outcome_date", "max"),
        ).reset_index()
        source = source.drop(
            columns=[column for column in daily_summary.columns[2:] if column in source],
            errors="ignore",
        ).merge(
            daily_summary,
            on=["base_candidate_id", "snapshot_date"],
            how="left",
            validate="many_to_one",
        )
    result = _normalise_schema(source, CONSTRUCTION_AUDIT_SCHEMA)
    return _sort_frame(result, PRIMARY_KEYS["PROFILE_CONSTRUCTION_AUDIT.csv"])


def aggregate_support_uncertainty(
    daily_rows: pd.DataFrame,
    order_scoring: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if daily_rows.empty:
        return _empty(SUPPORT_UNCERTAINTY_SCHEMA)
    daily = daily_rows.copy()
    daily["support_stratum"] = [
        _support_stratum(value) for value in daily["support"]
    ]
    daily["interval_width"] = pd.to_numeric(daily["upper_interval"], errors="coerce") - pd.to_numeric(
        daily["lower_interval"], errors="coerce"
    )
    exposure = pd.DataFrame()
    if order_scoring is not None and not order_scoring.empty:
        scored = order_scoring.copy()
        scored["snapshot_date"] = pd.to_datetime(scored["anchor_date"], errors="coerce")
        scored["support_stratum"] = [
            _support_stratum(value, status)
            for value, status in zip(scored["history_support"], scored["mapping_status"])
        ]
        exposure = scored.groupby(
            ["base_candidate_id", "snapshot_date", "support_stratum"],
            sort=True,
            dropna=False,
            observed=True,
        ).size().rename("order_exposure").reset_index()
    group_columns = [
        "base_candidate_id", "snapshot_date", "period", "target", "granularity",
        "scheme", "window_days", "lag_days", "estimator", "parent_structure",
        "kappa", "support_stratum",
    ]
    rows: list[dict[str, object]] = []
    # Drop communication-support duplicates before base-profile aggregation.
    base_daily = daily.drop_duplicates(["base_candidate_id", "snapshot_date", "entity_id"])
    for keys, group in base_daily.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        record = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        se = pd.to_numeric(group["posterior_se"], errors="coerce")
        widths = pd.to_numeric(group["interval_width"], errors="coerce")
        supports = pd.to_numeric(group["support"], errors="coerce")
        scores = pd.to_numeric(group["score"], errors="coerce")
        valid = np.isfinite(scores).any()
        record.update(
            {
                "entity_count": int(group["entity_id"].nunique()),
                "order_exposure": 0,
                "median_support": float(supports.median()) if supports.notna().any() else np.nan,
                "median_score": float(scores.median()) if scores.notna().any() else np.nan,
                "median_posterior_se": float(se.median()) if se.notna().any() else np.nan,
                "p90_posterior_se": float(se.quantile(0.90)) if se.notna().any() else np.nan,
                "median_interval_width": float(widths.median()) if widths.notna().any() else np.nan,
                "p90_interval_width": float(widths.quantile(0.90)) if widths.notna().any() else np.nan,
                "cold_start_count": int(_coerce_bool(group["cold_start"]).sum()),
                "valid": bool(valid),
                "invalid_reason": "" if valid else "no_finite_profile_score",
            }
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    if not exposure.empty and not result.empty:
        result = result.merge(
            exposure,
            on=["base_candidate_id", "snapshot_date", "support_stratum"],
            how="left",
            suffixes=("", "_observed"),
            validate="one_to_one",
        )
        result["order_exposure"] = pd.to_numeric(result["order_exposure_observed"], errors="coerce").fillna(0).astype(int)
        result = result.drop(columns=["order_exposure_observed"])
    return _sort_frame(
        _normalise_schema(result, SUPPORT_UNCERTAINTY_SCHEMA),
        PRIMARY_KEYS["PROFILE_SUPPORT_UNCERTAINTY.csv"],
    )


def aggregate_parent_structure(
    parent_profiles: pd.DataFrame | None,
    daily_rows: pd.DataFrame,
) -> pd.DataFrame:
    if parent_profiles is not None and not parent_profiles.empty:
        parents = parent_profiles.copy()
        aliases = {
            "support": "parent_support", "event_count": "parent_event_count",
            "score": "parent_score", "posterior_se": "parent_posterior_se",
            "lower_interval": "parent_interval_lower",
            "upper_interval": "parent_interval_upper",
            "within_variance": "parent_within_variance",
            "between_variance": "parent_between_variance",
        }
        for source, destination in aliases.items():
            if destination not in parents and source in parents:
                parents[destination] = parents[source]
        if "fallback_child_count" not in parents:
            parents["fallback_child_count"] = 0
        if "parent_supported" not in parents:
            parents["parent_supported"] = pd.to_numeric(
                _series(parents, "parent_support"), errors="coerce"
            ).fillna(0).ge(20)
        if "valid" not in parents:
            parents["valid"] = np.isfinite(pd.to_numeric(_series(parents, "parent_score"), errors="coerce"))
        if "invalid_reason" not in parents:
            parents["invalid_reason"] = np.where(_coerce_bool(parents["valid"]), "", "nonfinite_parent_score")
        result = _normalise_schema(parents, PARENT_STRUCTURE_SCHEMA)
        return _sort_frame(result, PRIMARY_KEYS["PROFILE_PARENT_STRUCTURE.csv"])
    if daily_rows.empty:
        return _empty(PARENT_STRUCTURE_SCHEMA)
    base = daily_rows.drop_duplicates(["base_candidate_id", "snapshot_date", "entity_id"])
    rows: list[dict[str, object]] = []
    group_columns = [
        "base_candidate_id", "snapshot_date", "target", "granularity",
        "parent_structure", "parent_id",
    ]
    for keys, group in base.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        record = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        parent_score = pd.to_numeric(group["parent_score"], errors="coerce")
        global_score = pd.to_numeric(group["global_score"], errors="coerce")
        valid = np.isfinite(parent_score).any()
        record.update(
            {
                "parent_support": np.nan,
                "parent_event_count": np.nan,
                "parent_score": float(parent_score.median()) if parent_score.notna().any() else np.nan,
                "global_score": float(global_score.median()) if global_score.notna().any() else np.nan,
                "parent_within_variance": np.nan,
                "parent_between_variance": np.nan,
                "parent_posterior_se": np.nan,
                "parent_interval_lower": np.nan,
                "parent_interval_upper": np.nan,
                "fallback_child_count": int(_coerce_bool(group["cold_start"]).sum()),
                "parent_supported": np.nan,
                "valid": bool(valid),
                "invalid_reason": "parent_detail_not_persisted" if valid else "nonfinite_parent_score",
            }
        )
        rows.append(record)
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), PARENT_STRUCTURE_SCHEMA),
        PRIMARY_KEYS["PROFILE_PARENT_STRUCTURE.csv"],
    )


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 2 or x[usable].nunique() < 2 or y[usable].nunique() < 2:
        return np.nan
    return float(x[usable].rank(method="average").corr(y[usable].rank(method="average")))


def aggregate_daily_stability(
    daily_rows: pd.DataFrame,
    hrd_days: pd.DataFrame | None = None,
    *,
    hrd_phases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if daily_rows.empty:
        return _empty(DAILY_STABILITY_SCHEMA)
    base = daily_rows.copy()
    base["snapshot_date"] = pd.to_datetime(
        base["snapshot_date"], errors="coerce"
    )
    base = base.sort_values(
        ["base_candidate_id", "snapshot_date", "entity_id"], kind="mergesort"
    ).drop_duplicates(["base_candidate_id", "snapshot_date", "entity_id"])
    regime_map: dict[pd.Timestamp, str] = {}
    if hrd_days is not None and not hrd_days.empty:
        hrd = hrd_days.copy()
        date_column = next((name for name in ("date", "snapshot_date", "purchase_date") if name in hrd), None)
        if date_column is not None:
            hrd[date_column] = pd.to_datetime(hrd[date_column], errors="coerce").dt.normalize()
            primary = next((name for name in ("both_top10", "is_hrd", "regime") if name in hrd), None)
            if primary is not None:
                for _, row in hrd.iterrows():
                    value = row[primary]
                    regime_map[pd.Timestamp(row[date_column])] = (
                        str(value) if primary == "regime" else (
                            "HRD" if bool(_coerce_bool(pd.Series([value])).iloc[0]) else "BAU"
                        )
                    )
        primary_definition = "both_top10"
        phase_table = _definition_phase_table(hrd_days, hrd_phases, primary_definition)
        if not phase_table.empty:
            for _, row in phase_table.iterrows():
                date = pd.Timestamp(row["_purchase_date"])
                phase = str(row["phase"])
                if phase == "event":
                    regime_map[date] = "HRD_event"
                elif phase == "pre_event":
                    regime_map[date] = "pre_event"
                elif phase.startswith("post_event"):
                    regime_map[date] = "post_event"
    rows: list[dict[str, object]] = []
    for base_id, candidate in base.groupby(
        "base_candidate_id", sort=True, observed=True
    ):
        dates = sorted(pd.to_datetime(candidate["snapshot_date"].dropna().unique()))
        for previous_date, current_date in zip(dates[:-1], dates[1:]):
            previous = candidate.loc[candidate["snapshot_date"].eq(previous_date)].copy()
            current = candidate.loc[candidate["snapshot_date"].eq(current_date)].copy()
            common = previous.merge(
                current,
                on="entity_id",
                how="inner",
                suffixes=("_previous", "_current"),
                validate="one_to_one",
            )
            score_previous = pd.to_numeric(_series(common, "score_previous"), errors="coerce")
            score_current = pd.to_numeric(_series(common, "score_current"), errors="coerce")
            finite = np.isfinite(score_previous) & np.isfinite(score_current)
            usable = common.loc[finite].copy()
            change = np.abs(score_current[finite] - score_previous[finite])
            valid = len(usable) >= 10 and score_previous[finite].nunique() > 1 and score_current[finite].nunique() > 1
            top_n = max(1, int(math.ceil(len(usable) * 0.20))) if len(usable) else 0
            left_top = set(
                usable.sort_values(["score_previous", "entity_id"], ascending=[False, True], kind="mergesort").head(top_n)["entity_id"]
            )
            right_top = set(
                usable.sort_values(["score_current", "entity_id"], ascending=[False, True], kind="mergesort").head(top_n)["entity_id"]
            )
            union = left_top | right_top
            support_previous = pd.to_numeric(_series(common, "support_previous", 0), errors="coerce").fillna(0)
            support_current = pd.to_numeric(_series(common, "support_current", 0), errors="coerce").fillna(0)
            newly = float((support_current - support_previous).clip(lower=0).sum())
            level_previous = _series(common, "level_previous", "Unknown").astype(str)
            level_current = _series(common, "level_current", "Unknown").astype(str)
            changes = int(level_previous.ne(level_current).sum())
            # Score stability is deliberately restricted to common mapped
            # entities.  Cold-start transitions need the adjacent-day union:
            # the shared builder persists rows only once an entity has
            # historical support, so absence on one side represents the
            # support-zero parent-fallback state for this retrospective
            # transition diagnostic.  Explicit support-zero rows, if supplied
            # by a richer runner table, follow the same rule.
            transition = previous.merge(
                current,
                on="entity_id",
                how="outer",
                suffixes=("_previous", "_current"),
                validate="one_to_one",
                indicator=True,
            )
            previous_present = transition["_merge"].ne("right_only")
            current_present = transition["_merge"].ne("left_only")
            previous_support = pd.to_numeric(
                _series(transition, "support_previous", 0), errors="coerce"
            ).fillna(0)
            current_support = pd.to_numeric(
                _series(transition, "support_current", 0), errors="coerce"
            ).fillna(0)
            cold_previous = (
                ~previous_present
                | previous_support.eq(0)
                | _coerce_bool(_series(transition, "cold_start_previous", False))
            )
            cold_current = (
                ~current_present
                | current_support.eq(0)
                | _coerce_bool(_series(transition, "cold_start_current", False))
            )
            transition_union_count = int(len(transition))
            cold_entries = int((~cold_previous & cold_current).sum())
            cold_exits = int((cold_previous & ~cold_current).sum())
            sample = current.iloc[0]
            rows.append(
                {
                    "base_candidate_id": base_id,
                    "target": sample["target"],
                    "granularity": sample["granularity"],
                    "previous_snapshot_date": previous_date,
                    "snapshot_date": current_date,
                    "period": _period_for_date(current_date),
                    "regime": regime_map.get(pd.Timestamp(current_date).normalize(), "BAU_or_unlabelled"),
                    "n_common_entities": int(len(usable)),
                    "newly_matured_support": newly,
                    "day_to_day_spearman": _rank_correlation(score_previous[finite], score_current[finite]) if valid else np.nan,
                    "median_absolute_score_change": float(change.median()) if len(change) else np.nan,
                    "p90_absolute_score_change": float(change.quantile(0.90)) if len(change) else np.nan,
                    "top20_jaccard": float(len(left_top & right_top) / len(union)) if union else np.nan,
                    "score_change_per_new_label": float(change.sum() / newly) if newly > 0 else np.nan,
                    "entities_changing_level": changes,
                    "pct_entities_changing_level": float(changes / len(common)) if len(common) else np.nan,
                    "transition_entity_union_count": transition_union_count,
                    "cold_start_entries": cold_entries,
                    "cold_start_exits": cold_exits,
                    "cold_start_entry_rate": (
                        cold_entries / transition_union_count
                        if transition_union_count else np.nan
                    ),
                    "cold_start_exit_rate": (
                        cold_exits / transition_union_count
                        if transition_union_count else np.nan
                    ),
                    "valid": bool(valid),
                    "invalid_reason": "" if valid else "fewer_than_10_or_constant_common_entities",
                }
            )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), DAILY_STABILITY_SCHEMA),
        PRIMARY_KEYS["PROFILE_DAILY_STABILITY.csv"],
    )


def _cluster_mean_ci(group: pd.DataFrame, value_column: str, identity: object) -> tuple[float, float]:
    usable = group.loc[
        group["entity_id"].notna()
        & np.isfinite(pd.to_numeric(group[value_column], errors="coerce")),
        ["entity_id", value_column],
    ].copy()
    if usable["entity_id"].nunique() < 2:
        return np.nan, np.nan
    entity = usable.groupby(
        "entity_id", sort=True, observed=True
    )[value_column].agg(["sum", "count"])
    sums = pd.to_numeric(entity["sum"], errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(entity["count"], errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(_stable_seed(RANDOM_SEED, "entity_mean", identity))
    replicates = np.empty(500, dtype=float)
    for index in range(500):
        sampled = rng.integers(0, len(entity), size=len(entity))
        replicates[index] = float(sums[sampled].sum() / counts[sampled].sum())
    return tuple(float(x) for x in np.quantile(replicates, [0.025, 0.975]))


def _level_lift_entity_cluster_ci(
    block: pd.DataFrame,
    numerator_level: str,
    outcome_column: str,
    identity: object,
) -> tuple[float, float]:
    """Entity-cluster bootstrap a level-mean ratio without iid order draws."""

    work = block.loc[
        block["entity_id"].notna()
        & _coerce_bool(block["target_observed"])
        & block["level"].astype(str).isin(["Low", numerator_level])
    ].copy()
    work["_outcome"] = pd.to_numeric(
        _series(work, outcome_column), errors="coerce"
    )
    work = work.loc[np.isfinite(work["_outcome"])].copy()
    if work["entity_id"].nunique() < 2:
        return np.nan, np.nan
    stats = work.groupby(
        ["entity_id", "level"], sort=True, observed=True
    )["_outcome"].agg(["sum", "count"]).reset_index()
    entities = sorted(stats["entity_id"].astype(str).unique())
    lookup = {entity: index for index, entity in enumerate(entities)}
    sums: dict[str, np.ndarray] = {
        level: np.zeros(len(entities), dtype=float)
        for level in ("Low", numerator_level)
    }
    counts: dict[str, np.ndarray] = {
        level: np.zeros(len(entities), dtype=float)
        for level in ("Low", numerator_level)
    }
    for row in stats.to_dict("records"):
        level = str(row["level"])
        index = lookup[str(row["entity_id"])]
        sums[level][index] = float(row["sum"])
        counts[level][index] = float(row["count"])
    rng = np.random.default_rng(
        _stable_seed(RANDOM_SEED, "level_lift_entity_cluster", identity)
    )
    replicates: list[float] = []
    for _ in range(500):
        sampled = rng.integers(0, len(entities), size=len(entities))
        numerator_count = counts[numerator_level][sampled].sum()
        denominator_count = counts["Low"][sampled].sum()
        if numerator_count <= 0 or denominator_count <= 0:
            continue
        numerator = sums[numerator_level][sampled].sum() / numerator_count
        denominator = sums["Low"][sampled].sum() / denominator_count
        if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0:
            replicates.append(float(numerator / denominator))
    if len(replicates) < 100:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(replicates, [0.025, 0.975]))


def aggregate_level_results(order_scoring: pd.DataFrame) -> pd.DataFrame:
    if order_scoring.empty:
        return _empty(LEVEL_RESULTS_SCHEMA)
    scored = order_scoring.copy()
    scored["target_observed"] = _coerce_bool(scored["target_observed"])
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days",
    ]
    for identity_values, block in scored.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, identity_values if isinstance(identity_values, tuple) else (identity_values,)))
        total = len(block)
        total_entities = int(block["entity_id"].dropna().nunique())
        percent_unknown = float(block["level"].astype(str).eq("Unknown").mean()) if total else np.nan
        estimates: dict[str, float] = {}
        original_day_means: dict[str, float] = {}
        block_rows: list[dict[str, object]] = []
        for level in ("Unknown", "Low", "Medium", "High"):
            part = block.loc[block["level"].astype(str).eq(level)]
            valid = part.loc[
                _coerce_bool(part["target_observed"])
                & np.isfinite(pd.to_numeric(part["target_value"], errors="coerce"))
            ].copy()
            estimate = float(pd.to_numeric(valid["target_value"], errors="coerce").mean()) if len(valid) else np.nan
            estimates[level] = estimate
            lower, upper = _cluster_mean_ci(
                valid, "target_value", (identity_values, level)
            ) if len(valid) else (np.nan, np.nan)
            common = {
                **identity,
                "level": level,
                "n_orders": int(len(part)),
                "n_entities": int(part["entity_id"].dropna().nunique()),
                "future_support": int(len(valid)),
                "percent_unknown": percent_unknown,
            }
            block_rows.append(
                {
                    **common,
                    "metric_name": "future_event_rate" if _target_family(identity["target"]) == "binary" else "future_mean_log_outcome",
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid": bool(len(valid)),
                    "invalid_reason": "" if len(valid) else "no_observed_future_outcome",
                }
            )
            block_rows.extend(
                [
                    {
                        **common,
                        "metric_name": "order_share",
                        "estimate": float(len(part) / total) if total else np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "valid": bool(total),
                        "invalid_reason": "" if total else "no_future_orders",
                    },
                    {
                        **common,
                        "metric_name": "entity_share",
                        "estimate": (
                            float(part["entity_id"].dropna().nunique() / total_entities)
                            if total_entities else np.nan
                        ),
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "valid": bool(total_entities),
                        "invalid_reason": "" if total_entities else "no_mapped_entities",
                    },
                ]
            )
            if _target_family(identity["target"]) == "continuous":
                raw = valid.copy()
                raw["_raw_days"] = pd.to_numeric(
                    _series(raw, "raw_target_value"), errors="coerce"
                )
                raw = raw.loc[np.isfinite(raw["_raw_days"])].copy()
                raw_mean = float(raw["_raw_days"].mean()) if len(raw) else np.nan
                raw_median = float(raw["_raw_days"].median()) if len(raw) else np.nan
                original_day_means[level] = raw_mean
                raw_lower, raw_upper = _cluster_mean_ci(
                    raw, "_raw_days", (identity_values, level, "original_days")
                ) if len(raw) else (np.nan, np.nan)
                block_rows.extend(
                    [
                        {
                            **common,
                            "metric_name": "future_original_days_mean",
                            "future_support": int(len(raw)),
                            "estimate": raw_mean,
                            "ci_lower": raw_lower,
                            "ci_upper": raw_upper,
                            "valid": bool(len(raw)),
                            "invalid_reason": "" if len(raw) else "no_observed_original_day_outcome",
                        },
                        {
                            **common,
                            "metric_name": "future_original_days_median",
                            "future_support": int(len(raw)),
                            "estimate": raw_median,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "valid": bool(len(raw)),
                            "invalid_reason": "" if len(raw) else "no_observed_original_day_outcome",
                        },
                    ]
                )
        monotone = bool(
            all(np.isfinite(estimates[level]) for level in ("Low", "Medium", "High"))
            and estimates["Low"] < estimates["Medium"] < estimates["High"]
        )
        for record in block_rows:
            record["monotone_lmh"] = monotone
            rows.append(record)
        lift_estimates = (
            original_day_means
            if _target_family(identity["target"]) == "continuous"
            else estimates
        )
        for numerator_level, metric_name in (
            ("High", "high_low_future_outcome_lift"),
            ("Medium", "medium_low_future_outcome_lift"),
        ):
            numerator = lift_estimates.get(numerator_level, np.nan)
            denominator = lift_estimates.get("Low", np.nan)
            lift_valid = bool(
                np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0
            )
            lift_outcome_column = (
                "raw_target_value"
                if _target_family(identity["target"]) == "continuous"
                else "target_value"
            )
            lift_lower, lift_upper = _level_lift_entity_cluster_ci(
                block,
                numerator_level,
                lift_outcome_column,
                (identity_values, metric_name),
            ) if lift_valid else (np.nan, np.nan)
            ci_supported = bool(np.isfinite(lift_lower) and np.isfinite(lift_upper))
            rows.append(
                {
                    **identity,
                    "level": "all",
                    "metric_name": metric_name,
                    "n_orders": int(total),
                    "n_entities": total_entities,
                    "future_support": int(
                        sum(
                            len(
                                block.loc[
                                    block["level"].astype(str).eq(level)
                                    & _coerce_bool(block["target_observed"])
                                ]
                            )
                            for level in ("Low", numerator_level)
                        )
                    ),
                    "estimate": float(numerator / denominator) if lift_valid else np.nan,
                    "ci_lower": lift_lower,
                    "ci_upper": lift_upper,
                    "monotone_lmh": monotone,
                    "percent_unknown": percent_unknown,
                    "valid": bool(lift_valid and ci_supported),
                    "invalid_reason": (
                        "" if lift_valid and ci_supported else (
                            "entity_cluster_lift_ci_not_supported"
                            if lift_valid else "low_level_outcome_missing_or_zero"
                        )
                    ),
                }
            )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), LEVEL_RESULTS_SCHEMA),
        PRIMARY_KEYS["PROFILE_LEVEL_RESULTS.csv"],
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def aggregate_level_transitions(daily_rows: pd.DataFrame) -> pd.DataFrame:
    if daily_rows.empty:
        return _empty(LEVEL_TRANSITIONS_SCHEMA)
    daily = daily_rows.copy()
    daily["snapshot_date"] = pd.to_datetime(daily["snapshot_date"], errors="coerce")
    levels = ("Unknown", "Low", "Medium", "High")
    transitions: list[pd.DataFrame] = []
    for candidate_id, group in daily.groupby(
        "candidate_id", sort=True, observed=True
    ):
        dates = sorted(group["snapshot_date"].dropna().unique())
        for previous_date, current_date in zip(dates[:-1], dates[1:]):
            previous = group.loc[group["snapshot_date"].eq(previous_date), ["entity_id", "level"]].rename(columns={"level": "from_level"})
            current = group.loc[group["snapshot_date"].eq(current_date), ["entity_id", "level"]].rename(columns={"level": "to_level"})
            joined = previous.merge(current, on="entity_id", how="inner", validate="one_to_one")
            joined["candidate_id"] = candidate_id
            joined["period"] = _period_for_date(current_date)
            transitions.append(joined)
    joined_all = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    persistence_values: dict[tuple[str, str, str], list[int]] = {}
    for (candidate_id, entity_id), history in daily.groupby(
        ["candidate_id", "entity_id"], sort=True, dropna=False, observed=True
    ):
        history = history.sort_values("snapshot_date", kind="mergesort")
        run_level: str | None = None
        run_period: str | None = None
        run_start: pd.Timestamp | None = None
        run_end: pd.Timestamp | None = None
        for _, record in history.iterrows():
            date = pd.Timestamp(record["snapshot_date"])
            level = str(record["level"])
            period = _period_for_date(date)
            consecutive = run_end is not None and (date - run_end).days == 1
            if run_level is None or level != run_level or period != run_period or not consecutive:
                if run_level is not None and run_start is not None and run_end is not None:
                    persistence_values.setdefault(
                        (str(candidate_id), str(run_period), str(run_level)), []
                    ).append((run_end - run_start).days + 1)
                run_level, run_period, run_start = level, period, date
            run_end = date
        if run_level is not None and run_start is not None and run_end is not None:
            persistence_values.setdefault(
                (str(candidate_id), str(run_period), str(run_level)), []
            ).append((run_end - run_start).days + 1)
    rows: list[dict[str, object]] = []
    metadata = daily.drop_duplicates("candidate_id").set_index("candidate_id")
    for candidate_id in sorted(metadata.index.astype(str)):
        candidate = metadata.loc[candidate_id]
        periods = sorted(
            set(
                joined_all.loc[
                    joined_all["candidate_id"].astype(str).eq(candidate_id), "period"
                ].astype(str)
            )
        ) if not joined_all.empty else []
        for period in periods:
            block = joined_all.loc[
                joined_all["candidate_id"].astype(str).eq(candidate_id)
                & joined_all["period"].eq(period)
            ]
            for from_level in levels:
                eligible = int(block["from_level"].astype(str).eq(from_level).sum())
                durations = persistence_values.get(
                    (candidate_id, str(period), from_level), []
                )
                for to_level in levels:
                    count = int(
                        (block["from_level"].astype(str).eq(from_level)
                         & block["to_level"].astype(str).eq(to_level)).sum()
                    )
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "base_candidate_id": candidate["base_candidate_id"],
                            "target": candidate["target"],
                            "granularity": candidate["granularity"],
                            "period": period,
                            "from_level": from_level,
                            "to_level": to_level,
                            "transition_count": count,
                            "eligible_from_count": eligible,
                            "transition_probability": count / eligible if eligible else np.nan,
                            "median_persistence_days": float(np.median(durations)) if durations else np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "valid": bool(eligible),
                            "invalid_reason": (
                                "transition_ci_not_supported_under_repeated_entity_time_structure"
                                if eligible else "no_eligible_from_level"
                            ),
                        }
                    )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), LEVEL_TRANSITIONS_SCHEMA),
        PRIMARY_KEYS["PROFILE_LEVEL_TRANSITIONS.csv"],
    )


def _top_quintile_lift(
    entity: pd.DataFrame,
    outcome_column: str = "future_mean",
) -> float:
    if entity.empty:
        return np.nan
    ranked = entity.sort_values(
        ["profile_score", "entity_id"], ascending=[False, True], kind="mergesort"
    )
    count = max(1, int(math.ceil(0.20 * len(ranked))))
    weights = pd.to_numeric(ranked["future_support"], errors="coerce").to_numpy(dtype=float)
    outcome = pd.to_numeric(
        ranked.get(outcome_column, pd.Series(index=ranked.index, dtype=float)),
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = np.isfinite(outcome) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    ranked = ranked.loc[finite].copy()
    weights = weights[finite]
    outcome = outcome[finite]
    count = max(1, int(math.ceil(0.20 * len(ranked))))
    overall = float(np.average(outcome, weights=weights)) if weights.sum() > 0 else np.nan
    top_weights = weights[:count]
    top = float(np.average(outcome[:count], weights=top_weights)) if top_weights.sum() > 0 else np.nan
    return top / overall if np.isfinite(overall) and overall != 0 else np.nan


def _entity_transfer_bootstrap(
    entity: pd.DataFrame,
    identity: object,
    *,
    lift_outcome_column: str = "future_mean",
) -> tuple[float, float, float, float]:
    if len(entity) < 10:
        return np.nan, np.nan, np.nan, np.nan
    n_rows = len(entity)
    profile_score = pd.to_numeric(
        entity["profile_score"], errors="coerce"
    ).to_numpy(dtype=float)
    future_mean = pd.to_numeric(
        entity["future_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    lift_outcome = pd.to_numeric(
        entity.get(
            lift_outcome_column,
            pd.Series(index=entity.index, dtype=float),
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    future_support = pd.to_numeric(
        entity["future_support"], errors="coerce"
    ).to_numpy(dtype=float)
    entity_text = entity["entity_id"].astype(str).to_numpy(dtype=str)

    # A bootstrap replicate only changes the multiplicity of values already
    # present in the entity table.  Encode the two rank variables once, then
    # recover average ranks from sampled category counts.  This is exactly the
    # ``Series.rank(method='average')`` definition used by _rank_correlation,
    # without allocating and sorting a DataFrame in every replicate.
    rank_usable = np.isfinite(profile_score) & np.isfinite(future_mean)
    profile_codes = np.full(n_rows, -1, dtype=np.int64)
    outcome_codes = np.full(n_rows, -1, dtype=np.int64)
    if rank_usable.any():
        _, usable_profile_codes = np.unique(
            profile_score[rank_usable], return_inverse=True
        )
        _, usable_outcome_codes = np.unique(
            future_mean[rank_usable], return_inverse=True
        )
        profile_codes[rank_usable] = usable_profile_codes
        outcome_codes[rank_usable] = usable_outcome_codes
        n_profile_values = int(usable_profile_codes.max()) + 1
        n_outcome_values = int(usable_outcome_codes.max()) + 1
    else:
        n_profile_values = 0
        n_outcome_values = 0

    lift_source_usable = (
        np.isfinite(lift_outcome)
        & np.isfinite(future_support)
        & (future_support > 0)
    )
    sample_suffix = np.asarray(
        [f"#{position}" for position in range(n_rows)], dtype=str
    )

    def sampled_rank_correlation(indices: np.ndarray) -> float:
        sampled_profile_codes = profile_codes[indices]
        usable = sampled_profile_codes >= 0
        if int(usable.sum()) < 2:
            return np.nan
        sampled_profile_codes = sampled_profile_codes[usable]
        sampled_outcome_codes = outcome_codes[indices][usable]
        profile_counts = np.bincount(
            sampled_profile_codes, minlength=n_profile_values
        )
        outcome_counts = np.bincount(
            sampled_outcome_codes, minlength=n_outcome_values
        )
        if (
            np.count_nonzero(profile_counts) < 2
            or np.count_nonzero(outcome_counts) < 2
        ):
            return np.nan

        profile_before = np.cumsum(profile_counts) - profile_counts
        outcome_before = np.cumsum(outcome_counts) - outcome_counts
        profile_rank_lookup = profile_before + (profile_counts + 1.0) / 2.0
        outcome_rank_lookup = outcome_before + (outcome_counts + 1.0) / 2.0
        profile_ranks = profile_rank_lookup[sampled_profile_codes]
        outcome_ranks = outcome_rank_lookup[sampled_outcome_codes]
        return float(np.corrcoef(profile_ranks, outcome_ranks)[0, 1])

    def sampled_top_quintile_lift(indices: np.ndarray) -> float:
        usable_positions = np.flatnonzero(lift_source_usable[indices])
        if not len(usable_positions):
            return np.nan
        sampled_sources = indices[usable_positions]
        sampled_scores = profile_score[sampled_sources]
        # The reference implementation makes resampled entities unique as
        # ``str(entity_id) + '#' + sample_position`` before its stable
        # score-descending/entity-lexical sort.  Construct precisely that key
        # only for lift-eligible rows; filtering after a sort preserves their
        # relative order, so the resulting order is identical.
        entity_keys = np.char.add(
            entity_text[sampled_sources], sample_suffix[usable_positions]
        )
        order = np.lexsort((entity_keys, -sampled_scores))
        ordered_sources = sampled_sources[order]
        weights = future_support[ordered_sources]
        outcomes = lift_outcome[ordered_sources]
        overall = float(np.average(outcomes, weights=weights))
        count = max(1, int(math.ceil(0.20 * len(ordered_sources))))
        top_weights = weights[:count]
        top = (
            float(np.average(outcomes[:count], weights=top_weights))
            if top_weights.sum() > 0
            else np.nan
        )
        return top / overall if np.isfinite(overall) and overall != 0 else np.nan

    rng = np.random.default_rng(_stable_seed(RANDOM_SEED, "entity_transfer", identity))
    correlations: list[float] = []
    lifts: list[float] = []
    for _ in range(500):
        indices = rng.integers(0, n_rows, size=n_rows)
        rho = sampled_rank_correlation(indices)
        lift = sampled_top_quintile_lift(indices)
        if np.isfinite(rho):
            correlations.append(rho)
        if np.isfinite(lift):
            lifts.append(lift)
    rho_ci = tuple(float(x) for x in np.quantile(correlations, [0.025, 0.975])) if correlations else (np.nan, np.nan)
    lift_ci = tuple(float(x) for x in np.quantile(lifts, [0.025, 0.975])) if lifts else (np.nan, np.nan)
    return rho_ci[0], rho_ci[1], lift_ci[0], lift_ci[1]


def aggregate_entity_transfer(
    entity_rows: pd.DataFrame | None,
    order_scoring: pd.DataFrame,
    selected_candidates: pd.DataFrame,
) -> pd.DataFrame:
    entity = pd.DataFrame() if entity_rows is None else entity_rows.copy()
    if entity.empty and not order_scoring.empty:
        valid = order_scoring.loc[_coerce_bool(order_scoring["eligible_for_metric"])].copy()
        if not valid.empty:
            aggregation: dict[str, tuple[str, str]] = {
                "profile_score": ("profile_score", "first"),
                "future_mean": ("target_value", "mean"),
                "future_support": ("target_value", "count"),
                "history_support": ("history_support", "first"),
                "level": ("level", "first"),
            }
            if "raw_target_value" in valid:
                aggregation["future_raw_mean"] = ("raw_target_value", "mean")
            entity = valid.groupby(
                [
                    "candidate_id", "base_candidate_id", "target", "granularity",
                    "period", "anchor_date", "horizon_days", "entity_id",
                ],
                sort=True,
                dropna=False,
                observed=True,
            ).agg(**aggregation).reset_index()
    if entity.empty:
        return _empty(ENTITY_TRANSFER_SCHEMA)
    metadata = _candidate_metadata(selected_candidates)
    if "candidate_id" not in entity and "base_candidate_id" in entity and not metadata.empty:
        entity = entity.merge(metadata[["base_candidate_id", "candidate_id"]], on="base_candidate_id", how="left")
    if "base_candidate_id" not in entity:
        entity["base_candidate_id"] = entity["candidate_id"].astype(str).str.replace(r"\|min_support=\d+$", "", regex=True)
    if not metadata.empty:
        add = metadata[["candidate_id", "target", "granularity"]].rename(columns={"target": "_target", "granularity": "_granularity"})
        entity = entity.merge(add, on="candidate_id", how="left", validate="many_to_one")
        for column in ("target", "granularity"):
            if column not in entity:
                entity[column] = entity[f"_{column}"]
            else:
                entity[column] = entity[column].fillna(entity[f"_{column}"])
        entity = entity.drop(columns=["_target", "_granularity"])
    if "period" not in entity:
        entity["period"] = pd.to_datetime(entity["anchor_date"], errors="coerce").map(_period_for_date)
    group_columns = [
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "anchor_date", "horizon_days",
    ]
    rows: list[dict[str, object]] = []
    for values, group in entity.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        numeric = group.copy()
        numeric["profile_score"] = pd.to_numeric(numeric["profile_score"], errors="coerce")
        numeric["future_mean"] = pd.to_numeric(numeric["future_mean"], errors="coerce")
        if "future_raw_mean" in numeric:
            numeric["future_raw_mean"] = pd.to_numeric(
                numeric["future_raw_mean"], errors="coerce"
            )
        numeric["future_support"] = pd.to_numeric(numeric["future_support"], errors="coerce")
        numeric = numeric.loc[
            np.isfinite(numeric["profile_score"])
            & np.isfinite(numeric["future_mean"])
            & numeric["future_support"].gt(0)
        ].copy()
        valid = len(numeric) >= 10 and numeric["profile_score"].nunique() > 1 and numeric["future_mean"].nunique() > 1
        unweighted = _rank_correlation(numeric["profile_score"], numeric["future_mean"]) if valid else np.nan
        weighted = weighted_spearman(numeric["profile_score"], numeric["future_mean"], numeric["future_support"]) if valid else np.nan
        continuous = _target_family(identity["target"]) == "continuous"
        lift_outcome = "future_raw_mean" if continuous else "future_mean"
        lift = _top_quintile_lift(numeric, lift_outcome) if valid else np.nan
        high_low = np.nan
        if "level" in numeric:
            high = numeric.loc[numeric["level"].astype(str).eq("High")]
            low = numeric.loc[numeric["level"].astype(str).eq("Low")]
            if not high.empty and not low.empty:
                high_outcome = pd.to_numeric(
                    high.get(lift_outcome, pd.Series(index=high.index, dtype=float)),
                    errors="coerce",
                )
                low_outcome = pd.to_numeric(
                    low.get(lift_outcome, pd.Series(index=low.index, dtype=float)),
                    errors="coerce",
                )
                high_valid = np.isfinite(high_outcome) & high["future_support"].gt(0)
                low_valid = np.isfinite(low_outcome) & low["future_support"].gt(0)
                if high_valid.any() and low_valid.any():
                    high_mean = np.average(
                        high_outcome[high_valid], weights=high.loc[high_valid, "future_support"]
                    )
                    low_mean = np.average(
                        low_outcome[low_valid], weights=low.loc[low_valid, "future_support"]
                    )
                    high_low = float(high_mean / low_mean) if low_mean != 0 else np.nan
        rho_l, rho_u, lift_l, lift_u = _entity_transfer_bootstrap(
            numeric,
            values,
            lift_outcome_column=lift_outcome,
        ) if valid else (np.nan,) * 4
        rows.append(
            {
                **identity,
                "stratum_type": "scale_contract" if continuous else "overall",
                "stratum_value": (
                    "rank_log_outcome__lift_original_days" if continuous else "all"
                ),
                "n_common_entities": int(len(numeric)),
                "future_support": int(numeric["future_support"].sum()),
                "unweighted_spearman": unweighted,
                "weighted_spearman": weighted,
                "top_quintile_lift": lift,
                "high_low_risk_ratio": high_low,
                "spearman_ci_lower": rho_l,
                "spearman_ci_upper": rho_u,
                "lift_ci_lower": lift_l,
                "lift_ci_upper": lift_u,
                "bootstrap_unit": "entity_cluster",
                "bootstrap_resamples": 500 if valid else 0,
                "valid": bool(valid),
                "invalid_reason": "" if valid else "fewer_than_10_or_constant_common_entities",
            }
        )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), ENTITY_TRANSFER_SCHEMA),
        PRIMARY_KEYS["PROFILE_FUTURE_ENTITY_TRANSFER.csv"],
    )


def _binary_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    usable = np.isfinite(y) & np.isfinite(p)
    y, p = y[usable], p[usable]
    if not len(y):
        return {"log_loss": np.nan, "brier": np.nan}
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((y - p) ** 2)),
    }


def _continuous_scores(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    usable = np.isfinite(y) & np.isfinite(score)
    y, score = y[usable], score[usable]
    if not len(y):
        return {"log_mae": np.nan, "log_rmse": np.nan}
    error = y - score
    return {
        "log_mae": float(np.mean(np.abs(error))),
        "log_rmse": float(math.sqrt(np.mean(error * error))),
    }


def aggregate_support_strata(order_scoring: pd.DataFrame) -> pd.DataFrame:
    if order_scoring.empty:
        return _empty(SUPPORT_STRATA_SCHEMA)
    scored = order_scoring.copy()
    scored["support_stratum"] = [
        _support_stratum(value, status)
        for value, status in zip(scored["history_support"], scored["mapping_status"])
    ]
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days", "support_stratum",
    ]
    for values, group in scored.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        valid = group.loc[
            _coerce_bool(group["eligible_for_metric"])
            & np.isfinite(pd.to_numeric(group["target_value"], errors="coerce"))
        ].copy()
        event_count = float(pd.to_numeric(valid["target_value"], errors="coerce").sum()) if _target_family(identity["target"]) == "binary" else np.nan
        base = {
            **identity,
            "n_orders": int(len(group)),
            "n_events": event_count,
            "n_entities": int(valid["entity_id"].nunique()),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
        outcome = float(pd.to_numeric(valid["target_value"], errors="coerce").mean()) if len(valid) else np.nan
        lower, upper = _cluster_mean_ci(valid, "target_value", (values, "support_outcome")) if len(valid) else (np.nan, np.nan)
        rows.append(
            {
                **base,
                "metric_name": "future_event_rate" if _target_family(identity["target"]) == "binary" else "future_mean_log_outcome",
                "reference_id": "observed_future",
                "estimate": outcome,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid": bool(len(valid)),
                "invalid_reason": "" if len(valid) else "no_observed_future_outcome",
            }
        )
        rows.append(
            {
                **base,
                "metric_name": "order_share",
                "reference_id": "all_placed_future_exposure",
                "estimate": float(len(group) / len(scored.loc[
                    scored["candidate_id"].eq(identity["candidate_id"])
                    & scored["period"].eq(identity["period"])
                    & pd.to_numeric(scored["horizon_days"], errors="coerce").eq(float(identity["horizon_days"]))
                ])),
                "valid": True,
                "invalid_reason": "",
            }
        )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), SUPPORT_STRATA_SCHEMA),
        PRIMARY_KEYS["PROFILE_SUPPORT_STRATA.csv"],
    )


def aggregate_cold_start(order_scoring: pd.DataFrame) -> pd.DataFrame:
    if order_scoring.empty:
        return _empty(COLD_START_SCHEMA)
    scored = order_scoring.copy()
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days", "mapping_status",
    ]
    denominators = scored.groupby(
        ["candidate_id", "period", "horizon_days"], sort=True, dropna=False,
        observed=True,
    ).size()
    for values, group in scored.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        valid = group.loc[
            _coerce_bool(group["target_observed"])
            & np.isfinite(pd.to_numeric(group["target_value"], errors="coerce"))
        ].copy()
        denominator = int(denominators.loc[(identity["candidate_id"], identity["period"], identity["horizon_days"])])
        estimate = float(pd.to_numeric(valid["target_value"], errors="coerce").mean()) if len(valid) else np.nan
        lower, upper = _cluster_mean_ci(valid, "target_value", (values, "cold_start")) if len(valid) else (np.nan, np.nan)
        rows.append(
            {
                **identity,
                "n_orders": int(len(group)),
                "order_share": len(group) / denominator if denominator else np.nan,
                "metric_name": "future_event_rate" if _target_family(identity["target"]) == "binary" else "future_mean_log_outcome",
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid": bool(len(valid)),
                "invalid_reason": "" if len(valid) else "no_observed_future_outcome",
            }
        )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), COLD_START_SCHEMA),
        PRIMARY_KEYS["PROFILE_COLD_START_RESULTS.csv"],
    )


def _hrd_date_column(frame: pd.DataFrame) -> str | None:
    return next((name for name in ("date", "purchase_date", "snapshot_date") if name in frame), None)


def _definition_phase_table(
    hrd_days: pd.DataFrame,
    hrd_phases: pd.DataFrame | None,
    definition: str,
) -> pd.DataFrame:
    """Return one deterministic date/phase table for an HRD definition.

    Both definition-specific wide columns and V1.1's long
    ``definition,date,phase`` table are accepted.  Overlapping event clusters
    are collapsed by the frozen descriptive priority event, pre-event, nearest
    post-event; this prevents the same future order being counted repeatedly.
    """

    for column in (f"{definition}_phase", f"phase_{definition}"):
        if column in hrd_days:
            date_column = _hrd_date_column(hrd_days)
            if date_column is None:
                return pd.DataFrame(columns=["_purchase_date", "phase"])
            result = hrd_days[[date_column, column]].rename(columns={column: "phase"})
            result["_purchase_date"] = pd.to_datetime(result[date_column], errors="coerce").dt.normalize()
            raw_phase = result["phase"].astype(str)
            result["phase"] = np.select(
                [
                    raw_phase.eq("event"),
                    raw_phase.eq("pre_event"),
                    raw_phase.str.startswith("post_event"),
                ],
                ["event", "pre_event", "post_event"],
                default=raw_phase,
            )
            return result[["_purchase_date", "phase"]].dropna(subset=["_purchase_date"]).drop_duplicates("_purchase_date")
    phases = pd.DataFrame() if hrd_phases is None else hrd_phases.copy()
    if phases.empty:
        return pd.DataFrame(columns=["_purchase_date", "phase"])
    definition_column = next((name for name in ("definition", "hrd_definition") if name in phases), None)
    date_column = _hrd_date_column(phases)
    if definition_column is None or date_column is None or "phase" not in phases:
        return pd.DataFrame(columns=["_purchase_date", "phase"])
    phases = phases.loc[phases[definition_column].astype(str).eq(definition)].copy()
    phases["_purchase_date"] = pd.to_datetime(phases[date_column], errors="coerce").dt.normalize()
    raw_phase = phases["phase"].astype(str)
    phases["phase"] = np.select(
        [raw_phase.eq("event"), raw_phase.eq("pre_event"), raw_phase.str.startswith("post_event")],
        ["event", "pre_event", "post_event"],
        default=raw_phase,
    )
    priority = {"event": 0, "pre_event": 1, "post_event": 2}
    phases["_priority"] = phases["phase"].map(priority).fillna(3)
    phase_day = pd.to_numeric(_series(phases, "phase_day"), errors="coerce").abs().fillna(np.inf)
    phases["_phase_day"] = phase_day
    phases = phases.sort_values(
        ["_purchase_date", "_priority", "_phase_day", definition_column],
        kind="mergesort",
        na_position="last",
    ).drop_duplicates("_purchase_date", keep="first")
    return phases[["_purchase_date", "phase"]].reset_index(drop=True)


def aggregate_hrd_diagnostics(
    order_scoring: pd.DataFrame,
    hrd_days: pd.DataFrame | None,
    config: Mapping[str, object] | None = None,
    *,
    hrd_phases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = load_config() if config is None else config
    if order_scoring.empty:
        return _empty(HRD_SCHEMA)
    if hrd_days is None or hrd_days.empty or _hrd_date_column(hrd_days) is None:
        selected = order_scoring.drop_duplicates(
            ["candidate_id", "base_candidate_id", "target", "granularity", "period", "horizon_days"]
        )
        rows = []
        for _, row in selected.iterrows():
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "base_candidate_id": row["base_candidate_id"],
                    "target": row["target"],
                    "granularity": row["granularity"],
                    "period": row["period"],
                    "hrd_definition": cfg["hrd"]["primary_definition"],
                    "regime": "unavailable",
                    "phase": "unavailable",
                    "horizon_days": row["horizon_days"],
                    "n_days": 0,
                    "n_orders": 0,
                    "n_entities": 0,
                    "historical_support": np.nan,
                    "metric_name": "future_outcome",
                    "estimate": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "valid": False,
                    "invalid_reason": "hrd_daily_labels_not_persisted",
                }
            )
        return _sort_frame(_normalise_schema(pd.DataFrame(rows), HRD_SCHEMA), PRIMARY_KEYS["PROFILE_HRD_DIAGNOSTICS.csv"])
    hrd = hrd_days.copy()
    date_column = str(_hrd_date_column(hrd))
    hrd["_purchase_date"] = pd.to_datetime(hrd[date_column], errors="coerce").dt.normalize()
    definitions = [name for name in cfg["hrd"]["all_definitions"] if name in hrd]
    if not definitions and "regime" in hrd:
        definitions = [cfg["hrd"]["primary_definition"]]
        hrd[definitions[0]] = hrd["regime"].astype(str).str.upper().eq("HRD")
    scored = order_scoring.copy()
    scored["_purchase_date"] = pd.to_datetime(scored["purchase_timestamp"], errors="coerce").dt.normalize()
    rows: list[dict[str, object]] = []
    for definition in definitions:
        labels = hrd[["_purchase_date", definition]].drop_duplicates("_purchase_date")
        joined = scored.merge(labels, on="_purchase_date", how="left", validate="many_to_one")
        joined["regime"] = np.where(_coerce_bool(joined[definition]), "HRD", "BAU")
        phases = _definition_phase_table(hrd, hrd_phases, definition)
        if not phases.empty:
            joined = joined.merge(
                phases, on="_purchase_date", how="left", validate="many_to_one"
            )
        else:
            joined["phase"] = np.nan
        joined["phase"] = joined["phase"].fillna(
            pd.Series(
                np.where(joined["regime"].eq("HRD"), "event_unclustered", "BAU"),
                index=joined.index,
            )
        )
        group_columns = [
            "candidate_id", "base_candidate_id", "target", "granularity", "period",
            "regime", "phase", "horizon_days",
        ]
        for values, group in joined.groupby(
            group_columns, sort=True, dropna=False, observed=True
        ):
            identity = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
            outcome = group.loc[
                _coerce_bool(group["target_observed"])
                & np.isfinite(pd.to_numeric(group["target_value"], errors="coerce"))
            ].copy()
            mapped = group.loc[
                group["mapping_status"].astype(str).ne("missing_mapping")
            ].copy() if "mapping_status" in group else group.copy()
            mapped["_profile_score"] = pd.to_numeric(
                _series(mapped, "profile_score"), errors="coerce"
            )
            mapped["_history_support"] = pd.to_numeric(
                _series(mapped, "history_support"), errors="coerce"
            )
            score_rows = mapped.loc[np.isfinite(mapped["_profile_score"])].copy()
            support_rows = mapped.loc[np.isfinite(mapped["_history_support"])].copy()
            historical_support = (
                float(support_rows["_history_support"].median())
                if len(support_rows) else np.nan
            )

            def append_metric(
                metric_name: str,
                estimate: object,
                subset: pd.DataFrame,
                *,
                ci: tuple[float, float] = (np.nan, np.nan),
                invalid_reason: str,
            ) -> None:
                finite_estimate = bool(np.isfinite(pd.to_numeric(pd.Series([estimate]), errors="coerce").iloc[0]))
                rows.append(
                    {
                        **identity,
                        "hrd_definition": definition,
                        "n_days": int(subset["_purchase_date"].nunique()) if not subset.empty else 0,
                        "n_orders": int(len(subset)),
                        "n_entities": int(subset["entity_id"].dropna().nunique()) if "entity_id" in subset else 0,
                        "historical_support": historical_support,
                        "metric_name": metric_name,
                        "estimate": estimate,
                        "ci_lower": ci[0],
                        "ci_upper": ci[1],
                        "valid": finite_estimate,
                        "invalid_reason": "" if finite_estimate else invalid_reason,
                    }
                )

            outcome_estimate = (
                float(pd.to_numeric(outcome["target_value"], errors="coerce").mean())
                if len(outcome) else np.nan
            )
            outcome_ci = _cluster_mean_ci(
                outcome, "target_value", (definition, values, "future_outcome")
            ) if len(outcome) else (np.nan, np.nan)
            append_metric(
                "future_event_rate" if _target_family(identity["target"]) == "binary" else "future_mean_log_outcome",
                outcome_estimate,
                outcome,
                ci=outcome_ci,
                invalid_reason="no_observed_future_outcome",
            )
            for metric_name, quantile in (
                ("profile_score_p25", 0.25),
                ("profile_score_median", 0.50),
                ("profile_score_p75", 0.75),
            ):
                append_metric(
                    metric_name,
                    float(score_rows["_profile_score"].quantile(quantile)) if len(score_rows) else np.nan,
                    score_rows,
                    invalid_reason="no_finite_mapped_profile_scores",
                )
            append_metric(
                "historical_support_median",
                historical_support,
                support_rows,
                invalid_reason="no_finite_mapped_historical_support",
            )
            for level in ("Unknown", "Low", "Medium", "High"):
                level_outcome = outcome.loc[
                    outcome["level"].astype(str).eq(level)
                ].copy() if "level" in outcome else outcome.iloc[0:0].copy()
                level_estimate = (
                    float(pd.to_numeric(level_outcome["target_value"], errors="coerce").mean())
                    if len(level_outcome) else np.nan
                )
                level_ci = _cluster_mean_ci(
                    level_outcome,
                    "target_value",
                    (definition, values, "level", level),
                ) if len(level_outcome) else (np.nan, np.nan)
                append_metric(
                    f"future_outcome_by_level_{level}",
                    level_estimate,
                    level_outcome,
                    ci=level_ci,
                    invalid_reason="no_observed_future_outcome_in_level",
                )
    return _sort_frame(_normalise_schema(pd.DataFrame(rows), HRD_SCHEMA), PRIMARY_KEYS["PROFILE_HRD_DIAGNOSTICS.csv"])


def aggregate_ablations(
    rich_scoring: pd.DataFrame,
    selected_candidates: pd.DataFrame,
) -> pd.DataFrame:
    if rich_scoring.empty:
        return _empty(ABLATION_SCHEMA)
    scored = rich_scoring.copy()
    public, scored = normalize_order_scoring(scored, selected_candidates)
    # ``normalize_order_scoring`` returns a rich key-aligned frame as its
    # second value.  In particular, retain runner-persisted ``raw_score`` and
    # the matched P1 ``shrinkage_score`` so a P2 adjustment is never compared
    # with itself under two labels.
    metadata = selected_candidates.set_index("candidate_id") if not selected_candidates.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id", "base_candidate_id", "target", "granularity", "period", "horizon_days",
    ]
    for values, group in scored.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        valid = group.loc[
            _coerce_bool(group["eligible_for_metric"])
            & np.isfinite(pd.to_numeric(group["target_value"], errors="coerce"))
        ].copy()
        estimator = str(metadata.loc[identity["candidate_id"], "estimator"]) if not metadata.empty and identity["candidate_id"] in metadata.index else ""
        variants: list[tuple[str, str, str, str]] = [
            ("parent_only", "parent_score", "overall", "all"),
        ]
        if "raw_score" in valid:
            variants.append(("raw_score_only", "raw_score", "overall", "all"))
        if estimator == "P2":
            if "shrinkage_score" in valid:
                variants.append(
                    ("shrinkage_score", "shrinkage_score", "overall", "all")
                )
            variants.extend(
                [
                    ("adjusted_score", "profile_score", "overall", "all"),
                    ("adjusted_without_support_threshold", "profile_score", "overall", "all"),
                ]
            )
        else:
            variants.append(("profile_score", "profile_score", "overall", "all"))
        for ablation_id, score_column, stratum_type, stratum_value in variants:
            subset = valid
            if ablation_id != "adjusted_without_support_threshold" and not metadata.empty and identity["candidate_id"] in metadata.index:
                minimum = float(metadata.loc[identity["candidate_id"], "min_support"])
                subset = subset.loc[pd.to_numeric(subset["history_support"], errors="coerce").ge(minimum)]
            y = pd.to_numeric(subset["target_value"], errors="coerce").to_numpy(dtype=float)
            prediction = pd.to_numeric(subset.get(score_column, np.nan), errors="coerce").to_numpy(dtype=float)
            metrics = _binary_scores(y, prediction) if _target_family(identity["target"]) == "binary" else _continuous_scores(y, prediction)
            for metric, estimate in metrics.items():
                rows.append(
                    {
                        **identity,
                        "ablation_id": ablation_id,
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "metric_name": metric,
                        "reference_id": "none",
                        "n_orders": int(len(subset)),
                        "n_events": float(np.nansum(y)) if _target_family(identity["target"]) == "binary" else np.nan,
                        "n_entities": int(subset["entity_id"].nunique()),
                        "estimate": estimate,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "valid": bool(np.isfinite(estimate)),
                        "invalid_reason": "" if np.isfinite(estimate) else "no_valid_ablation_rows",
                    }
                )
        # Required score-by-support ablation.
        valid["_support_stratum"] = [
            _support_stratum(value, status)
            for value, status in zip(valid["history_support"], valid["mapping_status"])
        ]
        for stratum, subset in valid.groupby(
            "_support_stratum", sort=True, observed=True
        ):
            y = pd.to_numeric(subset["target_value"], errors="coerce").to_numpy(dtype=float)
            prediction = pd.to_numeric(subset["profile_score"], errors="coerce").to_numpy(dtype=float)
            metrics = _binary_scores(y, prediction) if _target_family(identity["target"]) == "binary" else _continuous_scores(y, prediction)
            for metric, estimate in metrics.items():
                rows.append(
                    {
                        **identity,
                        "ablation_id": "score_stratified_by_support",
                        "stratum_type": "history_support",
                        "stratum_value": stratum,
                        "metric_name": metric,
                        "reference_id": "none",
                        "n_orders": int(len(subset)),
                        "n_events": float(np.nansum(y)) if _target_family(identity["target"]) == "binary" else np.nan,
                        "n_entities": int(subset["entity_id"].nunique()),
                        "estimate": estimate,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "valid": bool(np.isfinite(estimate)),
                        "invalid_reason": "" if np.isfinite(estimate) else "no_valid_ablation_rows",
                    }
                )
    return _sort_frame(_normalise_schema(pd.DataFrame(rows), ABLATION_SCHEMA), PRIMARY_KEYS["PROFILE_ABLATIONS.csv"])


def _valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "valid" in frame:
        return frame.loc[_coerce_bool(frame["valid"])].copy()
    return frame.copy()


def _figure_applicability(output: Path) -> dict[str, dict[str, object]]:
    """Return a persisted-design-aware applicability receipt for every figure."""

    selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
    pareto = _read_csv(output / "PROFILE_PARETO_FRONTIER.csv")
    selected_granularities = (
        set(selected["granularity"].dropna().astype(str))
        if "granularity" in selected else set()
    )
    selected_estimators = (
        set(selected["estimator"].dropna().astype(str))
        if "estimator" in selected else set()
    )
    has_seller = "seller_id" in selected_granularities
    has_route = bool(selected_granularities & {"state_od", "region_od"})
    has_shrunk_seller = has_seller and bool(selected_estimators & {"P1", "P2"})
    has_shrunk_route = has_route and bool(selected_estimators & {"P1", "P2"})
    has_p2 = "P2" in selected_estimators
    pareto_windows = (
        set(pd.to_numeric(pareto["window_days"], errors="coerce").dropna().astype(int))
        if "window_days" in pareto else set()
    )
    pareto_schemes = (
        set(pareto["scheme"].dropna().astype(str)) if "scheme" in pareto else set()
    )
    pareto_granularities = (
        set(pareto["granularity"].dropna().astype(str))
        if "granularity" in pareto else set()
    )
    default_applicable = not selected.empty
    receipt = {
        stem: {
            "applicable": default_applicable,
            "reason": (
                "promoted_candidates_present"
                if default_applicable else "no_promoted_candidates"
            ),
        }
        for stem in FIGURE_STEMS
    }
    conditions = {
        "01_seller_support_vs_uncertainty": (has_seller, "requires_promoted_seller_profile"),
        "02_route_support_vs_uncertainty": (has_route, "requires_promoted_route_profile"),
        "03_raw_vs_eb_seller_scores": (has_shrunk_seller, "requires_promoted_P1_or_P2_seller_profile"),
        "04_raw_vs_eb_route_scores": (has_shrunk_route, "requires_promoted_P1_or_P2_route_profile"),
        "05_adjusted_vs_unadjusted_scores": (has_p2, "requires_promoted_P2_profile"),
        "06_window_30_60_90_comparison": (
            {30, 60, 90}.issubset(pareto_windows),
            "requires_complete_frozen_30_60_90_development_design",
        ),
        "07_scheme_a_vs_c_comparison": (
            {"A", "C"}.issubset(pareto_schemes),
            "requires_both_frozen_A_and_C_development_designs",
        ),
        "15_seller_cold_start": (has_seller, "requires_promoted_seller_profile"),
        "16_state_od_vs_region_od": (
            {"state_od", "region_od"}.issubset(pareto_granularities),
            "requires_both_state_od_and_region_od_development_designs",
        ),
    }
    for stem, (design_applicable, requirement) in conditions.items():
        applicable = bool(default_applicable and design_applicable)
        receipt[stem] = {
            "applicable": applicable,
            "reason": (
                "applicable_under_frozen_design"
                if applicable else (
                    "no_promoted_candidates" if not default_applicable else requirement
                )
            ),
        }
    return receipt


def _figure_frame(
    stem: str,
    rows: list[dict[str, object]],
    *,
    applicable: bool = True,
    applicability_reason: str = "",
) -> pd.DataFrame:
    if not rows:
        rows = [
            {
                "figure_id": stem,
                "panel": "all",
                "x_value": np.nan,
                "x_label": (
                    "No valid data" if applicable
                    else "Not applicable to promoted/frozen design"
                ),
                "y_value": np.nan,
                "series": "No valid data" if applicable else "Not applicable",
                "weight": 0,
                "valid": False,
                "invalid_reason": (
                    "no_valid_persisted_rows" if applicable
                    else f"not_applicable:{applicability_reason}"
                ),
            }
        ]
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA),
        ("panel", "series", "x_label", "x_value", "y_value"),
    )


def _daily_rows_from_artifact(output: Path) -> pd.DataFrame:
    path = output / "PROFILE_DAILY_SCORES.csv.gz"
    columns = (
        "candidate_id", "target", "granularity", "estimator", "snapshot_date",
        "entity_id", "raw_score", "score", "support",
    )
    return (
        _read_csv(
            path,
            usecols=columns,
            dtype={
                "candidate_id": "category", "target": "category",
                "granularity": "category", "estimator": "category",
                "snapshot_date": "category", "entity_id": "category",
                "support": "Int32",
            },
        )
        if path.exists() else pd.DataFrame()
    )


def build_figure_source(stem: str, output_dir: str | Path) -> pd.DataFrame:
    """Build one concise source table exclusively from persisted outputs."""

    if stem not in FIGURE_STEMS:
        raise ValueError(f"unknown frozen figure stem: {stem}")
    output = Path(output_dir)
    applicability = _figure_applicability(output)[stem]
    # Conditional figures must not accidentally display rows from an
    # inapplicable target/granularity/estimator.  The persisted no-data receipt
    # is itself the auditable result for that frozen design.
    if not bool(applicability["applicable"]):
        return _figure_frame(
            stem,
            [],
            applicable=False,
            applicability_reason=str(applicability["reason"]),
        )
    rows: list[dict[str, object]] = []

    def append(
        *, panel: object = "all", x_value: object = np.nan,
        x_label: object = "", y_value: object = np.nan,
        series: object = "all", weight: object = 1,
        valid: bool = True, invalid_reason: str = "",
    ) -> None:
        rows.append(
            {
                "figure_id": stem, "panel": panel, "x_value": x_value,
                "x_label": x_label, "y_value": y_value, "series": series,
                "weight": weight, "valid": valid,
                "invalid_reason": invalid_reason,
            }
        )

    if stem in {"01_seller_support_vs_uncertainty", "02_route_support_vs_uncertainty"}:
        table = _valid_rows(_read_csv(output / "PROFILE_SUPPORT_UNCERTAINTY.csv"))
        seller = stem.startswith("01_")
        table = table.loc[table["granularity"].astype(str).eq("seller_id") if seller else ~table["granularity"].astype(str).eq("seller_id")]
        for _, row in table.iterrows():
            append(
                panel=row["target"], x_value=row["median_support"],
                x_label=row["support_stratum"],
                y_value=row["median_posterior_se"] if pd.notna(row["median_posterior_se"]) else row["median_interval_width"],
                series=row["granularity"], weight=row["entity_count"],
                valid=bool(pd.notna(row["median_support"])),
            )
    elif stem in {"03_raw_vs_eb_seller_scores", "04_raw_vs_eb_route_scores"}:
        table = _daily_rows_from_artifact(output)
        seller = stem.startswith("03_")
        if not table.empty:
            mask = table["granularity"].astype(str).eq("seller_id") if seller else ~table["granularity"].astype(str).eq("seller_id")
            table = table.loc[mask & table["estimator"].astype(str).isin(["P1", "P2"])].copy()
            table = _sort_frame(table, ("candidate_id", "snapshot_date", "entity_id"))
            if len(table) > 4000:
                table = table.iloc[np.linspace(0, len(table) - 1, 4000, dtype=int)]
            for _, row in table.iterrows():
                append(
                    panel=row["target"], x_value=row["raw_score"],
                    x_label=str(row["entity_id"]), y_value=row["score"],
                    series=row["estimator"], weight=max(float(row.get("support", 1) or 1), 1),
                    valid=bool(pd.notna(row["raw_score"]) and pd.notna(row["score"])),
                )
    elif stem == "05_adjusted_vs_unadjusted_scores":
        table = _valid_rows(_read_csv(output / "PROFILE_ABLATIONS.csv"))
        selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
        p2_ids = set(
            selected.loc[
                selected["estimator"].astype(str).eq("P2"), "candidate_id"
            ].dropna().astype(str)
        ) if {"candidate_id", "estimator"}.issubset(selected.columns) else set()
        table = table.loc[
            table["candidate_id"].astype(str).isin(p2_ids)
            &
            table["ablation_id"].astype(str).isin(
                ["raw_score_only", "shrinkage_score", "adjusted_score"]
            )
        ]
        grouped = table.groupby(
            ["target", "period", "ablation_id", "metric_name"],
            sort=True, observed=True,
        )["estimate"].median().reset_index() if not table.empty else table
        for _, row in grouped.iterrows():
            append(panel=row["target"], x_label=row["ablation_id"], y_value=row["estimate"], series=f"{row['period']}:{row['metric_name']}")
    elif stem in {"06_window_30_60_90_comparison", "07_scheme_a_vs_c_comparison", "16_state_od_vs_region_od"}:
        metrics = _valid_rows(_read_csv(output / "PROFILE_DEVELOPMENT_RESULTS.csv"))
        # These are design-space comparisons, so their metadata must come from
        # the complete frozen development frontier rather than only the small
        # promoted subset.  Falling back to selected metadata keeps an explicit
        # no/frontier synthetic fixture renderable without changing production
        # semantics.
        metadata = _read_csv(output / "PROFILE_PARETO_FRONTIER.csv")
        if metadata.empty:
            metadata = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
        metadata_columns = [
            "candidate_id", "window_days", "scheme", "granularity"
        ]
        metadata = (
            metadata.loc[:, metadata_columns].drop_duplicates("candidate_id")
            if not metadata.empty and set(metadata_columns).issubset(metadata.columns)
            else pd.DataFrame(columns=metadata_columns)
        )
        table = metrics.merge(
            metadata.rename(columns={"granularity": "selected_granularity"}),
            on="candidate_id", how="left", validate="many_to_one",
        ) if not metrics.empty and not metadata.empty else pd.DataFrame()
        wanted = table["metric_name"].astype(str).isin([
            "delta_log_loss_candidate_minus_reference",
            "parent_minus_candidate_log_mae",
            "weighted_future_spearman",
        ]) if not table.empty else pd.Series(dtype=bool)
        table = table.loc[wanted].copy() if not table.empty else table
        if stem.startswith("06_"):
            grouped = table.groupby(
                ["target", "window_days", "metric_name"],
                sort=True, observed=True,
            )["estimate"].median().reset_index() if not table.empty else table
            for _, row in grouped.iterrows():
                append(panel=row["target"], x_value=row["window_days"], x_label=str(row["window_days"]), y_value=row["estimate"], series=row["metric_name"])
        elif stem.startswith("07_"):
            grouped = table.groupby(
                ["target", "scheme", "metric_name"],
                sort=True, observed=True,
            )["estimate"].median().reset_index() if not table.empty else table
            for _, row in grouped.iterrows():
                append(panel=row["target"], x_label=row["scheme"], y_value=row["estimate"], series=row["metric_name"])
        else:
            table = table.loc[table["selected_granularity"].astype(str).isin(["state_od", "region_od"])] if not table.empty else table
            grouped = table.groupby(
                ["target", "selected_granularity", "metric_name"],
                sort=True, observed=True,
            )["estimate"].median().reset_index() if not table.empty else table
            for _, row in grouped.iterrows():
                append(panel=row["target"], x_label=row["selected_granularity"], y_value=row["estimate"], series=row["metric_name"])
    elif stem in {"08_development_future_rank_transfer", "09_confirmation_future_rank_transfer", "10_top_quintile_future_lift"}:
        raw_transfer = _read_csv(output / "PROFILE_FUTURE_ENTITY_TRANSFER.csv")
        table = _valid_rows(raw_transfer)
        if stem.startswith("08_"):
            table = table.loc[table["period"].astype(str).eq("development")]
            raw_period = raw_transfer.loc[
                raw_transfer["period"].astype(str).eq("development")
            ] if not raw_transfer.empty else raw_transfer
            metric = "weighted_spearman"
        elif stem.startswith("09_"):
            table = table.loc[table["period"].astype(str).eq("confirmation")]
            raw_period = raw_transfer.loc[
                raw_transfer["period"].astype(str).eq("confirmation")
            ] if not raw_transfer.empty else raw_transfer
            metric = "weighted_spearman"
        else:
            raw_period = raw_transfer
            metric = "top_quintile_lift"
        for _, row in table.iterrows():
            append(
                panel=row["target"], x_value=pd.Timestamp(row["anchor_date"]).toordinal() if pd.notna(row["anchor_date"]) else np.nan,
                x_label=str(row["anchor_date"]), y_value=row[metric],
                series=f"{row['period']}:{row['granularity']}", weight=row["future_support"],
                valid=bool(pd.notna(row[metric])),
            )
        if not rows and stem.startswith("09_") and not raw_period.empty:
            reasons = sorted(
                set(
                    raw_period["invalid_reason"].fillna("").astype(str)
                    .loc[lambda values: values.str.len().gt(0)]
                )
            )
            append(
                panel="confirmation",
                x_label="Statistically invalid confirmation rank evidence",
                series="explicit_invalid_evaluation_receipt",
                valid=False,
                invalid_reason="evaluation_invalid:" + (";".join(reasons) or "reason_missing"),
            )
    elif stem == "11_future_outcome_by_level":
        table = _valid_rows(_read_csv(output / "PROFILE_LEVEL_RESULTS.csv"))
        table = table.loc[
            table["metric_name"].astype(str).isin(
                ["future_event_rate", "future_original_days_mean"]
            )
            & table["level"].astype(str).isin(
                ["Unknown", "Low", "Medium", "High"]
            )
        ]
        for _, row in table.iterrows():
            append(
                panel=f"{row['target']}:{row['period']}",
                x_label=row["level"],
                y_value=row["estimate"],
                series=f"{row['granularity']}:{row['horizon_days']}d:{row['metric_name']}",
                weight=row["future_support"],
            )
    elif stem == "12_daily_profile_stability":
        raw_stability = _read_csv(output / "PROFILE_DAILY_STABILITY.csv")
        table = _valid_rows(raw_stability)
        if not table.empty:
            table["_month"] = pd.to_datetime(
                table["snapshot_date"], errors="coerce"
            ).dt.to_period("M").astype(str)
            table["_rho"] = pd.to_numeric(
                table["day_to_day_spearman"], errors="coerce"
            )
            table = table.loc[np.isfinite(table["_rho"])].copy()
        grouped = (
            table.groupby(
                ["target", "granularity", "period", "_month"],
                sort=True, observed=True,
            ).agg(
                median=("_rho", "median"),
                p25=("_rho", lambda values: values.quantile(0.25)),
                p75=("_rho", lambda values: values.quantile(0.75)),
                n_days=("_rho", "size"),
            ).reset_index()
            if not table.empty else table
        )
        for _, row in grouped.iterrows():
            month = pd.Timestamp(f"{row['_month']}-01")
            for statistic in ("median", "p25", "p75"):
                append(
                    panel=f"{row['target']}:{row['period']}",
                    x_value=month.toordinal(),
                    x_label=row["_month"],
                    y_value=row[statistic],
                    series=f"{row['granularity']}:{statistic}",
                    weight=row["n_days"],
                )
        if not rows and not raw_stability.empty:
            reasons = sorted(
                set(
                    raw_stability["invalid_reason"].fillna("").astype(str)
                    .loc[lambda values: values.str.len().gt(0)]
                )
            )
            append(
                panel="all_periods",
                x_label="Statistically invalid daily stability evidence",
                series="explicit_invalid_evaluation_receipt",
                valid=False,
                invalid_reason="evaluation_invalid:" + (";".join(reasons) or "reason_missing"),
            )
    elif stem == "13_level_transition_heatmap":
        table = _valid_rows(_read_csv(output / "PROFILE_LEVEL_TRANSITIONS.csv"))
        grouped = table.groupby(
            ["period", "from_level", "to_level"], sort=True, observed=True
        ).agg(
            y_value=("transition_probability", "mean"),
            weight=("transition_count", "sum"),
        ).reset_index() if not table.empty else table
        for _, row in grouped.iterrows():
            append(panel=row["period"], x_label=row["to_level"], y_value=row["y_value"], series=row["from_level"], weight=row["weight"])
    elif stem == "14_coverage_by_support_threshold":
        table = _valid_rows(_read_csv(output / "PROFILE_DEVELOPMENT_RESULTS.csv"))
        selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
        table = table.loc[table["metric_name"].astype(str).eq("support_qualified_coverage")].merge(selected[["candidate_id", "min_support"]], on="candidate_id", how="left", validate="many_to_one") if not table.empty and not selected.empty else pd.DataFrame()
        for _, row in table.iterrows():
            append(panel=row["target"], x_value=row["min_support"], x_label=str(row["min_support"]), y_value=row["estimate"], series=row["granularity"])
    elif stem == "15_seller_cold_start":
        table = _valid_rows(_read_csv(output / "PROFILE_COLD_START_RESULTS.csv"))
        table = table.loc[table["granularity"].astype(str).eq("seller_id")]
        for _, row in table.iterrows():
            append(panel=f"{row['target']}:{row['period']}", x_label=row["mapping_status"], y_value=row["order_share"], series=f"{row['horizon_days']}d", weight=row["n_orders"])
    elif stem == "17_development_vs_confirmation":
        tables = []
        for period, name in (("development", "PROFILE_DEVELOPMENT_RESULTS.csv"), ("confirmation", "PROFILE_CONFIRMATION_RESULTS.csv")):
            table = _valid_rows(_read_csv(output / name))
            if not table.empty:
                table = table.loc[table["metric_name"].astype(str).isin([
                    "delta_log_loss_candidate_minus_reference",
                    "parent_minus_candidate_log_mae",
                    "weighted_future_spearman",
                ])]
                table["_period"] = period
                tables.append(table)
        table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        grouped = table.groupby(
            ["target", "_period", "metric_name"], sort=True, observed=True
        )["estimate"].median().reset_index() if not table.empty else table
        for _, row in grouped.iterrows():
            append(panel=row["target"], x_label=row["_period"], y_value=row["estimate"], series=row["metric_name"])
    elif stem == "18_terminal_stress":
        table = _valid_rows(_read_csv(output / "PROFILE_TERMINAL_STRESS.csv"))
        table = table.loc[table["metric_name"].astype(str).isin([
            "delta_log_loss_candidate_minus_reference",
            "parent_minus_candidate_log_mae",
            "weighted_future_spearman",
            "future_seen_coverage",
            "profile_score_p10_difference",
            "profile_score_median_difference",
            "profile_score_p90_difference",
            "profile_score_wasserstein_distance",
        ])]
        for _, row in table.iterrows():
            append(panel=row["target"], x_label=f"{row['horizon_days']}d", y_value=row["estimate"], series=row["metric_name"], weight=row["n_orders"])
    return _figure_frame(
        stem,
        rows,
        applicable=bool(applicability["applicable"]),
        applicability_reason=str(applicability["reason"]),
    )


def render_figure_from_source(source_path: str | Path, figure_path: str | Path) -> None:
    """Render a PNG after reopening (never receiving) its persisted source."""

    source = Path(source_path)
    figure = Path(figure_path)
    stem = source.stem
    if stem not in FIGURE_STEMS:
        raise ValueError(f"source filename is not a frozen figure stem: {stem}")
    data = _read_csv(source)
    if tuple(data.columns) != FIGURE_SOURCE_SCHEMA:
        raise ValueError(f"figure source schema mismatch: {source}")
    valid = data.loc[
        _coerce_bool(data["valid"])
        & np.isfinite(pd.to_numeric(data["y_value"], errors="coerce"))
    ].copy()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 8,
            "axes.titlesize": 10, "axes.labelsize": 8,
            "legend.fontsize": 7, "figure.dpi": 120,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
    ax.set_title(FIGURE_TITLES[stem])
    if valid.empty:
        reason = "; ".join(sorted(set(data["invalid_reason"].dropna().astype(str)))) or "no_valid_persisted_rows"
        ax.text(0.5, 0.55, "No valid data", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.43, reason, ha="center", va="center", fontsize=8, color="0.35", transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
    elif stem == "13_level_transition_heatmap":
        order = ["Unknown", "Low", "Medium", "High"]
        pivot = valid.pivot_table(index="series", columns="x_label", values="y_value", aggfunc="mean").reindex(index=order, columns=order)
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(order)), order)
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel("To level")
        ax.set_ylabel("From level")
        for y in range(len(order)):
            for x in range(len(order)):
                if np.isfinite(matrix[y, x]):
                    ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Transition probability")
    elif stem in {
        "01_seller_support_vs_uncertainty", "02_route_support_vs_uncertainty",
        "03_raw_vs_eb_seller_scores", "04_raw_vs_eb_route_scores",
    }:
        for series, group in valid.groupby(
            "series", sort=True, observed=True
        ):
            size = np.clip(pd.to_numeric(group["weight"], errors="coerce").fillna(1).to_numpy(dtype=float), 1, None)
            size = 12 + 28 * np.sqrt(size / max(size.max(), 1))
            ax.scatter(pd.to_numeric(group["x_value"], errors="coerce"), pd.to_numeric(group["y_value"], errors="coerce"), s=size, alpha=0.65, label=str(series), edgecolors="none")
        ax.set_xlabel("Support" if stem.startswith(("01_", "02_")) else "Raw score")
        ax.set_ylabel("Uncertainty" if stem.startswith(("01_", "02_")) else "EB/adjusted score")
        if valid["series"].nunique() <= 12:
            ax.legend(frameon=False)
    else:
        labels = list(dict.fromkeys(valid["x_label"].astype(str)))
        numeric_x = pd.to_numeric(valid["x_value"], errors="coerce")
        use_numeric = np.isfinite(numeric_x).all() and numeric_x.nunique() > 1
        positions = {label: index for index, label in enumerate(labels)}
        for series, group in valid.groupby(
            "series", sort=True, observed=True
        ):
            group = group.copy()
            x = pd.to_numeric(group["x_value"], errors="coerce").to_numpy(dtype=float) if use_numeric else group["x_label"].astype(str).map(positions).to_numpy(dtype=float)
            order_index = np.argsort(x, kind="mergesort")
            ax.plot(x[order_index], pd.to_numeric(group["y_value"], errors="coerce").to_numpy(dtype=float)[order_index], marker="o", markersize=3.5, linewidth=1.2, label=str(series))
        if not use_numeric:
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_ylabel("Persisted estimate")
        if valid["series"].nunique() <= 12:
            ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    temporary = figure.with_name(f".{figure.name}.tmp.{os.getpid()}")
    fig.savefig(
        temporary,
        format="png",
        dpi=120,
        metadata={"Software": "dynamic_profile_profile_validation_v1"},
    )
    plt.close(fig)
    os.replace(temporary, figure)


def create_required_figures(output_dir: str | Path) -> dict[str, dict[str, object]]:
    """Create exactly the frozen 18 source/PNG pairs in deterministic order."""

    output = Path(output_dir)
    source_dir = output / "figure_sources"
    figure_dir = output / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for stem in FIGURE_STEMS:
        source = source_dir / f"{stem}.csv"
        figure = figure_dir / f"{stem}.png"
        table = build_figure_source(stem, output)
        write_csv(table, source, FIGURE_SOURCE_SCHEMA, ("panel", "series", "x_label", "x_value", "y_value"))
        render_figure_from_source(source, figure)
        records[figure.name] = {
            "relative_path": str(figure.relative_to(output)),
            "source_relative_path": str(source.relative_to(output)),
            "sha256": sha256_file(figure),
            "source_sha256": sha256_file(source),
            "source_rows": int(len(table)),
        }
    return records


def write_data_dictionary(output_dir: str | Path) -> None:
    output = Path(output_dir)
    lines = [
        "# Profile Validation Data Dictionary",
        "",
        "Generated from the reporting schema contract. Analytical tables use deterministic CSV formatting; missing values are empty fields.",
        "",
        "## Sample semantics",
        "",
        "- Profile history and primary observed labels: target-valid rows from the 96,470-order canonical delivered sample.",
        "- Future exposure, mapping, cold-start and unresolved denominator: all 99,441 placed orders; each anchor's future denominator is the purchase cohort in `[t, t+7d)` without delivered-only filtering.",
        "- Missing mappings and mapped cold starts are distinct states; neither is coded as a negative outcome.",
        "- All profile histories obey strict `label_available_at < snapshot_date`.",
        "",
        "## Tables",
        "",
    ]
    descriptions = {
        "PROFILE_CONSTRUCTION_AUDIT.csv": "Construction provenance. `future_denominator_sample=all_placed_99441_next_7d_purchase_cohort` denotes the all-placed `[t, t+7d)` purchase-cohort denominator; a more precise persisted runner label is preserved byte-for-value rather than overwritten.",
        "PROFILE_DAILY_SCORES.csv": "Compact index for the selected daily row artifact PROFILE_DAILY_SCORES.csv.gz.",
        "PROFILE_FUTURE_ORDER_SCORING.csv": "All-placed future exposure with separately flagged observed target rows.",
        "PROFILE_HRD_DIAGNOSTICS.csv": "Retrospective regime descriptions only; HRD is never a profile predictor.",
        "PROFILE_ABLATIONS.csv": "Standalone profile-estimator ablations, not a final order-model ladder.",
        "PROFILE_DAILY_STABILITY.csv": "Consecutive-day score stability on common entities; cold-start entry/exit rates use the adjacent-day entity union as their explicit denominator. Monthly medians/IQRs are persisted in figure source 12.",
        "PROFILE_LEVEL_TRANSITIONS.csv": "Repeated entity/time transition point estimates; iid Wilson intervals are prohibited and unsupported CIs remain NA with an explicit reason.",
        "PROFILE_FUTURE_ENTITY_TRANSFER.csv": "Continuous weighted rank transfer uses the log-outcome scale, while continuous top-quintile lift and High/Low ratio use original days; entity-cluster bootstrap intervals are reported.",
        "PROFILE_TERMINAL_STRESS.csv": "Terminal label availability is the unconditional all-placed proportion available by the fixed retrospective proxy. Score-shift metrics are terminal minus locked confirmation over repeated future-order exposure and are descriptive, not independent-order inference.",
    }
    for name in sorted(CSV_SCHEMAS):
        persisted = _read_csv(output / name)
        columns = (
            tuple(persisted.columns)
            if name in FROZEN_WIDE_SELECTION_TABLES and (output / name).exists()
            else CSV_SCHEMAS[name]
        )
        lines.extend(
            [
                f"### `{name}`",
                "",
                descriptions.get(name, "Deterministic analytical output under the frozen protocol."),
                "",
                "Columns: " + ", ".join(f"`{column}`" for column in columns) + ".",
                "",
                "Primary key: " + ", ".join(f"`{column}`" for column in PRIMARY_KEYS[name]) + ".",
                "",
            ]
        )
    lines.extend(
        [
            "### `PROFILE_DAILY_SCORES.csv.gz`",
            "",
            "Full selected-profile daily rows. Columns begin with the frozen profile base schema and add the frozen candidate/support-level assignment.",
            "",
            "Columns: " + ", ".join(f"`{column}`" for column in DAILY_ROW_SCHEMA) + ".",
            "",
            "Primary key: `candidate_id`, `snapshot_date`, `entity_id`.",
            "",
        ]
    )
    path = output / "PROFILE_DATA_DICTIONARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_number(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "NA"
    if float(numeric).is_integer() and abs(float(numeric)) >= 10:
        return f"{int(numeric):,}"
    return f"{float(numeric):.{digits}g}"


def _summary_counts(values: pd.Series) -> str:
    if values.empty:
        return "none"
    counts = values.fillna("missing").astype(str).value_counts().sort_index()
    return ", ".join(f"{key}={int(value)}" for key, value in counts.items())


def _summary_metric_medians(
    frame: pd.DataFrame,
    metric_names: Sequence[str] | None = None,
    *,
    prefix_columns: Sequence[str] = (),
    maximum: int = 16,
) -> str:
    table = _valid_rows(frame)
    if table.empty or "metric_name" not in table or "estimate" not in table:
        return "not estimable from persisted valid rows"
    if metric_names is not None:
        table = table.loc[table["metric_name"].astype(str).isin(metric_names)].copy()
    table["estimate"] = pd.to_numeric(table["estimate"], errors="coerce")
    table = table.loc[np.isfinite(table["estimate"])].copy()
    if table.empty:
        return "not estimable from persisted valid rows"
    groups = [
        column for column in prefix_columns
        if column in table and column != "metric_name"
    ] + ["metric_name"]
    medians = table.groupby(
        groups, sort=True, dropna=False, observed=True
    )["estimate"].median()
    parts: list[str] = []
    for keys, value in medians.iloc[:maximum].items():
        labels = keys if isinstance(keys, tuple) else (keys,)
        parts.append(f"{'/'.join(map(str, labels))}={_summary_number(value)}")
    if len(medians) > maximum:
        parts.append(f"+{len(medians) - maximum} more groups")
    return "; ".join(parts)


def _support_finding(table: pd.DataFrame, seller: bool) -> str:
    valid = _valid_rows(table)
    if valid.empty:
        return "not estimable from persisted valid rows"
    mask = valid["granularity"].astype(str).eq("seller_id")
    valid = valid.loc[mask if seller else ~mask].copy()
    if valid.empty:
        return "not estimable from persisted valid rows"
    return (
        f"{len(valid):,} valid snapshot×stratum rows; median support="
        f"{_summary_number(pd.to_numeric(valid['median_support'], errors='coerce').median())}; "
        f"median posterior SE={_summary_number(pd.to_numeric(valid['median_posterior_se'], errors='coerce').median())}; "
        f"median 95% width={_summary_number(pd.to_numeric(valid['median_interval_width'], errors='coerce').median())}; "
        f"p90 width={_summary_number(pd.to_numeric(valid['p90_interval_width'], errors='coerce').median())}"
    )


def _design_finding(pareto: pd.DataFrame, column: str) -> str:
    if pareto.empty or column not in pareto:
        return "not estimable from persisted frontier"
    work = pareto.copy()
    for metric in (
        "median_delta_logloss", "median_parent_minus_candidate_mae",
        "median_weighted_spearman", "median_support_qualified_coverage",
    ):
        work[metric] = pd.to_numeric(_series(work, metric), errors="coerce")
    parts: list[str] = []
    for value, group in work.groupby(
        column, sort=True, dropna=False, observed=True
    ):
        eligible = int(_coerce_bool(_series(group, "pareto_eligible", False)).sum())
        binary = group.loc[group["target_family"].astype(str).eq("binary")]
        continuous = group.loc[group["target_family"].astype(str).eq("continuous")]
        parts.append(
            f"{value}: n={len(group)}, eligible={eligible}, "
            f"binary candidate−reference Δlog-loss median={_summary_number(binary['median_delta_logloss'].median())}, "
            f"continuous parent−candidate log-MAE median={_summary_number(continuous['median_parent_minus_candidate_mae'].median())}, "
            f"weighted future Spearman median={_summary_number(group['median_weighted_spearman'].median())}, "
            f"coverage median={_summary_number(group['median_support_qualified_coverage'].median())}"
        )
    return "; ".join(parts)


def _ablation_finding(table: pd.DataFrame, ablations: Sequence[str]) -> str:
    valid = _valid_rows(table)
    if valid.empty:
        return "not estimable from persisted valid rows"
    valid = valid.loc[
        valid["ablation_id"].astype(str).isin(ablations)
        & valid["metric_name"].astype(str).isin(
            ["log_loss", "brier", "log_mae", "log_rmse"]
        )
    ]
    return _summary_metric_medians(
        valid,
        prefix_columns=("ablation_id",),
        maximum=20,
    )


def _persisted_summary_values(
    output: Path,
    work: Path | None = None,
) -> dict[str, object]:
    working = output / "working" if work is None else work

    def read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
    pareto = _read_csv(output / "PROFILE_PARETO_FRONTIER.csv")
    development = _read_csv(output / "PROFILE_DEVELOPMENT_RESULTS.csv")
    confirmation = _read_csv(output / "PROFILE_CONFIRMATION_RESULTS.csv")
    terminal = _read_csv(output / "PROFILE_TERMINAL_STRESS.csv")
    support = _read_csv(output / "PROFILE_SUPPORT_UNCERTAINTY.csv")
    cold = _read_csv(output / "PROFILE_COLD_START_RESULTS.csv")
    strata = _read_csv(output / "PROFILE_SUPPORT_STRATA.csv")
    hrd = _read_csv(output / "PROFILE_HRD_DIAGNOSTICS.csv")
    levels = _read_csv(output / "PROFILE_LEVEL_RESULTS.csv")
    transitions = _read_csv(output / "PROFILE_LEVEL_TRANSITIONS.csv")
    stability = _read_csv(output / "PROFILE_DAILY_STABILITY.csv")
    ablations = _read_csv(output / "PROFILE_ABLATIONS.csv")
    construction = _read_csv(output / "PROFILE_CONSTRUCTION_AUDIT.csv")
    labels_table = _read_csv(working / "CONFIRMATION_LABELS.csv")
    negative = _read_csv(working / "NEGATIVE_DURATION_AUDIT.csv")
    config = read_json(output / "PROFILE_FROZEN_CONFIG.json")
    tails = read_json(output / "FROZEN_TAIL_THRESHOLDS.json")
    freeze = read_json(output / "PROFILE_SELECTION_FREEZE.json")
    manifest = read_json(output / "RUN_MANIFEST.json")
    state = read_json(working / "RUN_STATE.json")

    selected_ids = selected["candidate_id"].astype(str).tolist() if not selected.empty else []
    label_counts = (
        _summary_counts(labels_table["confirmation_label"])
        if not labels_table.empty and "confirmation_label" in labels_table else "none"
    )
    label_by_id = (
        labels_table.set_index("candidate_id")["confirmation_label"].astype(str).to_dict()
        if not labels_table.empty and {"candidate_id", "confirmation_label"}.issubset(labels_table.columns)
        else {}
    )
    recommended = [
        candidate_id for candidate_id in selected_ids
        if label_by_id.get(candidate_id) in {"Strongly confirmed", "Partially confirmed"}
    ]
    not_confirmed = [
        candidate_id for candidate_id in selected_ids
        if label_by_id.get(candidate_id) == "Not confirmed"
    ]

    candidate_definitions: list[str] = []
    for row in selected.to_dict("records"):
        candidate_definitions.append(
            f"`{row.get('candidate_id')}` (target={row.get('target')}, granularity={row.get('granularity')}, "
            f"scheme={row.get('scheme')}, window={row.get('window_days')}d, lag={row.get('lag_days')}d, "
            f"estimator={row.get('estimator')}, parent={row.get('parent_structure')}, "
            f"kappa={row.get('kappa')}, min_support={row.get('min_support', row.get('support_threshold'))}, "
            f"cutoffs={row.get('low_medium_cutoff', row.get('q33'))}/{row.get('medium_high_cutoff', row.get('q67'))})"
        )

    rejection_reasons: list[str] = []
    if not pareto.empty:
        rejected = pareto.loc[
            ~_coerce_bool(_series(pareto, "selected_for_confirmation", False))
        ].copy()
        reason = _series(rejected, "minimum_evidence_failure_reasons", "").astype(str)
        fallback = _series(rejected, "pareto_ineligible_reason", "").astype(str)
        dominated = _series(rejected, "dominated_by", "").astype(str)
        reason = reason.mask(reason.isin(["", "nan"]), fallback)
        reason = reason.mask(reason.isin(["", "nan"]), np.where(dominated.isin(["", "nan"]), "not_selected_by_frozen_tie_break", "pareto_dominated"))
        rejection_reasons.append(_summary_counts(reason))
    if not_confirmed:
        rejection_reasons.append(f"not_confirmed={len(not_confirmed)}")

    negative_parts: list[str] = []
    for row in negative.to_dict("records"):
        target = str(row.get("target", "unknown"))
        affected = row.get("affected_sellers_canonical", row.get("affected_routes_canonical", np.nan))
        negative_parts.append(
            f"{target}: all-placed excluded={_summary_number(row.get('all_placed_excluded'))}, "
            f"canonical excluded={_summary_number(row.get('canonical_excluded'))}, "
            f"canonical affected entities={_summary_number(affected)}, clipped_to_zero={row.get('clipped_to_zero')}"
        )
    if not negative_parts and not construction.empty:
        negative_parts.append(
            f"construction-audit excluded-negative sum={_summary_number(pd.to_numeric(construction['source_orders_excluded_negative'], errors='coerce').sum())}; no clipping was permitted"
        )

    expected_anchors = config.get("time", {}).get("expected_valid_anchor_counts", {})
    anchor_text = ", ".join(
        f"{key}={value}" for key, value in sorted(expected_anchors.items())
    ) or "not persisted"
    tail_text = (
        f"handling={_summary_number(tails.get('handling_tail_threshold_days'))} days; "
        f"transit={_summary_number(tails.get('transit_tail_threshold_days'))} days; "
        f"q={tails.get('quantile', 'NA')}, method={tails.get('method', 'NA')}, "
        f"pre-development availability cutoff={tails.get('availability_end_exclusive', 'NA')}, operator={tails.get('event_operator', 'NA')}"
        if tails else "not persisted"
    )

    primary_metrics = (
        "delta_log_loss_candidate_minus_reference",
        "delta_brier_candidate_minus_reference",
        "parent_minus_candidate_log_mae",
        "weighted_future_spearman",
        "top_quintile_future_lift",
        "support_qualified_coverage",
    )
    selected_dev = (
        development.loc[development["candidate_id"].astype(str).isin(selected_ids)]
        if selected_ids and not development.empty else development.iloc[0:0]
    )
    confirmation_metrics = confirmation.loc[
        ~confirmation["stratum_type"].astype(str).eq("confirmation_label")
    ] if not confirmation.empty else confirmation

    level_primary = _valid_rows(levels)
    level_primary = level_primary.loc[
        level_primary["metric_name"].astype(str).isin(
            ["future_event_rate", "future_original_days_mean"]
        )
    ] if not level_primary.empty else level_primary
    level_text = _summary_metric_medians(
        level_primary,
        prefix_columns=("level", "metric_name"),
        maximum=12,
    )
    monotone_rate = (
        float(_coerce_bool(level_primary["monotone_lmh"]).mean())
        if not level_primary.empty else np.nan
    )
    stable = _valid_rows(stability)
    stability_text = "not estimable from persisted valid rows"
    if not stable.empty:
        stable["_month"] = pd.to_datetime(
            stable["snapshot_date"], errors="coerce"
        ).dt.to_period("M").astype(str)
        stable["_rho"] = pd.to_numeric(stable["day_to_day_spearman"], errors="coerce")
        monthly = stable.loc[np.isfinite(stable["_rho"])].groupby(
            "_month", sort=True, observed=True
        )["_rho"].median()
        stability_text = (
            f"daily Spearman median={_summary_number(stable['_rho'].median())}; "
            f"monthly-median variation median={_summary_number(monthly.median())}, "
            f"IQR=[{_summary_number(monthly.quantile(0.25))}, {_summary_number(monthly.quantile(0.75))}]; "
            f"median level-change share={_summary_number(pd.to_numeric(stable['pct_entities_changing_level'], errors='coerce').median())}; "
            f"cold-start entries/exits={_summary_number(pd.to_numeric(stable['cold_start_entries'], errors='coerce').sum())}/"
            f"{_summary_number(pd.to_numeric(stable['cold_start_exits'], errors='coerce').sum())}; "
            f"median entry/exit rates over adjacent-day entity unions="
            f"{_summary_number(pd.to_numeric(stable['cold_start_entry_rate'], errors='coerce').median())}/"
            f"{_summary_number(pd.to_numeric(stable['cold_start_exit_rate'], errors='coerce').median())}"
        )
    diagonal = _valid_rows(transitions)
    diagonal = diagonal.loc[
        diagonal["from_level"].astype(str).eq(diagonal["to_level"].astype(str))
    ] if not diagonal.empty else diagonal
    transition_text = (
        f"median diagonal persistence probability={_summary_number(pd.to_numeric(diagonal['transition_probability'], errors='coerce').median())}; CIs intentionally NA under repeated entity/time dependence"
        if not diagonal.empty else "not estimable from persisted valid rows"
    )

    cold_text = _summary_metric_medians(
        cold, prefix_columns=("mapping_status", "metric_name"), maximum=12
    )
    cold_share = _valid_rows(cold)
    if not cold_share.empty:
        share = cold_share.groupby(
            "mapping_status", sort=True, observed=True
        )["order_share"].median()
        cold_text += "; median order shares: " + ", ".join(
            f"{key}={_summary_number(value)}" for key, value in share.items()
        )
    high_support = _valid_rows(strata)
    high_support = high_support.loc[
        high_support["support_stratum"].astype(str).isin(
            ["support_5_9", "support_10_19", "support_20_plus"]
        )
    ] if not high_support.empty else high_support
    high_support_text = _summary_metric_medians(
        high_support,
        metric_names=("future_event_rate", "future_mean_log_outcome", "order_share"),
        prefix_columns=("support_stratum", "metric_name"),
        maximum=12,
    )

    hrd_valid = _valid_rows(hrd)
    if not hrd_valid.empty:
        hrd_valid["target_family"] = hrd_valid["target"].map(_target_family)
    hrd_text = _summary_metric_medians(
        hrd_valid,
        metric_names=(
            "future_event_rate", "future_mean_log_outcome",
            "profile_score_median", "historical_support_median",
        ),
        prefix_columns=("target_family", "regime", "metric_name"),
        maximum=16,
    )
    hrd_definitions = (
        sorted(hrd_valid["hrd_definition"].astype(str).unique())
        if not hrd_valid.empty else []
    )

    terminal_valid = _valid_rows(terminal)
    terminal_text = _summary_metric_medians(
        terminal_valid,
        (*primary_metrics,
         "profile_score_p10_difference", "profile_score_median_difference",
         "profile_score_p90_difference", "profile_score_wasserstein_distance"),
        maximum=16,
    )
    maturity_rate = (
        pd.to_numeric(terminal_valid["outcome_observation_rate"], errors="coerce").median()
        if not terminal_valid.empty else np.nan
    )
    maturity_flags = (
        int(_coerce_bool(terminal_valid["maturity_censoring_flag"]).sum())
        if not terminal_valid.empty else 0
    )

    blockers_text = (
        (output / "BLOCKERS.md").read_text(encoding="utf-8")
        if (output / "BLOCKERS.md").exists() else ""
    )
    blockers = [
        line[2:].strip() for line in blockers_text.splitlines()
        if line.startswith("- ")
    ]
    test_receipt = _parse_test_results(output / "TEST_RESULTS.txt")
    commands = list(manifest.get("commands", state.get("commands", [])))
    test_command = str(test_receipt.get("command") or "")
    if test_command and test_command not in commands:
        commands.append(test_command)

    protection_ok = bool(
        manifest
        and manifest.get("protected_hashes_before")
        == manifest.get("protected_hashes_after")
        == manifest.get("protected_hashes_after_tests")
        and manifest.get("control_file_hashes_before")
        == manifest.get("control_file_hashes_after")
        == manifest.get("control_file_hashes_after_tests")
        and manifest.get("stage_gate", {}).get("development_artifact_hashes_match") is True
    )
    freeze_path = output / "PROFILE_SELECTION_FREEZE.json"
    freeze_sidecar = output / "PROFILE_SELECTION_FREEZE.sha256"
    freeze_ok = bool(
        freeze_path.exists() and freeze_sidecar.exists()
        and sha256_file(freeze_path)
        == freeze_sidecar.read_text(encoding="utf-8").strip().split()[0]
    )
    existing_required = [name for name in REQUIRED_ARTIFACTS if (output / name).exists()]
    figures_created = len(list((output / "figures").glob("*.png"))) if (output / "figures").exists() else 0

    return {
        "commands": "\n".join(f"- `{command}`" for command in commands) or "- Not persisted.",
        "source_verdict": (
            f"{'PASS' if protection_ok and freeze_ok else 'FAIL/INCOMPLETE'}; freeze sidecar match={freeze_ok}; "
            f"protected/control hashes unchanged={protection_ok}; embedded frozen development artifacts={len(freeze.get('development_artifact_hashes', {}))}; "
            f"raw hash records={len(manifest.get('raw_file_hashes', {}))}; V1.1 hash records={len(manifest.get('v1_1_file_hashes', {}))}"
        ),
        "sample": (
            f"profile history/primary labels={config.get('sample_contract', {}).get('profile_history', 'NA')}; "
            f"future exposure={config.get('sample_contract', {}).get('future_exposure_denominator', 'NA')}; "
            f"noncanonical observed six={config.get('sample_contract', {}).get('observed_noncanonical_six', 'NA')}; "
            + ("; ".join(negative_parts) if negative_parts else "negative audit not persisted")
        ),
        "tails": tail_text,
        "anchors": anchor_text,
        "seller_support": _support_finding(support, True),
        "route_support": _support_finding(support, False),
        "raw_vs_eb": _ablation_finding(
            ablations, ("raw_score_only", "profile_score", "shrinkage_score")
        ),
        "adjusted_vs_unadjusted": _ablation_finding(
            ablations, ("raw_score_only", "shrinkage_score", "adjusted_score")
        ),
        "windows": _design_finding(pareto, "window_days"),
        "schemes": _design_finding(pareto, "scheme"),
        "pareto": (
            f"candidates={len(pareto):,}; evidence-pass={int(_coerce_bool(_series(pareto, 'minimum_evidence_pass', False)).sum())}; "
            f"eligible={int(_coerce_bool(_series(pareto, 'pareto_eligible', False)).sum())}; "
            f"non-dominated={int(_coerce_bool(_series(pareto, 'pareto_nondominated', False)).sum())}; selected={len(selected):,}"
        ),
        "selected": "\n".join(f"- {item}" for item in candidate_definitions) or "- None (valid negative selection outcome).",
        "confirmation": (
            f"labels: {label_counts}; development medians: {_summary_metric_medians(selected_dev, primary_metrics)}; "
            f"confirmation medians: {_summary_metric_medians(confirmation_metrics, primary_metrics)}"
        ),
        "levels": f"{level_text}; monotone Low<Medium<High share={_summary_number(monotone_rate)}; {stability_text}; {transition_text}",
        "support_cold": f"{cold_text}; high-support strata: {high_support_text}",
        "hrd": (
            f"definitions={','.join(hrd_definitions) or 'none'}; {hrd_text}; HRD remains retrospective and non-predictive"
        ),
        "terminal": (
            f"{terminal_text}; unconditional label-availability median={_summary_number(maturity_rate)}; "
            f"rows below frozen {config.get('terminal_maturity_availability_threshold', 0.95)} threshold={maturity_flags}; fixed proxy is not administrative closure"
        ),
        "failed_confirmation": f"Not confirmed={len(not_confirmed)}; IDs={', '.join(not_confirmed) or 'none'}",
        "recommended": "\n".join(f"- `{value}`" for value in recommended) or "- None.",
        "rejected": "; ".join(rejection_reasons) or "none",
        "blockers": "\n".join(f"- {value}" for value in blockers) or "- None.",
        "tests": (
            f"command=`{test_receipt.get('command') or 'NA'}`; collected={test_receipt.get('collected')}; "
            f"passed={test_receipt.get('passed')}; failed={test_receipt.get('failed')}; "
            f"skipped={test_receipt.get('skipped')}; deselected={test_receipt.get('deselected')}; "
            f"errors={test_receipt.get('errors')}; return_code={test_receipt.get('return_code')}"
        ),
        "files": (
            f"{len(existing_required)}/{len(REQUIRED_ARTIFACTS)} required artifacts currently present; "
            f"{figures_created}/{len(FIGURE_STEMS)} PNGs and "
            f"{len(list((output / 'figure_sources').glob('*.csv'))) if (output / 'figure_sources').exists() else 0}/{len(FIGURE_STEMS)} paired sources"
        ),
    }


def write_summary_skeletons(
    output_dir: str | Path,
    *,
    work_dir: str | Path | None = None,
) -> None:
    """Write the exact 25-item completion report from persisted evidence."""

    output = Path(output_dir)
    work = Path(work_dir) if work_dir is not None else output / "working"
    value = _persisted_summary_values(output, work)
    en = f"""# Dynamic Profile Validation V1 — Completion Report

Every numerical statement below is derived from persisted artifacts. A zero-candidate outcome is valid under the frozen evidence/Pareto rules; unavailable findings are reported rather than imputed.

## 1. Scripts and commands executed

{value['commands']}

## 2. Source, hash, and preservation verdict

{value['source_verdict']}.

## 3. Sample and negative-duration handling

{value['sample']}.

## 4. Frozen tail thresholds

{value['tails']}.

## 5. Development and confirmation anchor counts

{value['anchors']}.

## 6. Seller support and uncertainty

{value['seller_support']}.

## 7. Route support and uncertainty

{value['route_support']}.

## 8. Raw versus empirical-Bayes/shrinkage findings

Median estimates across valid candidate-period-horizon rows: {value['raw_vs_eb']}.

## 9. Adjusted versus unadjusted findings

Matched P2 comparison on the same future IDs: {value['adjusted_vs_unadjusted']}.

## 10. 30/60/90-day findings

{value['windows']}.

## 11. Scheme A versus Scheme C findings

{value['schemes']}.

## 12. Development Pareto frontier

{value['pareto']}.

## 13. Selected candidate definitions

{value['selected']}

## 14. Locked confirmation results

{value['confirmation']}.

## 15. Level separation and churn

{value['levels']}.

## 16. Seen, cold-start, and high-support findings

{value['support_cold']}. Missing mapping is kept separate from mapped cold start.

## 17. HRD descriptive findings

{value['hrd']}.

## 18. Terminal stress findings

{value['terminal']}. Terminal evidence did not redesign or reselect a profile.

## 19. Candidates failing confirmation

{value['failed_confirmation']}.

## 20. Profiles recommended for the next order-prediction phase

Only strongly or partially confirmed promoted profiles are listed, subject to independent audit:

{value['recommended']}

## 21. Rejected profiles and reasons

{value['rejected']}.

## 22. Blockers

{value['blockers']}

## 23. Tests passed and failed

{value['tests']}.

## 24. Files created

{value['files']}. Exact hashes and row counts are in `RUN_MANIFEST.json` and `ARTIFACT_VALIDATION_REPORT.md`.

## 25. Scope confirmation

No final breach/severity order-level prediction ladder or business policy was run. These results validate standalone historical profiles only; they do not establish causality, policy value, production readiness, or incremental value in the later full order model.
"""
    zh = f"""# 动态画像统计验证 V1——完成报告

以下数值均来自已落盘产物。按冻结的证据与 Pareto 规则，零晋级候选是有效负向结果；无法估计的项目不会被填补或猜测。

## 1. 已执行脚本与命令

{value['commands']}

## 2. 来源、哈希与保护结论

{value['source_verdict']}。

## 3. 样本与负时长处理

{value['sample']}。

## 4. 冻结尾部阈值

{value['tails']}。

## 5. 开发期与确认期锚点数

{value['anchors']}。

## 6. 卖家支持度与不确定性

{value['seller_support']}。

## 7. 路线支持度与不确定性

{value['route_support']}。

## 8. 原始估计与经验贝叶斯/收缩估计

有效候选×时期×预测窗行的中位数：{value['raw_vs_eb']}。

## 9. 调整前后比较

相同未来订单 ID 上的 P2 配对比较：{value['adjusted_vs_unadjusted']}。

## 10. 30/60/90 天窗口

{value['windows']}。

## 11. Scheme A 与 Scheme C

{value['schemes']}。

## 12. 开发期 Pareto 前沿

{value['pareto']}。

## 13. 晋级候选的精确定义

{value['selected']}

## 14. 锁定确认期结果

{value['confirmation']}。

## 15. 等级区分与变动

{value['levels']}。

## 16. 已见、冷启动与高支持度结果

{value['support_cold']}。实体映射缺失与已映射冷启动分开报告。

## 17. HRD 描述性结果

{value['hrd']}。

## 18. 终端压力结果

{value['terminal']}。终端结果没有用于重新设计或重新选择画像。

## 19. 未通过确认的候选

{value['failed_confirmation']}。

## 20. 推荐进入下一订单预测阶段的画像

仅列出“强确认”或“部分确认”的晋级画像，且仍须独立审计：

{value['recommended']}

## 21. 被拒画像及原因

{value['rejected']}。

## 22. 阻断项

{value['blockers']}

## 23. 测试通过/失败

{value['tests']}。

## 24. 已创建文件

{value['files']}。精确哈希与行数见 `RUN_MANIFEST.json` 和 `ARTIFACT_VALIDATION_REPORT.md`。

## 25. 范围确认

本阶段没有运行最终违约/严重度订单级预测阶梯，也没有模拟业务政策。结果只验证独立历史画像，不证明因果关系、政策价值、生产可用性或画像在后续完整订单模型中的增量价值。
"""
    (output / "PROFILE_RESULTS_SUMMARY.md").write_text(en, encoding="utf-8")
    (output / "PROFILE_RESULTS_SUMMARY_ZH.md").write_text(zh, encoding="utf-8")


def write_blockers(output_dir: str | Path, blockers: Sequence[str]) -> None:
    output = Path(output_dir)
    unique = sorted({str(item).strip() for item in blockers if str(item).strip()})
    lines = ["# Blockers", ""]
    if unique:
        lines.extend(["The following completion blockers remain:", ""])
        lines.extend(f"- {item}" for item in unique)
    else:
        lines.append("No reporting-layer blocker was detected. A zero-candidate selection, if produced by the frozen rules, is a valid negative result rather than a blocker.")
    lines.extend(
        [
            "",
            "No final order-level breach/severity model or business policy was run by the reporting layer.",
            "",
        ]
    )
    (output / "BLOCKERS.md").write_text("\n".join(lines), encoding="utf-8")


def _protected_hashes_now() -> dict[str, dict[str, str]]:
    return {name: recursive_hashes(path) for name, path in sorted(PROTECTED.items())}


def _csv_rows(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path, usecols=[0], low_memory=False)))
    except (pd.errors.EmptyDataError, ValueError):
        return 0


def _parse_test_results(path: Path) -> dict[str, object]:
    """Parse the runner's persisted, non-self-hashed pytest receipt."""

    result: dict[str, object] = {
        "command": "",
        "return_code": None,
        "collected": None,
        "passed": None,
        "failed": None,
        "skipped": None,
        "deselected": None,
        "errors": None,
        "duration_seconds": None,
    }
    if not path.exists():
        return result
    prefixes = {
        "COMMAND": "command",
        "RETURN_CODE": "return_code",
        "COLLECTED": "collected",
        "PASSED": "passed",
        "FAILED": "failed",
        "SKIPPED": "skipped",
        "DESELECTED": "deselected",
        "ERRORS": "errors",
        "DURATION_SECONDS": "duration_seconds",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        label, separator, value = line.partition(":")
        if not separator or label not in prefixes:
            continue
        key = prefixes[label]
        text_value = value.strip()
        if key == "command":
            result[key] = text_value
        elif key == "duration_seconds":
            try:
                result[key] = float(text_value)
            except ValueError:
                result[key] = None
        else:
            try:
                result[key] = int(text_value)
            except ValueError:
                result[key] = None
    return result


def build_manifest(
    output_dir: str | Path,
    figure_records: Mapping[str, Mapping[str, object]],
    *,
    work_dir: str | Path | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    work = Path(work_dir) if work_dir is not None else output / "working"
    prestate_path = work / "PRE_EXECUTION_STATE.json"
    state_path = work / "RUN_STATE.json"
    prestate = json.loads(prestate_path.read_text(encoding="utf-8")) if prestate_path.exists() else {}
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    before = prestate.get("protected_hashes", {})
    after = _protected_hashes_now() if before else {}
    control_before = prestate.get("control_file_hashes", {})
    control_after = control_file_hashes() if control_before else {}
    inventory: dict[str, dict[str, object]] = {}
    # TEST_RESULTS is intentionally outside the hashed inventory: the final
    # full artifact-aware suite is allowed to overwrite its own receipt after
    # the manifest is written, without creating a test-log/manifest hash cycle.
    excluded = {
        "RUN_MANIFEST.json", "ARTIFACT_VALIDATION_REPORT.md", "TEST_RESULTS.txt"
    }
    for relative in sorted(set(REQUIRED_ARTIFACTS) - excluded):
        path = output / relative
        if path.exists():
            record: dict[str, object] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if path.suffix == ".csv":
                record["rows"] = _csv_rows(path)
            inventory[relative] = record
    for relative in (
        "PROFILE_SELECTION_FREEZE.sha256",
        "ANCHOR_SCHEDULE.csv",
        "FROZEN_TAIL_THRESHOLDS.json",
    ):
        path = output / relative
        if path.exists():
            record = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if path.suffix == ".csv":
                record["rows"] = _csv_rows(path)
            inventory[relative] = record
    confirmation_label_path = work / "CONFIRMATION_LABELS.csv"
    confirmation_label_audit: dict[str, object] = {
        "present": confirmation_label_path.exists(),
        "relative_path": "working/CONFIRMATION_LABELS.csv",
        "sha256": "",
        "rows": 0,
        "label_counts": {},
    }
    if confirmation_label_path.exists():
        labels = _read_csv(confirmation_label_path)
        confirmation_label_audit.update(
            {
                "sha256": sha256_file(confirmation_label_path),
                "rows": int(len(labels)),
                "label_counts": (
                    labels["confirmation_label"].astype(str).value_counts().sort_index().to_dict()
                    if "confirmation_label" in labels else {}
                ),
            }
        )
        inventory["working/CONFIRMATION_LABELS.csv"] = {
            "sha256": sha256_file(confirmation_label_path),
            "bytes": confirmation_label_path.stat().st_size,
            "rows": int(len(labels)),
        }
    raw_label_evidence: dict[str, dict[str, object]] = {}
    for name in (
        "CONFIRMATION_BY_MONTH_FOR_LABELS.csv",
        "CONFIRMATION_LABEL_AUDIT.json",
    ):
        path = work / name
        relative = f"working/{name}"
        if not path.exists():
            continue
        if path.suffix == ".csv":
            row_count = _csv_rows(path)
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row_count = len(payload) if isinstance(payload, list) else 1
            except (json.JSONDecodeError, OSError):
                row_count = 0
        record = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": int(row_count),
        }
        inventory[relative] = record
        raw_label_evidence[relative] = record
    rich_scoring_path = work / "SELECTED_ORDER_SCORING_RICH.csv.gz"
    if rich_scoring_path.exists():
        inventory["working/SELECTED_ORDER_SCORING_RICH.csv.gz"] = {
            "sha256": sha256_file(rich_scoring_path),
            "bytes": rich_scoring_path.stat().st_size,
            "rows": _csv_rows(rich_scoring_path),
        }
    selected_daily_paths = (
        work / "SELECTED_DAILY_PROFILE_ROWS.csv.gz",
        work / "SELECTED_DAILY_PROFILES.csv.gz",
        work / "PROFILE_DAILY_PROFILE_ROWS.csv.gz",
        work / "SELECTED_DAILY_PROFILE_ROWS.csv",
        work / "SELECTED_DAILY_PROFILES.csv",
    )
    selected_daily_path = next(
        (path for path in selected_daily_paths if path.exists()), None
    )
    if selected_daily_path is not None:
        relative = f"working/{selected_daily_path.name}"
        inventory[relative] = {
            "sha256": sha256_file(selected_daily_path),
            "bytes": selected_daily_path.stat().st_size,
            "rows": _csv_rows(selected_daily_path),
        }
    resource_audit_path = work / "REPORTING_RESOURCE_AUDIT.json"
    if resource_audit_path.exists():
        inventory["working/REPORTING_RESOURCE_AUDIT.json"] = {
            "sha256": sha256_file(resource_audit_path),
            "bytes": resource_audit_path.stat().st_size,
            "rows": 1,
        }
    gzip_path = output / "PROFILE_DAILY_SCORES.csv.gz"
    if gzip_path.exists():
        inventory[gzip_path.name] = {"sha256": sha256_file(gzip_path), "bytes": gzip_path.stat().st_size}
    for stem in FIGURE_STEMS:
        for relative in (f"figures/{stem}.png", f"figure_sources/{stem}.csv"):
            path = output / relative
            if path.exists():
                record = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                if path.suffix == ".csv":
                    record["rows"] = _csv_rows(path)
                inventory[relative] = record
    raw_events = list(state.get("stage_events", []))
    aliases = {
        "selection_freeze_written": "freeze_written",
        "selection_freeze_hashed": "freeze_hashed",
        "selection_freeze_recorded": "freeze_recorded",
    }
    events = [{**event, "event": aliases.get(str(event.get("event")), event.get("event"))} for event in raw_events]
    development_pids = [int(event["pid"]) for event in events if event.get("stage") in {"development", "selection"} and "pid" in event]
    confirmation_pids = [int(event["pid"]) for event in events if event.get("stage") == "confirmation" and "pid" in event]
    freeze_path = output / "PROFILE_SELECTION_FREEZE.json"
    sidecar = output / "PROFILE_SELECTION_FREEZE.sha256"
    freeze_sha = sha256_file(freeze_path) if freeze_path.exists() else ""
    freeze_payload: dict[str, Any] = {}
    if freeze_path.exists():
        try:
            freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            freeze_payload = {}
    development_hash_checks: dict[str, dict[str, object]] = {}
    for relative, expected in sorted(
        freeze_payload.get("development_artifact_hashes", {}).items()
    ):
        path = output / str(relative)
        actual = sha256_file(path) if path.exists() else ""
        development_hash_checks[str(relative)] = {
            "expected_sha256": str(expected),
            "actual_sha256": actual,
            "matches": bool(actual and actual == str(expected)),
        }
    selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
    tests = _parse_test_results(output / "TEST_RESULTS.txt")
    commands = list(state.get("commands", []))
    test_command = str(tests.get("command") or "")
    if test_command and test_command not in commands:
        commands.append(test_command)
    manifest: dict[str, object] = {
        "analysis_id": "dynamic_profile_profile_validation_v1",
        "artifact_inventory": inventory,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_figure_stems": list(FIGURE_STEMS),
        "figures": dict(figure_records),
        "figure_applicability": _figure_applicability(output),
        "commands": commands,
        "tests": tests,
        "environment": prestate.get("environment", {}),
        "repository": prestate.get("repository", {}),
        "raw_file_hashes": prestate.get("raw_file_hashes", {}),
        "v1_1_file_hashes": prestate.get("v1_1_file_hashes", {}),
        "assembler_sha256": prestate.get("assembler_sha256", ""),
        "config_sha256": sha256_file(output / "PROFILE_FROZEN_CONFIG.json") if (output / "PROFILE_FROZEN_CONFIG.json").exists() else "",
        "protocol_sha256": sha256_file(output / "PROFILE_PROTOCOL.md") if (output / "PROFILE_PROTOCOL.md").exists() else "",
        "selection_freeze_sha256": freeze_sha,
        "selection_freeze_sidecar_value": sidecar.read_text(encoding="utf-8").strip().split()[0] if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() else "",
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_hashes_after_tests": after,
        "control_file_hashes_before": control_before,
        "control_file_hashes_after": control_after,
        "control_file_hashes_after_tests": control_after,
        "stage_gate": {
            "events": events,
            "development_pid": development_pids[-1] if development_pids else None,
            "confirmation_pid": confirmation_pids[0] if confirmation_pids else None,
            "fresh_process_required": True,
            "development_artifact_hash_checks": development_hash_checks,
            "development_artifact_hashes_match": bool(development_hash_checks)
            and all(bool(record["matches"]) for record in development_hash_checks.values()),
        },
        "scope_flags": {
            "final_order_model_fitted": False,
            "business_policy_simulated": False,
            "thesis_modified": False,
            "phase2a_reinterpreted": False,
        },
        "selection_outcome": {
            "promoted_candidate_count": int(len(selected)),
            "zero_candidates_is_valid_negative_result": True,
        },
        "confirmation_label_audit": confirmation_label_audit,
        "confirmation_label_raw_evidence": raw_label_evidence,
        "determinism": {
            "float_format": FLOAT_FORMAT,
            "date_format": DATE_FORMAT,
            "na_rep": NA_REP,
            "gzip_mtime": 0,
            "figure_renderer_reads_paired_csv": True,
        },
    }
    return manifest


def validate_artifacts(
    output_dir: str | Path,
    *,
    write_report: bool = True,
) -> dict[str, object]:
    """Validate required artifacts, schemas, keys, hashes, figures and guards."""

    output = Path(output_dir)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    artifact_rows: list[dict[str, object]] = []
    for name in REQUIRED_ARTIFACTS:
        path = output / name
        is_self = name == "ARTIFACT_VALIDATION_REPORT.md"
        exists = path.is_file() or (write_report and is_self)
        check(f"required_artifact:{name}", exists, "self-written after validation" if is_self and not path.exists() else "")
        artifact_rows.append(
            {
                "relative_path": name,
                "exists": exists,
                "sha256": sha256_file(path) if path.is_file() and not is_self else "self_hash_intentionally_omitted",
                "bytes": path.stat().st_size if path.is_file() else 0,
                "rows": _csv_rows(path) if path.is_file() and path.suffix == ".csv" else "",
            }
        )

    for name, schema in CSV_SCHEMAS.items():
        path = output / name
        if not path.exists():
            continue
        actual = tuple(_read_csv(path).columns)
        if name in FROZEN_WIDE_SELECTION_TABLES:
            required_columns = set(PRIMARY_KEYS[name]) | {
                "base_candidate_id", "target", "granularity", "scheme",
                "window_days", "lag_days", "estimator", "parent_structure",
            }
            support_present = "support_threshold" in actual or "min_support" in actual
            schema_ok = required_columns.issubset(actual) and support_present
            detail = (
                "persisted wide schema is authoritative under freeze; "
                f"missing_required={sorted(required_columns - set(actual))}; "
                f"support_rule_present={support_present}; actual={actual}"
            )
        else:
            schema_ok = actual == schema
            detail = f"expected={schema}; actual={actual}"
        check(f"schema:{name}", schema_ok, detail)
        frame = _read_csv(path)
        key = list(PRIMARY_KEYS[name])
        missing_key = [column for column in key if column not in frame]
        duplicated = bool(frame.duplicated(key).any()) if not missing_key and not frame.empty else False
        null_key = bool(frame[key].isna().any().any()) if not missing_key and not frame.empty else False
        check(
            f"primary_key:{name}",
            not missing_key and not duplicated and not null_key,
            f"missing={missing_key}; duplicated={duplicated}; null={null_key}",
        )

    gzip_path = output / "PROFILE_DAILY_SCORES.csv.gz"
    gzip_ok = False
    validated_daily = pd.DataFrame()
    if gzip_path.exists():
        header = gzip_path.read_bytes()[:10]
        mtime = int.from_bytes(header[4:8], byteorder="little", signed=False) if len(header) >= 8 else -1
        try:
            validated_daily = _read_csv(gzip_path, dtype=DAILY_PROFILE_DTYPES)
            schema_ok = tuple(validated_daily.columns) == DAILY_ROW_SCHEMA
            key = ["candidate_id", "snapshot_date", "entity_id"]
            key_ok = validated_daily.empty or (
                not validated_daily[key].isna().any().any()
                and not validated_daily.duplicated(key).any()
            )
            gzip_ok = header[:2] == b"\x1f\x8b" and mtime == 0 and schema_ok and key_ok
            check(
                "daily_gzip_schema", schema_ok,
                str(tuple(validated_daily.columns)),
            )
            check("daily_gzip_primary_key", key_ok)
        except (OSError, EOFError, ValueError, KeyError) as exc:
            check("daily_gzip_readable", False, str(exc))
        check("daily_gzip_fixed_mtime", mtime == 0, f"mtime={mtime}")
    check("daily_gzip_present_and_valid", gzip_ok)
    validated_daily_rows = int(len(validated_daily))
    validated_daily_ids = (
        set(validated_daily["candidate_id"].dropna().astype(str))
        if "candidate_id" in validated_daily else set()
    )
    del validated_daily

    expected_png = {f"{stem}.png" for stem in FIGURE_STEMS}
    expected_csv = {f"{stem}.csv" for stem in FIGURE_STEMS}
    actual_png = {path.name for path in (output / "figures").glob("*.png")} if (output / "figures").exists() else set()
    actual_csv = {path.name for path in (output / "figure_sources").glob("*.csv")} if (output / "figure_sources").exists() else set()
    check("exact_18_figure_pngs", actual_png == expected_png, f"missing={sorted(expected_png - actual_png)}; extra={sorted(actual_png - expected_png)}")
    check("exact_18_figure_sources", actual_csv == expected_csv, f"missing={sorted(expected_csv - actual_csv)}; extra={sorted(actual_csv - expected_csv)}")
    for stem in FIGURE_STEMS:
        source = output / "figure_sources" / f"{stem}.csv"
        figure = output / "figures" / f"{stem}.png"
        source_ok = source.exists() and tuple(_read_csv(source).columns) == FIGURE_SOURCE_SCHEMA and len(_read_csv(source)) >= 1
        figure_ok = figure.exists() and figure.stat().st_size > 0
        check(f"figure_pair:{stem}", source_ok and figure_ok, f"source={source_ok}; png={figure_ok}")

    blockers_text = (
        (output / "BLOCKERS.md").read_text(encoding="utf-8")
        if (output / "BLOCKERS.md").exists() else ""
    )
    active_blockers = "The following completion blockers remain:" in blockers_text
    check("no_active_completion_blockers", not active_blockers)

    selected_table = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
    selected_ids = (
        set(selected_table["candidate_id"].dropna().astype(str))
        if "candidate_id" in selected_table else set()
    )
    rich_scoring_path = output / "working" / "SELECTED_ORDER_SCORING_RICH.csv.gz"
    rich_scoring_table = _read_csv(
        rich_scoring_path, dtype=RICH_SCORING_DTYPES
    )
    rich_key = [
        "candidate_id", "period", "anchor_date", "horizon_days", "order_id"
    ]
    rich_schema_ok = tuple(rich_scoring_table.columns) == RICH_ORDER_SCORING_SCHEMA
    rich_key_ok = bool(
        rich_schema_ok
        and (
            rich_scoring_table.empty
            or (
                not rich_scoring_table[rich_key].isna().any().any()
                and not rich_scoring_table.duplicated(rich_key).any()
            )
        )
    )
    rich_mtime = -1
    if rich_scoring_path.exists():
        header = rich_scoring_path.read_bytes()[:10]
        rich_mtime = (
            int.from_bytes(header[4:8], byteorder="little", signed=False)
            if len(header) >= 8 else -1
        )
    check(
        "rich_order_scoring_schema_key_and_gzip",
        rich_scoring_path.exists() and rich_schema_ok and rich_key_ok and rich_mtime == 0,
        f"present={rich_scoring_path.exists()}; schema={rich_schema_ok}; key={rich_key_ok}; mtime={rich_mtime}",
    )
    resource_audit_path = output / "working" / "REPORTING_RESOURCE_AUDIT.json"
    resource_audit_ok = False
    selected_daily_resource_ok = False
    if resource_audit_path.exists():
        try:
            resource_audit = json.loads(
                resource_audit_path.read_text(encoding="utf-8")
            )
            rich_receipt = resource_audit.get("rich_scoring", {})
            resource_audit_ok = bool(
                resource_audit.get("allowed") is True
                and int(resource_audit.get("row_count", -1)) == len(rich_scoring_table)
                and resource_audit.get("sha256") == sha256_file(rich_scoring_path)
                and rich_receipt.get("schema_ok") is True
            )
            daily_receipt = resource_audit.get("selected_daily_profiles", {})
            daily_relative = str(daily_receipt.get("relative_path", ""))
            daily_work_path = output / daily_relative if daily_relative else None
            if daily_receipt.get("present") is True and daily_work_path is not None:
                selected_daily_resource_ok = bool(
                    daily_work_path.exists()
                    and daily_receipt.get("schema_ok") is True
                    and daily_receipt.get("sha256") == sha256_file(daily_work_path)
                    and int(daily_receipt.get("row_count", -1))
                    == _csv_rows(daily_work_path)
                )
            else:
                selected_daily_resource_ok = not selected_ids
        except (json.JSONDecodeError, OSError, ValueError, FileNotFoundError):
            resource_audit_ok = False
            selected_daily_resource_ok = False
    check("rich_order_scoring_resource_audit", resource_audit_ok)
    check("selected_daily_resource_audit", selected_daily_resource_ok)
    if selected_ids:
        daily_index = _read_csv(output / "PROFILE_DAILY_SCORES.csv")
        scoring_table = _read_csv(
            output / "PROFILE_FUTURE_ORDER_SCORING.csv",
            usecols=("candidate_id",),
            dtype={"candidate_id": "category"},
        )
        labels_table = _read_csv(output / "working" / "CONFIRMATION_LABELS.csv")
        daily_ids = validated_daily_ids
        scoring_ids = (
            set(scoring_table["candidate_id"].dropna().astype(str))
            if "candidate_id" in scoring_table else set()
        )
        label_ids = (
            set(labels_table["candidate_id"].dropna().astype(str))
            if "candidate_id" in labels_table else set()
        )
        rich_scoring_ids = (
            set(rich_scoring_table["candidate_id"].dropna().astype(str))
            if "candidate_id" in rich_scoring_table else set()
        )
        rich_key_ok = bool(rich_key_ok and not rich_scoring_table.empty)
        check(
            "promoted_daily_profiles_nonempty",
            not daily_index.empty and validated_daily_rows > 0 and selected_ids <= daily_ids,
            f"index_rows={len(daily_index)}; daily_rows={validated_daily_rows}; missing_ids={sorted(selected_ids - daily_ids)}",
        )
        check(
            "promoted_order_scoring_nonempty",
            not scoring_table.empty and selected_ids <= scoring_ids,
            f"rows={len(scoring_table)}; missing_ids={sorted(selected_ids - scoring_ids)}",
        )
        check(
            "promoted_rich_order_scoring_nonempty",
            rich_key_ok and selected_ids <= rich_scoring_ids,
            f"rows={len(rich_scoring_table)}; schema={rich_schema_ok}; key={rich_key_ok}; missing_ids={sorted(selected_ids - rich_scoring_ids)}",
        )
        check(
            "promoted_confirmation_labels_complete",
            len(labels_table) == len(selected_ids) and label_ids == selected_ids,
            f"rows={len(labels_table)}; expected={len(selected_ids)}; ids_match={label_ids == selected_ids}",
        )
        selected_metric_checks = {
            "development": "PROFILE_DEVELOPMENT_RESULTS.csv",
            "confirmation": "PROFILE_CONFIRMATION_RESULTS.csv",
            "terminal": "PROFILE_TERMINAL_STRESS.csv",
        }
        for period_name, filename in selected_metric_checks.items():
            frame = _read_csv(output / filename)
            if period_name == "confirmation" and not frame.empty:
                frame = frame.loc[
                    ~frame["stratum_type"].astype(str).eq("confirmation_label")
                ]
            valid_frame = _valid_rows(frame)
            metric_ids = (
                set(frame["candidate_id"].dropna().astype(str))
                if "candidate_id" in frame else set()
            )
            invalid_receipts_complete = bool(
                not frame.empty
                and (
                    _coerce_bool(frame["valid"])
                    | frame["invalid_reason"].fillna("").astype(str).str.len().gt(0)
                ).all()
            ) if {"valid", "invalid_reason"}.issubset(frame.columns) else False
            if period_name == "confirmation":
                evidence_ok = (
                    not frame.empty
                    and selected_ids <= metric_ids
                    and invalid_receipts_complete
                )
            else:
                evidence_ok = not valid_frame.empty and selected_ids <= metric_ids
            check(
                f"promoted_{period_name}_metrics_nonempty",
                evidence_ok,
                f"rows={len(frame)}; valid_rows={len(valid_frame)}; missing_ids={sorted(selected_ids - metric_ids)}; invalid_receipts_complete={invalid_receipts_complete}",
            )
        key_analytical_tables = (
            "PROFILE_CONSTRUCTION_AUDIT.csv",
            "PROFILE_SUPPORT_UNCERTAINTY.csv",
            "PROFILE_PARENT_STRUCTURE.csv",
            "PROFILE_CONFIRMATION_BY_MONTH.csv",
            "PROFILE_LEVEL_RESULTS.csv",
            "PROFILE_LEVEL_TRANSITIONS.csv",
            "PROFILE_DAILY_STABILITY.csv",
            "PROFILE_FUTURE_ENTITY_TRANSFER.csv",
            "PROFILE_SUPPORT_STRATA.csv",
            "PROFILE_COLD_START_RESULTS.csv",
            "PROFILE_HRD_DIAGNOSTICS.csv",
            "PROFILE_ABLATIONS.csv",
        )
        for filename in key_analytical_tables:
            raw_rows = _read_csv(output / filename)
            valid_rows = _valid_rows(raw_rows)
            explicit_invalid_rows = bool(
                not raw_rows.empty
                and {"valid", "invalid_reason"}.issubset(raw_rows.columns)
                and (
                    _coerce_bool(raw_rows["valid"])
                    | raw_rows["invalid_reason"].fillna("").astype(str).str.len().gt(0)
                ).all()
            )
            invalid_evidence_allowed = filename in {
                "PROFILE_DAILY_STABILITY.csv",
                "PROFILE_FUTURE_ENTITY_TRANSFER.csv",
            }
            check(
                f"promoted_key_analysis_nonempty:{filename}",
                not valid_rows.empty or (invalid_evidence_allowed and explicit_invalid_rows),
                f"rows={len(raw_rows)}; valid_rows={len(valid_rows)}; explicit_invalid={explicit_invalid_rows}",
            )
        applicability_receipt = _figure_applicability(output)
        for stem in FIGURE_STEMS:
            source = _read_csv(output / "figure_sources" / f"{stem}.csv")
            source_has_data = bool(
                not source.empty
                and _coerce_bool(source["valid"]).any()
                and np.isfinite(pd.to_numeric(source["y_value"], errors="coerce")).any()
            ) if {"valid", "y_value"}.issubset(source.columns) else False
            applicable = bool(applicability_receipt[stem]["applicable"])
            explicit_not_applicable = bool(
                not source.empty
                and source["invalid_reason"].astype(str).str.startswith(
                    "not_applicable:"
                ).all()
            ) if "invalid_reason" in source else False
            explicit_invalid_evidence = bool(
                not source.empty
                and source["invalid_reason"].astype(str).str.startswith(
                    "evaluation_invalid:"
                ).all()
            ) if "invalid_reason" in source else False
            allowed_invalid_evidence = stem in {
                "09_confirmation_future_rank_transfer",
                "12_daily_profile_stability",
            }
            check(
                f"promoted_figure_source_applicability:{stem}",
                (
                    source_has_data
                    or (allowed_invalid_evidence and explicit_invalid_evidence)
                    if applicable else explicit_not_applicable
                ),
                f"applicable={applicable}; substantive={source_has_data}; explicit_invalid={explicit_invalid_evidence}; explicit_not_applicable={explicit_not_applicable}; reason={applicability_receipt[stem]['reason']}",
            )

    manifest_path = output / "RUN_MANIFEST.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            check("manifest_json", True)
        except json.JSONDecodeError as exc:
            check("manifest_json", False, str(exc))
    else:
        check("manifest_json", False, "missing")
    if manifest:
        for relative, record in manifest.get("artifact_inventory", {}).items():
            path = output / relative
            hash_ok = path.exists() and sha256_file(path) == record.get("sha256")
            rows_ok = True
            if (
                path.exists()
                and (path.suffix == ".csv" or path.name.endswith(".csv.gz"))
                and "rows" in record
            ):
                rows_ok = _csv_rows(path) == int(record["rows"])
            elif path.exists() and path.suffix == ".json" and "rows" in record:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    actual_rows = len(payload) if isinstance(payload, list) else 1
                    rows_ok = actual_rows == int(record["rows"])
                except (json.JSONDecodeError, OSError, ValueError):
                    rows_ok = False
            check(f"manifest_inventory:{relative}", hash_ok and rows_ok, f"hash={hash_ok}; rows={rows_ok}")
        check(
            "test_log_excluded_from_manifest_inventory",
            "TEST_RESULTS.txt" not in manifest.get("artifact_inventory", {}),
        )
        figures = manifest.get("figures", {})
        check("manifest_exact_figure_set", set(figures) == expected_png)
        check(
            "manifest_figure_applicability_receipt",
            manifest.get("figure_applicability") == _figure_applicability(output),
        )
        for name, record in figures.items():
            figure = output / str(record.get("relative_path", f"figures/{name}"))
            source = output / str(record.get("source_relative_path", ""))
            ok = (
                figure.exists() and source.exists()
                and sha256_file(figure) == record.get("sha256")
                and sha256_file(source) == record.get("source_sha256")
                and _csv_rows(source) == int(record.get("source_rows", -1))
            )
            check(f"manifest_figure:{name}", ok)
        before = manifest.get("protected_hashes_before", {})
        after = manifest.get("protected_hashes_after", {})
        after_tests = manifest.get("protected_hashes_after_tests", {})
        check("protected_hashes_unchanged", bool(before) and before == after == after_tests)
        control_before = manifest.get("control_file_hashes_before", {})
        control_after = manifest.get("control_file_hashes_after", {})
        control_after_tests = manifest.get("control_file_hashes_after_tests", {})
        check(
            "control_file_hashes_unchanged",
            bool(control_before)
            and control_before == control_after == control_after_tests,
        )
        stage_gate = manifest.get("stage_gate", {})
        hash_checks = stage_gate.get("development_artifact_hash_checks", {})
        check(
            "freeze_embedded_development_hashes_revalidated",
            bool(hash_checks)
            and stage_gate.get("development_artifact_hashes_match") is True
            and all(bool(record.get("matches")) for record in hash_checks.values()),
        )
        event_names = [str(event.get("event")) for event in stage_gate.get("events", [])]
        required_events = [
            "freeze_written", "freeze_hashed", "freeze_recorded",
            "confirmation_labels_opened",
        ]
        ordered_events = all(name in event_names for name in required_events)
        if ordered_events:
            positions = [event_names.index(name) for name in required_events]
            ordered_events = positions == sorted(positions)
        check("stage_gate_event_order", ordered_events)
        development_pid = stage_gate.get("development_pid")
        confirmation_pid = stage_gate.get("confirmation_pid")
        check(
            "fresh_confirmation_process",
            development_pid is not None
            and confirmation_pid is not None
            and development_pid != confirmation_pid,
        )
        flags = manifest.get("scope_flags", {})
        scope_ok = all(
            flags.get(name) is False
            for name in (
                "final_order_model_fitted", "business_policy_simulated",
                "thesis_modified", "phase2a_reinterpreted",
            )
        )
        check("scope_flags", scope_ok)
        check(
            "zero_candidates_is_valid_negative_result",
            manifest.get("selection_outcome", {}).get("zero_candidates_is_valid_negative_result") is True,
        )
        confirmation_audit = manifest.get("confirmation_label_audit", {})
        promoted_count = int(
            manifest.get("selection_outcome", {}).get("promoted_candidate_count", 0)
        )
        audit_path = output / str(
            confirmation_audit.get(
                "relative_path", "working/CONFIRMATION_LABELS.csv"
            )
        )
        audit_ok = (
            confirmation_audit.get("present") is True
            and audit_path.exists()
            and sha256_file(audit_path) == confirmation_audit.get("sha256")
            and _csv_rows(audit_path) == int(confirmation_audit.get("rows", -1))
            and int(confirmation_audit.get("rows", -1)) == promoted_count
        )
        check("confirmation_label_audit", audit_ok)
        evidence = manifest.get("confirmation_label_raw_evidence", {})
        expected_evidence = {
            "working/CONFIRMATION_BY_MONTH_FOR_LABELS.csv",
            "working/CONFIRMATION_LABEL_AUDIT.json",
        }
        evidence_ok = set(evidence) == expected_evidence
        for relative in sorted(expected_evidence):
            record = evidence.get(relative, {})
            path = output / relative
            hash_ok = path.exists() and sha256_file(path) == record.get("sha256")
            if path.suffix == ".csv":
                rows_ok = path.exists() and _csv_rows(path) == int(record.get("rows", -1))
            else:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    actual_rows = len(payload) if isinstance(payload, list) else 1
                    rows_ok = actual_rows == int(record.get("rows", -1))
                except (json.JSONDecodeError, OSError, ValueError):
                    rows_ok = False
            evidence_ok = evidence_ok and hash_ok and rows_ok
            check(
                f"confirmation_label_raw_evidence:{relative}",
                hash_ok and rows_ok,
                f"hash={hash_ok}; rows={rows_ok}",
            )
        audit_json_path = output / "working" / "CONFIRMATION_LABEL_AUDIT.json"
        cross_hash_ok = False
        if audit_json_path.exists():
            try:
                label_audit_payload = json.loads(
                    audit_json_path.read_text(encoding="utf-8")
                )
                cross_hash_ok = (
                    label_audit_payload.get("confirmation_month_input_sha256")
                    == sha256_file(
                        output / "working" / "CONFIRMATION_BY_MONTH_FOR_LABELS.csv"
                    )
                    and label_audit_payload.get("confirmation_labels_sha256")
                    == sha256_file(output / "working" / "CONFIRMATION_LABELS.csv")
                    and int(label_audit_payload.get("candidate_count", -1))
                    == promoted_count
                )
            except (json.JSONDecodeError, OSError, ValueError, FileNotFoundError):
                cross_hash_ok = False
        check(
            "confirmation_label_raw_evidence_complete",
            evidence_ok and cross_hash_ok,
            f"inventory={evidence_ok}; cross_hashes={cross_hash_ok}",
        )
        manifest_tests = manifest.get("tests", {})
        manifest_test_ok = (
            manifest_tests.get("return_code") == 0
            and manifest_tests.get("collected") == 111
            and manifest_tests.get("passed") == 111
            and manifest_tests.get("failed") == 0
            and manifest_tests.get("skipped") == 0
            and manifest_tests.get("deselected") == 0
            and manifest_tests.get("errors") == 0
            and "-B" in str(manifest_tests.get("command", ""))
            and "no:cacheprovider" in str(manifest_tests.get("command", ""))
        )
        check("manifest_exact_full_test_receipt", manifest_test_ok)

    freeze = output / "PROFILE_SELECTION_FREEZE.json"
    sidecar = output / "PROFILE_SELECTION_FREEZE.sha256"
    sidecar_value = sidecar.read_text(encoding="utf-8").strip().split()[0] if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() else ""
    check("selection_freeze_sidecar", freeze.exists() and bool(sidecar_value) and sha256_file(freeze) == sidecar_value)

    tests = output / "TEST_RESULTS.txt"
    test_receipt = _parse_test_results(tests)
    test_ok = (
        test_receipt.get("return_code") == 0
        and test_receipt.get("collected") == 111
        and test_receipt.get("passed") == 111
        and test_receipt.get("failed") == 0
        and test_receipt.get("skipped") == 0
        and test_receipt.get("deselected") == 0
        and test_receipt.get("errors") == 0
        and "-B" in str(test_receipt.get("command", ""))
        and "no:cacheprovider" in str(test_receipt.get("command", ""))
    )
    check("required_test_log", test_ok, str(test_receipt))

    overall = bool(all(record["passed"] for record in checks))
    report = {
        "overall_pass": overall,
        "checks_passed": int(sum(bool(record["passed"]) for record in checks)),
        "checks_failed": int(sum(not bool(record["passed"]) for record in checks)),
        "checks": checks,
        "artifacts": artifact_rows,
    }
    if write_report:
        lines = [
            "# Artifact Validation Report",
            "",
            f"Overall verdict: **{'PASS' if overall else 'FAIL'}**",
            "",
            f"Checks passed: {report['checks_passed']}; checks failed: {report['checks_failed']}.",
            "",
            "The report validates the 30 required artifacts, the deterministic daily gzip artifact, exact schemas and primary keys, all 18 figure/source pairs, manifest hashes, the freeze sidecar, protected trees, scope flags, and the persisted test log. This report omits its own SHA-256 to avoid a self-referential hash.",
            "",
            "## Artifact inventory",
            "",
            "| Path | Exists | Rows | Bytes | SHA-256 |",
            "|---|---:|---:|---:|---|",
        ]
        for row in artifact_rows:
            lines.append(f"| `{row['relative_path']}` | {row['exists']} | {row['rows']} | {row['bytes']} | `{row['sha256']}` |")
        lines.extend(["", "## Checks", "", "| Check | Pass | Detail |", "|---|---:|---|"])
        for record in checks:
            detail = str(record["detail"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{record['check']}` | {record['passed']} | {detail} |")
        lines.append("")
        (output / "ARTIFACT_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _wasserstein_distance_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Exact unweighted empirical 1-D Wasserstein distance."""

    x = np.sort(np.asarray(left, dtype=float))
    y = np.sort(np.asarray(right, dtype=float))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if not len(x) or not len(y):
        return np.nan
    support = np.sort(np.concatenate([x, y]))
    if len(support) < 2:
        return 0.0
    deltas = np.diff(support)
    x_cdf = np.searchsorted(x, support[:-1], side="right") / len(x)
    y_cdf = np.searchsorted(y, support[:-1], side="right") / len(y)
    return float(np.sum(np.abs(x_cdf - y_cdf) * deltas))


def aggregate_terminal_score_shift(
    scoring: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Describe locked-confirmation-to-terminal score-distribution shift.

    Persisted future-order rows are intentionally retained as repeated
    descriptive exposure.  These metrics are not treated as independent-order
    inference and therefore carry no iid confidence intervals.
    """

    if scoring.empty:
        return _empty(TERMINAL_SCHEMA)
    # Keep the complete all-placed future cohort for maturity/follow-up.  Score
    # distribution metrics have a narrower, explicitly mapped-and-finite
    # analysis subset, but that subset must never become the maturity
    # denominator.
    full = scoring.copy()
    full["profile_score"] = pd.to_numeric(full["profile_score"], errors="coerce")
    full = full.loc[
        full["period"].astype(str).isin(["confirmation", "terminal"])
    ].copy()
    if full.empty:
        return _empty(TERMINAL_SCHEMA)
    endpoint = pd.Timestamp(config.get("audit_endpoint_proxy", "2018-10-17 13:22:46"))
    maturity_threshold = float(
        config.get("terminal_maturity_availability_threshold", 0.95)
    )
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id", "base_candidate_id", "target", "granularity",
        "horizon_days",
    ]
    metric_names = {
        "p10_difference": "profile_score_p10_difference",
        "median_difference": "profile_score_median_difference",
        "p90_difference": "profile_score_p90_difference",
        "wasserstein_distance": "profile_score_wasserstein_distance",
    }
    configured_metrics = config.get("terminal_score_distribution_shift", {}).get(
        "metrics", tuple(metric_names)
    )
    for values, group in full.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        score_group = group.loc[np.isfinite(group["profile_score"])].copy()
        if "mapping_status" in score_group:
            score_group = score_group.loc[
                score_group["mapping_status"].astype(str).ne("missing_mapping")
            ].copy()
        confirmation = pd.to_numeric(
            score_group.loc[
                score_group["period"].astype(str).eq("confirmation"),
                "profile_score",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        terminal = pd.to_numeric(
            score_group.loc[
                score_group["period"].astype(str).eq("terminal"),
                "profile_score",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        # This is the all-placed terminal future cohort: it intentionally
        # includes missing mappings, non-finite scores, and target-invalid rows.
        terminal_rows = group.loc[group["period"].astype(str).eq("terminal")].copy()
        terminal_score_rows = score_group.loc[
            score_group["period"].astype(str).eq("terminal")
        ].copy()
        if not len(confirmation) or not len(terminal):
            estimates = {name: np.nan for name in metric_names}
        else:
            estimates = {
                "p10_difference": float(
                    np.quantile(terminal, 0.10) - np.quantile(confirmation, 0.10)
                ),
                "median_difference": float(
                    np.median(terminal) - np.median(confirmation)
                ),
                "p90_difference": float(
                    np.quantile(terminal, 0.90) - np.quantile(confirmation, 0.90)
                ),
                "wasserstein_distance": _wasserstein_distance_1d(
                    terminal, confirmation
                ),
            }
        label_available = pd.to_datetime(
            _series(terminal_rows, "label_available_at"), errors="coerce"
        )
        maturity_rate = float(
            (label_available.notna() & label_available.le(endpoint)).mean()
        ) if len(terminal_rows) else np.nan
        anchors = pd.to_datetime(
            _series(terminal_rows, "anchor_date"), errors="coerce"
        )
        followup = (
            endpoint
            - (
                anchors
                + pd.to_timedelta(
                    pd.to_numeric(_series(terminal_rows, "horizon_days"), errors="coerce"),
                    unit="D",
                )
            )
        ).dt.total_seconds() / 86400.0
        for configured in configured_metrics:
            configured = str(configured)
            if configured not in metric_names:
                raise ValueError(
                    f"unknown frozen terminal score-shift metric: {configured}"
                )
            estimate = estimates[configured]
            valid = bool(np.isfinite(estimate))
            rows.append(
                {
                    **identity,
                    "period": "terminal",
                    "anchor_date": np.nan,
                    "calendar_month": "",
                    "stratum_type": "profile_score_distribution_shift",
                    "stratum_value": "terminal_minus_locked_confirmation",
                    "metric_name": metric_names[configured],
                    "reference_id": "locked_confirmation",
                    "aggregation": "future_order_rows_descriptive_repeated_exposure",
                    "n_scheduled_anchors": int(terminal_rows["anchor_date"].nunique()),
                    "n_valid_anchors": int(terminal_score_rows["anchor_date"].nunique()) if valid else 0,
                    "n_orders": int(len(terminal)),
                    "n_events": np.nan,
                    "n_entities": int(terminal_score_rows["entity_id"].dropna().nunique()),
                    "n_common_entities": 0,
                    "estimate": estimate,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "outcome_observation_rate": maturity_rate,
                    "followup_available_days": float(followup.median()) if np.isfinite(followup).any() else np.nan,
                    "maturity_censoring_flag": bool(
                        np.isfinite(maturity_rate) and maturity_rate < maturity_threshold
                    ),
                    "distribution_shift_reference": "locked_confirmation_future_order_rows_descriptive_repeated_exposure",
                    "valid": valid,
                    "invalid_reason": "" if valid else "confirmation_or_terminal_profile_scores_missing",
                }
            )
    return _sort_frame(
        _normalise_schema(pd.DataFrame(rows), TERMINAL_SCHEMA),
        PRIMARY_KEYS["PROFILE_TERMINAL_STRESS.csv"],
    )


def _augment_terminal_followup(
    anchor_metrics: pd.DataFrame,
    scoring: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    if anchor_metrics.empty:
        return anchor_metrics
    result = anchor_metrics.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["anchor_date"] = pd.to_datetime(result["anchor_date"], errors="coerce")
    result["horizon_days"] = pd.to_numeric(
        result["horizon_days"], errors="coerce"
    ).astype("Int64")
    if "period" not in result:
        result["period"] = result["anchor_date"].map(_period_for_date)
    if scoring.empty:
        return result
    terminal = scoring.loc[scoring["period"].astype(str).eq("terminal")].copy()
    if terminal.empty:
        return result
    terminal["candidate_id"] = terminal["candidate_id"].astype(str)
    terminal["anchor_date"] = pd.to_datetime(
        terminal["anchor_date"], errors="coerce"
    )
    terminal["horizon_days"] = pd.to_numeric(
        terminal["horizon_days"], errors="coerce"
    ).astype("Int64")
    endpoint = pd.Timestamp(config.get("audit_endpoint_proxy", "2018-10-17 13:22:46"))
    # Maturity is unconditional label availability among every all-placed row
    # in the terminal future cohort.  ``target_observed`` is target validity
    # (for example positive-severity eligibility) and must not be substituted
    # for this censoring audit.
    terminal["_label_available_at"] = pd.to_datetime(
        terminal["label_available_at"], errors="coerce"
    )
    terminal["_label_mature_by_proxy"] = (
        terminal["_label_available_at"].notna()
        & terminal["_label_available_at"].le(endpoint)
    )
    audit = terminal.groupby(
        ["candidate_id", "anchor_date", "horizon_days"], sort=True,
        observed=True,
    ).agg(
        outcome_observation_rate=(
            "_label_mature_by_proxy", lambda values: float(pd.Series(values).mean())
        ),
    ).reset_index()
    audit["followup_available_days"] = (
        endpoint - (pd.to_datetime(audit["anchor_date"]) + pd.to_timedelta(pd.to_numeric(audit["horizon_days"]), unit="D"))
    ).dt.total_seconds() / 86400.0
    maturity_threshold = float(
        config.get("terminal_maturity_availability_threshold", 0.95)
    )
    audit["maturity_censoring_flag"] = audit["outcome_observation_rate"].lt(
        maturity_threshold
    )
    keys = ["candidate_id", "anchor_date", "horizon_days"]
    result = result.merge(audit, on=keys, how="left", validate="many_to_one")
    return result


def _frozen_development_paths(output: Path) -> set[str]:
    freeze_path = output / "PROFILE_SELECTION_FREEZE.json"
    if not freeze_path.exists():
        return set()
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {str(path) for path in payload.get("development_artifact_hashes", {})}


def verify_frozen_development_artifacts(output_dir: str | Path) -> dict[str, str]:
    """Hard-stop unless every selection-freeze source still matches byte-for-byte."""

    output = Path(output_dir)
    freeze_path = output / "PROFILE_SELECTION_FREEZE.json"
    sidecar_path = output / "PROFILE_SELECTION_FREEZE.sha256"
    if not freeze_path.exists() or not sidecar_path.exists():
        raise RuntimeError("reporting requires selection freeze and SHA sidecar")
    calculated = sha256_file(freeze_path)
    sidecar_text = sidecar_path.read_text(encoding="utf-8").strip()
    sidecar = sidecar_text.split()[0] if sidecar_text else ""
    if not sidecar or calculated != sidecar:
        raise RuntimeError("selection freeze SHA sidecar mismatch before reporting")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    hashes = {
        str(relative): str(expected)
        for relative, expected in payload.get("development_artifact_hashes", {}).items()
    }
    if not hashes:
        raise RuntimeError("selection freeze lacks embedded development artifact hashes")
    mismatches: list[str] = []
    for relative, expected in sorted(hashes.items()):
        path = output / relative
        if not path.exists():
            mismatches.append(f"missing:{relative}")
        elif sha256_file(path) != expected:
            mismatches.append(f"hash_mismatch:{relative}")
    if mismatches:
        raise RuntimeError(
            "frozen development artifacts changed before reporting: "
            + ";".join(mismatches)
        )
    return hashes


def audit_rich_scoring_resources(
    output_dir: str | Path,
    work_dir: str | Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed before pandas loads the two potentially largest tables.

    Both tables are counted, schema-checked and hashed without constructing a
    DataFrame.  The allowance reflects the explicit typed loaders and is based
    on their *combined* estimated peak, rather than candidate count.
    """

    def table_receipt(path: Path | None, expected: Sequence[str]) -> dict[str, object]:
        if path is None or not path.exists():
            return {
                "relative_path": "", "present": False, "row_count": 0,
                "sha256": "", "compressed_bytes": 0, "schema_ok": False,
                "actual_columns": [],
            }
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n")
            row_count = sum(1 for _ in handle)
        actual = tuple(header.split(",")) if header else tuple()
        return {
            "relative_path": f"working/{path.name}",
            "present": True,
            "row_count": int(row_count),
            "sha256": sha256_file(path),
            "compressed_bytes": path.stat().st_size,
            "schema_ok": actual == tuple(expected),
            "actual_columns": list(actual),
        }

    output = Path(output_dir)
    work = Path(work_dir)
    selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
    selected_count = int(len(selected))
    rich_path = work / "SELECTED_ORDER_SCORING_RICH.csv.gz"
    daily_candidates = (
        work / "SELECTED_DAILY_PROFILE_ROWS.csv.gz",
        work / "SELECTED_DAILY_PROFILES.csv.gz",
        work / "PROFILE_DAILY_PROFILE_ROWS.csv.gz",
        work / "SELECTED_DAILY_PROFILE_ROWS.csv",
        work / "SELECTED_DAILY_PROFILES.csv",
    )
    daily_path = next((candidate for candidate in daily_candidates if candidate.exists()), None)
    rich = table_receipt(rich_path if rich_path.exists() else None, RICH_ORDER_SCORING_SCHEMA)
    daily = table_receipt(daily_path, RAW_SELECTED_DAILY_ROW_SCHEMA)
    configured_limit = config.get("reporting_resource_limits", {})
    max_rich_rows = int(
        configured_limit.get(
            "max_in_memory_rich_scoring_rows",
            DEFAULT_MAX_IN_MEMORY_RICH_SCORING_ROWS,
        )
    ) if isinstance(configured_limit, Mapping) else DEFAULT_MAX_IN_MEMORY_RICH_SCORING_ROWS
    max_daily_rows = int(
        configured_limit.get(
            "max_in_memory_daily_profile_rows",
            DEFAULT_MAX_IN_MEMORY_DAILY_PROFILE_ROWS,
        )
    ) if isinstance(configured_limit, Mapping) else DEFAULT_MAX_IN_MEMORY_DAILY_PROFILE_ROWS
    physical_memory = 0
    try:
        physical_memory = int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, OSError, ValueError):
        physical_memory = 0
    configured_memory_budget = int(
        configured_limit.get(
            "max_reporting_peak_bytes", DEFAULT_REPORTING_MEMORY_BUDGET_BYTES
        )
    ) if isinstance(configured_limit, Mapping) else DEFAULT_REPORTING_MEMORY_BUDGET_BYTES
    physical_memory_budget = (
        int(physical_memory * DEFAULT_REPORTING_MEMORY_FRACTION)
        if physical_memory > 0 else configured_memory_budget
    )
    effective_memory_budget = max(
        1, min(configured_memory_budget, physical_memory_budget)
    )
    rich_estimate = int(rich["row_count"]) * ESTIMATED_PEAK_BYTES_PER_RICH_ROW
    daily_estimate = int(daily["row_count"]) * ESTIMATED_PEAK_BYTES_PER_DAILY_ROW
    combined_estimate = rich_estimate + daily_estimate
    rich_present_ok = bool(rich["present"] or selected_count == 0)
    daily_present_ok = bool(daily["present"] or selected_count == 0)
    schema_ok = bool(
        (rich["schema_ok"] or not rich["present"])
        and (daily["schema_ok"] or not daily["present"])
    )
    rows_ok = bool(
        int(rich["row_count"]) <= max_rich_rows
        and int(daily["row_count"]) <= max_daily_rows
    )
    memory_ok = combined_estimate <= effective_memory_budget
    allowed = bool(
        rich_present_ok and daily_present_ok and schema_ok and rows_ok and memory_ok
    )
    failure_reasons: list[str] = []
    if not rich_present_ok:
        failure_reasons.append("rich_scoring_missing_for_promoted_candidates")
    if not daily_present_ok:
        failure_reasons.append("selected_daily_missing_for_promoted_candidates")
    if not schema_ok:
        failure_reasons.append("preload_schema_mismatch")
    if not rows_ok:
        failure_reasons.append("configured_table_row_limit_exceeded")
    if not memory_ok:
        failure_reasons.append("combined_typed_reporting_peak_exceeds_memory_budget")
    audit = {
        "rich_scoring_relative_path": "working/SELECTED_ORDER_SCORING_RICH.csv.gz",
        # Backward-compatible rich-scoring fields retained for manifest/tests.
        "present": bool(rich["present"]),
        "selected_candidate_count": selected_count,
        "row_count": int(rich["row_count"]),
        "sha256": str(rich["sha256"]),
        "compressed_bytes": int(rich["compressed_bytes"]),
        "rich_scoring": rich,
        "selected_daily_profiles": daily,
        "estimated_peak_bytes_per_row": ESTIMATED_PEAK_BYTES_PER_RICH_ROW,
        "estimated_peak_bytes": rich_estimate,
        "rich_estimated_peak_bytes": rich_estimate,
        "daily_estimated_peak_bytes_per_row": ESTIMATED_PEAK_BYTES_PER_DAILY_ROW,
        "daily_estimated_peak_bytes": daily_estimate,
        "combined_estimated_peak_bytes": combined_estimate,
        "physical_memory_bytes": physical_memory,
        "configured_max_rows": max_rich_rows,
        "configured_max_rich_rows": max_rich_rows,
        "configured_max_daily_rows": max_daily_rows,
        "configured_memory_budget_bytes": configured_memory_budget,
        "physical_memory_budget_bytes": physical_memory_budget,
        "effective_memory_budget_bytes": effective_memory_budget,
        "allowed": allowed,
        "failure_reason": ";".join(failure_reasons),
    }
    write_json(work / "REPORTING_RESOURCE_AUDIT.json", audit, canonical=True)
    if not allowed:
        raise RuntimeError(
            "reporting resource audit failed before rich-scoring load: "
            f"{audit['failure_reason']}; rich_rows={rich['row_count']}; "
            f"daily_rows={daily['row_count']}; combined_estimated_peak_bytes="
            f"{combined_estimate}; budget={effective_memory_budget}; "
            "use a larger audited host or implement chunked reporting aggregation"
        )
    return audit


def _write_unless_frozen_existing(
    frame: pd.DataFrame,
    path: Path,
    schema: Sequence[str],
    keys: Sequence[str],
    frozen_relative_paths: set[str],
    output: Path,
) -> None:
    relative = path.relative_to(output).as_posix()
    if path.exists() and relative in frozen_relative_paths:
        return
    write_csv(frame, path, schema, keys)


def finalize_reporting(
    output_dir: str | Path = OUT,
    work_dir: str | Path | None = None,
    *,
    inputs: ReportingInputs | None = None,
    test_results_path: str | Path | None = None,
    additional_blockers: Sequence[str] = (),
) -> dict[str, object]:
    """Runner-callable deterministic reporting entry point.

    The selection freeze must already exist; this function never creates or
    changes it.  It supports a valid zero-promoted-candidate outcome by writing
    exact empty analytical tables and explicit no-data figure panels.
    """

    output = Path(output_dir)
    work = Path(work_dir) if work_dir is not None else output / "working"
    output.mkdir(parents=True, exist_ok=True)
    verify_frozen_development_artifacts(output)
    config_path = output / "PROFILE_FROZEN_CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else load_config()
    discovered_inputs = inputs is None
    if discovered_inputs:
        audit_rich_scoring_resources(output, work, config)
        supplied = discover_reporting_inputs(output, work)
    else:
        supplied = inputs

    frozen_relative_paths = _frozen_development_paths(output)
    existing_selected = _read_csv(output / "PROFILE_SELECTED_CANDIDATES.csv")
    existing_pareto = _read_csv(output / "PROFILE_PARETO_FRONTIER.csv")
    selected = normalize_selected_candidates(
        existing_selected if not existing_selected.empty or (output / "PROFILE_SELECTED_CANDIDATES.csv").exists()
        else supplied.selected_candidates
    )
    pareto = normalize_pareto(
        existing_pareto if not existing_pareto.empty or (output / "PROFILE_PARETO_FRONTIER.csv").exists()
        else supplied.pareto
    )
    daily = prepare_daily_profiles(supplied.daily_profiles, selected)
    if discovered_inputs:
        supplied.daily_profiles = None
        gc.collect()
    scoring, rich_scoring = normalize_order_scoring(supplied.order_scoring, selected)
    if discovered_inputs:
        supplied.order_scoring = None
        gc.collect()
    anchor_metrics = pd.DataFrame() if supplied.anchor_metrics is None else supplied.anchor_metrics.copy()
    anchor_metrics = _augment_terminal_followup(anchor_metrics, scoring, config)

    _write_unless_frozen_existing(
        selected, output / "PROFILE_SELECTED_CANDIDATES.csv", SELECTED_SCHEMA,
        SORT_KEYS["PROFILE_SELECTED_CANDIDATES.csv"], frozen_relative_paths, output,
    )
    _write_unless_frozen_existing(
        pareto, output / "PROFILE_PARETO_FRONTIER.csv", PARETO_SCHEMA,
        SORT_KEYS["PROFILE_PARETO_FRONTIER.csv"], frozen_relative_paths, output,
    )
    write_daily_artifacts(daily, output)

    development_results = aggregate_anchor_results(
        anchor_metrics, selected, "development", config=config
    )
    development_by_month = aggregate_anchor_results(
        anchor_metrics, selected, "development", by_month=True, config=config
    )
    confirmation_results = aggregate_anchor_results(
        anchor_metrics, selected, "confirmation", config=config
    )
    confirmation_by_month = aggregate_anchor_results(
        anchor_metrics, selected, "confirmation", by_month=True, config=config
    )
    persisted_development = _read_csv(output / "PROFILE_DEVELOPMENT_RESULTS.csv")
    development_for_labels = (
        persisted_development if not persisted_development.empty else development_results
    )
    confirmation_audit = (
        pd.DataFrame() if supplied.confirmation_labels is None
        else supplied.confirmation_labels.copy()
    )
    if not confirmation_audit.empty:
        confirmation_results = append_persisted_confirmation_labels(
            confirmation_results, confirmation_audit, selected
        )
    else:
        confirmation_results = append_confirmation_label_rows(
            confirmation_results,
            anchor_metrics,
            supplied.support_strata,
            selected,
            development_for_labels,
        )
    terminal_results = aggregate_anchor_results(
        anchor_metrics, selected, "terminal", config=config
    )
    terminal_shift = aggregate_terminal_score_shift(
        rich_scoring if not rich_scoring.empty else scoring,
        config,
    )
    terminal_results = _sort_frame(
        _normalise_schema(
            pd.concat([terminal_results, terminal_shift], ignore_index=True),
            TERMINAL_SCHEMA,
        ),
        PRIMARY_KEYS["PROFILE_TERMINAL_STRESS.csv"],
    )

    tables: dict[str, pd.DataFrame] = {
        "PROFILE_CONSTRUCTION_AUDIT.csv": aggregate_construction_audit(supplied.construction_audit, daily, selected),
        "PROFILE_SUPPORT_UNCERTAINTY.csv": aggregate_support_uncertainty(daily, scoring),
        "PROFILE_PARENT_STRUCTURE.csv": aggregate_parent_structure(supplied.parent_profiles, daily),
        "PROFILE_DEVELOPMENT_RESULTS.csv": development_results,
        "PROFILE_DEVELOPMENT_BY_MONTH.csv": development_by_month,
        "PROFILE_CONFIRMATION_RESULTS.csv": confirmation_results,
        "PROFILE_CONFIRMATION_BY_MONTH.csv": confirmation_by_month,
        "PROFILE_TERMINAL_STRESS.csv": terminal_results,
        "PROFILE_LEVEL_RESULTS.csv": aggregate_level_results(
            rich_scoring if not rich_scoring.empty else scoring
        ),
        "PROFILE_LEVEL_TRANSITIONS.csv": aggregate_level_transitions(daily),
        "PROFILE_DAILY_STABILITY.csv": aggregate_daily_stability(
            daily, supplied.hrd_days, hrd_phases=supplied.hrd_phases
        ),
        "PROFILE_FUTURE_ENTITY_TRANSFER.csv": aggregate_entity_transfer(
            supplied.entity_rows,
            rich_scoring if not rich_scoring.empty else scoring,
            selected,
        ),
        "PROFILE_FUTURE_ORDER_SCORING.csv": scoring,
        "PROFILE_SUPPORT_STRATA.csv": aggregate_support_strata(scoring),
        "PROFILE_COLD_START_RESULTS.csv": aggregate_cold_start(scoring),
        "PROFILE_HRD_DIAGNOSTICS.csv": aggregate_hrd_diagnostics(
            scoring, supplied.hrd_days, config, hrd_phases=supplied.hrd_phases
        ),
        "PROFILE_ABLATIONS.csv": aggregate_ablations(rich_scoring, selected),
    }
    for name, table in tables.items():
        _write_unless_frozen_existing(
            table, output / name, CSV_SCHEMAS[name], SORT_KEYS[name],
            frozen_relative_paths, output,
        )

    write_data_dictionary(output)
    blockers = list(additional_blockers)
    if len(selected) and daily.empty:
        blockers.append("promoted candidates exist but no selected daily profile rows were persisted")
    if len(selected) and scoring.empty:
        blockers.append("promoted candidates exist but no selected future order-scoring rows were persisted")
    if test_results_path is not None:
        source = Path(test_results_path)
        if not source.exists():
            raise FileNotFoundError(f"test result log does not exist: {source}")
        shutil.copyfile(source, output / "TEST_RESULTS.txt")
    elif not (output / "TEST_RESULTS.txt").exists():
        (output / "TEST_RESULTS.txt").write_text(
            "COMMAND: not supplied to reporting layer\nRETURN_CODE: NOT_RUN\nPASSED: UNKNOWN\nFAILED: UNKNOWN\n",
            encoding="utf-8",
        )
        blockers.append("required -B/-p no:cacheprovider test log was not supplied")
    write_blockers(output, blockers)
    figure_records = create_required_figures(output)
    manifest = build_manifest(output, figure_records, work_dir=work)
    write_json(output / "RUN_MANIFEST.json", manifest)
    # The completion summaries consume the persisted manifest verdict.  Rebuild
    # once afterwards so their final hashes enter the inventory; RUN_MANIFEST
    # itself is excluded, avoiding a self-reference.
    write_summary_skeletons(output, work_dir=work)
    manifest = build_manifest(output, figure_records, work_dir=work)
    write_json(output / "RUN_MANIFEST.json", manifest)
    report = validate_artifacts(output, write_report=True)
    return {
        "overall_pass": report["overall_pass"],
        "promoted_candidate_count": int(len(selected)),
        "zero_candidates_valid": len(selected) == 0,
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "figure_pair_count": len(figure_records),
        "checks_passed": report["checks_passed"],
        "checks_failed": report["checks_failed"],
        "manifest_sha256": sha256_file(output / "RUN_MANIFEST.json"),
        "validation_report_sha256": sha256_file(output / "ARTIFACT_VALIDATION_REPORT.md"),
    }


__all__ = [
    "ReportingInputs",
    "REQUIRED_ARTIFACTS",
    "FIGURE_STEMS",
    "FIGURE_SOURCE_SCHEMA",
    "CSV_SCHEMAS",
    "PRIMARY_KEYS",
    "DAILY_ROW_SCHEMA",
    "discover_reporting_inputs",
    "normalize_selected_candidates",
    "prepare_daily_profiles",
    "normalize_order_scoring",
    "aggregate_anchor_results",
    "aggregate_construction_audit",
    "aggregate_support_uncertainty",
    "aggregate_parent_structure",
    "aggregate_level_results",
    "aggregate_level_transitions",
    "aggregate_daily_stability",
    "aggregate_entity_transfer",
    "aggregate_support_strata",
    "aggregate_cold_start",
    "aggregate_hrd_diagnostics",
    "aggregate_ablations",
    "build_figure_source",
    "render_figure_from_source",
    "create_required_figures",
    "write_summary_skeletons",
    "build_manifest",
    "validate_artifacts",
    "verify_frozen_development_artifacts",
    "finalize_reporting",
]
