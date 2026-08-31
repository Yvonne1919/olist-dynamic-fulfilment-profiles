#!/usr/bin/env python3
"""Deterministic reporting for the direct-promise profile extension.

This module intentionally does not fit, tune, calibrate, or score a model.  It
reads the persisted direct-extension CSVs and the protected, already executed
Order V1/profile-confirmation CSVs, then writes only reporting artifacts below
``analysis/direct_promise_profile_extension_v1``.

The central design constraint is estimand separation.  The robustness output
places the new direct-promise estimand and the existing current-context
estimand in separate, side-by-side columns.  It never calculates a difference
between their gains, ranks them, or selects one because its numerical gain is
larger.

Expected minimal semantic schema
--------------------------------
Loaders accept the aliases in ``COLUMN_ALIASES``.  The preferred schemas are:

* ``DIRECT_BREACH_MONTHLY.csv``: ``cohort_month``, ``family``, ``model_id``,
  ``representation``, ``log_loss``, ``brier`` and, for candidate rows,
  ``delta_log_loss``/``delta_brier`` plus any precomputed guard columns.
* ``DIRECT_BREACH_POOLED.csv``: ``family``, ``model_id``, ``representation``,
  ``log_loss``, ``brier``; AP/AUC/top-10 lift and candidate deltas are optional.
* ``DIRECT_BREACH_CALIBRATION.csv``: ``cohort_month``, ``family``,
  ``model_id``, ``representation``, ``wace``, ``calibration_slope``.
* ``DIRECT_BREACH_SUPPORT_STRATA.csv``: ``cohort_month``, ``family``,
  ``model_id``, ``representation`` and an explicit high-support indicator or
  stratum plus proper-score deltas.
* ``DIRECT_SEVERITY_MONTHLY.csv``: ``cohort_month``, ``family``, ``quantile``,
  ``model_id``, ``representation``, ``pinball_loss`` and ``skill``.
* ``DIRECT_SEVERITY_POOLED.csv``: the same keys with pooled ``pinball_loss``,
  ``skill``, ``coverage`` and ``coverage_error`` where available.
* ``DIRECT_SEVERITY_COVERAGE.csv``: the severity keys plus ``coverage`` and
  ``coverage_error``.
* ``DIRECT_TERMINAL.csv``: either the preceding wide schemas with
  ``period=terminal`` or a long schema with ``task``, ``family``, ``model_id``,
  ``representation``, ``quantile``, ``metric`` and ``estimate``.

Where candidate deltas/skills are absent, they are reconstructed only against
the same-family direct baseline (DP0/DQ0) inside the same persisted table.  No
quantity is reconstructed across the direct and current-context studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FLOAT_FORMAT = "%.12g"
SORT_KIND = "mergesort"

DIRECT_ESTIMAND = (
    "issued promise -> issued promise + held-fixed validated process profiles"
)
CONDITIONAL_ESTIMAND = (
    "issued promise + purchase-time current context -> issued promise + "
    "purchase-time current context + held-fixed validated process profiles"
)
DIRECT_TUNING_REFERENCE = "direct promise baseline (DP0/DQ0), development only"
CONDITIONAL_TUNING_REFERENCE = (
    "Order V1 current-context baseline (M1/Q1), development only"
)
TERMINAL_CAVEAT = (
    "July-August 2018 terminal-regime stress; not pooled with January-June, "
    "not used for selection, tuning, calibration selection, or evidence labels"
)
IMPLEMENTATION_DEVIATION = (
    "No separate Order V1 amendment existed in the repository. The protected "
    "Order V1 authority is ORDER_PROTOCOL.md, ORDER_FROZEN_CONFIG.json, and "
    "ORDER_MODEL_SELECTION_FREEZE.json; the unrelated Phase 2A amendment was "
    "not substituted. Execution-environment retries, the byte-identical charter-"
    "mirror recovery, and the explicitly named concurrent-workspace integrity "
    "exclusion are fully receipted in RETRY_LOG.md and RUN_MANIFEST.json."
)

DIRECT_FILES = {
    "breach_monthly": "DIRECT_BREACH_MONTHLY.csv",
    "breach_pooled": "DIRECT_BREACH_POOLED.csv",
    "breach_calibration": "DIRECT_BREACH_CALIBRATION.csv",
    "breach_support": "DIRECT_BREACH_SUPPORT_STRATA.csv",
    "breach_ablations": "DIRECT_BREACH_ABLATIONS.csv",
    "severity_monthly": "DIRECT_SEVERITY_MONTHLY.csv",
    "severity_pooled": "DIRECT_SEVERITY_POOLED.csv",
    "severity_coverage": "DIRECT_SEVERITY_COVERAGE.csv",
    "severity_support": "DIRECT_SEVERITY_SUPPORT_STRATA.csv",
    "severity_ablations": "DIRECT_SEVERITY_ABLATIONS.csv",
    "terminal": "DIRECT_TERMINAL.csv",
    "model_selection": "MODEL_SELECTION.csv",
    "evidence_labels": "EVIDENCE_LABELS.csv",
}

PROTECTED_FILES = {
    "order_comparisons": "analysis/order_breach_severity_v1/MODEL_COMPARISON_SUMMARY.csv",
    "order_breach_results": "analysis/order_breach_severity_v1/ORDER_BREACH_RESULTS.csv",
    "order_terminal": "analysis/order_breach_severity_v1/ORDER_TERMINAL_STRESS.csv",
    "order_breach_ablations": "analysis/order_breach_severity_v1/ORDER_PROFILE_ABLATIONS.csv",
    "order_severity_results": "analysis/order_breach_severity_v1/SEVERITY_RESULTS.csv",
    "order_severity_skill": "analysis/order_breach_severity_v1/SEVERITY_PINBALL_SKILL.csv",
    "order_severity_coverage": "analysis/order_breach_severity_v1/SEVERITY_COVERAGE.csv",
    "order_severity_ablations": "analysis/order_breach_severity_v1/SEVERITY_PROFILE_ABLATIONS.csv",
    "profile_confirmation": "analysis/dynamic_profile_profile_validation_v1/working/CONFIRMATION_LABELS.csv",
}

# Public, machine-readable description used by tests/callers before execution.
EXPECTED_MINIMAL_SCHEMA: Mapping[str, tuple[str, ...]] = {
    "DIRECT_BREACH_MONTHLY.csv": (
        "cohort_month",
        "family",
        "model_id",
        "representation",
        "log_loss",
        "brier",
    ),
    "DIRECT_BREACH_POOLED.csv": (
        "family",
        "model_id",
        "representation",
        "log_loss",
        "brier",
    ),
    "DIRECT_BREACH_CALIBRATION.csv": (
        "cohort_month",
        "family",
        "model_id",
        "representation",
        "wace",
        "calibration_slope",
    ),
    "DIRECT_BREACH_SUPPORT_STRATA.csv": (
        "cohort_month",
        "family",
        "model_id",
        "representation",
        "delta_log_loss",
        "delta_brier",
    ),
    "DIRECT_SEVERITY_MONTHLY.csv": (
        "cohort_month",
        "family",
        "quantile",
        "model_id",
        "representation",
        "pinball_loss",
        "skill",
    ),
    "DIRECT_SEVERITY_POOLED.csv": (
        "family",
        "quantile",
        "model_id",
        "representation",
        "pinball_loss",
        "skill",
    ),
    "DIRECT_SEVERITY_COVERAGE.csv": (
        "family",
        "quantile",
        "model_id",
        "representation",
        "coverage",
        "coverage_error",
    ),
    "DIRECT_SEVERITY_SUPPORT_STRATA.csv": (
        "cohort_month",
        "family",
        "quantile",
        "model_id",
        "representation",
        "skill",
    ),
    "DIRECT_TERMINAL.csv": (
        "task",
        "family",
        "model_id",
        "representation",
        "metric",
        "estimate",
    ),
}

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "task": ("task", "analysis", "outcome"),
    "period": ("period", "evaluation_period", "split"),
    "cohort": ("cohort", "evaluation_cohort"),
    "cohort_month": ("cohort_month", "month", "evaluation_month"),
    "family": ("family", "model_family"),
    "model_id": (
        "model_id",
        "specification",
        "spec_id",
        "model",
        "candidate_model",
        "candidate_specification",
        "reference_main_model",
    ),
    "representation": (
        "representation",
        "profile_representation",
        "payload",
        "payload_type",
    ),
    "comparison": ("comparison", "contrast"),
    "reference_model_id": (
        "reference_model_id",
        "reference_specification",
        "baseline_model_id",
        "reference_model",
        "reference_specification",
    ),
    "probability_variant": (
        "probability_variant",
        "probability_type",
        "calibration_variant",
    ),
    "quantile": ("quantile", "q", "tau"),
    "n_orders": ("n_orders", "n_obs", "n", "sample_size"),
    "n_events": ("n_events", "breaches", "n_breaches"),
    "n_months": ("n_months", "later_month_count"),
    "prevalence": ("prevalence", "event_rate", "breach_rate"),
    "log_loss": ("log_loss", "logloss"),
    "brier": ("brier", "brier_score"),
    "average_precision": ("average_precision", "ap"),
    "roc_auc": ("roc_auc", "auc"),
    "top_10pct_lift": ("top_10pct_lift", "top10_lift", "top_10_lift"),
    "calibration_intercept": ("calibration_intercept", "cal_intercept"),
    "calibration_slope": ("calibration_slope", "cal_slope"),
    "wace": ("wace", "weighted_absolute_calibration_error"),
    "delta_log_loss": ("delta_log_loss", "log_loss_delta"),
    "delta_brier": ("delta_brier", "delta_brier_score", "brier_delta"),
    "delta_average_precision": ("delta_average_precision", "delta_ap"),
    "delta_roc_auc": ("delta_roc_auc", "delta_auc"),
    "delta_top_10pct_lift": (
        "delta_top_10pct_lift",
        "delta_top10_lift",
        "top10_lift_delta",
    ),
    "pinball_loss": ("pinball_loss", "loss"),
    "baseline_pinball_loss": (
        "baseline_pinball_loss",
        "reference_pinball_loss",
        "q1_reference_loss",
    ),
    "skill": ("skill", "pinball_skill", "skill_vs_baseline", "skill_vs_q1"),
    "median_pinball_skill": ("median_pinball_skill", "median_skill"),
    "months_nonnegative_skill": (
        "months_nonnegative_skill",
        "favourable_month_count",
    ),
    "months_both_improved": (
        "months_both_improved",
        "both_improved_month_count",
    ),
    "coverage": ("coverage", "empirical_coverage"),
    "coverage_error": ("coverage_error",),
    "metric": ("metric", "metric_name"),
    "estimate": ("estimate", "value"),
    "evidence_label": ("evidence_label", "evidence_status", "label"),
    "evidence_reason": ("evidence_reason", "label_reason", "reason"),
    "high_support_guard": (
        "high_support_guard",
        "high_support_ok",
        "high_support_no_material_reversal",
        "support_ge20_gain_present",
    ),
    "calibration_guard": (
        "calibration_guard",
        "calibration_ok",
        "calibration_not_systematically_worse",
    ),
    "score_contribution_guard": (
        "score_contribution_guard",
        "score_not_metadata_only_guard",
        "score_contributes",
        "benefit_not_metadata_only",
    ),
    "coverage_guard": (
        "coverage_guard",
        "coverage_ok",
        "coverage_not_materially_worse",
    ),
    "support_stratum": ("support_stratum", "stratum", "support_group"),
    "minimum_support": ("minimum_support", "min_support", "support_min"),
}

DIRECT_BREACH_BASELINE = "DP0"
DIRECT_SEVERITY_BASELINE = "DQ0"
DIRECT_BREACH_BLOCKS = {"DPS": "seller", "DPG": "state_od", "DPB": "both"}
DIRECT_SEVERITY_BLOCKS = {"DQS": "seller", "DQG": "state_od", "DQB": "both"}
CONDITIONAL_BREACH_BLOCKS = {
    "M2-M1": "seller",
    "M3-M1": "state_od",
    "M4-M1": "both",
}
CONDITIONAL_SEVERITY_BLOCKS = {
    "Q2-Q1": "seller",
    "Q3-Q1": "state_od",
    "Q4-Q1": "both",
}
CONDITIONAL_BREACH_MODELS = {"M2": "seller", "M3": "state_od", "M4": "both"}
CONDITIONAL_SEVERITY_MODELS = {"Q2": "seller", "Q3": "state_od", "Q4": "both"}

PROFILE_IDS = {
    "S1": "handling_level|seller_id|C|w90|l14|P1|parent=global|kappa=na|min_support=5",
    "S2": "handling_tail|seller_id|A|w90|l0|P1|parent=global|kappa=10|min_support=5",
    "R1": "transit_level|state_od|A|w90|l0|P0|parent=global|kappa=na|min_support=5",
    "R2": "transit_tail|state_od|A|w90|l0|P1|parent=global|kappa=10|min_support=5",
}

ROBUSTNESS_COLUMNS = [
    "outcome",
    "period_summary",
    "model_family",
    "quantile",
    "profile_block",
    "representation",
    "metric",
    "metric_direction",
    "direct_estimand",
    "direct_tuning_reference",
    "direct_estimate",
    "direct_evidence_label",
    "direct_evidence_reason",
    "direct_profile_label_availability",
    "direct_source_path",
    "direct_source_sha256",
    "conditional_estimand",
    "conditional_tuning_reference",
    "conditional_estimate",
    "conditional_evidence_label",
    "conditional_evidence_reason",
    "conditional_profile_label_availability",
    "conditional_source_path",
    "conditional_source_sha256",
    "interpretation_boundary",
]

EVIDENCE_COLUMNS = [
    "label_namespace",
    "evidence_role",
    "task",
    "outcome",
    "model_family",
    "quantile",
    "profile_id",
    "profile_block",
    "specification",
    "comparison",
    "representation",
    "n_months",
    "median_delta_log_loss",
    "median_delta_brier",
    "months_both_improved",
    "median_pinball_skill",
    "months_nonnegative_skill",
    "high_support_guard",
    "calibration_guard",
    "score_contribution_guard",
    "coverage_guard",
    "evidence_label",
    "evidence_reason",
    "source_path",
    "source_sha256",
    "caveat",
]


class ReportingError(RuntimeError):
    """Raised when persisted evidence cannot support deterministic reporting."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 receipt for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise ReportingError(f"Required persisted CSV is missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - pandas supplies useful detail
        raise ReportingError(f"Could not read persisted CSV {path}: {exc}") from exc


def _write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    table = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in table.columns:
                table[column] = np.nan
        table = table.loc[:, list(columns)]
    table.to_csv(
        path,
        index=False,
        float_format=FLOAT_FORMAT,
        lineterminator="\n",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _canonicalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Add canonical semantic columns without discarding original columns."""

    table = frame.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in table.columns:
            continue
        for alias in aliases:
            if alias in table.columns:
                table[canonical] = table[alias]
                break
    if "family" in table.columns:
        table["family"] = table["family"].astype(str).str.strip().str.lower()
    if "model_id" in table.columns:
        table["model_id"] = table["model_id"].astype(str).str.strip().str.upper()
    if "representation" not in table.columns:
        table["representation"] = "full"
    else:
        table["representation"] = (
            table["representation"]
            .fillna("full")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"primary": "full", "complete": "full", "scores": "score_only"})
        )
    if "quantile" in table.columns:
        values = (
            table["quantile"]
            .astype(str)
            .str.upper()
            .str.replace("Q", "", regex=False)
        )
        numeric = pd.to_numeric(values, errors="coerce")
        table["quantile"] = np.where(numeric > 1, numeric / 100.0, numeric)
    if "metric" in table.columns:
        table["metric"] = (
            table["metric"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "brier_score": "brier",
                    "delta_brier_score": "delta_brier",
                    "top10_lift": "top_10pct_lift",
                    "delta_top10_lift": "delta_top_10pct_lift",
                    "coverage": "empirical_coverage",
                    "skill": "pinball_skill",
                }
            )
        )
    return table


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_value(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "supported"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed", "not-supported"}:
        return False
    return None


def _guard_from_rows(frame: pd.DataFrame, column: str) -> bool | None:
    if column not in frame.columns or frame.empty:
        return None
    values = [_bool_value(value) for value in frame[column].tolist()]
    known = [value for value in values if value is not None]
    if not known:
        return None
    return all(known)


def _later_rows(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if "period" in table.columns:
        period = table["period"].astype(str).str.lower()
        mask = period.str.contains("later") | period.str.contains("2018-0[1-6]", regex=True)
        if mask.any():
            table = table.loc[mask].copy()
    if "cohort_month" in table.columns:
        month = table["cohort_month"].astype(str).str.slice(0, 7)
        mask = month.isin([f"2018-{value:02d}" for value in range(1, 7)])
        if mask.any():
            table = table.loc[mask].copy()
            table["cohort_month"] = month.loc[mask]
    return table


def _full_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "representation" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["representation"].astype(str).str.lower().eq("full")].copy()


def _select_probability_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one persisted probability variant without recalibration/reselection."""

    if "probability_variant" not in frame.columns or frame.empty:
        return frame.copy()
    values = frame["probability_variant"].astype(str).str.lower()
    for preferred in ("selected", "calibrated", "raw"):
        mask = values.eq(preferred)
        if mask.any():
            return frame.loc[mask].copy()
    return frame.copy()


def _ensure_breach_deltas(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    table = _canonicalise(frame)
    required = {"family", "model_id", "log_loss", "brier"}
    if not required.issubset(table.columns):
        for column in ("delta_log_loss", "delta_brier"):
            if column in table.columns:
                table[column] = _numeric(table, column)
        return table
    keys = [column for column in ("family", "cohort_month", "cohort", "period", "probability_variant") if column in table.columns]
    base = table.loc[table["model_id"].eq(baseline)].copy()
    if base.empty:
        return table
    base = base.groupby(keys, dropna=False, as_index=False).agg(
        baseline_log_loss=("log_loss", "mean"), baseline_brier=("brier", "mean")
    )
    merged = table.merge(base, on=keys, how="left", validate="m:1")
    computed_ll = _numeric(merged, "log_loss") - _numeric(merged, "baseline_log_loss")
    computed_brier = _numeric(merged, "brier") - _numeric(merged, "baseline_brier")
    if "delta_log_loss" in merged.columns:
        merged["delta_log_loss"] = _numeric(merged, "delta_log_loss").fillna(computed_ll)
    else:
        merged["delta_log_loss"] = computed_ll
    if "delta_brier" in merged.columns:
        merged["delta_brier"] = _numeric(merged, "delta_brier").fillna(computed_brier)
    else:
        merged["delta_brier"] = computed_brier
    for metric in ("average_precision", "roc_auc", "top_10pct_lift"):
        if metric not in merged.columns:
            continue
        base_metric = table.loc[table["model_id"].eq(baseline)].groupby(
            keys, dropna=False, as_index=False
        )[metric].mean().rename(columns={metric: f"baseline_{metric}"})
        merged = merged.merge(base_metric, on=keys, how="left", validate="m:1")
        computed = _numeric(merged, metric) - _numeric(merged, f"baseline_{metric}")
        delta_column = f"delta_{metric}"
        if delta_column in merged.columns:
            merged[delta_column] = _numeric(merged, delta_column).fillna(computed)
        else:
            merged[delta_column] = computed
    return merged


def _ensure_severity_skill(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    table = _canonicalise(frame)
    if "skill" in table.columns and _numeric(table, "skill").notna().any():
        table["skill"] = _numeric(table, "skill")
        return table
    if "pinball_loss" not in table.columns or "model_id" not in table.columns:
        return table
    keys = [column for column in ("family", "quantile", "cohort_month", "cohort", "period") if column in table.columns]
    base = table.loc[table["model_id"].eq(baseline)].copy()
    if base.empty:
        return table
    base = base.groupby(keys, dropna=False, as_index=False)["pinball_loss"].mean().rename(
        columns={"pinball_loss": "baseline_pinball_loss"}
    )
    merged = table.merge(base, on=keys, how="left", validate="m:1")
    denominator = _numeric(merged, "baseline_pinball_loss")
    merged["skill"] = np.where(
        denominator > 0,
        1.0 - _numeric(merged, "pinball_loss") / denominator,
        np.nan,
    )
    return merged


def _is_high_support(frame: pd.DataFrame) -> pd.Series:
    if "minimum_support" in frame.columns:
        numeric = _numeric(frame, "minimum_support")
        if numeric.notna().any():
            return numeric.ge(20)
    for column in ("support_ge20", "high_support"):
        if column in frame.columns:
            return frame[column].map(_bool_value).fillna(False).astype(bool)
    if "support_stratum" in frame.columns:
        text = frame["support_stratum"].astype(str).str.lower()
        return text.str.contains(r"ge\s*20|>=\s*20|20\+|support_ge20|adequate", regex=True)
    return pd.Series(False, index=frame.index)


def load_config(extension_dir: Path) -> dict[str, Any]:
    path = extension_dir / "DIRECT_FROZEN_CONFIG.json"
    if not path.is_file():
        raise ReportingError(f"Frozen config is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protected_sources(
    config: Mapping[str, Any], repo_root: Path
) -> pd.DataFrame:
    """Verify every source receipt frozen in the extension config."""

    rows: list[dict[str, Any]] = []
    for source_id, value in sorted(config.get("sources", {}).items()):
        if not isinstance(value, list) or len(value) != 2:
            raise ReportingError(f"Malformed source receipt for {source_id!r}")
        relative, expected = value
        path = repo_root / str(relative)
        if not path.is_file():
            raise ReportingError(f"Protected source is missing: {relative}")
        observed = sha256_file(path)
        matched = observed == str(expected)
        rows.append(
            {
                "source_id": source_id,
                "receipt_role": "frozen_config_source",
                "source_path": str(relative),
                "expected_sha256": str(expected),
                "observed_sha256": observed,
                "hash_matches_frozen_receipt": matched,
            }
        )
        if not matched:
            raise ReportingError(
                f"Protected source hash mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )
    return pd.DataFrame(rows).sort_values("source_id", kind=SORT_KIND).reset_index(drop=True)


def reporting_input_receipts(
    paths: Mapping[str, Path], repo_root: Path
) -> pd.DataFrame:
    """Receipt every direct/protected CSV actually consumed by reporting."""

    rows: list[dict[str, Any]] = []
    for source_id, path in sorted(paths.items()):
        if not path.is_file():
            continue
        rows.append(
            {
                "source_id": f"reporting_input:{source_id}",
                "receipt_role": "persisted_reporting_input",
                "source_path": _repo_relative(path, repo_root),
                "expected_sha256": "",
                "observed_sha256": sha256_file(path),
                "hash_matches_frozen_receipt": "not_applicable; observed receipt",
            }
        )
    return pd.DataFrame(rows)


def load_reporting_inputs(
    extension_dir: Path, repo_root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Load direct and protected reporting inputs without modifying them."""

    tables: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    required_direct = {
        "breach_monthly",
        "breach_pooled",
        "breach_calibration",
        "breach_support",
        "severity_monthly",
        "severity_pooled",
        "severity_coverage",
        "terminal",
        "model_selection",
    }
    for key, filename in DIRECT_FILES.items():
        path = extension_dir / filename
        paths[key] = path
        if key == "evidence_labels":
            continue
        tables[key] = _read_csv(path, required=key in required_direct)
    for key, relative in PROTECTED_FILES.items():
        path = repo_root / relative
        tables[key] = _read_csv(path, required=True)
        paths[key] = path
    for key, filename in (
        ("direct_protocol", "DIRECT_EXTENSION_PROTOCOL.md"),
        ("direct_config", "DIRECT_FROZEN_CONFIG.json"),
        ("direct_feature_manifest", "EXACT_FEATURE_MANIFEST.md"),
    ):
        path = extension_dir / filename
        if not path.is_file():
            raise ReportingError(f"Required frozen extension control is missing: {path}")
        paths[key] = path
    return tables, paths


def _source_receipt(path: Path, repo_root: Path) -> tuple[str, str]:
    return _repo_relative(path, repo_root), sha256_file(path)


def _breach_guards(
    family: str,
    model_id: str,
    monthly: pd.DataFrame,
    calibration: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[bool | None, bool | None, bool | None]:
    candidate = monthly.loc[
        monthly["family"].eq(family) & monthly["model_id"].eq(model_id)
    ].copy()
    high_support = _guard_from_rows(candidate, "high_support_guard")
    calibration_guard = _guard_from_rows(candidate, "calibration_guard")
    score_guard = _guard_from_rows(candidate, "score_contribution_guard")

    if high_support is None and not support.empty:
        high = _ensure_breach_deltas(_canonicalise(support), DIRECT_BREACH_BASELINE)
        high = high.loc[
            high.get("family", pd.Series("", index=high.index)).eq(family)
            & high.get("model_id", pd.Series("", index=high.index)).eq(model_id)
            & _is_high_support(high)
        ].copy()
        if not high.empty:
            reversal = (
                _numeric(high, "delta_log_loss").median() > 0
                and _numeric(high, "delta_brier").median() > 0
            )
            high_support = not reversal

    if calibration_guard is None:
        source = _canonicalise(calibration) if not calibration.empty else monthly
        source = _later_rows(_full_rows(_select_probability_rows(source)))
        required = {"family", "model_id", "wace", "calibration_slope"}
        if required.issubset(source.columns):
            keys = [column for column in ("family", "cohort_month", "cohort") if column in source.columns]
            base = source.loc[source["model_id"].eq(DIRECT_BREACH_BASELINE), keys + ["wace", "calibration_slope"]].copy()
            base = base.rename(columns={"wace": "base_wace", "calibration_slope": "base_slope"})
            cand = source.loc[
                source["family"].eq(family) & source["model_id"].eq(model_id),
                keys + ["wace", "calibration_slope"],
            ].copy()
            if not base.empty and not cand.empty:
                merged = cand.merge(base, on=keys, how="inner", validate="1:1")
                wace_delta = _numeric(merged, "wace") - _numeric(merged, "base_wace")
                slope_delta = (
                    (_numeric(merged, "calibration_slope") - 1).abs()
                    - (_numeric(merged, "base_slope") - 1).abs()
                )
                if wace_delta.notna().any() and slope_delta.notna().any():
                    calibration_guard = bool(
                        wace_delta.median() <= 0.005 and slope_delta.median() <= 0.1
                    )

    if score_guard is None:
        family_rows = monthly.loc[
            monthly["family"].eq(family) & monthly["model_id"].eq(model_id)
        ].copy()
        score_rows = family_rows.loc[family_rows["representation"].eq("score_only")]
        if not score_rows.empty:
            score_rows = _ensure_breach_deltas(score_rows, DIRECT_BREACH_BASELINE)
            if _numeric(score_rows, "delta_log_loss").notna().any():
                score_guard = bool(_numeric(score_rows, "delta_log_loss").median() < 0)
        if score_guard is not True:
            full = family_rows.loc[family_rows["representation"].eq("full")].copy()
            metadata = family_rows.loc[family_rows["representation"].eq("metadata_only")].copy()
            keys = [column for column in ("family", "cohort_month", "cohort") if column in full.columns]
            if not full.empty and not metadata.empty and {"log_loss", "brier"}.issubset(full.columns):
                merged = full[keys + ["log_loss", "brier"]].merge(
                    metadata[keys + ["log_loss", "brier"]],
                    on=keys,
                    suffixes=("_full", "_metadata"),
                    validate="1:1",
                )
                ll = _numeric(merged, "log_loss_full") - _numeric(merged, "log_loss_metadata")
                br = _numeric(merged, "brier_full") - _numeric(merged, "brier_metadata")
                if ll.notna().any() and br.notna().any():
                    score_guard = bool(ll.median() < 0 and br.median() <= 0)
    return high_support, calibration_guard, score_guard


def _severity_guards(
    family: str,
    model_id: str,
    quantile: float,
    monthly: pd.DataFrame,
    coverage: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[bool | None, bool | None]:
    candidate = monthly.loc[
        monthly["family"].eq(family)
        & monthly["model_id"].eq(model_id)
        & np.isclose(_numeric(monthly, "quantile"), quantile, equal_nan=False)
    ].copy()
    high_support = _guard_from_rows(candidate, "high_support_guard")
    coverage_guard = _guard_from_rows(candidate, "coverage_guard")

    if high_support is None and not support.empty:
        high = _ensure_severity_skill(_canonicalise(support), DIRECT_SEVERITY_BASELINE)
        high = high.loc[
            high.get("family", pd.Series("", index=high.index)).eq(family)
            & high.get("model_id", pd.Series("", index=high.index)).eq(model_id)
            & np.isclose(_numeric(high, "quantile"), quantile, equal_nan=False)
            & _is_high_support(high)
        ].copy()
        if not high.empty and _numeric(high, "skill").notna().any():
            high_skill = _numeric(high, "skill")
            high_support = bool(
                high_skill.median() > 0 and high_skill.ge(0).sum() >= 4
            )

    if not np.isclose(quantile, 0.9):
        coverage_guard = None
    elif coverage_guard is None and not coverage.empty:
        cov = _canonicalise(coverage)
        cov = _later_rows(_full_rows(cov))
        required = {"family", "model_id", "quantile", "coverage"}
        if required.issubset(cov.columns):
            keys = [column for column in ("family", "quantile", "cohort_month", "cohort") if column in cov.columns]
            base = cov.loc[cov["model_id"].eq(DIRECT_SEVERITY_BASELINE), keys + ["coverage"]].rename(
                columns={"coverage": "base_coverage"}
            )
            cand = cov.loc[
                cov["family"].eq(family)
                & cov["model_id"].eq(model_id)
                & np.isclose(_numeric(cov, "quantile"), quantile, equal_nan=False),
                keys + ["coverage"],
            ]
            if not base.empty and not cand.empty:
                merged = cand.merge(base, on=keys, how="inner", validate="1:1")
                worsening = (
                    (_numeric(merged, "coverage") - quantile).abs()
                    - (_numeric(merged, "base_coverage") - quantile).abs()
                )
                if worsening.notna().any():
                    coverage_guard = bool(worsening.median() <= 0.02)
    return high_support, coverage_guard


def _profile_confirmation_label_rows(
    tables: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Return only the protected RQ2 labels, in their own namespace."""

    confirmation = _canonicalise(tables["profile_confirmation"])
    confirm_source, confirm_hash = _source_receipt(paths["profile_confirmation"], repo_root)
    rows: list[dict[str, Any]] = []
    for profile_id, candidate_id in PROFILE_IDS.items():
        selected = confirmation.loc[
            confirmation["candidate_id"].astype(str).eq(candidate_id)
        ].copy()
        if len(selected) != 1:
            raise ReportingError(
                f"Expected exactly one protected confirmation row for {profile_id}; found {len(selected)}"
            )
        row = selected.iloc[0]
        block = "seller" if profile_id.startswith("S") else "state_od"
        reversal = _bool_value(row.get("high_support_material_reversal", False))
        rows.append(
            {
                "label_namespace": "RQ2_profile_confirmation",
                "evidence_role": "held_fixed_standalone_profile_confirmation",
                "task": "profile_confirmation",
                "outcome": "future_process_validity",
                "model_family": "",
                "quantile": np.nan,
                "profile_id": profile_id,
                "profile_block": block,
                "specification": candidate_id,
                "comparison": "development_selection_to_locked_future_process_confirmation",
                "representation": "rq2_confirmation",
                "n_months": row.get("n_valid_confirmation_months", np.nan),
                "median_delta_log_loss": np.nan,
                "median_delta_brier": np.nan,
                "months_both_improved": row.get("n_favourable_confirmation_months", np.nan),
                "median_pinball_skill": np.nan,
                "months_nonnegative_skill": np.nan,
                "high_support_guard": None if reversal is None else not reversal,
                "calibration_guard": np.nan,
                "score_contribution_guard": np.nan,
                "coverage_guard": np.nan,
                "evidence_label": row.get("confirmation_label", ""),
                "evidence_reason": row.get("confirmation_label_reason", ""),
                "source_path": confirm_source,
                "source_sha256": confirm_hash,
                "caveat": (
                    "Descriptive RQ2 confirmation label; not an order-level Supported/Mixed/"
                    "Not-supported label and not causal entity-quality evidence."
                ),
            }
        )
    return rows


def build_evidence_labels(
    tables: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
    repo_root: Path,
) -> pd.DataFrame:
    """Build direct RQ3 labels and retain RQ2 confirmations in a separate namespace."""

    rows: list[dict[str, Any]] = []
    monthly = _ensure_breach_deltas(
        _later_rows(_full_rows(_select_probability_rows(_canonicalise(tables["breach_monthly"])))),
        DIRECT_BREACH_BASELINE,
    )
    calibration = _canonicalise(tables.get("breach_calibration", pd.DataFrame()))
    support = _canonicalise(tables.get("breach_support", pd.DataFrame()))
    breach_source, breach_hash = _source_receipt(paths["breach_monthly"], repo_root)

    if not {"family", "model_id", "delta_log_loss", "delta_brier"}.issubset(monthly.columns):
        raise ReportingError(
            "DIRECT_BREACH_MONTHLY.csv cannot support evidence labels: expected "
            "family/model_id and proper-score values or deltas"
        )
    for (family, model_id), group in monthly.loc[
        monthly["model_id"].isin(DIRECT_BREACH_BLOCKS)
    ].groupby(["family", "model_id"], sort=True, dropna=False):
        if "cohort_month" in group.columns:
            month_rows = group.groupby("cohort_month", as_index=False).agg(
                delta_log_loss=("delta_log_loss", "mean"),
                delta_brier=("delta_brier", "mean"),
            )
        else:
            month_rows = group[["delta_log_loss", "delta_brier"]].copy()
        n_months = int(len(month_rows))
        median_ll = float(_numeric(month_rows, "delta_log_loss").median())
        median_brier = float(_numeric(month_rows, "delta_brier").median())
        months_both = int(
            (
                _numeric(month_rows, "delta_log_loss").lt(0)
                & _numeric(month_rows, "delta_brier").lt(0)
            ).sum()
        )
        high, calibration_ok, score_ok = _breach_guards(
            str(family), str(model_id), _canonicalise(tables["breach_monthly"]), calibration, support
        )
        if n_months < 6:
            label = "Mixed"
            reason = f"incomplete_later_cohort_months:{n_months}/6"
        elif median_ll >= 0 or median_brier >= 0:
            label = "Not-supported"
            reason = "median_proper_score_non_improvement"
        elif months_both <= 2:
            label = "Not-supported"
            reason = "both_scores_improved_in_at_most_two_months"
        elif months_both == 3:
            label = "Mixed"
            reason = "exactly_three_of_six_months_improved_both_scores"
        elif not all(value is True for value in (high, calibration_ok, score_ok)):
            label = "Mixed"
            missing = [
                name
                for name, value in (
                    ("high_support", high),
                    ("calibration", calibration_ok),
                    ("score_contribution", score_ok),
                )
                if value is not True
            ]
            reason = "guard_failed_or_unavailable:" + "|".join(missing)
        else:
            label = "Supported"
            reason = "all_frozen_direct_breach_rules_passed"
        rows.append(
            {
                "label_namespace": "RQ3_direct_order_evidence",
                "evidence_role": "primary_direct_operational_estimand",
                "task": "breach",
                "outcome": "breach_probability",
                "model_family": family,
                "quantile": np.nan,
                "profile_id": "",
                "profile_block": DIRECT_BREACH_BLOCKS[str(model_id)],
                "specification": model_id,
                "comparison": f"{model_id}-{DIRECT_BREACH_BASELINE}",
                "representation": "full",
                "n_months": n_months,
                "median_delta_log_loss": median_ll,
                "median_delta_brier": median_brier,
                "months_both_improved": months_both,
                "median_pinball_skill": np.nan,
                "months_nonnegative_skill": np.nan,
                "high_support_guard": high,
                "calibration_guard": calibration_ok,
                "score_contribution_guard": score_ok,
                "coverage_guard": np.nan,
                "evidence_label": label,
                "evidence_reason": reason,
                "source_path": breach_source,
                "source_sha256": breach_hash,
                "caveat": "Order-level evidence label; not an RQ2 profile-confirmation label.",
            }
        )

    severity = _ensure_severity_skill(
        _later_rows(_full_rows(_canonicalise(tables["severity_monthly"]))),
        DIRECT_SEVERITY_BASELINE,
    )
    severity_support = _canonicalise(tables.get("severity_support", pd.DataFrame()))
    if severity_support.empty:
        severity_support = _canonicalise(tables.get("breach_support", pd.DataFrame()))
    # A compatible implementation may persist both tasks in the breach-named
    # support table; task filtering prevents breach rows from being mistaken
    # for severity evidence.
    if "task" in severity_support.columns:
        severity_support = severity_support.loc[
            severity_support["task"].astype(str).str.lower().str.contains("severity")
        ].copy()
    severity_coverage = _canonicalise(tables.get("severity_coverage", pd.DataFrame()))
    severity_source, severity_hash = _source_receipt(paths["severity_monthly"], repo_root)
    required = {"family", "model_id", "quantile", "skill"}
    if not required.issubset(severity.columns):
        raise ReportingError(
            "DIRECT_SEVERITY_MONTHLY.csv cannot support evidence labels: expected "
            "family/model_id/quantile and pinball loss or skill"
        )
    candidates = severity.loc[severity["model_id"].isin(DIRECT_SEVERITY_BLOCKS)].copy()
    for (family, model_id, quantile), group in candidates.groupby(
        ["family", "model_id", "quantile"], sort=True, dropna=False
    ):
        if "cohort_month" in group.columns:
            month_rows = group.groupby("cohort_month", as_index=False)["skill"].mean()
        else:
            month_rows = group[["skill"]].copy()
        values = _numeric(month_rows, "skill").dropna()
        n_months = int(len(values))
        median_skill = float(values.median())
        favourable = int(values.ge(0).sum())
        high, coverage_ok = _severity_guards(
            str(family),
            str(model_id),
            float(quantile),
            severity,
            severity_coverage,
            severity_support,
        )
        if n_months < 6:
            label = "Mixed"
            reason = f"incomplete_later_cohort_months:{n_months}/6"
        elif median_skill <= 0 or favourable < 4:
            label = "Not-supported"
            reason = "median_skill_nonpositive_or_fewer_than_four_nonnegative_months"
        elif high is False:
            label = "Not-supported"
            reason = "low_support_only_or_high_support_non_improvement"
        elif high is True and (not np.isclose(float(quantile), 0.9) or coverage_ok is True):
            label = "Supported"
            reason = "all_frozen_direct_severity_rules_passed"
        else:
            label = "Mixed"
            missing = ["high_support"] if high is not True else []
            if np.isclose(float(quantile), 0.9) and coverage_ok is not True:
                missing.append("coverage")
            reason = "guard_failed_or_unavailable:" + "|".join(missing)
        rows.append(
            {
                "label_namespace": "RQ3_direct_order_evidence",
                "evidence_role": "primary_direct_operational_estimand",
                "task": "severity",
                "outcome": "conditional_positive_lateness",
                "model_family": family,
                "quantile": float(quantile),
                "profile_id": "",
                "profile_block": DIRECT_SEVERITY_BLOCKS[str(model_id)],
                "specification": model_id,
                "comparison": f"{model_id}-{DIRECT_SEVERITY_BASELINE}",
                "representation": "full",
                "n_months": n_months,
                "median_delta_log_loss": np.nan,
                "median_delta_brier": np.nan,
                "months_both_improved": np.nan,
                "median_pinball_skill": median_skill,
                "months_nonnegative_skill": favourable,
                "high_support_guard": high,
                "calibration_guard": np.nan,
                "score_contribution_guard": np.nan,
                "coverage_guard": coverage_ok if np.isclose(float(quantile), 0.9) else np.nan,
                "evidence_label": label,
                "evidence_reason": reason,
                "source_path": severity_source,
                "source_sha256": severity_hash,
                "caveat": "Order-level evidence label; not an RQ2 profile-confirmation label.",
            }
        )

    rows.extend(_profile_confirmation_label_rows(tables, paths, repo_root))
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["label_namespace", "outcome", "model_family", "quantile", "profile_block", "profile_id"],
        kind=SORT_KIND,
        na_position="last",
    ).reset_index(drop=True)


def load_or_create_evidence_labels(
    extension_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
    repo_root: Path,
) -> tuple[pd.DataFrame, bool]:
    """Use core labels if present; otherwise create deterministic labels."""

    path = extension_dir / DIRECT_FILES["evidence_labels"]
    if path.is_file():
        labels = _normalise_existing_evidence_labels(_read_csv(path))
        if "label_namespace" not in labels.columns:
            labels["label_namespace"] = "RQ3_direct_order_evidence"
        if "evidence_label" not in labels.columns:
            raise ReportingError("Existing EVIDENCE_LABELS.csv has no evidence-label column")
        # The modelling core may reasonably persist only the newly adjudicated
        # order labels.  Add protected RQ2 confirmations in memory for the
        # summaries/figure-data, but honour the caller contract by leaving the
        # core-supplied EVIDENCE_LABELS.csv byte-for-byte untouched.
        if not labels["label_namespace"].astype(str).eq("RQ2_profile_confirmation").any():
            rq2 = pd.DataFrame(_profile_confirmation_label_rows(tables, paths, repo_root))
            labels = pd.concat([labels, rq2], ignore_index=True, sort=False)
        return labels, False
    labels = build_evidence_labels(tables, paths, repo_root)
    _write_csv(labels, path, EVIDENCE_COLUMNS)
    return labels, True


def _normalise_existing_evidence_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a core-supplied evidence table without changing its values."""

    table = _canonicalise(frame)
    task = table.get("task", pd.Series("", index=table.index)).astype(str).str.lower()
    breach_rows = task.str.contains("breach")
    severity_rows = task.str.contains("severity")
    # The modelling core persists breach and severity labels in one wide table.
    # Coalesce the task-specific guards into the reader-facing generic column;
    # otherwise the severity-only generic field is NaN on breach rows and masks
    # the populated breach-specific no-reversal guard.
    if "high_support_no_material_reversal" in table.columns:
        available = table["high_support_no_material_reversal"].notna()
        table.loc[
            breach_rows & available, "high_support_guard"
        ] = table.loc[breach_rows & available, "high_support_no_material_reversal"]
    if "support_ge20_gain_present" in table.columns:
        available = table["support_ge20_gain_present"].notna()
        table.loc[
            severity_rows & available, "high_support_guard"
        ] = table.loc[severity_rows & available, "support_ge20_gain_present"]
    if "evidence_label" not in table.columns and "evidence_status" in table.columns:
        table["evidence_label"] = table["evidence_status"]
    if "model_family" not in table.columns and "family" in table.columns:
        table["model_family"] = table["family"]
    if "specification" not in table.columns and "model_id" in table.columns:
        table["specification"] = table["model_id"]
    if "label_namespace" not in table.columns:
        table["label_namespace"] = "RQ3_direct_order_evidence"

    if "outcome" not in table.columns:
        table["outcome"] = np.select(
            [task.str.contains("breach"), task.str.contains("severity")],
            ["breach_probability", "conditional_positive_lateness"],
            default="",
        )
    if "profile_block" not in table.columns:
        specification = table.get(
            "specification", table.get("model_id", pd.Series("", index=table.index))
        ).astype(str).str.upper()
        comparison = table.get("comparison", pd.Series("", index=table.index)).astype(str).str.upper()
        block = specification.map({**DIRECT_BREACH_BLOCKS, **DIRECT_SEVERITY_BLOCKS})
        block = block.fillna(
            comparison.map(
                {
                    "DPS-DP0": "seller",
                    "DPG-DP0": "state_od",
                    "DPB-DP0": "both",
                    "DQS-DQ0": "seller",
                    "DQG-DQ0": "state_od",
                    "DQB-DQ0": "both",
                }
            )
        )
        table["profile_block"] = block.fillna("")
    return table


def _label_lookup(labels: pd.DataFrame) -> dict[tuple[str, str, str, float | None], tuple[str, str]]:
    lookup: dict[tuple[str, str, str, float | None], tuple[str, str]] = {}
    table = _canonicalise(labels)
    if "label_namespace" in table.columns:
        table = table.loc[
            table["label_namespace"].astype(str).str.contains("order", case=False, na=False)
        ].copy()
    for _, row in table.iterrows():
        outcome = str(row.get("outcome", ""))
        family = str(row.get("model_family", row.get("family", ""))).lower()
        block = str(row.get("profile_block", ""))
        quantile = pd.to_numeric(pd.Series([row.get("quantile")]), errors="coerce").iloc[0]
        qkey = None if pd.isna(quantile) else float(quantile)
        lookup[(outcome, family, block, qkey)] = (
            str(row.get("evidence_label", row.get("evidence_status", ""))),
            str(row.get("evidence_reason", "")),
        )
    return lookup


def _tidy_row(
    *,
    outcome: str,
    period: str,
    family: str,
    block: str,
    metric: str,
    estimate: Any,
    source_path: Path,
    repo_root: Path,
    quantile: float | None = None,
    representation: str = "full",
    label: str = "",
    reason: str = "",
    role: str,
) -> dict[str, Any]:
    path, receipt = _source_receipt(source_path, repo_root)
    return {
        "outcome": outcome,
        "period_summary": period,
        "model_family": family,
        "quantile": quantile,
        "profile_block": block,
        "representation": representation,
        "metric": metric,
        "estimate": pd.to_numeric(pd.Series([estimate]), errors="coerce").iloc[0],
        "evidence_label": label,
        "evidence_reason": reason,
        "source_path": path,
        "source_sha256": receipt,
        "estimand_role": role,
    }


def _direct_tidy(
    tables: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
    labels: pd.DataFrame,
    repo_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = _label_lookup(labels)
    breach_labels = labels.loc[
        labels.get("label_namespace", pd.Series("", index=labels.index))
        .astype(str)
        .eq("RQ3_direct_order_evidence")
        & labels.get("outcome", pd.Series("", index=labels.index)).astype(str).eq("breach_probability")
    ].copy()
    for _, row in breach_labels.iterrows():
        family = str(row.get("model_family", row.get("family", ""))).lower()
        block = str(row.get("profile_block", ""))
        label = str(row.get("evidence_label", row.get("evidence_status", "")))
        reason = str(row.get("evidence_reason", ""))
        for metric, column in (
            ("delta_log_loss", "median_delta_log_loss"),
            ("delta_brier", "median_delta_brier"),
            ("favourable_month_count", "months_both_improved"),
        ):
            rows.append(
                _tidy_row(
                    outcome="breach_probability",
                    period="later_monthly_median",
                    family=family,
                    block=block,
                    metric=metric,
                    estimate=row.get(column),
                    source_path=paths["evidence_labels"],
                    repo_root=repo_root,
                    label=label,
                    reason=reason,
                    role="primary_direct_operational_estimand",
                )
            )

    pooled = _ensure_breach_deltas(
        _full_rows(_select_probability_rows(_canonicalise(tables["breach_pooled"]))),
        DIRECT_BREACH_BASELINE,
    )
    for _, row in pooled.loc[pooled["model_id"].isin(DIRECT_BREACH_BLOCKS)].iterrows():
        family = str(row["family"])
        block = DIRECT_BREACH_BLOCKS[str(row["model_id"])]
        label, reason = lookup.get(("breach_probability", family, block, None), ("", ""))
        for metric in (
            "delta_log_loss",
            "delta_brier",
            "delta_average_precision",
            "delta_roc_auc",
            "delta_top_10pct_lift",
        ):
            if metric in pooled.columns and pd.notna(row.get(metric)):
                rows.append(
                    _tidy_row(
                        outcome="breach_probability",
                        period="later_pooled",
                        family=family,
                        block=block,
                        metric=metric,
                        estimate=row.get(metric),
                        source_path=paths["breach_pooled"],
                        repo_root=repo_root,
                        label=label,
                        reason=reason,
                        role="primary_direct_operational_estimand",
                    )
                )

    severity_labels = labels.loc[
        labels.get("label_namespace", pd.Series("", index=labels.index))
        .astype(str)
        .eq("RQ3_direct_order_evidence")
        & labels.get("outcome", pd.Series("", index=labels.index))
        .astype(str)
        .eq("conditional_positive_lateness")
    ].copy()
    for _, row in severity_labels.iterrows():
        family = str(row.get("model_family", row.get("family", ""))).lower()
        block = str(row.get("profile_block", ""))
        quantile = pd.to_numeric(pd.Series([row.get("quantile")]), errors="coerce").iloc[0]
        label = str(row.get("evidence_label", row.get("evidence_status", "")))
        reason = str(row.get("evidence_reason", ""))
        for metric, column in (
            ("pinball_skill", "median_pinball_skill"),
            ("favourable_month_count", "months_nonnegative_skill"),
        ):
            rows.append(
                _tidy_row(
                    outcome="conditional_positive_lateness",
                    period="later_monthly_median",
                    family=family,
                    block=block,
                    metric=metric,
                    estimate=row.get(column),
                    source_path=paths["evidence_labels"],
                    repo_root=repo_root,
                    quantile=float(quantile),
                    label=label,
                    reason=reason,
                    role="primary_direct_operational_estimand",
                )
            )

    severity_pooled = _ensure_severity_skill(
        _full_rows(_canonicalise(tables["severity_pooled"])), DIRECT_SEVERITY_BASELINE
    )
    for _, row in severity_pooled.loc[
        severity_pooled["model_id"].isin(DIRECT_SEVERITY_BLOCKS)
    ].iterrows():
        family = str(row["family"])
        block = DIRECT_SEVERITY_BLOCKS[str(row["model_id"])]
        quantile = float(row["quantile"])
        label, reason = lookup.get(
            ("conditional_positive_lateness", family, block, quantile), ("", "")
        )
        for metric, column in (
            ("pinball_skill", "skill"),
            ("empirical_coverage", "coverage"),
            ("coverage_error", "coverage_error"),
        ):
            if metric in {"empirical_coverage", "coverage_error"} and not np.isclose(
                quantile, 0.9
            ):
                continue
            if column in severity_pooled.columns and pd.notna(row.get(column)):
                rows.append(
                    _tidy_row(
                        outcome="conditional_positive_lateness",
                        period="later_pooled",
                        family=family,
                        block=block,
                        metric=metric,
                        estimate=row.get(column),
                        source_path=paths["severity_pooled"],
                        repo_root=repo_root,
                        quantile=quantile,
                        label=label,
                        reason=reason,
                        role="primary_direct_operational_estimand",
                    )
                )

    terminal = _canonicalise(tables["terminal"])
    if {"metric", "estimate"}.issubset(terminal.columns):
        for _, row in terminal.iterrows():
            model_id = str(row.get("model_id", ""))
            family = str(row.get("family", "")).lower()
            metric = str(row.get("metric", ""))
            task = str(row.get("task", "")).lower()
            quantile = pd.to_numeric(pd.Series([row.get("quantile")]), errors="coerce").iloc[0]
            if model_id in DIRECT_BREACH_BLOCKS and metric in {"delta_log_loss", "delta_brier"}:
                block = DIRECT_BREACH_BLOCKS[model_id]
                label, reason = lookup.get(("breach_probability", family, block, None), ("", ""))
                rows.append(
                    _tidy_row(
                        outcome="breach_probability",
                        period="terminal_stress",
                        family=family,
                        block=block,
                        metric=metric,
                        estimate=row.get("estimate"),
                        source_path=paths["terminal"],
                        repo_root=repo_root,
                        label=label,
                        reason=reason,
                        role="primary_direct_operational_estimand",
                    )
                )
            elif model_id in DIRECT_SEVERITY_BLOCKS and metric in {
                "pinball_skill",
                "empirical_coverage",
                "coverage_error",
            }:
                block = DIRECT_SEVERITY_BLOCKS[model_id]
                qkey = None if pd.isna(quantile) else float(quantile)
                if metric in {"empirical_coverage", "coverage_error"} and (
                    qkey is None or not np.isclose(qkey, 0.9)
                ):
                    continue
                label, reason = lookup.get(
                    ("conditional_positive_lateness", family, block, qkey), ("", "")
                )
                normal_metric = {"skill": "pinball_skill", "coverage": "empirical_coverage"}.get(metric, metric)
                rows.append(
                    _tidy_row(
                        outcome="conditional_positive_lateness",
                        period="terminal_stress",
                        family=family,
                        block=block,
                        metric=normal_metric,
                        estimate=row.get("estimate"),
                        source_path=paths["terminal"],
                        repo_root=repo_root,
                        quantile=qkey,
                        label=label,
                        reason=reason,
                        role="primary_direct_operational_estimand",
                    )
                )
            elif (
                model_id in PROFILE_IDS
                or "profile" in task
                or any(metric.startswith(f"{profile.lower()}_") for profile in PROFILE_IDS)
            ) and any(
                token in metric
                for token in ("availability", "available", "cold_start", "mapping", "support")
            ):
                if model_id in PROFILE_IDS:
                    block = "seller" if model_id.startswith("S") else "state_od"
                elif metric.startswith(("s1_", "s2_", "seller_")):
                    block = "seller"
                elif metric.startswith(("r1_", "r2_", "state_od_", "geographic_")):
                    block = "state_od"
                else:
                    block = "both"
                rows.append(
                    _tidy_row(
                        outcome="profile_availability",
                        period="terminal_stress",
                        family=family or "descriptive",
                        block=block,
                        metric=metric,
                        estimate=row.get("estimate"),
                        source_path=paths["terminal"],
                        repo_root=repo_root,
                        quantile=None,
                        label="descriptive-only",
                        reason="terminal_profile_availability_not_an_evidence_label",
                        role="primary_direct_operational_estimand",
                    )
                )
            elif "regime_context" in task or str(row.get("outcome", "")).lower() == "terminal_regime_context":
                rows.append(
                    _tidy_row(
                        outcome="terminal_regime_context",
                        period="terminal_stress",
                        family=family or "all",
                        block="all",
                        metric=metric,
                        estimate=row.get("estimate"),
                        source_path=paths["terminal"],
                        repo_root=repo_root,
                        quantile=None,
                        representation="not_applicable",
                        label="descriptive-only",
                        reason="terminal_regime_context_not_an_evidence_label",
                        role="primary_direct_operational_estimand",
                    )
                )
    return pd.DataFrame(rows)


def _conditional_tidy(
    tables: Mapping[str, pd.DataFrame], paths: Mapping[str, Path], repo_root: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = _canonicalise(tables["order_comparisons"])
    for _, row in comparisons.loc[
        comparisons["comparison"].astype(str).isin(CONDITIONAL_BREACH_BLOCKS)
    ].iterrows():
        comparison = str(row["comparison"])
        block = CONDITIONAL_BREACH_BLOCKS[comparison]
        family = str(row["family"])
        label = str(row.get("evidence_label", row.get("evidence_status", "")))
        reason = str(row.get("evidence_reason", ""))
        for metric, column in (
            ("delta_log_loss", "median_delta_log_loss"),
            ("delta_brier", "median_delta_brier"),
            ("favourable_month_count", "months_both_improved"),
        ):
            rows.append(
                _tidy_row(
                    outcome="breach_probability",
                    period="later_monthly_median",
                    family=family,
                    block=block,
                    metric=metric,
                    estimate=row.get(column),
                    source_path=paths["order_comparisons"],
                    repo_root=repo_root,
                    label=label,
                    reason=reason,
                    role="secondary_current_context_robustness",
                )
            )

    breach = _canonicalise(tables["order_breach_results"])
    if "period" in breach.columns:
        breach = breach.loc[
            breach["period"].astype(str).str.lower().eq("aggregate")
            | breach.get("cohort", pd.Series("", index=breach.index)).astype(str).eq("later_pooled")
        ].copy()
    breach = _select_probability_rows(breach)
    for family in sorted(breach["family"].dropna().astype(str).unique()):
        base = breach.loc[
            breach["family"].eq(family) & breach["model_id"].eq("M1")
        ]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for model_id, block in CONDITIONAL_BREACH_MODELS.items():
            cand = breach.loc[
                breach["family"].eq(family) & breach["model_id"].eq(model_id)
            ]
            if cand.empty:
                continue
            cand_row = cand.iloc[0]
            comparison = f"{model_id}-M1"
            label_row = comparisons.loc[
                comparisons["family"].eq(family) & comparisons["comparison"].astype(str).eq(comparison)
            ]
            label = str(label_row.iloc[0].get("evidence_label", label_row.iloc[0].get("evidence_status", ""))) if not label_row.empty else ""
            reason = str(label_row.iloc[0].get("evidence_reason", "")) if not label_row.empty else ""
            for metric in ("log_loss", "brier", "average_precision", "roc_auc", "top_10pct_lift"):
                if metric not in breach.columns:
                    continue
                estimate = pd.to_numeric(pd.Series([cand_row.get(metric)]), errors="coerce").iloc[0] - pd.to_numeric(
                    pd.Series([base_row.get(metric)]), errors="coerce"
                ).iloc[0]
                rows.append(
                    _tidy_row(
                        outcome="breach_probability",
                        period="later_pooled",
                        family=family,
                        block=block,
                        metric=f"delta_{metric}",
                        estimate=estimate,
                        source_path=paths["order_breach_results"],
                        repo_root=repo_root,
                        label=label,
                        reason=reason,
                        role="secondary_current_context_robustness",
                    )
                )

    severity_comparisons = comparisons.loc[
        comparisons["comparison"].astype(str).isin(CONDITIONAL_SEVERITY_BLOCKS)
    ].copy()
    for _, row in severity_comparisons.iterrows():
        comparison = str(row["comparison"])
        block = CONDITIONAL_SEVERITY_BLOCKS[comparison]
        family = str(row["family"])
        quantile = float(row["quantile"])
        label = str(row.get("evidence_label", row.get("evidence_status", "")))
        reason = str(row.get("evidence_reason", ""))
        for metric, column in (
            ("pinball_skill", "median_pinball_skill"),
            ("favourable_month_count", "months_nonnegative_skill"),
        ):
            rows.append(
                _tidy_row(
                    outcome="conditional_positive_lateness",
                    period="later_monthly_median",
                    family=family,
                    block=block,
                    metric=metric,
                    estimate=row.get(column),
                    source_path=paths["order_comparisons"],
                    repo_root=repo_root,
                    quantile=quantile,
                    label=label,
                    reason=reason,
                    role="secondary_current_context_robustness",
                )
            )

    severity = _canonicalise(tables["order_severity_results"])
    severity = severity.loc[
        severity.get("period", pd.Series("", index=severity.index)).astype(str).eq("aggregate")
        | severity.get("cohort", pd.Series("", index=severity.index)).astype(str).eq("later_pooled")
    ].copy()
    for _, row in severity.loc[severity["model_id"].isin(CONDITIONAL_SEVERITY_MODELS)].iterrows():
        family = str(row["family"])
        model_id = str(row["model_id"])
        block = CONDITIONAL_SEVERITY_MODELS[model_id]
        quantile = float(row["quantile"])
        comparison = f"{model_id}-Q1"
        label_row = severity_comparisons.loc[
            severity_comparisons["family"].eq(family)
            & severity_comparisons["comparison"].astype(str).eq(comparison)
            & np.isclose(_numeric(severity_comparisons, "quantile"), quantile, equal_nan=False)
        ]
        label = str(label_row.iloc[0].get("evidence_label", label_row.iloc[0].get("evidence_status", ""))) if not label_row.empty else ""
        reason = str(label_row.iloc[0].get("evidence_reason", "")) if not label_row.empty else ""
        for metric, column in (
            ("pinball_skill", "skill"),
            ("empirical_coverage", "coverage"),
            ("coverage_error", "coverage_error"),
        ):
            if metric in {"empirical_coverage", "coverage_error"} and not np.isclose(
                quantile, 0.9
            ):
                continue
            if column in severity.columns and pd.notna(row.get(column)):
                rows.append(
                    _tidy_row(
                        outcome="conditional_positive_lateness",
                        period="later_pooled",
                        family=family,
                        block=block,
                        metric=metric,
                        estimate=row.get(column),
                        source_path=paths["order_severity_results"],
                        repo_root=repo_root,
                        quantile=quantile,
                        label=label,
                        reason=reason,
                        role="secondary_current_context_robustness",
                    )
                )

    terminal = _canonicalise(tables["order_terminal"])
    if {"metric", "estimate", "comparison"}.issubset(terminal.columns):
        for _, row in terminal.iterrows():
            comparison = str(row.get("comparison", ""))
            family = str(row.get("family", ""))
            metric = str(row.get("metric", ""))
            quantile = pd.to_numeric(pd.Series([row.get("quantile")]), errors="coerce").iloc[0]
            if comparison in CONDITIONAL_BREACH_BLOCKS and metric in {"delta_log_loss", "delta_brier"}:
                block = CONDITIONAL_BREACH_BLOCKS[comparison]
                label_row = comparisons.loc[
                    comparisons["family"].eq(family) & comparisons["comparison"].astype(str).eq(comparison)
                ]
                label = str(label_row.iloc[0].get("evidence_label", label_row.iloc[0].get("evidence_status", ""))) if not label_row.empty else ""
                reason = str(label_row.iloc[0].get("evidence_reason", "")) if not label_row.empty else ""
                rows.append(
                    _tidy_row(
                        outcome="breach_probability",
                        period="terminal_stress",
                        family=family,
                        block=block,
                        metric=metric,
                        estimate=row.get("estimate"),
                        source_path=paths["order_terminal"],
                        repo_root=repo_root,
                        label=label,
                        reason=reason,
                        role="secondary_current_context_robustness",
                    )
                )
            elif comparison in CONDITIONAL_SEVERITY_BLOCKS and metric == "pinball_skill":
                block = CONDITIONAL_SEVERITY_BLOCKS[comparison]
                qkey = None if pd.isna(quantile) else float(quantile)
                label_row = severity_comparisons.loc[
                    severity_comparisons["family"].eq(family)
                    & severity_comparisons["comparison"].astype(str).eq(comparison)
                    & np.isclose(_numeric(severity_comparisons, "quantile"), qkey, equal_nan=False)
                ]
                label = str(label_row.iloc[0].get("evidence_label", label_row.iloc[0].get("evidence_status", ""))) if not label_row.empty else ""
                reason = str(label_row.iloc[0].get("evidence_reason", "")) if not label_row.empty else ""
                rows.append(
                    _tidy_row(
                        outcome="conditional_positive_lateness",
                        period="terminal_stress",
                        family=family,
                        block=block,
                        metric="pinball_skill",
                        estimate=row.get("estimate"),
                        source_path=paths["order_terminal"],
                        repo_root=repo_root,
                        quantile=qkey,
                        label=label,
                        reason=reason,
                        role="secondary_current_context_robustness",
                    )
                )

    terminal_severity = _canonicalise(tables["order_severity_results"])
    terminal_severity = terminal_severity.loc[
        terminal_severity.get("period", pd.Series("", index=terminal_severity.index)).astype(str).eq("terminal")
        & terminal_severity["model_id"].isin(CONDITIONAL_SEVERITY_MODELS)
    ].copy()
    for _, row in terminal_severity.iterrows():
        family = str(row["family"])
        model_id = str(row["model_id"])
        block = CONDITIONAL_SEVERITY_MODELS[model_id]
        quantile = float(row["quantile"])
        if not np.isclose(quantile, 0.9):
            continue
        comparison = f"{model_id}-Q1"
        label_row = severity_comparisons.loc[
            severity_comparisons["family"].eq(family)
            & severity_comparisons["comparison"].astype(str).eq(comparison)
            & np.isclose(_numeric(severity_comparisons, "quantile"), quantile, equal_nan=False)
        ]
        label = str(label_row.iloc[0].get("evidence_label", label_row.iloc[0].get("evidence_status", ""))) if not label_row.empty else ""
        reason = str(label_row.iloc[0].get("evidence_reason", "")) if not label_row.empty else ""
        rows.append(
            _tidy_row(
                outcome="conditional_positive_lateness",
                period="terminal_stress",
                family=family,
                block=block,
                metric="empirical_coverage",
                estimate=row.get("coverage"),
                source_path=paths["order_severity_results"],
                repo_root=repo_root,
                quantile=quantile,
                label=label,
                reason=reason,
                role="secondary_current_context_robustness",
            )
        )
    return pd.DataFrame(rows)


def build_robustness_table(
    tables: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
    labels: pd.DataFrame,
    repo_root: Path,
) -> pd.DataFrame:
    """Return a wide side-by-side comparison with no cross-estimand arithmetic."""

    direct = _direct_tidy(tables, paths, labels, repo_root)
    conditional = _conditional_tidy(tables, paths, repo_root)
    if direct.empty:
        raise ReportingError("No direct estimand rows could be constructed")
    if conditional.empty:
        raise ReportingError("No protected current-context robustness rows could be constructed")
    keys = [
        "outcome",
        "period_summary",
        "model_family",
        "quantile",
        "profile_block",
        "representation",
        "metric",
    ]
    direct = direct.rename(
        columns={
            "estimate": "direct_estimate",
            "evidence_label": "direct_evidence_label",
            "evidence_reason": "direct_evidence_reason",
            "source_path": "direct_source_path",
            "source_sha256": "direct_source_sha256",
        }
    )
    conditional = conditional.rename(
        columns={
            "estimate": "conditional_estimate",
            "evidence_label": "conditional_evidence_label",
            "evidence_reason": "conditional_evidence_reason",
            "source_path": "conditional_source_path",
            "source_sha256": "conditional_source_sha256",
        }
    )
    direct = direct.drop_duplicates(keys, keep="first")
    conditional = conditional.drop_duplicates(keys, keep="first")
    merged = direct[keys + [
        "direct_estimate",
        "direct_evidence_label",
        "direct_evidence_reason",
        "direct_source_path",
        "direct_source_sha256",
    ]].merge(
        conditional[keys + [
            "conditional_estimate",
            "conditional_evidence_label",
            "conditional_evidence_reason",
            "conditional_source_path",
            "conditional_source_sha256",
        ]],
        on=keys,
        how="outer",
        validate="1:1",
    )
    merged["metric_direction"] = merged["metric"].map(
        {
            "delta_log_loss": "negative_is_better",
            "delta_brier": "negative_is_better",
            "delta_average_precision": "positive_is_better",
            "delta_roc_auc": "positive_is_better",
            "delta_top_10pct_lift": "positive_is_better",
            "pinball_skill": "positive_is_better",
            "favourable_month_count": "larger_is_more_consistent",
            "empirical_coverage": "compare_with_nominal_quantile;not_a_guarantee",
            "coverage_error": "closer_to_zero_is_better",
        }
    ).fillna("see_metric_definition")
    merged["direct_estimand"] = DIRECT_ESTIMAND
    merged["direct_tuning_reference"] = DIRECT_TUNING_REFERENCE
    merged["conditional_estimand"] = CONDITIONAL_ESTIMAND
    merged["conditional_tuning_reference"] = CONDITIONAL_TUNING_REFERENCE
    terminal_mask = merged["period_summary"].eq("terminal_stress")
    availability_mask = merged["outcome"].eq("profile_availability")
    regime_mask = merged["outcome"].eq("terminal_regime_context")
    merged["direct_profile_label_availability"] = np.select(
        [availability_mask, regime_mask, terminal_mask],
        [
            "descriptive terminal availability only; no evidence label",
            "descriptive terminal regime context only; no evidence label",
            "January-June label shown for context only; no terminal evidence label",
        ],
        default="January-June direct order label available",
    )
    merged["conditional_profile_label_availability"] = np.select(
        [availability_mask, regime_mask, terminal_mask],
        [
            "not applicable; protected Order V1 not recomputed",
            "not applicable; shared terminal cohort context",
            "January-June Order V1 label shown for context only; no terminal evidence label",
        ],
        default="January-June protected Order V1 label available",
    )
    merged["interpretation_boundary"] = np.where(
        merged["period_summary"].eq("terminal_stress"),
        TERMINAL_CAVEAT,
        "Side-by-side interpretive robustness only; do not subtract, rank, or select estimands by gain magnitude.",
    )
    merged = merged.sort_values(keys, kind=SORT_KIND, na_position="last").reset_index(drop=True)
    return merged


def build_figure_data(robustness: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Create long, plot-ready data while retaining estimand provenance."""

    rows: list[dict[str, Any]] = []
    for _, row in robustness.iterrows():
        for prefix, role in (
            ("direct", "primary_direct_operational_estimand"),
            ("conditional", "secondary_current_context_robustness"),
        ):
            estimate = row.get(f"{prefix}_estimate")
            if pd.isna(estimate):
                continue
            rows.append(
                {
                    "panel": f"{row['outcome']}__{row['period_summary']}",
                    "estimand_role": role,
                    "estimand": row.get(f"{prefix}_estimand"),
                    "outcome": row["outcome"],
                    "period_summary": row["period_summary"],
                    "model_family": row["model_family"],
                    "quantile": row["quantile"],
                    "profile_block": row["profile_block"],
                    "representation": row["representation"],
                    "metric": row["metric"],
                    "metric_direction": row["metric_direction"],
                    "estimate": estimate,
                    "evidence_label": row.get(f"{prefix}_evidence_label", ""),
                    "label_namespace": (
                        "RQ3_direct_order_evidence"
                        if prefix == "direct"
                        else "RQ3_current_context_order_evidence"
                    ),
                    "source_path": row.get(f"{prefix}_source_path", ""),
                    "source_sha256": row.get(f"{prefix}_source_sha256", ""),
                    "note": row["interpretation_boundary"],
                }
            )
    profile_rows = labels.loc[
        labels.get("label_namespace", pd.Series("", index=labels.index))
        .astype(str)
        .eq("RQ2_profile_confirmation")
    ]
    for _, row in profile_rows.iterrows():
        rows.append(
            {
                "panel": "rq2_profile_confirmation",
                "estimand_role": "held_fixed_standalone_profile_confirmation",
                "estimand": "locked future-process confirmation",
                "outcome": "future_process_validity",
                "period_summary": "January-June_2018_confirmation",
                "model_family": "",
                "quantile": np.nan,
                "profile_block": row.get("profile_block", ""),
                "representation": row.get("profile_id", ""),
                "metric": "favourable_confirmation_month_count",
                "metric_direction": "descriptive_confirmation_rubric",
                "estimate": row.get("months_both_improved", np.nan),
                "evidence_label": row.get("evidence_label", ""),
                "label_namespace": "RQ2_profile_confirmation",
                "source_path": row.get("source_path", ""),
                "source_sha256": row.get("source_sha256", ""),
                "note": row.get("caveat", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["panel", "estimand_role", "model_family", "quantile", "profile_block", "metric"],
        kind=SORT_KIND,
        na_position="last",
    ).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 40) -> str:
    table = frame.copy().head(max_rows)
    if table.empty:
        return "_No persisted rows available._"
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, list(columns)].copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else format(float(value), ".8g")
            )
        else:
            table[column] = table[column].fillna("").astype(str)
        table[column] = table[column].str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in table.astype(str).values.tolist()]
    return "\n".join([header, separator, *body])


def _profile_summary(labels: pd.DataFrame) -> pd.DataFrame:
    table = labels.loc[
        labels.get("label_namespace", pd.Series("", index=labels.index))
        .astype(str)
        .eq("RQ2_profile_confirmation")
    ].copy()
    return table


def _direct_label_summary(labels: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return labels.loc[
        labels.get("label_namespace", pd.Series("", index=labels.index))
        .astype(str)
        .eq("RQ3_direct_order_evidence")
        & labels.get("outcome", pd.Series("", index=labels.index)).astype(str).eq(outcome)
    ].copy()


def _receipt_markdown(receipts: pd.DataFrame) -> str:
    return _markdown_table(
        receipts,
        [
            "source_id",
            "receipt_role",
            "source_path",
            "expected_sha256",
            "observed_sha256",
            "hash_matches_frozen_receipt",
        ],
        max_rows=100,
    )


def render_summary_en(
    labels: pd.DataFrame,
    robustness: pd.DataFrame,
    receipts: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> str:
    breach = _direct_label_summary(labels, "breach_probability")
    severity = _direct_label_summary(labels, "conditional_positive_lateness")
    profiles = _profile_summary(labels)
    monthly_robustness = robustness.loc[
        robustness["period_summary"].eq("later_monthly_median")
    ]
    pooled = robustness.loc[robustness["period_summary"].eq("later_pooled")]
    terminal = robustness.loc[robustness["period_summary"].eq("terminal_stress")]
    terminal_context = terminal.loc[terminal["outcome"].eq("terminal_regime_context")]
    terminal_availability = terminal.loc[terminal["outcome"].eq("profile_availability")]
    terminal_performance = terminal.loc[
        ~terminal["outcome"].isin(["terminal_regime_context", "profile_availability"])
    ]
    return f"""# Direct Promise + Validated-Profile Extension V1 — Result Summary

## Scope and evidence roles

Primary operational estimand: **{DIRECT_ESTIMAND}**. The direct baseline contains
exactly `promised_delivery_days`; hyperparameters are selected on DP0/DQ0 using
development data only and shared within each direct ladder.

Secondary robustness estimand: **{CONDITIONAL_ESTIMAND}**. Its values are read
from protected Order V1 outputs. The two gain columns are interpretive and are
not subtracted, ranked, or used to choose between estimands.

## Evidence-label namespaces

RQ2 labels (`Strongly confirmed`, `Partially confirmed`, `Not confirmed`) describe
held-fixed standalone future-process confirmation. RQ3 order labels (`Supported`,
`Mixed`, `Not-supported`) evaluate profile increments in a specified order-level
information ladder. They are not interchangeable.

### Held-fixed RQ2 profile confirmations

{_markdown_table(profiles, ['profile_id', 'profile_block', 'evidence_label', 'n_months', 'months_both_improved', 'evidence_reason'])}

### January–June direct breach labels

{_markdown_table(breach, ['model_family', 'profile_block', 'median_delta_log_loss', 'median_delta_brier', 'months_both_improved', 'high_support_guard', 'calibration_guard', 'score_contribution_guard', 'evidence_label', 'evidence_reason'])}

### January–June direct conditional-severity labels

{_markdown_table(severity, ['model_family', 'quantile', 'profile_block', 'median_pinball_skill', 'months_nonnegative_skill', 'high_support_guard', 'coverage_guard', 'evidence_label', 'evidence_reason'])}

## Direct versus current-context robustness

The two columns below are side-by-side estimates of different information-set
contrasts. They are not a difference-in-differences and do not identify a
preferred estimand by relative magnitude.

{_markdown_table(monthly_robustness, ['outcome', 'model_family', 'quantile', 'profile_block', 'metric', 'direct_estimate', 'direct_evidence_label', 'conditional_estimate', 'conditional_evidence_label'], max_rows=60)}

## Selected direct settings

{_markdown_table(model_selection, list(model_selection.columns)[:10], max_rows=30)}

## Later-pooled boundary and coverage

Pooled rows remain separate from six-month medians. Coverage is empirical, is
not a production guarantee, and does not by itself establish quantile skill.

{_markdown_table(pooled, ['outcome', 'model_family', 'quantile', 'profile_block', 'metric', 'direct_estimate', 'conditional_estimate'])}

## Terminal stress and profile availability

{TERMINAL_CAVEAT}. Cold-start observations remain in-sample through the frozen
fallback and metadata rules. Any availability statistic is descriptive for this
terminal population and does not create a terminal evidence label.

### Regime context

{_markdown_table(terminal_context, ['metric', 'direct_estimate', 'direct_source_path', 'direct_source_sha256'])}

### Profile-label availability

{_markdown_table(terminal_availability, ['model_family', 'profile_block', 'representation', 'metric', 'direct_estimate', 'direct_profile_label_availability'])}

### Unchanged-model transfer

{_markdown_table(terminal_performance, ['outcome', 'model_family', 'quantile', 'profile_block', 'metric', 'direct_estimate', 'conditional_estimate', 'direct_profile_label_availability'])}

## Implementation deviation

{IMPLEMENTATION_DEVIATION}

## Frozen protected-source receipts

{_receipt_markdown(receipts)}

All quantitative text above is generated from persisted CSVs. No prior empirical
output, governance file, Results Registry entry, evidence ledger, or thesis prose
is modified by this reporter.
"""


def render_summary_zh(
    labels: pd.DataFrame,
    robustness: pd.DataFrame,
    receipts: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> str:
    breach = _direct_label_summary(labels, "breach_probability")
    severity = _direct_label_summary(labels, "conditional_positive_lateness")
    profiles = _profile_summary(labels)
    monthly_robustness = robustness.loc[
        robustness["period_summary"].eq("later_monthly_median")
    ]
    pooled = robustness.loc[robustness["period_summary"].eq("later_pooled")]
    terminal = robustness.loc[robustness["period_summary"].eq("terminal_stress")]
    terminal_context = terminal.loc[terminal["outcome"].eq("terminal_regime_context")]
    terminal_availability = terminal.loc[terminal["outcome"].eq("profile_availability")]
    terminal_performance = terminal.loc[
        ~terminal["outcome"].isin(["terminal_regime_context", "profile_availability"])
    ]
    return f"""# 直接承诺 + 已验证画像扩展 V1 — 结果摘要

## 范围与证据角色

主要操作估计量：**已发布承诺 → 已发布承诺 + 固定的已验证过程画像**。直接基线
仅包含 `promised_delivery_days`；超参数只在开发期的 DP0/DQ0 上选择，并在同一
直接梯度内共享。

次要稳健性估计量：**已发布承诺 + 购买时当前订单背景 → 已发布承诺 + 购买时
当前订单背景 + 固定的已验证过程画像**。其数值直接读取受保护的 Order V1
持久化结果。两组增量只作并列解释，不相减、不排名，也不根据增量大小选择估计量。

## 证据标签命名空间

RQ2 标签（`Strongly confirmed`、`Partially confirmed`、`Not confirmed`）描述画像
本身在锁定未来过程确认中的结果；RQ3 订单标签（`Supported`、`Mixed`、
`Not-supported`）描述特定订单信息梯度中的画像增量。两类标签不可互换。

### 固定的 RQ2 画像确认

{_markdown_table(profiles, ['profile_id', 'profile_block', 'evidence_label', 'n_months', 'months_both_improved', 'evidence_reason'])}

### 1–6 月直接违约概率标签

{_markdown_table(breach, ['model_family', 'profile_block', 'median_delta_log_loss', 'median_delta_brier', 'months_both_improved', 'high_support_guard', 'calibration_guard', 'score_contribution_guard', 'evidence_label', 'evidence_reason'])}

### 1–6 月直接条件正迟到严重度标签

{_markdown_table(severity, ['model_family', 'quantile', 'profile_block', 'median_pinball_skill', 'months_nonnegative_skill', 'high_support_guard', 'coverage_guard', 'evidence_label', 'evidence_reason'])}

## 直接估计量与当前背景估计量的稳健性对照

下表两列对应不同信息集对比，只作并列解释；它们不是双重差分，也不能根据相对
增量大小决定哪个估计量更优先。

{_markdown_table(monthly_robustness, ['outcome', 'model_family', 'quantile', 'profile_block', 'metric', 'direct_estimate', 'direct_evidence_label', 'conditional_estimate', 'conditional_evidence_label'], max_rows=60)}

## 直接梯度选定设置

{_markdown_table(model_selection, list(model_selection.columns)[:10], max_rows=30)}

## 后期合并边界与覆盖率

六个月合并结果与月度中位数严格分开。覆盖率是经验量，不是生产保证，也不能单独
证明分位数模型有技能。

{_markdown_table(pooled, ['outcome', 'model_family', 'quantile', 'profile_block', 'metric', 'direct_estimate', 'conditional_estimate'])}

## 终端压力与画像可用性

7–8 月仅为终端制度压力证据，不与 1–6 月合并，不参与模型、超参数、校准或标签
选择。冷启动订单按照冻结的回退值与元数据规则保留在样本内。终端期可用性统计只作
描述，不产生新的终端证据标签。

### 制度背景

{_markdown_table(terminal_context, ['metric', 'direct_estimate', 'direct_source_path', 'direct_source_sha256'])}

### 画像标签可用性

{_markdown_table(terminal_availability, ['model_family', 'profile_block', 'representation', 'metric', 'direct_estimate', 'direct_profile_label_availability'])}

### 未经改变的模型迁移

{_markdown_table(terminal_performance, ['outcome', 'model_family', 'quantile', 'profile_block', 'metric', 'direct_estimate', 'conditional_estimate', 'direct_profile_label_availability'])}

## 实现偏差记录

仓库中不存在单独的 Order V1 amendment。受保护的 Order V1 权威文件是
`ORDER_PROTOCOL.md`、`ORDER_FROZEN_CONFIG.json` 和
`ORDER_MODEL_SELECTION_FREEZE.json`；没有用无关的 Phase 2A amendment 替代。

## 冻结受保护来源收据

{_receipt_markdown(receipts)}

以上定量内容仅由持久化 CSV 生成。本报告器不修改任何既有实证输出、治理文件、
Results Registry、证据台账或论文正文。
"""


def render_readme(receipts: pd.DataFrame, evidence_created: bool) -> str:
    evidence_note = (
        "The modelling core did not provide EVIDENCE_LABELS.csv, so this reporter "
        "created it deterministically from persisted monthly results and frozen guards."
        if evidence_created
        else "The modelling core supplied EVIDENCE_LABELS.csv; this reporter read it without overwriting it."
    )
    return f"""# Direct Promise + Validated-Profile Extension V1

This directory is an isolated, authorised extension. It asks whether held-fixed,
already validated seller and state-OD profiles add predictive information beyond
the recorded issued promise itself.

## Information ladders

Primary: `{DIRECT_ESTIMAND}`. The baseline is exactly one feature,
`promised_delivery_days`, defined as fractional elapsed seconds from purchase
timestamp to the recorded estimated-delivery timestamp divided by 86,400.

Secondary robustness: `{CONDITIONAL_ESTIMAND}`. These protected Order V1 results
are read, not recomputed. `DIRECT_VS_CURRENT_CONTEXT_ROBUSTNESS.csv` contains no
cross-estimand subtraction, difference-in-differences, rank, winner, or selection
field.

## Generated reporting artifacts

- `EVIDENCE_LABELS.csv` — the 18 direct RQ3 order labels produced by the modelling
  core. Protected RQ2 confirmation labels are joined in memory for summaries and
  figure data, in an explicit non-interchangeable namespace. {evidence_note}
- `DIRECT_VS_CURRENT_CONTEXT_ROBUSTNESS.csv` — wide side-by-side primary/secondary
  estimands with source paths and SHA-256 receipts.
- `FIGURE_DATA.csv` — long plot-ready rows with estimand role and provenance.
- `RESULT_SUMMARY.md` and `RESULT_SUMMARY_EN.md` — identical persisted-value
  English summaries (the first is the frozen validator-facing filename).
- `RESULT_SUMMARY_ZH.md` — persisted-value Chinese summary.

## Evidence boundaries

- RQ2 profile confirmation labels do not become order-level evidence labels.
- January–June monthly medians and later-pooled estimates are separate estimands.
- {TERMINAL_CAVEAT}.
- Score-only is a signal sensitivity; metadata-only is a breach guard diagnostic.
- Current-context Order V1 remains verified secondary robustness and is not
  invalidated or reinterpreted.
- No business, causal, deployment, production-quality, promise-policy, or original
  Olist-model claim is licensed by these outputs.

## Implementation deviation

{IMPLEMENTATION_DEVIATION}

## Frozen source receipts

{_receipt_markdown(receipts)}

## Reproduction

After the modelling core has persisted all required direct CSVs, run:

```bash
python analysis/direct_promise_profile_extension_v1/scripts/direct_reporting.py
```

The reporter validates frozen protected-source hashes before writing. It does not
fit models, tune hyperparameters, change prior evidence, edit governance files, or
edit thesis prose.
"""


def write_reporting_outputs(
    extension_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, str]:
    """Write all deterministic reporting outputs and return their SHA-256 hashes."""

    module_path = Path(__file__).resolve()
    extension = (
        Path(extension_dir).resolve()
        if extension_dir is not None
        else module_path.parent.parent
    )
    repository = (
        Path(repo_root).resolve()
        if repo_root is not None
        else module_path.parents[3]
    )
    config = load_config(extension)
    receipts = validate_protected_sources(config, repository)
    tables, paths = load_reporting_inputs(extension, repository)
    labels, evidence_created = load_or_create_evidence_labels(
        extension, tables, paths, repository
    )
    receipts = pd.concat(
        [receipts, reporting_input_receipts(paths, repository)],
        ignore_index=True,
        sort=False,
    ).sort_values(["receipt_role", "source_id"], kind=SORT_KIND).reset_index(drop=True)
    robustness = build_robustness_table(tables, paths, labels, repository)
    figure_data = build_figure_data(robustness, labels)
    model_selection = _canonicalise(tables["model_selection"])

    targets = {
        "DIRECT_VS_CURRENT_CONTEXT_ROBUSTNESS.csv": robustness,
        "FIGURE_DATA.csv": figure_data,
    }
    for filename, frame in targets.items():
        columns = ROBUSTNESS_COLUMNS if filename.startswith("DIRECT_VS") else None
        _write_csv(frame, extension / filename, columns)
    english_summary = render_summary_en(labels, robustness, receipts, model_selection)
    _write_text(extension / "RESULT_SUMMARY.md", english_summary)
    _write_text(extension / "RESULT_SUMMARY_EN.md", english_summary)
    _write_text(
        extension / "RESULT_SUMMARY_ZH.md",
        render_summary_zh(labels, robustness, receipts, model_selection),
    )
    _write_text(extension / "README.md", render_readme(receipts, evidence_created))

    output_names = [
        "EVIDENCE_LABELS.csv",
        "DIRECT_VS_CURRENT_CONTEXT_ROBUSTNESS.csv",
        "FIGURE_DATA.csv",
        "RESULT_SUMMARY.md",
        "RESULT_SUMMARY_EN.md",
        "RESULT_SUMMARY_ZH.md",
        "README.md",
    ]
    return {
        name: sha256_file(extension / name)
        for name in output_names
        if (extension / name).is_file()
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=None,
        help="Direct extension directory (defaults to the script's parent directory).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (defaults to three parents above the script).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipts = write_reporting_outputs(args.extension_dir, args.repo_root)
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
