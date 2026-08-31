"""Reporting and final artifact validation for supplementary RQ1 V1.

All writes are confined to ``analysis/rq1_speed_reliability_review_v1``.
The module persists the frozen data/statistical tables, exactly three main
figures with one source CSV each, bilingual evidence summaries, the mechanical
extension recommendation, and the final reproducibility/validation receipts.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .rq1_io import (
        FIGURE_DIR,
        FIGURE_SOURCE_DIR,
        WORKING_DIR,
        WORKSPACE,
        ensure_workspace_dirs,
        sha256_file,
        table_receipt,
        write_csv,
        write_json,
        write_text,
    )
    from .rq1_preflight import PRESTATE_PATH, verify_protected_unchanged
except ImportError:  # pragma: no cover - direct-script fallback
    from rq1_io import (  # type: ignore
        FIGURE_DIR,
        FIGURE_SOURCE_DIR,
        WORKING_DIR,
        WORKSPACE,
        ensure_workspace_dirs,
        sha256_file,
        table_receipt,
        write_csv,
        write_json,
        write_text,
    )
    from rq1_preflight import PRESTATE_PATH, verify_protected_unchanged  # type: ignore


os.environ.setdefault("MPLCONFIGDIR", str(WORKING_DIR / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


CONFIG_PATH = WORKSPACE / "RQ1_SPEED_RELIABILITY_FROZEN_CONFIG.json"
REPORTING_STATE_PATH = WORKING_DIR / "REPORTING_STATE.json"
ANALYSIS_RECEIPTS_PATH = WORKING_DIR / "ANALYSIS_RECEIPTS.json"
MANIFEST_PATH = WORKSPACE / "RUN_MANIFEST.json"
VALIDATION_PATH = WORKSPACE / "ARTIFACT_VALIDATION_REPORT.md"
TEST_RESULTS_PATH = WORKSPACE / "TEST_RESULTS.txt"

DATA_TABLE_FILES = (
    "RQ1_SAMPLE_AUDIT.csv",
    "RQ1_DATE_IDENTITY_AUDIT.csv",
    "RQ1_REVIEW_COVERAGE.csv",
    "RQ1_DURATION_ERROR_CELL_COUNTS.csv",
    "RQ1_DURATION_ERROR_REVIEW_RATES.csv",
)
DURATION_DISTRIBUTIONS_FILE = "RQ1_DURATION_DISTRIBUTIONS.csv"
STAT_TABLE_FILES = {
    "model_results": "RQ1_MODEL_RESULTS.csv",
    "covariance": "RQ1_ROBUST_COVARIANCE.csv",
    "wald": "RQ1_ROBUST_WALD_TESTS.csv",
    "probabilities": "RQ1_ADJUSTED_PROBABILITIES.csv",
    "contrasts": "RQ1_ADJUSTED_CONTRASTS.csv",
    "comparison": "RQ1_CONTRAST_COMPARISON.csv",
    "low_review_3_sensitivity": "RQ1_LOW_REVIEW_3_SENSITIVITY.csv",
    "review_timing_sensitivity": "RQ1_REVIEW_TIMING_SENSITIVITY.csv",
    "duration_bin_sensitivity": "RQ1_DURATION_BIN_SENSITIVITY.csv",
}
FIGURE_FILES = (
    "01_absolute_duration_review.png",
    "02_duration_error_review_heatmap.png",
    "03_adjusted_speed_reliability_associations.png",
)
FIGURE_SOURCE_FILES = (
    "01_absolute_duration_review.csv",
    "02_duration_error_review_heatmap.csv",
    "03_adjusted_speed_reliability_associations.csv",
)
FIGURE_PAIRS = {
    FIGURE_FILES[index]: FIGURE_SOURCE_FILES[index] for index in range(3)
}
DECISION_LABELS = {
    "EXPAND_RQ1_TO_SPEED_AND_RELIABILITY",
    "RETAIN_SIGNED_ERROR_ONLY_RQ1",
    "ACTUAL_DURATION_ASSOCIATION_WITHOUT_INCREMENTAL_PROMISE_ERROR",
    "INCONCLUSIVE_RQ1_EXTENSION",
}
FIXED_PROVENANCE_WARNINGS = (
    "The raw payment and review CSV hashes had no prior independent Registry hash anchor; trust is bounded by the recorded current SHA-256 values, exact deterministic sample reproduction, and pre/post protected-state verification.",
    "Before formal preflight, a read-only in-memory technical-design smoke diagnosed the rank deficiency of an uncentred Patsy natural-cubic-regression-spline basis and exercised common-support mechanics. It persisted no empirical result and did not select the frozen specification using substantive outcomes; formal parameters and sources were frozen before the authorised run.",
    "Before formal preflight, one read-only in-memory 95,824-row integration smoke executed the already frozen model/contrast/decision rules and observed the provisional label EXPAND_RQ1_TO_SPEED_AND_RELIABILITY. It wrote no artifact and did not alter any model rule, threshold, contrast, or decision rubric. The exact ad-hoc command is unavailable and is not reconstructed; the formal run must independently repeat the analysis after the source freeze and complete protected-state preflight.",
    "The first formal-preflight invocation used the runner file path and exited before importing the project package with ModuleNotFoundError: analysis. It read no empirical input and wrote no formal receipt; the recorded formal invocation therefore uses the package module entry point instead.",
    "A first complete formal attempt passed 36/36 tests and 87/87 artifact checks, but its command receipts rendered package-module invocations as non-replayable runner file paths. That attempt is superseded for provenance only; no analytical rule or empirical result changed, and the complete chain was rerun after correcting command serialization.",
    "A second complete attempt recorded the exact internal pytest command but omitted the surrounding run_rq1_tests module invocation from the manifest command list. It also passed 36/36 tests and 87/87 artifact checks and is superseded for command provenance only; the final fresh chain records both commands.",
    "A third complete attempt recorded the test wrapper and internal pytest in RUN_STATE and the test receipt, but the manifest parser did not promote the internal pytest command into RUN_MANIFEST.test_command. It passed 36/36 tests and 87/87 artifact checks and is superseded for manifest provenance only.",
)
PRE_FORMAL_EXECUTION_DISCLOSURES = (
    {
        "stage": "technical_design_smoke_before_formal_preflight",
        "command": "unavailable_not_reconstructed",
        "scope": "read-only in-memory Patsy spline-rank and common-support mechanics",
        "persisted_artifact": False,
        "substantive_result_used_for_selection": False,
    },
    {
        "stage": "full_sample_integration_smoke_before_formal_preflight",
        "command": "unavailable_not_reconstructed",
        "sample_rows": 95_824,
        "scope": "read-only in-memory execution of already frozen model, contrast, and decision rules",
        "provisional_label_observed": "EXPAND_RQ1_TO_SPEED_AND_RELIABILITY",
        "persisted_artifact": False,
        "model_rule_threshold_contrast_or_rubric_changed": False,
        "formal_post_preflight_rerun_required": True,
    },
    {
        "stage": "failed_preflight_startup_before_import",
        "command": "env PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=.cache/rq1_mpl_formal .venv/bin/python -B analysis/rq1_speed_reliability_review_v1/scripts/run_rq1_speed_reliability.py --stage preflight --data-dir 'data/olist_data'",
        "return_code": 1,
        "failure": "ModuleNotFoundError: No module named 'analysis'",
        "empirical_input_read": False,
        "formal_receipt_written": False,
        "resolution": "invoke the unchanged runner through python -m from the repository root",
    },
    {
        "stage": "superseded_complete_attempt_with_inexact_command_serialization",
        "manifest_sha256": "3f1c28a6f5e62d97a535a6390619733737da4f3ede578a1f3d567d5060a2809f",
        "artifact_validation": "87_of_87_pass",
        "tests": "36_of_36_pass",
        "analytical_result_changed": False,
        "reason_superseded": "module invocations were serialized as runner file paths and were not replayable as recorded",
        "resolution": "correct command serialization and repeat fresh preflight, analysis, tests, and finalize",
    },
    {
        "stage": "superseded_complete_attempt_with_missing_test_wrapper_command",
        "manifest_sha256": "31b26b66cbedb6c1558f2498d5f5c8617bfdef0acbea77c970450a692e60a051",
        "artifact_validation": "87_of_87_pass",
        "tests": "36_of_36_pass",
        "analytical_result_changed": False,
        "reason_superseded": "the manifest recorded pytest but omitted the outer run_rq1_tests module invocation",
        "resolution": "record the wrapper command in RUN_STATE and the internal pytest command in the test receipt, then repeat the full chain",
    },
    {
        "stage": "superseded_complete_attempt_with_null_manifest_test_command",
        "manifest_sha256": "5745de4102cf073c60833935eb037016ea18a46e37ad365f20fb1a34105f1356",
        "artifact_validation": "87_of_87_pass",
        "tests": "36_of_36_pass",
        "analytical_result_changed": False,
        "reason_superseded": "RUN_MANIFEST.test_command was null although the wrapper and pytest commands existed in lower-level receipts",
        "resolution": "promote both commands through the strict test-receipt parser and add validator gates",
    },
)
REQUIRED_TOP_LEVEL_FILES = {
    "RQ1_SPEED_RELIABILITY_PROTOCOL.md",
    "RQ1_SPEED_RELIABILITY_FROZEN_CONFIG.json",
    "RQ1_MODEL_SPECIFICATIONS.md",
    *DATA_TABLE_FILES,
    DURATION_DISTRIBUTIONS_FILE,
    *STAT_TABLE_FILES.values(),
    "RQ1_RESULTS_SUMMARY.md",
    "RQ1_RESULTS_SUMMARY_ZH.md",
    "RQ1_EXTENSION_DECISION.md",
    "DATA_DICTIONARY.md",
    "RUN_MANIFEST.json",
    "TEST_RESULTS.txt",
    "ARTIFACT_VALIDATION_REPORT.md",
    "BLOCKERS.md",
}

_COLUMN_DESCRIPTIONS = {
    "order_id": "Olist order identifier used only for row-level audit receipts.",
    "actual_delivery_days": "Integer normalised customer-delivery date minus purchase date.",
    "promised_lead_days": "Integer normalised estimated-delivery date minus purchase date.",
    "promise_error_days": "Integer normalised actual-delivery date minus estimated-delivery date.",
    "actual_duration_group": "Frozen actual-delivery-duration group.",
    "promise_error_group": "Frozen eight-level signed promise-error group.",
    "reviewed_orders": "Orders with one deterministically selected usable review.",
    "analytical_orders": "Canonical delivered orders in the relevant cell.",
    "review_coverage": "Reviewed orders divided by analytical orders.",
    "low_review_2_orders": "Selected reviews with score at most two.",
    "low_review_2_rate": "Share of selected reviews with score at most two.",
    "low_review_2_ci_lower": "Lower endpoint of the 95% interval.",
    "low_review_2_ci_upper": "Upper endpoint of the 95% interval.",
    "low_support_cell": "True when the frozen reviewed-order support threshold is not met.",
    "model_id": "Frozen statistical model identifier.",
    "analysis_id": "Primary or sensitivity analysis identifier.",
    "estimate": "Model-based point estimate on the scale stated by the row.",
    "ci_lower": "Lower endpoint of the reported 95% interval.",
    "ci_upper": "Upper endpoint of the reported 95% interval.",
    "p_value": "HC1 robust test p-value where applicable.",
    "population": "Canonical delivered-order or deterministically reviewed-order population.",
    "variable": "Frozen calendar-day duration/error variable summarised by the row.",
    "p05": "Empirical fifth percentile using linear interpolation.",
    "p10": "Empirical tenth percentile using linear interpolation.",
    "p25": "Empirical twenty-fifth percentile using linear interpolation.",
    "p50": "Empirical median using linear interpolation.",
    "p75": "Empirical seventy-fifth percentile using linear interpolation.",
    "p90": "Empirical ninetieth percentile using linear interpolation.",
    "p95": "Empirical ninety-fifth percentile using linear interpolation.",
    "reference_duration_days": "Pair-specific supported reference duration used for this adjusted estimand, when applicable.",
    "support_rule": "Frozen deterministic support rule used for the estimand.",
    "support_note": "Reader-facing copy of the frozen support rule for figure-source audit.",
    "support_lower": "Lower endpoint of the supported continuous-duration region.",
    "support_upper": "Upper endpoint of the supported continuous-duration region.",
    "support_n": "Observed orders in the support population recorded for the estimand.",
    "estimand_id": "Deterministic adjusted-probability estimand identifier.",
    "contrast_id": "Deterministic adjusted risk-contrast identifier.",
    "fixed_settings_json": "Exact fixed covariate/reference settings for the simulated contrast.",
}


class ReportingError(RuntimeError):
    """Raised when reporting cannot satisfy the frozen artifact contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportingError(f"{label} must be a mapping")
    return dict(value)


def _diagnostics_receipt(value: Any) -> dict[str, Any]:
    """Return a JSON-safe receipt for either supported diagnostics schema."""

    if isinstance(value, pd.DataFrame):
        if value.empty:
            raise ReportingError("stats.diagnostics must not be an empty DataFrame")
        return {
            "format": "dataframe_records",
            "columns": [str(column) for column in value.columns],
            "rows": int(len(value)),
            "records": value.to_dict(orient="records"),
        }
    if isinstance(value, Mapping):
        return {"format": "mapping", "value": dict(value)}
    raise ReportingError("stats.diagnostics must be a mapping or pandas DataFrame")


def _require_frame(value: Any, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ReportingError(f"{label} must be a pandas DataFrame")
    if value.empty:
        raise ReportingError(f"{label} must not be empty")
    return value.copy()


def _first_column(
    frame: pd.DataFrame, candidates: Sequence[str], *, required: bool = True
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ReportingError(
            f"None of the required columns {list(candidates)} exists; "
            f"available={list(frame.columns)}"
        )
    return None


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _write_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "RQ1 Speed and Promise-Reliability Review Analysis V1"},
    )
    plt.close(figure)


def _build_figure_1_source(
    all_orders: pd.DataFrame,
    reviewed: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    labels = list(config["actual_duration_groups"]["labels"])
    rows: list[dict[str, Any]] = []
    for label in labels:
        all_subset = all_orders.loc[
            all_orders["actual_duration_group"].astype("string").eq(label)
        ]
        reviewed_subset = reviewed.loc[
            reviewed["actual_duration_group"].astype("string").eq(label)
        ]
        total = int(len(all_subset))
        n_reviewed = int(len(reviewed_subset))
        low = int(reviewed_subset["low_review_2"].sum())
        lower, upper = _wilson(low, n_reviewed)
        rows.append(
            {
                "actual_duration_group": label,
                "analytical_orders": total,
                "reviewed_orders": n_reviewed,
                "review_coverage": n_reviewed / total if total else np.nan,
                "low_review_2_orders": low,
                "low_review_2_rate": low / n_reviewed if n_reviewed else np.nan,
                "low_review_2_ci_lower": lower,
                "low_review_2_ci_upper": upper,
                "mean_review_score": reviewed_subset["review_score"].mean(),
                "median_review_score": reviewed_subset["review_score"].median(),
                "one_star_share": reviewed_subset["one_star"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot_figure_1(source: pd.DataFrame) -> None:
    labels = source["actual_duration_group"].astype(str).tolist()
    positions = np.arange(len(source))
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    axes[0].bar(positions, source["reviewed_orders"], color="#4C78A8")
    axes[0].set_title("Reviewed-order counts")
    axes[0].set_ylabel("Orders")
    axes[0].grid(axis="y", alpha=0.22)

    estimate = source["low_review_2_rate"].to_numpy(float)
    lower = source["low_review_2_ci_lower"].to_numpy(float)
    upper = source["low_review_2_ci_upper"].to_numpy(float)
    axes[1].errorbar(
        positions,
        estimate * 100,
        yerr=np.vstack(((estimate - lower) * 100, (upper - estimate) * 100)),
        fmt="o-",
        color="#D55E00",
        capsize=3,
        linewidth=1.5,
    )
    axes[1].set_title("Observed low-review rate")
    axes[1].set_ylabel("Review score ≤2 (%)")
    axes[1].grid(axis="y", alpha=0.22)

    axes[2].plot(
        positions,
        source["review_coverage"].to_numpy(float) * 100,
        "o-",
        color="#009E73",
    )
    axes[2].set_title("Review coverage")
    axes[2].set_ylabel("Canonical orders reviewed (%)")
    axes[2].grid(axis="y", alpha=0.22)

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Absolute delivery duration and observed review outcomes", y=1.02)
    figure.text(
        0.5,
        -0.04,
        "Observed reviews are selected; intervals are 95% Wilson intervals. "
        "These descriptive associations are observational, not causal.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout()
    _write_figure(FIGURE_DIR / FIGURE_FILES[0], figure)


def _plot_figure_2(source: pd.DataFrame, config: Mapping[str, Any]) -> None:
    duration_labels = list(config["actual_duration_groups"]["labels"])
    error_labels = list(config["promise_error_groups"]["labels"])
    rate = source.pivot(
        index="actual_duration_group",
        columns="promise_error_group",
        values="low_review_2_rate",
    ).reindex(index=duration_labels, columns=error_labels)
    count = source.pivot(
        index="actual_duration_group",
        columns="promise_error_group",
        values="reviewed_orders",
    ).reindex(index=duration_labels, columns=error_labels)
    low_support = source.pivot(
        index="actual_duration_group",
        columns="promise_error_group",
        values="low_support_cell",
    ).reindex(index=duration_labels, columns=error_labels)

    display = rate.to_numpy(float) * 100
    muted = low_support.fillna(True).to_numpy(bool)
    plotted = display.copy()
    plotted[muted] = np.nan
    figure, axis = plt.subplots(figsize=(13.2, 5.0))
    image = axis.imshow(plotted, cmap="YlOrRd", vmin=0, vmax=85, aspect="auto")
    for row in range(len(duration_labels)):
        for column in range(len(error_labels)):
            n_orders = int(count.iloc[row, column]) if pd.notna(count.iloc[row, column]) else 0
            value = display[row, column]
            if muted[row, column]:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="#E6E6E6",
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
            annotation = "NA" if not np.isfinite(value) else f"{value:.1f}%\nn={n_orders:,}"
            if muted[row, column]:
                annotation += "\nlow support"
            axis.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                fontsize=7.0,
                color="#555555" if muted[row, column] else "black",
            )
    axis.set_xticks(np.arange(len(error_labels)), error_labels, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(duration_labels)), duration_labels)
    axis.set_xlabel("Signed promise-error group")
    axis.set_ylabel("Actual delivery-duration group")
    axis.set_title("Observed low-review rate by actual duration and promise error")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Review score ≤2 (%)")
    figure.text(
        0.5,
        -0.05,
        "Descriptive and observational; no purchase-month adjustment. "
        "Cells with fewer than 50 reviewed orders are retained and muted.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout()
    _write_figure(FIGURE_DIR / FIGURE_FILES[1], figure)


def _model_filter(frame: pd.DataFrame, model: str) -> pd.Series:
    column = _first_column(frame, ("model_id", "model", "specification"))
    values = frame[column].astype(str).str.lower().str.replace(" ", "_", regex=False)
    aliases = {
        "a": {"a", "model_a", "primary_model_a"},
        "b": {"b", "model_b", "primary_model_b"},
        "c": {"c", "model_c", "secondary_model_c"},
    }[model.lower()]
    return (
        values.isin(aliases)
        | values.str.contains(f"model_{model.lower()}", regex=False)
        | values.str.startswith(f"{model.lower()}_")
    )


def _probability_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    return {
        "estimate": _first_column(
            frame,
            ("adjusted_probability", "predicted_probability", "probability", "estimate"),
        ),
        "lower": _first_column(
            frame,
            ("ci_lower", "probability_ci_lower", "adjusted_probability_ci_lower", "lower"),
        ),
        "upper": _first_column(
            frame,
            ("ci_upper", "probability_ci_upper", "adjusted_probability_ci_upper", "upper"),
        ),
        "duration": _first_column(
            frame,
            (
                "actual_delivery_days",
                "duration_days",
                "reference_duration_days",
                "continuous_value",
                "x_value",
            ),
            required=False,
        ),
        "group": _first_column(
            frame,
            ("promise_error_group", "promise_error_group_label", "error_group"),
            required=False,
        ),
        "row_type": _first_column(
            frame,
            ("estimand_type", "probability_type", "row_type", "scenario_type"),
            required=False,
        ),
    }


def _long_contrast_frame(
    comparison: pd.DataFrame, contrasts: pd.DataFrame
) -> pd.DataFrame:
    for candidate in (comparison, contrasts):
        name_column = _first_column(
            candidate,
            ("contrast_id", "contrast", "comparison_id", "estimand"),
            required=False,
        )
        estimate_column = _first_column(
            candidate,
            ("estimate", "risk_difference", "point_estimate"),
            required=False,
        )
        lower_column = _first_column(
            candidate,
            ("ci_lower", "q025", "simulation_ci_lower", "lower"),
            required=False,
        )
        upper_column = _first_column(
            candidate,
            ("ci_upper", "q975", "simulation_ci_upper", "upper"),
            required=False,
        )
        if all((name_column, estimate_column, lower_column, upper_column)):
            result = candidate.copy()
            result = result.rename(
                columns={
                    name_column: "contrast_id",
                    estimate_column: "estimate",
                    lower_column: "ci_lower",
                    upper_column: "ci_upper",
                }
            )
            return result
    raise ReportingError("Contrast tables do not expose a supported long-form schema")


def _build_figure_3_source(
    probabilities: pd.DataFrame,
    contrasts: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    columns = _probability_columns(probabilities)
    estimate = str(columns["estimate"])
    lower = str(columns["lower"])
    upper = str(columns["upper"])
    duration = columns["duration"]
    group = columns["group"]
    row_type = columns["row_type"]

    model_a = probabilities.loc[_model_filter(probabilities, "a")].copy()
    if "variant" in model_a:
        model_a = model_a.loc[model_a["variant"].astype(str).eq("primary")]
    if row_type is not None:
        curve_mask = model_a[row_type].astype(str).str.lower().str.contains("curve")
        if curve_mask.any():
            model_a = model_a.loc[curve_mask]
    if duration is None:
        raise ReportingError("Model-A probability rows lack an actual-duration column")
    model_a = model_a.loc[pd.to_numeric(model_a[duration], errors="coerce").notna()]
    curve = pd.DataFrame(
        {
            "panel": "model_a_duration_curve",
            "x_numeric": pd.to_numeric(model_a[duration], errors="coerce"),
            "x_label": model_a[duration].astype(str),
            "estimate": pd.to_numeric(model_a[estimate], errors="coerce"),
            "ci_lower": pd.to_numeric(model_a[lower], errors="coerce"),
            "ci_upper": pd.to_numeric(model_a[upper], errors="coerce"),
            "model_id": model_a["model_id"].astype(str),
            "estimand_id": model_a.get("estimand_id", ""),
            "contrast_id": "",
            "error_group": "",
            "reference_duration_days": pd.to_numeric(
                model_a[duration], errors="coerce"
            ),
            "support_rule": model_a.get("support_rule", ""),
            "support_note": model_a.get("support_rule", ""),
            "support_lower": model_a.get("support_lower", np.nan),
            "support_upper": model_a.get("support_upper", np.nan),
            "support_n": model_a.get("support_n", np.nan),
            "fixed_settings_json": "",
        }
    ).dropna(subset=["x_numeric", "estimate", "ci_lower", "ci_upper"])
    curve = curve.sort_values("x_numeric", kind="mergesort")
    if curve.empty:
        raise ReportingError("No usable Model-A duration-curve rows")

    model_b = probabilities.loc[_model_filter(probabilities, "b")].copy()
    if "variant" in model_b:
        model_b = model_b.loc[model_b["variant"].astype(str).eq("primary")]
    if group is None:
        raise ReportingError("Model-B probability rows lack a promise-error group")
    model_b = model_b.loc[model_b[group].notna()]
    if row_type is not None:
        group_mask = model_b[row_type].astype(str).str.lower().str.contains("group")
        if group_mask.any():
            model_b = model_b.loc[group_mask]
    model_b = model_b.reset_index(drop=True)
    group_rows = pd.DataFrame(
        {
            "panel": "model_b_error_groups",
            "x_numeric": np.arange(len(model_b), dtype=float),
            "x_label": model_b[group].astype(str).to_numpy(),
            "estimate": pd.to_numeric(model_b[estimate], errors="coerce").to_numpy(),
            "ci_lower": pd.to_numeric(model_b[lower], errors="coerce").to_numpy(),
            "ci_upper": pd.to_numeric(model_b[upper], errors="coerce").to_numpy(),
            "model_id": model_b["model_id"].astype(str).to_numpy(),
            "estimand_id": model_b.get(
                "estimand_id", pd.Series("", index=model_b.index)
            ).astype(str).to_numpy(),
            "contrast_id": "",
            "error_group": model_b[group].astype(str).to_numpy(),
            "reference_duration_days": pd.to_numeric(
                model_b[duration], errors="coerce"
            ).to_numpy(),
            "support_rule": model_b.get(
                "support_rule", pd.Series("", index=model_b.index)
            ).astype(str).to_numpy(),
            "support_note": model_b.get(
                "support_rule", pd.Series("", index=model_b.index)
            ).astype(str).to_numpy(),
            "support_lower": pd.to_numeric(
                model_b.get("support_lower", pd.Series(np.nan, index=model_b.index)),
                errors="coerce",
            ).to_numpy(),
            "support_upper": pd.to_numeric(
                model_b.get("support_upper", pd.Series(np.nan, index=model_b.index)),
                errors="coerce",
            ).to_numpy(),
            "support_n": pd.to_numeric(
                model_b.get("support_n", pd.Series(np.nan, index=model_b.index)),
                errors="coerce",
            ).to_numpy(),
            "fixed_settings_json": "",
        }
    ).dropna(subset=["estimate", "ci_lower", "ci_upper"])
    if group_rows.empty:
        raise ReportingError("No usable Model-B error-group probability rows")

    contrast = _long_contrast_frame(comparison, contrasts)
    required_contrasts = (
        "C_speed",
        "C_late_4_7",
        "C_difference_4_7_minus_speed",
    )
    selected = contrast.loc[contrast["contrast_id"].isin(required_contrasts)].copy()
    if selected.empty:
        normalised_name = contrast["contrast_id"].astype(str).str.lower()
        wanted = np.logical_or.reduce(
            [
                normalised_name.eq("c_speed"),
                normalised_name.str.contains("late_4_7", regex=False),
                normalised_name.str.contains("4_7_minus_speed", regex=False),
            ]
        )
        selected = contrast.loc[wanted].copy()
    selected["_order"] = pd.Categorical(
        selected["contrast_id"], categories=list(required_contrasts), ordered=True
    )
    selected = selected.sort_values("_order", kind="mergesort").drop(columns="_order")
    fixed_json = selected.get(
        "fixed_settings_json", pd.Series("", index=selected.index)
    ).astype(str)

    def fixed_value(payload: str, key: str) -> Any:
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return np.nan
        return parsed.get(key, np.nan) if isinstance(parsed, Mapping) else np.nan

    fixed_reference = fixed_json.map(lambda value: fixed_value(value, "D_reference"))
    fixed_support_lower = fixed_json.map(lambda value: fixed_value(value, "support_lower"))
    fixed_support_upper = fixed_json.map(lambda value: fixed_value(value, "support_upper"))
    fixed_support_n = fixed_json.map(lambda value: fixed_value(value, "support_n"))
    selected_reference = pd.to_numeric(
        selected.get("reference_value", pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    ).fillna(pd.to_numeric(fixed_reference, errors="coerce"))
    selected_support_lower = pd.to_numeric(
        selected.get("support_lower", pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    ).fillna(pd.to_numeric(fixed_support_lower, errors="coerce"))
    selected_support_upper = pd.to_numeric(
        selected.get("support_upper", pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    ).fillna(pd.to_numeric(fixed_support_upper, errors="coerce"))
    selected_support_n = pd.to_numeric(
        selected.get("support_n", pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    ).fillna(pd.to_numeric(fixed_support_n, errors="coerce"))
    selected_support_rule = selected.get(
        "support_rule", pd.Series("", index=selected.index)
    ).astype(str)
    selected_support_rule = selected_support_rule.mask(
        selected_support_rule.eq("") | selected_support_rule.eq("nan"),
        "fixed settings and pair-specific support recorded in fixed_settings_json",
    )
    contrast_rows = pd.DataFrame(
        {
            "panel": "primary_contrasts",
            "x_numeric": np.arange(len(selected), dtype=float),
            "x_label": selected["contrast_id"].astype(str).to_numpy(),
            "estimate": pd.to_numeric(selected["estimate"], errors="coerce").to_numpy(),
            "ci_lower": pd.to_numeric(selected["ci_lower"], errors="coerce").to_numpy(),
            "ci_upper": pd.to_numeric(selected["ci_upper"], errors="coerce").to_numpy(),
            "model_id": selected.get(
                "model_id", pd.Series("", index=selected.index)
            ).astype(str).to_numpy(),
            "estimand_id": "",
            "contrast_id": selected["contrast_id"].astype(str).to_numpy(),
            "error_group": selected.get(
                "error_group", pd.Series("", index=selected.index)
            ).astype(str).to_numpy(),
            "reference_duration_days": selected_reference.to_numpy(),
            "support_rule": selected_support_rule.to_numpy(),
            "support_note": selected_support_rule.to_numpy(),
            "support_lower": selected_support_lower.to_numpy(),
            "support_upper": selected_support_upper.to_numpy(),
            "support_n": selected_support_n.to_numpy(),
            "fixed_settings_json": fixed_json.to_numpy(),
        }
    ).dropna(subset=["estimate", "ci_lower", "ci_upper"])
    if contrast_rows.empty:
        raise ReportingError("No usable primary contrast rows")
    return pd.concat([curve, group_rows, contrast_rows], ignore_index=True)


def _plot_figure_3(source: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.25))
    curve = source.loc[source["panel"].eq("model_a_duration_curve")]
    axes[0].plot(curve["x_numeric"], curve["estimate"] * 100, color="#4C78A8")
    axes[0].fill_between(
        curve["x_numeric"].to_numpy(float),
        curve["ci_lower"].to_numpy(float) * 100,
        curve["ci_upper"].to_numpy(float) * 100,
        color="#4C78A8",
        alpha=0.2,
    )
    axes[0].set_title("A. Actual-duration association")
    axes[0].set_xlabel("Actual delivery duration (calendar days)")
    axes[0].set_ylabel("Adjusted low-review probability (%)")

    groups = source.loc[source["panel"].eq("model_b_error_groups")].reset_index(drop=True)
    positions = np.arange(len(groups))
    axes[1].errorbar(
        positions,
        groups["estimate"].to_numpy(float) * 100,
        yerr=np.vstack(
            (
                (groups["estimate"] - groups["ci_lower"]).to_numpy(float) * 100,
                (groups["ci_upper"] - groups["estimate"]).to_numpy(float) * 100,
            )
        ),
        fmt="o",
        color="#D55E00",
        capsize=3,
    )
    axes[1].set_xticks(positions, groups["x_label"], rotation=30, ha="right")
    axes[1].set_title("B. Promise groups at pair-supported Dref")
    axes[1].set_ylabel("Adjusted low-review probability (%)")

    contrasts = source.loc[source["panel"].eq("primary_contrasts")].reset_index(drop=True)
    positions = np.arange(len(contrasts))
    axes[2].axhline(0, color="0.45", linewidth=0.8)
    axes[2].errorbar(
        positions,
        contrasts["estimate"].to_numpy(float) * 100,
        yerr=np.vstack(
            (
                (contrasts["estimate"] - contrasts["ci_lower"]).to_numpy(float) * 100,
                (contrasts["ci_upper"] - contrasts["estimate"]).to_numpy(float) * 100,
            )
        ),
        fmt="o",
        color="#009E73",
        capsize=3,
    )
    axes[2].set_xticks(positions, contrasts["x_label"], rotation=25, ha="right")
    axes[2].set_title("C. Pre-specified risk contrasts")
    axes[2].set_ylabel("Risk difference (percentage points)")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Adjusted speed and promise-relative review associations", y=1.02)
    figure.text(
        0.5,
        -0.06,
        "Model-based adjusted associations with 95% intervals; panel B uses "
        "pair-specific supported reference durations recorded in the source CSV. "
        "Reviews and timing are selected; estimates are observational and non-causal.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout()
    _write_figure(FIGURE_DIR / FIGURE_FILES[2], figure)


def _decision_label(decision: Mapping[str, Any]) -> str:
    for key in ("label", "decision_label", "extension_label", "assigned_label"):
        if key in decision:
            label = str(decision[key])
            if label in DECISION_LABELS:
                return label
    raise ReportingError(f"Decision mapping lacks one frozen label: {dict(decision)}")


def _decision_wording(label: str) -> tuple[str, str]:
    allowed = {
        "EXPAND_RQ1_TO_SPEED_AND_RELIABILITY": (
            "How are actual delivery duration and performance relative to the promised date associated with observed order reviews?",
            "Both pre-specified observational association branches met their frozen support rules.",
        ),
        "RETAIN_SIGNED_ERROR_ONLY_RQ1": (
            "How is signed delivery-promise error—including breach incidence and positive-lateness severity—associated with observed customer review outcomes?",
            "The promise-relative branch met its frozen rule, while the actual-duration branch did not.",
        ),
        "ACTUAL_DURATION_ASSOCIATION_WITHOUT_INCREMENTAL_PROMISE_ERROR": (
            "Actual delivery duration was associated with observed reviews, but the existing signed-error headline requires qualification after duration adjustment.",
            "The actual-duration branch met its rule, while incremental promise-relative support did not.",
        ),
        "INCONCLUSIVE_RQ1_EXTENSION": (
            "The supplementary analysis does not justify expanding the current RQ1 wording.",
            "Neither branch met its complete frozen rule or a key conclusion was sensitivity/support dependent.",
        ),
    }
    return allowed[label]


def _flatten_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_scalars(value[key], child))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, value))
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _distribution_receipt(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for column in ("actual_delivery_days", "promised_lead_days", "promise_error_days"):
        values = pd.to_numeric(frame[column], errors="raise")
        result[column] = {
            "n": int(values.notna().sum()),
            "min": float(values.min()),
            "p10": float(values.quantile(0.10, interpolation="linear")),
            "p25": float(values.quantile(0.25, interpolation="linear")),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75, interpolation="linear")),
            "p90": float(values.quantile(0.90, interpolation="linear")),
            "max": float(values.max()),
        }
    return result


def _duration_distribution_table(
    all_orders: pd.DataFrame, reviewed: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, frame in (("canonical", all_orders), ("reviewed", reviewed)):
        for variable in (
            "actual_delivery_days",
            "promised_lead_days",
            "promise_error_days",
        ):
            values = pd.to_numeric(frame[variable], errors="raise")
            rows.append(
                {
                    "population": population,
                    "variable": variable,
                    "n": int(values.notna().sum()),
                    "min": float(values.min()),
                    "p05": float(values.quantile(0.05, interpolation="linear")),
                    "p10": float(values.quantile(0.10, interpolation="linear")),
                    "p25": float(values.quantile(0.25, interpolation="linear")),
                    "p50": float(values.quantile(0.50, interpolation="linear")),
                    "p75": float(values.quantile(0.75, interpolation="linear")),
                    "p90": float(values.quantile(0.90, interpolation="linear")),
                    "p95": float(values.quantile(0.95, interpolation="linear")),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def _primary_model_rows(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = frame.loc[_model_filter(frame, model)].copy()
    if "variant" in rows:
        rows = rows.loc[rows["variant"].astype(str).eq("primary")]
    return rows


def _one_result(
    frame: pd.DataFrame, mask: pd.Series, description: str
) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ReportingError(
            f"Expected exactly one {description} row; found {len(rows)}"
        )
    return rows.iloc[0]


def _p_value(value: Any) -> str:
    number = float(value)
    return f"{number:.3g}" if number >= 0.001 else f"{number:.2e}"


def _risk_interval(row: Mapping[str, Any], lower: str, upper: str) -> str:
    return (
        f"{100 * float(row['estimate']):.2f} pp "
        f"[95% CI {100 * float(row[lower]):.2f}, {100 * float(row[upper]):.2f}]"
    )


def _result_evidence_markdown(
    *,
    chinese: bool,
    audit: Mapping[str, Any],
    decision: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> str:
    """Render the pre-specified numerical findings from persisted result rows."""

    wald = tables["RQ1_ROBUST_WALD_TESTS.csv"]
    contrasts = tables["RQ1_ADJUSTED_CONTRASTS.csv"]
    probabilities = tables["RQ1_ADJUSTED_PROBABILITIES.csv"]
    comparison = tables["RQ1_CONTRAST_COMPARISON.csv"]

    def wald_row(model: str, block: str) -> pd.Series:
        rows = _primary_model_rows(wald, model)
        block_column = _first_column(rows, ("block", "term_block", "test_block"))
        return _one_result(
            rows,
            rows[block_column].astype(str).eq(block),
            f"Model {model.upper()} {block} Wald",
        )

    def contrast_row(model: str, contrast_id: str) -> pd.Series:
        rows = _primary_model_rows(contrasts, model)
        return _one_result(
            rows,
            rows["contrast_id"].astype(str).eq(contrast_id),
            f"Model {model.upper()} {contrast_id} contrast",
        )

    a_wald = wald_row("a", "duration_spline")
    b_wald = wald_row("b", "error_group")
    c_wald = wald_row("c", "error_group")
    a_p75_p25 = contrast_row("a", "model_a_full_sample_p75_minus_p25")

    late_groups = ("2-3 days late", "4-7 days late", ">=8 days late")
    b_group_rows = [
        contrast_row("b", f"error_group_rd::{group}") for group in late_groups
    ]
    c_group_rows = [
        contrast_row("c", f"error_group_rd::{group}") for group in late_groups
    ]

    primary_comparison = _primary_model_rows(comparison, "b")
    comparison_by_id: dict[str, pd.Series] = {}
    for contrast_id in (
        "C_speed",
        "C_late_4_7",
        "C_late_8_plus",
        "C_difference_4_7_minus_speed",
    ):
        comparison_by_id[contrast_id] = _one_result(
            primary_comparison,
            primary_comparison["contrast_id"].astype(str).eq(contrast_id),
            f"primary simulated {contrast_id}",
        )

    percentile_rows = _primary_model_rows(probabilities, "a")
    if "estimand_type" in percentile_rows:
        percentile_rows = percentile_rows.loc[
            percentile_rows["estimand_type"].astype(str).eq("duration_percentile")
        ]
    percentile_rows = percentile_rows.sort_values("continuous_value", kind="mergesort")
    percentile_text = "; ".join(
        f"{row.estimand_id.replace('duration_', '').upper()} D={float(row.continuous_value):g}: "
        f"{100 * float(row.estimate):.2f}% [{100 * float(row.ci_lower):.2f}, "
        f"{100 * float(row.ci_upper):.2f}]"
        for row in percentile_rows.itertuples(index=False)
    )

    rates = tables["RQ1_DURATION_ERROR_REVIEW_RATES.csv"].copy()
    supported = rates.loc[~rates["low_support_cell"].astype(bool)].copy()
    if supported.empty:
        raise ReportingError("Every duration-by-error descriptive cell is low support")
    highest = supported.loc[supported["low_review_2_rate"].astype(float).idxmax()]
    lowest = supported.loc[supported["low_review_2_rate"].astype(float).idxmin()]
    descriptive = (
        f"highest supported cell={highest['actual_duration_group']} × "
        f"{highest['promise_error_group']} ({100*float(highest['low_review_2_rate']):.2f}%, "
        f"n={int(highest['reviewed_orders']):,}); lowest supported cell="
        f"{lowest['actual_duration_group']} × {lowest['promise_error_group']} "
        f"({100*float(lowest['low_review_2_rate']):.2f}%, "
        f"n={int(lowest['reviewed_orders']):,}); low-support cells="
        f"{int(rates['low_support_cell'].astype(bool).sum())}/40"
    )

    sensitivity_lines: list[str] = []
    for filename, label in (
        ("RQ1_LOW_REVIEW_3_SENSITIVITY.csv", "review score <=3"),
        ("RQ1_REVIEW_TIMING_SENSITIVITY.csv", "reviews at/after delivery"),
        ("RQ1_DURATION_BIN_SENSITIVITY.csv", "fixed duration bins"),
    ):
        frame = tables[filename]
        rows = frame.loc[
            frame["record_type"].astype(str).eq("contrast")
            & _model_filter(frame, "b")
            & frame["record_id"].astype(str).isin(
                ["C_speed", "C_late_4_7", "C_late_8_plus"]
            )
        ].copy()
        if rows.empty:
            raise ReportingError(f"{filename} lacks the required Model-B contrasts")
        values = "; ".join(
            f"{row.record_id}={_risk_interval(row._asdict(), 'ci_lower', 'ci_upper')}"
            for row in rows.itertuples(index=False)
        )
        sensitivity_lines.append(f"- {label}: {values}.")

    def settings_suffix(row: pd.Series) -> str:
        try:
            settings = json.loads(str(row.get("fixed_settings_json", "")))
        except json.JSONDecodeError:
            settings = {}
        if not isinstance(settings, Mapping) or not settings:
            return ""
        selected = {
            key: settings[key]
            for key in (
                "D_low",
                "D_high",
                "D_reference",
                "group",
                "reference_group",
                "support_lower",
                "support_upper",
                "support_n",
            )
            if key in settings
        }
        return f" Fixed settings: `{json.dumps(selected, sort_keys=True, ensure_ascii=False)}`."

    comparison_lines = "\n".join(
        f"- `{contrast_id}`: {_risk_interval(row, 'q025', 'q975')}."
        f"{settings_suffix(row)}"
        for contrast_id, row in comparison_by_id.items()
    )
    b_lines = "\n".join(
        f"- {group}: {_risk_interval(row, 'ci_lower', 'ci_upper')}."
        for group, row in zip(late_groups, b_group_rows, strict=True)
    )
    c_lines = "\n".join(
        f"- {group}: {_risk_interval(row, 'ci_lower', 'ci_upper')}."
        for group, row in zip(late_groups, c_group_rows, strict=True)
    )
    comparison_supported = bool(
        float(comparison_by_id["C_difference_4_7_minus_speed"]["q025"]) > 0
    )
    comparison_wording = (
        "The pre-specified 4–7-day promise-relative contrast was larger than the "
        "pre-specified P25-to-P75 absolute-speed contrast."
        if comparison_supported
        else "The analysis does not establish that the pre-specified promise-relative contrast is larger than the pre-specified absolute-speed contrast."
    )
    decision_components = "\n".join(
        f"- `{key}`: `{_format_value(value)}`"
        for key, value in _flatten_scalars(decision)
    )
    distributions = _as_mapping(
        audit.get("duration_distributions", {}), "audit.duration_distributions"
    )
    distribution_lines = "\n".join(
        f"- `{name}`: min={values['min']:g}, p10={values['p10']:g}, "
        f"p25={values['p25']:g}, median={values['median']:g}, "
        f"p75={values['p75']:g}, p90={values['p90']:g}, max={values['max']:g}."
        for name, values in distributions.items()
    )
    if chinese:
        return f"""## 冻结结果明细

### 时长分布（reviewed 样本，日历日）

{distribution_lines}

### 两维描述表

{descriptive}。这是观察性描述，未控制 purchase month；低支持 cell 不用于单独实质性结论。

### Model A：绝对送达时长

- 时长 spline 联合 robust Wald：chi-square={float(a_wald['wald_chi2']):.3f}，df={int(a_wald['df'])}，p={_p_value(a_wald['p_value'])}。
- 全样本 P75-minus-P25 风险差：{_risk_interval(a_p75_p25, 'ci_lower', 'ci_upper')}。
- 调整后分位点概率（95% CI）：{percentile_text}。

### Model B：控制实际时长后的承诺相对表现

- error-group block 联合 robust Wald：chi-square={float(b_wald['wald_chi2']):.3f}，df={int(b_wald['df'])}，p={_p_value(b_wald['p_value'])}。
{b_lines}

同一模型化实际时长下，不同 signed error 同时意味着不同 promised lead；这些是调整后的观察性关联，不能称为因果 expectation-violation effect。

### Model C：displayed-speed 分解（次要）

- error-group block 联合 robust Wald：chi-square={float(c_wald['wald_chi2']):.3f}，df={int(c_wald['df'])}，p={_p_value(c_wald['p_value'])}。
{c_lines}

### 预设 Model-B 风险对比（10,000 次 HC1 系数模拟）

{comparison_lines}

{comparison_wording}

### 敏感性

{chr(10).join(sensitivity_lines)}

### 冻结决策组件

{decision_components}
"""
    return f"""## Frozen numerical findings

### Duration distributions (reviewed sample; calendar days)

{distribution_lines}

### Two-way descriptive table

{descriptive}. This is observational and unadjusted for purchase month; low-support cells do not support standalone substantive claims.

### Model A: absolute delivery duration

- Joint robust duration-spline Wald: chi-square={float(a_wald['wald_chi2']):.3f}, df={int(a_wald['df'])}, p={_p_value(a_wald['p_value'])}.
- Full-sample P75-minus-P25 risk difference: {_risk_interval(a_p75_p25, 'ci_lower', 'ci_upper')}.
- Adjusted percentile probabilities (95% CI): {percentile_text}.

### Model B: promise-relative performance conditional on actual duration

- Joint robust error-group Wald: chi-square={float(b_wald['wald_chi2']):.3f}, df={int(b_wald['df'])}, p={_p_value(b_wald['p_value'])}.
{b_lines}

At a common modelled actual duration, different signed-error groups also imply different promised lead times. These are adjusted observational associations, not causal expectation-violation effects.

### Model C: displayed-speed decomposition (secondary)

- Joint robust error-group Wald: chi-square={float(c_wald['wald_chi2']):.3f}, df={int(c_wald['df'])}, p={_p_value(c_wald['p_value'])}.
{c_lines}

### Pre-specified Model-B risk contrasts (10,000 HC1 coefficient draws)

{comparison_lines}

{comparison_wording}

### Sensitivities

{chr(10).join(sensitivity_lines)}

### Frozen decision components

{decision_components}
"""


def _summary_markdown(
    *,
    chinese: bool,
    audit: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    decision: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> str:
    label = _decision_label(decision)
    wording, rationale = _decision_wording(label)
    sample = _as_mapping(audit.get("sample", {}), "audit.sample")
    review = _as_mapping(audit.get("review_selection", {}), "audit.review_selection")
    date = _as_mapping(audit.get("date_identity", {}), "audit.date_identity")
    if isinstance(diagnostics, pd.DataFrame):
        primary_diagnostics = diagnostics.copy()
        if "variant" in primary_diagnostics:
            primary_diagnostics = primary_diagnostics.loc[
                primary_diagnostics["variant"].astype(str).eq("primary")
            ]
        diagnostic_lines = "\n".join(
            "- "
            + ", ".join(
                f"`{key}`=`{_format_value(row[key])}`"
                for key in (
                    "model_id",
                    "n_orders",
                    "converged",
                    "design_rank",
                    "design_columns",
                    "design_condition_number",
                    "covariance_type",
                    "hc1_cholesky_passed",
                )
                if key in row.index
            )
            for _, row in primary_diagnostics.iterrows()
        ) or "- No primary diagnostic row was supplied."
    else:
        diagnostic_rows = _flatten_scalars(diagnostics)
        diagnostic_lines = "\n".join(
            f"- `{key}`: `{_format_value(value)}`" for key, value in diagnostic_rows
        ) or "- No scalar diagnostic was supplied."
    table_lines = "\n".join(
        f"- `{name}`: {len(frame):,} rows, {len(frame.columns)} columns"
        for name, frame in sorted(tables.items())
    )
    evidence = _result_evidence_markdown(
        chinese=chinese, audit=audit, decision=decision, tables=tables
    )
    canonical_n = int(sample.get("canonical_delivered_orders", 0))
    reviewed_n = int(sample.get("reviewed_orders", 0))
    review_coverage = reviewed_n / canonical_n if canonical_n else float("nan")
    if chinese:
        return f"""# RQ1 速度与承诺可靠性补充分析结果摘要

## 样本与恒等式

- 规范 delivered orders：{int(sample.get('canonical_delivered_orders', 0)):,}。
- 含确定性选择 review 的 orders：{int(sample.get('reviewed_orders', 0)):,}。
- 整体 review coverage：{100 * review_coverage:.2f}%（{reviewed_n:,}/{canonical_n:,}）。
- review 时间在实际送达时间戳之前：{int(review.get('reviews_before_delivery', 0)):,}；时间敏感性保留 {int(review.get('reviews_at_or_after_delivery', 0)):,}。
- 实际时长范围：{date.get('actual_delivery_days_min')} 至 {date.get('actual_delivery_days_max')} 个日历日；承诺 lead 范围：{date.get('promised_lead_days_min')} 至 {date.get('promised_lead_days_max')}。
- `D=P+E` 失败数：{int(date.get('identity_failures', 0)):,}。

{evidence}

## 冻结统计诊断

{diagnostic_lines}

## 机械决策标签

`{label}`

理由：{rationale}

建议的未来表述：{wording}

这只是后续治理审查的建议，不会自动修改当前 RQ1、Registry、Ledger 或论文。所有结果都是模型化的观察性关联。Review 提交与时间具有选择性，review 也可能反映产品、seller interaction 与整体购买体验；本分析不识别因果、心理预期违背或 business-policy 效应。

## 已持久化统计表

{table_lines}
"""
    return f"""# RQ1 Speed and Promise-Reliability Results Summary

## Sample and date identity

- Canonical delivered orders: {int(sample.get('canonical_delivered_orders', 0)):,}.
- Orders with one deterministically selected usable review: {int(sample.get('reviewed_orders', 0)):,}.
- Overall review coverage: {100 * review_coverage:.2f}% ({reviewed_n:,}/{canonical_n:,}).
- Selected reviews before the recorded delivery timestamp: {int(review.get('reviews_before_delivery', 0)):,}; timing sensitivity retained {int(review.get('reviews_at_or_after_delivery', 0)):,}.
- Actual-duration range: {date.get('actual_delivery_days_min')} to {date.get('actual_delivery_days_max')} calendar days; promised-lead range: {date.get('promised_lead_days_min')} to {date.get('promised_lead_days_max')}.
- Failures of `D=P+E`: {int(date.get('identity_failures', 0)):,}.

{evidence}

## Frozen statistical diagnostics

{diagnostic_lines}

## Mechanical extension label

`{label}`

Rationale: {rationale}

Recommended future wording: {wording}

This is a recommendation for a later governance review; it does not automatically change RQ1, the Registry, the Ledger, or thesis prose. All estimates are model-based observational associations. Review submission and timing are selected, and reviews may reflect product quality, seller interaction, and the broader order experience. The analysis does not identify causality, psychological expectation violation, or business-policy effects.

## Persisted statistical tables

{table_lines}
"""


def _decision_markdown(decision: Mapping[str, Any]) -> str:
    label = _decision_label(decision)
    wording, rationale = _decision_wording(label)
    checklist = "\n".join(
        f"- `{key}`: `{_format_value(value)}`"
        for key, value in _flatten_scalars(decision)
        if key not in {"label", "decision_label", "extension_label", "assigned_label"}
    )
    return f"""# RQ1 Extension Decision

Assigned descriptive manuscript-direction label:

`{label}`

## Frozen-rule receipt

{checklist or '- No additional scalar decision receipt was supplied.'}

## Recommendation

{rationale}

{wording}

This label is an observational manuscript-direction recommendation only. It does not establish causality and does not automatically update the current RQ, title, governance controls, evidence architecture, Registry, Ledger, or thesis prose.

## Interpretation boundaries

Allowed only when supported by the persisted intervals and frozen rules:

- longer actual delivery duration is associated with observed low-review probability;
- signed lateness retains an adjusted association with observed reviews;
- one pre-specified promise-relative risk contrast is larger or smaller than one pre-specified absolute-speed contrast; and
- the two dimensions provide complementary observational information.

Forbidden wording:

- actual duration or promise breach caused poor reviews;
- customers generally care more about breach than speed;
- changing a promise improves reviews, conversion, retention, or sales;
- review is a pure logistics-satisfaction measure;
- the analysis identifies psychological expectation violation; or
- the public Olist extract is representative of the full platform.
"""


def _blockers_markdown(blockers: Sequence[Any]) -> str:
    if not blockers:
        return "# Blockers\n\nNone.\n"
    lines = "\n".join(f"- {item}" for item in blockers)
    return f"# Blockers\n\n{lines}\n"


def _data_dictionary(tables: Mapping[str, pd.DataFrame]) -> str:
    sections = [
        "# Data Dictionary",
        "",
        "All durations use the frozen normalised calendar-date convention. Model outputs are observational associations, not causal effects.",
        "",
    ]
    for filename, frame in sorted(tables.items()):
        sections.extend(
            [
                f"## `{filename}`",
                "",
                f"Rows: {len(frame):,}; columns: {len(frame.columns)}.",
                "",
                "| Column | Persisted dtype | Definition |",
                "|---|---|---|",
            ]
        )
        for column in frame.columns:
            description = _COLUMN_DESCRIPTIONS.get(
                str(column),
                "See the frozen protocol/model specification and source table context.",
            )
            sections.append(f"| `{column}` | `{frame[column].dtype}` | {description} |")
        sections.append("")
    return "\n".join(sections)


def _artifact_paths(*, include_manifest: bool = False) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(WORKSPACE.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "__init__.py":
            continue
        if path == MANIFEST_PATH and not include_manifest:
            continue
        paths.append(path)
    for directory in (FIGURE_DIR, FIGURE_SOURCE_DIR):
        if directory.is_dir():
            paths.extend(sorted(item for item in directory.iterdir() if item.is_file()))
    return paths


def _inventory(*, include_manifest: bool = False) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(WORKSPACE)): table_receipt(path)
        for path in _artifact_paths(include_manifest=include_manifest)
    }


def write_analysis_artifacts(
    all_orders: pd.DataFrame,
    reviewed: pd.DataFrame,
    data_tables: Mapping[str, pd.DataFrame],
    statistics: Mapping[str, Any],
    audit: Mapping[str, Any],
    prestate: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Persist all non-manifest analysis artifacts in the new workspace."""

    ensure_workspace_dirs()
    config = _load_config() if config is None else _as_mapping(config, "config")
    if prestate.get("status") != "passed":
        raise ReportingError("Reporting requires a passed pre-execution state")
    if len(all_orders) != 96_470 or len(reviewed) != 95_824:
        raise ReportingError("Reporting sample does not match the frozen population")
    report_audit = dict(audit)
    report_audit["duration_distributions"] = _distribution_receipt(reviewed)

    persisted_tables: dict[str, pd.DataFrame] = {}
    for filename in DATA_TABLE_FILES:
        if filename not in data_tables:
            raise ReportingError(f"Missing required data table: {filename}")
        frame = _require_frame(data_tables[filename], filename)
        write_csv(WORKSPACE / filename, frame)
        persisted_tables[filename] = frame
    duration_distributions = _duration_distribution_table(all_orders, reviewed)
    write_csv(WORKSPACE / DURATION_DISTRIBUTIONS_FILE, duration_distributions)
    persisted_tables[DURATION_DISTRIBUTIONS_FILE] = duration_distributions

    stats_mapping = _as_mapping(statistics, "statistics")
    for key, filename in STAT_TABLE_FILES.items():
        if key not in stats_mapping:
            raise ReportingError(f"Missing required statistical result: {key}")
        frame = _require_frame(stats_mapping[key], key)
        write_csv(WORKSPACE / filename, frame)
        persisted_tables[filename] = frame
    diagnostics = stats_mapping.get("diagnostics", {})
    diagnostics_receipt = _diagnostics_receipt(diagnostics)
    decision = _as_mapping(stats_mapping.get("decision", {}), "stats.decision")
    label = _decision_label(decision)

    figure_1_source = _build_figure_1_source(all_orders, reviewed, config)
    figure_2_source = persisted_tables["RQ1_DURATION_ERROR_REVIEW_RATES.csv"].copy()
    figure_3_source = _build_figure_3_source(
        persisted_tables["RQ1_ADJUSTED_PROBABILITIES.csv"],
        persisted_tables["RQ1_ADJUSTED_CONTRASTS.csv"],
        persisted_tables["RQ1_CONTRAST_COMPARISON.csv"],
    )
    figure_sources = (figure_1_source, figure_2_source, figure_3_source)
    for filename, frame in zip(FIGURE_SOURCE_FILES, figure_sources, strict=True):
        write_csv(FIGURE_SOURCE_DIR / filename, frame)
        persisted_tables[f"figure_sources/{filename}"] = frame
    _plot_figure_1(figure_1_source)
    _plot_figure_2(figure_2_source, config)
    _plot_figure_3(figure_3_source)

    blockers = list(stats_mapping.get("blockers", []))
    warnings = [*FIXED_PROVENANCE_WARNINGS, *list(stats_mapping.get("warnings", []))]
    write_text(
        WORKSPACE / "RQ1_RESULTS_SUMMARY.md",
        _summary_markdown(
            chinese=False,
            audit=report_audit,
            diagnostics=diagnostics,
            decision=decision,
            tables=persisted_tables,
        ),
    )
    write_text(
        WORKSPACE / "RQ1_RESULTS_SUMMARY_ZH.md",
        _summary_markdown(
            chinese=True,
            audit=report_audit,
            diagnostics=diagnostics,
            decision=decision,
            tables=persisted_tables,
        ),
    )
    write_text(WORKSPACE / "RQ1_EXTENSION_DECISION.md", _decision_markdown(decision))
    write_text(WORKSPACE / "DATA_DICTIONARY.md", _data_dictionary(persisted_tables))
    write_text(WORKSPACE / "BLOCKERS.md", _blockers_markdown(blockers))

    state = {
        "schema_version": 1,
        "analysis_id": "RQ1_SPEED_RELIABILITY_REVIEW_V1",
        "written_at_utc": _utc_now(),
        "sample": report_audit.get("sample", {}),
        "review_selection": report_audit.get("review_selection", {}),
        "date_identity": report_audit.get("date_identity", {}),
        "duration_distributions": report_audit.get("duration_distributions", {}),
        "month_pooling": report_audit.get("month_pooling", {}),
        "canonical_assembler": report_audit.get("canonical_assembler", {}),
        "legacy_reconciliation": report_audit.get("legacy_reconciliation", {}),
        "diagnostics": diagnostics_receipt,
        "decision": decision,
        "assigned_extension_label": label,
        "warnings": warnings,
        "pre_formal_execution_disclosures": list(PRE_FORMAL_EXECUTION_DISCLOSURES),
        "blockers": blockers,
        "prestate_path": str(PRESTATE_PATH),
        "prestate_sha256": sha256_file(PRESTATE_PATH) if PRESTATE_PATH.is_file() else None,
        "figure_pairs": {
            f"figures/{figure}": f"figure_sources/{source}"
            for figure, source in FIGURE_PAIRS.items()
        },
        "artifact_receipts_before_tests": {},
    }
    # The runner/test boundary consumes this explicit pre-test receipt.  Keep a
    # separate reporting state for finalisation so neither tests nor the
    # manifest finaliser need to infer completion from top-level outputs.
    receipts = _inventory()
    state["artifact_receipts_before_tests"] = receipts
    write_json(REPORTING_STATE_PATH, state)
    write_json(ANALYSIS_RECEIPTS_PATH, receipts)
    return receipts


def _parse_test_results(text: str) -> dict[str, Any]:
    receipt_text = text.split("--- pytest output ---", maxsplit=1)[0].strip()
    try:
        receipt = json.loads(receipt_text)
    except json.JSONDecodeError:
        receipt = {}
    keys = ("collected", "passed", "failed", "skipped", "deselected", "errors")
    counts: dict[str, int] = {}
    for key in keys:
        value = receipt.get(key, -1)
        counts[key] = int(value) if isinstance(value, (int, np.integer)) else -1
    return_code = receipt.get("return_code")
    return_code = int(return_code) if isinstance(return_code, (int, np.integer)) else -1
    command_argv = receipt.get("command")
    if not (
        isinstance(command_argv, list)
        and command_argv
        and all(isinstance(value, str) for value in command_argv)
    ):
        command_argv = []
    pytest_command = shlex.join(command_argv) if command_argv else ""
    wrapper_command = receipt.get("wrapper_command")
    wrapper_command = wrapper_command if isinstance(wrapper_command, str) else ""
    passed = counts["passed"]
    collected = counts["collected"]
    clean = all(counts[key] == 0 for key in ("failed", "skipped", "deselected", "errors"))
    return {
        **counts,
        "return_code": return_code,
        "wrapper_command": wrapper_command,
        "pytest_command": pytest_command,
        "pytest_command_argv": command_argv,
        "passed_tests": passed,
        "failed_tests": counts["failed"],
        "error_tests": counts["errors"],
        "minimum_required_tests": 30,
        "receipt_json_parsed": bool(receipt),
        "passed": bool(
            receipt
            and return_code == 0
            and collected >= 30
            and passed == collected
            and clean
        ),
    }


def _read_decision_label() -> str:
    text = (WORKSPACE / "RQ1_EXTENSION_DECISION.md").read_text(encoding="utf-8")
    matches = [label for label in DECISION_LABELS if label in text]
    if len(matches) != 1:
        raise ReportingError(f"Decision file does not contain exactly one label: {matches}")
    return matches[0]


def _check(
    checks: list[dict[str, str]], check_id: str, condition: bool, detail: str
) -> None:
    checks.append(
        {"check_id": check_id, "status": "PASS" if condition else "FAIL", "detail": detail}
    )


def _validate_artifacts(
    *, test_audit: Mapping[str, Any], protected_audit: Mapping[str, Any]
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    existing = {path.name for path in WORKSPACE.iterdir() if path.is_file()}
    for filename in sorted(REQUIRED_TOP_LEVEL_FILES):
        _check(checks, f"required:{filename}", filename in existing, "Required top-level artifact")

    figures = sorted(path.name for path in FIGURE_DIR.glob("*.png"))
    sources = sorted(path.name for path in FIGURE_SOURCE_DIR.glob("*.csv"))
    _check(checks, "figures:exact_three", figures == sorted(FIGURE_FILES), str(figures))
    _check(checks, "figure_sources:exact_three", sources == sorted(FIGURE_SOURCE_FILES), str(sources))
    for figure, source in FIGURE_PAIRS.items():
        figure_path = FIGURE_DIR / figure
        source_path = FIGURE_SOURCE_DIR / source
        _check(
            checks,
            f"figure_pair:{figure}",
            figure_path.is_file()
            and figure_path.stat().st_size > 10_000
            and source_path.is_file()
            and len(pd.read_csv(source_path)) > 0,
            f"{figure} <- {source}",
        )
    if len(sources) == 3:
        source_one = pd.read_csv(FIGURE_SOURCE_DIR / FIGURE_SOURCE_FILES[0])
        source_two = pd.read_csv(FIGURE_SOURCE_DIR / FIGURE_SOURCE_FILES[1])
        source_three = pd.read_csv(FIGURE_SOURCE_DIR / FIGURE_SOURCE_FILES[2])
        _check(
            checks,
            "figure_source:one_five_duration_groups",
            len(source_one) == 5
            and {
                "reviewed_orders",
                "low_review_2_rate",
                "low_review_2_ci_lower",
                "low_review_2_ci_upper",
                "review_coverage",
            }.issubset(source_one.columns),
            f"rows={len(source_one)}",
        )
        _check(
            checks,
            "figure_source:two_complete_matrix",
            len(source_two) == 40
            and {"reviewed_orders", "low_review_2_rate", "low_support_cell"}.issubset(
                source_two.columns
            ),
            f"rows={len(source_two)}",
        )
        required_three_columns = {
            "panel",
            "estimate",
            "ci_lower",
            "ci_upper",
            "reference_duration_days",
            "support_rule",
            "support_note",
            "support_lower",
            "support_upper",
            "support_n",
            "estimand_id",
            "contrast_id",
            "fixed_settings_json",
        }
        panel_counts = source_three["panel"].value_counts().to_dict() if "panel" in source_three else {}
        _check(
            checks,
            "figure_source:three_estimand_metadata",
            required_three_columns.issubset(source_three.columns)
            and int(panel_counts.get("model_a_duration_curve", 0)) > 1
            and int(panel_counts.get("model_b_error_groups", 0)) == 8
            and int(panel_counts.get("primary_contrasts", 0)) == 3,
            f"columns={sorted(source_three.columns)}; panels={panel_counts}",
        )

    sample_path = WORKSPACE / "RQ1_SAMPLE_AUDIT.csv"
    sample = pd.read_csv(sample_path) if sample_path.is_file() else pd.DataFrame()
    _check(
        checks,
        "sample_audit:all_pass",
        not sample.empty and "status" in sample and sample["status"].eq("PASS").all(),
        "All sample-reconciliation rows PASS",
    )
    date_path = WORKSPACE / "RQ1_DATE_IDENTITY_AUDIT.csv"
    date = pd.read_csv(date_path) if date_path.is_file() else pd.DataFrame()
    _check(checks, "date_identity:rows", len(date) == 96_470, f"rows={len(date)}")
    _check(
        checks,
        "date_identity:holds",
        not date.empty
        and date.get("identity_holds", pd.Series(False, index=date.index)).astype(bool).all()
        and not date.get("missing_component", pd.Series(True, index=date.index)).astype(bool).any(),
        "D=P+E and no missing component",
    )
    rates_path = WORKSPACE / "RQ1_DURATION_ERROR_REVIEW_RATES.csv"
    rates = pd.read_csv(rates_path) if rates_path.is_file() else pd.DataFrame()
    _check(checks, "duration_error:40_cells", len(rates) == 40, f"rows={len(rates)}")
    distribution_path = WORKSPACE / DURATION_DISTRIBUTIONS_FILE
    distributions = (
        pd.read_csv(distribution_path) if distribution_path.is_file() else pd.DataFrame()
    )
    expected_distribution_pairs = {
        (population, variable)
        for population in ("canonical", "reviewed")
        for variable in (
            "actual_delivery_days",
            "promised_lead_days",
            "promise_error_days",
        )
    }
    observed_distribution_pairs = (
        set(zip(distributions["population"], distributions["variable"]))
        if {"population", "variable"}.issubset(distributions.columns)
        else set()
    )
    _check(
        checks,
        "duration_distributions:complete",
        len(distributions) == 6
        and observed_distribution_pairs == expected_distribution_pairs,
        f"rows={len(distributions)}; pairs={sorted(observed_distribution_pairs)}",
    )
    for filename in STAT_TABLE_FILES.values():
        path = WORKSPACE / filename
        rows = len(pd.read_csv(path)) if path.is_file() else 0
        _check(checks, f"stats:{filename}", rows > 0, f"rows={rows}")

    try:
        label = _read_decision_label()
        label_valid = label in DECISION_LABELS
    except ReportingError:
        label = "invalid"
        label_valid = False
    _check(checks, "decision:one_frozen_label", label_valid, label)
    for filename in ("RQ1_RESULTS_SUMMARY.md", "RQ1_RESULTS_SUMMARY_ZH.md"):
        path = WORKSPACE / filename
        text = path.read_text(encoding="utf-8").lower() if path.is_file() else ""
        _check(
            checks,
            f"boundary:{filename}",
            ("observational" in text or "观察性" in text)
            and ("causal" in text or "因果" in text),
            "Observational/non-causal boundary present",
        )
    blockers_text = (WORKSPACE / "BLOCKERS.md").read_text(encoding="utf-8")
    _check(checks, "blockers:none", "None." in blockers_text, blockers_text.strip())
    _check(checks, "tests:minimum_30", bool(test_audit.get("passed")), str(dict(test_audit)))
    _check(
        checks,
        "tests:wrapper_command_recorded",
        bool(test_audit.get("wrapper_command")),
        str(test_audit.get("wrapper_command")),
    )
    _check(
        checks,
        "tests:pytest_command_recorded",
        bool(test_audit.get("pytest_command")),
        str(test_audit.get("pytest_command")),
    )
    _check(
        checks,
        "protected:byte_unchanged",
        bool(protected_audit.get("passed")),
        str(protected_audit.get("preservation_verdict")),
    )
    manifest_valid = False
    manifest_inventory: dict[str, Any] = {}
    if MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest_valid = (
                manifest.get("analysis_id") == "RQ1_SPEED_RELIABILITY_REVIEW_V1"
                and manifest.get("self_hash_policy", {}).get("run_manifest_sha256_in_manifest")
                == "excluded_impossible_self_reference"
            )
            raw_inventory = manifest.get("output_inventory", {})
            if isinstance(raw_inventory, Mapping):
                manifest_inventory = dict(raw_inventory)
        except json.JSONDecodeError:
            manifest_valid = False
    _check(checks, "manifest:schema", manifest_valid, "Manifest JSON and self-reference exception")
    current_inventory = _inventory()
    _check(
        checks,
        "manifest:inventory_paths",
        set(manifest_inventory) == set(current_inventory),
        f"manifest={len(manifest_inventory)} current={len(current_inventory)}",
    )
    for relative_path in sorted(set(manifest_inventory) | set(current_inventory)):
        recorded = manifest_inventory.get(relative_path, {})
        current = current_inventory.get(relative_path, {})
        _check(
            checks,
            f"manifest:inventory:{relative_path}",
            recorded == current,
            f"SHA-256/bytes/schema receipt for {relative_path}",
        )
    return checks


def _validation_markdown(checks: Sequence[Mapping[str, str]]) -> str:
    failed = [check for check in checks if check["status"] != "PASS"]
    lines = [
        "# Artifact Validation Report",
        "",
        f"Overall: **{'PASS' if not failed else 'FAIL'}**",
        "",
        f"Checks: {len(checks) - len(failed)}/{len(checks)} PASS.",
        "",
        "The final manifest is excluded from its own SHA-256 inventory because a file cannot contain its final self-hash. The manifest is written after this report is hashed, then re-read and validated without mutating this report.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{check['check_id']}` | {check['status']} | {detail} |")
    return "\n".join(lines) + "\n"


def _build_manifest(
    *,
    state: Mapping[str, Any],
    prestate: Mapping[str, Any],
    protected_audit: Mapping[str, Any],
    test_audit: Mapping[str, Any],
    commands: Sequence[str],
    output_inventory: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    config = _load_config()
    protected_paths = _as_mapping(
        protected_audit.get("protected_paths", {}), "protected_audit.protected_paths"
    )
    protected_after = _as_mapping(protected_paths.get("after", {}), "protected after")
    return {
        "schema_version": 1,
        "analysis_id": "RQ1_SPEED_RELIABILITY_REVIEW_V1",
        "task_status": status,
        "completed_at_utc": _utc_now(),
        "authorised_scope": config["authorised_scope"],
        "repository_preflight": prestate.get("repository", {}),
        "environment": prestate.get("environment", {}),
        "commands": list(commands),
        "test_command": test_audit.get("pytest_command"),
        "test_result": dict(test_audit),
        "source_input_audit": prestate.get("source_input_audit", {}),
        "raw_file_hashes": prestate.get("raw_file_hashes", {}),
        "raw_file_paths": prestate.get("raw_file_paths", {}),
        "source_code_hashes": prestate.get("local_frozen_input_hashes", {}),
        "existing_rq1_output_hashes": prestate.get("existing_rq1_output_hashes", {}),
        "registry_manifest_assembler_chain": prestate.get(
            "registry_manifest_assembler_chain", {}
        ),
        "protected_hashes_before": prestate.get("protected_hashes", {}),
        "protected_hashes_after": protected_after.get("hashes", {}),
        "protected_preservation": {
            key: protected_audit.get(key)
            for key in (
                "passed",
                "preservation_verdict",
                "trusted_inputs_unchanged",
                "local_frozen_inputs_unchanged",
                "repository_unchanged_outside_workspace",
            )
        },
        "sample_counts": state.get("sample", {}),
        "review_selection_counts": state.get("review_selection", {}),
        "date_identity_audit": state.get("date_identity", {}),
        "duration_distributions": state.get("duration_distributions", {}),
        "model_formulas": config["models"],
        "reference_categories": {
            "promise_error_group": config["promise_error_groups"]["reference"],
            "purchase_month": config["models"]["month_reference"],
            "duration_bin": config["actual_duration_groups"]["labels"][0],
        },
        "spline_degrees_of_freedom": config["models"]["spline_df"],
        "purchase_month_pooling_rule": {
            "minimum_rows": config["models"]["month_min_orders"],
            "pooled_label": config["models"]["sparse_month_label"],
            "primary_audit": state.get("month_pooling", {}),
        },
        "covariance_estimator": config["models"]["covariance"],
        "contrast_definitions": config["contrasts"],
        "simulation_seed": config["contrasts"]["simulation_seed"],
        "simulation_draws": config["contrasts"]["simulation_draws"],
        "assigned_rq1_extension_label": state.get("assigned_extension_label"),
        "statistical_diagnostics": state.get("diagnostics", {}),
        "warnings": state.get("warnings", []),
        "pre_formal_execution_disclosures": state.get(
            "pre_formal_execution_disclosures", []
        ),
        "blockers": state.get("blockers", []),
        "figure_pairs": state.get("figure_pairs", {}),
        "output_inventory": dict(output_inventory),
        "self_hash_policy": {
            "run_manifest_sha256_in_manifest": "excluded_impossible_self_reference",
            "explanation": (
                "All other final deliverables, including the validation report, are "
                "hashed in output_inventory. RUN_MANIFEST.json is hashed only after "
                "its final write and returned by finalize_manifest."
            ),
        },
        "scope_confirmation": {
            "profile_model_run": False,
            "breach_model_run": False,
            "severity_model_run": False,
            "weather_branch_run": False,
            "customer_profile_run": False,
            "governance_file_changed": False,
            "thesis_rewrite_run": False,
        },
    }


def finalize_manifest(
    test_results_path: str | Path,
    commands: Sequence[str],
) -> dict[str, Any]:
    """Verify preservation and write the final manifest/report fixed point."""

    if not REPORTING_STATE_PATH.is_file():
        raise ReportingError("REPORTING_STATE.json is missing; write artifacts first")
    state = json.loads(REPORTING_STATE_PATH.read_text(encoding="utf-8"))
    prestate = json.loads(PRESTATE_PATH.read_text(encoding="utf-8"))
    source_test_path = Path(test_results_path)
    if not source_test_path.is_file():
        raise ReportingError(f"Test results file is missing: {source_test_path}")
    test_text = source_test_path.read_text(encoding="utf-8")
    test_audit = _parse_test_results(test_text)
    write_text(TEST_RESULTS_PATH, test_text)
    if not test_audit["passed"]:
        raise ReportingError(f"Required deterministic tests did not pass: {test_audit}")
    if state.get("blockers"):
        raise ReportingError(f"Analysis blockers prevent completion: {state['blockers']}")

    # This is intentionally after empirical execution and the full test suite.
    protected_audit = verify_protected_unchanged(prestate)
    command_list = [str(command) for command in commands]

    # Seed the validation path before the first inventory.  This deterministic
    # placeholder is replaced immediately after the draft manifest exists; it
    # prevents a fresh-workspace run from failing its own required-file check.
    write_text(
        VALIDATION_PATH,
        "# Artifact Validation Report\n\nOverall: **PENDING FIXED-POINT VALIDATION**\n",
    )
    draft = _build_manifest(
        state=state,
        prestate=prestate,
        protected_audit=protected_audit,
        test_audit=test_audit,
        commands=command_list,
        output_inventory=_inventory(),
        status="completed_pending_validation_fixed_point",
    )
    write_json(MANIFEST_PATH, draft)
    initial_checks = _validate_artifacts(
        test_audit=test_audit, protected_audit=protected_audit
    )
    write_text(VALIDATION_PATH, _validation_markdown(initial_checks))

    initial_failed = [check for check in initial_checks if check["status"] != "PASS"]
    final_status = "completed" if not initial_failed else "blocked_validation_failed"
    final_manifest = _build_manifest(
        state=state,
        prestate=prestate,
        protected_audit=protected_audit,
        test_audit=test_audit,
        commands=command_list,
        output_inventory=_inventory(),
        status=final_status,
    )
    write_json(MANIFEST_PATH, final_manifest)
    final_checks = _validate_artifacts(
        test_audit=test_audit, protected_audit=protected_audit
    )
    final_failed = [check for check in final_checks if check["status"] != "PASS"]
    if final_failed:
        write_text(VALIDATION_PATH, _validation_markdown(final_checks))
        failed_manifest = _build_manifest(
            state=state,
            prestate=prestate,
            protected_audit=protected_audit,
            test_audit=test_audit,
            commands=command_list,
            output_inventory=_inventory(),
            status="blocked_validation_failed",
        )
        write_json(MANIFEST_PATH, failed_manifest)
        raise ReportingError(f"Artifact validation failed: {final_failed}")
    return {
        "overall_pass": True,
        "manifest": final_manifest,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "validation_path": str(VALIDATION_PATH),
        "validation_sha256": sha256_file(VALIDATION_PATH),
        "validation_checks_passed": len(final_checks),
        "validation_checks_failed": 0,
    }


__all__ = [
    "ReportingError",
    "finalize_manifest",
    "write_analysis_artifacts",
]
