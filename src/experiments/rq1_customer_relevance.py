"""Persist the descriptive and observational evidence required for RQ1.

This is not a predictive modelling experiment.  The only fitted specification
is a pre-specified binomial association sensitivity:

    low_review_2 ~ promise_error_group + purchase_month

No classification, quantile, mixed-effects, weather or adjustment-policy model
is fitted here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlds_thesis_matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.data.olist import load_order_level_data
from src.data.reviews import (
    REVIEW_COLUMNS,
    load_review_records,
    select_latest_usable_review,
)
from src.features.targets import build_targets

ERROR_GROUP_LABELS = [
    "very early: <= -14 days",
    "early: -13 to -7 days",
    "slightly early: -6 to -1 days",
    "on promised date",
    "1 day late",
    "2-3 days late",
    "4-7 days late",
    ">=8 days late",
]
ERROR_GROUP_EDGES = [-np.inf, -14, -7, -1, 0, 1, 3, 7, np.inf]
MIN_REVIEWED_GROUP_SIZE = 100
TERMINAL_CAUTION_MONTH = "2018-05"
MIN_CALENDAR_LEVEL_SIZE = 500
CALENDAR_REFERENCE_MONTH = "2018-01"
SPARSE_CALENDAR_LABEL = "pooled sparse purchase months (<500 orders each)"


def promise_error_groups(values: pd.Series) -> pd.Categorical:
    """Apply the pre-specified, interpretable RQ1 promise-error groups."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=ERROR_GROUP_EDGES,
        labels=ERROR_GROUP_LABELS,
        right=True,
        include_lowest=True,
        ordered=True,
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def promise_error_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    errors = frame["promise_error_days"].dropna().astype(float)
    if len(errors) != len(frame):
        raise AssertionError("The delivered analytical sample contains missing promise errors.")
    quantiles = errors.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return pd.DataFrame([{
        "sample_size": len(errors),
        "mean": errors.mean(),
        "median": errors.median(),
        "standard_deviation_sample": errors.std(ddof=1),
        "minimum": errors.min(),
        "p05": quantiles.loc[0.05],
        "p10": quantiles.loc[0.10],
        "p25": quantiles.loc[0.25],
        "p50": quantiles.loc[0.50],
        "p75": quantiles.loc[0.75],
        "p90": quantiles.loc[0.90],
        "p95": quantiles.loc[0.95],
        "p99": quantiles.loc[0.99],
        "maximum": errors.max(),
        "proportion_early": errors.lt(0).mean(),
        "proportion_on_date": errors.eq(0).mean(),
        "proportion_late": errors.gt(0).mean(),
        "proportion_severe_late_2d": errors.ge(2).mean(),
    }])


def promise_error_tail_counts(frame: pd.DataFrame) -> pd.DataFrame:
    errors = frame["promise_error_days"].astype(float)
    definitions = [
        ("very_early_le_minus_14d", "promise_error_days <= -14", errors.le(-14)),
        ("extreme_early_le_minus_30d", "promise_error_days <= -30", errors.le(-30)),
        ("extreme_early_le_minus_60d", "promise_error_days <= -60", errors.le(-60)),
        ("late_ge_8d", "promise_error_days >= 8", errors.ge(8)),
        ("late_ge_14d", "promise_error_days >= 14", errors.ge(14)),
        ("extreme_late_ge_30d", "promise_error_days >= 30", errors.ge(30)),
        ("extreme_late_ge_60d", "promise_error_days >= 60", errors.ge(60)),
        ("extreme_late_ge_90d", "promise_error_days >= 90", errors.ge(90)),
    ]
    return pd.DataFrame([
        {
            "tail_id": tail_id,
            "definition": definition,
            "order_count": int(mask.sum()),
            "proportion": float(mask.mean()),
            "sample_size": len(errors),
        }
        for tail_id, definition, mask in definitions
    ])


def monthly_promise_error_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["purchase_month"] = pd.to_datetime(
        work["order_purchase_timestamp"], errors="coerce"
    ).dt.to_period("M").astype("string")
    if work["purchase_month"].isna().any():
        raise AssertionError("Purchase month is missing in the analytical sample.")
    summary = work.groupby("purchase_month", observed=True).agg(
        order_count=("order_id", "size"),
        late_orders=("late_delivery", "sum"),
        severe_late_orders=("severe_late_2d", "sum"),
        late_rate=("late_delivery", "mean"),
        severe_late_rate=("severe_late_2d", "mean"),
        mean_promise_error=("promise_error_days", "mean"),
        median_promise_error=("promise_error_days", "median"),
        p90_promise_error=("promise_error_days", lambda s: s.quantile(0.90)),
        p95_promise_error=("promise_error_days", lambda s: s.quantile(0.95)),
    ).reset_index()
    summary["purchase_month_start"] = pd.to_datetime(summary["purchase_month"] + "-01")
    summary["terminal_maturity_caution"] = summary["purchase_month"].ge(
        TERMINAL_CAUTION_MONTH
    )
    summary["coverage_note"] = np.where(
        summary["terminal_maturity_caution"],
        "Overlaps endpoint-sensitive terminal purchase cohorts; observed delivered outcomes only.",
        "All observed delivered outcomes in the analytical sample.",
    )
    return summary


def review_join_and_audit(
    frame: pd.DataFrame, review_records: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    selected, all_order_audit = select_latest_usable_review(review_records)
    analytical_ids = set(frame["order_id"])
    raw_linked = review_records.loc[review_records["order_id"].isin(analytical_ids)].copy()
    selected_linked = selected.loc[selected["order_id"].isin(analytical_ids)].copy()
    order_audit = all_order_audit.loc[
        all_order_audit["order_id"].isin(analytical_ids)
    ].copy()

    join_columns = [
        "order_id",
        "selected_review_id",
        "selected_review_score",
        "selected_review_creation_date",
        "selected_review_answer_timestamp",
        "raw_review_records",
        "usable_review_records",
        "distinct_review_ids",
        "distinct_review_scores",
        "minimum_review_score",
        "maximum_review_score",
        "conflicting_scores",
    ]
    joined = frame.merge(
        selected_linked[join_columns], on="order_id", how="left", validate="one_to_one"
    )
    if len(joined) != len(frame) or not joined["order_id"].is_unique:
        raise AssertionError("Review join changed the analytical order population.")
    joined["usable_review"] = joined["selected_review_score"].notna()
    joined["purchase_month"] = pd.to_datetime(
        joined["order_purchase_timestamp"], errors="coerce"
    ).dt.to_period("M").astype("string")
    joined["promise_error_group"] = promise_error_groups(joined["promise_error_days"])

    reviewed = joined.loc[joined["usable_review"]].copy()
    reviewed["selected_review_score"] = reviewed["selected_review_score"].astype(int)
    reviewed["review_answer_before_delivery"] = (
        reviewed["selected_review_answer_timestamp"]
        < reviewed["order_delivered_customer_date"]
    )
    reviewed["review_answer_after_promise_before_delivery"] = (
        reviewed["selected_review_answer_timestamp"]
        >= reviewed["order_estimated_delivery_date"]
    ) & reviewed["review_answer_before_delivery"]
    reviewed["review_answer_before_promised_date"] = (
        reviewed["selected_review_answer_timestamp"]
        < reviewed["order_estimated_delivery_date"]
    )
    reviewed["review_answer_before_delivery_and_before_promised_date"] = (
        reviewed["review_answer_before_delivery"]
        & reviewed["review_answer_before_promised_date"]
    )
    joined = joined.drop(columns=[
        "review_answer_before_delivery",
        "review_answer_after_promise_before_delivery",
        "review_answer_before_promised_date",
        "review_answer_before_delivery_and_before_promised_date",
    ], errors="ignore")
    joined = joined.merge(
        reviewed[[
            "order_id",
            "review_answer_before_delivery",
            "review_answer_after_promise_before_delivery",
            "review_answer_before_promised_date",
            "review_answer_before_delivery_and_before_promised_date",
        ]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    coverage_group = joined.groupby("promise_error_group", observed=False).agg(
        analytical_orders=("order_id", "size"),
        usable_review_orders=("usable_review", "sum"),
    ).reset_index()
    coverage_group["missing_review_orders"] = (
        coverage_group["analytical_orders"] - coverage_group["usable_review_orders"]
    )
    coverage_group["review_coverage"] = (
        coverage_group["usable_review_orders"] / coverage_group["analytical_orders"]
    )

    coverage_month = joined.groupby("purchase_month", observed=True).agg(
        analytical_orders=("order_id", "size"),
        usable_review_orders=("usable_review", "sum"),
    ).reset_index()
    coverage_month["missing_review_orders"] = (
        coverage_month["analytical_orders"] - coverage_month["usable_review_orders"]
    )
    coverage_month["review_coverage"] = (
        coverage_month["usable_review_orders"] / coverage_month["analytical_orders"]
    )
    coverage_month["terminal_maturity_caution"] = coverage_month["purchase_month"].ge(
        TERMINAL_CAUTION_MONTH
    )

    score_distribution = reviewed.groupby("selected_review_score", as_index=False).agg(
        reviewed_orders=("order_id", "size")
    )
    score_distribution = score_distribution.rename(
        columns={"selected_review_score": "review_score"}
    )
    score_distribution["share_reviewed_orders"] = (
        score_distribution["reviewed_orders"] / len(reviewed)
    )

    multiple = selected_linked.loc[selected_linked["raw_review_records"].gt(1)].copy()
    multiple = multiple.sort_values("order_id").reset_index(drop=True)
    # ``load_review_records`` appends a unique source-row index for stable
    # tie-breaking.  Exclude it when auditing duplicates in the raw fields.
    exact_duplicate_rows = int(
        review_records.duplicated(subset=list(REVIEW_COLUMNS)).sum()
    )
    duplicate_order_review_pairs = int(
        review_records.duplicated(["order_id", "review_id"]).sum()
    )
    repeated_ids = review_records.groupby("review_id")["order_id"].nunique()
    repeated_ids_across_orders = int(repeated_ids.gt(1).sum())
    dense_months = coverage_month.loc[coverage_month["analytical_orders"].ge(500)]

    audit_rows = [
        ("analytical_delivered_orders", len(frame), "orders", "Current delivered-outcome sample."),
        ("raw_review_rows_linked", len(raw_linked), "review rows", "Before order-level selection."),
        ("orders_with_any_review_record", int(order_audit["order_id"].nunique()), "orders", "Any linked raw record, valid or invalid."),
        ("orders_with_usable_review_score", len(selected_linked), "orders", "Selected valid score in {1,...,5}."),
        ("usable_review_coverage", len(selected_linked) / len(frame), "proportion", "Review-analysis coverage of delivered analytical orders."),
        ("orders_without_usable_review_score", len(frame) - len(selected_linked), "orders", "Retained in descriptive sample; excluded only from review outcome analysis."),
        ("orders_with_multiple_review_records", int(order_audit["raw_review_records"].gt(1).sum()), "orders", "Counted before deterministic selection."),
        ("orders_with_multiple_review_records_proportion", float(order_audit["raw_review_records"].gt(1).sum() / len(frame)), "proportion", "Share of all delivered analytical orders, before deterministic selection."),
        ("orders_with_conflicting_review_scores", int(order_audit["conflicting_scores"].sum()), "orders", "Multiple distinct valid scores."),
        ("orders_with_conflicting_review_scores_proportion", float(order_audit["conflicting_scores"].sum() / len(frame)), "proportion", "Share of all delivered analytical orders."),
        ("maximum_review_records_per_order", int(order_audit["raw_review_records"].max()), "review rows", "Maximum within the delivered analytical sample."),
        ("exact_duplicate_review_rows", exact_duplicate_rows, "review rows", "Across the raw review file."),
        ("duplicate_order_review_id_pairs", duplicate_order_review_pairs, "review rows", "Across the raw review file."),
        ("review_ids_reused_across_orders", repeated_ids_across_orders, "review IDs", "Review ID is not used as the join key."),
        ("selected_reviews_answered_before_delivery", int(reviewed["review_answer_before_delivery"].sum()), "orders", "Valid observed reviews, not uniformly post-delivery."),
        ("selected_reviews_answered_before_delivery_proportion", float(reviewed["review_answer_before_delivery"].mean()), "proportion", "Share of orders with a selected usable review."),
        ("selected_reviews_after_promise_before_delivery", int(reviewed["review_answer_after_promise_before_delivery"].sum()), "orders", "Promise date passed before actual delivery."),
        ("selected_reviews_after_promise_before_delivery_proportion", float(reviewed["review_answer_after_promise_before_delivery"].mean()), "proportion", "Share of orders with a selected usable review."),
        ("selected_reviews_answered_before_promised_date", int(reviewed["review_answer_before_promised_date"].sum()), "orders", "Often follows an early actual delivery but precedes the more conservative promised date."),
        ("selected_reviews_before_delivery_and_before_promised_date", int(reviewed["review_answer_before_delivery_and_before_promised_date"].sum()), "orders", "Answered before both actual delivery and promised date."),
        ("minimum_error_group_review_coverage", float(coverage_group["review_coverage"].min()), "proportion", "Across the eight pre-specified promise-error groups."),
        ("maximum_error_group_review_coverage", float(coverage_group["review_coverage"].max()), "proportion", "Across the eight pre-specified promise-error groups."),
        ("minimum_dense_month_review_coverage", float(dense_months["review_coverage"].min()), "proportion", "Purchase months with at least 500 analytical orders."),
        ("maximum_dense_month_review_coverage", float(dense_months["review_coverage"].max()), "proportion", "Purchase months with at least 500 analytical orders."),
        ("selection_rule", "latest valid answer", "rule", "Latest answer timestamp; latest creation timestamp; smallest review ID; source row."),
    ]
    audit = pd.DataFrame(audit_rows, columns=["metric", "value", "unit", "definition"])
    tables = {
        "audit": audit,
        "multiple": multiple,
        "score_distribution": score_distribution,
        "coverage_group": coverage_group,
        "coverage_month": coverage_month,
    }
    return joined, tables


def review_outcomes_by_group(joined: pd.DataFrame) -> pd.DataFrame:
    reviewed = joined.loc[joined["usable_review"]].copy()
    reviewed["review_score"] = reviewed["selected_review_score"].astype(int)
    reviewed["one_star"] = reviewed["review_score"].eq(1)
    reviewed["low_review_2"] = reviewed["review_score"].le(2)
    reviewed["low_review_3"] = reviewed["review_score"].le(3)
    table = reviewed.groupby("promise_error_group", observed=False).agg(
        reviewed_orders=("order_id", "size"),
        mean_review_score=("review_score", "mean"),
        median_review_score=("review_score", "median"),
        one_star_orders=("one_star", "sum"),
        one_star_share=("one_star", "mean"),
        low_review_2_orders=("low_review_2", "sum"),
        low_review_2_share=("low_review_2", "mean"),
        low_review_3_orders=("low_review_3", "sum"),
        low_review_3_share=("low_review_3", "mean"),
    ).reset_index()
    if table["reviewed_orders"].min() < MIN_REVIEWED_GROUP_SIZE:
        raise RuntimeError(
            "A pre-specified review group has fewer than "
            f"{MIN_REVIEWED_GROUP_SIZE} reviewed orders; document a bin change before interpreting."
        )
    intervals_2 = [
        wilson_interval(int(row.low_review_2_orders), int(row.reviewed_orders))
        for row in table.itertuples()
    ]
    intervals_3 = [
        wilson_interval(int(row.low_review_3_orders), int(row.reviewed_orders))
        for row in table.itertuples()
    ]
    table["low_review_2_ci_lower"] = [v[0] for v in intervals_2]
    table["low_review_2_ci_upper"] = [v[1] for v in intervals_2]
    table["low_review_3_ci_lower"] = [v[0] for v in intervals_3]
    table["low_review_3_ci_upper"] = [v[1] for v in intervals_3]
    return table


def adjusted_low_review_association(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the pre-specified calendar-adjusted observational sensitivity."""
    analysis = joined.loc[joined["usable_review"]].copy()
    analysis["low_review_2"] = analysis["selected_review_score"].astype(int).le(2).astype(int)
    analysis["promise_error_group_label"] = analysis["promise_error_group"].astype("string")
    analysis["purchase_month_label"] = analysis["purchase_month"].astype("string")
    month_counts = analysis["purchase_month_label"].value_counts()
    sparse_months = set(
        month_counts.loc[month_counts.lt(MIN_CALENDAR_LEVEL_SIZE)].index
    )
    analysis["purchase_month_adjustment"] = analysis["purchase_month_label"].where(
        ~analysis["purchase_month_label"].isin(sparse_months),
        SPARSE_CALENDAR_LABEL,
    )
    if CALENDAR_REFERENCE_MONTH not in set(analysis["purchase_month_adjustment"]):
        raise AssertionError("The pre-specified dense calendar reference is unavailable.")
    reference = "on promised date"
    formula = (
        'low_review_2 ~ C(promise_error_group_label, Treatment(reference="on promised date")) '
        '+ C(purchase_month_adjustment, Treatment(reference="2018-01"))'
    )
    fitted = smf.glm(
        formula=formula, data=analysis, family=sm.families.Binomial()
    ).fit(cov_type="HC1")
    confidence = fitted.conf_int()
    full = pd.DataFrame({
        "term": fitted.params.index,
        "coefficient_log_odds": fitted.params.to_numpy(),
        "standard_error_hc1": fitted.bse.to_numpy(),
        "odds_ratio": np.exp(fitted.params.to_numpy()),
        "odds_ratio_ci_lower": np.exp(confidence.iloc[:, 0].to_numpy()),
        "odds_ratio_ci_upper": np.exp(confidence.iloc[:, 1].to_numpy()),
        "p_value": fitted.pvalues.to_numpy(),
    })

    group_rows = []
    for label in ERROR_GROUP_LABELS:
        if label == reference:
            group_rows.append({
                "promise_error_group": label,
                "reference_group": True,
                "coefficient_log_odds": 0.0,
                "standard_error_hc1": np.nan,
                "odds_ratio": 1.0,
                "odds_ratio_ci_lower": 1.0,
                "odds_ratio_ci_upper": 1.0,
                "p_value": np.nan,
            })
            continue
        suffix = f"[T.{label}]"
        matches = [term for term in fitted.params.index if term.endswith(suffix)]
        if len(matches) != 1:
            raise AssertionError(f"Could not identify adjusted coefficient for {label!r}.")
        term = matches[0]
        group_rows.append({
            "promise_error_group": label,
            "reference_group": False,
            "coefficient_log_odds": fitted.params[term],
            "standard_error_hc1": fitted.bse[term],
            "odds_ratio": np.exp(fitted.params[term]),
            "odds_ratio_ci_lower": np.exp(confidence.loc[term, 0]),
            "odds_ratio_ci_upper": np.exp(confidence.loc[term, 1]),
            "p_value": fitted.pvalues[term],
        })
    group_effects = pd.DataFrame(group_rows)
    calendar_cell_audit = analysis.groupby("purchase_month_adjustment").agg(
        orders=("order_id", "size"),
        low_reviews=("low_review_2", "sum"),
    )
    diagnostics = pd.DataFrame([{
        "analysis": "calendar_adjusted_observational_sensitivity",
        "outcome": "review_score <= 2",
        "formula": formula,
        "n_orders": int(fitted.nobs),
        "low_review_orders": int(analysis["low_review_2"].sum()),
        "low_review_rate": analysis["low_review_2"].mean(),
        "raw_purchase_month_levels": analysis["purchase_month_label"].nunique(),
        "adjustment_calendar_levels": analysis["purchase_month_adjustment"].nunique(),
        "pooled_sparse_months": ", ".join(sorted(sparse_months)),
        "sparse_month_pool_rule": f"raw purchase months with <{MIN_CALENDAR_LEVEL_SIZE} reviewed orders",
        "calendar_reference": CALENDAR_REFERENCE_MONTH,
        "minimum_calendar_level_orders": int(calendar_cell_audit["orders"].min()),
        "minimum_calendar_level_low_reviews": int(calendar_cell_audit["low_reviews"].min()),
        "promise_error_groups": analysis["promise_error_group_label"].nunique(),
        "converged": bool(fitted.converged),
        "all_coefficients_finite": bool(np.isfinite(fitted.params).all()),
        "all_standard_errors_finite": bool(np.isfinite(fitted.bse).all()),
        "covariance": "HC1",
        "reference_group": reference,
        "deviance": fitted.deviance,
        "null_deviance": fitted.null_deviance,
        "aic": fitted.aic,
        "interpretation_scope": "association sensitivity only; not prediction or causality",
    }])
    return group_effects, full, diagnostics


def plot_monthly_promise_error(monthly: pd.DataFrame, path: str | Path) -> None:
    # Keep all months in the machine-readable table, but omit one-/few-order
    # cohorts from the thesis figure so they do not determine the axes.
    work = monthly.loc[monthly["order_count"].ge(MIN_CALENDAR_LEVEL_SIZE)].copy()
    if work.empty:
        raise AssertionError("No purchase month meets the figure stability threshold.")
    dates = pd.to_datetime(work["purchase_month_start"])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].bar(dates, work["order_count"], width=22, color="#9ECAE1")
    axes[0].set_ylabel("Orders")
    axes[0].set_title("(a) Delivered analytical orders by purchase month", loc="left")

    axes[1].plot(dates, 100 * work["late_rate"], marker="o", color="#D55E00", label="Late (>0 days)")
    axes[1].plot(dates, 100 * work["severe_late_rate"], marker="s", color="#CC79A7", label="Severe late (>=2 days)")
    axes[1].set_ylabel("Orders (%)")
    axes[1].set_title("(b) Promise-breach incidence", loc="left")
    axes[1].legend(frameon=False, ncol=2)

    for column, label, color, marker in [
        ("median_promise_error", "Median", "#0072B2", "o"),
        ("p90_promise_error", "p90", "#009E73", "s"),
        ("p95_promise_error", "p95", "#D55E00", "^"),
    ]:
        axes[2].plot(dates, work[column], marker=marker, color=color, label=label)
    axes[2].axhline(0, color="0.45", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Promise error (days)")
    axes[2].set_title("(c) Promise-error location and upper tail", loc="left")
    axes[2].legend(frameon=False, ncol=3)

    terminal_start = pd.Timestamp(f"{TERMINAL_CAUTION_MONTH}-01")
    terminal_end = dates.max() + pd.offsets.MonthEnd(1)
    for ax in axes:
        ax.axvspan(terminal_start, terminal_end, color="0.75", alpha=0.28)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_xlabel("Purchase / promise-issuance month")
    fig.suptitle("Delivery-promise errors over purchase time")
    fig.text(
        0.5,
        0.008,
        "Figure shows months with >=500 delivered orders; sparse 2016 cohorts remain in the table. Grey months overlap endpoint-sensitive terminal cohorts. Descriptive, not causal.",
        ha="center",
        fontsize=8.5,
    )
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_review_outcomes(table: pd.DataFrame, path: str | Path) -> None:
    positions = np.arange(len(table))
    rates = table["low_review_2_share"].to_numpy()
    lower = rates - table["low_review_2_ci_lower"].to_numpy()
    upper = table["low_review_2_ci_upper"].to_numpy() - rates
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.errorbar(
        positions,
        100 * rates,
        yerr=100 * np.vstack([lower, upper]),
        fmt="o-",
        color="#D55E00",
        markerfacecolor="white",
        markeredgewidth=1.7,
        markersize=7,
        capsize=4,
        linewidth=1.5,
    )
    for x, row in zip(positions, table.itertuples()):
        ax.annotate(
            f"{100 * row.low_review_2_share:.1f}%",
            (x, 100 * row.low_review_2_ci_upper),
            xytext=(0, 19),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="semibold",
        )
        ax.annotate(
            f"n={int(row.reviewed_orders):,}",
            (x, 100 * row.low_review_2_ci_upper),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
            color="0.35",
        )
    group_labels = [
        "<= -14 days\n(very early)",
        "-13 to -7 days\n(early)",
        "-6 to -1 days\n(slightly early)",
        "On promised\ndate",
        "1 day late",
        "2-3 days late",
        "4-7 days late",
        ">= 8 days late",
    ]
    if len(table) != len(group_labels):
        raise AssertionError("Expected the eight fixed promise-error groups.")
    ax.set_xticks(positions, group_labels, rotation=18, ha="right")
    ax.set_ylim(0, 88)
    ax.set_ylabel("Low reviews: score <=2 (%)")
    ax.set_xlabel("Realised promise-error group")
    ax.set_title("Observed low-review rate increases with promise-error severity")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.005,
        "Labels show observed rates and reviewed orders; error bars are 95% Wilson intervals. Association is observational, not causal.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(data_dir: str | Path | None, output_root: str | Path = "results") -> dict[str, pd.DataFrame]:
    output = Path(output_root)
    for subdir in ["tables", "metrics", "figures"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    raw = load_order_level_data(data_dir)
    frame = build_targets(raw).dropna(
        subset=["order_id", "order_purchase_timestamp", "promise_error_days"]
    ).copy()
    if len(frame) != 96_470 or not frame["order_id"].is_unique:
        raise AssertionError("RQ1 sample does not match the current 96,470 unique-order sample.")

    distribution = promise_error_distribution(frame)
    tails = promise_error_tail_counts(frame)
    monthly = monthly_promise_error_summary(frame)
    review_records = load_review_records(data_dir)
    joined, audit_tables = review_join_and_audit(frame, review_records)
    outcomes = review_outcomes_by_group(joined)
    adjusted, adjusted_full, adjusted_diagnostics = adjusted_low_review_association(joined)

    distribution.to_csv(output / "tables/rq1_promise_error_distribution.csv", index=False)
    tails.to_csv(output / "tables/rq1_promise_error_extreme_tails.csv", index=False)
    monthly.to_csv(output / "tables/rq1_promise_error_monthly.csv", index=False)
    audit_tables["audit"].to_csv(output / "tables/rq1_review_join_audit.csv", index=False)
    audit_tables["multiple"].to_csv(output / "tables/rq1_review_multiple_records.csv", index=False)
    audit_tables["score_distribution"].to_csv(output / "tables/rq1_review_score_distribution.csv", index=False)
    audit_tables["coverage_group"].to_csv(output / "tables/rq1_review_coverage_by_error_group.csv", index=False)
    audit_tables["coverage_month"].to_csv(output / "tables/rq1_review_coverage_by_purchase_month.csv", index=False)
    outcomes.to_csv(output / "tables/rq1_review_outcomes_by_error_group.csv", index=False)
    adjusted.to_csv(output / "metrics/rq1_adjusted_low_review_group_effects.csv", index=False)
    adjusted_full.to_csv(output / "metrics/rq1_adjusted_low_review_full_coefficients.csv", index=False)
    adjusted_diagnostics.to_csv(output / "metrics/rq1_adjusted_low_review_model_diagnostics.csv", index=False)

    plot_monthly_promise_error(monthly, output / "figures/rq1_promise_error_over_time.pdf")
    plot_review_outcomes(outcomes, output / "figures/rq1_low_review_by_promise_error.pdf")

    manifest = {
        "experiment_id": "RQ1-DESC-REV-001",
        "run_date": "2026-08-08",
        "sample": "96,470 unique delivered orders with observed actual and estimated delivery dates",
        "target_convention": "normalized calendar-date actual delivery minus estimated delivery date",
        "temporal_index": "order purchase / promise-issuance month",
        "review_selection": [
            "valid integer score in 1..5",
            "latest review_answer_timestamp",
            "latest review_creation_date",
            "lexicographically smallest review_id",
            "source row",
        ],
        "promise_error_groups": ERROR_GROUP_LABELS,
        "primary_low_review": "review_score <= 2",
        "sensitivity_low_review": "review_score <= 3",
        "adjusted_formula": adjusted_diagnostics.loc[0, "formula"],
        "calendar_adjustment": (
            "purchase-month fixed effects; raw levels with fewer than 500 reviewed "
            "orders pooled for numerical stability; January 2018 reference"
        ),
        "adjusted_role": "observational calendar-composition sensitivity only",
        "endpoint_limitation": "terminal purchase cohorts are conditional on observed delivered outcomes and may be outcome-immature",
        "models_not_run": [
            "classification",
            "quantile",
            "mixed_effects",
            "weather",
            "policy",
        ],
    }
    (output / "metrics/rq1_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "distribution": distribution,
        "tails": tails,
        "monthly": monthly,
        "review_audit": audit_tables["audit"],
        "review_outcomes": outcomes,
        "adjusted": adjusted,
        "adjusted_diagnostics": adjusted_diagnostics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    result = run(args.data_dir, args.output_root)
    for name, table in result.items():
        print(f"\n[{name}]\n{table.to_string(index=False)}")
