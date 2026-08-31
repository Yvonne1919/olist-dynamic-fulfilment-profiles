"""Frozen data construction for RQ1 Speed and Promise-Reliability V1.

This module is deliberately side-effect free: it reads protected inputs and
returns in-memory frames, but it never writes an empirical or governance file.
The public entry point :func:`build_analysis_frames` combines the current
Phase-2A canonical order assembler with the existing deterministic RQ1 review
selection and then reconciles the result against the legacy RQ1 assembler.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from src.data.olist import load_order_level_data
from src.data.reviews import load_review_records, select_latest_usable_review
from src.experiments.rq1_customer_relevance import (
    ERROR_GROUP_EDGES,
    ERROR_GROUP_LABELS,
    promise_error_groups,
    review_join_and_audit,
    wilson_interval,
)
from src.features.targets import build_targets


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ASSEMBLER = (
    REPOSITORY_ROOT / "analysis/profile_pivot_phase2a/scripts/data_pipeline.py"
)

EXPECTED_SOURCE_SHA256 = {
    "analysis/profile_pivot_phase2a/scripts/data_pipeline.py": (
        "0c4cad3c99db268292253abd26e2070ccf2a286337bcfe9fb76fb3768c44ab8d"
    ),
    "src/data/olist.py": (
        "5fcc289fe79756bc9d3e08b037d5c23c474ab986f054dfbf4a75ffeade27cab0"
    ),
    "src/data/reviews.py": (
        "7d8b5182188ef73f03c200b74b7ebd546a6d98dae4628ec26239fd9296c88520"
    ),
    "src/features/targets.py": (
        "303913e67144d7721ffa46ca2567ba3cbffef1c901ce0410d64cccee0edaa1ea"
    ),
    # The current protected working-tree version differs from HEAD only in the
    # already-authorised rendering of the existing RQ1 figure.  The empirical
    # construction and group function are unchanged.
    "src/experiments/rq1_customer_relevance.py": (
        "647024c67312a6899670e199d43b2982d3b0d06027a42057f11620567f69fbbf"
    ),
}

EXPECTED_RAW_SHA256 = {
    "olist_customers_dataset.csv": (
        "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2"
    ),
    "olist_geolocation_dataset.csv": (
        "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6"
    ),
    "olist_order_items_dataset.csv": (
        "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279"
    ),
    "olist_order_payments_dataset.csv": (
        "4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0"
    ),
    "olist_order_reviews_dataset.csv": (
        "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7"
    ),
    "olist_orders_dataset.csv": (
        "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf"
    ),
    "olist_products_dataset.csv": (
        "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc"
    ),
    "olist_sellers_dataset.csv": (
        "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e"
    ),
    "product_category_name_translation.csv": (
        "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278"
    ),
}

EXPECTED_CANONICAL_ORDERS = 96_470
EXPECTED_REVIEWED_ORDERS = 95_824
EXPECTED_LOW_REVIEW_2 = 12_272
EXPECTED_LOW_REVIEW_3 = 20_188
EXPECTED_POST_DELIVERY_REVIEWS = 91_171
EXPECTED_RAW_REVIEW_ROWS_LINKED = 96_353
EXPECTED_MULTIPLE_REVIEW_ORDERS = 525
EXPECTED_CONFLICTING_REVIEW_ORDERS = 189
EXPECTED_REVIEWS_BEFORE_DELIVERY = 4_653
EXPECTED_REVIEWS_AFTER_PROMISE_BEFORE_DELIVERY = 4_432
EXPECTED_REVIEWS_BEFORE_DELIVERY_AND_BEFORE_PROMISE = 221
EXPECTED_SELECTED_REVIEW_SHA256 = (
    "3b89feb2bbb0ca0c985374dfcc6de903726a4d6aa5ad558dacc828d947df0570"
)
EXPECTED_REVIEWED_ERROR_GROUP_COUNTS = {
    "very early: <= -14 days": 41_861,
    "early: -13 to -7 days": 33_883,
    "slightly early: -6 to -1 days": 12_419,
    "on promised date": 1_280,
    "1 day late": 820,
    "2-3 days late": 1_032,
    "4-7 days late": 1_748,
    ">=8 days late": 2_781,
}
EXPECTED_SPARSE_MONTHS = ("2016-09", "2016-10", "2016-12")

DEFAULT_DURATION_LABELS = (
    "0-3 days",
    "4-7 days",
    "8-14 days",
    "15-21 days",
    "22+ days",
)
DEFAULT_DURATION_EDGES = (-1.0, 3.0, 7.0, 14.0, 21.0, np.inf)

_PHASE2A_MODULE_NAME = "_rq1_frozen_phase2a_data_pipeline"

__all__ = [
    "build_analysis_frames",
    "pool_sparse_purchase_months",
    "build_sample_audit",
    "build_date_identity_audit",
    "build_review_coverage",
    "build_duration_error_cell_counts",
    "build_duration_error_review_rates",
    "build_data_audit_tables",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_edge(value: Any) -> float:
    if isinstance(value, str):
        if value == "Infinity":
            return np.inf
        if value == "-Infinity":
            return -np.inf
    return float(value)


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = config["expected_sample"]
    assert int(expected["canonical_delivered_orders"]) == EXPECTED_CANONICAL_ORDERS
    assert int(expected["reviewed_orders"]) == EXPECTED_REVIEWED_ORDERS
    assert expected["purchase_month_min"] == "2016-09"
    assert expected["purchase_month_max"] == "2018-08"

    configured_error_labels = list(config["promise_error_groups"]["labels"])
    configured_error_edges = [
        _parse_edge(value) for value in config["promise_error_groups"]["edges"]
    ]
    assert configured_error_labels == list(ERROR_GROUP_LABELS)
    assert np.array_equal(
        np.asarray(configured_error_edges), np.asarray(ERROR_GROUP_EDGES, dtype=float)
    )
    assert bool(config["promise_error_groups"]["right_closed"])
    assert config["promise_error_groups"]["reference"] == "on promised date"

    duration = config["actual_duration_groups"]
    assert tuple(duration["labels"]) == DEFAULT_DURATION_LABELS
    assert np.array_equal(
        np.asarray([_parse_edge(value) for value in duration["edges"]]),
        np.asarray(DEFAULT_DURATION_EDGES),
    )
    assert bool(duration["right_closed"])
    assert int(duration["low_support_threshold"]) == 50

    models = config["models"]
    assert int(models["month_min_orders"]) == 500
    assert models["month_reference"] == "2018-01"
    assert models["sparse_month_label"] == (
        "pooled sparse purchase months (<500 orders each)"
    )
    assert int(models["spline_df"]) == 4
    assert models["covariance"] == "HC1"


def _verify_source_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected in EXPECTED_SOURCE_SHA256.items():
        path = REPOSITORY_ROOT / relative_path
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Protected source hash mismatch for {relative_path}: "
                f"expected {expected}, observed {actual}"
            )
        observed[relative_path] = actual
    return observed


def _verify_raw_hashes(data_dir: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename, expected in EXPECTED_RAW_SHA256.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required raw Olist input is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Raw input hash mismatch for {filename}: expected {expected}, "
                f"observed {actual}"
            )
        observed[filename] = actual
    return observed


def _load_phase2a_module() -> ModuleType:
    existing = sys.modules.get(_PHASE2A_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _PHASE2A_MODULE_NAME, CANONICAL_ASSEMBLER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical assembler: {CANONICAL_ASSEMBLER}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during execution.
    sys.modules[_PHASE2A_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _reconcile_legacy_order_base(
    canonical: pd.DataFrame, data_dir: Path
) -> dict[str, Any]:
    legacy = build_targets(load_order_level_data(data_dir)).dropna(
        subset=["order_id", "order_purchase_timestamp", "promise_error_days"]
    )
    if len(legacy) != EXPECTED_CANONICAL_ORDERS or not legacy["order_id"].is_unique:
        raise AssertionError("Legacy RQ1 assembler no longer yields 96,470 unique orders")

    canonical_columns = [
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "promise_error_days",
    ]
    left = canonical.loc[:, canonical_columns].sort_values(
        "order_id", kind="mergesort"
    ).reset_index(drop=True)
    right = legacy.loc[:, canonical_columns].sort_values(
        "order_id", kind="mergesort"
    ).reset_index(drop=True)
    id_mismatches = int(left["order_id"].ne(right["order_id"]).sum())
    timestamp_mismatches: dict[str, int] = {}
    for column in canonical_columns[1:4]:
        equal = left[column].eq(right[column]) | (
            left[column].isna() & right[column].isna()
        )
        timestamp_mismatches[column] = int((~equal).sum())
    left_error = pd.to_numeric(left["promise_error_days"], errors="coerce")
    right_error = pd.to_numeric(right["promise_error_days"], errors="coerce")
    error_equal = left_error.eq(right_error) | (
        left_error.isna() & right_error.isna()
    )
    error_mismatches = int((~error_equal).sum())
    if id_mismatches or any(timestamp_mismatches.values()) or error_mismatches:
        raise AssertionError(
            "Canonical/legacy RQ1 order reconciliation failed: "
            f"ids={id_mismatches}, timestamps={timestamp_mismatches}, "
            f"promise_error={error_mismatches}"
        )
    if canonical["order_id"].tolist() != legacy["order_id"].tolist():
        raise AssertionError("Canonical and legacy RQ1 order row order changed")

    return {
        "legacy_orders": int(len(legacy)),
        "id_mismatches": id_mismatches,
        "timestamp_mismatches": timestamp_mismatches,
        "promise_error_mismatches": error_mismatches,
        "row_order_equal": True,
    }


def _selected_review_digest(selected: pd.DataFrame) -> str:
    columns = [
        "order_id",
        "selected_review_id",
        "selected_review_score",
        "selected_review_creation_date",
        "selected_review_answer_timestamp",
        "_source_row",
    ]
    serialised = selected.loc[:, columns].sort_values(
        "order_id", kind="mergesort"
    ).to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _derive_timing_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    purchase = pd.to_datetime(
        frame["order_purchase_timestamp"], errors="coerce"
    ).dt.normalize()
    actual = pd.to_datetime(
        frame["order_delivered_customer_date"], errors="coerce"
    ).dt.normalize()
    promise = pd.to_datetime(
        frame["order_estimated_delivery_date"], errors="coerce"
    ).dt.normalize()
    frame["purchase_date"] = purchase
    frame["actual_delivery_date"] = actual
    frame["estimated_delivery_date"] = promise
    frame["actual_delivery_days"] = (actual - purchase).dt.days.astype("Int64")
    frame["promised_lead_days"] = (promise - purchase).dt.days.astype("Int64")
    recomputed_error = (actual - promise).dt.days.astype("Int64")
    existing_error = pd.to_numeric(frame["promise_error_days"], errors="coerce").astype(
        "Int64"
    )
    if not recomputed_error.equals(existing_error):
        raise AssertionError("Date-normalised promise error differs from canonical target")
    frame["promise_error_days"] = recomputed_error

    duration = config["actual_duration_groups"]
    duration_edges = [_parse_edge(value) for value in duration["edges"]]
    frame["actual_duration_group"] = pd.cut(
        frame["actual_delivery_days"].astype(float),
        bins=duration_edges,
        labels=list(duration["labels"]),
        right=True,
        include_lowest=True,
        ordered=True,
    )
    # Re-apply the authoritative function rather than reconstructing the bins.
    frame["promise_error_group"] = promise_error_groups(
        frame["promise_error_days"]
    )
    frame["promise_error_group_label"] = frame["promise_error_group"].astype(
        "string"
    )


def _derive_review_columns(frame: pd.DataFrame) -> None:
    score = pd.to_numeric(frame["selected_review_score"], errors="coerce")
    frame["review_score"] = score.astype("Int64")
    frame["low_review_2"] = score.le(2).astype("Int8").where(score.notna())
    frame["low_review_3"] = score.le(3).astype("Int8").where(score.notna())
    frame["one_star"] = score.eq(1).astype("Int8").where(score.notna())

    answer = pd.to_datetime(
        frame["selected_review_answer_timestamp"], errors="coerce"
    )
    actual_timestamp = pd.to_datetime(
        frame["order_delivered_customer_date"], errors="coerce"
    )
    timing_observed = score.notna() & answer.notna() & actual_timestamp.notna()
    at_or_after = answer.ge(actual_timestamp)
    frame["review_at_or_after_delivery"] = at_or_after.astype("boolean").where(
        timing_observed
    )


def pool_sparse_purchase_months(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | str | Path,
    *,
    month_column: str = "purchase_month",
) -> tuple[pd.Series, dict[str, Any]]:
    """Pool raw purchase months with fewer than the frozen row threshold.

    The counts are intentionally calculated on the supplied analysis frame.
    Passing the primary reviewed frame exactly reproduces the old RQ1 GLM;
    passing a sensitivity subset applies the same frozen rule to that subset.
    """

    frozen = _load_config(config)
    models = frozen["models"]
    minimum = int(models["month_min_orders"])
    pooled_label = str(models["sparse_month_label"])
    reference = str(models["month_reference"])
    months = frame[month_column].astype("string")
    if months.isna().any():
        raise AssertionError(f"Missing values in {month_column}")
    counts = months.value_counts().sort_index()
    sparse_months = tuple(counts.loc[counts.lt(minimum)].index.astype(str))
    adjustment = months.where(~months.isin(sparse_months), pooled_label)
    if reference not in set(adjustment.astype(str)):
        raise AssertionError(f"Frozen purchase-month reference is absent: {reference}")
    adjusted_counts = adjustment.value_counts().sort_index()
    audit = {
        "source_column": month_column,
        "threshold": minimum,
        "rule": f"raw purchase months with <{minimum} analysis rows",
        "sparse_months": list(sparse_months),
        "raw_month_counts": {str(k): int(v) for k, v in counts.items()},
        "raw_month_levels": int(len(counts)),
        "adjustment_levels": int(len(adjusted_counts)),
        "pooled_label": pooled_label,
        "pooled_rows": int(months.isin(sparse_months).sum()),
        "reference": reference,
    }
    return adjustment.astype("string"), audit


def build_analysis_frames(
    data_dir: str | Path | None,
    config: Mapping[str, Any] | str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build and hard-audit the frozen all-order and reviewed-order frames."""

    frozen = _load_config(config)
    _validate_config(frozen)
    source_hashes = _verify_source_hashes()
    phase2a = _load_phase2a_module()
    resolved_data_dir = Path(phase2a.resolve_data_dir(data_dir))
    raw_hashes = _verify_raw_hashes(resolved_data_dir)

    canonical = phase2a.build_order_base(resolved_data_dir)
    canonical_audit = phase2a.audit_order_base(
        canonical, enforce_reference_counts=True
    )
    if len(canonical) != EXPECTED_CANONICAL_ORDERS:
        raise AssertionError("Canonical assembler did not yield exactly 96,470 orders")
    if canonical["order_id"].isna().any() or not canonical["order_id"].is_unique:
        raise AssertionError("Canonical order_id is not non-null and unique")
    legacy_audit = _reconcile_legacy_order_base(canonical, resolved_data_dir)

    review_records = load_review_records(resolved_data_dir)
    selected, _ = select_latest_usable_review(review_records)
    selected = selected.loc[selected["order_id"].isin(set(canonical["order_id"]))].copy()
    selected_digest = _selected_review_digest(selected)
    if len(selected) != EXPECTED_REVIEWED_ORDERS or not selected["order_id"].is_unique:
        raise AssertionError("Deterministic review selection did not yield 95,824 orders")
    if selected_digest != EXPECTED_SELECTED_REVIEW_SHA256:
        raise AssertionError(
            "Deterministic selected-review digest mismatch: "
            f"expected {EXPECTED_SELECTED_REVIEW_SHA256}, observed {selected_digest}"
        )
    if selected[
        ["selected_review_creation_date", "selected_review_answer_timestamp"]
    ].isna().any().any():
        raise AssertionError("A selected canonical review has a missing timing field")

    all_orders, old_review_audits = review_join_and_audit(canonical, review_records)
    old_review_metrics = (
        old_review_audits["audit"].set_index("metric")["value"].to_dict()
    )
    exact_old_review_metrics = {
        "raw_review_rows_linked": EXPECTED_RAW_REVIEW_ROWS_LINKED,
        "orders_with_multiple_review_records": EXPECTED_MULTIPLE_REVIEW_ORDERS,
        "orders_with_conflicting_review_scores": EXPECTED_CONFLICTING_REVIEW_ORDERS,
        "selected_reviews_answered_before_delivery": EXPECTED_REVIEWS_BEFORE_DELIVERY,
        "selected_reviews_after_promise_before_delivery": (
            EXPECTED_REVIEWS_AFTER_PROMISE_BEFORE_DELIVERY
        ),
        "selected_reviews_before_delivery_and_before_promised_date": (
            EXPECTED_REVIEWS_BEFORE_DELIVERY_AND_BEFORE_PROMISE
        ),
    }
    for metric, expected_value in exact_old_review_metrics.items():
        observed_value = int(old_review_metrics[metric])
        if observed_value != expected_value:
            raise AssertionError(
                f"Existing RQ1 review-audit metric changed for {metric}: "
                f"expected {expected_value}, observed {observed_value}"
            )
    _derive_timing_columns(all_orders, frozen)
    _derive_review_columns(all_orders)

    identity_residual = all_orders["actual_delivery_days"] - (
        all_orders["promised_lead_days"] + all_orders["promise_error_days"]
    )
    missing_date_components = int(
        all_orders[
            ["purchase_date", "actual_delivery_date", "estimated_delivery_date"]
        ].isna().any(axis=1).sum()
    )
    identity_failures = int(identity_residual.ne(0).fillna(True).sum())
    negative_actual = int(all_orders["actual_delivery_days"].lt(0).sum())
    negative_promised = int(all_orders["promised_lead_days"].lt(0).sum())
    unbinned_duration = int(all_orders["actual_duration_group"].isna().sum())
    unbinned_error = int(all_orders["promise_error_group"].isna().sum())
    if any(
        [
            missing_date_components,
            identity_failures,
            negative_actual,
            negative_promised,
            unbinned_duration,
            unbinned_error,
        ]
    ):
        raise AssertionError(
            "Frozen date/duration audit failed: "
            f"missing={missing_date_components}, identity={identity_failures}, "
            f"negative_actual={negative_actual}, negative_promised={negative_promised}, "
            f"unbinned_duration={unbinned_duration}, unbinned_error={unbinned_error}"
        )

    reviewed = all_orders.loc[all_orders["usable_review"]].copy()
    if len(reviewed) != EXPECTED_REVIEWED_ORDERS or not reviewed["order_id"].is_unique:
        raise AssertionError("Reviewed analysis frame did not retain 95,824 unique orders")
    if reviewed["review_score"].isna().any():
        raise AssertionError("Reviewed analysis frame contains a missing selected score")
    if int(reviewed["low_review_2"].sum()) != EXPECTED_LOW_REVIEW_2:
        raise AssertionError("Primary low-review target no longer reproduces 12,272 cases")
    if int(reviewed["low_review_3"].sum()) != EXPECTED_LOW_REVIEW_3:
        raise AssertionError("Sensitivity low-review target no longer reproduces 20,188 cases")
    if int(reviewed["review_at_or_after_delivery"].sum()) != EXPECTED_POST_DELIVERY_REVIEWS:
        raise AssertionError("Timestamp-level post-delivery review count changed")

    month_min = str(reviewed["purchase_month"].min())
    month_max = str(reviewed["purchase_month"].max())
    if (month_min, month_max) != ("2016-09", "2018-08"):
        raise AssertionError(
            f"Reviewed purchase-month range changed: {month_min} through {month_max}"
        )
    pooled, month_audit = pool_sparse_purchase_months(reviewed, frozen)
    if tuple(month_audit["sparse_months"]) != EXPECTED_SPARSE_MONTHS:
        raise AssertionError(
            f"Sparse purchase months changed: {month_audit['sparse_months']}"
        )
    if month_audit["pooled_rows"] != 264 or month_audit["adjustment_levels"] != 21:
        raise AssertionError("Primary RQ1 sparse-month pooling no longer reproduces")
    reviewed["purchase_month_adjustment"] = pooled.to_numpy()
    month_mapping = (
        reviewed[["purchase_month", "purchase_month_adjustment"]]
        .drop_duplicates()
        .set_index("purchase_month")["purchase_month_adjustment"]
    )
    if not month_mapping.index.is_unique:
        raise AssertionError("A raw purchase month maps to multiple adjustment levels")
    all_orders["purchase_month_adjustment"] = all_orders["purchase_month"].map(
        month_mapping
    ).astype("string")
    reviewed = all_orders.loc[all_orders["usable_review"]].copy()

    observed_group_counts = {
        str(label): int(count)
        for label, count in reviewed["promise_error_group"].value_counts(
            sort=False, dropna=False
        ).items()
    }
    if observed_group_counts != EXPECTED_REVIEWED_ERROR_GROUP_COUNTS:
        raise AssertionError(
            "Reviewed promise-error group counts changed: "
            f"{observed_group_counts}"
        )

    audit: dict[str, Any] = {
        "source_hashes": source_hashes,
        "raw_hashes": raw_hashes,
        "data_dir": str(resolved_data_dir),
        "canonical_assembler": {
            "path": str(CANONICAL_ASSEMBLER.relative_to(REPOSITORY_ROOT)),
            "sha256": source_hashes[
                "analysis/profile_pivot_phase2a/scripts/data_pipeline.py"
            ],
        },
        "canonical_audit": canonical_audit,
        "legacy_reconciliation": legacy_audit,
        "sample": {
            "canonical_delivered_orders": int(len(all_orders)),
            "canonical_unique_order_ids": int(all_orders["order_id"].nunique()),
            "reviewed_orders": int(len(reviewed)),
            "reviewed_unique_order_ids": int(reviewed["order_id"].nunique()),
            "orders_without_usable_review": int((~all_orders["usable_review"]).sum()),
            "purchase_month_min": month_min,
            "purchase_month_max": month_max,
        },
        "review_selection": {
            "raw_review_rows_total": int(len(review_records)),
            "raw_review_rows_linked": int(
                old_review_metrics["raw_review_rows_linked"]
            ),
            "selected_canonical_reviews": int(len(selected)),
            "selected_review_sha256": selected_digest,
            "orders_with_multiple_records": int(
                old_review_metrics["orders_with_multiple_review_records"]
            ),
            "orders_with_conflicting_scores": int(
                old_review_metrics["orders_with_conflicting_review_scores"]
            ),
            "low_review_2_orders": int(reviewed["low_review_2"].sum()),
            "low_review_3_orders": int(reviewed["low_review_3"].sum()),
            "reviews_before_delivery": int(
                old_review_metrics["selected_reviews_answered_before_delivery"]
            ),
            "reviews_after_promise_before_delivery": int(
                old_review_metrics[
                    "selected_reviews_after_promise_before_delivery"
                ]
            ),
            "reviews_before_delivery_and_before_promised_date": int(
                old_review_metrics[
                    "selected_reviews_before_delivery_and_before_promised_date"
                ]
            ),
            "reviews_at_or_after_delivery": int(
                reviewed["review_at_or_after_delivery"].sum()
            ),
        },
        "date_identity": {
            "missing_components": missing_date_components,
            "identity_failures": identity_failures,
            "negative_actual_delivery_days": negative_actual,
            "negative_promised_lead_days": negative_promised,
            "unbinned_actual_duration": unbinned_duration,
            "unbinned_promise_error": unbinned_error,
            "actual_delivery_days_min": int(all_orders["actual_delivery_days"].min()),
            "actual_delivery_days_max": int(all_orders["actual_delivery_days"].max()),
            "promised_lead_days_min": int(all_orders["promised_lead_days"].min()),
            "promised_lead_days_max": int(all_orders["promised_lead_days"].max()),
        },
        "month_pooling": month_audit,
        "reviewed_error_group_counts": observed_group_counts,
    }
    return all_orders.reset_index(drop=True), reviewed.reset_index(drop=True), audit


def build_sample_audit(
    all_orders: pd.DataFrame,
    reviewed: pd.DataFrame,
    audit: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the deterministic compact sample-reconciliation receipt."""

    sample = audit["sample"]
    review = audit["review_selection"]
    legacy = audit["legacy_reconciliation"]
    date = audit["date_identity"]
    assembler = audit["canonical_assembler"]
    rows = [
        ("canonical_assembler_sha256", assembler["sha256"], EXPECTED_SOURCE_SHA256["analysis/profile_pivot_phase2a/scripts/data_pipeline.py"], "sha256", "Frozen Phase2A canonical assembler."),
        ("canonical_delivered_orders", len(all_orders), EXPECTED_CANONICAL_ORDERS, "orders", "Canonical delivered analytical population."),
        ("canonical_unique_order_ids", all_orders["order_id"].nunique(), EXPECTED_CANONICAL_ORDERS, "orders", "One row per canonical order."),
        ("legacy_reconciled_orders", legacy["legacy_orders"], EXPECTED_CANONICAL_ORDERS, "orders", "Legacy RQ1 loader/target reconciliation."),
        ("legacy_id_mismatches", legacy["id_mismatches"], 0, "orders", "Per-order canonical/legacy ID comparison."),
        ("legacy_promise_error_mismatches", legacy["promise_error_mismatches"], 0, "orders", "Per-order normalised error comparison."),
        ("reviewed_orders", len(reviewed), EXPECTED_REVIEWED_ORDERS, "orders", "Exactly one selected usable review per order."),
        ("reviewed_unique_order_ids", reviewed["order_id"].nunique(), EXPECTED_REVIEWED_ORDERS, "orders", "One reviewed row per order."),
        ("orders_without_usable_review", sample["orders_without_usable_review"], 646, "orders", "Excluded only from review-outcome analysis."),
        ("raw_review_rows_linked", review["raw_review_rows_linked"], EXPECTED_RAW_REVIEW_ROWS_LINKED, "review rows", "Raw review rows linked to the canonical delivered population before order-level selection."),
        ("orders_with_multiple_review_records", review["orders_with_multiple_records"], EXPECTED_MULTIPLE_REVIEW_ORDERS, "orders", "Canonical orders with multiple linked raw review records."),
        ("orders_with_conflicting_review_scores", review["orders_with_conflicting_scores"], EXPECTED_CONFLICTING_REVIEW_ORDERS, "orders", "Canonical orders with multiple distinct valid review scores."),
        ("selected_review_sha256", review["selected_review_sha256"], EXPECTED_SELECTED_REVIEW_SHA256, "sha256", "Sorted deterministic selected-review mapping."),
        ("low_review_2_orders", review["low_review_2_orders"], EXPECTED_LOW_REVIEW_2, "orders", "Primary review_score <=2 target."),
        ("low_review_3_orders", review["low_review_3_orders"], EXPECTED_LOW_REVIEW_3, "orders", "Sensitivity review_score <=3 target."),
        ("reviews_before_delivery", review["reviews_before_delivery"], EXPECTED_REVIEWS_BEFORE_DELIVERY, "orders", "Selected review answer timestamp precedes recorded customer delivery."),
        ("reviews_after_promise_before_delivery", review["reviews_after_promise_before_delivery"], EXPECTED_REVIEWS_AFTER_PROMISE_BEFORE_DELIVERY, "orders", "Selected review follows the promised date but precedes recorded customer delivery."),
        ("reviews_before_delivery_and_before_promised_date", review["reviews_before_delivery_and_before_promised_date"], EXPECTED_REVIEWS_BEFORE_DELIVERY_AND_BEFORE_PROMISE, "orders", "Selected review precedes both recorded customer delivery and the promised date."),
        ("reviews_at_or_after_delivery", review["reviews_at_or_after_delivery"], EXPECTED_POST_DELIVERY_REVIEWS, "orders", "Timestamp-level timing sensitivity sample."),
        ("purchase_month_min", sample["purchase_month_min"], "2016-09", "YYYY-MM", "Earliest reviewed purchase month."),
        ("purchase_month_max", sample["purchase_month_max"], "2018-08", "YYYY-MM", "Latest reviewed purchase month."),
        ("date_identity_failures", date["identity_failures"], 0, "orders", "D must equal P + E."),
        ("negative_actual_delivery_days", date["negative_actual_delivery_days"], 0, "orders", "No invalid actual duration retained."),
        ("negative_promised_lead_days", date["negative_promised_lead_days"], 0, "orders", "No invalid promised lead retained."),
    ]
    result = pd.DataFrame(
        rows, columns=["metric", "value", "expected", "unit", "definition"]
    )
    result["status"] = np.where(
        result["value"].astype(str).eq(result["expected"].astype(str)), "PASS", "FAIL"
    )
    return result[["metric", "value", "expected", "unit", "status", "definition"]]


def build_date_identity_audit(all_orders: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable identity row for every canonical delivered order."""

    result = all_orders[
        [
            "order_id",
            "purchase_date",
            "actual_delivery_date",
            "estimated_delivery_date",
            "actual_delivery_days",
            "promised_lead_days",
            "promise_error_days",
        ]
    ].copy()
    result["identity_rhs_days"] = (
        result["promised_lead_days"] + result["promise_error_days"]
    )
    result["identity_residual_days"] = (
        result["actual_delivery_days"] - result["identity_rhs_days"]
    )
    result["identity_holds"] = result["identity_residual_days"].eq(0)
    result["missing_component"] = result[
        ["purchase_date", "actual_delivery_date", "estimated_delivery_date"]
    ].isna().any(axis=1)
    result["negative_actual_duration"] = result["actual_delivery_days"].lt(0)
    result["negative_promised_lead"] = result["promised_lead_days"].lt(0)
    return result.sort_values("order_id", kind="mergesort").reset_index(drop=True)


def _coverage_summary(
    subset: pd.DataFrame,
    *,
    dimension: str,
    actual_duration_group: str | None = None,
    promise_error_group: str | None = None,
    purchase_month: str | None = None,
    review_status: str = "all",
) -> dict[str, Any]:
    reviewed_mask = subset["usable_review"].astype(bool)
    total = int(len(subset))
    reviewed_orders = int(reviewed_mask.sum())
    return {
        "dimension": dimension,
        "actual_duration_group": actual_duration_group,
        "promise_error_group": promise_error_group,
        "purchase_month": purchase_month,
        "review_status": review_status,
        "analytical_orders": total,
        "reviewed_orders": reviewed_orders,
        "missing_review_orders": total - reviewed_orders,
        "review_coverage": reviewed_orders / total if total else np.nan,
        "mean_actual_delivery_days": subset["actual_delivery_days"].mean(),
        "median_actual_delivery_days": subset["actual_delivery_days"].median(),
        "mean_promised_lead_days": subset["promised_lead_days"].mean(),
        "median_promised_lead_days": subset["promised_lead_days"].median(),
        "mean_promise_error_days": subset["promise_error_days"].mean(),
        "median_promise_error_days": subset["promise_error_days"].median(),
    }


def build_review_coverage(all_orders: pd.DataFrame) -> pd.DataFrame:
    """Return long-format review coverage and reviewed/non-reviewed diagnostics."""

    rows = [_coverage_summary(all_orders, dimension="overall")]
    for label in DEFAULT_DURATION_LABELS:
        subset = all_orders.loc[
            all_orders["actual_duration_group"].astype("string").eq(label)
        ]
        rows.append(
            _coverage_summary(
                subset, dimension="actual_duration_group", actual_duration_group=label
            )
        )
    for label in ERROR_GROUP_LABELS:
        subset = all_orders.loc[
            all_orders["promise_error_group"].astype("string").eq(label)
        ]
        rows.append(
            _coverage_summary(
                subset, dimension="promise_error_group", promise_error_group=label
            )
        )
    for duration_label, error_label in product(
        DEFAULT_DURATION_LABELS, ERROR_GROUP_LABELS
    ):
        subset = all_orders.loc[
            all_orders["actual_duration_group"].astype("string").eq(duration_label)
            & all_orders["promise_error_group"].astype("string").eq(error_label)
        ]
        rows.append(
            _coverage_summary(
                subset,
                dimension="duration_x_error",
                actual_duration_group=duration_label,
                promise_error_group=error_label,
            )
        )
    for month in sorted(all_orders["purchase_month"].astype(str).unique()):
        subset = all_orders.loc[all_orders["purchase_month"].astype(str).eq(month)]
        rows.append(
            _coverage_summary(
                subset, dimension="purchase_month", purchase_month=month
            )
        )
    for review_status, mask in [
        ("reviewed", all_orders["usable_review"].astype(bool)),
        ("not_reviewed", ~all_orders["usable_review"].astype(bool)),
    ]:
        rows.append(
            _coverage_summary(
                all_orders.loc[mask],
                dimension="review_status",
                review_status=review_status,
            )
        )
    return pd.DataFrame(rows)


def build_duration_error_cell_counts(
    all_orders: pd.DataFrame,
    config: Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    """Return the complete 5-by-8 cell population and review-coverage counts."""

    frozen = _load_config(config)
    threshold = int(frozen["actual_duration_groups"]["low_support_threshold"])
    rows: list[dict[str, Any]] = []
    for duration_label, error_label in product(
        frozen["actual_duration_groups"]["labels"],
        frozen["promise_error_groups"]["labels"],
    ):
        subset = all_orders.loc[
            all_orders["actual_duration_group"].astype("string").eq(duration_label)
            & all_orders["promise_error_group"].astype("string").eq(error_label)
        ]
        reviewed_orders = int(subset["usable_review"].astype(bool).sum())
        analytical_orders = int(len(subset))
        rows.append(
            {
                "actual_duration_group": duration_label,
                "promise_error_group": error_label,
                "analytical_orders": analytical_orders,
                "reviewed_orders": reviewed_orders,
                "missing_review_orders": analytical_orders - reviewed_orders,
                "review_coverage": (
                    reviewed_orders / analytical_orders
                    if analytical_orders
                    else np.nan
                ),
                "low_support_cell": reviewed_orders < threshold,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 40 or int(result["analytical_orders"].sum()) != len(all_orders):
        raise AssertionError("Duration-by-error cell-count table is not a complete partition")
    return result


def build_duration_error_review_rates(
    reviewed: pd.DataFrame,
    config: Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    """Return frozen two-way review summaries with Wilson intervals."""

    frozen = _load_config(config)
    threshold = int(frozen["actual_duration_groups"]["low_support_threshold"])
    rows: list[dict[str, Any]] = []
    for duration_label, error_label in product(
        frozen["actual_duration_groups"]["labels"],
        frozen["promise_error_groups"]["labels"],
    ):
        subset = reviewed.loc[
            reviewed["actual_duration_group"].astype("string").eq(duration_label)
            & reviewed["promise_error_group"].astype("string").eq(error_label)
        ]
        n_orders = int(len(subset))
        low_reviews = int(subset["low_review_2"].sum())
        lower, upper = wilson_interval(low_reviews, n_orders)
        one_star_orders = int(subset["one_star"].sum())
        rows.append(
            {
                "actual_duration_group": duration_label,
                "promise_error_group": error_label,
                "reviewed_orders": n_orders,
                "low_review_2_orders": low_reviews,
                "low_review_2_rate": low_reviews / n_orders if n_orders else np.nan,
                "low_review_2_ci_lower": lower,
                "low_review_2_ci_upper": upper,
                "mean_review_score": subset["review_score"].mean(),
                "median_review_score": subset["review_score"].median(),
                "one_star_orders": one_star_orders,
                "one_star_share": one_star_orders / n_orders if n_orders else np.nan,
                "low_support_cell": n_orders < threshold,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 40 or int(result["reviewed_orders"].sum()) != len(reviewed):
        raise AssertionError("Duration-by-error review-rate table is not a complete partition")
    return result


def build_data_audit_tables(
    all_orders: pd.DataFrame,
    reviewed: pd.DataFrame,
    audit: Mapping[str, Any],
    config: Mapping[str, Any] | str | Path,
) -> dict[str, pd.DataFrame]:
    """Return all required data-stage CSV payloads without writing them."""

    return {
        "RQ1_SAMPLE_AUDIT.csv": build_sample_audit(all_orders, reviewed, audit),
        "RQ1_DATE_IDENTITY_AUDIT.csv": build_date_identity_audit(all_orders),
        "RQ1_REVIEW_COVERAGE.csv": build_review_coverage(all_orders),
        "RQ1_DURATION_ERROR_CELL_COUNTS.csv": build_duration_error_cell_counts(
            all_orders, config
        ),
        "RQ1_DURATION_ERROR_REVIEW_RATES.csv": build_duration_error_review_rates(
            reviewed, config
        ),
    }
