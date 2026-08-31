"""Frozen purchase-time feature construction for the order-level experiment.

The input is the exact canonical Phase 2A assembler frame.  This module does
not aggregate raw Olist tables, alter the canonical issued-lead-time field, fit
preprocessing, or construct historical profiles.  It adds only the explicitly
frozen multi-seller and deterministic calendar/event fields used by
``order_breach_severity_v1``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.events import add_event_features, brazil_event_calendar


EXPECTED_HOLIDAYS_VERSION = "0.76"

PROMISE_NUMERIC_FEATURES = ("promised_delivery_days",)

CANONICAL_CONTEXT_NUMERIC_FEATURES = (
    "n_items",
    "n_unique_products",
    "n_unique_sellers",
    "multi_item",
    "multi_product",
    "total_price",
    "total_freight_value",
    "freight_to_price_ratio",
    "avg_product_weight_g",
    "avg_product_volume_cm3",
    "max_product_dimension_cm",
    "category_diversity",
    "customer_seller_same_state",
    "distance_km",
    "purchase_month_num",
    "purchase_weekday",
    "purchase_hour",
    "is_weekend_purchase",
)

# These are the active, previously executed static-calendar transforms.  The
# separate Carnival field is retained by the frozen config as an explicit
# all-zero audit field; it did not represent an active exposure in Phase 1.
ACTIVE_EVENT_NUMERIC_FEATURES = (
    "days_to_next_holiday",
    "days_since_previous_holiday",
    "holiday_within_next_3d",
    "holiday_within_next_7d",
    "holiday_in_promise_window",
    "num_holidays_in_promise_window",
    "num_weekend_days_in_promise_window",
    "black_friday_period",
    "christmas_new_year_period",
)
INACTIVE_AUDIT_EVENT_FEATURES = ("carnival_period",)
KNOWN_EVENT_COMPONENTS = (
    "holiday_within_next_7d",
    "black_friday_period",
    "christmas_new_year_period",
    "carnival_period",
)

CONTEXT_NUMERIC_FEATURES = (
    "n_items",
    "n_unique_products",
    "n_unique_sellers",
    "multi_item",
    "multi_product",
    "multi_seller",
    "total_price",
    "total_freight_value",
    "freight_to_price_ratio",
    "avg_product_weight_g",
    "avg_product_volume_cm3",
    "max_product_dimension_cm",
    "category_diversity",
    "customer_seller_same_state",
    "distance_km",
    "purchase_month_num",
    "purchase_weekday",
    "purchase_hour",
    "is_weekend_purchase",
    *ACTIVE_EVENT_NUMERIC_FEATURES,
    *INACTIVE_AUDIT_EVENT_FEATURES,
    "known_event_indicator",
)

CONTEXT_CATEGORICAL_FEATURES = (
    "customer_state",
    "main_seller_state",
    "main_product_category",
    "route_region",
    "distance_band",
)

CURRENT_ORDER_NUMERIC_FEATURES = PROMISE_NUMERIC_FEATURES + CONTEXT_NUMERIC_FEATURES
CURRENT_ORDER_CATEGORICAL_FEATURES = CONTEXT_CATEGORICAL_FEATURES
CURRENT_ORDER_FEATURES = CURRENT_ORDER_NUMERIC_FEATURES + CURRENT_ORDER_CATEGORICAL_FEATURES

REQUIRED_CANONICAL_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            *PROMISE_NUMERIC_FEATURES,
            *CANONICAL_CONTEXT_NUMERIC_FEATURES,
            *CONTEXT_CATEGORICAL_FEATURES,
        )
    )
)

# Exact column names that may exist on the canonical analytical frame but may
# never enter a purchase-time predictor list.
FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "order_id",
        "customer_id",
        "customer_unique_id",
        "main_seller_id",
        "seller_id",
        "main_product_id",
        "product_id",
        "state_od",
        "route_state",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_status",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "shipping_limit_date",
        "approval_delay",
        "purchase_to_carrier",
        "post_approval_handling",
        "handling_duration",
        "transit_time",
        "transit_duration",
        "total_delivery_time",
        "promise_error_days",
        "late_delivery",
        "severe_late_2d",
        "positive_late_days",
        "review_score",
        "n_reviews",
        "total_payment_value",
        "n_payment_installments",
        "main_payment_type",
        "payment_value",
        "payment_installments",
        "payment_type",
        "realised_daily_order_count",
        "realised_daily_gmv",
        "retrospective_hrd_label",
        "hrd_label",
    }
)

FORBIDDEN_FEATURE_PREFIXES = (
    "review_",
    "selected_review_",
    "payment_",
    "order_delivered_",
    "label_available_",
    "final_breach_available_",
    "late_delivery_available_",
    "severe_late_2d_available_",
    "positive_late_days_available_",
    "post_approval_handling_available_",
    "purchase_to_carrier_available_",
    "handling_available_",
    "transit_available_",
    "realised_daily_",
)


@dataclass(frozen=True)
class FeatureAvailability:
    """Machine-readable information-time declaration for one predictor."""

    feature: str
    feature_type: str
    block: str
    available_at: str = "purchase_or_promise_decision_proxy"


FEATURE_AVAILABILITY = {
    feature: FeatureAvailability(
        feature=feature,
        feature_type=(
            "categorical" if feature in CURRENT_ORDER_CATEGORICAL_FEATURES else "numeric"
        ),
        block=("promise" if feature in PROMISE_NUMERIC_FEATURES else "current_context"),
    )
    for feature in CURRENT_ORDER_FEATURES
}


def forbidden_feature_violations(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Return forbidden purchase-time predictor names in stable lexical order."""

    violations: set[str] = set()
    for feature in feature_names:
        name = str(feature)
        lower = name.lower()
        if lower in FORBIDDEN_FEATURE_COLUMNS or lower.startswith(FORBIDDEN_FEATURE_PREFIXES):
            violations.add(name)
    return tuple(sorted(violations))


def validate_no_forbidden_features(feature_names: Iterable[str]) -> None:
    """Hard-stop if a proposed predictor list contains a leakage/identity field."""

    violations = forbidden_feature_violations(feature_names)
    if violations:
        raise ValueError(f"Forbidden current-order predictors: {list(violations)}")


def validate_feature_availability(feature_names: Iterable[str]) -> None:
    """Require every proposed predictor to have the frozen purchase-time status."""

    names = tuple(str(feature) for feature in feature_names)
    validate_no_forbidden_features(names)
    unknown = tuple(sorted(set(names) - set(FEATURE_AVAILABILITY)))
    if unknown:
        raise ValueError(f"Predictors absent from the frozen availability contract: {list(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError("Predictor list contains duplicate feature names")


def validate_feature_contract(
    numeric: Iterable[str] = CURRENT_ORDER_NUMERIC_FEATURES,
    categorical: Iterable[str] = CURRENT_ORDER_CATEGORICAL_FEATURES,
    *,
    require_full_current_order_block: bool = True,
) -> None:
    """Validate types, overlap, availability, and the frozen full-block schema."""

    numeric_names = tuple(str(feature) for feature in numeric)
    categorical_names = tuple(str(feature) for feature in categorical)
    overlap = tuple(sorted(set(numeric_names) & set(categorical_names)))
    if overlap:
        raise ValueError(f"Numeric/categorical feature overlap: {list(overlap)}")
    validate_feature_availability(numeric_names + categorical_names)
    mistyped_numeric = tuple(
        feature for feature in numeric_names if FEATURE_AVAILABILITY[feature].feature_type != "numeric"
    )
    mistyped_categorical = tuple(
        feature
        for feature in categorical_names
        if FEATURE_AVAILABILITY[feature].feature_type != "categorical"
    )
    if mistyped_numeric or mistyped_categorical:
        raise ValueError(
            "Feature type mismatch: "
            f"numeric={list(mistyped_numeric)}, categorical={list(mistyped_categorical)}"
        )
    if require_full_current_order_block:
        if numeric_names != CURRENT_ORDER_NUMERIC_FEATURES:
            raise ValueError("Numeric features differ from ORDER_FROZEN_CONFIG order/content")
        if categorical_names != CURRENT_ORDER_CATEGORICAL_FEATURES:
            raise ValueError("Categorical features differ from ORDER_FROZEN_CONFIG order/content")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = tuple(column for column in columns if column not in frame.columns)
    if missing:
        raise KeyError(f"Canonical frame is missing required columns: {list(missing)}")


def _assert_exact_numeric_equal(actual: pd.Series, expected: pd.Series, feature: str) -> None:
    left = pd.to_numeric(actual, errors="coerce")
    right = pd.to_numeric(expected, errors="coerce")
    if left.isna().any() or right.isna().any() or not left.eq(right).all():
        raise AssertionError(f"Canonical {feature} is inconsistent with its frozen definition")


def build_current_order_features(canonical_frame: pd.DataFrame) -> pd.DataFrame:
    """Add the frozen purchase-time context fields to a canonical order frame.

    The returned frame retains every input row and column.  No imputation is
    performed here: missing-value handling belongs inside each training-only
    model pipeline.  In particular, ``promised_delivery_days`` is copied from
    the canonical assembler and is never recomputed from normalised dates.
    """

    _require_columns(canonical_frame, REQUIRED_CANONICAL_INPUT_COLUMNS)
    validate_feature_contract()

    result = canonical_frame.copy()
    canonical_promise = result["promised_delivery_days"].copy(deep=True)
    purchase = pd.to_datetime(result["order_purchase_timestamp"], errors="coerce")
    promise = pd.to_datetime(result["order_estimated_delivery_date"], errors="coerce")
    if purchase.isna().any() or promise.isna().any():
        raise AssertionError("Canonical frame contains a missing purchase or issued-promise timestamp")

    seller_count = pd.to_numeric(result["n_unique_sellers"], errors="coerce")
    if seller_count.isna().any() or seller_count.lt(0).any():
        raise AssertionError("Canonical n_unique_sellers must be finite and nonnegative")
    result["multi_seller"] = seller_count.gt(1).astype("int8")

    _assert_exact_numeric_equal(
        result["multi_item"],
        pd.to_numeric(result["n_items"], errors="coerce").gt(1).astype("int8"),
        "multi_item",
    )
    _assert_exact_numeric_equal(
        result["multi_product"],
        pd.to_numeric(result["n_unique_products"], errors="coerce").gt(1).astype("int8"),
        "multi_product",
    )
    _assert_exact_numeric_equal(result["purchase_month_num"], purchase.dt.month, "purchase_month_num")
    _assert_exact_numeric_equal(result["purchase_weekday"], purchase.dt.weekday, "purchase_weekday")
    _assert_exact_numeric_equal(result["purchase_hour"], purchase.dt.hour, "purchase_hour")
    _assert_exact_numeric_equal(
        result["is_weekend_purchase"], purchase.dt.weekday.ge(5).astype("int8"), "is_weekend_purchase"
    )

    first_year = int(purchase.dt.year.min())
    last_year = int(max(purchase.dt.year.max(), promise.dt.year.max()))
    calendar = brazil_event_calendar(list(range(first_year, last_year + 1)))
    if calendar.empty or not calendar["source_version"].eq(EXPECTED_HOLIDAYS_VERSION).all():
        versions = sorted(calendar.get("source_version", pd.Series(dtype=str)).dropna().astype(str).unique())
        raise AssertionError(
            f"Frozen calendar requires holidays=={EXPECTED_HOLIDAYS_VERSION}; observed {versions}"
        )
    event_input = pd.DataFrame(
        {
            "order_purchase_timestamp": purchase,
            "order_estimated_delivery_date": promise,
        },
        index=result.index,
    )
    event_frame = add_event_features(event_input, calendar)
    generated_event_fields = ACTIVE_EVENT_NUMERIC_FEATURES + INACTIVE_AUDIT_EVENT_FEATURES
    for feature in generated_event_fields:
        result[feature] = event_frame[feature].to_numpy()

    generated_carnival = pd.to_numeric(result["carnival_period"], errors="coerce")
    if generated_carnival.isna().any() or generated_carnival.ne(0).any():
        raise AssertionError(
            "Frozen PUBLIC Brazil calendar must yield an identically-zero Carnival audit field"
        )
    result["carnival_period"] = np.int8(0)

    event_components = result.loc[:, KNOWN_EVENT_COMPONENTS].apply(
        pd.to_numeric, errors="coerce"
    )
    if event_components.isna().any().any():
        raise AssertionError("Known-event components contain missing values")
    if not event_components.isin([0, 1]).all().all():
        raise AssertionError("Known-event components must be binary")
    result["known_event_indicator"] = event_components.max(axis=1).astype("int8")

    pd.testing.assert_series_equal(
        result["promised_delivery_days"],
        canonical_promise,
        check_dtype=True,
        check_exact=True,
        check_names=True,
    )
    validate_built_current_order_features(result)
    return result


def validate_built_current_order_features(frame: pd.DataFrame) -> None:
    """Validate the constructed full current-order block without fitting data."""

    _require_columns(frame, CURRENT_ORDER_FEATURES)
    validate_feature_contract()
    carnival = pd.to_numeric(frame["carnival_period"], errors="coerce")
    if carnival.isna().any() or carnival.ne(0).any():
        raise AssertionError("carnival_period must remain the frozen all-zero audit field")
    seller_count = pd.to_numeric(frame["n_unique_sellers"], errors="coerce")
    expected_multi_seller = seller_count.gt(1).astype("int8")
    _assert_exact_numeric_equal(frame["multi_seller"], expected_multi_seller, "multi_seller")
    expected_known_event = frame.loc[:, KNOWN_EVENT_COMPONENTS].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)
    _assert_exact_numeric_equal(
        frame["known_event_indicator"], expected_known_event, "known_event_indicator"
    )


__all__ = [
    "ACTIVE_EVENT_NUMERIC_FEATURES",
    "CANONICAL_CONTEXT_NUMERIC_FEATURES",
    "CONTEXT_CATEGORICAL_FEATURES",
    "CONTEXT_NUMERIC_FEATURES",
    "CURRENT_ORDER_CATEGORICAL_FEATURES",
    "CURRENT_ORDER_FEATURES",
    "CURRENT_ORDER_NUMERIC_FEATURES",
    "EXPECTED_HOLIDAYS_VERSION",
    "FEATURE_AVAILABILITY",
    "FORBIDDEN_FEATURE_COLUMNS",
    "INACTIVE_AUDIT_EVENT_FEATURES",
    "KNOWN_EVENT_COMPONENTS",
    "PROMISE_NUMERIC_FEATURES",
    "REQUIRED_CANONICAL_INPUT_COLUMNS",
    "build_current_order_features",
    "forbidden_feature_violations",
    "validate_built_current_order_features",
    "validate_feature_availability",
    "validate_feature_contract",
    "validate_no_forbidden_features",
]
