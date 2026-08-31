#!/usr/bin/env python3
"""Leakage-safe Phase 2A Olist data assembly and as-of history helpers.

This module deliberately contains no model fitting, calibration, outer scoring,
or result persistence.  The command-line interface supports only a read-only
``--audit-only`` mode that assembles the canonical order base in memory,
verifies frozen reference counts, and prints a JSON audit to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

ORDER_TIME_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)

B0_NUMERIC = (
    "promised_delivery_days",
    "n_items",
    "n_unique_products",
    "n_unique_sellers",
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
    "multi_item",
    "multi_product",
)

B0_CATEGORICAL = (
    "customer_state",
    "main_seller_state",
    "main_product_category",
    "route_region",
    "distance_band",
)

B0_COLUMNS = B0_NUMERIC + B0_CATEGORICAL

SUPPORT_ONLY_IDS = ("main_product_id", "customer_unique_id")

REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North",
    "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast",
    "MA": "Northeast", "PB": "Northeast", "PE": "Northeast",
    "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MS": "Central-West",
    "MT": "Central-West", "ES": "Southeast", "MG": "Southeast",
    "RJ": "Southeast", "SP": "Southeast", "PR": "South",
    "RS": "South", "SC": "South",
}

SAMPLE_SINGLE_SELLER = "single_seller"
SAMPLE_DETERMINISTIC_MAIN_SELLER = "all_orders_main_seller"

WINDOW_H0 = "H0_expanding"
WINDOW_H1 = "H1_180d"
ALLOWED_REPORTING_LAGS = (0, 30)


@dataclass(frozen=True)
class OutcomeSpec:
    """Column contract for a target-specific historical information time."""

    value_column: str
    availability_column: str
    valid_column: str
    kind: str


OUTCOME_SPECS = {
    "late_delivery": OutcomeSpec(
        "late_delivery", "late_delivery_available_at", "late_delivery_valid", "binary"
    ),
    "severe_late_2d": OutcomeSpec(
        "severe_late_2d", "severe_late_2d_available_at", "severe_late_2d_valid", "binary"
    ),
    "positive_late_days": OutcomeSpec(
        "positive_late_days", "positive_late_days_available_at", "positive_late_days_valid", "continuous"
    ),
    "post_approval_handling": OutcomeSpec(
        "post_approval_handling",
        "post_approval_handling_available_at",
        "post_approval_handling_valid",
        "continuous",
    ),
    "purchase_to_carrier": OutcomeSpec(
        "purchase_to_carrier",
        "purchase_to_carrier_available_at",
        "purchase_to_carrier_valid",
        "continuous",
    ),
    "transit_time": OutcomeSpec(
        "transit_time", "transit_time_available_at", "transit_time_valid", "continuous"
    ),
}

REFERENCE_COUNTS = {
    "canonical_orders": 96_470,
    "unique_order_ids": 96_470,
    "late_events": 6_534,
    "severe_events": 5_709,
    "single_seller_orders": 95_195,
    "deterministic_main_seller_orders": 96_470,
    "single_seller_levels": 2_948,
    "single_seller_route_levels": 408,
    "negative_post_approval_handling": 1_350,
    "negative_purchase_to_carrier": 165,
    "negative_transit_time": 23,
    "valid_post_approval_handling": 95_105,
    "valid_transit_time": 96_446,
    "missing_distance_km": 480,
    "missing_main_product_category": 1_332,
    "missing_product_physics": 16,
}


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Resolve a directory containing the seven raw tables used here."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("OLIST_DATA_DIR"):
        candidates.append(Path(os.environ["OLIST_DATA_DIR"]))
    candidates.extend((Path("data/olist_data"), Path("olist_data")))
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in RAW_FILES.values()):
            return candidate.resolve()
    required = ", ".join(RAW_FILES.values())
    raise FileNotFoundError(
        f"Olist raw tables not found. Pass --data-dir or set OLIST_DATA_DIR. Required: {required}"
    )


def read_raw_tables(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Read only the raw tables required for the frozen Phase 2A order base."""

    root = resolve_data_dir(data_dir)
    return {
        key: pd.read_csv(root / filename, encoding="utf-8-sig", low_memory=False)
        for key, filename in RAW_FILES.items()
    }


def raw_file_sha256s(data_dir: str | Path | None = None) -> dict[str, str]:
    """Return raw-input hashes without modifying source files."""

    root = resolve_data_dir(data_dir)
    hashes: dict[str, str] = {}
    for key, filename in RAW_FILES.items():
        digest = hashlib.sha256()
        with (root / filename).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[key] = digest.hexdigest()
    return hashes


def mode_deterministic(series: pd.Series):
    """Return the modal non-null string with a lexical tie-break."""

    values = series.dropna().astype(str)
    if values.empty:
        return pd.NA
    counts = values.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance using the audited Earth radius."""

    radius = 6371.0088
    a1, o1, a2, o2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = a2 - a1, o2 - o1
    a = np.sin(dlat / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def _require_unique(table: pd.DataFrame, key: str, name: str) -> None:
    if table[key].isna().any() or table[key].duplicated().any():
        raise AssertionError(f"{name}.{key} must be non-null and unique")


def _max_timestamp_strict(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Row maximum that remains missing unless every required time is present."""

    return frame[columns].max(axis=1, skipna=False)


def assemble_order_base(raw: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the frozen one-row-per-order base directly from raw Olist tables.

    The construction matches the audited feasibility assembler where that
    assembler is authoritative for B0: item-level volume is averaged after
    multiplication, maximum dimension is an item-level maximum, seller
    geography is joined from the deterministically selected seller, and ZIP
    coordinates use valid-coordinate medians.
    """

    missing_tables = set(RAW_FILES) - set(raw)
    if missing_tables:
        raise KeyError(f"Missing raw tables: {sorted(missing_tables)}")

    orders = raw["orders"].copy()
    customers = raw["customers"].copy()
    items = raw["items"].copy()
    products = raw["products"].copy()
    sellers = raw["sellers"].copy()
    categories = raw["categories"].copy()
    geolocation = raw["geolocation"].copy()

    _require_unique(orders, "order_id", "orders")
    _require_unique(customers, "customer_id", "customers")
    _require_unique(products, "product_id", "products")
    _require_unique(sellers, "seller_id", "sellers")
    _require_unique(categories, "product_category_name", "categories")

    for column in ORDER_TIME_COLUMNS:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    canonical_mask = (
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    )
    delivered = orders.loc[canonical_mask].copy()
    if delivered["order_purchase_timestamp"].isna().any():
        raise AssertionError("Canonical delivered orders contain missing purchase timestamps")

    product_lookup = products.merge(
        categories,
        on="product_category_name",
        how="left",
        validate="m:1",
    )
    product_lookup["product_category"] = product_lookup[
        "product_category_name_english"
    ].fillna(product_lookup["product_category_name"])

    enriched_items = items.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="m:1",
        indicator="_product_join",
    )
    if (enriched_items["_product_join"] != "both").any():
        raise AssertionError("At least one item product_id is absent from products")
    enriched_items = enriched_items.drop(columns="_product_join")
    if (~enriched_items["seller_id"].isin(sellers["seller_id"])).any():
        raise AssertionError("At least one item seller_id is absent from sellers")

    enriched_items["product_volume_cm3"] = (
        enriched_items["product_length_cm"]
        * enriched_items["product_height_cm"]
        * enriched_items["product_width_cm"]
    )
    enriched_items["product_max_dimension_cm"] = enriched_items[
        ["product_length_cm", "product_height_cm", "product_width_cm"]
    ].max(axis=1)

    item_agg = enriched_items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        n_unique_products=("product_id", "nunique"),
        n_unique_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        main_seller_id=("seller_id", mode_deterministic),
        main_product_id=("product_id", mode_deterministic),
        main_product_category=("product_category", mode_deterministic),
        category_diversity=("product_category", "nunique"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_volume_cm3=("product_volume_cm3", "mean"),
        max_product_dimension_cm=("product_max_dimension_cm", "max"),
    )

    seller_lookup = sellers[
        ["seller_id", "seller_state", "seller_zip_code_prefix"]
    ].rename(
        columns={
            "seller_id": "main_seller_id",
            "seller_state": "main_seller_state",
            "seller_zip_code_prefix": "main_seller_zip",
        }
    )
    item_agg = item_agg.merge(
        seller_lookup,
        on="main_seller_id",
        how="left",
        validate="m:1",
    )

    frame = delivered.merge(
        customers,
        on="customer_id",
        how="left",
        validate="m:1",
        indicator="_customer_join",
    )
    if (frame["_customer_join"] != "both").any():
        raise AssertionError("At least one canonical order has no customer row")
    frame = frame.drop(columns="_customer_join").merge(
        item_agg,
        on="order_id",
        how="left",
        validate="1:1",
        indicator="_item_join",
    )
    if (frame["_item_join"] != "both").any():
        raise AssertionError("At least one canonical order has no item aggregate")
    frame = frame.drop(columns="_item_join")

    geo_lat = pd.to_numeric(geolocation["geolocation_lat"], errors="coerce")
    geo_lon = pd.to_numeric(geolocation["geolocation_lng"], errors="coerce")
    valid_coordinate = geo_lat.between(-34.5, 6.0) & geo_lon.between(-74.5, -30.0)
    valid_geo = geolocation.loc[valid_coordinate].copy()
    valid_geo["geolocation_lat"] = geo_lat.loc[valid_coordinate]
    valid_geo["geolocation_lng"] = geo_lon.loc[valid_coordinate]
    geo_centroid = valid_geo.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        geo_lat=("geolocation_lat", "median"),
        geo_lng=("geolocation_lng", "median"),
    )

    frame["main_seller_zip"] = pd.to_numeric(
        frame["main_seller_zip"], errors="coerce"
    ).astype("Int64")
    frame["customer_zip_code_prefix"] = pd.to_numeric(
        frame["customer_zip_code_prefix"], errors="coerce"
    ).astype("Int64")
    seller_geo = geo_centroid.rename(
        columns={
            "geolocation_zip_code_prefix": "main_seller_zip",
            "geo_lat": "seller_lat",
            "geo_lng": "seller_lng",
        }
    )
    customer_geo = geo_centroid.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geo_lat": "customer_lat",
            "geo_lng": "customer_lng",
        }
    )
    frame = frame.merge(seller_geo, on="main_seller_zip", how="left", validate="m:1")
    frame = frame.merge(
        customer_geo,
        on="customer_zip_code_prefix",
        how="left",
        validate="m:1",
    )
    complete_coordinates = frame[
        ["seller_lat", "seller_lng", "customer_lat", "customer_lng"]
    ].notna().all(axis=1)
    frame["distance_km"] = np.nan
    frame.loc[complete_coordinates, "distance_km"] = haversine_km(
        frame.loc[complete_coordinates, "seller_lat"],
        frame.loc[complete_coordinates, "seller_lng"],
        frame.loc[complete_coordinates, "customer_lat"],
        frame.loc[complete_coordinates, "customer_lng"],
    )
    frame.loc[~frame["distance_km"].between(0, 5000), "distance_km"] = np.nan

    purchase = frame["order_purchase_timestamp"]
    frame["purchase_month_start"] = purchase.dt.to_period("M").dt.to_timestamp()
    frame["purchase_month"] = frame["purchase_month_start"].dt.strftime("%Y-%m")
    frame["purchase_month_num"] = purchase.dt.month
    frame["purchase_weekday"] = purchase.dt.weekday
    frame["purchase_hour"] = purchase.dt.hour
    frame["is_weekend_purchase"] = purchase.dt.weekday.ge(5).astype("int8")
    frame["promised_delivery_days"] = (
        frame["order_estimated_delivery_date"] - purchase
    ).dt.total_seconds() / 86_400
    frame["freight_to_price_ratio"] = frame["total_freight_value"] / frame[
        "total_price"
    ].replace(0, np.nan)
    frame["is_single_seller"] = frame["n_unique_sellers"].eq(1)
    frame["is_single_product"] = frame["n_unique_products"].eq(1)
    frame["is_single_item"] = frame["n_items"].eq(1)
    frame["multi_item"] = frame["n_items"].gt(1).astype("int8")
    frame["multi_product"] = frame["n_unique_products"].gt(1).astype("int8")
    frame["has_deterministic_main_seller"] = frame["main_seller_id"].notna()

    frame["seller_region"] = frame["main_seller_state"].map(REGION).fillna("Unknown")
    frame["customer_region"] = frame["customer_state"].map(REGION).fillna("Unknown")
    frame["route_region"] = frame["seller_region"] + " -> " + frame["customer_region"]
    seller_state = frame["main_seller_state"].astype("string").fillna("Unknown")
    customer_state = frame["customer_state"].astype("string").fillna("Unknown")
    frame["route_state"] = seller_state + " -> " + customer_state
    both_states = frame["main_seller_state"].notna() & frame["customer_state"].notna()
    frame["customer_seller_same_state"] = (
        both_states & frame["main_seller_state"].eq(frame["customer_state"])
    ).astype("int8")
    frame["distance_band"] = pd.cut(
        frame["distance_km"],
        [-np.inf, 50, 150, 500, 1000, 2000, np.inf],
        labels=["0-50", "50-150", "150-500", "500-1000", "1000-2000", ">2000"],
    ).astype("string").fillna("unknown")

    actual_date = frame["order_delivered_customer_date"].dt.normalize()
    promise_date = frame["order_estimated_delivery_date"].dt.normalize()
    frame["promise_error_days"] = (actual_date - promise_date).dt.days.astype("int64")
    frame["late_delivery"] = frame["promise_error_days"].gt(0).astype("int8")
    frame["severe_late_2d"] = frame["promise_error_days"].ge(2).astype("int8")
    frame["positive_late_days"] = frame["promise_error_days"].clip(lower=0).astype("int64")

    frame["purchase_to_carrier"] = (
        frame["order_delivered_carrier_date"] - purchase
    ).dt.total_seconds() / 86_400
    frame["post_approval_handling"] = (
        frame["order_delivered_carrier_date"] - frame["order_approved_at"]
    ).dt.total_seconds() / 86_400
    frame["transit_time"] = (
        frame["order_delivered_customer_date"] - frame["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86_400

    actual = frame["order_delivered_customer_date"]
    frame["late_delivery_available_at"] = actual
    frame["severe_late_2d_available_at"] = actual
    frame["positive_late_days_available_at"] = actual
    frame["post_approval_handling_available_at"] = _max_timestamp_strict(
        frame, ["order_approved_at", "order_delivered_carrier_date"]
    )
    frame["purchase_to_carrier_available_at"] = frame["order_delivered_carrier_date"]
    frame["transit_time_available_at"] = _max_timestamp_strict(
        frame, ["order_delivered_carrier_date", "order_delivered_customer_date"]
    )
    # Stable runner/profile aliases.  The long names above remain the explicit
    # target-to-availability contract used by OUTCOME_SPECS.
    frame["late_available_at"] = frame["late_delivery_available_at"]
    frame["handling_available_at"] = frame["post_approval_handling_available_at"]
    frame["transit_available_at"] = frame["transit_time_available_at"]

    frame["late_delivery_valid"] = frame["late_delivery_available_at"].notna()
    frame["severe_late_2d_valid"] = frame["severe_late_2d_available_at"].notna()
    frame["positive_late_days_valid"] = frame["positive_late_days_available_at"].notna()
    frame["purchase_to_carrier_valid"] = (
        frame["purchase_to_carrier_available_at"].notna()
        & frame["purchase_to_carrier"].ge(0)
    )
    frame["post_approval_handling_valid"] = (
        frame["post_approval_handling_available_at"].notna()
        & frame["post_approval_handling"].ge(0)
    )
    frame["transit_time_valid"] = (
        frame["transit_time_available_at"].notna() & frame["transit_time"].ge(0)
    )

    if frame["order_id"].duplicated().any():
        raise AssertionError("Order-level assembly produced duplicate order IDs")
    return frame.reset_index(drop=True)


def build_order_base(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read raw inputs and return the in-memory canonical order base."""

    return assemble_order_base(read_raw_tables(data_dir))


def get_outcome_spec(outcome: str) -> OutcomeSpec:
    try:
        return OUTCOME_SPECS[outcome]
    except KeyError as exc:
        raise ValueError(
            f"Unknown outcome {outcome!r}; expected one of {sorted(OUTCOME_SPECS)}"
        ) from exc


def sample_mask(frame: pd.DataFrame, sample: str = SAMPLE_SINGLE_SELLER) -> pd.Series:
    """Return the frozen primary or deterministic-main-seller sample mask."""

    if sample == SAMPLE_SINGLE_SELLER:
        return frame["is_single_seller"].fillna(False).astype(bool)
    if sample == SAMPLE_DETERMINISTIC_MAIN_SELLER:
        return frame["has_deterministic_main_seller"].fillna(False).astype(bool)
    raise ValueError(
        f"Unknown sample {sample!r}; expected {SAMPLE_SINGLE_SELLER!r} or "
        f"{SAMPLE_DETERMINISTIC_MAIN_SELLER!r}"
    )


def normalise_origin(origin: str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-naive first-of-month origin and reject mid-month use."""

    value = pd.Timestamp(origin)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    if value != value.to_period("M").start_time:
        raise ValueError("Profile origins must be the first instant of a calendar month")
    return value


def normalise_window(window: str) -> str:
    aliases = {
        "H0": WINDOW_H0,
        "expanding": WINDOW_H0,
        WINDOW_H0: WINDOW_H0,
        "H1": WINDOW_H1,
        "180d": WINDOW_H1,
        WINDOW_H1: WINDOW_H1,
    }
    try:
        return aliases[window]
    except KeyError as exc:
        raise ValueError(f"Unknown history window {window!r}") from exc


def validate_reporting_lag(reporting_lag_days: int) -> int:
    """Restrict information-time lag to the frozen primary/sensitivity rules."""

    if reporting_lag_days not in ALLOWED_REPORTING_LAGS:
        raise ValueError(
            f"reporting_lag_days must be one of {ALLOWED_REPORTING_LAGS}, "
            f"got {reporting_lag_days!r}"
        )
    return int(reporting_lag_days)


def asof_eligibility_mask(
    frame: pd.DataFrame,
    origin: str | pd.Timestamp,
    outcome: str,
    *,
    window: str = WINDOW_H0,
    sample: str = SAMPLE_SINGLE_SELLER,
    reporting_lag_days: int = 0,
) -> pd.Series:
    """Return a strict target-specific historical eligibility mask.

    Eligibility always requires purchase strictly before ``origin``, a valid
    outcome, and target availability strictly before ``origin - lag``.  H1
    additionally requires purchase on or after ``origin - 180 days``.  The
    lower H1 boundary is inclusive exactly as frozen.
    """

    reporting_lag_days = validate_reporting_lag(reporting_lag_days)
    t = normalise_origin(origin)
    history_window = normalise_window(window)
    spec = get_outcome_spec(outcome)
    cutoff = t - pd.Timedelta(days=reporting_lag_days)
    mask = (
        sample_mask(frame, sample)
        & frame["order_purchase_timestamp"].lt(t)
        & frame[spec.valid_column].fillna(False).astype(bool)
        & frame[spec.availability_column].lt(cutoff)
    )
    if history_window == WINDOW_H1:
        mask &= frame["order_purchase_timestamp"].ge(t - pd.Timedelta(days=180))
    return mask.fillna(False)


def eligible_history(
    frame: pd.DataFrame,
    origin: str | pd.Timestamp,
    outcome: str,
    **kwargs,
) -> pd.DataFrame:
    """Return a copy of rows passing :func:`asof_eligibility_mask`."""

    return frame.loc[asof_eligibility_mask(frame, origin, outcome, **kwargs)].copy()


def asof_eligibility_counts(
    frame: pd.DataFrame,
    origin: str | pd.Timestamp,
    outcome: str,
    *,
    window: str = WINDOW_H0,
    sample: str = SAMPLE_SINGLE_SELLER,
    reporting_lag_days: int = 0,
) -> dict[str, int | str]:
    """Return auditable counts for one origin without exposing any test result."""

    t = normalise_origin(origin)
    spec = get_outcome_spec(outcome)
    history_window = normalise_window(window)
    reporting_lag_days = validate_reporting_lag(reporting_lag_days)
    in_sample = sample_mask(frame, sample)
    prior_purchase = in_sample & frame["order_purchase_timestamp"].lt(t)
    in_window = prior_purchase.copy()
    if history_window == WINDOW_H1:
        in_window &= frame["order_purchase_timestamp"].ge(t - pd.Timedelta(days=180))
    valid = in_window & frame[spec.valid_column].fillna(False).astype(bool)
    cutoff = t - pd.Timedelta(days=reporting_lag_days)
    mature = valid & frame[spec.availability_column].lt(cutoff)
    return {
        "origin": t.isoformat(),
        "outcome": outcome,
        "history_window": history_window,
        "sample": sample,
        "reporting_lag_days": int(reporting_lag_days),
        "sample_orders": int(in_sample.sum()),
        "purchased_before_origin": int(prior_purchase.sum()),
        "within_purchase_window": int(in_window.sum()),
        "valid_target_rows": int(valid.sum()),
        "invalid_or_missing_target_rows": int((in_window & ~valid).sum()),
        "target_mature_rows": int(mature.sum()),
        "target_immature_rows": int((valid & ~mature).sum()),
    }


def outer_test_mask(
    frame: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    sample: str = SAMPLE_SINGLE_SELLER,
    outcome: str | None = None,
    require_valid_outcome: bool = False,
) -> pd.Series:
    """Return a calendar purchase-month test mask without an as-of maturity gate.

    Test outcomes are scored retrospectively after they mature.  Passing
    ``require_valid_outcome=True`` excludes impossible/missing process outcomes
    but never requires them to have been available at the prediction origin.
    """

    t = normalise_origin(origin)
    end = t + pd.offsets.MonthBegin(1)
    mask = (
        sample_mask(frame, sample)
        & frame["order_purchase_timestamp"].ge(t)
        & frame["order_purchase_timestamp"].lt(end)
    )
    if require_valid_outcome:
        if outcome is None:
            raise ValueError("outcome is required when require_valid_outcome=True")
        spec = get_outcome_spec(outcome)
        mask &= frame[spec.valid_column].fillna(False).astype(bool)
    return mask.fillna(False)


def process_q90(
    frame: pd.DataFrame,
    origin: str | pd.Timestamp,
    outcome: str,
    *,
    sample: str = SAMPLE_SINGLE_SELLER,
    reporting_lag_days: int = 0,
) -> tuple[float, int]:
    """Calculate the frozen linear q90 from expanding eligible history.

    The process-event definition is common to H0 and H1 at a given
    target/sample/lag/origin.  H1 changes the entity profile source window, not
    the q90 target definition, so this helper intentionally has no window
    argument and always uses expanding target-mature history.
    """

    spec = get_outcome_spec(outcome)
    if outcome not in {"post_approval_handling", "purchase_to_carrier", "transit_time"}:
        raise ValueError("process_q90 is only defined for process-duration outcomes")
    history = eligible_history(
        frame,
        origin,
        outcome,
        window=WINDOW_H0,
        sample=sample,
        reporting_lag_days=reporting_lag_days,
    )
    values = pd.to_numeric(history[spec.value_column], errors="coerce").dropna()
    if values.empty:
        raise ValueError("No eligible process durations for q90")
    return float(np.quantile(values.to_numpy(), 0.90, method="linear")), int(len(values))


def build_entity_snapshot(
    frame: pd.DataFrame,
    entity_column: str,
    origin: str | pd.Timestamp,
    outcome: str,
    *,
    window: str = WINDOW_H0,
    sample: str = SAMPLE_SINGLE_SELLER,
    reporting_lag_days: int = 0,
    event_threshold: float | None = None,
) -> pd.DataFrame:
    """Aggregate a leakage-safe raw entity snapshot at one origin.

    Binary outcomes are aggregated directly.  A continuous process outcome is
    aggregated as a binary ``value > event_threshold`` event when a threshold
    is supplied; without a threshold it returns transparent continuous P0
    summaries.  This helper does not estimate P1/P2 or inspect test outcomes.
    """

    if entity_column not in frame.columns:
        raise KeyError(f"Missing entity column {entity_column!r}")
    t = normalise_origin(origin)
    history_window = normalise_window(window)
    spec = get_outcome_spec(outcome)
    history = eligible_history(
        frame,
        t,
        outcome,
        window=history_window,
        sample=sample,
        reporting_lag_days=reporting_lag_days,
    )
    history = history.loc[history[entity_column].notna()].copy()
    common = {
        "origin": t,
        "outcome": outcome,
        "history_window": history_window,
        "sample": sample,
        "reporting_lag_days": int(reporting_lag_days),
    }
    if history.empty:
        base_columns = [
            entity_column, "history_support", "active_months", "first_purchase",
            "last_purchase", "last_availability",
        ]
        if spec.kind == "binary" or event_threshold is not None:
            base_columns += ["event_count", "raw_rate", "event_threshold"]
        else:
            base_columns += ["raw_mean", "raw_sd"]
        return pd.DataFrame(columns=base_columns + list(common))

    grouped = history.groupby(entity_column, dropna=False)
    snapshot = grouped.agg(
        history_support=("order_id", "size"),
        active_months=("purchase_month_start", "nunique"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        last_availability=(spec.availability_column, "max"),
    ).reset_index()

    if spec.kind == "binary" or event_threshold is not None:
        if spec.kind == "binary":
            event = pd.to_numeric(history[spec.value_column], errors="coerce").astype("int8")
            threshold_value = np.nan
        else:
            if not np.isfinite(event_threshold):
                raise ValueError("event_threshold must be finite")
            event = pd.to_numeric(history[spec.value_column], errors="coerce").gt(
                float(event_threshold)
            ).astype("int8")
            threshold_value = float(event_threshold)
        event_count = event.groupby(history[entity_column], dropna=False).sum().rename(
            "event_count"
        )
        snapshot = snapshot.merge(event_count, on=entity_column, how="left", validate="1:1")
        snapshot["raw_rate"] = snapshot["event_count"] / snapshot["history_support"]
        snapshot["event_threshold"] = threshold_value
    else:
        values = pd.to_numeric(history[spec.value_column], errors="coerce")
        continuous = pd.DataFrame(
            {entity_column: history[entity_column], "_value": values}
        ).groupby(entity_column, dropna=False)["_value"].agg(
            raw_mean="mean", raw_sd="std"
        ).reset_index()
        snapshot = snapshot.merge(continuous, on=entity_column, how="left", validate="1:1")

    for column, value in common.items():
        snapshot[column] = value
    return snapshot.sort_values(entity_column, kind="mergesort").reset_index(drop=True)


def build_support_snapshot(
    frame: pd.DataFrame,
    entity_column: str,
    origin: str | pd.Timestamp,
    *,
    outcome: str = "late_delivery",
    window: str = WINDOW_H0,
    sample: str = SAMPLE_DETERMINISTIC_MAIN_SELLER,
    reporting_lag_days: int = 0,
) -> pd.DataFrame:
    """Build target-mature seen/support counts for a support-only identity."""

    snapshot = build_entity_snapshot(
        frame,
        entity_column,
        origin,
        outcome,
        window=window,
        sample=sample,
        reporting_lag_days=reporting_lag_days,
    )
    keep = [
        entity_column,
        "history_support",
        "active_months",
        "first_purchase",
        "last_purchase",
        "last_availability",
        "origin",
        "outcome",
        "history_window",
        "sample",
        "reporting_lag_days",
    ]
    return snapshot[[column for column in keep if column in snapshot.columns]].copy()


def audit_order_base(
    frame: pd.DataFrame,
    *,
    enforce_reference_counts: bool = True,
) -> dict[str, int | float | str | list[str]]:
    """Validate schema/invariants and return non-monthly deterministic counts."""

    missing_b0 = set(B0_COLUMNS) - set(frame.columns)
    missing_support = set(SUPPORT_ONLY_IDS) - set(frame.columns)
    required_outcome_columns = {
        field
        for spec in OUTCOME_SPECS.values()
        for field in (spec.value_column, spec.availability_column, spec.valid_column)
    }
    missing_outcomes = required_outcome_columns - set(frame.columns)
    if missing_b0 or missing_support or missing_outcomes:
        raise AssertionError(
            "Missing assembled fields: "
            f"B0={sorted(missing_b0)}, support={sorted(missing_support)}, "
            f"outcomes={sorted(missing_outcomes)}"
        )
    if frame["order_id"].isna().any() or frame["order_id"].duplicated().any():
        raise AssertionError("Canonical order_id must be non-null and unique")
    if not frame["order_status"].eq("delivered").all():
        raise AssertionError("Canonical frame contains non-delivered status")
    if frame[["order_delivered_customer_date", "order_estimated_delivery_date"]].isna().any().any():
        raise AssertionError("Canonical frame contains a missing actual or estimated delivery date")
    for name, spec in OUTCOME_SPECS.items():
        invalid_available = frame[spec.valid_column] & frame[spec.availability_column].isna()
        if invalid_available.any():
            raise AssertionError(f"Valid {name} rows contain missing availability timestamps")

    single = sample_mask(frame, SAMPLE_SINGLE_SELLER)
    deterministic = sample_mask(frame, SAMPLE_DETERMINISTIC_MAIN_SELLER)
    summary: dict[str, int | float | str | list[str]] = {
        "canonical_orders": int(len(frame)),
        "unique_order_ids": int(frame["order_id"].nunique()),
        "late_events": int(frame["late_delivery"].sum()),
        "severe_events": int(frame["severe_late_2d"].sum()),
        "single_seller_orders": int(single.sum()),
        "deterministic_main_seller_orders": int(deterministic.sum()),
        "single_seller_levels": int(frame.loc[single, "main_seller_id"].nunique()),
        "single_seller_route_levels": int(frame.loc[single, "route_state"].nunique()),
        "negative_post_approval_handling": int(frame["post_approval_handling"].lt(0).sum()),
        "negative_purchase_to_carrier": int(frame["purchase_to_carrier"].lt(0).sum()),
        "negative_transit_time": int(frame["transit_time"].lt(0).sum()),
        "valid_post_approval_handling": int(frame["post_approval_handling_valid"].sum()),
        "valid_transit_time": int(frame["transit_time_valid"].sum()),
        "missing_distance_km": int(frame["distance_km"].isna().sum()),
        "missing_main_product_category": int(frame["main_product_category"].isna().sum()),
        "missing_product_physics": int(frame["avg_product_weight_g"].isna().sum()),
        "purchase_min": frame["order_purchase_timestamp"].min().isoformat(),
        "purchase_max": frame["order_purchase_timestamp"].max().isoformat(),
        "b0_numeric": list(B0_NUMERIC),
        "b0_categorical": list(B0_CATEGORICAL),
        "support_only_ids": list(SUPPORT_ONLY_IDS),
    }
    if enforce_reference_counts:
        mismatches = {
            key: {"expected": expected, "actual": summary.get(key)}
            for key, expected in REFERENCE_COUNTS.items()
            if summary.get(key) != expected
        }
        if mismatches:
            raise AssertionError(f"Frozen Olist reference-count mismatch: {mismatches}")
    return summary


def audit_data_dir(data_dir: str | Path | None = None) -> dict:
    """Run the read-only full input/base audit and return JSON-ready metadata."""

    root = resolve_data_dir(data_dir)
    raw = read_raw_tables(root)
    frame = assemble_order_base(raw)
    summary = audit_order_base(frame, enforce_reference_counts=True)
    return {
        "status": "audit_passed_no_artifacts_written",
        "source_data_directory": str(root),
        "raw_rows": {key: int(len(table)) for key, table in raw.items()},
        "raw_sha256": raw_file_sha256s(root),
        "order_base": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Assemble and verify in memory; print JSON and write no artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audit_only:
        raise SystemExit("This data-layer CLI only supports --audit-only; import its API for execution.")
    print(json.dumps(audit_data_dir(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
