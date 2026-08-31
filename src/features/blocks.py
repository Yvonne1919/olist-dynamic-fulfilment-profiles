"""Audited M0/M1/M2 information sets."""
from __future__ import annotations

import numpy as np
import pandas as pd

M0_NUMERIC = ["promised_delivery_days"]
M1_NUMERIC = [
    "n_items", "n_unique_products", "n_unique_sellers", "total_price",
    "total_freight_value", "freight_to_price_ratio", "avg_product_weight_g",
    "avg_product_volume_cm3", "customer_seller_same_state",
]
M1_CATEGORICAL = ["customer_state", "seller_state", "main_product_category"]
M2_NUMERIC = [
    "purchase_month", "purchase_weekday", "purchase_hour", "is_weekend_purchase",
    "days_to_next_holiday", "days_since_previous_holiday",
    "holiday_within_next_3d", "holiday_within_next_7d",
    "holiday_in_promise_window", "num_holidays_in_promise_window",
    "num_weekend_days_in_promise_window", "black_friday_period",
    "christmas_new_year_period", "carnival_period",
]


def add_structural_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    purchase = pd.to_datetime(result["order_purchase_timestamp"])
    promise = pd.to_datetime(result["order_estimated_delivery_date"])
    result["promised_delivery_days"] = (promise - purchase).dt.total_seconds() / 86400
    result["freight_to_price_ratio"] = result["total_freight_value"] / result["total_price"].replace(0, np.nan)
    result["avg_product_volume_cm3"] = (
        result["avg_product_length_cm"] * result["avg_product_height_cm"]
        * result["avg_product_width_cm"]
    )
    result["customer_seller_same_state"] = (
        result["customer_state"].notna() & result["seller_state"].notna()
        & result["customer_state"].eq(result["seller_state"])
    ).astype(int)
    return result


def feature_sets() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "M0": (M0_NUMERIC, []),
        "M0_M1": (M0_NUMERIC + M1_NUMERIC, M1_CATEGORICAL),
        "M0_M1_M2": (M0_NUMERIC + M1_NUMERIC + M2_NUMERIC, M1_CATEGORICAL),
    }
