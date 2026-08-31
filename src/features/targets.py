"""Single source of truth for delivery-promise outcomes."""
from __future__ import annotations

import pandas as pd

TARGET_COLUMNS = ("promise_error_days", "late_delivery", "severe_late_2d")


def build_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Build calendar-date outcomes; positive error means delivered after promise.

    Missing actual/estimated timestamps remain missing. The severe label means at
    least two full calendar dates after the estimated delivery date.
    """
    required = {"order_delivered_customer_date", "order_estimated_delivery_date"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing target inputs: {sorted(missing)}")
    result = frame.copy()
    actual = pd.to_datetime(result["order_delivered_customer_date"], errors="coerce").dt.normalize()
    promise = pd.to_datetime(result["order_estimated_delivery_date"], errors="coerce").dt.normalize()
    error = (actual - promise).dt.days.astype("Int64")
    result["promise_error_days"] = error
    result["late_delivery"] = (error > 0).astype("Int64").where(error.notna())
    result["severe_late_2d"] = (error >= 2).astype("Int64").where(error.notna())
    return result
