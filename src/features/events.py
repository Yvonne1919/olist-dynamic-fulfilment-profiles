"""Reproducible Brazil calendar and explicitly defined retail-event features."""
from __future__ import annotations

from importlib.metadata import version

import holidays
import numpy as np
import pandas as pd


def brazil_event_calendar(years: list[int]) -> pd.DataFrame:
    calendar = holidays.Brazil(years=years, language="en_US")
    rows = [{"date": pd.Timestamp(day), "event": name, "source": "holidays.Brazil",
             "source_version": version("holidays")} for day, name in sorted(calendar.items())]
    return pd.DataFrame(rows)


def _count_weekend_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    return int(sum(day.weekday() >= 5 for day in pd.date_range(start.normalize(), end.normalize())))


def add_event_features(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    purchase = pd.to_datetime(result["order_purchase_timestamp"]).dt.normalize()
    promise = pd.to_datetime(result["order_estimated_delivery_date"]).dt.normalize()
    holiday_dates = pd.DatetimeIndex(calendar["date"].drop_duplicates().sort_values())
    holiday_set = set(holiday_dates)

    def previous_gap(day):
        eligible = holiday_dates[holiday_dates <= day]
        return (day - eligible.max()).days if len(eligible) else np.nan

    def next_gap(day):
        eligible = holiday_dates[holiday_dates >= day]
        return (eligible.min() - day).days if len(eligible) else np.nan

    result["purchase_weekday"] = purchase.dt.dayofweek
    result["purchase_month"] = purchase.dt.month
    result["purchase_hour"] = pd.to_datetime(result["order_purchase_timestamp"]).dt.hour
    result["is_weekend_purchase"] = result["purchase_weekday"].isin([5, 6]).astype(int)
    result["days_to_next_holiday"] = purchase.map(next_gap)
    result["days_since_previous_holiday"] = purchase.map(previous_gap)
    result["holiday_within_next_3d"] = result["days_to_next_holiday"].between(0, 3).astype(int)
    result["holiday_within_next_7d"] = result["days_to_next_holiday"].between(0, 7).astype(int)
    result["num_holidays_in_promise_window"] = [
        sum(start <= day <= end for day in holiday_set) if end >= start else 0
        for start, end in zip(purchase, promise)
    ]
    result["holiday_in_promise_window"] = (result["num_holidays_in_promise_window"] > 0).astype(int)
    result["num_weekend_days_in_promise_window"] = [
        _count_weekend_days(start, end) for start, end in zip(purchase, promise)
    ]
    # Project definitions: surrounding periods are fixed before model evaluation.
    result["black_friday_period"] = ((purchase.dt.month == 11) & (purchase.dt.day >= 20)).astype(int)
    result["christmas_new_year_period"] = (
        ((purchase.dt.month == 12) & (purchase.dt.day >= 15))
        | ((purchase.dt.month == 1) & (purchase.dt.day <= 7))
    ).astype(int)
    carnival_names = calendar.loc[calendar["event"].str.contains("Carnival", case=False), "date"]
    result["carnival_period"] = purchase.map(
        lambda day: int(any(abs((day - event).days) <= 3 for event in carnival_names))
    )
    return result
