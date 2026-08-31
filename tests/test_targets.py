import pandas as pd

from src.features.targets import build_targets


def test_calendar_date_sign_rounding_and_missingness():
    frame = pd.DataFrame({
        "order_delivered_customer_date": ["2018-01-03 01:00", "2018-01-02 23:59", None],
        "order_estimated_delivery_date": ["2018-01-02 23:00", "2018-01-03 00:01", "2018-01-03"],
    })
    result = build_targets(frame)
    assert result.promise_error_days.tolist()[:2] == [1, -1]
    assert result.late_delivery.tolist()[:2] == [1, 0]
    assert result.severe_late_2d.tolist()[:2] == [0, 0]
    assert pd.isna(result.promise_error_days.iloc[2])
    assert pd.isna(result.late_delivery.iloc[2])


def test_severe_threshold_is_two_calendar_days():
    frame = pd.DataFrame({
        "order_delivered_customer_date": ["2018-01-04", "2018-01-03"],
        "order_estimated_delivery_date": ["2018-01-02", "2018-01-02"],
    })
    result = build_targets(frame)
    assert result.severe_late_2d.tolist() == [1, 0]
