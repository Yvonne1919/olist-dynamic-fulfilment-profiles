import pandas as pd

from src.evaluation.splits import development_final_holdout, expanding_month_splits


def test_final_holdout_has_no_overlap_and_is_future():
    frame = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(10)],
        "order_purchase_timestamp": pd.date_range("2018-01-01", periods=10),
    })
    development, final = development_final_holdout(frame, 0.2)
    assert len(development) == 8 and len(final) == 2
    assert set(development.order_id).isdisjoint(final.order_id)
    assert development.order_purchase_timestamp.max() < final.order_purchase_timestamp.min()


def test_expanding_splits_exclude_boundary_months():
    rows = []
    for month in pd.period_range("2017-01", "2017-09", freq="M"):
        for i in range(500):
            rows.append({"order_id": f"{month}-{i}", "order_purchase_timestamp": month.start_time + pd.Timedelta(minutes=i), "late_delivery": int(i < 25)})
    frame = pd.DataFrame(rows)
    folds = expanding_month_splits(frame, min_train_months=6)
    assert folds[0][1] == "2017-08"
    assert all(period != "2017-09" for _, period, _, _ in folds)
