import numpy as np
import pandas as pd

from src.data.reviews import prepare_review_records, select_latest_usable_review
from src.experiments.rq1_customer_relevance import (
    ERROR_GROUP_LABELS,
    promise_error_distribution,
    promise_error_groups,
    wilson_interval,
)


def test_promise_error_groups_match_pre_specified_boundaries():
    values = pd.Series([-100, -14, -13, -7, -6, -1, 0, 1, 2, 3, 4, 7, 8, 100])
    observed = promise_error_groups(values).astype(str).tolist()
    expected = [
        ERROR_GROUP_LABELS[0], ERROR_GROUP_LABELS[0],
        ERROR_GROUP_LABELS[1], ERROR_GROUP_LABELS[1],
        ERROR_GROUP_LABELS[2], ERROR_GROUP_LABELS[2],
        ERROR_GROUP_LABELS[3], ERROR_GROUP_LABELS[4],
        ERROR_GROUP_LABELS[5], ERROR_GROUP_LABELS[5],
        ERROR_GROUP_LABELS[6], ERROR_GROUP_LABELS[6],
        ERROR_GROUP_LABELS[7], ERROR_GROUP_LABELS[7],
    ]
    assert observed == expected


def test_latest_review_selection_is_deterministic_and_keeps_orders_separate():
    records = pd.DataFrame({
        "review_id": ["z", "a", "same", "same", "bad"],
        "order_id": ["o1", "o1", "o2", "o3", "o4"],
        "review_score": [1, 5, 4, 4, 9],
        "review_creation_date": [
            "2018-01-01", "2018-01-02", "2018-01-01", "2018-01-01", "2018-01-01"
        ],
        "review_answer_timestamp": [
            "2018-01-03", "2018-01-03", "2018-01-02", "2018-01-02", "2018-01-02"
        ],
    })
    prepared = prepare_review_records(records)
    selected, audit = select_latest_usable_review(prepared)
    selected = selected.set_index("order_id")
    assert selected.loc["o1", "selected_review_id"] == "a"
    assert selected.loc["o1", "selected_review_score"] == 5
    assert set(selected.index) == {"o1", "o2", "o3"}
    assert audit.set_index("order_id").loc["o1", "conflicting_scores"]


def test_latest_answer_precedes_creation_and_id_tie_breaks():
    records = pd.DataFrame({
        "review_id": ["b", "a", "c"],
        "order_id": ["o1", "o1", "o1"],
        "review_score": [2, 3, 5],
        "review_creation_date": ["2018-01-02", "2018-01-02", "2018-01-01"],
        "review_answer_timestamp": ["2018-01-03", "2018-01-03", "2018-01-04"],
    })
    selected, _ = select_latest_usable_review(records)
    assert selected.loc[0, "selected_review_id"] == "c"
    records.loc[2, "review_answer_timestamp"] = "2018-01-03"
    selected, _ = select_latest_usable_review(records)
    assert selected.loc[0, "selected_review_id"] == "a"


def test_distribution_preserves_calendar_day_states():
    frame = pd.DataFrame({"promise_error_days": [-2, -1, 0, 1, 2]})
    summary = promise_error_distribution(frame).iloc[0]
    assert summary.sample_size == 5
    assert summary.proportion_early == 0.4
    assert summary.proportion_on_date == 0.2
    assert summary.proportion_late == 0.4
    assert summary.proportion_severe_late_2d == 0.2
    assert np.isclose(summary.standard_deviation_sample, np.std([-2, -1, 0, 1, 2], ddof=1))


def test_wilson_interval_is_bounded_and_contains_observed_proportion():
    lower, upper = wilson_interval(2, 10)
    assert 0 <= lower <= 0.2 <= upper <= 1
    assert all(np.isfinite(wilson_interval(0, 10)))
    assert all(np.isfinite(wilson_interval(10, 10)))
