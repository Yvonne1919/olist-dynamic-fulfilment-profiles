from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.all_mature_history_sensitivity_v1.scripts.sensitivity_core import (
    aggregate_anchor_outputs,
    all_mature_history_slice,
    favourable_direction,
    paired_record,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["ancient", "recent", "boundary", "unmatured"],
        "order_purchase_timestamp": pd.to_datetime([
            "2017-01-01", "2018-05-01", "2018-05-18", "2017-01-01",
        ]),
        "in_canonical": [True, True, True, True],
        "handling_available_at": pd.to_datetime([
            "2017-01-03", "2018-05-02", "2018-05-19", "2018-05-20",
        ]),
        "handling_level_value": [1.0, 2.0, 3.0, 4.0],
        "handling_duration": [1.0, 2.0, 3.0, 4.0],
        "seller_id": ["s1", "s1", "s2", "s3"],
        "main_seller_state": ["SP", "SP", "RJ", "MG"],
        "region_od": ["a", "a", "b", "c"],
    })


def test_scheme_a_all_mature_has_no_lower_bound_and_strict_snapshot() -> None:
    source = {
        "target": "handling_level", "granularity": "seller_id",
        "scheme": "A", "window_days": 90, "lag_days": 0,
    }
    history = all_mature_history_slice(_frame(), source, pd.Timestamp("2018-05-20"))
    assert set(history["order_id"]) == {"ancient", "recent", "boundary"}
    assert history["handling_available_at"].lt(pd.Timestamp("2018-05-20")).all()


def test_scheme_c_preserves_lag_and_removes_only_lower_bound() -> None:
    source = {
        "target": "handling_level", "granularity": "seller_id",
        "scheme": "C", "window_days": 90, "lag_days": 14,
    }
    history = all_mature_history_slice(_frame(), source, pd.Timestamp("2018-05-20"))
    assert set(history["order_id"]) == {"ancient", "recent"}
    assert history["order_purchase_timestamp"].lt(pd.Timestamp("2018-05-06")).all()


def test_literal_delta_and_direction() -> None:
    row = paired_record({}, "log_loss", 0.2, 0.18)
    assert np.isclose(row["all_mature_minus_90d"], -0.02)
    assert row["all_mature_favourable"]
    assert favourable_direction("weighted_spearman") == "higher_is_favourable"


def test_equivalence_uses_separate_period_medians_not_median_paired_delta() -> None:
    rows = []
    values = {
        "log_loss": ([0.10, 0.20, 0.21], [0.1001, 0.101, 0.30]),
        "brier": ([0.05, 0.06, 0.07], [0.0501, 0.0601, 0.0701]),
    }
    for metric, (left, right) in values.items():
        for index, (selected, all_mature) in enumerate(zip(left, right)):
            rows.append({
                "profile_code": "S2",
                "candidate_id": "candidate",
                "base_candidate_id": "base",
                "target": "handling_tail",
                "target_kind": "binary",
                "period": "development",
                "anchor_date": pd.Timestamp("2017-04-01") + pd.Timedelta(days=7 * index),
                "calendar_month": "2017-04",
                "horizon_days": 7,
                "metric": metric,
                "favourable_direction": "lower_is_favourable",
                "selected_90d_value": selected,
                "all_mature_value": all_mature,
                "all_mature_minus_90d": all_mature - selected,
                "all_mature_favourable": all_mature < selected,
            })
    cfg = {
        "practical_equivalence": {
            "binary_absolute_log_loss": 0.001,
            "binary_absolute_brier": 0.0005,
            "continuous_absolute_weighted_spearman": 0.02,
            "continuous_relative_log_mae": 0.01,
        }
    }
    _, summary = aggregate_anchor_outputs(pd.DataFrame(rows), cfg)
    log_row = summary.loc[summary["metric"].eq("log_loss")].iloc[0]
    assert np.isclose(log_row["all_mature_minus_90d"], 0.0001)
    assert np.isclose(log_row["aggregate_median_difference"], -0.099)
    assert log_row["practical_equivalence_assessment"] == "outside_frozen_tolerances_numeric_only"
    assert log_row["all_mature_minus_90d_aggregation"] == "median_of_paired_anchor_differences"
