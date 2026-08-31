"""Focused unit tests for the Phase 2A data layer; no real outer results."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.profile_pivot_phase2a.scripts.data_pipeline import (
    SAMPLE_DETERMINISTIC_MAIN_SELLER,
    WINDOW_H0,
    WINDOW_H1,
    assemble_order_base,
    asof_eligibility_mask,
    build_entity_snapshot,
    outer_test_mask,
    process_q90,
    sample_mask,
)


def _synthetic_raw() -> dict[str, pd.DataFrame]:
    return {
        "orders": pd.DataFrame(
            {
                "order_id": ["o1", "o2", "excluded"],
                "customer_id": ["c1", "c2", "c1"],
                "order_status": ["delivered", "delivered", "canceled"],
                "order_purchase_timestamp": [
                    "2018-01-01 10:00:00",
                    "2018-02-01 10:00:00",
                    "2018-02-02 10:00:00",
                ],
                "order_approved_at": [
                    "2018-01-01 11:00:00",
                    "2018-02-01 12:00:00",
                    None,
                ],
                "order_delivered_carrier_date": [
                    "2018-01-02 10:00:00",
                    "2018-02-03 10:00:00",
                    None,
                ],
                "order_delivered_customer_date": [
                    "2018-01-05 10:00:00",
                    "2018-02-10 10:00:00",
                    None,
                ],
                "order_estimated_delivery_date": [
                    "2018-01-06 00:00:00",
                    "2018-02-08 00:00:00",
                    "2018-02-20 00:00:00",
                ],
            }
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["c1", "c2"],
                "customer_unique_id": ["u1", "u2"],
                "customer_zip_code_prefix": [1000, 2000],
                "customer_city": ["one", "two"],
                "customer_state": ["RJ", "MG"],
            }
        ),
        "items": pd.DataFrame(
            {
                "order_id": ["o1", "o2", "o2", "excluded"],
                "order_item_id": [1, 1, 2, 1],
                "product_id": ["p1", "p1", "p2", "p1"],
                "seller_id": ["s2", "z_seller", "a_seller", "s2"],
                "shipping_limit_date": ["2018-01-03"] * 4,
                "price": [10.0, 10.0, 20.0, 5.0],
                "freight_value": [1.0, 1.0, 2.0, 1.0],
            }
        ),
        "products": pd.DataFrame(
            {
                "product_id": ["p1", "p2"],
                "product_category_name": ["cat1", "cat2"],
                "product_name_lenght": [1, 1],
                "product_description_lenght": [1, 1],
                "product_photos_qty": [1, 1],
                "product_weight_g": [100.0, 200.0],
                "product_length_cm": [2.0, 10.0],
                "product_height_cm": [3.0, 1.0],
                "product_width_cm": [4.0, 1.0],
            }
        ),
        "sellers": pd.DataFrame(
            {
                "seller_id": ["s2", "z_seller", "a_seller"],
                "seller_zip_code_prefix": [3000, 4000, 5000],
                "seller_city": ["s2", "z", "a"],
                "seller_state": ["RJ", "SP", "RS"],
            }
        ),
        "categories": pd.DataFrame(
            {
                "product_category_name": ["cat1", "cat2"],
                "product_category_name_english": ["category one", "category two"],
            }
        ),
        "geolocation": pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [1000, 2000, 3000, 4000, 5000],
                "geolocation_lat": [-22.9, -19.9, -22.8, -23.5, -30.0],
                "geolocation_lng": [-43.2, -43.9, -43.1, -46.6, -51.2],
                "geolocation_city": ["a", "b", "c", "d", "e"],
                "geolocation_state": ["RJ", "MG", "RJ", "SP", "RS"],
            }
        ),
    }


def test_order_assembly_uses_exact_attribution_and_item_physics() -> None:
    frame = assemble_order_base(_synthetic_raw()).set_index("order_id")

    assert list(frame.index) == ["o1", "o2"]
    assert bool(frame.loc["o1", "is_single_seller"])
    assert not bool(frame.loc["o2", "is_single_seller"])
    assert frame.loc["o2", "main_seller_id"] == "a_seller"
    assert frame.loc["o2", "main_seller_state"] == "RS"
    assert frame.loc["o2", "route_state"] == "RS -> MG"
    assert frame.loc["o2", "main_product_id"] == "p1"
    assert frame.loc["o2", "avg_product_volume_cm3"] == 17.0
    assert frame.loc["o2", "max_product_dimension_cm"] == 10.0
    assert frame.loc["o2", "category_diversity"] == 2
    assert frame.loc["o2", "promise_error_days"] == 2
    assert frame.loc["o2", "late_delivery"] == 1
    assert frame.loc["o2", "late_available_at"] == frame.loc[
        "o2", "late_delivery_available_at"
    ]
    assert frame.loc["o2", "handling_available_at"] == frame.loc[
        "o2", "post_approval_handling_available_at"
    ]
    assert frame.loc["o2", "transit_available_at"] == frame.loc[
        "o2", "transit_time_available_at"
    ]
    assert frame.loc["o1", "distance_km"] >= 0
    assert sample_mask(frame, SAMPLE_DETERMINISTIC_MAIN_SELLER).sum() == 2


def test_asof_masks_are_strict_target_specific_and_windowed() -> None:
    origin = pd.Timestamp("2018-04-01")
    frame = pd.DataFrame(
        {
            "order_id": ["eligible", "equal_cutoff", "future", "old", "multi"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-02-01", "2018-02-02", "2018-04-01", "2017-01-01", "2018-02-01"]
            ),
            "purchase_month_start": pd.to_datetime(
                ["2018-02-01", "2018-02-01", "2018-04-01", "2017-01-01", "2018-02-01"]
            ),
            "late_delivery": [1, 0, 1, 0, 1],
            "late_delivery_valid": [True] * 5,
            "late_delivery_available_at": pd.to_datetime(
                ["2018-02-20", "2018-04-01", "2018-04-10", "2017-01-20", "2018-02-20"]
            ),
            "is_single_seller": [True, True, True, True, False],
            "has_deterministic_main_seller": [True] * 5,
            "main_seller_id": ["s1", "s1", "s2", "s_old", "s3"],
        }
    )

    h0 = asof_eligibility_mask(frame, origin, "late_delivery", window=WINDOW_H0)
    h1 = asof_eligibility_mask(frame, origin, "late_delivery", window=WINDOW_H1)
    assert frame.loc[h0, "order_id"].tolist() == ["eligible", "old"]
    assert frame.loc[h1, "order_id"].tolist() == ["eligible"]

    lagged = asof_eligibility_mask(
        frame,
        origin,
        "late_delivery",
        window=WINDOW_H0,
        reporting_lag_days=30,
    )
    assert frame.loc[lagged, "order_id"].tolist() == ["eligible", "old"]

    all_sample = asof_eligibility_mask(
        frame,
        origin,
        "late_delivery",
        sample=SAMPLE_DETERMINISTIC_MAIN_SELLER,
    )
    assert frame.loc[all_sample, "order_id"].tolist() == ["eligible", "old", "multi"]

    test = outer_test_mask(frame, origin, sample=SAMPLE_DETERMINISTIC_MAIN_SELLER)
    assert frame.loc[test, "order_id"].tolist() == ["future"]

    with pytest.raises(ValueError, match="first instant"):
        asof_eligibility_mask(frame, "2018-04-15", "late_delivery")
    with pytest.raises(ValueError, match="must be one of"):
        asof_eligibility_mask(
            frame, origin, "late_delivery", reporting_lag_days=15
        )


def test_snapshot_uses_only_eligible_history() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["a", "b", "immature"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03"]
            ),
            "purchase_month_start": pd.to_datetime(
                ["2018-01-01", "2018-01-01", "2018-01-01"]
            ),
            "late_delivery": [1, 0, 1],
            "late_delivery_valid": [True, True, True],
            "late_delivery_available_at": pd.to_datetime(
                ["2018-01-10", "2018-01-11", "2018-03-01"]
            ),
            "is_single_seller": [True, True, True],
            "has_deterministic_main_seller": [True, True, True],
            "main_seller_id": ["s", "s", "s"],
        }
    )
    snapshot = build_entity_snapshot(
        frame, "main_seller_id", "2018-02-01", "late_delivery"
    )
    assert len(snapshot) == 1
    assert snapshot.loc[0, "history_support"] == 2
    assert snapshot.loc[0, "event_count"] == 1
    assert snapshot.loc[0, "raw_rate"] == 0.5


def test_process_q90_uses_common_expanding_target_definition() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["old", "recent"],
            "order_purchase_timestamp": pd.to_datetime(["2017-01-01", "2018-03-01"]),
            "post_approval_handling": [100.0, 1.0],
            "post_approval_handling_valid": [True, True],
            "post_approval_handling_available_at": pd.to_datetime(
                ["2017-01-10", "2018-03-10"]
            ),
            "is_single_seller": [True, True],
            "has_deterministic_main_seller": [True, True],
        }
    )
    threshold, support = process_q90(
        frame, "2018-04-01", "post_approval_handling"
    )
    assert support == 2
    assert threshold == 90.1
