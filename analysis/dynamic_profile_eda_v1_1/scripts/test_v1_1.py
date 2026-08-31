from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/dynamic-profile-v1-1-mpl-tests")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pytest

from analysis.dynamic_profile_eda_v1_1.scripts import core
from analysis.dynamic_profile_eda_v1_1.scripts import fast_engine
from analysis.dynamic_profile_eda_v1_1.scripts import run_eda_v1_1 as run


CONFIG = json.loads((core.OUT / "EDA_V1_1_FROZEN_CONFIG.json").read_text())
DATA_DIR = Path(CONFIG["data_dir"])


def output(name: str) -> Path:
    return core.OUT / name


def tiny_frame() -> pd.DataFrame:
    t = pd.Timestamp
    rows = [
        # Old slow delivery: enters Scheme A near 1 February.
        dict(
            order_id="slow", order_status="delivered",
            order_purchase_timestamp=t("2017-10-01"), order_approved_at=t("2017-10-02"),
            order_delivered_carrier_date=t("2017-10-05"), order_delivered_customer_date=t("2018-01-28"),
            order_estimated_delivery_date=t("2018-01-20"), purchase_date=t("2017-10-01"),
            late_delivery=True, positive_late_days=8.0, handling_duration=3.0, transit_duration=115.0,
            final_breach_available_at=t("2018-01-28"), positive_late_days_available_at=t("2018-01-28"),
            handling_available_at=t("2017-10-05"), transit_available_at=t("2018-01-28"),
            purchase_to_carrier=4.0, promise_error_days=8.0,
            gmv_observed=True, total_price=100.0, total_freight_value=10.0,
            main_seller_id="s1", seller_x_customer_region="s1 -> R",
            seller_x_customer_state="s1 -> ST", seller_x_state_od="s1 -> ST -> ST",
            region_od="R -> R", state_od="ST -> ST", zip2_od="12 -> 54",
            n_unique_sellers=1, is_multi_seller=False,
        ),
        # Recent unresolved order: remains in the purchase-cohort denominator.
        dict(
            order_id="unresolved", order_status="shipped",
            order_purchase_timestamp=t("2018-01-25"), order_approved_at=t("2018-01-26"),
            order_delivered_carrier_date=t("2018-01-27"), order_delivered_customer_date=pd.NaT,
            order_estimated_delivery_date=t("2018-02-10"), purchase_date=t("2018-01-25"),
            late_delivery=np.nan, positive_late_days=np.nan, handling_duration=1.0, transit_duration=np.nan,
            final_breach_available_at=pd.NaT, positive_late_days_available_at=pd.NaT,
            handling_available_at=t("2018-01-27"), transit_available_at=pd.NaT,
            purchase_to_carrier=2.0, promise_error_days=np.nan,
            gmv_observed=True, total_price=80.0, total_freight_value=8.0,
            main_seller_id="s2", seller_x_customer_region="s2 -> R",
            seller_x_customer_state="s2 -> ST", seller_x_state_od="s2 -> ST -> ST",
            region_od="R -> R", state_od="ST -> ST", zip2_od="12 -> 54",
            n_unique_sellers=1, is_multi_seller=False,
        ),
        # Delivered after t; handling is already available.
        dict(
            order_id="future", order_status="delivered",
            order_purchase_timestamp=t("2018-01-01"), order_approved_at=t("2018-01-02"),
            order_delivered_carrier_date=t("2018-01-05"), order_delivered_customer_date=t("2018-02-10"),
            order_estimated_delivery_date=t("2018-02-08"), purchase_date=t("2018-01-01"),
            late_delivery=True, positive_late_days=2.0, handling_duration=3.0, transit_duration=36.0,
            final_breach_available_at=t("2018-02-10"), positive_late_days_available_at=t("2018-02-10"),
            handling_available_at=t("2018-01-05"), transit_available_at=t("2018-02-10"),
            purchase_to_carrier=4.0, promise_error_days=2.0,
            gmv_observed=True, total_price=120.0, total_freight_value=12.0,
            main_seller_id="s1", seller_x_customer_region="s1 -> R",
            seller_x_customer_state="s1 -> ST", seller_x_state_od="s1 -> ST -> ST",
            region_od="R -> R", state_od="ST -> ST", zip2_od="12 -> 54",
            n_unique_sellers=1, is_multi_seller=False,
        ),
        # Negative handling anomaly: timestamp-observed but not descriptively valid.
        dict(
            order_id="negative", order_status="delivered",
            order_purchase_timestamp=t("2018-01-03"), order_approved_at=t("2018-01-06"),
            order_delivered_carrier_date=t("2018-01-05"), order_delivered_customer_date=t("2018-01-20"),
            order_estimated_delivery_date=t("2018-01-22"), purchase_date=t("2018-01-03"),
            late_delivery=False, positive_late_days=0.0, handling_duration=-1.0, transit_duration=15.0,
            final_breach_available_at=t("2018-01-20"), positive_late_days_available_at=t("2018-01-20"),
            handling_available_at=t("2018-01-06"), transit_available_at=t("2018-01-20"),
            purchase_to_carrier=2.0, promise_error_days=-2.0,
            gmv_observed=True, total_price=60.0, total_freight_value=6.0,
            main_seller_id="s3", seller_x_customer_region="s3 -> R",
            seller_x_customer_state="s3 -> ST", seller_x_state_od="s3 -> ST -> ST",
            region_od="R -> R", state_od="ST -> ST", zip2_od="12 -> 54",
            n_unique_sellers=2, is_multi_seller=True,
        ),
    ]
    return pd.DataFrame(rows)


def _minimal_scheme_frame(ids: list[str], purchases: list[pd.Timestamp], available: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ids,
            "order_purchase_timestamp": pd.to_datetime(purchases),
            "final_breach_available_at": pd.to_datetime(available),
        }
    )


def coverage_fixture() -> pd.DataFrame:
    """Four-order snapshot fixture: one history row and three future mappings."""
    history = tiny_frame().iloc[[3]].copy()
    history.loc[:, "order_id"] = "history_seen"
    history.loc[:, "order_purchase_timestamp"] = pd.Timestamp("2018-01-03")
    history.loc[:, "purchase_date"] = pd.Timestamp("2018-01-03")
    history.loc[:, "main_seller_id"] = "s1"
    history.loc[:, "seller_x_customer_region"] = "s1 -> R"
    history.loc[:, "seller_x_customer_state"] = "s1 -> ST"
    history.loc[:, "seller_x_state_od"] = "s1 -> ST -> ST"
    history.loc[:, "handling_duration"] = 1.0

    future_rows = []
    for oid, seller, purchase in [
        ("future_seen", "s1", "2018-02-02"),
        ("future_unseen", "s9", "2018-02-03"),
        ("future_missing", pd.NA, "2018-02-04"),
    ]:
        row = history.iloc[0].to_dict()
        row.update(
            order_id=oid,
            order_status="shipped",
            order_purchase_timestamp=pd.Timestamp(purchase),
            purchase_date=pd.Timestamp(purchase),
            order_approved_at=pd.NaT,
            order_delivered_carrier_date=pd.NaT,
            order_delivered_customer_date=pd.NaT,
            late_delivery=np.nan,
            positive_late_days=np.nan,
            handling_duration=np.nan,
            transit_duration=np.nan,
            purchase_to_carrier=np.nan,
            promise_error_days=np.nan,
            final_breach_available_at=pd.NaT,
            positive_late_days_available_at=pd.NaT,
            handling_available_at=pd.NaT,
            transit_available_at=pd.NaT,
            main_seller_id=seller,
            seller_x_customer_region=(f"{seller} -> R" if pd.notna(seller) else pd.NA),
            seller_x_customer_state=(f"{seller} -> ST" if pd.notna(seller) else pd.NA),
            seller_x_state_od=(f"{seller} -> ST -> ST" if pd.notna(seller) else pd.NA),
            region_od=("R -> R" if pd.notna(seller) else pd.NA),
            state_od=("ST -> ST" if pd.notna(seller) else pd.NA),
            zip2_od=("12 -> 54" if pd.notna(seller) else pd.NA),
            n_unique_sellers=(1 if pd.notna(seller) else np.nan),
            is_multi_seller=False,
        )
        future_rows.append(row)
    return pd.concat([history, pd.DataFrame(future_rows)], ignore_index=True)


def interval_fixture() -> tuple[pd.DataFrame, pd.Timestamp]:
    """A two-candidate-day fixture that makes the exact interval auditable."""
    endpoint = pd.Timestamp("2018-04-02 12:00:00")

    def row(oid: str, purchase: str | pd.Timestamp, available: str | pd.Timestamp) -> dict:
        purchase_ts = pd.Timestamp(purchase)
        available_ts = pd.Timestamp(available)
        return {
            "order_id": oid,
            "order_purchase_timestamp": purchase_ts,
            "order_delivered_customer_date": available_ts,
            "final_breach_available_at": available_ts,
            "positive_late_days_available_at": available_ts,
            "handling_available_at": available_ts,
            "transit_available_at": available_ts,
        }

    rows = [row("raw_start", "2018-01-01", "2018-01-02")]
    rows.extend(row(f"anchor_{i:03d}", "2018-02-15", "2018-04-02 10:00:00") for i in range(100))
    rows.extend(
        [
            row("left_included", "2018-02-01 00:00:00", "2018-04-02 10:00:00"),
            row("left_excluded", "2018-01-31 23:59:59.999999", "2018-04-02 10:00:00"),
            row("right_included_endpoint_excluded", "2018-04-01 23:59:59.999999", "2018-04-02 12:00:00.000001"),
            row("right_excluded", "2018-04-02 00:00:00", "2018-04-02 10:00:00"),
        ]
    )
    return pd.DataFrame(rows), endpoint


def canonical_quality_fixture() -> pd.DataFrame:
    """Canonical-shaped fixture kept distinct from the all-placed audit frame."""
    canonical = tiny_frame().loc[lambda x: x.order_status.eq("delivered")].copy()
    canonical["route_state"] = canonical["state_od"]
    canonical["distance_km"] = [100.0, 200.0, np.nan]
    canonical["post_approval_handling"] = canonical["handling_duration"]
    canonical["transit_time"] = canonical["transit_duration"]
    return canonical


@pytest.fixture(scope="session")
def canonical_order_ids() -> frozenset[str]:
    """Execute the protected canonical assembler once, read-only and without bytecode."""
    from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp

    raw = dp.read_raw_tables(DATA_DIR)
    canonical = dp.assemble_order_base(raw)
    assert len(canonical) == 96_470
    assert canonical.order_id.notna().all() and canonical.order_id.is_unique
    return frozenset(canonical.order_id.astype(str))


@pytest.fixture(scope="session")
def synthetic_snapshot_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot = pd.Timestamp("2018-02-01")
    run.snapshot_worker_init(coverage_fixture(), pd.Timestamp("2018-12-01"), {snapshot})
    return tuple(pd.DataFrame(rows) for rows in run.process_snapshot(str(snapshot)))


def test_01_canonical_assembler_hash() -> None:
    assert core.sha256_file(core.ASSEMBLER) == core.ASSEMBLER_SHA


def test_02_reconciliation_is_one_row_per_raw_order() -> None:
    reconciliation = pd.read_csv(output("CANONICAL_SAMPLE_RECONCILIATION.csv"), dtype={"order_id": "string"})
    assert len(reconciliation) == 99_441
    assert reconciliation.order_id.notna().all() and reconciliation.order_id.is_unique


def test_03_canonical_96470_unique_members_match_real_assembler(canonical_order_ids: frozenset[str]) -> None:
    reconciliation = pd.read_csv(output("CANONICAL_SAMPLE_RECONCILIATION.csv"), dtype={"order_id": "string"})
    recorded = reconciliation.loc[reconciliation.in_canonical, "order_id"]
    assert len(recorded) == 96_470 and recorded.is_unique
    assert frozenset(recorded.astype(str)) == canonical_order_ids


def test_04_strict_equality_is_excluded_by_real_scheme_function() -> None:
    snapshot = pd.Timestamp("2018-02-01")
    frame = _minimal_scheme_frame(
        ["before", "equal"],
        [snapshot - pd.Timedelta(days=2), snapshot - pd.Timedelta(days=2)],
        [snapshot - pd.Timedelta(microseconds=1), snapshot],
    )
    completion, _, _ = core.scheme_cohort(frame, "final_breach", snapshot, 30, "A", 0, snapshot + pd.Timedelta(days=30))
    full, mature, eventual = core.scheme_cohort(frame, "final_breach", snapshot, 30, "B", 0, snapshot + pd.Timedelta(days=30))
    assert set(completion.order_id) == {"before"}
    assert set(full.order_id) == {"before", "equal"}
    assert set(mature.order_id) == {"before"}
    assert set(eventual.order_id) == {"before", "equal"}


def test_05_future_delivery_not_profile_eligible() -> None:
    frame = tiny_frame()
    _, mature, _ = core.scheme_cohort(frame, "final_breach", pd.Timestamp("2018-02-01"), 30, "B", 0, pd.Timestamp("2018-12-01"))
    assert "future" not in set(mature.order_id)


def test_06_slow_old_purchase_enters_scheme_a() -> None:
    completion, _, _ = core.scheme_cohort(tiny_frame(), "final_breach", pd.Timestamp("2018-02-01"), 30, "A", 0, pd.Timestamp("2018-12-01"))
    assert "slow" in set(completion.order_id)


def test_07_unresolved_stays_in_scheme_b_full_denominator() -> None:
    full, mature, eventual = core.scheme_cohort(tiny_frame(), "final_breach", pd.Timestamp("2018-02-01"), 30, "B", 0, pd.Timestamp("2018-12-01"))
    assert set(full.order_id) == {"unresolved", "negative"}
    assert set(mature.order_id) == {"negative"}
    assert set(eventual.order_id) == {"negative"}


def test_08_handling_available_before_customer_delivery() -> None:
    frame = tiny_frame().set_index("order_id")
    assert frame.loc["future", "handling_available_at"] < pd.Timestamp("2018-02-01")
    assert frame.loc["future", "final_breach_available_at"] > pd.Timestamp("2018-02-01")


@pytest.mark.parametrize("window_days", core.WINDOWS)
def test_09_scheme_b_exact_30_60_90_boundaries(window_days: int) -> None:
    snapshot = pd.Timestamp("2018-04-01")
    epsilon = pd.Timedelta(microseconds=1)
    left = snapshot - pd.Timedelta(days=window_days)
    frame = _minimal_scheme_frame(
        ["left_in", "left_out", "right_in", "right_out"],
        [left, left - epsilon, snapshot - epsilon, snapshot],
        [snapshot - pd.Timedelta(days=1)] * 4,
    )
    full, mature, eventual = core.scheme_cohort(frame, "final_breach", snapshot, window_days, "B", 0, snapshot)
    expected = {"left_in", "right_in"}
    assert set(full.order_id) == expected
    assert set(mature.order_id) == expected
    assert set(eventual.order_id) == expected


@pytest.mark.parametrize("window_days", core.WINDOWS)
def test_10_scheme_a_exact_30_60_90_availability_boundaries(window_days: int) -> None:
    snapshot = pd.Timestamp("2018-04-01")
    epsilon = pd.Timedelta(microseconds=1)
    left = snapshot - pd.Timedelta(days=window_days)
    frame = _minimal_scheme_frame(
        ["left_in", "left_out", "right_in", "right_out"],
        [snapshot - pd.Timedelta(days=100)] * 4,
        [left, left - epsilon, snapshot - epsilon, snapshot],
    )
    full, mature, eventual = core.scheme_cohort(frame, "final_breach", snapshot, window_days, "A", 0, snapshot)
    expected = {"left_in", "right_in"}
    assert set(full.order_id) == expected
    assert set(mature.order_id) == expected
    assert set(eventual.order_id) == expected


@pytest.mark.parametrize("window_days", core.WINDOWS)
@pytest.mark.parametrize("lag_days", core.LAGS)
def test_11_scheme_c_exact_window_and_lag_boundaries(window_days: int, lag_days: int) -> None:
    snapshot = pd.Timestamp("2018-06-01")
    epsilon = pd.Timedelta(microseconds=1)
    left = snapshot - pd.Timedelta(days=lag_days + window_days)
    right = snapshot - pd.Timedelta(days=lag_days)
    frame = _minimal_scheme_frame(
        ["left_in", "left_out", "right_in", "right_out"],
        [left, left - epsilon, right - epsilon, right],
        [snapshot - pd.Timedelta(days=1)] * 4,
    )
    full, mature, eventual = core.scheme_cohort(frame, "final_breach", snapshot, window_days, "C", lag_days, snapshot)
    expected = {"left_in", "right_in"}
    assert set(full.order_id) == expected
    assert set(mature.order_id) == expected
    assert set(eventual.order_id) == expected


def test_12_eventual_endpoint_is_inclusive_and_later_label_excluded() -> None:
    snapshot = pd.Timestamp("2018-04-01")
    endpoint = pd.Timestamp("2018-04-15")
    frame = _minimal_scheme_frame(
        ["at_endpoint", "after_endpoint", "never"],
        [snapshot - pd.Timedelta(days=1)] * 3,
        [endpoint, endpoint + pd.Timedelta(microseconds=1), pd.NaT],
    )
    full, mature, eventual = core.scheme_cohort(frame, "final_breach", snapshot, 60, "B", 0, endpoint)
    assert set(full.order_id) == {"at_endpoint", "after_endpoint", "never"}
    assert mature.empty
    assert set(eventual.order_id) == {"at_endpoint"}


def test_13_unconditional_and_conditional_maturity_denominators() -> None:
    curves, quantiles, _ = core.maturity_outputs(tiny_frame(), pd.Timestamp("2018-12-01"))
    assert curves.denominator_orders.eq(4).all()
    assert set(quantiles.conditioning) == {"all_placed_orders", "conditional_on_eventual_observation"}
    assert len(curves) == len(core.TARGETS) * len(core.AGES)
    assert len(quantiles) == len(core.TARGETS) * len(core.MATURITY_THRESHOLDS) * 2


def test_14_unreachable_unconditional_threshold_returns_na() -> None:
    _, quantiles, _ = core.maturity_outputs(tiny_frame(), pd.Timestamp("2018-12-01"))
    row = quantiles[(quantiles.target == "final_breach") & (quantiles.conditioning == "all_placed_orders") & (quantiles.threshold == 0.99)].iloc[0]
    assert not bool(row.threshold_reached)
    assert pd.isna(row.age_days_required)
    assert row.reason == "eventual_observation_plateau_below_threshold"


def test_15_unresolved_is_never_zero_severity() -> None:
    assert pd.isna(tiny_frame().set_index("order_id").loc["unresolved", "positive_late_days"])


def test_16_negative_process_duration_is_observed_but_invalid() -> None:
    frame = tiny_frame()
    assert frame.handling_duration.lt(0).sum() == 1
    assert core.availability_mask(frame, "handling").sum() == 4
    assert core.target_observed_mask(frame, "handling").sum() == 4
    assert core.target_valid_mask(frame, "handling").sum() == 3


def test_17_all_seven_entity_identifiers_are_present() -> None:
    frame = tiny_frame()
    assert len(core.ENTITIES) == 7
    assert all(column in frame.columns for column in core.ENTITIES.values())


def test_18_missing_mapping_is_not_a_cold_start(synthetic_snapshot_outputs) -> None:
    _, _, coverage, _ = synthetic_snapshot_outputs
    row = coverage[
        (coverage.target == "final_breach")
        & (coverage.window_days == 30)
        & (coverage.scheme == "A")
        & (coverage.granularity == "seller_id")
        & (coverage.future_horizon_days == 7)
    ].iloc[0]
    assert row.total_future_placed_orders == 3
    assert row.orders_with_valid_entity_mapping == 2
    assert row.historical_seen_orders == 1
    assert row.mapped_cold_start_orders == 1
    assert row.missing_mapping_count == 1
    assert row.mapped_cold_start_orders + row.missing_mapping_count != row.mapped_cold_start_orders
    assert row.cold_start_rate == pytest.approx(1 / 3)
    assert row.cold_start_rate_among_mapped == pytest.approx(1 / 2)


def test_19_future_seen_and_support_qualified_coverage(synthetic_snapshot_outputs) -> None:
    _, _, coverage, _ = synthetic_snapshot_outputs
    row = coverage[
        (coverage.target == "final_breach")
        & (coverage.window_days == 30)
        & (coverage.scheme == "A")
        & (coverage.granularity == "seller_id")
        & (coverage.future_horizon_days == 7)
    ].iloc[0]
    assert row.orders_support_ge_1 == 1
    assert row.orders_support_ge_5 == 0
    assert row.order_weighted_support_ge_1_rate == pytest.approx(1 / 3)
    assert row.support_quantile_denominator == "mapped_future_orders_including_seen_and_unseen"


def test_20_rank_output_has_every_granularity() -> None:
    rank = pd.read_csv(output("ENTITY_RANK_AGREEMENT_ALL_GRANULARITIES.csv"), low_memory=False)
    assert set(rank.granularity) == set(core.ENTITIES)


def test_21_invalid_rank_reason_is_explicit() -> None:
    small = pd.DataFrame({"entity": ["a", "b"], "support": [5, 5], "mean": [1.0, 1.0]})
    result = core.spearman_row(small, small, "entity", "mean", 1)
    assert not result["valid"]
    assert result["invalid_reason"] == "fewer_than_10_common_entities"


def test_22_constant_rank_vector_is_explicit_after_minimum_support() -> None:
    entities = [f"e{i}" for i in range(10)]
    left = pd.DataFrame({"entity": entities, "support": [5] * 10, "mean": [1.0] * 10})
    right = pd.DataFrame({"entity": entities, "support": [5] * 10, "mean": range(10)})
    result = core.spearman_row(left, right, "entity", "mean", 5)
    assert not result["valid"]
    assert result["invalid_reason"] == "constant_vector"
    assert result["constant_vector_a"] and not result["constant_vector_b"]


def test_23_hrd_daily_order_count_uses_all_placed_orders() -> None:
    daily = pd.read_csv(output("DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv"))
    assert daily.order_count.sum() == 99_441


def test_24_gmv_join_coverage_reconciles() -> None:
    daily = pd.read_csv(output("DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv"))
    assert {"orders_with_gmv", "orders_missing_gmv", "gmv_join_coverage"}.issubset(daily.columns)
    assert (daily.orders_with_gmv + daily.orders_missing_gmv).equals(daily.order_count)
    nonempty = daily.order_count.gt(0)
    assert np.allclose(
        daily.loc[nonempty, "gmv_join_coverage"],
        daily.loc[nonempty, "orders_with_gmv"] / daily.loc[nonempty, "order_count"],
    )


def test_25_calendar_contiguous_clustering_uses_actual_dates() -> None:
    daily = pd.DataFrame(
        {"date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-04"]), **{flag: [True, True, True] for flag in core.HRD_DEFS}}
    )
    clusters, _ = core.calendar_clusters(daily)
    assert clusters.loc[clusters.definition == "order_top10", "duration_days"].tolist() == [2, 1]


def test_26_interval_uses_approved_recent_60d_cohort_and_endpoint() -> None:
    frame, endpoint = interval_fixture()
    audit, first, completion_last, common_last = core.derive_intervals(frame, endpoint)
    assert first == pd.Timestamp("2018-04-01")
    assert completion_last == pd.Timestamp("2018-04-02")
    assert common_last == pd.Timestamp("2018-04-02")
    target_rows = audit[audit.boundary_scope == "target_purchase_comparison"]
    assert len(target_rows) == len(core.TARGETS)
    assert target_rows.last_eligible_snapshot.eq(pd.Timestamp("2018-04-02")).all()
    # Exact [t-60,t): 100 anchors + left boundary + right-minus-epsilon.
    assert target_rows.cohort_total_orders.eq(102).all()
    # The right-minus-epsilon order is in the denominator but after the endpoint.
    assert target_rows.available_by_audit_endpoint.eq(101).all()
    assert np.allclose(target_rows.unconditional_availability, 101 / 102)


def test_27_scheme_c_full_denominator_keeps_unresolved_and_immature() -> None:
    snapshot = pd.Timestamp("2018-04-01")
    endpoint = pd.Timestamp("2018-12-01")
    frame = _minimal_scheme_frame(
        ["mature", "immature", "never"],
        [pd.Timestamp("2018-02-10")] * 3,
        [pd.Timestamp("2018-03-01"), pd.Timestamp("2018-04-05"), pd.NaT],
    )
    full, mature, eventual = core.scheme_cohort(frame, "final_breach", snapshot, 30, "C", 30, endpoint)
    assert len(full) == 3
    assert set(mature.order_id) == {"mature"}
    assert set(eventual.order_id) == {"mature", "immature"}
    assert len(full) - len(mature) == 2
    assert len(full) - len(eventual) == 1


def test_28_scheme_a_purchase_maturity_is_not_applicable() -> None:
    slicing = pd.read_csv(output("SLICING_SCHEME_COMPARISON_V1_1.csv"), low_memory=False)
    scheme_a = slicing[slicing.scheme == "A"]
    assert scheme_a.unconditional_maturity_fraction.isna().all()
    assert scheme_a.purchase_cohort_maturity_reason.eq("not_applicable_for_completion_window").all()


def test_29_bc_persisted_denominators_reconcile_exactly() -> None:
    slicing = pd.read_csv(output("SLICING_SCHEME_COMPARISON_V1_1.csv"), low_memory=False)
    cohorts = slicing[slicing.scheme.isin(["B", "C"])]
    assert (cohorts.mature_asof_orders + cohorts.unresolved_asof_orders == cohorts.cohort_total_orders_all_placed).all()
    assert (
        cohorts.eventually_available_orders_by_audit_end + cohorts.never_observed_by_audit_end_orders
        == cohorts.cohort_total_orders_all_placed
    ).all()
    nonempty = cohorts.cohort_total_orders_all_placed.gt(0)
    assert np.allclose(
        cohorts.loc[nonempty, "unconditional_maturity_fraction"],
        cohorts.loc[nonempty, "mature_asof_orders"] / cohorts.loc[nonempty, "cohort_total_orders_all_placed"],
    )


EXACT_SCHEMAS = {
    "CANONICAL_SAMPLE_RECONCILIATION.csv": {
        "order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "customer_join", "item_join", "in_canonical", "status_delivered",
        "customer_delivery_observed", "estimate_observed", "purchase_observed", "deterministic_reconciliation_reason",
    },
    "MATURITY_CURVES_UNCONDITIONAL.csv": {
        "target", "conditioning", "age_days", "denominator_orders", "available_by_age_orders",
        "unconditional_cumulative_availability", "eventual_observation_plateau",
    },
    "MATURITY_QUANTILES_CONDITIONAL.csv": {
        "target", "conditioning", "threshold", "threshold_reached", "age_days_required",
        "eventual_observation_plateau", "reason",
    },
    "MATURITY_COMPONENT_COUNTS.csv": {
        "target", "all_placed_orders", "availability_timestamp_observed", "availability_by_audit_endpoint",
        "target_value_observed", "target_value_valid_for_descriptive_summary", "negative_duration_count",
        "missing_component_count", "eventual_observation_plateau",
    },
    "DATA_QUALITY_AUDIT_V1_1.csv": {"scope", "check", "count", "denominator", "percentage", "detail"},
    "PROCESS_DURATION_DIAGNOSTICS.csv": {"target", "scope", "count", "mean", "p01", "median", "p90", "p95", "p99"},
    "SNAPSHOT_INTERVAL_AUDIT.csv": {
        "boundary_scope", "target", "raw_data_start", "first_90d_warmup_snapshot", "last_eligible_snapshot",
        "candidate_snapshot_domain_start", "candidate_snapshot_domain_end",
        "audit_endpoint_proxy", "cohort_total_orders", "available_by_audit_endpoint", "unconditional_availability", "reason",
        "maturity_threshold", "maturity_achieved",
    },
    "HRD_EVENT_CLUSTERS.csv": {"definition", "cluster_id", "start_date", "end_date", "duration_days", "n_hrd_days"},
    "HRD_EVENT_PHASES.csv": {"definition", "cluster_id", "date", "phase", "phase_day", "date_present_in_marketplace_table"},
    "HRD_DEFINITION_OVERLAP.csv": {"definition_a", "definition_b", "days_intersection", "days_union", "jaccard"},
}

REQUIRED_SCHEMA_SUBSETS = {
    "SLICING_SCHEME_COMPARISON_V1_1.csv": {
        "snapshot_date", "sample", "target", "target_kind", "window_days", "scheme", "lag_days", "granularity",
        "source_records", "entity_id_available_orders", "timestamp_observed_asof_orders", "valid_outcomes_asof",
        "invalid_or_anomalous_outcomes_asof", "cohort_total_orders_all_placed", "mature_asof_orders",
        "unresolved_asof_orders", "eventually_available_orders_by_audit_end", "never_observed_by_audit_end_orders",
        "unconditional_maturity_fraction", "conditional_maturity_fraction", "eventual_observed_fraction",
        "purchase_cohort_maturity_reason", "asof_breach_count", "asof_breach_rate", "eventual_breach_rate",
        "asof_zero_severity_share", "asof_raw_duration_count", "asof_negative_duration_count",
    },
    "ENTITY_GRANULARITY_SUPPORT_V1_1.csv": {
        "snapshot_date", "sample", "target", "window_days", "scheme", "lag_days", "granularity", "active_entities",
        "entity_id_nonmissing_rate", "future_7d_seen_rate", "future_30d_seen_rate", "support_median",
        "entities_support_ge_1", "entities_support_ge_5", "entities_support_ge_10", "entities_support_ge_20", "entities_support_ge_50",
    },
    "FUTURE_PROFILE_COVERAGE.csv": {
        "snapshot_date", "sample", "target", "window_days", "scheme", "lag_days", "granularity", "future_horizon_days",
        "total_future_placed_orders", "orders_with_valid_entity_mapping", "entity_id_nonmissing_rate",
        "historical_seen_orders", "historical_seen_rate", "mapped_cold_start_orders", "cold_start_rate",
        "cold_start_rate_among_mapped", "missing_mapping_count", "multi_seller_count", "orders_support_ge_1",
        "orders_support_ge_5", "orders_support_ge_10", "orders_support_ge_20", "orders_support_ge_50",
    },
    "ENTITY_RANK_AGREEMENT_ALL_GRANULARITIES.csv": {
        "snapshot_date", "target", "window_days", "granularity", "entity_statistic", "scheme_a", "lag_a_days",
        "scheme_b", "lag_b_days", "n_entities_source_a", "n_entities_source_b", "n_common_entities",
        "support_threshold", "spearman_correlation", "p_value", "constant_vector_a", "constant_vector_b",
        "valid", "invalid_reason", "interpretation",
    },
    "DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv": {
        "date", "order_count", "delivered_count", "unresolved_or_cancelled_count", "active_sellers", "active_routes",
        "orders_with_gmv", "orders_missing_gmv", "gmv_join_coverage", "total_gmv", "freight_value",
        *core.HRD_DEFS,
    },
    "HRD_FEASIBILITY_V1_1.csv": {
        "definition", "granularity", "n_hrd_days", "n_event_clusters", "n_orders", "pct_all_placed_orders",
        "active_entities", "entity_id_nonmissing_rate", "historical_hrd_clusters_available",
        "entities_support_ge_1", "entities_support_ge_5", "entities_support_ge_10", "entities_support_ge_20", "entities_support_ge_50",
    },
}


def test_30_output_schema_contracts() -> None:
    for name, expected in EXACT_SCHEMAS.items():
        actual = set(pd.read_csv(output(name), nrows=0).columns)
        assert actual == expected, f"{name}: missing={expected-actual}, extra={actual-expected}"
    for name, required in REQUIRED_SCHEMA_SUBSETS.items():
        actual = set(pd.read_csv(output(name), nrows=0).columns)
        assert required.issubset(actual), f"{name}: missing={required-actual}"


def test_31_manifest_csv_rows_and_sha256_recompute_exactly() -> None:
    manifest = json.loads(output("RUN_MANIFEST.json").read_text())
    assert manifest["csv_artifacts"]
    for relative, recorded in manifest["csv_artifacts"].items():
        path = core.OUT / relative
        assert path.is_file(), relative
        assert core.sha256_file(path) == recorded["sha256"], relative
        row_count = len(pd.read_csv(path, usecols=[0], low_memory=False))
        assert row_count == recorded["rows"], relative


def test_32_design_artifact_row_counts_follow_frozen_formula() -> None:
    manifest = json.loads(output("RUN_MANIFEST.json").read_text())
    snapshots = int(manifest["snapshot_intervals"]["completion"]["days"])
    design_cells = snapshots * len(core.TARGETS) * len(core.WINDOWS) * (2 + len(core.LAGS)) * len(core.ENTITIES)
    slicing = pd.read_csv(output("SLICING_SCHEME_COMPARISON_V1_1.csv"), usecols=["snapshot_date"])
    support = pd.read_csv(output("ENTITY_GRANULARITY_SUPPORT_V1_1.csv"), usecols=["snapshot_date"])
    coverage = pd.read_csv(output("FUTURE_PROFILE_COVERAGE.csv"), usecols=["snapshot_date"])
    assert len(slicing) == design_cells
    assert len(support) == design_cells
    assert len(coverage) == design_cells * len(core.HORIZONS)
    assert len(pd.read_csv(output("MATURITY_CURVES_UNCONDITIONAL.csv"))) == len(core.TARGETS) * len(core.AGES)
    assert len(pd.read_csv(output("MATURITY_QUANTILES_CONDITIONAL.csv"))) == len(core.TARGETS) * len(core.MATURITY_THRESHOLDS) * 2
    assert len(pd.read_csv(output("MATURITY_COMPONENT_COUNTS.csv"))) == len(core.TARGETS)
    assert len(pd.read_csv(output("HRD_FEASIBILITY_V1_1.csv"))) == len(core.HRD_DEFS) * len(core.ENTITIES)
    assert len(pd.read_csv(output("HRD_DEFINITION_OVERLAP.csv"))) == len(core.HRD_DEFS) ** 2


def test_33_rank_row_count_follows_frozen_pair_statistic_formula() -> None:
    rank = pd.read_csv(output("ENTITY_RANK_AGREEMENT_ALL_GRANULARITIES.csv"), usecols=["snapshot_date"])
    rank_snapshots = rank.snapshot_date.nunique()
    comparison_pairs = 1 + len(core.LAGS) + len(core.LAGS) + (len(core.LAGS) - 1)
    statistics_across_targets = sum(1 if spec["kind"] == "binary" else 3 for spec in core.TARGETS.values())
    expected = rank_snapshots * len(core.WINDOWS) * len(core.ENTITIES) * comparison_pairs * 4 * statistics_across_targets
    assert len(rank) == expected


def test_34_reconciliation_96476_96470_and_six_reasons() -> None:
    reconciliation = pd.read_csv(output("CANONICAL_SAMPLE_RECONCILIATION.csv"))
    six = reconciliation[reconciliation.customer_delivery_observed & ~reconciliation.in_canonical]
    assert reconciliation.customer_delivery_observed.sum() == 96_476
    assert reconciliation.in_canonical.sum() == 96_470
    assert len(six) == 6
    assert six.order_status.eq("canceled").all()
    assert six.deterministic_reconciliation_reason.str.startswith("status_inconsistency:").all()


def test_35_v1_and_phase2a_protected_hashes_unchanged() -> None:
    manifest = json.loads(output("RUN_MANIFEST.json").read_text())
    assert manifest["protected_hashes_before"]["v1"] == manifest["protected_hashes_after"]["v1"]
    assert manifest["protected_hashes_before"]["phase2a"] == manifest["protected_hashes_after"]["phase2a"]


def test_36_figure_source_rows_and_hashes_recompute() -> None:
    manifest = json.loads(output("RUN_MANIFEST.json").read_text())
    assert len(manifest["figures"]) == 14
    for figure_name, recorded in manifest["figures"].items():
        figure = core.OUT / "figures" / figure_name
        source = core.OUT / recorded["source_csv"]
        assert core.sha256_file(figure) == recorded["sha256"]
        assert core.sha256_file(source) == recorded["source_sha256"]
        assert len(pd.read_csv(source)) == recorded["source_rows"]


def test_37_no_hrd_pass_fail_or_winner_field() -> None:
    columns = pd.read_csv(output("HRD_FEASIBILITY_V1_1.csv"), nrows=0).columns
    assert not any(any(word in column.lower() for word in ["pass", "feasible", "winner"]) for column in columns)


def test_38_minimal_end_to_end_synthetic_fixture(tmp_path: Path) -> None:
    frame = coverage_fixture()
    snapshot = pd.Timestamp("2018-02-01")
    endpoint = pd.Timestamp("2018-12-01")
    run.snapshot_worker_init(frame, endpoint, {snapshot})

    part_dir = tmp_path / "parts"
    part_dir.mkdir()
    run.chunk_worker((0, [str(snapshot)], str(part_dir)))
    expected_rows = {"slicing": 672, "support": 672, "coverage": 1_344, "rank": 15_120}
    sort_columns = {
        "slicing": ["snapshot_date", "target", "window_days", "scheme", "lag_days", "granularity"],
        "support": ["snapshot_date", "target", "window_days", "scheme", "lag_days", "granularity"],
        "coverage": ["snapshot_date", "target", "window_days", "scheme", "lag_days", "granularity", "future_horizon_days"],
        "rank": ["snapshot_date", "target", "window_days", "granularity", "scheme_a", "lag_a_days", "scheme_b", "lag_b_days", "entity_statistic", "support_threshold"],
    }
    first_hashes = {}
    for kind, expected in expected_rows.items():
        destination = tmp_path / f"{kind}.csv"
        combined = run.combine_parts(part_dir, kind, destination, sort_columns[kind])
        assert len(combined) == expected
        first_hashes[kind] = core.sha256_file(destination)
        second = tmp_path / f"{kind}_repeat.csv"
        run.combine_parts(part_dir, kind, second, sort_columns[kind])
        assert core.sha256_file(second) == first_hashes[kind]

    curves, quantiles, components = core.maturity_outputs(frame, endpoint)
    daily = run.all_placed_daily(frame)
    clusters, phases = core.calendar_clusters(daily)
    hrd, overlap = run.hrd_outputs(frame, daily, clusters)
    quality = core.data_quality(frame, canonical_quality_fixture())
    durations = core.duration_diagnostics(frame)
    assert (len(curves), len(quantiles), len(components)) == (36, 32, 4)
    assert daily.order_count.sum() == len(frame)
    assert len(hrd) == 42 and len(overlap) == 36
    assert not clusters.empty and not phases.empty
    assert not quality.empty and len(durations) == 4


def test_39_fast_engine_matches_reference_full_synthetic_snapshot() -> None:
    frame=coverage_fixture(); snapshot=pd.Timestamp("2018-02-01"); endpoint=pd.Timestamp("2018-12-01")
    run.snapshot_worker_init(frame,endpoint,{snapshot})
    reference=tuple(pd.DataFrame(rows) for rows in run.process_snapshot(str(snapshot.date())))
    candidate=fast_engine.compute_snapshot_outputs(frame,endpoint,[snapshot],{snapshot})
    for name,left,right in zip(["slicing","support","coverage","rank"],reference,candidate):
        assert set(left.columns)==set(right.columns)
        columns=sorted(left.columns); sort_columns=run.PARITY_SORT_COLUMNS[name]
        left=left.sort_values(sort_columns,kind="mergesort").reset_index(drop=True)[columns]
        right=right.sort_values(sort_columns,kind="mergesort").reset_index(drop=True)[columns]
        pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=False,rtol=1e-11,atol=1e-11)


def test_40_empty_nonrank_chunk_combines_without_parser_failure(tmp_path: Path) -> None:
    frame=coverage_fixture(); snapshot=pd.Timestamp("2018-02-02"); endpoint=pd.Timestamp("2018-12-01")
    run.snapshot_worker_init(frame,endpoint,set()); part_dir=tmp_path/"parts"; part_dir.mkdir()
    run.chunk_worker((0,[str(snapshot.date())],str(part_dir)))
    assert not list(part_dir.glob("*_rank.csv"))
    combined=run.combine_parts(part_dir,"rank",tmp_path/"rank.csv",run.PARITY_SORT_COLUMNS["rank"])
    assert combined.empty


def test_41_manifest_records_exact_orig_argv_including_b_flag() -> None:
    """`sys.argv` drops interpreter flags; provenance must use `sys.orig_argv`."""
    manifest=json.loads(output("RUN_MANIFEST.json").read_text())
    pipeline_tokens=shlex.split(manifest["commands"][0])
    expected_pipeline_tail=[
        "-B", "analysis/dynamic_profile_eda_v1_1/scripts/run_eda_v1_1.py",
        "--data-dir", str(DATA_DIR), "--workers", str(manifest["environment"]["workers"]),
    ]
    assert Path(pipeline_tokens[0]).resolve()==Path(sys.executable).resolve()
    assert pipeline_tokens[1:]==expected_pipeline_tail, "pipeline command did not preserve sys.orig_argv exactly"

    actual_test_tokens=list(sys.orig_argv)
    recorded_test_tokens=shlex.split(manifest["tests"]["command"])
    assert "-B" in actual_test_tokens[1:], "deterministic suite itself was not launched with -B"
    assert Path(recorded_test_tokens[0]).resolve()==Path(actual_test_tokens[0]).resolve()
    assert recorded_test_tokens[1:]==[
        "-B", "-m", "pytest", "analysis/dynamic_profile_eda_v1_1/scripts/test_v1_1.py",
        "-q", "-p", "no:cacheprovider",
    ]
