from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/profile-validation-v1-mpl-tests")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pytest

from analysis.dynamic_profile_profile_validation_v1.scripts import profile_core as core


CONFIG = core.load_config()
OUT = core.OUT
V11 = core.V11

REQUIRED_OUTPUTS = (
    "PROFILE_PROTOCOL.md",
    "PROFILE_FROZEN_CONFIG.json",
    "PROFILE_SELECTION_FREEZE.json",
    "PROFILE_DATA_DICTIONARY.md",
    "PROFILE_CONSTRUCTION_AUDIT.csv",
    "PROFILE_DAILY_SCORES.csv",
    "PROFILE_SUPPORT_UNCERTAINTY.csv",
    "PROFILE_PARENT_STRUCTURE.csv",
    "PROFILE_DEVELOPMENT_RESULTS.csv",
    "PROFILE_DEVELOPMENT_BY_MONTH.csv",
    "PROFILE_PARETO_FRONTIER.csv",
    "PROFILE_SELECTED_CANDIDATES.csv",
    "PROFILE_CONFIRMATION_RESULTS.csv",
    "PROFILE_CONFIRMATION_BY_MONTH.csv",
    "PROFILE_TERMINAL_STRESS.csv",
    "PROFILE_LEVEL_RESULTS.csv",
    "PROFILE_LEVEL_TRANSITIONS.csv",
    "PROFILE_DAILY_STABILITY.csv",
    "PROFILE_FUTURE_ENTITY_TRANSFER.csv",
    "PROFILE_FUTURE_ORDER_SCORING.csv",
    "PROFILE_SUPPORT_STRATA.csv",
    "PROFILE_COLD_START_RESULTS.csv",
    "PROFILE_HRD_DIAGNOSTICS.csv",
    "PROFILE_ABLATIONS.csv",
    "PROFILE_RESULTS_SUMMARY.md",
    "PROFILE_RESULTS_SUMMARY_ZH.md",
    "BLOCKERS.md",
    "RUN_MANIFEST.json",
    "TEST_RESULTS.txt",
    "ARTIFACT_VALIDATION_REPORT.md",
)

FIGURE_STEMS = (
    "01_seller_support_vs_uncertainty",
    "02_route_support_vs_uncertainty",
    "03_raw_vs_eb_seller_scores",
    "04_raw_vs_eb_route_scores",
    "05_adjusted_vs_unadjusted_scores",
    "06_window_30_60_90_comparison",
    "07_scheme_a_vs_c_comparison",
    "08_development_future_rank_transfer",
    "09_confirmation_future_rank_transfer",
    "10_top_quintile_future_lift",
    "11_future_outcome_by_level",
    "12_daily_profile_stability",
    "13_level_transition_heatmap",
    "14_coverage_by_support_threshold",
    "15_seller_cold_start",
    "16_state_od_vs_region_od",
    "17_development_vs_confirmation",
    "18_terminal_stress",
)

METRIC_SCHEMA = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "anchor_date", "calendar_month", "horizon_days", "stratum_type",
    "stratum_value", "metric_name", "reference_id", "aggregation",
    "n_scheduled_anchors", "n_valid_anchors", "n_orders", "n_events",
    "n_entities", "n_common_entities", "estimate", "ci_lower", "ci_upper",
    "valid", "invalid_reason",
)

EXACT_ARTIFACT_SCHEMAS = {
    "PROFILE_CONSTRUCTION_AUDIT.csv": (
        "base_candidate_id", "snapshot_date", "period", "target", "granularity",
        "scheme", "window_days", "lag_days", "estimator", "parent_structure",
        "kappa", "history_sample", "future_denominator_sample",
        "source_interval_axis", "source_interval_start", "source_interval_end",
        "availability_cutoff", "entity_domain_count", "source_orders_observed",
        "source_orders_valid", "source_orders_excluded_negative",
        "affected_entities_negative", "profile_rows", "parent_rows",
        "cold_start_rows", "coverage_before_negative_exclusion",
        "coverage_after_negative_exclusion", "max_source_purchase_at",
        "max_source_label_available_at", "last_mature_outcome_date",
        "strict_asof_pass", "window_pass", "valid", "invalid_reason",
    ),
    "PROFILE_DAILY_SCORES.csv": (
        "schema_version", "relative_path", "target", "granularity", "scheme",
        "window_days", "lag_days", "estimator", "snapshot_date_min",
        "snapshot_date_max", "row_count", "sha256", "primary_key_columns",
        "sort_columns",
    ),
    "PROFILE_SUPPORT_UNCERTAINTY.csv": (
        "base_candidate_id", "snapshot_date", "period", "target", "granularity",
        "scheme", "window_days", "lag_days", "estimator", "parent_structure",
        "kappa", "support_stratum", "entity_count", "order_exposure",
        "median_support", "median_score", "median_posterior_se",
        "p90_posterior_se", "median_interval_width", "p90_interval_width",
        "cold_start_count", "valid", "invalid_reason",
    ),
    "PROFILE_PARENT_STRUCTURE.csv": (
        "base_candidate_id", "snapshot_date", "target", "granularity",
        "parent_structure", "parent_id", "parent_support", "parent_event_count",
        "parent_score", "global_score", "parent_within_variance",
        "parent_between_variance", "parent_posterior_se", "parent_interval_lower",
        "parent_interval_upper", "fallback_child_count", "parent_supported",
        "valid", "invalid_reason",
    ),
    "PROFILE_DEVELOPMENT_RESULTS.csv": METRIC_SCHEMA,
    "PROFILE_DEVELOPMENT_BY_MONTH.csv": METRIC_SCHEMA,
    "PROFILE_CONFIRMATION_RESULTS.csv": METRIC_SCHEMA,
    "PROFILE_CONFIRMATION_BY_MONTH.csv": METRIC_SCHEMA,
    "PROFILE_TERMINAL_STRESS.csv": METRIC_SCHEMA + (
        "outcome_observation_rate", "followup_available_days",
        "maturity_censoring_flag", "distribution_shift_reference",
    ),
    "PROFILE_LEVEL_RESULTS.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days", "level", "metric_name", "n_orders", "n_entities",
        "future_support", "estimate", "ci_lower", "ci_upper", "monotone_lmh",
        "percent_unknown", "valid", "invalid_reason",
    ),
    "PROFILE_LEVEL_TRANSITIONS.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "from_level", "to_level", "transition_count", "eligible_from_count",
        "transition_probability", "median_persistence_days", "ci_lower",
        "ci_upper", "valid", "invalid_reason",
    ),
    "PROFILE_DAILY_STABILITY.csv": (
        "base_candidate_id", "target", "granularity", "previous_snapshot_date",
        "snapshot_date", "period", "regime", "n_common_entities",
        "newly_matured_support", "day_to_day_spearman",
        "median_absolute_score_change", "p90_absolute_score_change",
        "top20_jaccard", "score_change_per_new_label",
        "entities_changing_level", "pct_entities_changing_level",
        "cold_start_entries", "cold_start_exits",
        "transition_entity_union_count", "cold_start_entry_rate",
        "cold_start_exit_rate", "valid", "invalid_reason",
    ),
    "PROFILE_FUTURE_ENTITY_TRANSFER.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "anchor_date", "horizon_days", "stratum_type", "stratum_value",
        "n_common_entities", "future_support", "unweighted_spearman",
        "weighted_spearman", "top_quintile_lift", "high_low_risk_ratio",
        "spearman_ci_lower", "spearman_ci_upper", "lift_ci_lower",
        "lift_ci_upper", "bootstrap_unit", "bootstrap_resamples", "valid",
        "invalid_reason",
    ),
    "PROFILE_FUTURE_ORDER_SCORING.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "anchor_date", "horizon_days", "order_id", "purchase_timestamp",
        "entity_id", "mapping_status", "history_support", "cold_start",
        "profile_score", "parent_score", "level", "target_observed",
        "target_value", "label_available_at", "eligible_for_metric",
    ),
    "PROFILE_SUPPORT_STRATA.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days", "support_stratum", "metric_name", "reference_id",
        "n_orders", "n_events", "n_entities", "estimate", "ci_lower",
        "ci_upper", "valid", "invalid_reason",
    ),
    "PROFILE_COLD_START_RESULTS.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days", "mapping_status", "n_orders", "order_share",
        "metric_name", "estimate", "ci_lower", "ci_upper", "valid",
        "invalid_reason",
    ),
    "PROFILE_HRD_DIAGNOSTICS.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "hrd_definition", "regime", "phase", "horizon_days", "n_days",
        "n_orders", "n_entities", "historical_support", "metric_name",
        "estimate", "ci_lower", "ci_upper", "valid", "invalid_reason",
    ),
    "PROFILE_ABLATIONS.csv": (
        "candidate_id", "base_candidate_id", "target", "granularity", "period",
        "horizon_days", "ablation_id", "stratum_type", "stratum_value",
        "metric_name", "reference_id", "n_orders", "n_events", "n_entities",
        "estimate", "ci_lower", "ci_upper", "valid", "invalid_reason",
    ),
}


def _source(target: str = "final_breach", granularity: str = "seller_id", **overrides) -> dict[str, object]:
    result: dict[str, object] = {
        "target": target,
        "granularity": granularity,
        "scheme": "A",
        "window_days": 30,
        "lag_days": 0,
    }
    result.update(overrides)
    return result


def _binary_history() -> pd.DataFrame:
    values = [1.0, 1.0, 0.0, 0.0]
    sellers = ["s1", "s1", "s1", "s2"]
    states = ["SP", "SP", "SP", "RJ"]
    return pd.DataFrame({
        "order_id": [f"b{i}" for i in range(4)],
        "order_purchase_timestamp": pd.to_datetime(["2017-05-01", "2017-05-02", "2017-05-03", "2017-05-04"]),
        "final_breach_available_at": pd.to_datetime(["2017-05-10", "2017-05-11", "2017-05-12", "2017-05-13"]),
        "late_delivery": values,
        "seller_id": sellers,
        "main_seller_state": states,
        "state_od": ["SP -> RJ", "SP -> RJ", "SP -> RJ", "RJ -> SP"],
        "region_od": ["Southeast -> Southeast"] * 4,
        "expected_final_breach": [0.2, 0.8, 0.5, 0.5],
    })


def _supported_seller_parent_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(40):
        state = "SP" if i < 20 else "RJ"
        seller = "s1" if i < 4 else ("s_sp_other" if i < 20 else "s_rj")
        if i < 4:
            event = [1.0, 1.0, 1.0, 0.0][i]
        elif i < 20:
            event = 1.0 if i < 11 else 0.0  # SP total: 10/20.
        else:
            event = 1.0 if i < 30 else 0.0  # RJ total: 10/20.
        rows.append({
            "order_id": f"p{i:02d}",
            "order_purchase_timestamp": pd.Timestamp("2017-04-01") + pd.Timedelta(days=i),
            "final_breach_available_at": pd.Timestamp("2017-04-03") + pd.Timedelta(days=i),
            "late_delivery": event,
            "seller_id": seller,
            "main_seller_state": state,
            "state_od": f"{state} -> MG",
            "region_od": "Southeast -> Southeast",
            "expected_final_breach": 0.5,
        })
    return pd.DataFrame(rows)


def _continuous_history(degenerate: bool = False) -> pd.DataFrame:
    values = [1.0, 1.0, 1.0, 1.0] if degenerate else [0.0, 2.0, 4.0, 6.0]
    return pd.DataFrame({
        "order_id": [f"c{i}" for i in range(4)],
        "order_purchase_timestamp": pd.to_datetime(["2017-05-01", "2017-05-02", "2017-05-03", "2017-05-04"]),
        "handling_available_at": pd.to_datetime(["2017-05-05", "2017-05-06", "2017-05-07", "2017-05-08"]),
        "handling_level_value": values,
        "handling_duration": np.expm1(values),
        "seller_id": ["s1", "s1", "s2", "s2"],
        "main_seller_state": ["SP", "SP", "RJ", "RJ"],
        "region_od": ["Southeast -> Southeast"] * 4,
        "expected_handling_level": [0.0, 0.0, 0.0, 0.0],
    })


def _history_boundary_frame(snapshot: pd.Timestamp, window: int, lag: int = 0) -> pd.DataFrame:
    epsilon = pd.Timedelta(microseconds=1)
    left = snapshot - pd.Timedelta(days=window + lag)
    right = snapshot - pd.Timedelta(days=lag)
    return pd.DataFrame({
        "order_id": ["left_in", "left_out", "right_in", "right_out", "availability_equal"],
        "order_purchase_timestamp": [left, left - epsilon, right - epsilon, right, left],
        "final_breach_available_at": [snapshot - pd.Timedelta(days=1)] * 4 + [snapshot],
        "late_delivery": [1.0, 0.0, 1.0, 0.0, 1.0],
        "in_canonical": [True] * 5,
        "seller_id": ["s1"] * 5,
        "main_seller_state": ["SP"] * 5,
        "region_od": ["Southeast -> Southeast"] * 5,
    })


def _future_mapping_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    base_id = "final_breach|seller_id|A|w30|l0|P1|parent=global|kappa=10"
    profile = pd.DataFrame({
        "base_candidate_id": [base_id], "entity_id": ["s1"],
        "parent_structure": ["global"], "score": [0.8], "raw_score": [0.75],
        "support": [8], "posterior_se": [0.05], "lower_interval": [0.70],
        "upper_interval": [0.90], "global_score": [0.20],
    })
    parents = pd.DataFrame({
        "base_candidate_id": [base_id], "parent_id": ["__GLOBAL__"],
        "parent_score": [0.20],
    })
    future = pd.DataFrame({
        "order_id": ["seen", "cold", "missing", "unresolved"],
        "order_purchase_timestamp": pd.to_datetime(["2018-02-01", "2018-02-02", "2018-02-03", "2018-02-04"]),
        "seller_id": pd.Series(["s1", "s9", pd.NA, "s1"], dtype="string"),
        "main_seller_state": ["SP", "RJ", pd.NA, "SP"],
        "state_od": ["SP -> RJ", "RJ -> SP", pd.NA, "SP -> RJ"],
        "region_od": ["Southeast -> Southeast", "Southeast -> Southeast", pd.NA, "Southeast -> Southeast"],
        "in_canonical": [True, True, False, False],
        "late_delivery": [1.0, 0.0, np.nan, np.nan],
        "promise_error_days": [2.0, -1.0, np.nan, np.nan],
        "final_breach_available_at": pd.to_datetime(["2018-02-10", "2018-02-11", pd.NaT, pd.NaT]),
    })
    source = _source()
    return future, profile, parents, source, base_id


def _row_origin_fixture() -> pd.DataFrame:
    historic_dates = list(pd.date_range("2016-09-01", periods=110, freq="D")) * 2
    future_dates = list(pd.date_range("2017-01-02", periods=20, freq="D")) * 2
    purchases = pd.to_datetime(historic_dates + future_dates)
    n = len(purchases)
    idx = np.arange(n)
    frame = pd.DataFrame({
        "order_id": [f"n{i:03d}" for i in range(n)],
        "order_purchase_timestamp": purchases,
        "in_canonical": True,
        "seller_id": np.where(idx % 2 == 0, "s1", "s2"),
        "main_seller_state": np.where(idx % 2 == 0, "SP", "RJ"),
        "state_od": np.where(idx % 2 == 0, "SP -> RJ", "RJ -> SP"),
        "region_od": "Southeast -> Southeast",
    })
    available = purchases + pd.Timedelta(days=1)
    binary = (idx % 2).astype(float)
    frame["final_breach_available_at"] = available
    frame["late_delivery"] = binary
    frame["promise_error_days"] = np.where(binary == 1, 2.0, -2.0)
    frame["handling_available_at"] = available
    frame["handling_duration"] = 1.0 + (idx % 5)
    frame["handling_level_value"] = np.log1p(frame["handling_duration"])
    frame["handling_tail"] = binary
    frame["transit_available_at"] = available
    frame["transit_duration"] = 2.0 + (idx % 7)
    frame["transit_level_value"] = np.log1p(frame["transit_duration"])
    frame["transit_tail"] = binary
    frame["positive_late_days_available_at"] = available
    frame["positive_late_days"] = 1.0 + (idx % 4)
    frame["positive_late_severity_value"] = np.log1p(frame["positive_late_days"])
    for position, name in enumerate(CONFIG["p2"]["features_numeric"]):
        frame[name] = (idx % (position + 3)).astype(float)
    for name in CONFIG["p2"]["features_categorical"]:
        frame[name] = np.where(idx % 2 == 0, "A", "B")
    return frame


def _require_stage_api(*names: str) -> None:
    missing = [name for name in names if not hasattr(core, name)]
    if missing:
        pytest.skip(f"stage-gate API not implemented yet: {missing}")


def _require_artifacts() -> None:
    if not (OUT / "RUN_MANIFEST.json").is_file():
        pytest.skip("production artifacts not generated yet")


def test_001_frozen_config_scope_flags_prohibit_expansive_work() -> None:
    assert CONFIG["scope"] == {
        "final_order_model_allowed": False,
        "business_policy_allowed": False,
        "thesis_edit_allowed": False,
        "phase2a_reinterpretation_allowed": False,
    }


def test_002_canonical_assembler_hash_is_frozen() -> None:
    assert core.sha256_file(core.ASSEMBLER) == core.ASSEMBLER_SHA


def test_003_v11_manifest_matches_raw_and_assembler_contract() -> None:
    manifest = json.loads((V11 / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["raw_file_hashes"] == core.EXPECTED_RAW_HASHES
    assert manifest["canonical_assembler"]["sha256"] == core.ASSEMBLER_SHA


def test_004_all_placed_and_canonical_counts_remain_distinct() -> None:
    reconciliation = pd.read_csv(V11 / "CANONICAL_SAMPLE_RECONCILIATION.csv")
    assert len(reconciliation) == 99_441
    assert reconciliation["order_id"].notna().all() and reconciliation["order_id"].is_unique
    assert int(reconciliation["customer_delivery_observed"].sum()) == 96_476
    assert int(reconciliation["in_canonical"].sum()) == 96_470


def test_005_six_observed_noncanonical_orders_are_audit_only() -> None:
    reconciliation = pd.read_csv(V11 / "CANONICAL_SAMPLE_RECONCILIATION.csv")
    six = reconciliation.loc[reconciliation["customer_delivery_observed"] & ~reconciliation["in_canonical"]]
    assert len(six) == 6
    assert six["order_status"].eq("canceled").all()
    assert six["deterministic_reconciliation_reason"].str.startswith("status_inconsistency:").all()


def test_006_negative_process_counts_match_v11_audit() -> None:
    quality = pd.read_csv(V11 / "DATA_QUALITY_AUDIT_V1_1.csv")
    lookup = quality.set_index(["scope", "check"])["count"]
    assert lookup[("all_placed", "negative_handling")] == 1_359
    assert lookup[("canonical_delivered", "negative_handling")] == 1_350
    assert lookup[("all_placed", "negative_transit")] == 23
    assert lookup[("canonical_delivered", "negative_transit")] == 23


def test_007_protected_paths_do_not_include_new_workspace() -> None:
    assert set(core.PROTECTED) == {
        "dynamic_profile_eda_v1", "dynamic_profile_eda_v1_1",
        "profile_pivot_phase2a", "profile_pivot_phase1_audit",
        "docs_thesis", "report_thesis", "results", "src",
    }
    assert all(not path.is_relative_to(OUT) for path in core.PROTECTED.values())
    assert set(core.control_file_hashes()) == {
        "AGENTS.md", "PROJECT_CONTEXT.md", "RESULTS_REGISTRY.md", "DECISION_LOG.md",
    }


def test_008_recursive_protected_hashes_are_deterministic() -> None:
    first = {name: core.recursive_hashes(path) for name, path in core.PROTECTED.items()}
    second = {name: core.recursive_hashes(path) for name, path in core.PROTECTED.items()}
    assert first == second
    assert all(first[name] for name in ("dynamic_profile_eda_v1", "dynamic_profile_eda_v1_1", "profile_pivot_phase2a"))


def test_009_protocol_contains_required_stage_and_scope_language() -> None:
    text = (OUT / "PROFILE_PROTOCOL.md").read_text(encoding="utf-8")
    for phrase in (
        "strict `label_available_at < snapshot_date`",
        "Development and confirmation execute in separate processes",
        "does not invoke a final order-level model or business-policy routine",
    ):
        assert phrase in text


def test_010_no_forbidden_final_model_family_is_present() -> None:
    source = Path(core.__file__).read_text(encoding="utf-8")
    for token in ("RandomForest", "XGBClassifier", "XGBRegressor", "LightGBM", "CatBoost", "MLPClassifier"):
        assert token not in source


def test_011_hrd_is_frozen_as_nonpredictive() -> None:
    assert CONFIG["hrd"]["predictor_allowed"] is False
    assert CONFIG["hrd"]["primary_definition"] == "both_top10"
    assert len(CONFIG["hrd"]["all_definitions"]) == 6


def test_012_p2_feature_allowlist_excludes_every_forbidden_field() -> None:
    allowed = set(CONFIG["p2"]["features_numeric"]) | set(CONFIG["p2"]["features_categorical"])
    forbidden = set(CONFIG["p2"]["forbidden_features"])
    assert allowed.isdisjoint(forbidden)
    assert all(name.startswith("b0__") for name in allowed)


def test_013_profile_base_schema_is_exact_and_unique() -> None:
    assert len(core.PROFILE_BASE_COLUMNS) == len(set(core.PROFILE_BASE_COLUMNS))
    assert core.PROFILE_BASE_COLUMNS[:6] == [
        "entity_id", "snapshot_date", "target", "granularity", "scheme", "window_days",
    ]
    assert {"score", "raw_score", "support", "parent_score", "posterior_se", "invalid_reason"}.issubset(core.PROFILE_BASE_COLUMNS)


def test_014_daily_gzip_determinism_is_frozen(tmp_path: Path) -> None:
    from analysis.dynamic_profile_profile_validation_v1.scripts import (
        profile_reporting,
        run_profile_validation,
        selected_daily,
    )

    assert CONFIG["daily_storage"]["row_artifact"] == "PROFILE_DAILY_SCORES.csv.gz"
    assert CONFIG["daily_storage"]["index_artifact"] == "PROFILE_DAILY_SCORES.csv"
    assert CONFIG["daily_storage"]["gzip_mtime"] == 0
    assert CONFIG["determinism"]["float_format"] == "%.12g"
    assert (
        profile_reporting.RAW_SELECTED_DAILY_ROW_SCHEMA
        == selected_daily.SELECTED_DAILY_COLUMNS
    )
    assert (
        profile_reporting.RICH_ORDER_SCORING_SCHEMA
        == tuple(run_profile_validation.SCORING_WORK_COLUMNS)
    )

    base_candidate_id = (
        "final_breach|seller_id|A|w90|l30|P0|parent=global|kappa=na"
    )
    candidate_id = f"{base_candidate_id}|min_support=5"
    profile_spec_id = run_profile_validation.profile_spec_id(base_candidate_id)
    score_30d = 3.1862233972223803
    score_90d = 3.6094270293301634
    raw_row = {column: np.nan for column in selected_daily.SELECTED_DAILY_COLUMNS}
    raw_row.update(
        {
            "entity_id": "seller_1",
            "snapshot_date": pd.Timestamp("2017-04-01"),
            "target": "final_breach",
            "granularity": "seller_id",
            "scheme": "A",
            "window_days": 90,
            "lag_days": 30,
            "estimator": "P0",
            "parent_structure": "global",
            "base_candidate_id": base_candidate_id,
            "parent_id": "__GLOBAL__",
            "score": 0.8,
            "raw_score": 0.8,
            "support": 10,
            "event_count": 8,
            "parent_score": 0.1,
            "global_score": 0.1,
            "posterior_se": 0.01,
            "lower_interval": 0.7,
            "upper_interval": 0.9,
            "cold_start": False,
            "profile_freshness_days": 0,
            "active_days": 10,
            "last_mature_outcome_date": pd.Timestamp("2017-03-31"),
            "invalid_reason": "",
            "candidate_id": candidate_id,
            "profile_spec_id": profile_spec_id,
            "min_support": 5,
            "q33": 0.2,
            "q67": 0.5,
            "level": "High",
            "unknown_reason": "",
            "period": "development",
            "score_30d": score_30d,
            "support_30d": 10,
            "score_90d": score_90d,
            "support_90d": 10,
            "short_long_trend": score_30d - score_90d,
        }
    )
    raw_daily = pd.DataFrame(
        [raw_row], columns=selected_daily.SELECTED_DAILY_COLUMNS,
    )
    part_path = tmp_path / "selected_daily_part.csv"
    gzip_path = tmp_path / "SELECTED_DAILY_PROFILE_ROWS.csv.gz"
    second_gzip_path = tmp_path / "SELECTED_DAILY_PROFILE_ROWS.second.csv.gz"
    run_profile_validation.write_csv(
        raw_daily, part_path, ["candidate_id", "snapshot_date", "entity_id"],
    )
    run_profile_validation.concatenate_csv_files_to_deterministic_gzip(
        [part_path], gzip_path, selected_daily.SELECTED_DAILY_COLUMNS,
    )
    run_profile_validation.concatenate_csv_files_to_deterministic_gzip(
        [part_path], second_gzip_path, selected_daily.SELECTED_DAILY_COLUMNS,
    )
    assert gzip_path.read_bytes() == second_gzip_path.read_bytes()
    gzip_header = gzip_path.read_bytes()[:10]
    assert gzip_header[:2] == b"\x1f\x8b"
    assert struct.unpack("<I", gzip_header[4:8])[0] == 0

    discovered = profile_reporting.discover_reporting_inputs(tmp_path, tmp_path)
    round_trip = discovered.daily_profiles
    assert round_trip is not None
    assert tuple(round_trip.columns) == selected_daily.SELECTED_DAILY_COLUMNS
    residual = abs(
        float(round_trip.loc[0, "short_long_trend"])
        - (
            float(round_trip.loc[0, "score_30d"])
            - float(round_trip.loc[0, "score_90d"])
        )
    )
    assert residual > 1e-12
    assert residual <= profile_reporting.DAILY_TREND_SERIALIZATION_ATOL

    selected = profile_reporting.normalize_selected_candidates(
        pd.DataFrame(
            [
                {
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_candidate_id,
                    "profile_spec_id": profile_spec_id,
                    "target": "final_breach",
                    "granularity": "seller_id",
                    "scheme": "A",
                    "window_days": 90,
                    "lag_days": 30,
                    "estimator": "P0",
                    "parent_structure": "global",
                    "kappa": np.nan,
                    "min_support": 5,
                    "low_medium_cutoff": 0.2,
                    "medium_high_cutoff": 0.5,
                }
            ]
        )
    )
    prepared = profile_reporting.prepare_daily_profiles(round_trip, selected)
    expected_trend = float(round_trip.loc[0, "score_30d"]) - float(
        round_trip.loc[0, "score_90d"]
    )
    assert tuple(prepared.columns) == profile_reporting.DAILY_ROW_SCHEMA
    assert float(prepared.loc[0, "short_long_trend"]) == expected_trend
    assert float(prepared.loc[0, "low_medium_cutoff"]) == 0.2
    assert float(prepared.loc[0, "medium_high_cutoff"]) == 0.5

    corrupt = round_trip.copy()
    corrupt.loc[0, "short_long_trend"] = (
        float(corrupt.loc[0, "short_long_trend"]) + 1e-6
    )
    with pytest.raises(ValueError, match="short_long_trend"):
        profile_reporting.prepare_daily_profiles(corrupt, selected)

    augmented = profile_reporting._augment_terminal_followup(
        pd.DataFrame(
            {
                "candidate_id": ["c1"],
                "period": ["terminal"],
                "anchor_date": ["2018-07-07"],
                "horizon_days": ["7"],
            }
        ),
        pd.DataFrame(
            {
                "candidate_id": ["c1"],
                "period": ["terminal"],
                "anchor_date": [pd.Timestamp("2018-07-07")],
                "horizon_days": [7],
                "label_available_at": [pd.Timestamp("2018-07-20")],
            }
        ),
        {
            "audit_endpoint_proxy": "2018-10-17 13:22:46",
            "terminal_maturity_availability_threshold": 0.95,
        },
    )
    assert augmented.loc[0, "outcome_observation_rate"] == 1.0
    assert not bool(augmented.loc[0, "maturity_censoring_flag"])


def test_015_weekly_anchors_are_left_closed_and_seven_days_apart() -> None:
    anchors = core.weekly_anchors("2017-04-01", "2017-05-01")
    assert anchors[0] == pd.Timestamp("2017-04-01")
    assert anchors[-1] == pd.Timestamp("2017-04-29")
    assert np.all(np.diff(anchors.values).astype("timedelta64[D]").astype(int) == 7)


@pytest.mark.parametrize(
    ("period", "horizon", "expected"),
    [("development", 7, 39), ("development", 30, 36),
     ("confirmation", 7, 25), ("confirmation", 30, 21),
     ("terminal", 7, 7), ("terminal", 30, 4)],
)
def test_016_anchor_schedule_has_exact_frozen_counts(period: str, horizon: int, expected: int) -> None:
    schedule = core.anchor_schedule(CONFIG)
    actual = len(schedule.loc[(schedule["period"] == period) & (schedule["horizon_days"] == horizon)])
    assert actual == expected


def test_017_anchor_cadence_is_global_not_restarted_at_phase_boundary() -> None:
    schedule = core.anchor_schedule(CONFIG)
    confirmation = schedule.loc[schedule["period"].eq("confirmation"), "anchor_date"]
    terminal = schedule.loc[schedule["period"].eq("terminal"), "anchor_date"]
    assert confirmation.min() == pd.Timestamp("2018-01-06")
    assert terminal.min() == pd.Timestamp("2018-07-07")


def test_018_every_evaluation_horizon_is_fully_contained() -> None:
    schedule = core.anchor_schedule(CONFIG)
    assert schedule["full_phase_containment"].all()
    phase_ends = {
        "development": pd.Timestamp("2018-01-01"),
        "confirmation": pd.Timestamp("2018-07-01"),
        "terminal": pd.Timestamp("2018-08-31"),
    }
    for row in schedule.itertuples(index=False):
        assert row.future_start == row.anchor_date
        assert row.future_end_exclusive <= phase_ends[row.period]


@pytest.mark.parametrize("horizon", [7, 30])
def test_019_future_purchase_cohort_has_exact_half_open_boundaries(horizon: int) -> None:
    snapshot = pd.Timestamp("2018-02-01")
    epsilon = pd.Timedelta(microseconds=1)
    frame = pd.DataFrame({
        "order_id": ["left", "inside", "right"],
        "order_purchase_timestamp": [snapshot, snapshot + pd.Timedelta(days=horizon) - epsilon, snapshot + pd.Timedelta(days=horizon)],
    })
    result = core.future_cohort(frame, snapshot, horizon)
    assert set(result["order_id"]) == {"left", "inside"}


def test_020_development_view_excludes_locked_purchase_rows() -> None:
    frame = pd.DataFrame({
        "order_id": ["dev", "equal", "later"],
        "order_purchase_timestamp": pd.to_datetime(["2017-12-31", "2018-01-01", "2018-02-01"]),
        "late_delivery": [0.0, 1.0, 1.0],
    })
    result = core.mask_locked_outcomes_for_development(frame)
    assert set(result["order_id"]) == {"dev"}
    assert result.attrs["locked_outcomes_masked"] is True


def test_021_confirmation_label_changes_cannot_change_development_view() -> None:
    base = pd.DataFrame({
        "order_id": ["dev", "confirmation"],
        "order_purchase_timestamp": pd.to_datetime(["2017-12-01", "2018-01-01"]),
        "late_delivery": [0.0, 0.0],
        "final_breach_available_at": pd.to_datetime(["2017-12-10", "2018-01-10"]),
    })
    poisoned = base.copy()
    poisoned.loc[poisoned["order_id"].eq("confirmation"), "late_delivery"] = 1.0
    pd.testing.assert_frame_equal(
        core.mask_locked_outcomes_for_development(base),
        core.mask_locked_outcomes_for_development(poisoned),
    )


def test_022_strict_asof_equality_is_excluded() -> None:
    snapshot = pd.Timestamp("2017-06-01")
    frame = _history_boundary_frame(snapshot, 30)
    result = core.history_slice(frame, _source(), snapshot)
    assert "availability_equal" not in set(result["order_id"])


def test_023_future_available_outcome_cannot_enter_profile() -> None:
    snapshot = pd.Timestamp("2017-06-01")
    frame = _history_boundary_frame(snapshot, 30)
    frame.loc[frame["order_id"].eq("left_in"), "final_breach_available_at"] = snapshot + pd.Timedelta(days=1)
    result = core.history_slice(frame, _source(), snapshot)
    assert "left_in" not in set(result["order_id"])


@pytest.mark.parametrize("window", [30, 60, 90])
def test_024_scheme_a_exact_windows(window: int) -> None:
    snapshot = pd.Timestamp("2017-06-01")
    epsilon = pd.Timedelta(microseconds=1)
    left = snapshot - pd.Timedelta(days=window)
    frame = pd.DataFrame({
        "order_id": ["left", "left_out", "right", "right_out"],
        "order_purchase_timestamp": [snapshot - pd.Timedelta(days=100)] * 4,
        "final_breach_available_at": [left, left - epsilon, snapshot - epsilon, snapshot],
        "late_delivery": [1.0, 0.0, 1.0, 0.0], "in_canonical": True,
        "seller_id": "s1", "main_seller_state": "SP", "region_od": "Southeast -> Southeast",
    })
    result = core.history_slice(frame, _source(window_days=window), snapshot)
    assert set(result["order_id"]) == {"left", "right"}


@pytest.mark.parametrize("lag", [14, 21])
def test_025_handling_scheme_c_exact_lags(lag: int) -> None:
    snapshot = pd.Timestamp("2017-06-01")
    frame = _history_boundary_frame(snapshot, 30, lag).rename(columns={
        "final_breach_available_at": "handling_available_at",
        "late_delivery": "handling_level_value",
    })
    frame["handling_duration"] = frame["handling_level_value"]
    result = core.history_slice(
        frame, _source("handling_level", scheme="C", lag_days=lag), snapshot,
    )
    assert set(result["order_id"]) == {"left_in", "right_in"}


@pytest.mark.parametrize("lag", [30, 45])
def test_026_nonhandling_scheme_c_exact_lags(lag: int) -> None:
    snapshot = pd.Timestamp("2017-06-01")
    frame = _history_boundary_frame(snapshot, 60, lag)
    result = core.history_slice(
        frame, _source(scheme="C", window_days=60, lag_days=lag), snapshot,
    )
    assert set(result["order_id"]) == {"left_in", "right_in"}


def test_027_scheme_b_or_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported scheme"):
        core.history_slice(_history_boundary_frame(pd.Timestamp("2017-06-01"), 30), _source(scheme="B"), pd.Timestamp("2017-06-01"))


def test_028_candidate_grid_contains_only_frozen_sources() -> None:
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_selection

    sources = core.candidate_sources()
    assert len(sources) == 108
    assert {row["scheme"] for row in sources} == {"A", "C"}
    assert {row["window_days"] for row in sources} == {30, 60, 90}
    assert all(row["lag_days"] == 0 for row in sources if row["scheme"] == "A")

    catalog_row = profile_selection.build_candidate_catalog(CONFIG).iloc[0]
    anchors = pd.DataFrame(
        {
            **{
                column: [catalog_row[column]] * 39
                for column in profile_selection.CATALOG_COLUMNS
            },
            "anchor_date": pd.date_range("2017-04-01", periods=39, freq="7D"),
            "valid": [True] * 39,
            "delta_logloss": [0.01] * 39,
            "delta_brier": [0.001] * 39,
        }
    )
    aggregate, _ = profile_selection.aggregate_anchor_metrics(anchors)
    expected = profile_selection.stable_profile_spec_id(
        str(catalog_row["base_candidate_id"])
    )
    assert aggregate.loc[0, "profile_spec_id"] == expected


def test_029_history_source_is_canonical_only() -> None:
    snapshot = pd.Timestamp("2017-06-01")
    frame = _history_boundary_frame(snapshot, 30)
    frame.loc[frame["order_id"].eq("left_in"), "in_canonical"] = False
    result = core.history_slice(frame, _source(), snapshot)
    assert "left_in" not in set(result["order_id"])


def test_030_positive_lateness_severity_excludes_zero() -> None:
    frame = pd.DataFrame({
        "positive_late_days_available_at": pd.to_datetime(["2017-01-02", "2017-01-02", pd.NaT]),
        "positive_late_severity_value": [0.0, math.log(3.0), np.nan],
        "positive_late_days": [0.0, 2.0, np.nan],
    })
    assert core.target_valid_mask(frame, "positive_late_severity").tolist() == [False, True, False]


def test_031_negative_process_duration_is_invalid_without_clipping() -> None:
    frame = pd.DataFrame({
        "handling_available_at": pd.to_datetime(["2017-01-02", "2017-01-02"]),
        "handling_level_value": [np.nan, math.log(2.0)],
        "handling_duration": [-1.0, 1.0],
    })
    assert core.target_valid_mask(frame, "handling_level").tolist() == [False, True]
    assert frame.loc[0, "handling_duration"] == -1.0


def test_032_tail_threshold_uses_only_strict_predevelopment_canonical_nonnegative_rows() -> None:
    cutoff = pd.Timestamp("2017-04-01")
    frame = pd.DataFrame({
        "in_canonical": [True] * 7 + [False],
        "handling_available_at": pd.to_datetime(["2017-03-01"] * 6 + [cutoff, "2017-03-01"]),
        "transit_available_at": pd.to_datetime(["2017-03-01"] * 6 + [cutoff, "2017-03-01"]),
        "handling_duration": [0.0, 1.0, 2.0, 3.0, 100.0, -5.0, 999.0, 888.0],
        "transit_duration": [0.0, 2.0, 4.0, 6.0, 200.0, -4.0, 999.0, 888.0],
    })
    thresholds = core.frozen_tail_thresholds(frame, cutoff)
    assert thresholds["handling_tail_threshold_days"] == pytest.approx(np.quantile([0, 1, 2, 3, 100], 0.9, method="linear"))
    assert thresholds["transit_tail_threshold_days"] == pytest.approx(np.quantile([0, 2, 4, 6, 200], 0.9, method="linear"))


def test_033_tail_event_operator_is_strict_greater_than() -> None:
    frame = pd.DataFrame({"handling_duration": [1.9, 2.0, 2.1], "transit_duration": [2.9, 3.0, 3.1]})
    result = core.attach_tail_targets(frame, {"handling_tail_threshold_days": 2.0, "transit_tail_threshold_days": 3.0})
    assert result["handling_tail"].tolist() == [False, False, True]
    assert result["transit_tail"].tolist() == [False, False, True]


def test_034_negative_tail_values_remain_missing_not_zero_events() -> None:
    frame = pd.DataFrame({"handling_duration": [-1.0], "transit_duration": [-0.1]})
    result = core.attach_tail_targets(frame, {"handling_tail_threshold_days": 2.0, "transit_tail_threshold_days": 3.0})
    assert pd.isna(result.loc[0, "handling_tail"])
    assert pd.isna(result.loc[0, "transit_tail"])


def test_035_tail_thresholds_are_order_deterministic() -> None:
    frame = pd.DataFrame({
        "in_canonical": True,
        "handling_available_at": pd.to_datetime(["2017-01-01"] * 10),
        "transit_available_at": pd.to_datetime(["2017-01-01"] * 10),
        "handling_duration": np.arange(10, dtype=float),
        "transit_duration": np.arange(10, dtype=float) * 2,
    })
    first = core.frozen_tail_thresholds(frame)
    second = core.frozen_tail_thresholds(frame.sample(frac=1, random_state=42).reset_index(drop=True))
    assert first == second


def test_036_confirmation_rows_cannot_recompute_predevelopment_tail() -> None:
    base = pd.DataFrame({
        "in_canonical": [True, True],
        "handling_available_at": pd.to_datetime(["2017-01-01", "2018-02-01"]),
        "transit_available_at": pd.to_datetime(["2017-01-01", "2018-02-01"]),
        "handling_duration": [1.0, 999.0], "transit_duration": [2.0, 999.0],
    })
    assert core.frozen_tail_thresholds(base) == core.frozen_tail_thresholds(base.iloc[[0]])


def test_037_candidate_variants_use_only_frozen_kappas() -> None:
    variants = core.candidate_variants(_source())
    assert {row["estimator"] for row in variants} == {"P0", "P1", "P2"}
    assert {row["kappa"] for row in variants if row["estimator"] in {"P1", "P2"}} == {10, 20, 50, 100}
    assert len({row["base_candidate_id"] for row in variants}) == len(variants)


def test_038_p0_binary_is_raw_event_rate() -> None:
    history = _binary_history()
    variant = {"estimator": "P0", "parent_structure": "global", "kappa": None, "base_candidate_id": "p0"}
    profile, _ = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    s1 = profile.set_index("entity_id").loc["s1"]
    assert s1["support"] == 3
    assert s1["event_count"] == 2
    assert s1["score"] == pytest.approx(2 / 3)
    assert s1["raw_score"] == pytest.approx(2 / 3)


def test_039_p1_binary_matches_beta_binomial_formula() -> None:
    history = _binary_history()
    variant = {"estimator": "P1", "parent_structure": "global", "kappa": 10, "base_candidate_id": "p1"}
    profile, _ = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    s1 = profile.set_index("entity_id").loc["s1"]
    global_rate = 0.5
    assert s1["score"] == pytest.approx((2 + 10 * global_rate) / (3 + 10))


def test_040_p1_binary_posterior_uncertainty_is_finite_and_ordered() -> None:
    history = _binary_history()
    variant = {"estimator": "P1", "parent_structure": "global", "kappa": 20, "base_candidate_id": "p1"}
    profile, _ = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    assert np.isfinite(profile["posterior_se"]).all()
    assert (profile["lower_interval"] <= profile["score"]).all()
    assert (profile["score"] <= profile["upper_interval"]).all()


def test_041_low_support_structural_parent_falls_back_to_global() -> None:
    history = _binary_history()
    variant = {"estimator": "P1", "parent_structure": "seller_state", "kappa": 10, "base_candidate_id": "p1-state"}
    profile, _ = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    assert np.allclose(profile["parent_score"], 0.5)


def test_042_supported_seller_state_parent_is_applied() -> None:
    history = _supported_seller_parent_history()
    variant = {"estimator": "P1", "parent_structure": "seller_state", "kappa": 10, "base_candidate_id": "p1-state"}
    profile, parents = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    s1 = profile.set_index("entity_id").loc["s1"]
    sp_parent = parents.set_index("parent_id").loc["SP"]
    assert sp_parent["parent_support"] == 20
    assert sp_parent["parent_score"] == pytest.approx(0.5)
    assert s1["parent_id"] == "SP"
    assert s1["score"] == pytest.approx((3 + 10 * 0.5) / (4 + 10))


def test_042b_supported_pure_parent_is_shrunk_not_discarded() -> None:
    rows = []
    for index in range(40):
        state = "SP" if index < 20 else "RJ"
        rows.append({
            "order_id": f"pure{index:02d}",
            "order_purchase_timestamp": pd.Timestamp("2017-04-01") + pd.Timedelta(days=index),
            "final_breach_available_at": pd.Timestamp("2017-05-15"),
            "late_delivery": float(index < 20),
            "seller_id": "s_sp" if index < 20 else "s_rj",
            "main_seller_state": state,
            "state_od": f"{state} -> MG",
            "region_od": "Southeast -> Southeast",
        })
    history = pd.DataFrame(rows)
    variant = {
        "estimator": "P1", "parent_structure": "seller_state", "kappa": 10,
        "base_candidate_id": "pure-parent",
    }
    profile, parents = core._binary_profile_variant(
        history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG,
    )
    sp_parent = parents.set_index("parent_id").loc["SP", "parent_score"]
    assert sp_parent == pytest.approx((20 + 10 * 0.5) / 30)
    assert profile.set_index("entity_id").loc["s_sp", "parent_score"] == pytest.approx(sp_parent)


def test_043_state_od_uses_region_od_parent_mapping() -> None:
    history = _supported_seller_parent_history().copy()
    history["state_od"] = np.where(history["main_seller_state"].eq("SP"), "SP -> MG", "RJ -> MG")
    history["region_od"] = "Southeast -> Southeast"
    source = _source(granularity="state_od")
    variant = {"estimator": "P1", "parent_structure": "region_od", "kappa": 10, "base_candidate_id": "route"}
    profile, parents = core._binary_profile_variant(history, source, pd.Timestamp("2017-06-01"), variant, CONFIG)
    assert set(profile["parent_id"]) == {"Southeast -> Southeast"}
    assert set(parents["parent_id"]) == {"Southeast -> Southeast"}


def test_043b_region_od_entity_key_is_not_duplicated_in_history_slice() -> None:
    history = _binary_history().copy()
    history["in_canonical"] = True
    source = _source(granularity="region_od")
    base_id = core.base_candidate_id(source, "P0", "global", None)
    profile, _ = core.build_profiles(
        history, source, pd.Timestamp("2017-06-01"), CONFIG,
        allowed_base_ids={base_id},
    )
    assert not profile.empty
    assert profile["entity_id"].is_unique


def test_044_invalid_parent_structure_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid parent"):
        core._parent_column(_binary_history(), "seller_id", "region_od")


def test_045_p0_continuous_is_raw_mean_on_model_scale() -> None:
    history = _continuous_history()
    source = _source("handling_level")
    variant = {"estimator": "P0", "parent_structure": "global", "kappa": None, "base_candidate_id": "p0c"}
    profile, _ = core._continuous_profile_variant(history, source, pd.Timestamp("2017-06-01"), variant, CONFIG)
    assert profile.set_index("entity_id").loc["s1", "score"] == pytest.approx(1.0)


def test_046_p1_continuous_matches_frozen_normal_normal_formula() -> None:
    history = _continuous_history()
    source = _source("handling_level")
    variant = {"estimator": "P1", "parent_structure": "global", "kappa": None, "base_candidate_id": "p1c"}
    profile, _ = core._continuous_profile_variant(history, source, pd.Timestamp("2017-06-01"), variant, CONFIG)
    s1 = profile.set_index("entity_id").loc["s1"]
    assert s1["within_variance"] == pytest.approx(2.0)
    assert s1["between_variance"] == pytest.approx(3.0)
    assert s1["score"] == pytest.approx(1.5)
    assert s1["posterior_se"] == pytest.approx(math.sqrt(0.75))


def test_047_degenerate_continuous_variance_uses_explicit_parent_fallback() -> None:
    history = _continuous_history(degenerate=True)
    source = _source("handling_level")
    variant = {"estimator": "P1", "parent_structure": "global", "kappa": None, "base_candidate_id": "p1c"}
    profile, _ = core._continuous_profile_variant(history, source, pd.Timestamp("2017-06-01"), variant, CONFIG)
    assert profile["invalid_reason"].eq("degenerate_variance_parent_fallback").all()
    assert np.allclose(profile["score"], profile["parent_score"])


def test_048_group_logistic_offset_zero_gradient_stays_zero() -> None:
    theta, hessian = core._group_logistic_offsets(
        np.array([0.0, 1.0]), np.array([0.2, 0.8]), np.array([0, 0]), 1, 2.5,
    )
    assert theta[0] == pytest.approx(0.0, abs=1e-12)
    assert hessian[0] > 2.5


def test_049_p2_binary_persists_expected_rate_and_oe() -> None:
    history = _binary_history().iloc[:2].copy()
    history["seller_id"] = "s1"
    history["late_delivery"] = [0.0, 1.0]
    history["expected_final_breach"] = [0.2, 0.8]
    variant = {"estimator": "P2", "parent_structure": "global", "kappa": 10, "base_candidate_id": "p2"}
    profile, _ = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    row = profile.iloc[0]
    assert row["expected_rate"] == pytest.approx(0.5)
    assert row["observed_expected_ratio"] == pytest.approx(1.0)
    assert row["score"] == pytest.approx(0.5)


def test_050_p2_binary_regularisation_avoids_raw_zero_or_one_scores() -> None:
    history = _binary_history().copy()
    history["seller_id"] = ["high", "high", "low", "low"]
    history["late_delivery"] = [1.0, 1.0, 0.0, 0.0]
    history["expected_final_breach"] = 0.5
    variant = {"estimator": "P2", "parent_structure": "global", "kappa": 10, "base_candidate_id": "p2"}
    profile, _ = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    indexed = profile.set_index("entity_id")
    assert 0.5 < indexed.loc["high", "score"] < 1.0
    assert 0.0 < indexed.loc["low", "score"] < 0.5


def test_051_p2_requires_expected_values_and_does_not_fallback_to_p1() -> None:
    history = _binary_history().copy()
    history["expected_final_breach"] = np.nan
    variant = {"estimator": "P2", "parent_structure": "global", "kappa": 10, "base_candidate_id": "p2"}
    profile, parents = core._binary_profile_variant(history, _source(), pd.Timestamp("2017-06-01"), variant, CONFIG)
    assert profile.empty and parents.empty


def test_052_base_candidate_id_is_deterministic_and_parameter_complete() -> None:
    source = _source(scheme="C", window_days=90, lag_days=45)
    first = core.base_candidate_id(source, "P1", "seller_state", 50)
    second = core.base_candidate_id(dict(reversed(list(source.items()))), "P1", "seller_state", 50)
    assert first == second
    assert first == "final_breach|seller_id|C|w90|l45|P1|parent=seller_state|kappa=50"


def test_053_build_profiles_has_exact_schema_and_unique_candidate_entity_rows() -> None:
    history = _binary_history().copy()
    history["in_canonical"] = True
    source = _source()
    profiles, _ = core.build_profiles(history, source, pd.Timestamp("2017-06-01"), CONFIG)
    assert list(profiles.columns) == core.PROFILE_BASE_COLUMNS
    assert not profiles.duplicated(["base_candidate_id", "entity_id"]).any()


def test_054_cold_start_maps_to_parent_not_zero() -> None:
    future, profile, parents, source, base_id = _future_mapping_fixture()
    mapped = core.map_future_orders(future, profile, parents, source, base_id).set_index("order_id")
    assert mapped.loc["cold", "mapping_status"] == "mapped_cold_start"
    assert bool(mapped.loc["cold", "cold_start"])
    assert mapped.loc["cold", "history_support"] == 0
    assert mapped.loc["cold", "profile_score"] == pytest.approx(0.20)


def test_055_missing_mapping_is_not_cold_start() -> None:
    future, profile, parents, source, base_id = _future_mapping_fixture()
    mapped = core.map_future_orders(future, profile, parents, source, base_id).set_index("order_id")
    assert mapped.loc["missing", "mapping_status"] == "missing_mapping"
    assert not bool(mapped.loc["missing", "cold_start"])


def test_056_seen_future_entity_retains_profile_support_and_score() -> None:
    future, profile, parents, source, base_id = _future_mapping_fixture()
    mapped = core.map_future_orders(future, profile, parents, source, base_id).set_index("order_id")
    assert mapped.loc["seen", "mapping_status"] == "seen"
    assert mapped.loc["seen", "history_support"] == 8
    assert mapped.loc["seen", "profile_score"] == pytest.approx(0.8)


def test_057_all_placed_unresolved_order_is_exposure_but_not_primary_label() -> None:
    future, profile, parents, source, base_id = _future_mapping_fixture()
    mapped = core.map_future_orders(future, profile, parents, source, base_id).set_index("order_id")
    assert "unresolved" in mapped.index
    assert not bool(mapped.loc["unresolved", "target_observed"])
    assert not bool(mapped.loc["unresolved", "eligible_for_metric"])


def test_058_mapping_statuses_partition_all_future_orders() -> None:
    future, profile, parents, source, base_id = _future_mapping_fixture()
    mapped = core.map_future_orders(future, profile, parents, source, base_id)
    counts = mapped["mapping_status"].value_counts()
    assert counts.sum() == len(future)
    assert set(counts.index) == {"seen", "mapped_cold_start", "missing_mapping"}


def test_059_support_strata_keep_cold_start_and_missing_mapping_separate() -> None:
    future, profile, parents, source, base_id = _future_mapping_fixture()
    mapped = core.map_future_orders(future, profile, parents, source, base_id)
    local = copy.deepcopy(CONFIG)
    local["validity"]["minimum_future_orders_for_primary_score"] = 1
    local["validity"]["minimum_binary_class_count_per_anchor"] = 1
    _, _, strata = core.evaluate_mapped_orders(mapped, 5, local)
    assert {"support_0_cold_start", "missing_mapping", "support_5_9"}.issubset(set(strata["support_stratum"]))


def test_060_weighted_spearman_matches_perfect_monotone_ranking() -> None:
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_reporting

    assert core.weighted_spearman([1, 2, 3], [10, 20, 30], [1, 5, 2]) == pytest.approx(1.0)
    assert core.weighted_spearman([1, 2, 3], [30, 20, 10], [1, 5, 2]) == pytest.approx(-1.0)

    # Frozen old-reference fixture for the optimized entity bootstrap.  It
    # exercises average-rank ties, lexical IDs (e1/e10/e2), repeated bootstrap
    # entities, the continuous original-day lift, a missing raw outcome and a
    # zero-support row.
    entity = pd.DataFrame(
        {
            "entity_id": [
                "e1", "e10", "e2", "e3", "e4", "e5",
                "e6", "e7", "e8", "e9", "e11", "e12",
            ],
            "profile_score": [
                0.1, 0.1, 0.2, 0.2, 0.3, 0.3,
                0.4, 0.5, 0.5, 0.6, 0.7, 0.7,
            ],
            "future_mean": [
                0.05, 0.10, 0.10, 0.20, 0.25, 0.30,
                0.45, 0.50, 0.55, 0.65, 0.80, 0.90,
            ],
            "future_raw_mean": [
                1.0, np.nan, 1.5, 2.0, 2.5, 3.0,
                3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
            ],
            "future_support": [1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 2, 1],
        }
    )
    identity = ("candidate_continuous", "development", "2017-04-01", 7)
    expected = (
        0.9319892157597032,
        0.9982096428251803,
        1.2105166666666667,
        2.0359280074936437,
    )
    first = profile_reporting._entity_transfer_bootstrap(
        entity, identity, lift_outcome_column="future_raw_mean",
    )
    second = profile_reporting._entity_transfer_bootstrap(
        entity, identity, lift_outcome_column="future_raw_mean",
    )
    assert first == second
    assert first == pytest.approx(expected, rel=0, abs=1e-12)


def test_061_rank_metric_is_invalid_below_ten_common_entities() -> None:
    mapped = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(9)], "entity_id": [f"e{i}" for i in range(9)],
        "profile_score": np.linspace(0.1, 0.9, 9), "parent_score": 0.5,
        "global_score": 0.5,
        "raw_score": np.linspace(0.1, 0.9, 9), "history_support": 10,
        "mapping_status": "seen", "eligible_for_metric": True,
        "target_value": [0, 1] * 4 + [0], "raw_target_value": [0, 1] * 4 + [0],
        "target": "final_breach", "granularity": "seller_id", "base_candidate_id": "b",
    })
    local = copy.deepcopy(CONFIG)
    local["validity"]["minimum_future_orders_for_primary_score"] = 1
    local["validity"]["minimum_binary_class_count_per_anchor"] = 1
    row, entity, _ = core.evaluate_mapped_orders(mapped, 5, local)
    assert row["n_common_entities"] == 9
    assert pd.isna(row["weighted_spearman"])
    assert not entity["rank_valid"].any()


def test_062_stability_is_deterministic_with_lexical_top_ties() -> None:
    entities = [f"e{i:02d}" for i in range(10)]
    previous = pd.DataFrame({"base_candidate_id": "b", "target": "final_breach", "granularity": "seller_id", "entity_id": entities, "score": np.arange(10)})
    current = previous.copy()
    result1 = core.stability_between_profiles(previous, current, pd.Timestamp("2017-05-01"), pd.Timestamp("2017-05-02"))
    result2 = core.stability_between_profiles(previous.sample(frac=1, random_state=2), current.sample(frac=1, random_state=3), pd.Timestamp("2017-05-01"), pd.Timestamp("2017-05-02"))
    pd.testing.assert_frame_equal(result1, result2)
    assert result1.iloc[0]["day_to_day_spearman"] == pytest.approx(1.0)
    assert result1.iloc[0]["top20_jaccard"] == pytest.approx(1.0)


def test_063_weighted_quantile_uses_first_score_reaching_weight_fraction() -> None:
    values = np.array([3.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert core._weighted_quantile(values, weights, 0.33) == 1.0
    assert core._weighted_quantile(values, weights, 0.67) == 3.0


def test_064_nuisance_preprocessor_uses_only_frozen_b0_features() -> None:
    prep = core._nuisance_preprocessor(CONFIG)
    selected = set(prep.transformers[0][2]) | set(prep.transformers[1][2])
    assert selected == set(CONFIG["p2"]["features_numeric"]) | set(CONFIG["p2"]["features_categorical"])
    assert selected.isdisjoint(CONFIG["p2"]["forbidden_features"])


def test_064b_nuisance_feature_view_normalises_nullable_missing_values() -> None:
    frame = _row_origin_fixture().iloc[:220].copy()
    numeric = CONFIG["p2"]["features_numeric"][0]
    categorical = CONFIG["p2"]["features_categorical"][0]
    frame.loc[frame.index[0], numeric] = pd.NA
    frame[categorical] = frame[categorical].astype("string")
    frame.loc[frame.index[1], categorical] = pd.NA
    view = core._nuisance_feature_frame(frame, CONFIG)
    assert view[numeric].dtype == float
    assert view[categorical].dtype == object
    transformed = core._nuisance_preprocessor(CONFIG).fit_transform(view)
    assert transformed.shape[0] == len(frame)


def test_065_row_origin_nuisance_models_are_strictly_prior() -> None:
    frame = _row_origin_fixture()
    enriched, audit = core.generate_row_origin_expectations(frame, CONFIG, "2017-02-01")
    january = audit.loc[(audit["origin"] == "2017-01-01") & audit["status"].eq("model")]
    assert set(january["target"]) == set(core.TARGET_SPECS)
    assert (pd.to_datetime(january["strict_prior_max_availability"]) < pd.Timestamp("2017-01-01")).all()
    scored = enriched["order_purchase_timestamp"].between("2017-01-01", "2017-02-01", inclusive="left")
    assert enriched.loc[scored, "expected_final_breach"].notna().all()


def test_066_insufficient_nuisance_training_is_explicit_and_has_no_fallback_prediction() -> None:
    frame = _row_origin_fixture().iloc[:20].copy()
    enriched, audit = core.generate_row_origin_expectations(frame, CONFIG, "2016-10-01")
    assert audit["status"].eq("invalid_insufficient_strict_prior_training").all()
    expected_columns = [f"expected_{target}" for target in core.TARGET_SPECS]
    assert enriched[expected_columns].isna().all().all()


def test_067_row_origin_outputs_are_deterministic_under_input_reordering() -> None:
    frame = _row_origin_fixture()
    first, first_audit = core.generate_row_origin_expectations(frame, CONFIG, "2017-02-01")
    second, second_audit = core.generate_row_origin_expectations(frame.sample(frac=1, random_state=17), CONFIG, "2017-02-01")
    expected = [f"expected_{target}" for target in core.TARGET_SPECS]
    first = first.set_index("order_id")[expected].sort_index()
    second = second.set_index("order_id")[expected].sort_index()
    pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(
        first_audit.sort_values(["target", "origin"]).reset_index(drop=True),
        second_audit.sort_values(["target", "origin"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_068_stage_guard_rejects_missing_freeze_placeholder(tmp_path: Path) -> None:
    _require_stage_api("verify_selection_freeze")
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        core.verify_selection_freeze(tmp_path / "missing.json", tmp_path / "missing.sha256", CONFIG)


def test_069_stage_guard_rejects_mutated_freeze_placeholder(tmp_path: Path) -> None:
    from analysis.dynamic_profile_profile_validation_v1.scripts import run_profile_validation

    _require_stage_api("write_selection_freeze", "verify_selection_freeze")
    freeze = tmp_path / "PROFILE_SELECTION_FREEZE.json"
    sidecar = tmp_path / "PROFILE_SELECTION_FREEZE.sha256"
    core.write_selection_freeze(freeze, sidecar, {"promoted_candidates": []}, CONFIG)
    freeze.write_text(freeze.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        core.verify_selection_freeze(freeze, sidecar, CONFIG)

    base_id = "final_breach|seller_id|A|w90|l0|P1|parent=global|kappa=10"
    valid = {
        "candidate_id": f"{base_id}|min_support=5",
        "base_candidate_id": base_id,
        "profile_spec_id": run_profile_validation.profile_spec_id(base_id),
    }
    run_profile_validation._validate_promoted_candidate_specs(
        [valid], context="test freeze denied",
    )
    with pytest.raises(RuntimeError, match="missing.*profile_spec_id"):
        run_profile_validation._validate_promoted_candidate_specs(
            [{key: value for key, value in valid.items() if key != "profile_spec_id"}],
            context="test freeze denied",
        )
    with pytest.raises(RuntimeError, match="specification ID mismatch"):
        run_profile_validation._validate_promoted_candidate_specs(
            [{**valid, "profile_spec_id": "ps_wrong"}],
            context="test freeze denied",
        )


def test_070_level_assignment_placeholder() -> None:
    if not hasattr(core, "assign_frozen_levels"):
        pytest.skip("level helper not implemented yet")
    rows = pd.DataFrame({
        "score": [0.1, 0.4, 0.8, 0.5], "support": [10, 10, 10, 0],
        "cold_start": [False, False, False, True],
        "lower_interval": [0.05, 0.35, 0.7, np.nan],
        "upper_interval": [0.2, 0.55, 0.9, np.nan],
    })
    levels = core.assign_frozen_levels(rows, 5, 0.33, 0.67)
    assert levels.tolist() == ["Low", "Medium", "High", "Unknown"]


def test_071_pareto_dominance_placeholder() -> None:
    if not hasattr(core, "pareto_frontier"):
        pytest.skip("Pareto helper not implemented yet")
    candidates = pd.DataFrame({
        "candidate_id": ["dominant", "dominated", "incomparable"],
        "proper": [1.0, 0.9, 1.1], "lift": [1.2, 1.1, 1.0],
        "coverage": [0.9, 0.8, 0.95], "stability": [0.9, 0.8, 0.7],
    })
    result = core.pareto_frontier(
        candidates, maximize=["proper", "lift", "coverage", "stability"], tolerance=1e-12,
    ).set_index("candidate_id")
    assert bool(result.loc["dominant", "is_pareto"])
    assert not bool(result.loc["dominated", "is_pareto"])
    assert bool(result.loc["incomparable", "is_pareto"])


def test_072_artifact_required_output_set() -> None:
    _require_artifacts()
    missing = [name for name in REQUIRED_OUTPUTS if not (OUT / name).is_file()]
    assert not missing


@pytest.mark.parametrize("name", sorted(EXACT_ARTIFACT_SCHEMAS))
def test_073_artifact_exact_csv_schemas(name: str) -> None:
    _require_artifacts()
    actual = tuple(pd.read_csv(OUT / name, nrows=0).columns)
    assert actual == EXACT_ARTIFACT_SCHEMAS[name]


def test_074_artifact_daily_gzip_schema_key_and_mtime() -> None:
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_reporting

    _require_artifacts()
    path = OUT / CONFIG["daily_storage"]["row_artifact"]
    assert path.is_file()
    header = path.read_bytes()[:10]
    assert header[:2] == b"\x1f\x8b"
    assert struct.unpack("<I", header[4:8])[0] == 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        columns = tuple(pd.read_csv(handle, nrows=0).columns)
    assert columns == profile_reporting.DAILY_ROW_SCHEMA


def test_075_artifact_manifest_hashes_and_rows_recompute() -> None:
    _require_artifacts()
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    inventory = manifest["artifact_inventory"]
    for relative, record in inventory.items():
        path = OUT / relative
        assert path.is_file(), relative
        assert core.sha256_file(path) == record["sha256"], relative
        if relative.endswith(".csv") and "rows" in record:
            assert len(pd.read_csv(path, usecols=[0], low_memory=False)) == int(record["rows"])


def test_076_artifact_protected_hashes_are_unchanged() -> None:
    _require_artifacts()
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["protected_hashes_before"] == manifest["protected_hashes_after"]
    assert manifest["protected_hashes_before"] == manifest["protected_hashes_after_tests"]


def test_077_artifact_selection_freeze_sidecar_and_manifest_match() -> None:
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_selection

    _require_artifacts()
    freeze = OUT / CONFIG["stage_gate"]["freeze_file"]
    sidecar = OUT / CONFIG["stage_gate"]["freeze_sha_sidecar"]
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    assert expected == core.sha256_file(freeze)
    assert manifest["selection_freeze_sha256"] == expected
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    promoted = pd.DataFrame(payload["promoted_candidates"])
    selected = pd.read_csv(OUT / "PROFILE_SELECTED_CANDIDATES.csv", low_memory=False)
    if promoted.empty:
        assert selected.empty
        return
    assert promoted["candidate_id"].is_unique
    assert promoted["profile_spec_id"].notna().all()
    expected_specs = promoted["base_candidate_id"].astype(str).map(
        profile_selection.stable_profile_spec_id
    )
    assert promoted["profile_spec_id"].astype(str).eq(expected_specs).all()
    frozen_map = promoted.set_index("candidate_id")["profile_spec_id"].astype(str).to_dict()
    public_map = selected.set_index("candidate_id")["profile_spec_id"].astype(str).to_dict()
    assert public_map == frozen_map


def test_078_artifact_stage_event_order_proves_gate() -> None:
    _require_artifacts()
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    events = manifest["stage_gate"]["events"]
    names = [event["event"] for event in events]
    required = ["freeze_written", "freeze_hashed", "freeze_recorded", "confirmation_labels_opened"]
    assert all(name in names for name in required)
    assert [names.index(name) for name in required] == sorted(names.index(name) for name in required)
    assert manifest["stage_gate"]["development_pid"] != manifest["stage_gate"]["confirmation_pid"]


def test_079_artifact_figure_source_pairs_are_exact_and_hashed() -> None:
    _require_artifacts()
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == {f"{stem}.png" for stem in FIGURE_STEMS}
    for stem in FIGURE_STEMS:
        figure = OUT / "figures" / f"{stem}.png"
        source = OUT / "figure_sources" / f"{stem}.csv"
        record = manifest["figures"][figure.name]
        assert figure.is_file() and source.is_file()
        assert core.sha256_file(figure) == record["sha256"]
        assert core.sha256_file(source) == record["source_sha256"]
        assert len(pd.read_csv(source)) == int(record["source_rows"])


def test_080_artifact_scope_flags_confirm_no_final_ladder_or_policy() -> None:
    _require_artifacts()
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    flags = manifest["scope_flags"]
    assert flags["final_order_model_fitted"] is False
    assert flags["business_policy_simulated"] is False
    assert flags["thesis_modified"] is False
    assert flags["phase2a_reinterpreted"] is False


def test_081_artifact_test_log_has_zero_failures() -> None:
    _require_artifacts()
    text = (OUT / "TEST_RESULTS.txt").read_text(encoding="utf-8")
    assert "RETURN_CODE: 0" in text
    assert "FAILED: 0" in text
    assert "-B" in text
    assert "no:cacheprovider" in text
