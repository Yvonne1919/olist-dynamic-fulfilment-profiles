"""Deterministic tests for RQ1 Speed and Promise-Reliability Review V1.

The numbered tests map one-for-one to the thirty required controls in the
authorisation prompt.  Additional tests cover the centred spline design,
DesignInfo reuse, HC1/Cholesky simulation, common support, and the required
three figure/source pairs.  The final test run is intentionally fail-closed:
there are no conditional skips or xfails.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from analysis.rq1_speed_reliability_review_v1.scripts import rq1_data, rq1_io
from analysis.rq1_speed_reliability_review_v1.scripts import rq1_preflight
from src.experiments.rq1_customer_relevance import (
    ERROR_GROUP_LABELS,
    promise_error_groups,
)


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "RQ1_SPEED_RELIABILITY_FROZEN_CONFIG.json"
STATS_PATH = WORKSPACE / "scripts/rq1_stats.py"
REPORTING_PATH = WORKSPACE / "scripts/rq1_reporting.py"

EXPECTED_SELECTED_DIGEST = (
    "3b89feb2bbb0ca0c985374dfcc6de903726a4d6aa5ad558dacc828d947df0570"
)
EXPECTED_ERROR_COUNTS = {
    "very early: <= -14 days": 41_861,
    "early: -13 to -7 days": 33_883,
    "slightly early: -6 to -1 days": 12_419,
    "on promised date": 1_280,
    "1 day late": 820,
    "2-3 days late": 1_032,
    "4-7 days late": 1_748,
    ">=8 days late": 2_781,
}

EXPECTED_DATA_TABLE_SCHEMAS = {
    "RQ1_SAMPLE_AUDIT.csv": [
        "metric",
        "value",
        "expected",
        "unit",
        "status",
        "definition",
    ],
    "RQ1_DATE_IDENTITY_AUDIT.csv": [
        "order_id",
        "purchase_date",
        "actual_delivery_date",
        "estimated_delivery_date",
        "actual_delivery_days",
        "promised_lead_days",
        "promise_error_days",
        "identity_rhs_days",
        "identity_residual_days",
        "identity_holds",
        "missing_component",
        "negative_actual_duration",
        "negative_promised_lead",
    ],
    "RQ1_REVIEW_COVERAGE.csv": [
        "dimension",
        "actual_duration_group",
        "promise_error_group",
        "purchase_month",
        "review_status",
        "analytical_orders",
        "reviewed_orders",
        "missing_review_orders",
        "review_coverage",
        "mean_actual_delivery_days",
        "median_actual_delivery_days",
        "mean_promised_lead_days",
        "median_promised_lead_days",
        "mean_promise_error_days",
        "median_promise_error_days",
    ],
    "RQ1_DURATION_ERROR_CELL_COUNTS.csv": [
        "actual_duration_group",
        "promise_error_group",
        "analytical_orders",
        "reviewed_orders",
        "missing_review_orders",
        "review_coverage",
        "low_support_cell",
    ],
    "RQ1_DURATION_ERROR_REVIEW_RATES.csv": [
        "actual_duration_group",
        "promise_error_group",
        "reviewed_orders",
        "low_review_2_orders",
        "low_review_2_rate",
        "low_review_2_ci_lower",
        "low_review_2_ci_upper",
        "mean_review_score",
        "median_review_score",
        "one_star_orders",
        "one_star_share",
        "low_support_cell",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name: str):
    path = WORKSPACE / f"scripts/{name}.py"
    assert path.is_file(), f"Required implementation module is absent: {path}"
    return importlib.import_module(
        f"analysis.rq1_speed_reliability_review_v1.scripts.{name}"
    )


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    candidates = []
    if os.environ.get("OLIST_DATA_DIR"):
        candidates.append(Path(os.environ["OLIST_DATA_DIR"]))
    candidates.extend(
        [
            Path("data/olist_data"),
            ROOT / "data/olist_data",
            ROOT / "olist_data",
        ]
    )
    for candidate in candidates:
        if all(
            (candidate / str(spec["filename"])).is_file()
            for spec in rq1_preflight.TRUSTED_RAW_INPUTS.values()
        ):
            return candidate.resolve()
    pytest.fail("The frozen raw Olist directory is unavailable")


@pytest.fixture(scope="session")
def analysis_frames(data_dir: Path, config: dict[str, Any]):
    return rq1_data.build_analysis_frames(data_dir, config)


@pytest.fixture(scope="session")
def all_orders(analysis_frames):
    return analysis_frames[0]


@pytest.fixture(scope="session")
def reviewed(analysis_frames):
    return analysis_frames[1]


@pytest.fixture(scope="session")
def data_audit(analysis_frames):
    return analysis_frames[2]


@pytest.fixture(scope="session")
def preservation_audit() -> dict[str, Any]:
    path = WORKSPACE / "working/PRE_TEST_PRESERVATION.json"
    assert path.is_file(), "Analysis-stage preservation receipt is absent"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def analysis_receipts() -> dict[str, dict[str, Any]]:
    path = WORKSPACE / "working/ANALYSIS_RECEIPTS.json"
    assert path.is_file(), "Analysis-stage artifact receipts are absent"
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipts = payload.get("artifact_receipts_before_tests", payload)
    assert isinstance(receipts, dict) and receipts
    return receipts


@pytest.fixture(scope="session")
def synthetic_reviewed(config: dict[str, Any]) -> pd.DataFrame:
    """A deterministic, non-separated sample spanning every frozen factor."""

    rng = np.random.default_rng(20260824)
    n = 3_200
    labels = np.asarray(config["promise_error_groups"]["labels"], dtype=object)
    error_values = np.asarray([-20, -10, -3, 0, 1, 2, 5, 10], dtype=int)
    group_index = np.arange(n) % len(labels)
    error = error_values[group_index]
    lower = np.maximum(error + 3, 0)
    actual = lower + rng.integers(0, 31, size=n)
    promised = actual - error
    months = np.asarray(["2018-01", "2018-02", "2018-03", "2018-04"])
    # Cross every error group with every month; a modulo assignment would make
    # the eight-level error group determine the four-level month factor.
    month = months[(np.arange(n) // len(labels)) % len(months)]
    linear = (
        -2.4
        + 0.025 * actual
        + 0.18 * np.maximum(error, 0)
        + 0.10 * (month == "2018-03")
    )
    probability = 1.0 / (1.0 + np.exp(-linear))
    low_2 = rng.binomial(1, np.clip(probability, 0.03, 0.90))
    low_3_probability = np.clip(probability + 0.12, 0.05, 0.95)
    low_3 = np.maximum(low_2, rng.binomial(1, low_3_probability))
    post_delivery = rng.random(n) >= 0.08
    score = np.where(low_2 == 1, 1, np.where(low_3 == 1, 3, 5))
    purchase_timestamp = pd.to_datetime(pd.Series(month) + "-01")
    actual_delivery_timestamp = purchase_timestamp + pd.to_timedelta(actual, unit="D")
    review_answer_timestamp = actual_delivery_timestamp + pd.to_timedelta(
        np.where(post_delivery, 1, -1), unit="D"
    )
    duration_edges = [
        -1.0,
        3.0,
        7.0,
        14.0,
        21.0,
        np.inf,
    ]
    frame = pd.DataFrame(
        {
            "order_id": [f"synthetic-{index:05d}" for index in range(n)],
            "actual_delivery_days": actual.astype(int),
            "promised_lead_days": promised.astype(int),
            "promise_error_days": error.astype(int),
            "promise_error_group": labels[group_index],
            "promise_error_group_label": labels[group_index],
            "purchase_month": month,
            "purchase_month_adjustment": month,
            "selected_review_score": score.astype(int),
            "low_review_2": low_2.astype(int),
            "low_review_3": low_3.astype(int),
            "review_at_or_after_delivery": post_delivery,
            "order_delivered_customer_date": actual_delivery_timestamp,
            "selected_review_answer_timestamp": review_answer_timestamp,
        }
    )
    frame["actual_duration_group"] = pd.cut(
        frame["actual_delivery_days"],
        bins=duration_edges,
        labels=config["actual_duration_groups"]["labels"],
        right=True,
        include_lowest=True,
        ordered=True,
    )
    assert (frame["actual_delivery_days"] == frame["promised_lead_days"] + frame["promise_error_days"]).all()
    assert set(frame["promise_error_group_label"]) == set(labels)
    return frame


@pytest.fixture(scope="session")
def synthetic_config(config: dict[str, Any], synthetic_reviewed: pd.DataFrame):
    local = deepcopy(config)
    local["expected_sample"]["reviewed_orders"] = len(synthetic_reviewed)
    local["expected_sample"]["purchase_month_min"] = str(
        synthetic_reviewed["purchase_month"].min()
    )
    local["expected_sample"]["purchase_month_max"] = str(
        synthetic_reviewed["purchase_month"].max()
    )
    return local


@pytest.fixture(scope="session")
def stats_module():
    return _module("rq1_stats")


@pytest.fixture(scope="session")
def reporting_module():
    return _module("rq1_reporting")


@pytest.fixture(scope="session")
def synthetic_stats(stats_module, synthetic_reviewed, synthetic_config):
    result = stats_module.run_statistical_analysis(
        synthetic_reviewed.copy(), synthetic_config
    )
    expected = {
        "model_results",
        "covariance",
        "wald",
        "probabilities",
        "contrasts",
        "comparison",
        "low_review_3_sensitivity",
        "review_timing_sensitivity",
        "duration_bin_sensitivity",
        "diagnostics",
        "decision",
    }
    assert set(result) == expected
    return result


def test_01_exact_canonical_assembler_hash():
    relative = "analysis/profile_pivot_phase2a/scripts/data_pipeline.py"
    expected = rq1_preflight.TRUSTED_SOURCE_ANCHORS[
        "programme_canonical_assembler"
    ]["sha256"]
    assert _sha256(ROOT / relative) == expected
    assert rq1_data.EXPECTED_SOURCE_SHA256[relative] == expected


def test_02_one_row_per_canonical_order(all_orders: pd.DataFrame):
    assert len(all_orders) == 96_470
    assert all_orders["order_id"].notna().all()
    assert all_orders["order_id"].is_unique


def test_03_exact_reviewed_order_reproduction(reviewed: pd.DataFrame):
    assert len(reviewed) == 95_824
    assert reviewed["order_id"].is_unique
    assert reviewed["usable_review"].all()


def test_04_exact_deterministic_review_selection(data_audit: dict[str, Any]):
    selection = data_audit["review_selection"]
    assert selection["raw_review_rows_linked"] == 96_353
    assert selection["selected_canonical_reviews"] == 95_824
    assert selection["selected_review_sha256"] == EXPECTED_SELECTED_DIGEST
    assert selection["orders_with_multiple_records"] == 525
    assert selection["orders_with_conflicting_scores"] == 189
    assert selection["reviews_before_delivery"] == 4_653
    assert selection["reviews_after_promise_before_delivery"] == 4_432
    assert selection["reviews_before_delivery_and_before_promised_date"] == 221
    assert selection["reviews_at_or_after_delivery"] == 91_171


def test_05_exact_existing_low_review_target(reviewed: pd.DataFrame):
    score = reviewed["selected_review_score"].astype(int)
    assert reviewed["low_review_2"].astype(int).equals(score.le(2).astype(int))
    assert int(reviewed["low_review_2"].sum()) == 12_272


def test_06_exact_existing_eight_error_groups(reviewed: pd.DataFrame):
    boundary_values = pd.Series(
        [-100, -14, -13, -7, -6, -1, 0, 1, 2, 3, 4, 7, 8, 100]
    )
    observed = promise_error_groups(boundary_values).astype(str).tolist()
    expected = [
        ERROR_GROUP_LABELS[0],
        ERROR_GROUP_LABELS[0],
        ERROR_GROUP_LABELS[1],
        ERROR_GROUP_LABELS[1],
        ERROR_GROUP_LABELS[2],
        ERROR_GROUP_LABELS[2],
        ERROR_GROUP_LABELS[3],
        ERROR_GROUP_LABELS[4],
        ERROR_GROUP_LABELS[5],
        ERROR_GROUP_LABELS[5],
        ERROR_GROUP_LABELS[6],
        ERROR_GROUP_LABELS[6],
        ERROR_GROUP_LABELS[7],
        ERROR_GROUP_LABELS[7],
    ]
    assert observed == expected
    counts = {
        str(key): int(value)
        for key, value in reviewed["promise_error_group"].value_counts(
            sort=False
        ).items()
    }
    assert counts == EXPECTED_ERROR_COUNTS


def test_07_exact_calendar_date_normalisation(all_orders: pd.DataFrame):
    purchase = pd.to_datetime(all_orders["order_purchase_timestamp"]).dt.normalize()
    actual = pd.to_datetime(all_orders["order_delivered_customer_date"]).dt.normalize()
    promise = pd.to_datetime(all_orders["order_estimated_delivery_date"]).dt.normalize()
    assert all_orders["purchase_date"].equals(purchase)
    assert all_orders["actual_delivery_date"].equals(actual)
    assert all_orders["estimated_delivery_date"].equals(promise)
    assert all_orders["actual_delivery_days"].astype(int).equals(
        (actual - purchase).dt.days
    )
    assert all_orders["promised_lead_days"].astype(int).equals(
        (promise - purchase).dt.days
    )


def test_08_exact_date_identity_d_equals_p_plus_e(all_orders: pd.DataFrame):
    residual = all_orders["actual_delivery_days"] - (
        all_orders["promised_lead_days"] + all_orders["promise_error_days"]
    )
    assert residual.notna().all()
    assert residual.eq(0).all()


def test_09_no_invalid_duration_silently_retained(all_orders: pd.DataFrame):
    columns = [
        "actual_delivery_days",
        "promised_lead_days",
        "promise_error_days",
        "actual_duration_group",
        "promise_error_group",
    ]
    assert not all_orders[columns].isna().any().any()
    assert all_orders["actual_delivery_days"].ge(0).all()
    assert all_orders["promised_lead_days"].ge(0).all()
    assert int(all_orders["actual_delivery_days"].max()) == 210
    assert int(all_orders["promised_lead_days"].max()) == 156


def test_10_fixed_duration_bin_boundaries(config: dict[str, Any]):
    values = pd.Series([0, 3, 4, 7, 8, 14, 15, 21, 22, 500])
    observed = pd.cut(
        values,
        bins=[-1, 3, 7, 14, 21, np.inf],
        labels=config["actual_duration_groups"]["labels"],
        right=True,
        include_lowest=True,
        ordered=True,
    ).astype(str).tolist()
    assert observed == [
        "0-3 days",
        "0-3 days",
        "4-7 days",
        "4-7 days",
        "8-14 days",
        "8-14 days",
        "15-21 days",
        "15-21 days",
        "22+ days",
        "22+ days",
    ]


def test_11_fixed_centered_spline_degrees_of_freedom(config: dict[str, Any]):
    models = config["models"]
    assert models["spline_df"] == 4
    assert "constraints='center'" in models["model_a"]
    assert "constraints='center'" in models["model_b"]
    assert "constraints='center'" in models["model_c"]


def test_12_exact_purchase_month_pooling_rule(
    reviewed: pd.DataFrame, config: dict[str, Any]
):
    adjustment, audit = rq1_data.pool_sparse_purchase_months(reviewed, config)
    assert audit["threshold"] == 500
    assert audit["sparse_months"] == ["2016-09", "2016-10", "2016-12"]
    assert audit["pooled_rows"] == 264
    assert audit["raw_month_levels"] == 23
    assert audit["adjustment_levels"] == 21
    assert audit["reference"] == "2018-01"
    assert adjustment.notna().all()


def test_13_hc1_covariance_is_used(stats_module, synthetic_stats):
    assert stats_module.COVARIANCE_TYPE == "HC1"
    fits = synthetic_stats["diagnostics"]
    assert isinstance(fits, pd.DataFrame)
    assert not fits.empty
    assert fits["covariance"].eq("HC1").all()
    assert not fits["use_t"].astype(bool).any()
    assert synthetic_stats["wald"]["covariance"].eq("HC1").all()
    assert synthetic_stats["model_results"]["standard_error_hc1"].notna().all()


def test_14_on_date_is_error_group_reference(
    stats_module, synthetic_stats, config: dict[str, Any]
):
    assert config["promise_error_groups"]["reference"] == "on promised date"
    for kind in ("B", "C", "BIN_B"):
        formula = stats_module.model_formula(kind, "low_review_2", config)
        assert 'Treatment(reference="on promised date")' in formula
    model_results = synthetic_stats["model_results"]
    error_terms = model_results["term"].astype(str)
    assert not error_terms.str.contains(r"\[T\.on promised date\]", regex=True).any()


def test_15_model_a_excludes_promise_error_group(stats_module, config):
    formula = stats_module.model_formula("A", "low_review_2", config)
    assert "actual_delivery_days" in formula
    assert "promise_error_group" not in formula
    assert "promised_lead_days" not in formula


def test_16_model_b_contains_duration_and_error_but_not_promised_lead(
    stats_module, config
):
    formula = stats_module.model_formula("B", "low_review_2", config)
    assert "actual_delivery_days" in formula
    assert "promise_error_group" in formula
    assert "promised_lead_days" not in formula


def test_17_model_c_contains_promised_lead_and_error_but_not_actual(
    stats_module, config
):
    formula = stats_module.model_formula("C", "low_review_2", config)
    assert "promised_lead_days" in formula
    assert "promise_error_group" in formula
    assert "actual_delivery_days" not in formula


def test_18_no_model_contains_unrestricted_d_p_and_e(stats_module, config):
    formulas = [
        stats_module.model_formula(kind, "low_review_2", config)
        for kind in ("A", "B", "C", "BIN_B")
    ]
    for formula in formulas:
        present = {
            name
            for name in (
                "actual_delivery_days",
                "promised_lead_days",
                "promise_error_days",
            )
            if name in formula
        }
        assert len(present) < 3


def test_19_deterministic_common_support_rule(
    stats_module, synthetic_reviewed, synthetic_config
):
    args = (
        synthetic_reviewed,
        "on promised date",
        "4-7 days late",
        synthetic_config,
    )
    result = stats_module.common_support_duration(*args)
    repeated = stats_module.common_support_duration(*args)
    assert result == repeated
    assert result["status"] == "supported"

    group_values = synthetic_reviewed.loc[
        synthetic_reviewed["promise_error_group"].eq("4-7 days late"),
        "actual_delivery_days",
    ]
    reference_values = synthetic_reviewed.loc[
        synthetic_reviewed["promise_error_group"].eq("on promised date"),
        "actual_delivery_days",
    ]
    expected_lower = max(
        group_values.quantile(0.05, interpolation="linear"),
        reference_values.quantile(0.05, interpolation="linear"),
    )
    expected_upper = min(
        group_values.quantile(0.95, interpolation="linear"),
        reference_values.quantile(0.95, interpolation="linear"),
    )
    pooled = synthetic_reviewed.loc[
        synthetic_reviewed["promise_error_group"].isin(
            ["on promised date", "4-7 days late"]
        )
        & synthetic_reviewed["actual_delivery_days"].between(
            expected_lower, expected_upper, inclusive="both"
        ),
        "actual_delivery_days",
    ]
    assert result["intersection_lower"] == pytest.approx(expected_lower)
    assert result["intersection_upper"] == pytest.approx(expected_upper)
    expected_reference = float(pooled.quantile(0.5, interpolation="nearest"))
    assert result["reference_duration"] == pytest.approx(expected_reference)
    assert float(result["reference_duration"]).is_integer()
    assert result["support_n"] == len(pooled)


def test_20_fixed_contrast_definitions(config: dict[str, Any], synthetic_stats):
    contrasts = config["contrasts"]
    assert "D75" in contrasts["speed"] and "D25" in contrasts["speed"]
    assert "4-7 days late" in contrasts["late_4_7"]
    assert ">=8 days late" in contrasts["late_8_plus"]
    assert contrasts["difference"] == "late_4_7 - speed"
    names = set(synthetic_stats["comparison"]["contrast_id"].astype(str))
    assert {
        "C_speed",
        "C_late_4_7",
        "C_late_8_plus",
        "C_difference_4_7_minus_speed",
        "C_difference_8_plus_minus_speed",
    } == names


def test_21_fixed_simulation_seed_and_draw_count(
    config: dict[str, Any], stats_module, synthetic_stats
):
    contrasts = config["contrasts"]
    assert contrasts["simulation_seed"] == 20260824
    assert contrasts["simulation_draws"] == 10_000
    comparison = synthetic_stats["comparison"]
    assert comparison["seed"].eq(20260824).all()
    assert comparison["n_draws"].eq(10_000).all()
    assert comparison["n_valid"].eq(10_000).all()
    assert comparison["mvn_method"].eq(stats_module.MVN_METHOD).all()


def test_22_alternative_low_review_target_sensitivity(reviewed: pd.DataFrame, synthetic_stats):
    assert reviewed["low_review_3"].astype(int).equals(
        reviewed["selected_review_score"].astype(int).le(3).astype(int)
    )
    assert int(reviewed["low_review_3"].sum()) == 20_188
    sensitivity = synthetic_stats["low_review_3_sensitivity"]
    assert not sensitivity.empty
    assert set(sensitivity["outcome"].astype(str)) == {"low_review_3"}


def test_23_timestamp_level_post_delivery_review_sensitivity(
    reviewed: pd.DataFrame, synthetic_stats
):
    direct = pd.to_datetime(reviewed["selected_review_answer_timestamp"]).ge(
        pd.to_datetime(reviewed["order_delivered_customer_date"])
    )
    assert reviewed["review_at_or_after_delivery"].astype(bool).equals(direct)
    assert int(direct.sum()) == 91_171
    sensitivity = synthetic_stats["review_timing_sensitivity"]
    assert not sensitivity.empty
    assert sensitivity["variant"].eq("post_delivery_reviews").all()
    assert set(sensitivity["record_type"]) == {"coefficient", "wald", "contrast"}
    assert sensitivity["n_orders"].nunique() == 1


def test_24_fixed_duration_bin_sensitivity(synthetic_stats, config: dict[str, Any]):
    sensitivity = synthetic_stats["duration_bin_sensitivity"]
    assert not sensitivity.empty
    assert sensitivity["variant"].eq("fixed_duration_bins").all()
    assert (
        sensitivity.loc[sensitivity["record_type"].eq("wald"), "record_id"]
        .eq("duration_bin")
        .any()
    )
    terms = sensitivity.loc[
        sensitivity["record_type"].eq("coefficient"), "record_id"
    ].astype(str)
    for label in config["actual_duration_groups"]["labels"][1:]:
        assert terms.str.contains(re.escape(label), regex=True).any()


def test_25_no_post_purchase_predictor_beyond_observational_targets(
    stats_module, config
):
    forbidden = {
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "selected_review_answer_timestamp",
        "review_text",
        "review_comment",
    }
    formulas = [
        stats_module.model_formula(kind, "low_review_2", config)
        for kind in ("A", "B", "C", "BIN_B")
    ]
    for formula in formulas:
        assert not any(name in formula for name in forbidden)


def test_26_no_weather_or_customer_profile_branch():
    forbidden_identifiers = {
        "weather",
        "weather_data",
        "customer_profile",
        "customer_profiles",
        "build_customer_profile",
    }
    observed_identifiers: set[str] = set()
    imported_modules: set[str] = set()
    for path in sorted((WORKSPACE / "scripts").glob("rq1_*.py")):
        if path.name == Path(__file__).name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                observed_identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                observed_identifiers.add(node.attr.lower())
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.lower())
    assert forbidden_identifiers.isdisjoint(observed_identifiers)
    assert not any(
        "weather" in module or "customer_profile" in module
        for module in imported_modules
    )


def test_27_no_protected_empirical_artifact_changed(preservation_audit):
    assert preservation_audit["passed"] is True
    assert preservation_audit["preservation_verdict"] == "unchanged"
    assert preservation_audit["protected_paths"]["passed"] is True


def test_28_no_governance_or_thesis_file_changed(preservation_audit):
    detail = preservation_audit["protected_paths"]["detail"]
    for protected_root in (
        "AGENTS.md",
        "PROJECT_CONTEXT.md",
        "DECISION_LOG.md",
        "RESULTS_REGISTRY.md",
        "docs",
        "report",
        "results",
    ):
        assert detail[protected_root]["unchanged"] is True


def test_29_deterministic_output_schemas_and_hash_receipts(
    analysis_receipts, stats_module, reporting_module
):
    receipts = analysis_receipts
    for relative_path, receipt in receipts.items():
        path = WORKSPACE / relative_path
        assert path.is_file()
        assert receipt["sha256"] == _sha256(path)
        assert receipt["bytes"] == path.stat().st_size
        if path.suffix.lower() == ".csv":
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            assert receipt["columns"] == columns
    for filename, columns in EXPECTED_DATA_TABLE_SCHEMAS.items():
        assert pd.read_csv(WORKSPACE / filename, nrows=0).columns.tolist() == columns
    statistical_schemas = {
        "RQ1_MODEL_RESULTS.csv": list(stats_module.MODEL_RESULT_COLUMNS),
        "RQ1_ROBUST_COVARIANCE.csv": list(stats_module.COVARIANCE_COLUMNS),
        "RQ1_ROBUST_WALD_TESTS.csv": list(stats_module.WALD_COLUMNS),
        "RQ1_ADJUSTED_PROBABILITIES.csv": list(stats_module.PROBABILITY_COLUMNS),
        "RQ1_ADJUSTED_CONTRASTS.csv": list(stats_module.CONTRAST_COLUMNS),
        "RQ1_CONTRAST_COMPARISON.csv": list(stats_module.COMPARISON_COLUMNS),
        "RQ1_LOW_REVIEW_3_SENSITIVITY.csv": list(stats_module.SENSITIVITY_COLUMNS),
        "RQ1_REVIEW_TIMING_SENSITIVITY.csv": list(stats_module.SENSITIVITY_COLUMNS),
        "RQ1_DURATION_BIN_SENSITIVITY.csv": list(stats_module.SENSITIVITY_COLUMNS),
    }
    assert reporting_module.STAT_TABLE_FILES == {
        "model_results": "RQ1_MODEL_RESULTS.csv",
        "covariance": "RQ1_ROBUST_COVARIANCE.csv",
        "wald": "RQ1_ROBUST_WALD_TESTS.csv",
        "probabilities": "RQ1_ADJUSTED_PROBABILITIES.csv",
        "contrasts": "RQ1_ADJUSTED_CONTRASTS.csv",
        "comparison": "RQ1_CONTRAST_COMPARISON.csv",
        "low_review_3_sensitivity": "RQ1_LOW_REVIEW_3_SENSITIVITY.csv",
        "review_timing_sensitivity": "RQ1_REVIEW_TIMING_SENSITIVITY.csv",
        "duration_bin_sensitivity": "RQ1_DURATION_BIN_SENSITIVITY.csv",
    }
    for filename, columns in statistical_schemas.items():
        path = WORKSPACE / filename
        assert analysis_receipts[filename]["columns"] == columns
        assert pd.read_csv(path, nrows=0).columns.tolist() == columns


def test_30_exact_decision_label_implementation(stats_module, synthetic_stats):
    cases = [
        (True, True, False, "EXPAND_RQ1_TO_SPEED_AND_RELIABILITY"),
        (False, True, False, "RETAIN_SIGNED_ERROR_ONLY_RQ1"),
        (
            True,
            False,
            False,
            "ACTUAL_DURATION_ASSOCIATION_WITHOUT_INCREMENTAL_PROMISE_ERROR",
        ),
        (False, False, False, "INCONCLUSIVE_RQ1_EXTENSION"),
        (True, True, True, "INCONCLUSIVE_RQ1_EXTENSION"),
        (False, True, True, "INCONCLUSIVE_RQ1_EXTENSION"),
        (True, False, True, "INCONCLUSIVE_RQ1_EXTENSION"),
        (False, False, True, "INCONCLUSIVE_RQ1_EXTENSION"),
    ]
    labels = {
        (duration, promise, override): expected
        for duration, promise, override, expected in cases
    }
    if hasattr(stats_module, "assign_extension_decision"):
        for duration, promise, override, expected in cases:
            assert stats_module.assign_extension_decision(
                duration_supported=duration,
                promise_relative_supported=promise,
                material_sensitivity_conflict=override,
            ) == expected

    decision = synthetic_stats["decision"]
    key = (
        bool(decision["actual_duration_association_supported"]),
        bool(decision["promise_relative_association_beyond_duration_supported"]),
        bool(decision["inconclusive_override"]),
    )
    assert decision["label"] == labels[key]
    assert decision["causal_claim_authorised"] is False
    assert decision["governance_update_authorised"] is False


def test_31_centered_spline_design_is_full_rank(synthetic_stats, stats_module):
    assert stats_module.SPLINE_DF == 4
    assert stats_module.SPLINE_CONSTRAINT == "center"
    diagnostics = synthetic_stats["diagnostics"]
    primary = diagnostics.loc[
        diagnostics["variant"].eq("primary")
        & diagnostics["model_id"].isin(["A", "B", "C"])
    ]
    assert len(primary) == 3
    assert (primary["design_rank"] == primary["design_columns"]).all()
    assert primary["spline_df"].eq(4).all()
    assert primary["spline_constraint"].eq("center").all()
    assert np.isfinite(primary["design_condition_number"]).all()


def test_32_predictions_reuse_fitted_design_info(synthetic_stats):
    diagnostics = synthetic_stats["diagnostics"]
    assert diagnostics["design_info_reused_for_prediction"].astype(bool).all()
    assert diagnostics["prediction_design_columns_match"].astype(bool).all()
    primary = diagnostics.loc[diagnostics["variant"].eq("primary")]
    hashes = set(primary["parameter_order_hash"].astype(str))
    comparison_hashes = set(
        synthetic_stats["comparison"]["parameter_order_hash"].astype(str)
    )
    assert comparison_hashes == {
        primary.loc[
            primary["model_id"].eq("B"),
            "parameter_order_hash",
        ].iloc[0]
    }
    assert comparison_hashes <= hashes


def test_33_hc1_covariance_passes_cholesky_gate(synthetic_stats, stats_module):
    assert stats_module.MVN_METHOD == "cholesky"
    diagnostics = synthetic_stats["diagnostics"]
    model_b = diagnostics.loc[
        diagnostics["variant"].eq("primary")
        & diagnostics["model_id"].eq("B")
    ].iloc[0]
    assert model_b["covariance"] == "HC1"
    assert bool(model_b["covariance_positive_definite"])
    assert float(model_b["covariance_min_eigenvalue"]) > 0
    covariance = synthetic_stats["covariance"]
    block = covariance.loc[
        covariance["variant"].eq("primary")
        & covariance["model_id"].eq("B")
    ].pivot(index="row_term", columns="column_term", values="covariance_hc1")
    order = list(model_b["parameter_order"])
    matrix = block.loc[order, order].to_numpy(dtype=float)
    assert np.isfinite(matrix).all()
    np.linalg.cholesky(matrix)


def test_34_mvn_draws_are_exact_and_deterministic(
    stats_module, config: dict[str, Any], synthetic_stats
):
    beta = np.asarray([0.1, -0.2, 0.3])
    covariance = np.asarray(
        [[0.09, 0.01, 0.00], [0.01, 0.04, 0.005], [0.00, 0.005, 0.16]]
    )
    first, first_audit = stats_module.draw_hc1_coefficients(beta, covariance, config)
    second, second_audit = stats_module.draw_hc1_coefficients(beta, covariance, config)
    assert first.shape == (10_000, 3)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first_audit == second_audit
    assert first_audit["valid_draws"] == 10_000
    assert first_audit["discarded_draws"] == 0
    expected_digest = hashlib.sha256(first.tobytes(order="C")).hexdigest()
    assert first_audit["draw_sha256"] == expected_digest
    comparison = synthetic_stats["comparison"]
    assert comparison["mvn_method"].eq("cholesky").all()
    assert comparison["quantile_method"].eq("linear").all()
    assert comparison["draws_all_finite"].astype(bool).all()
    assert comparison["probability_min"].between(0, 1, inclusive="both").all()
    assert comparison["probability_max"].between(0, 1, inclusive="both").all()


def test_35_common_support_failure_is_fail_closed(stats_module, config: dict[str, Any]):
    frame = pd.DataFrame(
        {
            "actual_delivery_days": [1, 2, 3, 4, 30, 31, 32, 33],
            "promise_error_group_label": ["on promised date"] * 4
            + ["4-7 days late"] * 4,
        }
    )
    result = stats_module.common_support_duration(
        frame, "on promised date", "4-7 days late", config
    )
    assert result["status"] == "insufficient_common_duration_support"
    assert result["reference_duration"] is None


def test_36_exact_three_figure_source_pairs(
    analysis_receipts, reporting_module
):
    figure_dir = WORKSPACE / "figures"
    source_dir = WORKSPACE / "figure_sources"
    figures = sorted(
        path for path in figure_dir.iterdir() if path.suffix.lower() in {".png", ".pdf"}
    )
    sources = sorted(path for path in source_dir.iterdir() if path.suffix == ".csv")
    assert [path.name for path in figures] == sorted(reporting_module.FIGURE_FILES)
    assert [path.name for path in sources] == sorted(
        reporting_module.FIGURE_SOURCE_FILES
    )
    assert reporting_module.FIGURE_PAIRS == {
        "01_absolute_duration_review.png": "01_absolute_duration_review.csv",
        "02_duration_error_review_heatmap.png": "02_duration_error_review_heatmap.csv",
        "03_adjusted_speed_reliability_associations.png":
            "03_adjusted_speed_reliability_associations.csv",
    }
    for path in figures + sources:
        relative = path.relative_to(WORKSPACE).as_posix()
        assert analysis_receipts[relative]["sha256"] == _sha256(path)
