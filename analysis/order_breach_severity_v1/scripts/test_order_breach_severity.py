"""Deterministic contract tests for ``order_breach_severity_v1``.

The suite deliberately separates two kinds of evidence:

* synthetic unit/contract tests, which always run and exercise the frozen
  information-time, model-ladder, metric, ablation, and determinism rules; and
* execution-receipt tests, which run only after the corresponding formal
  artifact exists.  A missing not-yet-produced artifact is skipped rather than
  interpreted as a successful zero-row result.

Run from the repository root with the project environment, for example::

    .venv/bin/python -B -m pytest -q -p no:cacheprovider \
      analysis/order_breach_severity_v1/scripts/test_order_breach_severity.py
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "analysis/order_breach_severity_v1"
WORKING = WORKSPACE / "working"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.order_breach_severity_v1.scripts import (  # noqa: E402
    order_experiment,
    order_features,
    order_io,
    order_modeling,
    order_preflight,
    order_profiles,
    order_reporting,
    run_order_experiment as order_runner,
)
from analysis.profile_pivot_phase2a.scripts import data_pipeline  # noqa: E402


PROFILE_FIELDS = (
    "name",
    "candidate_id",
    "base_candidate_id",
    "profile_spec_id",
    "entity",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent",
    "kappa",
    "min_support",
)

EXPECTED_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "S1": {
        "name": "seller_handling_level",
        "candidate_id": "handling_level|seller_id|C|w90|l14|P1|parent=global|kappa=na|min_support=5",
        "base_candidate_id": "handling_level|seller_id|C|w90|l14|P1|parent=global|kappa=na",
        "profile_spec_id": "ps_18f6d18af885ac9c1930",
        "entity": "seller_id",
        "scheme": "C",
        "window_days": 90,
        "lag_days": 14,
        "estimator": "P1",
        "parent": "global",
        "kappa": None,
        "min_support": 5,
    },
    "S2": {
        "name": "seller_handling_tail",
        "candidate_id": "handling_tail|seller_id|A|w90|l0|P1|parent=global|kappa=10|min_support=5",
        "base_candidate_id": "handling_tail|seller_id|A|w90|l0|P1|parent=global|kappa=10",
        "profile_spec_id": "ps_29c28f8f40eed03c1031",
        "entity": "seller_id",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 10,
        "min_support": 5,
    },
    "R1": {
        "name": "route_transit_level",
        "candidate_id": "transit_level|state_od|A|w90|l0|P0|parent=global|kappa=na|min_support=5",
        "base_candidate_id": "transit_level|state_od|A|w90|l0|P0|parent=global|kappa=na",
        "profile_spec_id": "ps_18f16966ac00ff520226",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P0",
        "parent": "global",
        "kappa": None,
        "min_support": 5,
    },
    "R2": {
        "name": "route_transit_tail",
        "candidate_id": "transit_tail|state_od|A|w90|l0|P1|parent=global|kappa=10|min_support=5",
        "base_candidate_id": "transit_tail|state_od|A|w90|l0|P1|parent=global|kappa=10",
        "profile_spec_id": "ps_9799491505b2347220fb",
        "entity": "state_od",
        "scheme": "A",
        "window_days": 90,
        "lag_days": 0,
        "estimator": "P1",
        "parent": "global",
        "kappa": 10,
        "min_support": 5,
    },
}

EXPECTED_ENDPOINT_SPEC = {
    "name": "route_historical_final_breach",
    "candidate_id": "final_breach|state_od|A|w90|l0|P1|parent=global|kappa=100|min_support=5",
    "base_candidate_id": "final_breach|state_od|A|w90|l0|P1|parent=global|kappa=100",
    "profile_spec_id": "ps_ef5d05dc7c0496cca415",
    "entity": "state_od",
    "scheme": "A",
    "window_days": 90,
    "lag_days": 0,
    "estimator": "P1",
    "parent": "global",
    "kappa": 100,
    "min_support": 5,
}


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    return json.loads((WORKSPACE / "ORDER_FROZEN_CONFIG.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "pass", "passed", "verified", "unchanged"}


def _assert_xgboost_quantile_params(
    params: Mapping[str, object], quantile: float
) -> None:
    assert params["objective"] == "reg:quantileerror"
    assert params["eval_metric"] == "quantile"
    assert float(params["quantile_alpha"]) == pytest.approx(float(quantile), abs=1e-12)
    receipt = order_modeling.stable_json(params)
    assert "binary:logistic" not in receipt
    assert "logloss" not in receipt


def _minimal_valid_model_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "order_id": ["order-0", "order-1"],
            "purchase_date": pd.to_datetime(["2017-06-01", "2017-06-02"]),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-06-01 10:00:00", "2017-06-02 11:00:00"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2017-06-05 10:00:00", "2017-06-10 11:00:00"]
            ),
            "late_delivery": [0, 1],
            "positive_late_days": [0.0, 2.0],
        }
    )
    for feature in order_features.CURRENT_ORDER_NUMERIC_FEATURES:
        frame[feature] = 1.0
    frame["carnival_period"] = 0
    frame["known_event_indicator"] = 0
    frame["multi_seller"] = 0
    for feature in order_features.CURRENT_ORDER_CATEGORICAL_FEATURES:
        frame[feature] = "SP"
    for block in order_experiment.PROFILE_BLOCKS:
        for suffix in order_experiment.PROFILE_PAYLOAD_SUFFIXES:
            frame[f"{block}_{suffix}"] = 0.0
    frame["S1_score_x_known_event"] = 0.0
    frame["R1_score_x_known_event"] = 0.0
    return frame


def _profile_join_inputs(
    *, last_mature: str = "2018-01-08",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    result = pd.DataFrame({"order_id": ["seen", "new", "missing"]})
    snapshot = pd.Series(pd.to_datetime(["2018-01-10"] * 3), index=result.index)
    entity = pd.Series(pd.array(["seller-seen", "seller-new", pd.NA], dtype="string"))
    profile = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2018-01-10"]),
            "entity_id": ["seller-seen"],
            "score": [0.8],
            "support": [6],
            "posterior_se": [0.12],
            "profile_freshness_days": [2],
            "last_mature_outcome_date": pd.to_datetime([last_mature]),
        }
    )
    parent = pd.Series(
        [0.25], index=pd.DatetimeIndex([pd.Timestamp("2018-01-10")]), name="parent_score"
    )
    return result, snapshot, entity, profile, parent


def _paired_predictions() -> pd.DataFrame:
    order_ids = [f"order-{index:03d}" for index in range(28)]
    target = (np.arange(28) % 5 == 0).astype(int)
    base = 0.06 + 0.62 * target
    parts: list[pd.DataFrame] = []
    for model_index, model_id in enumerate(order_experiment.model_feature_map()):
        probability = np.clip(base + (model_index - 3) * 0.002, 0.01, 0.99)
        parts.append(
            pd.DataFrame(
                {
                    "order_id": order_ids,
                    "purchase_date": pd.date_range("2018-01-01", periods=28, freq="D"),
                    "period": "later",
                    "cohort": "2018-01",
                    "family": "logistic_l2",
                    "model_id": model_id,
                    "target": target,
                    "raw_probability": probability,
                    "calibrated_probability": probability,
                    "multi_seller": (np.arange(28) % 2).astype(int),
                    "S1_support": 25,
                    "S2_support": 25,
                    "R1_support": 25,
                    "R2_support": 25,
                    "S1_cold_start": False,
                    "S2_cold_start": False,
                    "R1_cold_start": False,
                    "R2_cold_start": False,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Population, source, and frozen-profile controls
# ---------------------------------------------------------------------------


def test_canonical_one_row_per_order_contract() -> None:
    valid = _minimal_valid_model_frame()
    order_experiment.validate_model_frame(valid)

    duplicated = valid.copy()
    duplicated.loc[:, "order_id"] = "duplicate"
    with pytest.raises(AssertionError, match="one nonmissing row per order"):
        order_experiment.validate_model_frame(duplicated)

    missing = valid.copy()
    missing.loc[0, "order_id"] = None
    with pytest.raises(AssertionError, match="one nonmissing row per order"):
        order_experiment.validate_model_frame(missing)


def test_canonical_one_row_per_order_artifact_if_available() -> None:
    path = WORKSPACE / "ORDER_SAMPLE_AUDIT.csv"
    if not path.is_file():
        pytest.skip("formal ORDER_SAMPLE_AUDIT.csv has not been produced")
    audit = pd.read_csv(path)
    row = audit.loc[audit["sample"].eq("canonical_delivered_all_dates")]
    assert len(row) == 1
    record = row.iloc[0]
    assert int(record["n_orders"]) == 96_470
    assert int(record["n_unique_orders"]) == 96_470
    assert int(record["duplicate_order_ids"]) == 0
    assert int(record["unresolved_target_rows"]) == 0


def test_exact_source_hash_trust_anchors(config: Mapping[str, Any]) -> None:
    data = config["data"]
    assert data["canonical_assembler_sha256"] == order_preflight.EXPECTED_ASSEMBLER_SHA256
    assert data["canonical_assembler_sha256"] == _sha256(ROOT / data["canonical_assembler"])
    assert data["profile_daily_input_sha256"] == order_profiles.PROFILE_DAILY_SHA256
    assert data["profile_parent_input_sha256"] == order_profiles.PROFILE_PARENT_SHA256
    assert data["profile_selection_freeze_sha256"] == order_profiles.PROFILE_SELECTION_FREEZE_SHA256
    assert data["profile_selected_candidates_sha256"] == order_profiles.PROFILE_SELECTED_CANDIDATES_SHA256

    # The small frozen selection controls are hashed directly.  The 11+ GB
    # daily profile and all raw inputs are validated once by formal preflight
    # and checked below through its persisted receipt.
    assert _sha256(ROOT / data["profile_selection_freeze"]) == data["profile_selection_freeze_sha256"]
    assert _sha256(ROOT / data["profile_selected_candidates"]) == data["profile_selected_candidates_sha256"]


def test_local_frozen_control_hashes_match_preflight_trust_anchors() -> None:
    assert _sha256(WORKSPACE / "ORDER_FROZEN_CONFIG.json") == order_preflight.EXPECTED_ORDER_CONFIG_SHA256
    assert _sha256(WORKSPACE / "ORDER_PROTOCOL.md") == order_preflight.EXPECTED_ORDER_PROTOCOL_SHA256


def test_exact_source_hash_receipt_if_available() -> None:
    path = order_preflight.SOURCE_AUDIT_PATH
    if not path.is_file():
        pytest.skip("formal SOURCE_INPUT_AUDIT.csv has not been produced")
    audit = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert tuple(audit.columns) == tuple(order_preflight.SOURCE_AUDIT_COLUMNS)
    assert len(audit) == len(order_preflight.RAW_FILE_SPECS) + 1 + len(order_preflight.PROFILE_INPUT_SPECS)
    assert audit["exists"].eq("true").all()
    assert audit["status"].eq("verified").all()
    assert audit["actual_sha256"].eq(audit["expected_sha256"]).all()
    assert audit["actual_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


@pytest.mark.parametrize("block", ["S1", "S2", "R1", "R2"], ids=str.lower)
def test_exact_frozen_primary_profile_specification(
    block: str, config: Mapping[str, Any]
) -> None:
    expected = EXPECTED_PROFILE_SPECS[block]
    configured = config["profiles"][block]
    implemented = order_profiles.FROZEN_PROFILE_SPECS[block]
    preflight_anchor = order_preflight.EXPECTED_PROFILE_BLOCKS[block]
    for field in PROFILE_FIELDS:
        assert configured[field] == expected[field]
        assert implemented[field] == expected[field]
        assert preflight_anchor[field] == expected[field]


def test_endpoint_profile_is_resolved_by_preconfirmation_development_rank(
    config: Mapping[str, Any]
) -> None:
    endpoint = config["profiles"]["M5_ENDPOINT"]
    implemented = order_profiles.FROZEN_PROFILE_SPECS["M5_ENDPOINT"]
    for field, value in EXPECTED_ENDPOINT_SPEC.items():
        assert endpoint[field] == value
        assert implemented[field] == value
    assert endpoint["representative_rule"] == "lowest_frozen_development_selection_rank_then_lexical_candidate_id"
    assert endpoint["frozen_selection_rank"] == 1

    paths = order_profiles._validate_config(config)
    order_profiles._validate_selection_controls(paths)
    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    assert freeze["confirmation_outcomes_accessed"] is False

    selected = pd.read_csv(paths["selected"])
    family = selected.loc[
        selected["candidate_id"].astype(str).str.startswith(
            "final_breach|state_od|A|w90|l0|P1|parent=global|kappa="
        )
        & selected["candidate_id"].astype(str).str.endswith("|min_support=5")
        & selected["selection_decision"].eq("selected")
    ].copy()
    assert not family.empty
    resolved = family.sort_values(["selection_rank", "candidate_id"], kind="mergesort").iloc[0]
    assert resolved["candidate_id"] == endpoint["candidate_id"]
    assert int(resolved["selection_rank"]) == 1
    assert float(resolved["kappa"]) == 100.0


def test_profile_snapshot_is_purchase_asof_and_strictly_prefuture() -> None:
    result, snapshot, entity, profile, parent = _profile_join_inputs()
    purchase = pd.Series(pd.to_datetime(["2018-01-10 09:00:00"] * 3))
    assert snapshot.eq(purchase.dt.normalize()).all()
    assert snapshot.le(purchase).all()

    order_profiles._join_one_block(result, snapshot, entity, profile, parent, "S1")
    seen = result["S1_mapping_status"].eq("seen")
    assert pd.to_datetime(result.loc[seen, "S1_last_mature_outcome_date"]).lt(
        snapshot.loc[seen]
    ).all()

    bad_result, bad_snapshot, bad_entity, bad_profile, bad_parent = _profile_join_inputs(
        last_mature="2018-01-10"
    )
    with pytest.raises(AssertionError, match="strict pre-snapshot maturity"):
        order_profiles._join_one_block(
            bad_result, bad_snapshot, bad_entity, bad_profile, bad_parent, "S1"
        )


def test_profile_timing_receipt_if_available() -> None:
    path = WORKSPACE / "ORDER_PROFILE_JOIN_AUDIT.csv"
    if not path.is_file():
        pytest.skip("formal ORDER_PROFILE_JOIN_AUDIT.csv has not been produced")
    audit = pd.read_csv(path)
    assert set(audit["block"]) == set(order_profiles.PROFILE_BLOCKS)
    assert pd.to_numeric(audit["snapshot_after_purchase_violations"], errors="raise").eq(0).all()
    assert pd.to_numeric(audit["seen_history_time_violations"], errors="raise").eq(0).all()
    assert audit["snapshot_rule"].eq("snapshot_date_equals_normalized_order_purchase_date").all()


# ---------------------------------------------------------------------------
# Chronological isolation and paired-comparison controls
# ---------------------------------------------------------------------------


def test_identical_paired_order_ids(config: Mapping[str, Any]) -> None:
    predictions = _paired_predictions()
    lightweight = copy.deepcopy(config)
    lightweight["uncertainty"]["paired_calendar_block_bootstrap_replicates"] = 25
    paired = order_experiment.paired_classification_differences(predictions, lightweight)
    assert len(paired) == 2 * len(order_experiment.PRIMARY_COMPARISONS)
    assert paired["n_orders"].eq(28).all()
    assert paired["paired_order_id_sha256"].eq(
        order_modeling.order_id_hash(predictions.loc[predictions["model_id"].eq("M0"), "order_id"])
    ).all()

    unpaired = predictions.drop(
        predictions.index[
            predictions["model_id"].eq("M2") & predictions["order_id"].eq("order-000")
        ]
    )
    with pytest.raises(AssertionError, match="nonidentical paired sample"):
        order_experiment.paired_classification_differences(unpaired, lightweight)


def test_development_only_hyperparameter_tuning(
    config: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    folds = order_experiment.frozen_folds(config)
    assert len(folds) == 3
    assert config["classification"]["hyperparameter_selection_model"] == "M1"
    assert config["severity"]["xgboost_objective"] == "reg:quantileerror"
    assert config["severity"]["xgboost_eval_metric"] == "quantile"
    assert config["calibration"]["selection_source"] == "development_chronological_out_of_fold_predictions_only"
    assert config["periods"]["development_validation_label_rule"] == (
        "purchase_in_validation_window_and_actual_delivery_timestamp_strictly_before_"
        "validation_window_end"
    )
    for fold in folds:
        assert pd.Timestamp(fold["train_start"]) == pd.Timestamp("2017-04-01")
        assert pd.Timestamp(fold["train_end_exclusive"]) <= pd.Timestamp(fold["validation_start"])
        assert pd.Timestamp(fold["validation_start"]) < pd.Timestamp(fold["validation_end_exclusive"])
        assert pd.Timestamp(fold["validation_end_exclusive"]) <= pd.Timestamp("2018-01-01")

    frame = pd.DataFrame(
        {
            "order_id": [f"d-{index:04d}" for index in range(275)],
            "purchase_date": pd.date_range("2017-04-01", periods=275, freq="D"),
        }
    )
    frame["order_delivered_customer_date"] = frame["purchase_date"] + pd.Timedelta(days=3)
    frame["positive_late_days"] = 1.0 + (np.arange(len(frame)) % 7)
    for fold in folds:
        train, validation = order_experiment.chronological_masks(frame, fold)
        assert not (set(frame.loc[train, "order_id"]) & set(frame.loc[validation, "order_id"]))
        assert frame.loc[train, "purchase_date"].lt(pd.Timestamp(fold["train_end_exclusive"])).all()
        assert frame.loc[train, "order_delivered_customer_date"].lt(
            pd.Timestamp(fold["validation_start"])
        ).all()
        assert frame.loc[validation, "purchase_date"].ge(pd.Timestamp(fold["validation_start"])).all()
        assert frame.loc[validation, "purchase_date"].lt(
            pd.Timestamp(fold["validation_end_exclusive"])
        ).all()
        assert frame.loc[validation, "order_delivered_customer_date"].lt(
            pd.Timestamp(fold["validation_end_exclusive"])
        ).all()
        in_purchase_window_but_unmatured = (
            frame["purchase_date"].ge(pd.Timestamp(fold["validation_start"]))
            & frame["purchase_date"].lt(pd.Timestamp(fold["validation_end_exclusive"]))
            & frame["order_delivered_customer_date"].ge(
                pd.Timestamp(fold["validation_end_exclusive"])
            )
        )
        assert in_purchase_window_but_unmatured.any()
        assert not validation.loc[in_purchase_window_but_unmatured].any()

    selection_source = "\n".join(
        inspect.getsource(function).lower()
        for function in (
            order_experiment.tune_classification,
            order_experiment.tune_severity,
            order_experiment.development_oof_and_calibration,
        )
    )
    assert "confirmation" not in selection_source

    captured: list[tuple[str, float, dict[str, object]]] = []

    class DummyTunedQuantile:
        best_iteration = 1

        def __init__(self, quantile: float) -> None:
            self.quantile = quantile

        def predict(self, prediction_frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(prediction_frame), 2.0 + self.quantile)

    def fake_fit_quantile_model(
        training_frame: pd.DataFrame,
        target: Sequence[float],
        numeric: Sequence[str],
        categorical: Sequence[str],
        family: str,
        quantile: float,
        params: Mapping[str, object],
        **_: object,
    ) -> DummyTunedQuantile:
        del numeric, categorical
        values = np.asarray(target, dtype=float)
        assert len(training_frame) == len(values)
        assert np.all(values > 0)
        captured.append((family, quantile, dict(params)))
        return DummyTunedQuantile(quantile)

    monkeypatch.setattr(order_modeling, "fit_quantile_model", fake_fit_quantile_model)
    classifier_xgboost = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": 20,
        "n_jobs": 1,
        "random_state": 20260823,
    }
    chosen, tuning = order_experiment.tune_severity(
        frame, config, {"xgboost": classifier_xgboost}
    )
    xgb_calls = [item for item in captured if item[0] == "xgboost_quantile"]
    assert len(xgb_calls) == len(config["severity"]["quantiles"]) * len(folds)
    for _, quantile, params in xgb_calls:
        _assert_xgboost_quantile_params(params, quantile)
    for quantile in config["severity"]["quantiles"]:
        _assert_xgboost_quantile_params(
            chosen["xgboost_quantile"][str(float(quantile))], float(quantile)
        )
    xgb_tuning = tuning.loc[tuning["family"].eq("xgboost_quantile")]
    assert len(xgb_tuning) == len(xgb_calls)
    for row in xgb_tuning.itertuples(index=False):
        _assert_xgboost_quantile_params(json.loads(row.parameters_json), row.quantile)


def test_development_selection_freeze_receipt_if_available() -> None:
    path = WORKSPACE / "ORDER_MODEL_SELECTION_FREEZE.json"
    if not path.is_file():
        pytest.skip("formal ORDER_MODEL_SELECTION_FREEZE.json has not been produced")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    assert freeze["selection_source"] == "development_2017-04-01_to_2017-12-31_only"
    assert freeze["later_or_terminal_outcomes_used"] is False
    parameters = pd.read_csv(WORKSPACE / "ORDER_MODEL_PARAMETERS.csv")
    assert parameters["development_only"].map(_truthy).all()
    severity_xgb = parameters.loc[
        parameters["task"].eq("severity")
        & parameters["family"].eq("xgboost_quantile")
    ]
    assert not severity_xgb.empty
    for row in severity_xgb.itertuples(index=False):
        _assert_xgboost_quantile_params(json.loads(row.parameters_json), row.quantile)

    frozen_xgb = freeze["severity_parameters"]["xgboost_quantile"]
    for quantile, params in frozen_xgb.items():
        _assert_xgboost_quantile_params(params, float(quantile))

    tuning_path = WORKSPACE / "ORDER_DEVELOPMENT_TUNING.csv"
    tuning = pd.read_csv(tuning_path)
    tuning_xgb = tuning.loc[
        tuning["task"].eq("severity") & tuning["family"].eq("xgboost_quantile")
    ]
    assert not tuning_xgb.empty
    for row in tuning_xgb.itertuples(index=False):
        _assert_xgboost_quantile_params(json.loads(row.parameters_json), row.quantile)


def test_later_cohort_isolation() -> None:
    cohorts = order_experiment.evaluation_cohorts()
    later = [cohort for cohort in cohorts if cohort.period == "later"]
    assert [cohort.cohort for cohort in later] == [
        "2018-01",
        "2018-02",
        "2018-03",
        "2018-04",
        "2018-05",
        "2018-06",
    ]
    for cohort in later:
        assert cohort.origin == cohort.start
        assert cohort.start >= pd.Timestamp("2018-01-01")
        assert cohort.end_exclusive <= pd.Timestamp("2018-07-01")

    frame = pd.DataFrame(
        {
            "order_id": ["mature", "equal-origin", "jan", "feb"],
            "purchase_date": pd.to_datetime(
                ["2017-12-15", "2017-12-20", "2018-01-05", "2018-02-05"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2017-12-25", "2018-01-01", "2018-01-20", "2018-02-20"]
            ),
        }
    )
    train, test = order_experiment.cohort_masks(frame, later[0])
    assert frame.loc[train, "order_id"].tolist() == ["mature"]
    assert frame.loc[test, "order_id"].tolist() == ["jan"]


def test_terminal_isolation() -> None:
    terminal = [
        cohort for cohort in order_experiment.evaluation_cohorts() if cohort.period == "terminal"
    ]
    assert len(terminal) == 1
    cohort = terminal[0]
    assert cohort.origin == pd.Timestamp("2018-07-01")
    assert cohort.start == pd.Timestamp("2018-07-01")
    assert cohort.end_exclusive == pd.Timestamp("2018-08-31")

    frame = pd.DataFrame(
        {
            "order_id": ["old-mature", "old-unmatured", "july", "august"],
            "purchase_date": pd.to_datetime(
                ["2018-05-01", "2018-06-20", "2018-07-01", "2018-08-30"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-05-10", "2018-07-01", "2018-07-15", "2018-09-20"]
            ),
        }
    )
    train, test = order_experiment.cohort_masks(frame, cohort)
    assert frame.loc[train, "order_id"].tolist() == ["old-mature"]
    assert frame.loc[test, "order_id"].tolist() == ["july", "august"]

    # Terminal reporting must preserve raw versus calibrated probabilities and
    # express reliability movement against the actual Jan--Jun pooled rows.
    metric_defaults = {
        "log_loss": 0.20,
        "brier": 0.05,
        "average_precision": 0.30,
        "roc_auc": 0.70,
    }
    breach_rows: list[dict[str, object]] = []
    for probability_type, later_values, terminal_values in (
        ("raw", (0.10, 1.00, 0.05), (0.30, 0.80, 0.10)),
        ("calibrated", (0.00, 1.00, 0.04), (0.20, 0.90, 0.08)),
    ):
        for period, cohort_name, values, n_orders in (
            ("aggregate", "later_pooled", later_values, 600),
            ("terminal", "2018-07_to_2018-08", terminal_values, 200),
        ):
            breach_rows.append(
                {
                    "period": period,
                    "cohort": cohort_name,
                    "family": "logistic_l2",
                    "model_id": "M1",
                    "probability_type": probability_type,
                    "n_orders": n_orders,
                    **metric_defaults,
                    "calibration_intercept": values[0],
                    "calibration_slope": values[1],
                    "wace": values[2],
                }
            )
    terminal_predictions = pd.DataFrame(
        {
            "period": ["later", "later", "terminal", "terminal"],
            "family": "logistic_l2",
            "model_id": "M4",
            "S1_score": [1.0, 3.0, 4.0, 6.0],
            "S2_score": [2.0, 4.0, 5.0, 7.0],
            "R1_score": [3.0, 5.0, 6.0, 8.0],
            "R2_score": [4.0, 6.0, 7.0, 9.0],
        }
    )
    stress = order_runner._terminal_stress(
        pd.DataFrame(breach_rows),
        pd.DataFrame(columns=["period", "delta_log_loss", "delta_brier"]),
        pd.DataFrame(columns=["period"]),
        terminal_predictions,
    )
    breach_model = stress.loc[stress["analysis"].eq("breach_model")]
    assert set(breach_model["probability_type"]) == {"raw", "calibrated"}
    assert breach_model.groupby("probability_type")["metric"].nunique().eq(7).all()
    shift = stress.loc[stress["analysis"].eq("breach_calibration_shift")].copy()
    assert len(shift) == 6
    assert shift["comparison"].eq("terminal_minus_later_pooled").all()
    assert shift["interpretation"].eq(
        "terminal_minus_january_june_pooled_calibration_shift"
    ).all()
    observed = shift.set_index(["probability_type", "metric"])["estimate"].to_dict()
    assert observed[("raw", "terminal_minus_later_calibration_intercept")] == pytest.approx(0.20)
    assert observed[("raw", "terminal_minus_later_calibration_slope")] == pytest.approx(-0.20)
    assert observed[("raw", "terminal_minus_later_wace")] == pytest.approx(0.05)
    assert observed[("calibrated", "terminal_minus_later_calibration_intercept")] == pytest.approx(0.20)
    assert observed[("calibrated", "terminal_minus_later_calibration_slope")] == pytest.approx(-0.10)
    assert observed[("calibrated", "terminal_minus_later_wace")] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Leakage, target, and feature-representation controls
# ---------------------------------------------------------------------------


def test_no_retrospective_hrd_feature_leakage(config: Mapping[str, Any]) -> None:
    all_features = {
        feature
        for numeric, categorical in order_experiment.model_feature_map().values()
        for feature in (*numeric, *categorical)
    }
    retrospective = {
        "both_top10",
        "both_top10_phase",
        "retrospective_hrd_label",
        "realised_daily_order_count",
        "realised_daily_gmv",
    }
    assert all_features.isdisjoint(retrospective)
    assert config["hrd"]["predictor_allowed"] is False
    assert config["hrd"]["evaluation_only"] is True
    assert config["current_order_features"]["event_interactions"] == [
        "S1_score_x_known_event",
        "R1_score_x_known_event",
    ]
    with pytest.raises(ValueError):
        order_features.validate_feature_availability(["both_top10"])
    with pytest.raises(ValueError):
        order_features.validate_no_forbidden_features(["retrospective_hrd_label"])


@pytest.mark.parametrize(
    "feature",
    [
        "order_approved_at",
        "order_delivered_carrier_date",
        "handling_duration",
        "transit_duration",
        "payment_value",
        "realised_daily_order_count",
    ],
)
def test_no_postpurchase_feature_in_current_order_block(feature: str) -> None:
    assert feature not in order_features.CURRENT_ORDER_FEATURES
    with pytest.raises(ValueError, match="Forbidden current-order predictors"):
        order_features.validate_no_forbidden_features([feature])


@pytest.mark.parametrize(
    "feature",
    [
        "order_delivered_customer_date",
        "promise_error_days",
        "late_delivery",
        "positive_late_days",
        "review_score",
        "review_comment_message",
    ],
)
def test_no_delivery_target_or_review_leakage(feature: str) -> None:
    assert feature not in order_features.CURRENT_ORDER_FEATURES
    with pytest.raises(ValueError, match="Forbidden current-order predictors"):
        order_features.validate_no_forbidden_features([feature])


def test_no_unresolved_target_is_coded_as_zero() -> None:
    frame = _minimal_valid_model_frame()
    frame.loc[0, "late_delivery"] = np.nan
    with pytest.raises(AssertionError, match="observed binary"):
        order_experiment.validate_model_frame(frame)

    frame = _minimal_valid_model_frame()
    frame.loc[0, "positive_late_days"] = np.nan
    with pytest.raises(AssertionError, match="observed and nonnegative"):
        order_experiment.validate_model_frame(frame)


def test_score_metadata_and_full_profile_ablations_are_exact() -> None:
    variants = order_experiment.ablation_feature_map()
    base_numeric, base_categorical = order_experiment.model_feature_map()["M1"]
    assert set(variants) == {
        "seller_score_only",
        "seller_metadata_only",
        "seller_full",
        "seller_s1_score_only",
        "seller_s2_score_only",
        "route_score_only",
        "route_metadata_only",
        "route_full",
        "route_r1_score_only",
        "route_r2_score_only",
        "combined_score_only",
        "combined_metadata_only",
        "combined_full",
    }

    for name, blocks in (
        ("seller", ("S1", "S2")),
        ("route", ("R1", "R2")),
        ("combined", ("S1", "S2", "R1", "R2")),
    ):
        _, score_numeric, score_categorical = variants[f"{name}_score_only"]
        _, metadata_numeric, metadata_categorical = variants[f"{name}_metadata_only"]
        _, full_numeric, full_categorical = variants[f"{name}_full"]
        scores = {f"{block}_score" for block in blocks}
        metadata = {
            f"{block}_{suffix}"
            for block in blocks
            for suffix in ("log1p_support", "cold_start", "posterior_se", "freshness_days")
        }
        assert set(score_numeric) - set(base_numeric) == scores
        assert set(metadata_numeric) - set(base_numeric) == metadata
        assert set(full_numeric) - set(base_numeric) == scores | metadata
        assert score_categorical == metadata_categorical == full_categorical == base_categorical


def test_cold_start_global_parent_fallback_and_missing_mapping_are_distinct() -> None:
    result, snapshot, entity, profile, parent = _profile_join_inputs()
    audit = order_profiles._join_one_block(result, snapshot, entity, profile, parent, "S1")

    assert result["S1_mapping_status"].tolist() == [
        "seen",
        "mapped_cold_start",
        "missing_mapping",
    ]
    assert result["S1_cold_start"].tolist() == [False, True, False]
    assert result["S1_score"].tolist() == pytest.approx([0.8, 0.25, 0.25])
    assert result["S1_support"].tolist() == [6, 0, 0]
    assert result["S1_log1p_support"].tolist() == pytest.approx([math.log1p(6), 0.0, 0.0])
    assert pd.isna(result.loc[1, "S1_posterior_se"])
    assert pd.isna(result.loc[2, "S1_freshness_days"])
    assert audit["orders_seen"] == 1
    assert audit["orders_mapped_cold_start"] == 1
    assert audit["orders_missing_mapping"] == 1
    assert audit["orders_global_parent_fallback"] == 2


def test_multi_seller_handling_is_deterministic_and_retained(config: Mapping[str, Any]) -> None:
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-02 12:00:00", "2018-01-06 15:00:00"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-10 00:00:00", "2018-01-15 00:00:00"]
            ),
            "promised_delivery_days": [7.5, 8.375],
            "n_items": [1, 3],
            "n_unique_products": [1, 2],
            "n_unique_sellers": [1, 2],
            "multi_item": [0, 1],
            "multi_product": [0, 1],
            "total_price": [100.0, 200.0],
            "total_freight_value": [10.0, 20.0],
            "freight_to_price_ratio": [0.1, 0.1],
            "avg_product_weight_g": [500.0, 800.0],
            "avg_product_volume_cm3": [1000.0, 2000.0],
            "max_product_dimension_cm": [20.0, 30.0],
            "category_diversity": [1, 2],
            "customer_seller_same_state": [1, 0],
            "distance_km": [10.0, 300.0],
            "purchase_month_num": [1, 1],
            "purchase_weekday": [1, 5],
            "purchase_hour": [12, 15],
            "is_weekend_purchase": [0, 1],
            "customer_state": ["SP", "RJ"],
            "main_seller_state": ["SP", "MG"],
            "main_product_category": ["books", "furniture"],
            "route_region": ["Southeast -> Southeast", "Southeast -> Southeast"],
            "distance_band": ["0-50", "200-500"],
            "main_seller_id": ["seller-z", "seller-a"],
        }
    )
    original_promise = frame["promised_delivery_days"].copy()
    built = order_features.build_current_order_features(frame)
    assert built["multi_seller"].tolist() == [0, 1]
    pd.testing.assert_series_equal(built["promised_delivery_days"], original_promise)

    entities = order_profiles._deterministic_entities(built)
    assert entities["seller_id"].tolist() == ["seller-z", "seller-a"]
    assert entities["state_od"].tolist() == ["SP -> SP", "MG -> RJ"]
    assert data_pipeline.mode_deterministic(pd.Series(["seller-b", "seller-a"])) == "seller-a"
    assert data_pipeline.mode_deterministic(
        pd.Series(["seller-b", "seller-a", "seller-b"])
    ) == "seller-b"
    assert config["population"]["multi_seller_rule"] == "deterministic_modal_main_seller_then_lexical_tie_break"

    classification_strata = order_experiment.classification_support_strata(
        _paired_predictions()
    )
    composition = classification_strata.loc[
        classification_strata["block"].eq("order_composition")
    ]
    assert set(composition["support_stratum"]) == {"single_seller", "multi_seller"}
    assert composition["order_composition_stratum"].eq(
        composition["support_stratum"]
    ).all()
    assert composition["comparison"].eq("M4-M1").all()

    severity_parts: list[pd.DataFrame] = []
    actual = np.arange(1.0, 13.0)
    for model_id in ("Q2", "Q3", "Q4"):
        severity_parts.append(
            pd.DataFrame(
                {
                    "order_id": [f"severity-{index:02d}" for index in range(12)],
                    "period": "later",
                    "cohort": "2018-01",
                    "family": "linear_quantile",
                    "model_id": model_id,
                    "quantile": 0.9,
                    "actual_positive_late_days": actual,
                    "prediction": np.maximum(actual - 0.5, 0.0),
                    "q1_prediction": np.maximum(actual - 1.0, 0.0),
                    "multi_seller": (np.arange(12) % 2).astype(int),
                    "S1_support": 25,
                    "S2_support": 25,
                    "R1_support": 25,
                    "R2_support": 25,
                    "S1_cold_start": False,
                    "S2_cold_start": False,
                    "R1_cold_start": False,
                    "R2_cold_start": False,
                }
            )
        )
    severity_strata = order_experiment.severity_support_strata(
        pd.concat(severity_parts, ignore_index=True)
    )
    severity_composition = severity_strata.loc[
        severity_strata["block"].eq("order_composition")
    ]
    assert set(severity_composition["support_stratum"]) == {
        "single_seller",
        "multi_seller",
    }
    assert severity_composition["comparison"].eq("Q4-Q1").all()


# ---------------------------------------------------------------------------
# Severity and probability-metric implementations
# ---------------------------------------------------------------------------


def test_severity_evaluation_contains_breach_orders_only_and_generates_q50_q90(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = order_experiment.Cohort(
        "later",
        "2018-01",
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2018-02-01"),
    )
    monkeypatch.setattr(order_experiment, "evaluation_cohorts", lambda: [cohort])
    frame = pd.DataFrame(
        {
            "order_id": ["tp1", "tn1", "tp2", "tn2", "ep1", "en1", "ep2", "en2"],
            "purchase_date": pd.to_datetime(
                [
                    "2017-10-01",
                    "2017-10-02",
                    "2017-11-01",
                    "2017-11-02",
                    "2018-01-05",
                    "2018-01-06",
                    "2018-01-07",
                    "2018-01-08",
                ]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                [
                    "2017-10-10",
                    "2017-10-11",
                    "2017-11-10",
                    "2017-11-11",
                    "2018-01-15",
                    "2018-01-16",
                    "2018-01-17",
                    "2018-01-18",
                ]
            ),
            "positive_late_days": [2.0, 0.0, 5.0, 0.0, 3.0, 0.0, 8.0, 0.0],
        }
    )
    fitted_targets: list[np.ndarray] = []

    class DummyQuantile:
        def __init__(self, quantile: float) -> None:
            self.quantile = quantile

        def predict(self, prediction_frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(prediction_frame), 2.0 + self.quantile)

    def fake_fit_quantile_model(
        training_frame: pd.DataFrame,
        target: Sequence[float],
        numeric: Sequence[str],
        categorical: Sequence[str],
        family: str,
        quantile: float,
        params: Mapping[str, object],
        **_: object,
    ) -> DummyQuantile:
        values = np.asarray(target, dtype=float)
        assert len(training_frame) == len(values)
        assert np.all(values > 0)
        fitted_targets.append(values)
        return DummyQuantile(quantile)

    monkeypatch.setattr(order_modeling, "fit_quantile_model", fake_fit_quantile_model)
    monkeypatch.setattr(order_modeling, "fitted_model_sha256", lambda _: "d" * 64)
    lightweight_config = {
        "severity": {"quantiles": [0.5, 0.9], "families": ["linear_quantile"]}
    }
    selection = {
        "severity_parameters": {
            "linear_quantile": {"0.5": {"alpha": 0.001}, "0.9": {"alpha": 0.001}}
        }
    }
    results, predictions = order_experiment.evaluate_severity(
        frame, lightweight_config, selection
    )
    assert fitted_targets and all(np.all(values > 0) for values in fitted_targets)
    assert set(results["quantile"]) == {0.5, 0.9}
    assert set(results["model_id"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert results["n_orders"].eq(2).all()
    assert predictions["actual_positive_late_days"].gt(0).all()
    assert set(predictions["order_id"]) == {"ep1", "ep2"}


def test_quantile_fit_rejects_nonbreaches_and_generates_nonnegative_predictions() -> None:
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 30),
            "category": ["stable"] * 30,
        }
    )
    positive = 1.0 + 4.0 * frame["x"].to_numpy() + (np.arange(30) % 3) * 0.2
    with pytest.raises(ValueError, match="positive-lateness"):
        order_modeling.fit_quantile_model(
            frame,
            np.where(np.arange(30) == 0, 0.0, positive),
            ["x"],
            ["category"],
            "linear_quantile",
            0.5,
            {"alpha": 0.001, "solver": "highs"},
        )

    predictions: dict[float, np.ndarray] = {}
    fitted_hashes: dict[float, str] = {}
    for quantile in (0.5, 0.9):
        fitted = order_modeling.fit_quantile_model(
            frame,
            positive,
            ["x"],
            ["category"],
            "linear_quantile",
            quantile,
            {"alpha": 0.001, "solver": "highs"},
        )
        assert fitted.quantile == quantile
        fitted_hashes[quantile] = order_modeling.fitted_model_sha256(fitted)
        assert fitted_hashes[quantile] == order_modeling.fitted_model_sha256(fitted)
        assert len(fitted_hashes[quantile]) == 64
        assert set(fitted_hashes[quantile]) <= set("0123456789abcdef")
        predictions[quantile] = fitted.predict(frame.iloc[:7])
        assert predictions[quantile].shape == (7,)
        assert np.isfinite(predictions[quantile]).all()
        assert (predictions[quantile] >= 0).all()
    assert np.median(predictions[0.9]) >= np.median(predictions[0.5])

    repeated_q50 = order_modeling.fit_quantile_model(
        frame,
        positive,
        ["x"],
        ["category"],
        "linear_quantile",
        0.5,
        {"alpha": 0.001, "solver": "highs"},
    )
    assert order_modeling.fitted_model_sha256(repeated_q50) == fitted_hashes[0.5]

    valid_xgboost = {
        "objective": "reg:quantileerror",
        "eval_metric": "quantile",
        "quantile_alpha": 0.9,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": 2,
        "n_jobs": 1,
        "random_state": 20260823,
    }
    _assert_xgboost_quantile_params(valid_xgboost, 0.9)
    fitted_xgboost = order_modeling.fit_quantile_model(
        frame,
        positive,
        ["x"],
        ["category"],
        "xgboost_quantile",
        0.9,
        valid_xgboost,
    )
    xgboost_prediction = fitted_xgboost.predict(frame.iloc[:7])
    assert fitted_xgboost.quantile == 0.9
    assert xgboost_prediction.shape == (7,)
    assert np.isfinite(xgboost_prediction).all()
    assert (xgboost_prediction >= 0).all()

    invalid_variants = (
        ("objective", "binary:logistic", "objective mismatch"),
        ("eval_metric", "logloss", "eval_metric mismatch"),
        ("quantile_alpha", 0.5, "quantile_alpha mismatch"),
    )
    for field, invalid_value, message in invalid_variants:
        invalid = dict(valid_xgboost)
        invalid[field] = invalid_value
        with pytest.raises(ValueError, match=message):
            order_modeling.fit_quantile_model(
                frame,
                positive,
                ["x"],
                ["category"],
                "xgboost_quantile",
                0.9,
                invalid,
            )
        missing = dict(valid_xgboost)
        missing.pop(field)
        with pytest.raises(ValueError, match=message):
            order_modeling.fit_quantile_model(
                frame,
                positive,
                ["x"],
                ["category"],
                "xgboost_quantile",
                0.9,
                missing,
            )


def test_pinball_loss_implementation() -> None:
    target = np.array([1.0, 4.0, 7.0])
    prediction = np.array([2.0, 2.0, 8.0])
    for quantile in (0.5, 0.9):
        residual = target - prediction
        expected = np.mean(
            np.where(residual >= 0, quantile * residual, (quantile - 1.0) * residual)
        )
        assert order_modeling.pinball_loss(target, prediction, quantile) == pytest.approx(expected)


def test_empirical_coverage_calculation() -> None:
    metrics = order_modeling.quantile_metrics(
        target=[1.0, 2.0, 3.0, 4.0],
        prediction=[1.0, 2.0, 2.0, 5.0],
        quantile=0.9,
    )
    assert metrics["empirical_coverage"] == pytest.approx(0.75)
    assert metrics["coverage_error"] == pytest.approx(-0.15)


def test_calibration_intercept_and_slope_implementation() -> None:
    probability = np.repeat([0.2, 0.8], 100)
    target = np.concatenate(
        [
            np.r_[np.ones(20), np.zeros(80)],
            np.r_[np.ones(80), np.zeros(20)],
        ]
    ).astype(int)
    intercept, slope, reason = order_modeling.calibration_intercept_slope(target, probability)
    assert reason == ""
    assert intercept == pytest.approx(0.0, abs=1e-7)
    assert slope == pytest.approx(1.0, abs=1e-7)


def test_brier_and_log_loss_implementation() -> None:
    order_ids = ["a", "b", "c", "d"]
    target = np.array([0, 1, 1, 0])
    probability = np.array([0.1, 0.8, 0.6, 0.3])
    metrics, _ = order_modeling.classification_metrics(order_ids, target, probability, bins=2)
    expected_brier = np.mean((probability - target) ** 2)
    expected_log_loss = -np.mean(
        target * np.log(probability) + (1 - target) * np.log(1 - probability)
    )
    assert metrics["brier"] == pytest.approx(expected_brier)
    assert metrics["log_loss"] == pytest.approx(expected_log_loss)


# ---------------------------------------------------------------------------
# Preservation, scope, reproducibility, and schema controls
# ---------------------------------------------------------------------------


def test_no_prior_protected_output_modification_contract() -> None:
    before = {
        "analysis/old": {"artifact.csv": "a" * 64},
        "docs": {"thesis.tex": "b" * 64},
    }
    ok, detail = order_io.compare_hash_maps(before, copy.deepcopy(before))
    assert ok
    assert all(record["unchanged"] for record in detail.values())

    changed = copy.deepcopy(before)
    changed["analysis/old"]["artifact.csv"] = "c" * 64
    ok, detail = order_io.compare_hash_maps(before, changed)
    assert not ok
    assert detail["analysis/old"]["changed"] == ["artifact.csv"]

    protected = order_io.protected_roots()
    assert "AGENTS.md" in protected
    assert "PROJECT_CONTEXT.md" in protected
    assert "docs" in protected
    assert "report" in protected
    assert all(path.resolve() != WORKSPACE.resolve() for path in protected.values())


def test_no_thesis_or_prior_output_modification_receipt_if_available(
    config: Mapping[str, Any]
) -> None:
    prestate_path = order_preflight.PRESTATE_PATH
    run_state_path = WORKING / "RUN_STATE.json"
    if not prestate_path.is_file() or not run_state_path.is_file():
        pytest.skip("formal preflight/run-state preservation receipts have not been produced")
    prestate = json.loads(prestate_path.read_text(encoding="utf-8"))
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    assert prestate["status"] == "passed"
    protected_hashes = prestate["protected_baseline"]["hashes"]
    assert "docs" in protected_hashes
    assert "report" in protected_hashes
    assert "RESULTS_REGISTRY.md" in protected_hashes
    assert "DECISION_LOG.md" in protected_hashes
    assert run_state.get("events")
    assert config["scope"]["thesis_edit_allowed"] is False
    assert config["scope"]["results_registry_edit_allowed"] is False

    # Re-hash the bounded thesis/control subset directly.  The much larger
    # prior-analysis trees are re-verified by every formal runner stage before
    # its success event is appended; duplicating their multi-GB scan here would
    # make the unit suite needlessly expensive.
    targets = order_preflight._protected_targets()
    for root_name in (
        "docs",
        "report",
        "AGENTS.md",
        "PROJECT_CONTEXT.md",
        "RESULTS_REGISTRY.md",
        "DECISION_LOG.md",
    ):
        current_hashes, _ = order_preflight._path_inventory(targets[root_name])
        assert current_hashes == protected_hashes[root_name], root_name


def test_no_business_policy_simulation(
    config: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert config["scope"]["business_policy_allowed"] is False
    assert "policy" not in order_experiment.model_feature_map()
    assert set(config["classification"]["model_ladder"]) == {
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M4E",
    }
    manifest_path = WORKSPACE / "RUN_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scope = manifest.get("scope", {})
        assert scope.get("business_policy_optimisation_run") is False
        assert scope.get("thesis_rewrite_run") is False
        assert scope.get("prior_registry_edit_run") is False

    # Exercise the real runner-to-reporter finalize call boundary.  All
    # external checks/writes are replaced with bounded in-memory receipts, so
    # this validates the runtime keyword contract without mutating artifacts.
    output = tmp_path / "output"
    work = output / "working"
    output.mkdir()
    work.mkdir()
    test_results = tmp_path / "TEST_RESULTS.txt"
    test_results.write_text("67 passed in synthetic runtime contract\n", encoding="utf-8")
    captured: dict[str, object] = {}
    events: list[tuple[str, dict[str, object]]] = []

    def fake_finalize_reporting(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"overall_pass": True, "blockers": []}

    monkeypatch.setattr(order_runner, "_require_preflight", lambda: {"status": "passed"})
    monkeypatch.setattr(order_runner, "_verify_selection_freeze", lambda _: {})
    monkeypatch.setattr(order_runner.order_preflight, "verify_protected_unchanged", lambda _: {"passed": True})
    monkeypatch.setattr(order_runner.order_io, "OUT", output)
    monkeypatch.setattr(order_runner.order_io, "WORK", work)
    monkeypatch.setattr(
        order_runner.order_io,
        "append_run_event",
        lambda event, **payload: events.append((event, payload)),
    )
    monkeypatch.setattr(order_reporting, "finalize_reporting", fake_finalize_reporting)
    order_runner.run_finalize(test_results)
    assert captured == {
        "output_dir": output,
        "work_dir": work,
        "test_results_path": test_results,
    }
    assert events == [("reporting_complete", {"overall_pass": True})]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(order_runner.__file__)),
            "--stage",
            "finalize",
            "--test-results",
            str(test_results),
        ],
    )
    parsed = order_runner.parse_args()
    assert parsed.stage == "finalize"
    assert parsed.test_results == test_results
    assert parsed.workers == 4


def test_reproducible_hashes_and_deterministic_gzip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["b", "a"],
            "value": [1.25, np.nan],
            "date": pd.to_datetime(["2018-01-02", "2018-01-01"]),
        }
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    order_io.write_gzip_csv(first, frame)
    order_io.write_gzip_csv(second, frame)
    assert first.read_bytes() == second.read_bytes()
    assert _sha256(first) == _sha256(second)
    assert int.from_bytes(first.read_bytes()[4:8], byteorder="little") == 0
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        round_trip = pd.read_csv(handle)
    assert round_trip["order_id"].tolist() == ["b", "a"]

    assert order_modeling.order_id_hash(["b", "a"]) == order_modeling.order_id_hash(
        ["a", "b"]
    )
    payload_a = {"z": 1, "a": [2, 3]}
    payload_b = {"a": [2, 3], "z": 1}
    assert order_modeling.stable_json(payload_a) == order_modeling.stable_json(payload_b)
    assert order_modeling.stable_seed(20260823, "M4", "2018-01") == order_modeling.stable_seed(
        20260823, "M4", "2018-01"
    )
    composite_a = order_modeling.composite_fitted_model_sha256(["a" * 64, "b" * 64])
    composite_b = order_modeling.composite_fitted_model_sha256(["b" * 64, "a" * 64])
    assert composite_a == composite_b
    assert len(composite_a) == 64 and set(composite_a) <= set("0123456789abcdef")
    with pytest.raises(ValueError, match="at least one fitted-model hash"):
        order_modeling.composite_fitted_model_sha256([])


def test_reproducible_model_selection_hashes_if_available() -> None:
    freeze_path = WORKSPACE / "ORDER_MODEL_SELECTION_FREEZE.json"
    parameters_path = WORKSPACE / "ORDER_MODEL_PARAMETERS.csv"
    receipt_path = WORKING / "ORDER_MODEL_FRAME_RECEIPT.json"
    if not (freeze_path.is_file() and parameters_path.is_file() and receipt_path.is_file()):
        pytest.skip("formal model-selection/hash artifacts have not been produced")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    model_frame = WORKING / "ORDER_MODEL_FRAME.csv.gz"
    assert _sha256(model_frame) == receipt["sha256"] == freeze["model_frame_sha256"]
    assert _sha256(WORKSPACE / "ORDER_FROZEN_CONFIG.json") == freeze["config_sha256"]

    parameters = pd.read_csv(parameters_path)
    for value in parameters["parameters_json"]:
        assert order_modeling.stable_json(json.loads(value)) == value
    breach = parameters.loc[parameters["task"].eq("breach")]
    assert breach.groupby("family")["parameters_json"].nunique().eq(1).all()
    severity = parameters.loc[parameters["task"].eq("severity")]
    assert severity.groupby(["family", "quantile"])["parameters_json"].nunique().eq(1).all()


def test_deterministic_metric_result_schemas(config: Mapping[str, Any]) -> None:
    metrics, bins = order_modeling.classification_metrics(
        ["a", "b", "c", "d"], [0, 1, 0, 1], [0.1, 0.8, 0.3, 0.7], bins=2
    )
    assert tuple(metrics) == (
        "n_orders",
        "n_events",
        "prevalence",
        "log_loss",
        "brier",
        "average_precision",
        "roc_auc",
        "top_5pct_lift",
        "top_10pct_lift",
        "top_10pct_recall",
        "calibration_intercept",
        "calibration_slope",
        "wace",
        "calibration_invalid_reason",
        "order_id_sha256",
    )
    assert tuple(bins.columns) == (
        "bin",
        "n",
        "positives",
        "prevalence",
        "mean_probability",
        "min_probability",
        "max_probability",
        "absolute_calibration_error",
        "weight",
    )
    quantile = order_modeling.quantile_metrics([1.0, 2.0], [1.5, 2.5], 0.9)
    assert tuple(quantile) == (
        "n_orders",
        "pinball_loss",
        "empirical_coverage",
        "coverage_error",
        "median_prediction",
        "mean_prediction",
        "median_actual",
        "mean_actual",
        "mean_exceedance",
        "p90_absolute_error",
    )

    # Figure 06 must consume the real aggregate schema emitted by
    # _aggregate_classification_predictions, not a synthetic "later_aggregate"
    # alias that never appears in the persisted calibration bins.
    reliability_rows: list[dict[str, object]] = []
    for model_id in ("M1", "M2", "M3", "M4"):
        for bin_id, (mean_probability, prevalence) in enumerate(
            ((0.03, 0.02), (0.15, 0.18)), start=1
        ):
            reliability_rows.append(
                {
                    "period": "aggregate",
                    "cohort": "later_pooled",
                    "family": "logistic_l2",
                    "model_id": model_id,
                    "probability_type": "calibrated",
                    "bin": bin_id,
                    "n": 100,
                    "prevalence": prevalence,
                    "mean_probability": mean_probability,
                }
            )
    figure_spec = next(spec for spec in order_reporting.FIGURE_SPECS if spec.number == 6)
    figure_source, reason = order_reporting._prepare_figure_source(
        figure_spec,
        {"ORDER_CALIBRATION_BINS.csv": pd.DataFrame(reliability_rows)},
        config,
    )
    assert reason is None
    assert not figure_source.empty
    assert set(figure_source["model_id"]) == {"M1", "M2", "M3", "M4"}
    assert figure_source["period"].eq("aggregate").all()
    assert figure_source["cohort_month"].eq("later_pooled").all()
    assert figure_source["probability_variant"].eq("calibrated").all()
    assert {
        "mean_predicted_probability", "observed_rate", "n_orders"
    }.issubset(figure_source.columns)

    persisted_figure_source = WORKSPACE / "figure_sources/06_reliability.csv"
    if persisted_figure_source.is_file():
        persisted = pd.read_csv(persisted_figure_source)
        assert not persisted.empty
        assert set(persisted["model_id"]) == {"M1", "M2", "M3", "M4"}
        assert persisted["period"].eq("aggregate").all()
        assert persisted["cohort_month"].eq("later_pooled").all()
        assert persisted["probability_variant"].eq("calibrated").all()


ARTIFACT_SCHEMA_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ORDER_SAMPLE_AUDIT.csv": (
        "sample",
        "n_orders",
        "n_unique_orders",
        "n_breaches",
        "n_positive_severity",
        "unresolved_target_rows",
        "duplicate_order_ids",
    ),
    "ORDER_PROFILE_JOIN_AUDIT.csv": (
        "block",
        "candidate_id",
        "profile_spec_id",
        "orders_seen",
        "orders_mapped_cold_start",
        "orders_missing_mapping",
        "seen_history_time_violations",
    ),
    "ORDER_DEVELOPMENT_TUNING.csv": (
        "task",
        "family",
        "model_id",
        "parameter_index",
        "parameters_json",
        "fold",
        "selected",
    ),
    "ORDER_MODEL_PARAMETERS.csv": (
        "task",
        "family",
        "model_id",
        "quantile",
        "parameters_json",
        "development_only",
    ),
    "ORDER_BREACH_RESULTS.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "probability_type",
        "log_loss",
        "brier",
        "average_precision",
        "roc_auc",
        "order_id_sha256",
        "fitted_model_sha256",
        "model_hash_type",
    ),
    "ORDER_BREACH_BY_MONTH.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "probability_type",
        "log_loss",
        "brier",
        "fitted_model_sha256",
        "model_hash_type",
    ),
    "ORDER_BREACH_PAIRED_DIFFERENCES.csv": (
        "period",
        "cohort",
        "family",
        "comparison",
        "candidate_model",
        "reference_model",
        "paired_order_id_sha256",
        "delta_log_loss",
        "delta_brier",
    ),
    "ORDER_CALIBRATION_RESULTS.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "calibration_intercept",
        "calibration_slope",
        "wace",
        "fitted_model_sha256",
        "model_hash_type",
    ),
    "ORDER_CALIBRATION_BINS.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "probability_type",
        "bin",
        "n",
        "prevalence",
        "mean_probability",
    ),
    "ORDER_PROFILE_ABLATIONS.csv": (
        "period",
        "cohort",
        "family",
        "ablation_id",
        "representation",
        "log_loss",
        "brier",
        "fitted_model_sha256",
        "model_hash_type",
        "reference_fitted_model_sha256",
    ),
    "ORDER_PROFILE_SUPPORT_STRATA.csv": (
        "block",
        "period",
        "cohort",
        "family",
        "support_stratum",
        "log_loss",
        "brier",
    ),
    "ORDER_EVENT_STRATA.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "event_stratum",
        "predictor_used",
        "prevalence",
        "average_precision",
    ),
    "ORDER_TERMINAL_STRESS.csv": (
        "analysis",
        "period",
        "cohort",
        "family",
        "model_id",
        "probability_type",
        "comparison",
        "metric",
        "estimate",
        "interpretation",
    ),
    "SEVERITY_RESULTS.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "quantile",
        "pinball_loss",
        "empirical_coverage",
        "skill_vs_q1",
        "order_id_sha256",
        "fitted_model_sha256",
        "model_hash_type",
    ),
    "SEVERITY_BY_MONTH.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "quantile",
        "pinball_loss",
        "empirical_coverage",
        "fitted_model_sha256",
        "model_hash_type",
    ),
    "SEVERITY_PINBALL_SKILL.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "quantile",
        "pinball_loss",
        "skill_vs_unconditional",
        "skill_vs_q1",
    ),
    "SEVERITY_COVERAGE.csv": (
        "period",
        "cohort",
        "family",
        "model_id",
        "quantile",
        "empirical_coverage",
        "coverage_error",
    ),
    "SEVERITY_PROFILE_ABLATIONS.csv": (
        "period",
        "cohort",
        "family",
        "quantile",
        "ablation_id",
        "representation",
        "pinball_loss",
        "skill_vs_q1",
        "fitted_model_sha256",
        "model_hash_type",
        "reference_fitted_model_sha256",
    ),
    "SEVERITY_SUPPORT_STRATA.csv": (
        "block",
        "period",
        "cohort",
        "family",
        "quantile",
        "support_stratum",
        "pinball_loss",
        "skill_vs_q1",
    ),
    "MODEL_COMPARISON_SUMMARY.csv": (
        "task",
        "family",
        "comparison",
        "quantile",
        "evidence_status",
        "evidence_reason",
    ),
}


@pytest.mark.parametrize("filename", sorted(ARTIFACT_SCHEMA_REQUIREMENTS))
def test_persisted_result_schema_if_available(filename: str) -> None:
    path = WORKSPACE / filename
    if not path.is_file():
        pytest.skip(f"formal artifact has not been produced: {filename}")
    frame = pd.read_csv(path, nrows=10, low_memory=False)
    assert len(frame.columns) == len(set(frame.columns))
    missing = set(ARTIFACT_SCHEMA_REQUIREMENTS[filename]) - set(frame.columns)
    assert not missing, f"{filename} missing deterministic schema fields: {sorted(missing)}"

    hash_columns = [
        column
        for column in ("fitted_model_sha256", "reference_fitted_model_sha256")
        if column in frame.columns
    ]
    if hash_columns:
        hashes = pd.read_csv(path, usecols=hash_columns, dtype=str, keep_default_na=False)
        for column in hash_columns:
            present = hashes[column].loc[hashes[column].ne("")]
            assert not present.empty
            assert present.str.fullmatch(r"[0-9a-f]{64}").all(), f"{filename}:{column}"
        if "model_hash_type" in frame.columns:
            hash_types = pd.read_csv(path, usecols=["model_hash_type"], dtype=str)[
                "model_hash_type"
            ]
            assert set(hash_types.dropna()) <= {"fitted_model", "composite_fitted_models"}

    if filename == "ORDER_TERMINAL_STRESS.csv":
        terminal = pd.read_csv(path, low_memory=False)
        breach_rows = terminal.loc[terminal["analysis"].eq("breach_model")]
        assert set(breach_rows["probability_type"]) == {"raw", "calibrated"}
        shifts = terminal.loc[terminal["analysis"].eq("breach_calibration_shift")]
        assert not shifts.empty
        assert shifts["comparison"].eq("terminal_minus_later_pooled").all()
        assert set(shifts["metric"]) == {
            "terminal_minus_later_calibration_intercept",
            "terminal_minus_later_calibration_slope",
            "terminal_minus_later_wace",
        }

    if filename == "ORDER_BREACH_RESULTS.csv":
        import pyarrow.parquet as pq

        storage = json.loads(
            (WORKSPACE / "ORDER_FROZEN_CONFIG.json").read_text(encoding="utf-8")
        )["storage"]
        assert storage == {
            "row_predictions_format": "parquet",
            "parquet_engine": "pyarrow",
            "parquet_compression": "zstd",
            "parquet_index": False,
            "classification_sort_keys": [
                "period", "cohort", "family", "model_id", "order_id"
            ],
            "severity_sort_keys": [
                "period", "cohort", "family", "model_id", "quantile", "order_id"
            ],
        }
        parquet_specs = {
            "ORDER_BREACH_ROW_PREDICTIONS.parquet": (
                "breach_prediction_output",
                {"order_id", "period", "cohort", "family", "model_id", "fitted_model_sha256"},
            ),
            "SEVERITY_ROW_PREDICTIONS.parquet": (
                "severity_prediction_output",
                {
                    "order_id", "period", "cohort", "family", "model_id", "quantile",
                    "fitted_model_sha256",
                },
            ),
        }
        run_state = json.loads((WORKING / "RUN_STATE.json").read_text(encoding="utf-8"))
        evaluation_events = [
            event
            for event in run_state["events"]
            if event.get("event") == "later_and_terminal_evaluation_complete"
        ]
        assert evaluation_events
        receipt_event = evaluation_events[-1]
        for parquet_name, (receipt_key, required_columns) in parquet_specs.items():
            parquet_path = WORKSPACE / parquet_name
            assert parquet_path.is_file()
            parquet_file = pq.ParquetFile(parquet_path)
            assert parquet_file.metadata.num_rows > 0
            assert required_columns.issubset(parquet_file.schema_arrow.names)
            assert "__index_level_0__" not in parquet_file.schema_arrow.names
            for row_group in range(parquet_file.metadata.num_row_groups):
                metadata = parquet_file.metadata.row_group(row_group)
                assert all(
                    metadata.column(column).compression == "ZSTD"
                    for column in range(metadata.num_columns)
                )
            pandas_metadata = json.loads(parquet_file.schema_arrow.metadata[b"pandas"])
            assert pandas_metadata["index_columns"] == []
            receipt = receipt_event[receipt_key]
            assert receipt["path"] == parquet_name
            assert int(receipt["rows"]) == parquet_file.metadata.num_rows
            assert receipt["sha256"] == _sha256(parquet_path)


def test_manifest_output_hashes_if_available() -> None:
    path = WORKSPACE / "RUN_MANIFEST.json"
    if not path.is_file():
        pytest.skip("formal RUN_MANIFEST.json has not been produced")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["analysis_id"] == "order_breach_severity_v1"
    outputs = manifest.get("outputs")
    assert isinstance(outputs, dict) and outputs
    for relative, receipt in outputs.items():
        output_path = WORKSPACE / relative
        assert receipt["exists"] is True
        assert output_path.is_file(), relative
        assert int(receipt["bytes"]) == output_path.stat().st_size
        # A JSON file cannot contain its own final digest.  All other outputs
        # are immutable when the inventory is written and must match exactly.
        if relative == "RUN_MANIFEST.json":
            assert receipt["sha256"] is None
            assert receipt["hash_omitted_reason"] == "self_referential_manifest_hash"
        else:
            assert receipt["sha256"] == _sha256(output_path), relative
