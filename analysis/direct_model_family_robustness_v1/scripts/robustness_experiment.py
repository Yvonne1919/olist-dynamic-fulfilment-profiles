"""Frozen model-family robustness experiment for the direct-promise ladder."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from analysis.direct_promise_profile_extension_v1.scripts import direct_experiment as direct
from analysis.order_breach_severity_v1.scripts import order_modeling
from src.models.classification import estimators as historical_estimators
from src.models.classification import pipeline as historical_pipeline

from .recovered_severity_model_source import (
    SEED as RECOVERED_SEVERITY_SEED,
    lognormal_quantiles,
    make_lognormal_ridge,
    make_quantile_forest,
    weighted_quantile,
)

from .robustness_integrity import (
    ROOT,
    WORKSPACE,
    load_config,
    read_json,
    sha256_file,
    stable_json,
    utc_now,
    write_json,
)


sys_dont_write_bytecode_guard = True
PRIMARY_ROOT = ROOT / "analysis/direct_promise_profile_extension_v1"
FLOAT_FORMAT = "%.12g"
PRIMARY_SELECTION_PATH = PRIMARY_ROOT / "DIRECT_MODEL_SELECTION_FREEZE.json"

FAMILY_DISPLAY = {
    "logistic_l2": "L2 Logistic Regression",
    "spline_logistic": "Spline Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "linear_quantile": "Linear Quantile Regression",
    "random_forest_leaf_weighted_quantile": "Leaf-weighted Quantile Random Forest",
    "xgboost_quantile": "XGBoost Quantile Regression",
    "lognormal_ridge": "Lognormal Ridge",
}
PROFILE_BLOCK = {
    "DPS-DP0": "seller",
    "DPG-DP0": "state_od",
    "DPB-DP0": "both",
    "DQS-DQ0": "seller",
    "DQG-DQ0": "state_od",
    "DQB-DQ0": "both",
}


def _feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, features)) + "\n").encode()).hexdigest()


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "record_type", "task", "stage", "period", "cohort", "family", "model_id",
        "quantile", "representation", "probability_type", "comparison",
        "support_stratum", "fold", "parameter_index", "bin", "order_id", "metric",
    ]
    keys = [column for column in preferred if column in frame.columns]
    if not keys or frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".gz" if path.suffix == ".gz" else ".csv"
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        output = _sort_frame(frame)
        if path.suffix == ".gz":
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="", write_through=True) as text:
                        output.to_csv(text, index=False, float_format=FLOAT_FORMAT, date_format="%Y-%m-%d", na_rep="")
        else:
            output.to_csv(temporary, index=False, float_format=FLOAT_FORMAT, date_format="%Y-%m-%d", na_rep="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    buffer = io.StringIO()
    _sort_frame(frame).to_csv(
        buffer, index=False, float_format=FLOAT_FORMAT, date_format="%Y-%m-%d", na_rep=""
    )
    buffer.seek(0)
    return pd.read_csv(buffer, low_memory=False)


def _numeric_compatible(expected: pd.Series, actual: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(expected) or pd.api.types.is_numeric_dtype(actual):
        return True
    expected_nonempty = expected.dropna()
    actual_nonempty = actual.dropna()
    if expected_nonempty.empty and actual_nonempty.empty:
        return False
    expected_num = pd.to_numeric(expected_nonempty, errors="coerce")
    actual_num = pd.to_numeric(actual_nonempty, errors="coerce")
    return expected_num.notna().all() and actual_num.notna().all()


def compare_table(
    expected_path: Path,
    actual_frame: pd.DataFrame,
    *,
    atol: float,
    ignored_columns: Sequence[str] = (),
) -> dict[str, Any]:
    expected = pd.read_csv(expected_path, low_memory=False)
    actual = _canonical_roundtrip(actual_frame)
    expected = _sort_frame(expected)
    actual = _sort_frame(actual)
    schema_equal = list(expected.columns) == list(actual.columns)
    row_count_equal = len(expected) == len(actual)
    result: dict[str, Any] = {
        "expected_path": expected_path.relative_to(ROOT).as_posix(),
        "expected_sha256": sha256_file(expected_path),
        "schema_equal": schema_equal,
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "row_count_equal": row_count_equal,
        "ignored_columns": list(ignored_columns),
        "numeric_differing_cells": 0,
        "nonnumeric_differing_cells": 0,
        "max_absolute_difference": 0.0,
        "ignored_column_exact_match": True,
    }
    if not schema_equal or not row_count_equal:
        result["passed"] = False
        return result
    ignored = set(ignored_columns)
    for column in expected.columns:
        if column in ignored:
            left = expected[column].fillna("").astype(str)
            right = actual[column].fillna("").astype(str)
            result["ignored_column_exact_match"] = bool(
                result["ignored_column_exact_match"] and left.equals(right)
            )
            continue
        if _numeric_compatible(expected[column], actual[column]):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            valid = np.isclose(left, right, rtol=0.0, atol=atol, equal_nan=True)
            result["numeric_differing_cells"] += int((~valid).sum())
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                result["max_absolute_difference"] = max(
                    float(result["max_absolute_difference"]),
                    float(np.max(np.abs(left[finite] - right[finite]))),
                )
        else:
            left = expected[column].fillna("").astype(str)
            right = actual[column].fillna("").astype(str)
            result["nonnumeric_differing_cells"] += int((left != right).sum())
    result["passed"] = bool(
        result["numeric_differing_cells"] == 0
        and result["nonnumeric_differing_cells"] == 0
    )
    return result


def _core_reproduction_rows(expected: pd.DataFrame, actual: pd.DataFrame, atol: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["task", "family", "comparison", "quantile"]
    for _, old in expected.iterrows():
        mask = pd.Series(True, index=actual.index)
        for key in keys:
            if key not in actual or key not in old:
                continue
            if pd.isna(old[key]):
                mask &= actual[key].isna()
            elif key == "quantile":
                mask &= pd.to_numeric(actual[key], errors="coerce").eq(float(old[key]))
            else:
                mask &= actual[key].astype(str).eq(str(old[key]))
        match = actual.loc[mask]
        task = str(old["task"])
        metrics = (
            ["median_delta_log_loss", "median_delta_brier", "both_improved_month_count"]
            if task == "breach"
            else [
                "median_skill", "favourable_month_count",
                "median_absolute_coverage_error_deterioration",
            ]
        )
        passed = len(match) == 1
        detail: dict[str, Any] = {}
        if len(match) == 1:
            new = match.iloc[0]
            for metric in metrics:
                old_value = pd.to_numeric(pd.Series([old.get(metric)]), errors="coerce").iloc[0]
                new_value = pd.to_numeric(pd.Series([new.get(metric)]), errors="coerce").iloc[0]
                metric_pass = bool(
                    (pd.isna(old_value) and pd.isna(new_value))
                    or (pd.notna(old_value) and pd.notna(new_value) and abs(float(old_value) - float(new_value)) <= atol)
                )
                passed = passed and metric_pass
                detail[f"expected_{metric}"] = old_value
                detail[f"reproduced_{metric}"] = new_value
            expected_label = str(old.get("evidence_label", ""))
            actual_label = str(new.get("evidence_label", ""))
            passed = passed and expected_label == actual_label
        else:
            expected_label = str(old.get("evidence_label", ""))
            actual_label = ""
        rows.append(
            {
                "record_type": "core_comparison",
                "task": task,
                "family": old["family"],
                "comparison": old["comparison"],
                "quantile": old.get("quantile"),
                "expected_label": expected_label,
                "reproduced_label": actual_label,
                "matching_rows": len(match),
                "absolute_tolerance": atol,
                "passed": passed,
                **detail,
            }
        )
    return rows


def reproduce_primary(frame: pd.DataFrame, direct_config: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    selection, development = direct.run_development_selection(frame, direct_config)
    expected_selection = read_json(PRIMARY_SELECTION_PATH)
    selection_exact = stable_json(selection) == stable_json(expected_selection)
    primary = direct.evaluate_direct_extension(frame, selection, direct_config)
    expected_labels = pd.read_csv(PRIMARY_ROOT / "EVIDENCE_LABELS.csv", low_memory=False)
    audit_rows = _core_reproduction_rows(
        expected_labels, primary["evidence_labels"], float(config["reproduction_gate"]["absolute_tolerance"])
    )
    audit_rows.append(
        {
            "record_type": "selection_freeze",
            "task": "all_primary",
            "family": "all_primary",
            "comparison": "selection_json",
            "absolute_tolerance": 0.0,
            "passed": selection_exact,
            "expected_sha256": sha256_file(PRIMARY_SELECTION_PATH),
            "reproduced_object_sha256": hashlib.sha256(stable_json(selection).encode()).hexdigest(),
            "expected_object_sha256": hashlib.sha256(stable_json(expected_selection).encode()).hexdigest(),
        }
    )
    table_map = {
        "model_selection": development["model_selection"],
        "breach_monthly": primary["breach_monthly"],
        "breach_pooled": primary["breach_pooled"],
        "breach_calibration": primary["breach_calibration"],
        "severity_monthly": primary["severity_monthly"],
        "severity_pooled": primary["severity_pooled"],
        "severity_coverage": primary["severity_coverage"],
        "terminal": primary["terminal"],
        "evidence_labels": primary["evidence_labels"],
    }
    for key, actual in table_map.items():
        comparison = compare_table(
            PRIMARY_ROOT / str(config["direct_result_files"][key]),
            actual,
            atol=float(config["reproduction_gate"]["absolute_tolerance"]),
            ignored_columns=config["reproduction_gate"]["non_result_hash_columns"],
        )
        audit_rows.append(
            {
                "record_type": "artifact_table",
                "task": "primary_artifact",
                "family": "all_primary",
                "comparison": key,
                "absolute_tolerance": config["reproduction_gate"]["absolute_tolerance"],
                **comparison,
            }
        )
    audit = pd.DataFrame(audit_rows)
    _atomic_csv(audit, WORKSPACE / "PRIMARY_RESULT_REPRODUCTION_AUDIT.csv")
    if not audit["passed"].fillna(False).astype(bool).all():
        raise RuntimeError("primary-result reproduction gate failed; alternatives were not evaluated")
    return selection, {**development, **primary}, audit


@dataclass
class HistoricalRFFitted:
    family: str
    pipeline: object
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.pipeline.predict_proba(frame)[:, 1], dtype=float)
        return np.clip(probability, order_modeling.EPS, 1 - order_modeling.EPS)


def fit_historical_rf(frame: pd.DataFrame, target: Sequence[int], numeric: Sequence[str], categorical: Sequence[str]) -> HistoricalRFFitted:
    estimator = clone(historical_estimators(42, include_xgboost=False)["random_forest"])
    fitted = historical_pipeline(list(numeric), list(categorical), estimator)
    fitted.fit(frame, np.asarray(target, dtype=int))
    return HistoricalRFFitted("random_forest", fitted, tuple(numeric), tuple(categorical))


def validate_frozen_contract(config: Mapping[str, Any], direct_config: Mapping[str, Any]) -> None:
    for block in ("S1", "S2", "R1", "R2"):
        if config["profiles"][block] != direct_config["profiles"][block]:
            raise AssertionError(f"fixed profile definition drifted for {block}")
    if config["profiles"]["payload_suffixes"] != direct_config["profiles"]["payload_suffixes"]:
        raise AssertionError("fixed profile payload order drifted")
    if config["periods"]["development_inner_folds"] != direct_config["periods"]["development_inner_folds"]:
        raise AssertionError("development fold drift")
    if config["periods"]["later_months"] != direct_config["periods"]["later_months"]:
        raise AssertionError("later-month drift")
    if config["periods"]["terminal"] != direct_config["periods"]["terminal"]:
        raise AssertionError("terminal-period drift")
    if config["families"]["breach_primary"] != direct_config["breach"]["families"]:
        raise AssertionError("primary breach family drift")
    if config["families"]["severity_primary"] != direct_config["severity"]["families"]:
        raise AssertionError("primary severity family drift")

    rf_config = config["alternative_parameters"]["random_forest"]
    rf = historical_estimators(42, include_xgboost=False)["random_forest"].get_params()
    rf_expected = {
        "n_estimators": rf_config["n_estimators"],
        "min_samples_leaf": rf_config["min_samples_leaf"],
        "max_features": rf_config["max_features"],
        "class_weight": rf_config["class_weight"],
        "random_state": rf_config["random_state"],
        "n_jobs": rf_config["n_jobs"],
    }
    if any(rf[key] != value for key, value in rf_expected.items()):
        raise AssertionError("recovered breach RF source/config mismatch")

    qrf_config = config["alternative_parameters"]["random_forest_leaf_weighted_quantile"]
    qrf = make_quantile_forest().get_params()
    qrf_expected = {
        "n_estimators": qrf_config["n_estimators"],
        "min_samples_leaf": qrf_config["min_samples_leaf"],
        "max_features": qrf_config["max_features"],
        "random_state": qrf_config["random_state"],
        "n_jobs": qrf_config["n_jobs"],
    }
    if any(qrf[key] != value for key, value in qrf_expected.items()):
        raise AssertionError("recovered severity RF source/config mismatch")
    if RECOVERED_SEVERITY_SEED != int(qrf_config["random_state"]):
        raise AssertionError("recovered severity seed mismatch")

    ridge_config = config["alternative_parameters"]["lognormal_ridge"]
    ridge = make_lognormal_ridge().get_params()
    if ridge["alpha"] != ridge_config["alpha"] or ridge["solver"] != ridge_config["solver"]:
        raise AssertionError("recovered lognormal Ridge source/config mismatch")


def _model_manifest(
    *, task: str, stage: str, family: str, model_id: str, representation: str,
    fitted: object, parameters: Mapping[str, Any], numeric: Sequence[str], categorical: Sequence[str],
    cohort: str, origin: object, fold: int | None, quantile: float | None,
    train_ids: Iterable[object], evaluation_ids: Iterable[object],
) -> dict[str, Any]:
    return {
        "task": task, "stage": stage, "family": family, "model_family": family,
        "model_id": model_id, "specification": model_id, "representation": representation,
        "quantile": quantile, "cohort": cohort, "origin": origin, "fold": fold,
        "n_train": len(list(train_ids)) if not isinstance(train_ids, pd.Series) else len(train_ids),
        "n_evaluation": len(list(evaluation_ids)) if not isinstance(evaluation_ids, pd.Series) else len(evaluation_ids),
        "train_order_id_sha256": order_modeling.order_id_hash(train_ids),
        "evaluation_order_id_sha256": order_modeling.order_id_hash(evaluation_ids),
        "parameters_json": order_modeling.stable_json(dict(parameters)),
        "numeric_features_json": order_modeling.stable_json(list(numeric)),
        "categorical_features_json": order_modeling.stable_json(list(categorical)),
        "ordered_feature_sha256": _feature_hash(list(numeric) + list(categorical)),
        "fitted_model_sha256": order_modeling.fitted_model_sha256(fitted),
        "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
    }


def tune_and_calibrate_rf(frame: pd.DataFrame, direct_config: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric, categorical = direct.breach_feature_map("full")["DP0"]
    params = dict(config["alternative_parameters"]["random_forest"])
    tuning_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for fold in direct._normalised_folds(direct_config):
        train, valid = direct.chronological_masks(frame, fold)
        fitted = fit_historical_rf(frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical)
        probability = fitted.predict_raw(frame.loc[valid])
        metrics, _ = order_modeling.classification_metrics(
            frame.loc[valid, "order_id"], frame.loc[valid, "late_delivery"], probability,
            int(direct_config["breach"]["calibration_bins"]),
        )
        tuning_rows.append({
            "record_type": "development_tuning", "task": "breach", "stage": "development_tuning",
            "family": "random_forest", "model_family": "random_forest", "model_id": "DP0",
            "specification": "DP0", "quantile": np.nan, "parameter_index": 0,
            "parameters_json": order_modeling.stable_json(params), "fold": int(fold["fold"]),
            "n_train": int(train.sum()), "n_validation": int(valid.sum()),
            "train_order_id_sha256": order_modeling.order_id_hash(frame.loc[train, "order_id"]),
            "validation_order_id_sha256": order_modeling.order_id_hash(frame.loc[valid, "order_id"]),
            "log_loss": metrics["log_loss"], "brier": metrics["brier"], "brier_score": metrics["brier"],
            "selected": True, "grid_type": "recovered_singleton", "development_only": True,
            "later_or_terminal_outcomes_used": False,
        })
        manifests.append(_model_manifest(
            task="breach", stage="development_tuning", family="random_forest", model_id="DP0",
            representation="full", fitted=fitted, parameters=params, numeric=numeric, categorical=categorical,
            cohort=f"development_fold_{fold['fold']}", origin=fold["validation_start"], fold=int(fold["fold"]),
            quantile=None, train_ids=frame.loc[train, "order_id"], evaluation_ids=frame.loc[valid, "order_id"],
        ))
    calibrators: dict[str, dict[str, Any]] = {}
    oof_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    for model_id, (model_numeric, model_categorical) in direct.breach_feature_map("full").items():
        parts = []
        for fold in direct._normalised_folds(direct_config):
            train, valid = direct.chronological_masks(frame, fold)
            fitted = fit_historical_rf(
                frame.loc[train], frame.loc[train, "late_delivery"], model_numeric, model_categorical
            )
            probability = fitted.predict_raw(frame.loc[valid])
            part = pd.DataFrame({
                "order_id": frame.loc[valid, "order_id"].astype(str).to_numpy(),
                "purchase_date": frame.loc[valid, "purchase_date"].to_numpy(),
                "fold": int(fold["fold"]), "target": frame.loc[valid, "late_delivery"].to_numpy(int),
                "raw_probability": probability, "family": "random_forest", "model_id": model_id,
            })
            parts.append(part)
            manifests.append(_model_manifest(
                task="breach", stage="development_calibration_oof", family="random_forest", model_id=model_id,
                representation="full", fitted=fitted, parameters=params, numeric=model_numeric,
                categorical=model_categorical, cohort=f"development_fold_{fold['fold']}",
                origin=fold["validation_start"], fold=int(fold["fold"]), quantile=None,
                train_ids=frame.loc[train, "order_id"], evaluation_ids=frame.loc[valid, "order_id"],
            ))
        oof = pd.concat(parts, ignore_index=True)
        calibrator, audit = order_modeling.select_calibration_method(oof)
        oof["calibrated_probability"] = calibrator.predict(oof["raw_probability"])
        calibrators[model_id] = calibrator.as_dict()
        audit.insert(0, "representation", "full")
        audit.insert(0, "specification", model_id)
        audit.insert(0, "model_id", model_id)
        audit.insert(0, "model_family", "random_forest")
        audit.insert(0, "family", "random_forest")
        audit["calibrator_parameters_json"] = order_modeling.stable_json(calibrator.as_dict())
        audit["oof_n_orders"] = len(oof)
        audit["oof_order_id_sha256"] = order_modeling.order_id_hash(oof["order_id"])
        calibration_parts.append(audit)
        oof_parts.append(oof)
    selection_row = pd.DataFrame([{
        "record_type": "selected_specification", "task": "breach", "family": "random_forest",
        "model_family": "random_forest", "model_id": "DP0", "specification": "DP0", "quantile": np.nan,
        "tuning_baseline": "DP0", "selection_metric": "mean_chronological_log_loss_then_brier",
        "parameters_json": order_modeling.stable_json(params), "grid_type": "recovered_singleton",
        "development_only": True, "later_or_terminal_outcomes_used": False,
    }])
    return (
        {"parameters": params, "calibrators": calibrators},
        pd.DataFrame(tuning_rows), selection_row,
        pd.concat(calibration_parts, ignore_index=True), pd.DataFrame(manifests),
        pd.concat(oof_parts, ignore_index=True),
    )


def evaluate_rf_breach(frame: pd.DataFrame, direct_config: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    params = selected["parameters"]
    metric_rows: list[dict[str, Any]] = []
    reliability_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for cohort in direct.evaluation_cohorts(direct_config):
        train, test = direct.cohort_masks(frame, cohort, direct_config)
        test_hash = order_modeling.order_id_hash(frame.loc[test, "order_id"])
        for model_id, (numeric, categorical) in direct.breach_feature_map("full").items():
            fitted = fit_historical_rf(frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical)
            raw = fitted.predict_raw(frame.loc[test])
            payload = selected["calibrators"][model_id]
            calibrator = direct._calibrator_from_dict(payload)
            calibrated = calibrator.predict(raw)
            model_hash = order_modeling.fitted_model_sha256(fitted)
            prediction = pd.DataFrame({
                "order_id": frame.loc[test, "order_id"].astype(str).to_numpy(),
                "purchase_date": frame.loc[test, "purchase_date"].to_numpy(),
                "period": cohort.period, "cohort": cohort.cohort, "cohort_month": cohort.cohort,
                "origin": cohort.origin, "family": "random_forest", "model_family": "random_forest",
                "model_id": model_id, "specification": model_id, "model_name": direct.BREACH_NAMES[model_id],
                "representation": "full", "target": frame.loc[test, "late_delivery"].to_numpy(int),
                "raw_probability": raw, "calibrated_probability": calibrated,
                "calibration_method": calibrator.method, "fitted_model_sha256": model_hash,
                "order_id_sha256": test_hash,
            })
            for block in direct.PROFILE_BLOCKS:
                for suffix in ("support", "cold_start", "mapping_status", "score"):
                    prediction[f"{block}_{suffix}"] = frame.loc[test, f"{block}_{suffix}"].to_numpy()
            prediction_parts.append(prediction)
            for probability_type, probability in (("raw", raw), ("calibrated", calibrated)):
                metrics, reliability = order_modeling.classification_metrics(
                    prediction["order_id"], prediction["target"], probability,
                    int(direct_config["breach"]["calibration_bins"]),
                )
                metric_rows.append(direct._metric_aliases({
                    "period": cohort.period, "cohort": cohort.cohort, "cohort_month": cohort.cohort,
                    "origin": cohort.origin, "family": "random_forest", "model_id": model_id,
                    "model_name": direct.BREACH_NAMES[model_id], "representation": "full",
                    "probability_type": probability_type, "probability_variant": probability_type,
                    "calibration_method": calibrator.method if probability_type == "calibrated" else "none",
                    "n_train": int(train.sum()),
                    "train_order_id_sha256": order_modeling.order_id_hash(frame.loc[train, "order_id"]),
                    "fitted_model_sha256": model_hash, "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                    **metrics,
                }))
                reliability.insert(0, "probability_type", probability_type)
                reliability.insert(0, "representation", "full")
                reliability.insert(0, "model_id", model_id)
                reliability.insert(0, "family", "random_forest")
                reliability.insert(0, "cohort", cohort.cohort)
                reliability.insert(0, "period", cohort.period)
                reliability["fitted_model_sha256"] = model_hash
                reliability["order_id_sha256"] = test_hash
                reliability_parts.append(reliability)
            manifests.append(_model_manifest(
                task="breach", stage="later_evaluation" if cohort.period == "later" else "terminal_stress",
                family="random_forest", model_id=model_id, representation="full", fitted=fitted,
                parameters=params, numeric=numeric, categorical=categorical, cohort=cohort.cohort,
                origin=cohort.origin, fold=None, quantile=None, train_ids=frame.loc[train, "order_id"],
                evaluation_ids=frame.loc[test, "order_id"],
            ))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    pooled, pooled_bins = direct._aggregate_breach_predictions(
        predictions, period="later", bins=int(direct_config["breach"]["calibration_bins"])
    )
    pairs = direct._paired_breach_differences(predictions, direct_config)
    ablations, ablation_manifests = evaluate_rf_ablations(
        frame, direct_config, params, metrics, predictions
    )
    support = direct._breach_support_strata(predictions)
    family_config = json.loads(json.dumps(direct_config))
    family_config["breach"]["families"] = ["random_forest"]
    summary = direct._breach_evidence_summary(pairs, metrics, support, ablations, family_config)
    support = support.merge(
        summary,
        on=["family", "comparison", "candidate_model", "reference_model"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )
    metrics = direct._join_breach_evidence(metrics, summary, pairs)
    pooled = direct._join_breach_evidence(pooled, summary, pairs)
    pairs = pairs.merge(
        summary, on=["family", "comparison", "candidate_model", "reference_model"],
        how="left", validate="many_to_one", suffixes=("", "_evidence"),
    )
    reliability = pd.concat([*reliability_parts, pooled_bins], ignore_index=True, sort=False)
    return {
        "metrics": metrics, "pooled": pooled, "predictions": predictions, "pairs": pairs,
        "ablations": ablations, "support": support, "summary": summary,
        "reliability": reliability,
        "manifests": pd.concat([pd.DataFrame(manifests), ablation_manifests], ignore_index=True, sort=False),
    }


def evaluate_rf_ablations(
    frame: pd.DataFrame, direct_config: Mapping[str, Any], params: Mapping[str, Any],
    primary_metrics: pd.DataFrame, primary_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    comparisons = {candidate: comparison for comparison, candidate, _ in direct.BREACH_COMPARISONS}
    raw = primary_metrics.loc[primary_metrics["probability_type"].eq("raw")]
    for _, candidate in raw.loc[raw["model_id"].ne("DP0")].iterrows():
        reference = raw.loc[
            raw["period"].eq(candidate["period"]) & raw["cohort"].eq(candidate["cohort"])
            & raw["model_id"].eq("DP0")
        ]
        if len(reference) != 1:
            raise AssertionError("RF full ablation lacks unique DP0 reference")
        ref = reference.iloc[0]
        rows.append(direct._metric_aliases({
            **candidate.to_dict(), "comparison": comparisons[str(candidate["model_id"])],
            "candidate_model": candidate["model_id"], "reference_model": "DP0",
            "ablation_id": f"{candidate['model_id']}_full", "representation": "full",
            "reference_fitted_model_sha256": ref["fitted_model_sha256"],
            "delta_log_loss": candidate["log_loss"] - ref["log_loss"],
            "delta_brier": candidate["brier"] - ref["brier"],
            "delta_brier_score": candidate["brier"] - ref["brier"],
        }))
    for cohort in direct.evaluation_cohorts(direct_config):
        train, test = direct.cohort_masks(frame, cohort, direct_config)
        baseline = primary_predictions.loc[
            primary_predictions["period"].eq(cohort.period)
            & primary_predictions["cohort"].eq(cohort.cohort)
            & primary_predictions["model_id"].eq("DP0")
        ]
        if len(baseline) != int(test.sum()):
            raise AssertionError("RF ablation baseline cohort mismatch")
        baseline_metrics, _ = order_modeling.classification_metrics(
            baseline["order_id"], baseline["target"], baseline["raw_probability"]
        )
        for representation in ("score_only", "metadata_only"):
            for model_id in ("DPS", "DPG", "DPB"):
                numeric, categorical = direct.breach_feature_map(representation)[model_id]
                fitted = fit_historical_rf(
                    frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical
                )
                probability = fitted.predict_raw(frame.loc[test])
                metrics, _ = order_modeling.classification_metrics(
                    frame.loc[test, "order_id"], frame.loc[test, "late_delivery"], probability
                )
                rows.append(direct._metric_aliases({
                    "period": cohort.period, "cohort": cohort.cohort, "cohort_month": cohort.cohort,
                    "origin": cohort.origin, "family": "random_forest", "model_id": model_id,
                    "model_name": direct.BREACH_NAMES[model_id], "comparison": comparisons[model_id],
                    "candidate_model": model_id, "reference_model": "DP0",
                    "ablation_id": f"{model_id}_{representation}", "representation": representation,
                    "probability_type": "raw", "probability_variant": "raw", "n_train": int(train.sum()),
                    "train_order_id_sha256": order_modeling.order_id_hash(frame.loc[train, "order_id"]),
                    "fitted_model_sha256": order_modeling.fitted_model_sha256(fitted),
                    "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                    "reference_fitted_model_sha256": baseline["fitted_model_sha256"].iloc[0],
                    "delta_log_loss": metrics["log_loss"] - baseline_metrics["log_loss"],
                    "delta_brier": metrics["brier"] - baseline_metrics["brier"],
                    "delta_brier_score": metrics["brier"] - baseline_metrics["brier"],
                    **metrics,
                }))
                manifests.append(_model_manifest(
                    task="breach", stage=f"{cohort.period}_{representation}_sensitivity",
                    family="random_forest", model_id=model_id, representation=representation,
                    fitted=fitted, parameters=params, numeric=numeric, categorical=categorical,
                    cohort=cohort.cohort, origin=cohort.origin, fold=None, quantile=None,
                    train_ids=frame.loc[train, "order_id"], evaluation_ids=frame.loc[test, "order_id"],
                ))
    return pd.DataFrame(rows), pd.DataFrame(manifests)


@dataclass
class LeafWeightedQuantileForest:
    family: str
    preprocessor: object
    model: RandomForestRegressor
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    y_train: np.ndarray
    train_leaves: np.ndarray

    def predict_quantiles(self, frame: pd.DataFrame, quantiles: Sequence[float]) -> dict[float, np.ndarray]:
        x_test = self.preprocessor.transform(frame)
        test_leaves = self.model.apply(x_test)
        tree_maps: list[dict[int, np.ndarray]] = []
        for tree_index in range(self.train_leaves.shape[1]):
            leaves = self.train_leaves[:, tree_index]
            mapping = {
                int(leaf): self.y_train[leaves == leaf]
                for leaf in np.unique(leaves)
            }
            tree_maps.append(mapping)
        result = {float(q): np.empty(len(test_leaves), dtype=float) for q in quantiles}
        tree_weight = 1.0 / self.train_leaves.shape[1]
        for row_index in range(len(test_leaves)):
            values_parts: list[np.ndarray] = []
            weight_parts: list[np.ndarray] = []
            for tree_index, mapping in enumerate(tree_maps):
                values = mapping[int(test_leaves[row_index, tree_index])]
                values_parts.append(values)
                weight_parts.append(np.full(len(values), tree_weight / len(values)))
            values = np.concatenate(values_parts)
            weights = np.concatenate(weight_parts)
            for quantile in quantiles:
                result[float(quantile)][row_index] = weighted_quantile(
                    values, weights, float(quantile)
                )
        return result


@dataclass
class LognormalRidgeFitted:
    family: str
    preprocessor: object
    model: Ridge
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    residual_sigma: float

    def predict_quantiles(self, frame: pd.DataFrame, quantiles: Sequence[float]) -> dict[float, np.ndarray]:
        location = self.model.predict(self.preprocessor.transform(frame))
        return lognormal_quantiles(location, self.residual_sigma, tuple(map(float, quantiles)))


def fit_leaf_weighted_qrf(
    frame: pd.DataFrame, target: Sequence[float], numeric: Sequence[str], categorical: Sequence[str]
) -> LeafWeightedQuantileForest:
    y = np.asarray(target, dtype=float)
    preprocessor = order_modeling.make_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(frame)
    model = make_quantile_forest()
    model.fit(x_train, y)
    return LeafWeightedQuantileForest(
        "random_forest_leaf_weighted_quantile", preprocessor, model,
        tuple(numeric), tuple(categorical), y.copy(), model.apply(x_train),
    )


def fit_lognormal_ridge(
    frame: pd.DataFrame, target: Sequence[float], numeric: Sequence[str], categorical: Sequence[str]
) -> LognormalRidgeFitted:
    y = np.asarray(target, dtype=float)
    preprocessor = order_modeling.make_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(frame)
    model = make_lognormal_ridge()
    log_target = np.log(y)
    model.fit(x_train, log_target)
    sigma = float(np.std(log_target - model.predict(x_train), ddof=1))
    return LognormalRidgeFitted(
        "lognormal_ridge", preprocessor, model, tuple(numeric), tuple(categorical), sigma
    )


def fit_severity_alternative(
    family: str, frame: pd.DataFrame, target: Sequence[float],
    numeric: Sequence[str], categorical: Sequence[str],
) -> LeafWeightedQuantileForest | LognormalRidgeFitted:
    if family == "random_forest_leaf_weighted_quantile":
        return fit_leaf_weighted_qrf(frame, target, numeric, categorical)
    if family == "lognormal_ridge":
        return fit_lognormal_ridge(frame, target, numeric, categorical)
    raise ValueError(family)


def tune_severity_alternatives(
    frame: pd.DataFrame, direct_config: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric, categorical = direct.severity_feature_map("full")["DQ0"]
    quantiles = tuple(map(float, direct_config["severity"]["quantiles"]))
    tuning_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    for family in config["families"]["severity_additional"]:
        params = dict(config["alternative_parameters"][family])
        selections[family] = {str(q): params for q in quantiles}
        for fold in direct._normalised_folds(direct_config):
            train_all, valid_all = direct.chronological_masks(frame, fold)
            train = train_all & frame["positive_late_days"].gt(0)
            valid = valid_all & frame["positive_late_days"].gt(0)
            fitted = fit_severity_alternative(
                family, frame.loc[train], frame.loc[train, "positive_late_days"], numeric, categorical
            )
            predictions = fitted.predict_quantiles(frame.loc[valid], quantiles)
            for quantile in quantiles:
                loss = order_modeling.pinball_loss(
                    frame.loc[valid, "positive_late_days"], predictions[quantile], quantile
                )
                tuning_rows.append({
                    "record_type": "development_tuning", "task": "severity",
                    "stage": "development_tuning", "family": family, "model_family": family,
                    "model_id": "DQ0", "specification": "DQ0", "quantile": quantile,
                    "parameter_index": 0, "parameters_json": order_modeling.stable_json(params),
                    "fold": int(fold["fold"]), "n_train": int(train.sum()),
                    "n_validation": int(valid.sum()),
                    "train_order_id_sha256": order_modeling.order_id_hash(frame.loc[train, "order_id"]),
                    "validation_order_id_sha256": order_modeling.order_id_hash(frame.loc[valid, "order_id"]),
                    "pinball_loss": loss, "selected": True, "grid_type": "recovered_singleton",
                    "development_only": True, "later_or_terminal_outcomes_used": False,
                })
                manifests.append(_model_manifest(
                    task="severity", stage="development_tuning", family=family, model_id="DQ0",
                    representation="full", fitted=fitted, parameters=params, numeric=numeric,
                    categorical=categorical, cohort=f"development_fold_{fold['fold']}",
                    origin=fold["validation_start"], fold=int(fold["fold"]), quantile=quantile,
                    train_ids=frame.loc[train, "order_id"], evaluation_ids=frame.loc[valid, "order_id"],
                ))
        for quantile in quantiles:
            selection_rows.append({
                "record_type": "selected_specification", "task": "severity", "family": family,
                "model_family": family, "model_id": "DQ0", "specification": "DQ0",
                "quantile": quantile, "tuning_baseline": "DQ0",
                "selection_metric": "mean_chronological_pinball_loss",
                "parameters_json": order_modeling.stable_json(params),
                "grid_type": "recovered_singleton", "development_only": True,
                "later_or_terminal_outcomes_used": False,
            })
    return selections, pd.DataFrame(tuning_rows), pd.DataFrame(selection_rows), pd.DataFrame(manifests)


def evaluate_severity_alternatives(
    frame: pd.DataFrame, direct_config: Mapping[str, Any], config: Mapping[str, Any],
    selections: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    quantiles = tuple(map(float, direct_config["severity"]["quantiles"]))
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for cohort in direct.evaluation_cohorts(direct_config):
        train_all, test_all = direct.cohort_masks(frame, cohort, direct_config)
        train = train_all & frame["positive_late_days"].gt(0)
        test = test_all & frame["positive_late_days"].gt(0)
        if not train.any() or not test.any():
            raise RuntimeError(f"severity cohort has no breached orders: {cohort.cohort}")
        y_train = frame.loc[train, "positive_late_days"].to_numpy(float)
        y_test = frame.loc[test, "positive_late_days"].to_numpy(float)
        test_hash = order_modeling.order_id_hash(frame.loc[test, "order_id"])
        for family in config["families"]["severity_additional"]:
            local: dict[tuple[str, float], tuple[object, np.ndarray]] = {}
            for model_id, (numeric, categorical) in direct.severity_feature_map("full").items():
                fitted = fit_severity_alternative(
                    family, frame.loc[train], frame.loc[train, "positive_late_days"], numeric, categorical
                )
                predictions = fitted.predict_quantiles(frame.loc[test], quantiles)
                for quantile in quantiles:
                    local[(model_id, quantile)] = (fitted, predictions[quantile])
                    manifests.append(_model_manifest(
                        task="severity", stage="later_evaluation" if cohort.period == "later" else "terminal_stress",
                        family=family, model_id=model_id, representation="full", fitted=fitted,
                        parameters=selections[family][str(quantile)], numeric=numeric, categorical=categorical,
                        cohort=cohort.cohort, origin=cohort.origin, fold=None, quantile=quantile,
                        train_ids=frame.loc[train, "order_id"], evaluation_ids=frame.loc[test, "order_id"],
                    ))
            for quantile in quantiles:
                unconditional = float(np.quantile(y_train, quantile, method="linear"))
                unconditional_prediction = np.full(len(y_test), unconditional)
                unconditional_loss = order_modeling.pinball_loss(y_test, unconditional_prediction, quantile)
                baseline_prediction = local[("DQ0", quantile)][1]
                baseline_loss = order_modeling.pinball_loss(y_test, baseline_prediction, quantile)
                for model_id in direct.SEVERITY_MODEL_BLOCKS:
                    fitted, prediction = local[(model_id, quantile)]
                    metrics = order_modeling.quantile_metrics(y_test, prediction, quantile)
                    model_hash = order_modeling.fitted_model_sha256(fitted)
                    skill = 1 - float(metrics["pinball_loss"]) / baseline_loss if baseline_loss > 0 else np.nan
                    metric_rows.append(direct._metric_aliases({
                        "period": cohort.period, "cohort": cohort.cohort, "cohort_month": cohort.cohort,
                        "origin": cohort.origin, "family": family, "model_id": model_id,
                        "model_name": direct.SEVERITY_NAMES[model_id], "quantile": quantile,
                        "representation": "full", "n_train_breaches": int(train.sum()),
                        "train_order_id_sha256": order_modeling.order_id_hash(frame.loc[train, "order_id"]),
                        "fitted_model_sha256": model_hash,
                        "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                        "unconditional_training_quantile": unconditional,
                        "unconditional_reference_loss": unconditional_loss,
                        "baseline_pinball_loss": baseline_loss, "dq0_reference_loss": baseline_loss,
                        "skill": skill, "skill_vs_dq0": skill,
                        "skill_vs_unconditional": 1 - float(metrics["pinball_loss"]) / unconditional_loss if unconditional_loss > 0 else np.nan,
                        "nominal_coverage": quantile, "order_id_sha256": test_hash, **metrics,
                    }))
                    part = pd.DataFrame({
                        "order_id": frame.loc[test, "order_id"].astype(str).to_numpy(),
                        "purchase_date": frame.loc[test, "purchase_date"].to_numpy(),
                        "period": cohort.period, "cohort": cohort.cohort, "cohort_month": cohort.cohort,
                        "origin": cohort.origin, "family": family, "model_family": family,
                        "model_id": model_id, "specification": model_id,
                        "model_name": direct.SEVERITY_NAMES[model_id], "quantile": quantile,
                        "representation": "full", "actual_positive_late_days": y_test,
                        "prediction": prediction, "dq0_prediction": baseline_prediction,
                        "unconditional_prediction": unconditional_prediction,
                        "fitted_model_sha256": model_hash, "order_id_sha256": test_hash,
                    })
                    for block in direct.PROFILE_BLOCKS:
                        for suffix in ("support", "cold_start", "mapping_status"):
                            part[f"{block}_{suffix}"] = frame.loc[test, f"{block}_{suffix}"].to_numpy()
                    prediction_parts.append(part)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    pooled = direct._aggregate_severity_predictions(predictions, period="later")
    support = direct._severity_support_strata(predictions)
    family_config = json.loads(json.dumps(direct_config))
    family_config["severity"]["families"] = list(config["families"]["severity_additional"])
    summary = direct._severity_evidence_summary(metrics, support, family_config)
    metrics = direct._join_severity_evidence(metrics, summary)
    pooled = direct._join_severity_evidence(pooled, summary)
    return {
        "metrics": metrics, "pooled": pooled, "predictions": predictions,
        "support": support, "summary": summary, "manifests": pd.DataFrame(manifests),
    }


def _source_role(family: str) -> str:
    return "protected_pre_existing_direct_extension" if family in {
        "logistic_l2", "xgboost", "linear_quantile", "xgboost_quantile"
    } else "subsequent_model_family_robustness"


def build_breach_summary(
    evidence: pd.DataFrame, metrics: pd.DataFrame, pooled: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibrated = metrics.loc[
        metrics["period"].eq("later") & metrics["probability_type"].eq("calibrated")
    ]
    for family, baseline in calibrated.loc[calibrated["model_id"].eq("DP0")].groupby("family", sort=True):
        pooled_base = pooled.loc[
            pooled["family"].eq(family) & pooled["model_id"].eq("DP0")
            & pooled["probability_type"].eq("calibrated")
        ]
        rows.append({
            "row_type": "baseline_absolute", "family": family, "family_display": FAMILY_DISPLAY[family],
            "family_status": "evaluated", "source_role": _source_role(family),
            "comparison": "DP0_reference", "profile_block": "none", "model_id": "DP0",
            "median_log_loss": baseline["log_loss"].median(), "mean_log_loss": baseline["log_loss"].mean(),
            "sd_log_loss": baseline["log_loss"].std(ddof=1), "median_brier": baseline["brier"].median(),
            "mean_brier": baseline["brier"].mean(), "sd_brier": baseline["brier"].std(ddof=1),
            "median_average_precision": baseline["average_precision"].median(),
            "median_roc_auc": baseline["roc_auc"].median(),
            "median_top_10pct_lift": baseline["top_10pct_lift"].median(),
            "median_wace": baseline["wace"].median(),
            "median_absolute_calibration_slope_error": (baseline["calibration_slope"] - 1).abs().median(),
            "pooled_log_loss": pooled_base["log_loss"].iloc[0] if len(pooled_base) == 1 else np.nan,
            "pooled_brier": pooled_base["brier"].iloc[0] if len(pooled_base) == 1 else np.nan,
            "later_month_count": len(baseline), "evidence_label": "Reference",
        })
    for _, row in evidence.iterrows():
        family = str(row["family"])
        comparison = str(row["comparison"])
        candidate_id = str(row["candidate_model"])
        candidate = calibrated.loc[
            calibrated["family"].eq(family) & calibrated["model_id"].eq(candidate_id)
        ]
        deltas = pairs.loc[
            pairs["family"].eq(family) & pairs["comparison"].eq(comparison)
            & pairs["period"].eq("later")
        ]
        pooled_delta = pairs.loc[
            pairs["family"].eq(family) & pairs["comparison"].eq(comparison)
            & pairs["period"].eq("aggregate")
        ]
        rows.append({
            "row_type": "profile_increment", "family": family,
            "family_display": FAMILY_DISPLAY[family], "family_status": "evaluated",
            "source_role": _source_role(family), "comparison": comparison,
            "profile_block": PROFILE_BLOCK[comparison], "model_id": candidate_id,
            "median_log_loss": candidate["log_loss"].median(), "mean_log_loss": candidate["log_loss"].mean(),
            "sd_log_loss": candidate["log_loss"].std(ddof=1), "median_brier": candidate["brier"].median(),
            "mean_brier": candidate["brier"].mean(), "sd_brier": candidate["brier"].std(ddof=1),
            "median_average_precision": candidate["average_precision"].median(),
            "median_roc_auc": candidate["roc_auc"].median(),
            "median_top_10pct_lift": candidate["top_10pct_lift"].median(),
            "median_wace": candidate["wace"].median(),
            "median_absolute_calibration_slope_error": (candidate["calibration_slope"] - 1).abs().median(),
            "median_delta_log_loss": deltas["delta_log_loss"].median(),
            "mean_delta_log_loss": deltas["delta_log_loss"].mean(),
            "sd_delta_log_loss": deltas["delta_log_loss"].std(ddof=1),
            "median_delta_brier": deltas["delta_brier"].median(),
            "mean_delta_brier": deltas["delta_brier"].mean(),
            "sd_delta_brier": deltas["delta_brier"].std(ddof=1),
            "both_improved_month_count": int((deltas["delta_log_loss"].lt(0) & deltas["delta_brier"].lt(0)).sum()),
            "pooled_delta_log_loss": pooled_delta["delta_log_loss"].iloc[0] if len(pooled_delta) == 1 else np.nan,
            "pooled_delta_brier": pooled_delta["delta_brier"].iloc[0] if len(pooled_delta) == 1 else np.nan,
            "later_month_count": len(deltas),
            "calibration_guard_available": row.get("calibration_guard_available"),
            "calibration_not_systematically_worse": row.get("calibration_not_systematically_worse"),
            "high_support_no_material_reversal": row.get("high_support_no_material_reversal"),
            "score_contributes": row.get("score_contributes"),
            "all_guards_pass": row.get("all_guards_pass"),
            "evidence_label": row.get("evidence_label"), "evidence_reason": row.get("evidence_reason"),
            "label_type": "PROTECTED PRE-EXISTING DIRECT-EXTENSION LABEL" if _source_role(family).startswith("protected") else "ROBUSTNESS EVIDENCE LABEL",
        })
    for comparison in ("DPS-DP0", "DPG-DP0", "DPB-DP0"):
        rows.append({
            "row_type": "profile_increment", "family": "spline_logistic",
            "family_display": FAMILY_DISPLAY["spline_logistic"], "family_status": "blocked",
            "source_role": "not_evaluated", "comparison": comparison,
            "profile_block": PROFILE_BLOCK[comparison],
            "evidence_label": "Incomplete", "label_type": "ROBUSTNESS EVIDENCE LABEL",
            "evidence_reason": "no_applicable_exact_predictive_direct_feature_spline_specification_or_grid_recovered",
        })
    return pd.DataFrame(rows)


def build_severity_summary(
    evidence: pd.DataFrame, metrics: pd.DataFrame, pooled: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    later = metrics.loc[metrics["period"].eq("later")]
    for (family, quantile), baseline in later.loc[later["model_id"].eq("DQ0")].groupby(
        ["family", "quantile"], sort=True
    ):
        pooled_base = pooled.loc[
            pooled["family"].eq(family) & pooled["model_id"].eq("DQ0")
            & pd.to_numeric(pooled["quantile"]).eq(float(quantile))
        ]
        rows.append({
            "row_type": "baseline_absolute", "family": family, "family_display": FAMILY_DISPLAY[family],
            "family_status": "evaluated", "source_role": _source_role(family),
            "quantile": quantile, "comparison": "DQ0_reference", "profile_block": "none",
            "model_id": "DQ0", "median_pinball_loss": baseline["pinball_loss"].median(),
            "mean_pinball_loss": baseline["pinball_loss"].mean(),
            "sd_pinball_loss": baseline["pinball_loss"].std(ddof=1),
            "median_empirical_coverage": baseline["empirical_coverage"].median(),
            "median_absolute_coverage_error": baseline["coverage_error"].abs().median(),
            "pooled_pinball_loss": pooled_base["pinball_loss"].iloc[0] if len(pooled_base) == 1 else np.nan,
            "pooled_empirical_coverage": pooled_base["empirical_coverage"].iloc[0] if len(pooled_base) == 1 else np.nan,
            "later_month_count": len(baseline), "evidence_label": "Reference",
        })
    for _, row in evidence.iterrows():
        family = str(row["family"])
        quantile = float(row["quantile"])
        comparison = str(row["comparison"])
        candidate_id = str(row["candidate_model"])
        candidate = later.loc[
            later["family"].eq(family) & later["model_id"].eq(candidate_id)
            & pd.to_numeric(later["quantile"]).eq(quantile)
        ]
        pooled_candidate = pooled.loc[
            pooled["family"].eq(family) & pooled["model_id"].eq(candidate_id)
            & pd.to_numeric(pooled["quantile"]).eq(quantile)
        ]
        rows.append({
            "row_type": "profile_increment", "family": family, "family_display": FAMILY_DISPLAY[family],
            "family_status": "evaluated", "source_role": _source_role(family),
            "quantile": quantile, "comparison": comparison, "profile_block": PROFILE_BLOCK[comparison],
            "model_id": candidate_id, "median_pinball_loss": candidate["pinball_loss"].median(),
            "mean_pinball_loss": candidate["pinball_loss"].mean(),
            "sd_pinball_loss": candidate["pinball_loss"].std(ddof=1),
            "median_skill": candidate["skill"].median(), "mean_skill": candidate["skill"].mean(),
            "sd_skill": candidate["skill"].std(ddof=1),
            "favourable_month_count": int(candidate["skill"].ge(0).sum()),
            "median_empirical_coverage": candidate["empirical_coverage"].median(),
            "median_absolute_coverage_error": candidate["coverage_error"].abs().median(),
            "pooled_pinball_loss": pooled_candidate["pinball_loss"].iloc[0] if len(pooled_candidate) == 1 else np.nan,
            "pooled_skill": pooled_candidate["skill"].iloc[0] if len(pooled_candidate) == 1 else np.nan,
            "pooled_empirical_coverage": pooled_candidate["empirical_coverage"].iloc[0] if len(pooled_candidate) == 1 else np.nan,
            "later_month_count": len(candidate),
            "high_support_guard_available": row.get("high_support_guard_available"),
            "support_ge20_gain_present": row.get("support_ge20_gain_present"),
            "coverage_guard_available": row.get("coverage_guard_available"),
            "coverage_not_materially_worse": row.get("coverage_not_materially_worse"),
            "all_guards_pass": row.get("all_guards_pass"), "evidence_label": row.get("evidence_label"),
            "evidence_reason": row.get("evidence_reason"),
            "label_type": "PROTECTED PRE-EXISTING DIRECT-EXTENSION LABEL" if _source_role(family).startswith("protected") else "ROBUSTNESS EVIDENCE LABEL",
        })
    return pd.DataFrame(rows)


def build_robustness_labels(
    primary_labels: pd.DataFrame, rf_summary: pd.DataFrame, severity_summary: pd.DataFrame
) -> pd.DataFrame:
    protected = primary_labels.copy()
    protected["label_type"] = "PROTECTED PRE-EXISTING DIRECT-EXTENSION LABEL"
    protected["label_namespace_source"] = "direct_promise_profile_extension_v1"
    protected["source_evidence_labels_sha256"] = sha256_file(PRIMARY_ROOT / "EVIDENCE_LABELS.csv")
    protected["protected_label_unchanged"] = True
    breach = rf_summary.copy()
    breach["label_namespace"] = "direct_model_family_robustness_v1"
    breach["evidence_role"] = "subsequent_model_family_robustness"
    breach["outcome"] = "breach_probability"
    breach["profile_block"] = breach["comparison"].map(PROFILE_BLOCK)
    breach["specification"] = breach["candidate_model"]
    breach["quantile"] = np.nan
    breach["label_type"] = "ROBUSTNESS EVIDENCE LABEL"
    breach["protected_label_unchanged"] = np.nan
    severity = severity_summary.copy()
    severity["label_namespace"] = "direct_model_family_robustness_v1"
    severity["evidence_role"] = "subsequent_model_family_robustness"
    severity["outcome"] = "conditional_positive_lateness"
    severity["profile_block"] = severity["comparison"].map(PROFILE_BLOCK)
    severity["specification"] = severity["candidate_model"]
    severity["label_type"] = "ROBUSTNESS EVIDENCE LABEL"
    severity["protected_label_unchanged"] = np.nan
    blocked = pd.DataFrame([
        {
            "task": "breach", "family": "spline_logistic", "model_family": "spline_logistic",
            "comparison": comparison, "candidate_model": comparison.split("-")[0],
            "reference_model": "DP0", "representation": "full", "evidence_status": "Incomplete",
            "evidence_label": "Incomplete",
            "evidence_reason": "no_applicable_exact_predictive_direct_feature_spline_specification_or_grid_recovered",
            "label_namespace": "direct_model_family_robustness_v1",
            "evidence_role": "not_evaluated_blocker", "outcome": "breach_probability",
            "profile_block": PROFILE_BLOCK[comparison], "specification": comparison.split("-")[0],
            "label_type": "ROBUSTNESS EVIDENCE LABEL", "protected_label_unchanged": np.nan,
        }
        for comparison in ("DPS-DP0", "DPG-DP0", "DPB-DP0")
    ])
    return pd.concat([protected, breach, severity, blocked], ignore_index=True, sort=False)


def build_selection_table(
    primary: Mapping[str, pd.DataFrame], rf_tuning: pd.DataFrame, rf_selection: pd.DataFrame,
    severity_tuning: pd.DataFrame, severity_selection: pd.DataFrame,
) -> pd.DataFrame:
    direct_tuning = primary["development_tuning"].copy()
    direct_tuning["record_type"] = "development_tuning"
    direct_selection = primary["model_selection"].copy()
    direct_selection["record_type"] = "selected_specification"
    blocked = pd.DataFrame([{
        "record_type": "blocked_family", "task": "breach", "family": "spline_logistic",
        "model_family": "spline_logistic", "model_id": "DP0", "specification": "DP0",
        "tuning_baseline": "DP0", "grid_type": "unavailable", "selected": False,
        "development_only": True, "later_or_terminal_outcomes_used": False,
        "invalid_reason": "no_applicable_exact_predictive_direct_feature_spline_specification_or_grid_recovered",
    }])
    return pd.concat(
        [direct_tuning, direct_selection, rf_tuning, rf_selection, severity_tuning, severity_selection, blocked],
        ignore_index=True, sort=False,
    )


def run_model_experiment() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    direct_config = direct.load_config()
    validate_frozen_contract(config, direct_config)
    frame = direct.load_and_validate_frame(direct_config)
    selection, primary, audit = reproduce_primary(frame, direct_config, config)

    rf_selection, rf_tuning, rf_selection_table, rf_calibration_selection, rf_dev_manifests, rf_oof = tune_and_calibrate_rf(
        frame, direct_config, config
    )
    rf = evaluate_rf_breach(frame, direct_config, rf_selection)

    severity_selection, severity_tuning, severity_selection_table, severity_dev_manifests = tune_severity_alternatives(
        frame, direct_config, config
    )
    severity_alt = evaluate_severity_alternatives(
        frame, direct_config, config, severity_selection
    )

    primary_breach_metrics = primary["breach_calibration"].loc[
        primary["breach_calibration"]["period"].isin(["later", "terminal"])
    ].copy()
    primary_breach_monthly = primary["breach_monthly"].copy()
    primary_breach_pooled = primary["breach_pooled"].copy()
    breach_metrics = pd.concat([primary_breach_metrics, rf["metrics"]], ignore_index=True, sort=False)
    breach_monthly = pd.concat(
        [primary_breach_monthly, rf["metrics"].loc[rf["metrics"]["period"].eq("later")]],
        ignore_index=True, sort=False,
    )
    breach_pooled = pd.concat([primary_breach_pooled, rf["pooled"]], ignore_index=True, sort=False)
    breach_pairs = pd.concat([primary["breach_paired_differences"], rf["pairs"]], ignore_index=True, sort=False)
    breach_support = pd.concat([primary["breach_support"], rf["support"]], ignore_index=True, sort=False)
    breach_evidence = pd.concat(
        [primary["evidence_labels"].loc[primary["evidence_labels"]["task"].eq("breach")], rf["summary"]],
        ignore_index=True, sort=False,
    )
    breach_summary = build_breach_summary(breach_evidence, breach_metrics, breach_pooled, breach_pairs)

    calibration_metrics = pd.concat(
        [primary["breach_calibration"], rf["metrics"], rf["pooled"]], ignore_index=True, sort=False
    )
    calibration_metrics["record_type"] = "model_metric"
    reliability = pd.concat([primary["breach_reliability_bins"], rf["reliability"]], ignore_index=True, sort=False)
    reliability["record_type"] = "reliability_bin"
    calibration_table = pd.concat([calibration_metrics, reliability], ignore_index=True, sort=False)

    primary_severity_metrics = primary["severity_coverage"].loc[
        primary["severity_coverage"]["period"].isin(["later", "terminal"])
    ].copy()
    severity_metrics = pd.concat([primary_severity_metrics, severity_alt["metrics"]], ignore_index=True, sort=False)
    severity_monthly = pd.concat(
        [primary["severity_monthly"], severity_alt["metrics"].loc[severity_alt["metrics"]["period"].eq("later")]],
        ignore_index=True, sort=False,
    )
    severity_pooled = pd.concat([primary["severity_pooled"], severity_alt["pooled"]], ignore_index=True, sort=False)
    severity_support = pd.concat([primary["severity_support"], severity_alt["support"]], ignore_index=True, sort=False)
    severity_evidence = pd.concat(
        [primary["evidence_labels"].loc[primary["evidence_labels"]["task"].eq("severity")], severity_alt["summary"]],
        ignore_index=True, sort=False,
    )
    severity_summary = build_severity_summary(severity_evidence, severity_metrics, severity_pooled)
    severity_coverage = pd.concat(
        [severity_monthly, severity_pooled], ignore_index=True, sort=False
    )

    terminal = direct._terminal_long_table(
        frame, direct_config, breach_metrics, breach_pairs, severity_metrics
    )
    terminal["evidence_status"] = "Terminal stress only"
    terminal["evidence_label"] = ""
    terminal["label_type"] = "none_terminal_stress"

    labels = build_robustness_labels(
        primary["evidence_labels"], rf["summary"], severity_alt["summary"]
    )
    selection_table = build_selection_table(
        primary, rf_tuning, rf_selection_table, severity_tuning, severity_selection_table
    )

    breach_increment = breach_pairs.copy()
    severity_increment = pd.concat(
        [severity_monthly, severity_pooled], ignore_index=True, sort=False
    )
    severity_increment = severity_increment.loc[severity_increment["model_id"].ne("DQ0")].copy()
    severity_increment["comparison"] = severity_increment["model_id"].map(
        {"DQS": "DQS-DQ0", "DQG": "DQG-DQ0", "DQB": "DQB-DQ0"}
    )
    severity_increment["delta_pinball_loss"] = (
        severity_increment["pinball_loss"] - severity_increment["dq0_reference_loss"]
    )

    figure_breach = breach_summary.loc[breach_summary["row_type"].eq("profile_increment")].copy()
    figure_breach = figure_breach[[
        "family", "family_display", "family_status", "profile_block", "comparison",
        "median_delta_log_loss", "both_improved_month_count",
        "calibration_not_systematically_worse", "evidence_label", "label_type", "evidence_reason",
    ]]
    figure_severity = severity_summary.loc[severity_summary["row_type"].eq("profile_increment")].copy()
    figure_severity = figure_severity[[
        "family", "family_display", "family_status", "quantile", "profile_block", "comparison",
        "median_skill", "favourable_month_count", "median_empirical_coverage",
        "coverage_not_materially_worse", "evidence_label", "label_type",
    ]]

    model_manifests = pd.concat(
        [
            primary.get("development_model_manifests", pd.DataFrame()),
            primary.get("evaluation_model_manifests", pd.DataFrame()),
            rf_dev_manifests, rf["manifests"], severity_dev_manifests, severity_alt["manifests"],
        ], ignore_index=True, sort=False,
    )
    model_manifests["source_model_frame_sha256"] = config["sources"]["order_model_frame"][1]
    model_manifests["robustness_config_sha256"] = sha256_file(WORKSPACE / "ROBUSTNESS_FROZEN_CONFIG.json")

    outputs = {
        "MODEL_SELECTION_ALL_FAMILIES.csv": selection_table,
        "BREACH_MODEL_FAMILY_MONTHLY.csv": breach_monthly,
        "BREACH_MODEL_FAMILY_POOLED.csv": breach_pooled,
        "BREACH_MODEL_FAMILY_CALIBRATION.csv": calibration_table,
        "BREACH_PROFILE_INCREMENT_BY_FAMILY.csv": breach_increment,
        "BREACH_MODEL_FAMILY_SUMMARY.csv": breach_summary,
        "SEVERITY_MODEL_FAMILY_MONTHLY.csv": severity_monthly,
        "SEVERITY_MODEL_FAMILY_POOLED.csv": severity_pooled,
        "SEVERITY_MODEL_FAMILY_COVERAGE.csv": severity_coverage,
        "SEVERITY_PROFILE_INCREMENT_BY_FAMILY.csv": severity_increment,
        "SEVERITY_MODEL_FAMILY_SUMMARY.csv": severity_summary,
        "TERMINAL_MODEL_FAMILY_ROBUSTNESS.csv": terminal,
        "ROBUSTNESS_EVIDENCE_LABELS.csv": labels,
        "FIGURE_DATA_BREACH_MODEL_FAMILIES.csv": figure_breach,
        "FIGURE_DATA_SEVERITY_MODEL_FAMILIES.csv": figure_severity,
        "BREACH_SUPPORT_STRATA.csv": breach_support,
        "SEVERITY_SUPPORT_STRATA.csv": severity_support,
        "MODEL_MANIFESTS.csv": model_manifests,
        "RF_CALIBRATION_SELECTION.csv": rf_calibration_selection,
        "BREACH_RF_ABLATIONS.csv": rf["ablations"],
    }
    for filename, table in outputs.items():
        _atomic_csv(table, WORKSPACE / filename)
    _atomic_csv(rf["predictions"], WORKSPACE / "working/ALTERNATIVE_BREACH_PREDICTIONS.csv.gz")
    _atomic_csv(rf_oof, WORKSPACE / "working/RF_DEVELOPMENT_OOF_PREDICTIONS.csv.gz")
    _atomic_csv(severity_alt["predictions"], WORKSPACE / "working/ALTERNATIVE_SEVERITY_PREDICTIONS.csv.gz")

    selection_freeze = {
        "analysis_id": config["analysis_id"], "created_at_utc": utc_now(),
        "development_only": True, "later_or_terminal_outcomes_used": False,
        "primary_selection_reproduced": selection,
        "random_forest": rf_selection,
        "severity_alternatives": severity_selection,
        "spline_logistic": {
            "status": "blocked",
            "reason": "no_applicable_exact_predictive_direct_feature_spline_specification_or_grid_recovered",
        },
    }
    write_json(WORKSPACE / "ROBUSTNESS_MODEL_SELECTION_FREEZE.json", selection_freeze)
    receipt = {
        "analysis_id": config["analysis_id"], "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_frame_rows": len(frame),
        "model_frame_order_id_sha256": order_modeling.order_id_hash(frame["order_id"]),
        "profile_history_variant": "selected_90_day",
        "all_mature_workspace_consumed": False,
        "primary_reproduction_gate_passed": True,
        "primary_reproduction_audit_rows": len(audit),
        "breach_evaluable_families": ["logistic_l2", "random_forest", "xgboost"],
        "breach_blocked_families": ["spline_logistic"],
        "severity_evaluable_families": [
            "linear_quantile", "random_forest_leaf_weighted_quantile",
            "xgboost_quantile", "lognormal_ridge",
        ],
        "later_months": config["periods"]["later_months"],
        "terminal": config["periods"]["terminal"],
        "output_rows": {name: len(table) for name, table in outputs.items()},
    }
    write_json(WORKSPACE / "working/MODEL_RUN_RECEIPT.json", receipt)
    return receipt


__all__ = ["run_model_experiment"]
