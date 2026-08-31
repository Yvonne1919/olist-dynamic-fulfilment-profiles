"""Frozen experiment mechanics for order breach and positive-lateness severity.

This module contains no stage orchestration and performs no implicit I/O.  It
implements the chronological feature ladders, development-only tuning, later
cohort fitting, paired comparisons, and severity evaluation used by the
``order_breach_severity_v1`` runner.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from analysis.order_breach_severity_v1.scripts import order_features, order_modeling


PROFILE_BLOCKS = ("S1", "S2", "R1", "R2", "M5")
PROFILE_PAYLOAD_SUFFIXES = (
    "score",
    "log1p_support",
    "cold_start",
    "posterior_se",
    "freshness_days",
)


def profile_payload(block: str) -> list[str]:
    return [f"{block}_{suffix}" for suffix in PROFILE_PAYLOAD_SUFFIXES]


def model_feature_map() -> dict[str, tuple[list[str], list[str]]]:
    promise = list(order_features.PROMISE_NUMERIC_FEATURES)
    context = list(order_features.CONTEXT_NUMERIC_FEATURES)
    categories = list(order_features.CONTEXT_CATEGORICAL_FEATURES)
    seller = profile_payload("S1") + profile_payload("S2")
    route = profile_payload("R1") + profile_payload("R2")
    endpoint = profile_payload("M5")
    interactions = ["S1_score_x_known_event", "R1_score_x_known_event"]
    return {
        "M0": (promise, []),
        "M1": (promise + context, categories),
        "M2": (promise + context + seller, categories),
        "M3": (promise + context + route, categories),
        "M4": (promise + context + seller + route, categories),
        "M5": (promise + context + seller + route + endpoint, categories),
        "M4E": (promise + context + seller + route + interactions, categories),
    }


def severity_feature_map() -> dict[str, tuple[list[str], list[str]]]:
    main = model_feature_map()
    return {"Q1": main["M1"], "Q2": main["M2"], "Q3": main["M3"], "Q4": main["M4"]}


def ablation_feature_map() -> dict[str, tuple[str, list[str], list[str]]]:
    """Return frozen interpretation-only profile variants.

    The tuple is ``(reference_main_model, numeric, categorical)``.  All
    variants retain M1 and change only the historical profile payload.
    """

    main = model_feature_map()
    base_num, base_cat = main["M1"]

    def metadata(blocks: Sequence[str]) -> list[str]:
        return [
            f"{block}_{suffix}"
            for block in blocks
            for suffix in ("log1p_support", "cold_start", "posterior_se", "freshness_days")
        ]

    result: dict[str, tuple[str, list[str], list[str]]] = {}
    for name, blocks, reference in (
        ("seller", ("S1", "S2"), "M2"),
        ("route", ("R1", "R2"), "M3"),
        ("combined", ("S1", "S2", "R1", "R2"), "M4"),
    ):
        scores = [f"{block}_score" for block in blocks]
        result[f"{name}_score_only"] = (reference, base_num + scores, base_cat)
        result[f"{name}_metadata_only"] = (reference, base_num + metadata(blocks), base_cat)
        result[f"{name}_full"] = (
            reference,
            base_num + [feature for block in blocks for feature in profile_payload(block)],
            base_cat,
        )
        if name != "combined":
            result[f"{name}_{blocks[0].lower()}_score_only"] = (reference, base_num + [scores[0]], base_cat)
            result[f"{name}_{blocks[1].lower()}_score_only"] = (reference, base_num + [scores[1]], base_cat)
    return result


def add_frozen_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    event = pd.to_numeric(result["known_event_indicator"], errors="raise")
    for block in ("S1", "R1"):
        result[f"{block}_score_x_known_event"] = pd.to_numeric(
            result[f"{block}_score"], errors="coerce"
        ) * event
    return result


def validate_model_frame(frame: pd.DataFrame) -> None:
    required = {
        "order_id",
        "purchase_date",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "late_delivery",
        "positive_late_days",
        *order_features.CURRENT_ORDER_FEATURES,
        *[feature for block in PROFILE_BLOCKS for feature in profile_payload(block)],
        "S1_score_x_known_event",
        "R1_score_x_known_event",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"model frame missing required columns: {missing}")
    if frame["order_id"].isna().any() or frame["order_id"].duplicated().any():
        raise AssertionError("model frame must be one nonmissing row per order")
    target = pd.to_numeric(frame["late_delivery"], errors="raise")
    if not target.isin([0, 1]).all():
        raise AssertionError("breach target must be observed binary, never unresolved-as-zero")
    severity = pd.to_numeric(frame["positive_late_days"], errors="raise")
    if severity.isna().any() or severity.lt(0).any():
        raise AssertionError("positive lateness must be observed and nonnegative")
    purchase = pd.to_datetime(frame["purchase_date"], errors="raise")
    if purchase.min() < pd.Timestamp("2017-04-01") or purchase.max() > pd.Timestamp("2018-08-30"):
        raise AssertionError("model frame violates frozen purchase-date population")
    order_features.validate_feature_contract()


def frozen_folds(config: Mapping[str, object]) -> list[dict[str, object]]:
    return list(config["periods"]["development_inner_folds"])


def chronological_masks(frame: pd.DataFrame, fold: Mapping[str, object]) -> tuple[pd.Series, pd.Series]:
    purchase = pd.to_datetime(frame["purchase_date"], errors="raise")
    availability = pd.to_datetime(frame["order_delivered_customer_date"], errors="raise")
    train = (
        purchase.ge(pd.Timestamp(fold["train_start"]))
        & purchase.lt(pd.Timestamp(fold["train_end_exclusive"]))
        & availability.lt(pd.Timestamp(fold["validation_start"]))
    )
    validation_end = pd.Timestamp(fold["validation_end_exclusive"])
    validation = (
        purchase.ge(pd.Timestamp(fold["validation_start"]))
        & purchase.lt(validation_end)
        & availability.lt(validation_end)
    )
    if set(frame.loc[train, "order_id"]) & set(frame.loc[validation, "order_id"]):
        raise AssertionError("chronological fold contains overlapping order IDs")
    return train, validation


def _classifier_parameter_grid(config: Mapping[str, object], family: str) -> list[dict[str, object]]:
    classification = config["classification"]
    if family == "logistic_l2":
        base = dict(classification["logistic"])
        return [{**base, "C": value} for value in base.pop("C_grid")]
    if family == "xgboost":
        base = dict(classification["xgboost"])
        learning = base.pop("learning_rate_grid")
        depths = base.pop("max_depth_grid")
        child = base.pop("min_child_weight_grid")
        base["n_estimators"] = base.pop("max_estimators")
        return [
            {**base, "learning_rate": lr, "max_depth": depth, "min_child_weight": weight}
            for lr, depth, weight in itertools.product(learning, depths, child)
        ]
    raise ValueError(family)


def tune_classification(frame: pd.DataFrame, config: Mapping[str, object]) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    numeric, categorical = model_feature_map()["M1"]
    rows: list[dict[str, object]] = []
    chosen: dict[str, dict[str, object]] = {}
    for family in config["classification"]["families"]:
        for parameter_index, params in enumerate(_classifier_parameter_grid(config, family)):
            for fold in frozen_folds(config):
                train, valid = chronological_masks(frame, fold)
                model = order_modeling.fit_classifier(
                    frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical,
                    family, params,
                    validation_frame=frame.loc[valid] if family == "xgboost" else None,
                    validation_target=frame.loc[valid, "late_delivery"] if family == "xgboost" else None,
                )
                probability = model.predict_raw(frame.loc[valid])
                metrics, _ = order_modeling.classification_metrics(
                    frame.loc[valid, "order_id"], frame.loc[valid, "late_delivery"], probability
                )
                rows.append(
                    {
                        "task": "breach",
                        "family": family,
                        "model_id": "M1",
                        "quantile": np.nan,
                        "parameter_index": parameter_index,
                        "parameters_json": order_modeling.stable_json(params),
                        "fold": int(fold["fold"]),
                        "n_train": int(train.sum()),
                        "n_validation": int(valid.sum()),
                        "log_loss": metrics["log_loss"],
                        "brier": metrics["brier"],
                        "pinball_loss": np.nan,
                        "best_iteration": model.best_iteration,
                        "selected": False,
                        "invalid_reason": "",
                    }
                )
        table = pd.DataFrame([row for row in rows if row["family"] == family and row["task"] == "breach"])
        aggregate = table.groupby(["parameter_index", "parameters_json"], as_index=False).agg(
            mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"),
            valid_folds=("log_loss", "count"), median_best_iteration=("best_iteration", "median"),
        )
        expected = len(frozen_folds(config))
        best = aggregate.loc[aggregate["valid_folds"].eq(expected)].sort_values(
            ["mean_log_loss", "mean_brier", "parameters_json"], kind="mergesort"
        ).iloc[0]
        params = json.loads(best["parameters_json"])
        if family == "xgboost" and pd.notna(best["median_best_iteration"]):
            params["n_estimators"] = int(best["median_best_iteration"]) + 1
            params.pop("early_stopping_rounds", None)
        chosen[family] = params
        for row in rows:
            if row["task"] == "breach" and row["family"] == family:
                row["selected"] = row["parameter_index"] == int(best["parameter_index"])
    return chosen, pd.DataFrame(rows)


def development_oof_and_calibration(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    classifier_params: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, dict[str, object]]], pd.DataFrame, pd.DataFrame]:
    calibrators: dict[str, dict[str, dict[str, object]]] = {}
    oof_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    for family in config["classification"]["families"]:
        calibrators[family] = {}
        for model_id, (numeric, categorical) in model_feature_map().items():
            model_oof: list[pd.DataFrame] = []
            for fold in frozen_folds(config):
                train, valid = chronological_masks(frame, fold)
                model = order_modeling.fit_classifier(
                    frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical,
                    family, classifier_params[family],
                )
                probability = model.predict_raw(frame.loc[valid])
                model_oof.append(
                    pd.DataFrame(
                        {
                            "order_id": frame.loc[valid, "order_id"].astype(str).to_numpy(),
                            "purchase_date": frame.loc[valid, "purchase_date"].to_numpy(),
                            "fold": int(fold["fold"]),
                            "target": frame.loc[valid, "late_delivery"].astype(int).to_numpy(),
                            "raw_probability": probability,
                            "family": family,
                            "model_id": model_id,
                        }
                    )
                )
            oof = pd.concat(model_oof, ignore_index=True)
            calibrator, audit = order_modeling.select_calibration_method(oof)
            oof["calibrated_probability"] = calibrator.predict(oof["raw_probability"])
            calibrators[family][model_id] = calibrator.as_dict()
            audit.insert(0, "model_id", model_id)
            audit.insert(0, "family", family)
            calibration_parts.append(audit)
            oof_parts.append(oof)
    return calibrators, pd.concat(oof_parts, ignore_index=True), pd.concat(calibration_parts, ignore_index=True)


def tune_severity(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    classifier_params: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, dict[str, object]]], pd.DataFrame]:
    numeric, categorical = severity_feature_map()["Q1"]
    rows: list[dict[str, object]] = []
    chosen: dict[str, dict[str, dict[str, object]]] = {"linear_quantile": {}, "xgboost_quantile": {}}
    for quantile in config["severity"]["quantiles"]:
        linear_grid = [
            {"alpha": alpha, "solver": config["severity"]["linear_solver"]}
            for alpha in config["severity"]["linear_alpha_grid"]
        ]
        classifier_xgb = classifier_params["xgboost"]
        shared_xgb_keys = (
            "tree_method",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "n_estimators",
            "n_jobs",
            "random_state",
        )
        xgb_base = {key: classifier_xgb[key] for key in shared_xgb_keys}
        xgb_base["n_estimators"] = max(50, int(xgb_base.get("n_estimators", 300)))
        xgb_base["early_stopping_rounds"] = int(
            config["classification"]["xgboost"]["early_stopping_rounds"]
        )
        xgb_base["objective"] = str(config["severity"]["xgboost_objective"])
        xgb_base["eval_metric"] = str(config["severity"]["xgboost_eval_metric"])
        xgb_base["quantile_alpha"] = float(quantile)
        xgb_grid = [xgb_base]
        for family, grid in (("linear_quantile", linear_grid), ("xgboost_quantile", xgb_grid)):
            for parameter_index, params in enumerate(grid):
                for fold in frozen_folds(config):
                    train, valid = chronological_masks(frame, fold)
                    train &= frame["positive_late_days"].gt(0)
                    valid &= frame["positive_late_days"].gt(0)
                    model = order_modeling.fit_quantile_model(
                        frame.loc[train], frame.loc[train, "positive_late_days"], numeric, categorical,
                        family, float(quantile), params,
                        validation_frame=frame.loc[valid] if family == "xgboost_quantile" else None,
                        validation_target=frame.loc[valid, "positive_late_days"] if family == "xgboost_quantile" else None,
                    )
                    prediction = model.predict(frame.loc[valid])
                    loss = order_modeling.pinball_loss(frame.loc[valid, "positive_late_days"], prediction, float(quantile))
                    rows.append(
                        {
                            "task": "severity",
                            "family": family,
                            "model_id": "Q1",
                            "quantile": float(quantile),
                            "parameter_index": parameter_index,
                            "parameters_json": order_modeling.stable_json(params),
                            "fold": int(fold["fold"]),
                            "n_train": int(train.sum()),
                            "n_validation": int(valid.sum()),
                            "log_loss": np.nan,
                            "brier": np.nan,
                            "pinball_loss": loss,
                            "best_iteration": model.best_iteration,
                            "selected": False,
                            "invalid_reason": "",
                        }
                    )
            table = pd.DataFrame(
                [row for row in rows if row["family"] == family and row["quantile"] == float(quantile)]
            )
            aggregate = table.groupby(["parameter_index", "parameters_json"], as_index=False).agg(
                mean_pinball=("pinball_loss", "mean"), valid_folds=("pinball_loss", "count"),
                median_best_iteration=("best_iteration", "median"),
            )
            best = aggregate.loc[aggregate["valid_folds"].eq(len(frozen_folds(config)))].sort_values(
                ["mean_pinball", "parameters_json"], kind="mergesort"
            ).iloc[0]
            params = json.loads(best["parameters_json"])
            if family == "xgboost_quantile" and pd.notna(best["median_best_iteration"]):
                params["n_estimators"] = int(best["median_best_iteration"]) + 1
                params.pop("early_stopping_rounds", None)
            chosen[family][str(float(quantile))] = params
            for row in rows:
                if row["family"] == family and row["quantile"] == float(quantile):
                    row["selected"] = row["parameter_index"] == int(best["parameter_index"])
    return chosen, pd.DataFrame(rows)


def calibrator_from_dict(payload: Mapping[str, object]) -> order_modeling.FrozenCalibrator:
    return order_modeling.FrozenCalibrator(
        method=str(payload["method"]),
        platt_intercept=payload.get("platt_intercept"),
        platt_slope=payload.get("platt_slope"),
        isotonic_x=tuple(payload.get("isotonic_x", [])),
        isotonic_y=tuple(payload.get("isotonic_y", [])),
    )


@dataclass(frozen=True)
class Cohort:
    period: str
    cohort: str
    origin: pd.Timestamp
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


def evaluation_cohorts() -> list[Cohort]:
    later = [
        Cohort("later", start.strftime("%Y-%m"), start, start, start + pd.offsets.MonthBegin(1))
        for start in pd.date_range("2018-01-01", "2018-06-01", freq="MS")
    ]
    later.append(Cohort("terminal", "2018-07_to_2018-08", pd.Timestamp("2018-07-01"), pd.Timestamp("2018-07-01"), pd.Timestamp("2018-08-31")))
    return later


def cohort_masks(frame: pd.DataFrame, cohort: Cohort) -> tuple[pd.Series, pd.Series]:
    purchase = pd.to_datetime(frame["purchase_date"], errors="raise")
    availability = pd.to_datetime(frame["order_delivered_customer_date"], errors="raise")
    train = purchase.ge(pd.Timestamp("2017-04-01")) & purchase.lt(cohort.origin) & availability.lt(cohort.origin)
    test = purchase.ge(cohort.start) & purchase.lt(cohort.end_exclusive)
    if not train.any() or not test.any():
        raise RuntimeError(f"empty chronological train/test cohort: {cohort}")
    if set(frame.loc[train, "order_id"]) & set(frame.loc[test, "order_id"]):
        raise AssertionError("chronological evaluation order overlap")
    return train, test


def evaluate_classification(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    selection: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    bin_parts: list[pd.DataFrame] = []
    for cohort in evaluation_cohorts():
        train, test = cohort_masks(frame, cohort)
        for family in config["classification"]["families"]:
            params = selection["classification_parameters"][family]
            for model_id, (numeric, categorical) in model_feature_map().items():
                model = order_modeling.fit_classifier(
                    frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical, family, params
                )
                model_sha256 = order_modeling.fitted_model_sha256(model)
                raw_probability = model.predict_raw(frame.loc[test])
                calibrator = calibrator_from_dict(selection["calibrators"][family][model_id])
                calibrated = calibrator.predict(raw_probability)
                base = pd.DataFrame(
                    {
                        "order_id": frame.loc[test, "order_id"].astype(str).to_numpy(),
                        "purchase_date": frame.loc[test, "purchase_date"].to_numpy(),
                        "period": cohort.period,
                        "cohort": cohort.cohort,
                        "origin": cohort.origin,
                        "family": family,
                        "model_id": model_id,
                        "fitted_model_sha256": model_sha256,
                        "model_hash_type": "fitted_model",
                        "target": frame.loc[test, "late_delivery"].astype(int).to_numpy(),
                        "raw_probability": raw_probability,
                        "calibrated_probability": calibrated,
                    }
                )
                for column in (
                    "multi_seller", "known_event_indicator", "both_top10", "both_top10_phase",
                    *[f"{block}_{suffix}" for block in PROFILE_BLOCKS for suffix in ("support", "cold_start", "mapping_status", "score")],
                ):
                    if column in frame:
                        base[column] = frame.loc[test, column].to_numpy()
                predictions.append(base)
                for probability_type, values in (("raw", raw_probability), ("calibrated", calibrated)):
                    metric, bins = order_modeling.classification_metrics(
                        base["order_id"], base["target"], values, int(config["calibration"]["bins"])
                    )
                    metrics_rows.append(
                        {
                            "period": cohort.period, "cohort": cohort.cohort, "origin": cohort.origin,
                            "family": family, "model_id": model_id, "probability_type": probability_type,
                            "fitted_model_sha256": model_sha256,
                            "model_hash_type": "fitted_model",
                            "calibration_method": calibrator.method if probability_type == "calibrated" else "none",
                            "n_train": int(train.sum()), **metric,
                        }
                    )
                    bins.insert(0, "probability_type", probability_type)
                    bins.insert(0, "model_id", model_id)
                    bins.insert(0, "family", family)
                    bins.insert(0, "cohort", cohort.cohort)
                    bins.insert(0, "period", cohort.period)
                    bin_parts.append(bins)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return pd.DataFrame(metrics_rows), pd.concat(bin_parts, ignore_index=True), prediction_frame


def _aggregate_classification_predictions(predictions: pd.DataFrame, *, period: str = "later") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    bins: list[pd.DataFrame] = []
    selected = predictions.loc[predictions["period"].eq(period)]
    for (family, model_id), group in selected.groupby(["family", "model_id"], sort=True, observed=True):
        constituents = (
            group[["cohort", "fitted_model_sha256"]]
            .drop_duplicates()
            .sort_values(["cohort", "fitted_model_sha256"], kind="mergesort")
        )
        constituent_ids = [
            f"{row.cohort}:{row.fitted_model_sha256}" for row in constituents.itertuples(index=False)
        ]
        composite_hash = order_modeling.composite_fitted_model_sha256(constituent_ids)
        for probability_type, column in (("raw", "raw_probability"), ("calibrated", "calibrated_probability")):
            metric, table = order_modeling.classification_metrics(group["order_id"], group["target"], group[column])
            rows.append(
                {
                    "period": "aggregate", "cohort": f"{period}_pooled", "family": family,
                    "model_id": model_id, "probability_type": probability_type,
                    "fitted_model_sha256": composite_hash,
                    "model_hash_type": "composite_fitted_models",
                    "constituent_fitted_model_count": len(constituent_ids),
                    **metric,
                }
            )
            table.insert(0, "probability_type", probability_type)
            table.insert(0, "model_id", model_id)
            table.insert(0, "family", family)
            table.insert(0, "cohort", f"{period}_pooled")
            table.insert(0, "period", "aggregate")
            bins.append(table)
    return pd.DataFrame(rows), pd.concat(bins, ignore_index=True)


PRIMARY_COMPARISONS = (
    ("M1-M0", "M1", "M0"),
    ("M2-M1", "M2", "M1"),
    ("M3-M1", "M3", "M1"),
    ("M4-M1", "M4", "M1"),
    ("M4-M2", "M4", "M2"),
    ("M4-M3", "M4", "M3"),
    ("M5-M4", "M5", "M4"),
    ("M4E-M4", "M4E", "M4"),
)


def paired_classification_differences(predictions: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = list(predictions.groupby(["period", "cohort", "family"], sort=True, observed=True))
    for (period, cohort, family), group in groups:
        for comparison, candidate_id, reference_id in PRIMARY_COMPARISONS:
            candidate = group.loc[group["model_id"].eq(candidate_id)].copy()
            reference = group.loc[group["model_id"].eq(reference_id)].copy()
            paired = candidate.merge(
                reference[["order_id", "target", "calibrated_probability"]], on="order_id", how="inner",
                suffixes=("_candidate", "_reference"), validate="one_to_one",
            )
            if len(paired) != len(candidate) or len(paired) != len(reference):
                raise AssertionError(f"nonidentical paired sample: {period}/{cohort}/{family}/{comparison}")
            if not paired["target_candidate"].eq(paired["target_reference"]).all():
                raise AssertionError("paired target mismatch")
            y = paired["target_candidate"].to_numpy(int)
            pc = paired["calibrated_probability_candidate"].to_numpy(float)
            pr = paired["calibrated_probability_reference"].to_numpy(float)
            candidate_metrics, _ = order_modeling.classification_metrics(paired["order_id"], y, pc)
            reference_metrics, _ = order_modeling.classification_metrics(paired["order_id"], y, pr)
            bootstrap = order_modeling.paired_calendar_block_bootstrap(
                paired["purchase_date"], y, pc, pr,
                replicates=int(config["uncertainty"]["paired_calendar_block_bootstrap_replicates"]),
                seed=order_modeling.stable_seed(int(config["uncertainty"]["seed"]), period, cohort, family, comparison),
            )
            rows.append(
                {
                    "period": period, "cohort": cohort, "family": family, "comparison": comparison,
                    "candidate_model": candidate_id, "reference_model": reference_id,
                    "n_orders": len(paired), "paired_order_id_sha256": order_modeling.order_id_hash(paired["order_id"]),
                    "delta_log_loss": candidate_metrics["log_loss"] - reference_metrics["log_loss"],
                    "delta_brier": candidate_metrics["brier"] - reference_metrics["brier"],
                    "delta_average_precision": candidate_metrics["average_precision"] - reference_metrics["average_precision"],
                    "delta_roc_auc": candidate_metrics["roc_auc"] - reference_metrics["roc_auc"],
                    "delta_top_10pct_lift": candidate_metrics["top_10pct_lift"] - reference_metrics["top_10pct_lift"],
                    "delta_calibration_intercept": candidate_metrics["calibration_intercept"] - reference_metrics["calibration_intercept"],
                    "delta_calibration_slope": candidate_metrics["calibration_slope"] - reference_metrics["calibration_slope"],
                    **bootstrap,
                }
            )
    # Pooled Jan--Jun comparisons are the headline paired estimates.
    pooled = predictions.loc[predictions["period"].eq("later")].copy()
    if not pooled.empty:
        pooled["period"] = "aggregate"
        pooled["cohort"] = "later_pooled"
        rows.extend(paired_classification_differences(pooled, config).to_dict("records"))
    return pd.DataFrame(rows)


def evaluate_profile_ablations(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    selection: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = ablation_feature_map()
    for cohort in evaluation_cohorts():
        train, test = cohort_masks(frame, cohort)
        for family in config["classification"]["families"]:
            params = selection["classification_parameters"][family]
            m1_numeric, m1_categorical = model_feature_map()["M1"]
            m1_fitted = order_modeling.fit_classifier(
                frame.loc[train], frame.loc[train, "late_delivery"],
                m1_numeric, m1_categorical, family, params,
            )
            m1_model_sha256 = order_modeling.fitted_model_sha256(m1_fitted)
            m1_probability = m1_fitted.predict_raw(frame.loc[test])
            m1_metrics, _ = order_modeling.classification_metrics(
                frame.loc[test, "order_id"], frame.loc[test, "late_delivery"], m1_probability
            )
            for variant_id, (reference_model, numeric, categorical) in variants.items():
                fitted = order_modeling.fit_classifier(
                    frame.loc[train], frame.loc[train, "late_delivery"], numeric, categorical, family, params
                )
                probability = fitted.predict_raw(frame.loc[test])
                metrics, _ = order_modeling.classification_metrics(
                    frame.loc[test, "order_id"], frame.loc[test, "late_delivery"], probability
                )
                rows.append(
                    {
                        "period": cohort.period,
                        "cohort": cohort.cohort,
                        "origin": cohort.origin,
                        "family": family,
                        "ablation_id": variant_id,
                        "reference_main_model": reference_model,
                        "comparison": (
                            "M2-M1" if variant_id.startswith("seller_") else
                            "M3-M1" if variant_id.startswith("route_") else "M4-M1"
                        ),
                        "representation": (
                            "metadata_only" if variant_id.endswith("metadata_only") else
                            "full" if variant_id.endswith("full") else "score_only"
                        ),
                        "n_train": int(train.sum()),
                        "fitted_model_sha256": order_modeling.fitted_model_sha256(fitted),
                        "model_hash_type": "fitted_model",
                        "reference_fitted_model_sha256": m1_model_sha256,
                        "delta_log_loss": metrics["log_loss"] - m1_metrics["log_loss"],
                        "delta_brier": metrics["brier"] - m1_metrics["brier"],
                        **metrics,
                    }
                )
    table = pd.DataFrame(rows)
    aggregate = []
    for (family, ablation_id), group in table.loc[table["period"].eq("later")].groupby(
        ["family", "ablation_id"], sort=True, observed=True
    ):
        aggregate.append(
            {
                "period": "later_aggregate",
                "cohort": "monthly_median",
                "family": family,
                "ablation_id": ablation_id,
                "reference_main_model": group["reference_main_model"].iloc[0],
                "comparison": (
                    "M2-M1" if ablation_id.startswith("seller_") else
                    "M3-M1" if ablation_id.startswith("route_") else "M4-M1"
                ),
                "representation": group["representation"].iloc[0],
                "n_train": np.nan,
                "fitted_model_sha256": order_modeling.composite_fitted_model_sha256(
                    f"{row.cohort}:{row.fitted_model_sha256}"
                    for row in group[["cohort", "fitted_model_sha256"]].drop_duplicates().itertuples(index=False)
                ),
                "model_hash_type": "composite_fitted_models",
                "reference_fitted_model_sha256": order_modeling.composite_fitted_model_sha256(
                    f"{row.cohort}:{row.reference_fitted_model_sha256}"
                    for row in group[["cohort", "reference_fitted_model_sha256"]].drop_duplicates().itertuples(index=False)
                ),
                "constituent_fitted_model_count": group["fitted_model_sha256"].nunique(),
                **{
                    column: group[column].median()
                    for column in (
                        "n_orders", "n_events", "prevalence", "log_loss", "brier", "average_precision",
                        "roc_auc", "top_5pct_lift", "top_10pct_lift", "top_10pct_recall",
                        "calibration_intercept", "calibration_slope", "wace",
                    )
                },
                "calibration_invalid_reason": "",
                "order_id_sha256": "monthly_medians_no_single_order_hash",
                "delta_log_loss": group["delta_log_loss"].median(),
                "delta_brier": group["delta_brier"].median(),
            }
        )
    result = pd.concat([table, pd.DataFrame(aggregate)], ignore_index=True)
    result["benefit_not_metadata_only"] = False
    result["score_contributes"] = False
    later = result.loc[result["period"].eq("later")]
    for (family, comparison), group in later.groupby(["family", "comparison"], sort=True, observed=True):
        simple_prefix = {
            "M2-M1": "seller_", "M3-M1": "route_", "M4-M1": "combined_"
        }[comparison]
        score = group.loc[group["ablation_id"].eq(f"{simple_prefix}score_only")]
        metadata = group.loc[group["ablation_id"].eq(f"{simple_prefix}metadata_only")]
        full = group.loc[group["ablation_id"].eq(f"{simple_prefix}full")]
        score_improves = not score.empty and score["delta_log_loss"].median() < 0
        full_beats_metadata = (
            not metadata.empty and not full.empty
            and full["log_loss"].median() < metadata["log_loss"].median()
            and full["brier"].median() <= metadata["brier"].median()
        )
        guard = bool(score_improves or full_beats_metadata)
        mask = result["family"].eq(family) & result["comparison"].eq(comparison)
        result.loc[mask, ["benefit_not_metadata_only", "score_contributes"]] = guard
    return result


def _support_stratum(values: pd.Series) -> pd.Series:
    support = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [support.lt(5), support.between(5, 9), support.between(10, 19), support.ge(20)],
            ["0-4", "5-9", "10-19", "20+"],
            default="missing",
        ),
        index=values.index,
        dtype="string",
    )


def _order_composition_stratum(values: pd.Series) -> pd.Series:
    multi_seller = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [multi_seller.eq(0), multi_seller.eq(1)],
            ["single_seller", "multi_seller"],
            default="missing",
        ),
        index=values.index,
        dtype="string",
    )


def classification_support_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "seller": ("M2", "M1", "M2-M1", ["S1_support", "S2_support"], ["S1_cold_start", "S2_cold_start"]),
        "route": ("M3", "M1", "M3-M1", ["R1_support", "R2_support"], ["R1_cold_start", "R2_cold_start"]),
        "combined": ("M4", "M1", "M4-M1", ["S1_support", "S2_support", "R1_support", "R2_support"], ["S1_cold_start", "S2_cold_start", "R1_cold_start", "R2_cold_start"]),
    }
    for block, (model_id, reference_id, comparison, support_columns, cold_columns) in definitions.items():
        candidate = predictions.loc[predictions["model_id"].eq(model_id)].copy()
        reference = predictions.loc[predictions["model_id"].eq(reference_id), [
            "order_id", "period", "cohort", "family", "target", "calibrated_probability"
        ]].copy()
        subset = candidate.merge(
            reference,
            on=["order_id", "period", "cohort", "family"], how="inner", validate="one_to_one",
            suffixes=("_candidate", "_reference"),
        )
        if len(subset) != len(candidate):
            raise AssertionError(f"support-stratum paired sample mismatch for {comparison}")
        subset["minimum_support"] = subset[support_columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
        subset["any_cold_start"] = subset[cold_columns].fillna(False).astype(bool).any(axis=1)
        subset["support_stratum"] = _support_stratum(subset["minimum_support"])
        subset.loc[subset["any_cold_start"], "support_stratum"] = "cold_start"
        for keys, group in subset.groupby(["period", "cohort", "family", "support_stratum"], sort=True, observed=True):
            candidate_metric, _ = order_modeling.classification_metrics(
                group["order_id"], group["target_candidate"], group["calibrated_probability_candidate"]
            )
            reference_metric, _ = order_modeling.classification_metrics(
                group["order_id"], group["target_reference"], group["calibrated_probability_reference"]
            )
            rows.append(
                {
                    "block": block,
                    "comparison": comparison,
                    "period": keys[0],
                    "cohort": keys[1],
                    "family": keys[2],
                    "support_stratum": keys[3],
                    "median_support": group["minimum_support"].median(),
                    "cold_start_share": group["any_cold_start"].mean(),
                    "delta_log_loss": candidate_metric["log_loss"] - reference_metric["log_loss"],
                    "delta_brier": candidate_metric["brier"] - reference_metric["brier"],
                    **candidate_metric,
                }
            )
    if "multi_seller" in predictions.columns:
        candidate = predictions.loc[predictions["model_id"].eq("M4")].copy()
        reference = predictions.loc[predictions["model_id"].eq("M1"), [
            "order_id", "period", "cohort", "family", "target", "calibrated_probability"
        ]].copy()
        subset = candidate.merge(
            reference,
            on=["order_id", "period", "cohort", "family"], how="inner", validate="one_to_one",
            suffixes=("_candidate", "_reference"),
        )
        if len(subset) != len(candidate) or len(subset) != len(reference):
            raise AssertionError("order-composition paired sample mismatch for M4-M1")
        if not subset["target_candidate"].eq(subset["target_reference"]).all():
            raise AssertionError("order-composition paired target mismatch for M4-M1")
        support_columns = ["S1_support", "S2_support", "R1_support", "R2_support"]
        cold_columns = ["S1_cold_start", "S2_cold_start", "R1_cold_start", "R2_cold_start"]
        subset["minimum_support"] = subset[support_columns].apply(
            pd.to_numeric, errors="coerce"
        ).min(axis=1)
        subset["any_cold_start"] = subset[cold_columns].fillna(False).astype(bool).any(axis=1)
        subset["support_stratum"] = _order_composition_stratum(subset["multi_seller"])
        for keys, group in subset.groupby(
            ["period", "cohort", "family", "support_stratum"], sort=True, observed=True
        ):
            candidate_metric, _ = order_modeling.classification_metrics(
                group["order_id"], group["target_candidate"], group["calibrated_probability_candidate"]
            )
            reference_metric, _ = order_modeling.classification_metrics(
                group["order_id"], group["target_reference"], group["calibrated_probability_reference"]
            )
            rows.append(
                {
                    "block": "order_composition",
                    "comparison": "M4-M1",
                    "candidate_model": "M4",
                    "reference_model": "M1",
                    "period": keys[0],
                    "cohort": keys[1],
                    "family": keys[2],
                    "support_stratum": keys[3],
                    "order_composition_stratum": keys[3],
                    "median_support": group["minimum_support"].median(),
                    "cold_start_share": group["any_cold_start"].mean(),
                    "delta_log_loss": candidate_metric["log_loss"] - reference_metric["log_loss"],
                    "delta_brier": candidate_metric["brier"] - reference_metric["brier"],
                    **candidate_metric,
                }
            )
    result = pd.DataFrame(rows)
    result["high_support_no_material_reversal"] = False
    result["high_support_material_reversal"] = False
    for (family, comparison), group in result.loc[
        result["period"].eq("later") & result["support_stratum"].eq("20+")
    ].groupby(["family", "comparison"], sort=True, observed=True):
        reversal = bool(group["delta_log_loss"].median() > 0 and group["delta_brier"].median() > 0)
        mask = result["family"].eq(family) & result["comparison"].eq(comparison)
        result.loc[mask, "high_support_no_material_reversal"] = not reversal
        result.loc[mask, "high_support_material_reversal"] = reversal
    return result


def event_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison, candidate_id, reference_id in (("M4-M1", "M4", "M1"), ("M4E-M4", "M4E", "M4")):
        candidate = predictions.loc[predictions["model_id"].eq(candidate_id)].copy()
        candidate["event_stratum"] = candidate.get(
            "both_top10_phase", pd.Series("BAU", index=candidate.index)
        ).fillna("BAU").astype(str)
        reference = predictions.loc[predictions["model_id"].eq(reference_id), [
            "order_id", "period", "cohort", "family", "target", "calibrated_probability"
        ]]
        paired = candidate.merge(
            reference, on=["order_id", "period", "cohort", "family"], how="inner", validate="one_to_one",
            suffixes=("_candidate", "_reference"),
        )
        if len(paired) != len(candidate):
            raise AssertionError(f"event stratum paired sample mismatch: {comparison}")
        for keys, group in paired.groupby(
            ["period", "cohort", "family", "event_stratum"], sort=True, observed=True
        ):
            candidate_metric, _ = order_modeling.classification_metrics(
                group["order_id"], group["target_candidate"], group["calibrated_probability_candidate"]
            )
            reference_metric, _ = order_modeling.classification_metrics(
                group["order_id"], group["target_reference"], group["calibrated_probability_reference"]
            )
            rows.append(
                {
                    "period": keys[0], "cohort": keys[1], "family": keys[2], "model_id": candidate_id,
                    "comparison": comparison, "event_stratum": keys[3], "stratum": keys[3],
                    "retrospective_hrd_definition": "both_top10", "predictor_used": False,
                    "delta_log_loss": candidate_metric["log_loss"] - reference_metric["log_loss"],
                    "delta_brier": candidate_metric["brier"] - reference_metric["brier"],
                    **candidate_metric,
                }
            )
    return pd.DataFrame(rows)


def evaluate_severity(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    selection: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    feature_map = severity_feature_map()
    for cohort in evaluation_cohorts():
        train, test = cohort_masks(frame, cohort)
        train &= frame["positive_late_days"].gt(0)
        test &= frame["positive_late_days"].gt(0)
        if not train.any() or not test.any():
            raise RuntimeError(f"severity cohort has no breaches: {cohort}")
        y_train = frame.loc[train, "positive_late_days"].to_numpy(float)
        for quantile in config["severity"]["quantiles"]:
            unconditional = float(np.quantile(y_train, float(quantile), method="linear"))
            for family in config["severity"]["families"]:
                params = selection["severity_parameters"][family][str(float(quantile))]
                q1_prediction: np.ndarray | None = None
                cohort_models: list[tuple[str, np.ndarray, str]] = []
                for model_id, (numeric, categorical) in feature_map.items():
                    fitted = order_modeling.fit_quantile_model(
                        frame.loc[train], frame.loc[train, "positive_late_days"], numeric, categorical,
                        family, float(quantile), params,
                    )
                    model_sha256 = order_modeling.fitted_model_sha256(fitted)
                    prediction = fitted.predict(frame.loc[test])
                    if model_id == "Q1":
                        q1_prediction = prediction
                    cohort_models.append((model_id, prediction, model_sha256))
                if q1_prediction is None:
                    raise AssertionError("Q1 severity reference missing")
                y_test = frame.loc[test, "positive_late_days"].to_numpy(float)
                q1_loss = order_modeling.pinball_loss(y_test, q1_prediction, float(quantile))
                unconditional_prediction = np.full(len(y_test), unconditional)
                unconditional_loss = order_modeling.pinball_loss(y_test, unconditional_prediction, float(quantile))
                for model_id, prediction, model_sha256 in cohort_models:
                    metric = order_modeling.quantile_metrics(y_test, prediction, float(quantile))
                    rows.append(
                        {
                            "period": cohort.period, "cohort": cohort.cohort, "origin": cohort.origin,
                            "family": family, "model_id": model_id, "quantile": float(quantile),
                            "fitted_model_sha256": model_sha256,
                            "model_hash_type": "fitted_model",
                            "n_train_breaches": int(train.sum()), "unconditional_training_quantile": unconditional,
                            "unconditional_reference_loss": unconditional_loss,
                            "q1_reference_loss": q1_loss,
                            "skill_vs_unconditional": 1 - metric["pinball_loss"] / unconditional_loss if unconditional_loss > 0 else np.nan,
                            "skill_vs_q1": 1 - metric["pinball_loss"] / q1_loss if q1_loss > 0 else np.nan,
                            "order_id_sha256": order_modeling.order_id_hash(frame.loc[test, "order_id"]),
                            **metric,
                        }
                    )
                    part = pd.DataFrame(
                        {
                            "order_id": frame.loc[test, "order_id"].astype(str).to_numpy(),
                            "purchase_date": frame.loc[test, "purchase_date"].to_numpy(),
                            "period": cohort.period, "cohort": cohort.cohort, "origin": cohort.origin,
                            "family": family, "model_id": model_id, "quantile": float(quantile),
                            "fitted_model_sha256": model_sha256,
                            "model_hash_type": "fitted_model",
                            "actual_positive_late_days": y_test, "prediction": prediction,
                            "unconditional_prediction": unconditional,
                            "q1_prediction": q1_prediction,
                        }
                    )
                    for column in (
                        "both_top10_phase", "multi_seller",
                        *[f"{block}_{suffix}" for block in ("S1", "S2", "R1", "R2") for suffix in ("support", "cold_start")],
                    ):
                        if column in frame:
                            part[column] = frame.loc[test, column].to_numpy()
                    predictions.append(part)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def aggregate_severity_predictions(predictions: pd.DataFrame, period: str = "later") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = predictions.loc[predictions["period"].eq(period)]
    for (family, model_id, quantile), group in subset.groupby(
        ["family", "model_id", "quantile"], sort=True, observed=True
    ):
        constituents = (
            group[["cohort", "fitted_model_sha256"]]
            .drop_duplicates()
            .sort_values(["cohort", "fitted_model_sha256"], kind="mergesort")
        )
        constituent_ids = [
            f"{row.cohort}:{row.fitted_model_sha256}" for row in constituents.itertuples(index=False)
        ]
        composite_hash = order_modeling.composite_fitted_model_sha256(constituent_ids)
        metric = order_modeling.quantile_metrics(group["actual_positive_late_days"], group["prediction"], float(quantile))
        unconditional_loss = order_modeling.pinball_loss(
            group["actual_positive_late_days"], group["unconditional_prediction"], float(quantile)
        )
        q1_loss = order_modeling.pinball_loss(group["actual_positive_late_days"], group["q1_prediction"], float(quantile))
        rows.append(
            {
                "period": "aggregate", "cohort": f"{period}_pooled", "family": family,
                "model_id": model_id, "quantile": float(quantile),
                "fitted_model_sha256": composite_hash,
                "model_hash_type": "composite_fitted_models",
                "constituent_fitted_model_count": len(constituent_ids),
                "n_train_breaches": np.nan, "unconditional_training_quantile": np.nan,
                "unconditional_reference_loss": unconditional_loss, "q1_reference_loss": q1_loss,
                "skill_vs_unconditional": 1 - metric["pinball_loss"] / unconditional_loss if unconditional_loss > 0 else np.nan,
                "skill_vs_q1": 1 - metric["pinball_loss"] / q1_loss if q1_loss > 0 else np.nan,
                "order_id_sha256": order_modeling.order_id_hash(group["order_id"]), **metric,
            }
        )
    return pd.DataFrame(rows)


def evaluate_severity_ablations(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    selection: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = ablation_feature_map()
    # Severity interpretation focuses on the two full/mechanism blocks and the
    # frozen representation variants; endpoint history remains excluded.
    for cohort in evaluation_cohorts():
        train, test = cohort_masks(frame, cohort)
        train &= frame["positive_late_days"].gt(0)
        test &= frame["positive_late_days"].gt(0)
        for quantile in config["severity"]["quantiles"]:
            for family in config["severity"]["families"]:
                params = selection["severity_parameters"][family][str(float(quantile))]
                q1_num, q1_cat = severity_feature_map()["Q1"]
                q1_fitted = order_modeling.fit_quantile_model(
                    frame.loc[train], frame.loc[train, "positive_late_days"], q1_num, q1_cat,
                    family, float(quantile), params,
                )
                q1_model_sha256 = order_modeling.fitted_model_sha256(q1_fitted)
                q1 = q1_fitted.predict(frame.loc[test])
                reference_loss = order_modeling.pinball_loss(frame.loc[test, "positive_late_days"], q1, float(quantile))
                for variant_id, (_, numeric, categorical) in variants.items():
                    prediction_model = order_modeling.fit_quantile_model(
                        frame.loc[train], frame.loc[train, "positive_late_days"], numeric, categorical,
                        family, float(quantile), params,
                    )
                    prediction = prediction_model.predict(frame.loc[test])
                    metric = order_modeling.quantile_metrics(
                        frame.loc[test, "positive_late_days"], prediction, float(quantile)
                    )
                    rows.append(
                        {
                            "period": cohort.period, "cohort": cohort.cohort, "family": family,
                            "quantile": float(quantile), "ablation_id": variant_id,
                            "representation": (
                                "metadata_only" if variant_id.endswith("metadata_only") else
                                "full" if variant_id.endswith("full") else "score_only"
                            ),
                            "fitted_model_sha256": order_modeling.fitted_model_sha256(
                                prediction_model
                            ),
                            "model_hash_type": "fitted_model",
                            "reference_fitted_model_sha256": q1_model_sha256,
                            "q1_reference_loss": reference_loss,
                            "skill_vs_q1": 1 - metric["pinball_loss"] / reference_loss if reference_loss > 0 else np.nan,
                            "order_id_sha256": order_modeling.order_id_hash(frame.loc[test, "order_id"]),
                            **metric,
                        }
                    )
    return pd.DataFrame(rows)


def severity_support_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "seller": ("Q2", "Q2-Q1", ["S1_support", "S2_support"], ["S1_cold_start", "S2_cold_start"]),
        "route": ("Q3", "Q3-Q1", ["R1_support", "R2_support"], ["R1_cold_start", "R2_cold_start"]),
        "combined": ("Q4", "Q4-Q1", ["S1_support", "S2_support", "R1_support", "R2_support"], ["S1_cold_start", "S2_cold_start", "R1_cold_start", "R2_cold_start"]),
    }
    for block, (model_id, comparison, support_columns, cold_columns) in definitions.items():
        subset = predictions.loc[predictions["model_id"].eq(model_id)].copy()
        subset["minimum_support"] = subset[support_columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
        subset["any_cold_start"] = subset[cold_columns].fillna(False).astype(bool).any(axis=1)
        subset["support_stratum"] = _support_stratum(subset["minimum_support"])
        subset.loc[subset["any_cold_start"], "support_stratum"] = "cold_start"
        for keys, group in subset.groupby(
            ["period", "cohort", "family", "quantile", "support_stratum"], sort=True, observed=True
        ):
            metric = order_modeling.quantile_metrics(
                group["actual_positive_late_days"], group["prediction"], float(keys[3])
            )
            reference_loss = order_modeling.pinball_loss(
                group["actual_positive_late_days"], group["q1_prediction"], float(keys[3])
            )
            rows.append(
                {
                    "block": block, "period": keys[0], "cohort": keys[1], "family": keys[2],
                    "quantile": keys[3], "comparison": comparison, "support_stratum": keys[4],
                    "median_support": group["minimum_support"].median(),
                    "cold_start_share": group["any_cold_start"].mean(),
                    "q1_reference_loss": reference_loss,
                    "delta_pinball_loss": metric["pinball_loss"] - reference_loss,
                    "skill_vs_q1": 1 - metric["pinball_loss"] / reference_loss if reference_loss > 0 else np.nan,
                    **metric,
                }
            )
    if "multi_seller" in predictions.columns:
        subset = predictions.loc[predictions["model_id"].eq("Q4")].copy()
        support_columns = ["S1_support", "S2_support", "R1_support", "R2_support"]
        cold_columns = ["S1_cold_start", "S2_cold_start", "R1_cold_start", "R2_cold_start"]
        subset["minimum_support"] = subset[support_columns].apply(
            pd.to_numeric, errors="coerce"
        ).min(axis=1)
        subset["any_cold_start"] = subset[cold_columns].fillna(False).astype(bool).any(axis=1)
        subset["support_stratum"] = _order_composition_stratum(subset["multi_seller"])
        for keys, group in subset.groupby(
            ["period", "cohort", "family", "quantile", "support_stratum"],
            sort=True,
            observed=True,
        ):
            metric = order_modeling.quantile_metrics(
                group["actual_positive_late_days"], group["prediction"], float(keys[3])
            )
            reference_loss = order_modeling.pinball_loss(
                group["actual_positive_late_days"], group["q1_prediction"], float(keys[3])
            )
            rows.append(
                {
                    "block": "order_composition",
                    "period": keys[0],
                    "cohort": keys[1],
                    "family": keys[2],
                    "quantile": keys[3],
                    "comparison": "Q4-Q1",
                    "candidate_model": "Q4",
                    "reference_model": "Q1",
                    "support_stratum": keys[4],
                    "order_composition_stratum": keys[4],
                    "median_support": group["minimum_support"].median(),
                    "cold_start_share": group["any_cold_start"].mean(),
                    "q1_reference_loss": reference_loss,
                    "delta_pinball_loss": metric["pinball_loss"] - reference_loss,
                    "skill_vs_q1": 1 - metric["pinball_loss"] / reference_loss if reference_loss > 0 else np.nan,
                    **metric,
                }
            )
    result = pd.DataFrame(rows)
    result["support_ge20_gain_present"] = False
    result["gain_only_low_support"] = True
    high = result.loc[result["period"].eq("later") & result["support_stratum"].eq("20+")]
    for (family, comparison, quantile), group in high.groupby(
        ["family", "comparison", "quantile"], sort=True, observed=True
    ):
        guard = bool(group["skill_vs_q1"].median() > 0 and group["skill_vs_q1"].ge(0).sum() >= 4)
        mask = (
            result["family"].eq(family) & result["comparison"].eq(comparison)
            & pd.to_numeric(result["quantile"]).eq(float(quantile))
        )
        result.loc[mask, "support_ge20_gain_present"] = guard
        result.loc[mask, "gain_only_low_support"] = not guard
    return result


def add_breach_evidence_guards(
    paired: pd.DataFrame,
    calibration_results: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    result = paired.copy()
    result["calibration_not_systematically_worse"] = False
    result["calibration_systematically_worse"] = True
    thresholds = config["interpretation_thresholds"]
    wace_tolerance = float(thresholds["calibration_wace_worsening_tolerance"])
    slope_tolerance = float(thresholds["calibration_absolute_slope_error_worsening_tolerance"])
    metrics = calibration_results.loc[
        calibration_results["period"].eq("later")
        & calibration_results["probability_type"].eq("calibrated")
    ].copy()
    for comparison, candidate_id, reference_id in PRIMARY_COMPARISONS:
        if comparison not in {"M2-M1", "M3-M1", "M4-M1"}:
            continue
        candidate = metrics.loc[metrics["model_id"].eq(candidate_id)]
        reference = metrics.loc[metrics["model_id"].eq(reference_id)]
        merged = candidate.merge(
            reference[["cohort", "family", "wace", "calibration_slope"]],
            on=["cohort", "family"], how="inner", validate="one_to_one",
            suffixes=("_candidate", "_reference"),
        )
        for family, group in merged.groupby("family", sort=True, observed=True):
            wace_delta = (group["wace_candidate"] - group["wace_reference"]).median()
            slope_delta = (
                (group["calibration_slope_candidate"] - 1).abs()
                - (group["calibration_slope_reference"] - 1).abs()
            ).median()
            guard = bool(pd.notna(wace_delta) and pd.notna(slope_delta) and wace_delta <= wace_tolerance and slope_delta <= slope_tolerance)
            mask = result["family"].eq(family) & result["comparison"].eq(comparison)
            result.loc[mask, "calibration_not_systematically_worse"] = guard
            result.loc[mask, "calibration_systematically_worse"] = not guard
    return result


def add_severity_evidence_guards(
    skill: pd.DataFrame,
    coverage: pd.DataFrame,
    support: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skill_result = skill.copy()
    coverage_result = coverage.copy()
    for table in (skill_result, coverage_result):
        table["coverage_not_materially_worse"] = False
        table["coverage_materially_worse"] = True
        table["support_ge20_gain_present"] = False
        table["gain_only_low_support"] = True
    tolerance = float(config["interpretation_thresholds"]["q90_absolute_coverage_error_worsening_tolerance"])
    later_coverage = coverage_result.loc[coverage_result["period"].eq("later")]
    for family in sorted(later_coverage["family"].dropna().unique()):
        for model_id in ("Q2", "Q3", "Q4"):
            comparison = f"{model_id}-Q1"
            for quantile in (0.5, 0.9):
                candidate = later_coverage.loc[
                    later_coverage["family"].eq(family) & later_coverage["model_id"].eq(model_id)
                    & pd.to_numeric(later_coverage["quantile"]).eq(quantile)
                ]
                reference = later_coverage.loc[
                    later_coverage["family"].eq(family) & later_coverage["model_id"].eq("Q1")
                    & pd.to_numeric(later_coverage["quantile"]).eq(quantile)
                ]
                merged = candidate.merge(
                    reference[["cohort", "coverage_error"]], on="cohort", how="inner", validate="one_to_one",
                    suffixes=("_candidate", "_reference"),
                )
                deterioration = (
                    merged["coverage_error_candidate"].abs() - merged["coverage_error_reference"].abs()
                ).median() if not merged.empty else np.nan
                coverage_guard = bool(quantile != 0.9 or (pd.notna(deterioration) and deterioration <= tolerance))
                support_rows = support.loc[
                    support["family"].eq(family) & support["comparison"].eq(comparison)
                    & pd.to_numeric(support["quantile"]).eq(quantile)
                ]
                support_guard = bool(not support_rows.empty and support_rows["support_ge20_gain_present"].astype(bool).all())
                for table in (skill_result, coverage_result):
                    mask = (
                        table["family"].eq(family) & table["model_id"].eq(model_id)
                        & pd.to_numeric(table["quantile"]).eq(quantile)
                    )
                    table.loc[mask, "coverage_not_materially_worse"] = coverage_guard
                    table.loc[mask, "coverage_materially_worse"] = not coverage_guard
                    table.loc[mask, "support_ge20_gain_present"] = support_guard
                    table.loc[mask, "gain_only_low_support"] = not support_guard
    return skill_result, coverage_result


def development_eda(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    purchase = pd.to_datetime(frame["purchase_date"], errors="raise")
    dev = frame.loc[purchase.ge("2017-04-01") & purchase.lt("2018-01-01")].copy()
    numeric = list(order_features.CURRENT_ORDER_NUMERIC_FEATURES) + [f"{block}_score" for block in PROFILE_BLOCKS]
    correlation = dev[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    correlation = correlation.rename_axis("feature").reset_index().melt(
        id_vars="feature", var_name="other_feature", value_name="spearman"
    )
    profile_correlation = correlation.loc[
        correlation["feature"].isin([f"{block}_score" for block in PROFILE_BLOCKS])
        & correlation["other_feature"].isin(order_features.CURRENT_ORDER_NUMERIC_FEATURES)
    ].reset_index(drop=True)
    support_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for block in PROFILE_BLOCKS:
        score = pd.to_numeric(dev[f"{block}_score"], errors="coerce")
        support = pd.to_numeric(dev[f"{block}_support"], errors="coerce")
        se = pd.to_numeric(dev[f"{block}_posterior_se"], errors="coerce")
        support_rows.append(
            {
                "block": block, "n_orders": len(dev), "n_finite_score": score.notna().sum(),
                "median_support": support.median(), "median_posterior_se": se.median(),
                "spearman_support_score": support.corr(score, method="spearman"),
                "spearman_support_uncertainty": support.corr(se, method="spearman"),
                "cold_start_share": pd.to_numeric(dev[f"{block}_cold_start"], errors="coerce").mean(),
            }
        )
        for quantile, value in score.quantile([0, .05, .25, .5, .75, .95, 1]).items():
            distribution_rows.append({"block": block, "quantile": quantile, "score": value})
    missingness = pd.DataFrame(
        {
            "feature": list(order_features.CURRENT_ORDER_FEATURES) + [feature for block in PROFILE_BLOCKS for feature in profile_payload(block)],
        }
    )
    missingness["missing_count"] = missingness["feature"].map(dev.isna().sum())
    missingness["missing_share"] = missingness["missing_count"] / len(dev)
    category_rows: list[dict[str, object]] = []
    for feature in order_features.CONTEXT_CATEGORICAL_FEATURES:
        counts = dev[feature].astype("string").fillna("__MISSING__").value_counts(dropna=False)
        category_rows.append(
            {
                "feature": feature, "n_levels": len(counts), "n_rare_lt20": int(counts.lt(20).sum()),
                "orders_in_rare_lt20": int(counts.loc[counts.lt(20)].sum()),
                "rare_order_share": float(counts.loc[counts.lt(20)].sum() / len(dev)),
                "missing_count": int(dev[feature].isna().sum()),
            }
        )
    profile_promise = []
    for block in PROFILE_BLOCKS:
        profile_promise.append(
            {
                "block": block,
                "spearman_score_vs_promised_delivery_days": pd.to_numeric(dev[f"{block}_score"], errors="coerce").corr(
                    pd.to_numeric(dev["promised_delivery_days"], errors="coerce"), method="spearman"
                ),
            }
        )
    mechanism = pd.DataFrame(
        [
            {
                "seller_score": seller,
                "route_score": route,
                "spearman": pd.to_numeric(dev[seller], errors="coerce").corr(
                    pd.to_numeric(dev[route], errors="coerce"), method="spearman"
                ),
            }
            for seller in ("S1_score", "S2_score")
            for route in ("R1_score", "R2_score")
        ]
    )
    return {
        "numeric_spearman": correlation,
        "profile_current_correlation": profile_correlation,
        "support_uncertainty": pd.DataFrame(support_rows),
        "categorical_rarity": pd.DataFrame(category_rows),
        "missingness": missingness,
        "profile_distributions": pd.DataFrame(distribution_rows),
        "profile_vs_promise": pd.DataFrame(profile_promise),
        "seller_route_relationship": mechanism,
    }


__all__ = [
    "PRIMARY_COMPARISONS", "PROFILE_BLOCKS", "PROFILE_PAYLOAD_SUFFIXES", "Cohort",
    "ablation_feature_map", "add_frozen_interactions", "aggregate_severity_predictions",
    "calibrator_from_dict", "chronological_masks", "classification_support_strata",
    "cohort_masks", "development_eda", "development_oof_and_calibration",
    "evaluate_classification", "evaluate_profile_ablations", "evaluate_severity",
    "evaluate_severity_ablations", "evaluation_cohorts", "event_strata",
    "frozen_folds", "model_feature_map", "paired_classification_differences",
    "profile_payload", "severity_feature_map", "severity_support_strata",
    "add_breach_evidence_guards", "add_severity_evidence_guards",
    "tune_classification", "tune_severity", "validate_model_frame",
]
