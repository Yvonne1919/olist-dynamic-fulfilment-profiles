"""Deterministic modelling core for the direct-promise profile extension.

The module deliberately performs no work at import time.  It consumes the
already persisted Order V1 model frame, uses the protected Order V1 modelling
utilities read-only, and exposes functions for the extension runner.  No
current-order context feature is admitted to any model feature list here.

Run callers with ``python -B`` (or ``PYTHONDONTWRITEBYTECODE=1``) so importing
the protected modelling module cannot create bytecode inside the protected
Order V1 workspace.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.order_breach_severity_v1.scripts import order_modeling


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "analysis/direct_promise_profile_extension_v1"
CONFIG_PATH = WORKSPACE / "DIRECT_FROZEN_CONFIG.json"

BASELINE_FEATURE = "promised_delivery_days"
PROFILE_BLOCKS = ("S1", "S2", "R1", "R2")
PROFILE_PAYLOAD_SUFFIXES = (
    "score",
    "log1p_support",
    "cold_start",
    "posterior_se",
    "freshness_days",
)
PROFILE_METADATA_SUFFIXES = (
    "log1p_support",
    "cold_start",
    "posterior_se",
    "freshness_days",
)
PROFILE_AUDIT_SUFFIXES = ("support", "mapping_status", "last_mature_outcome_date")

BREACH_MODEL_BLOCKS: dict[str, tuple[str, ...]] = {
    "DP0": (),
    "DPS": ("S1", "S2"),
    "DPG": ("R1", "R2"),
    "DPB": ("S1", "S2", "R1", "R2"),
}
SEVERITY_MODEL_BLOCKS: dict[str, tuple[str, ...]] = {
    "DQ0": (),
    "DQS": ("S1", "S2"),
    "DQG": ("R1", "R2"),
    "DQB": ("S1", "S2", "R1", "R2"),
}
BREACH_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("DPS-DP0", "DPS", "DP0"),
    ("DPG-DP0", "DPG", "DP0"),
    ("DPB-DP0", "DPB", "DP0"),
)
SEVERITY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("DQS-DQ0", "DQS", "DQ0"),
    ("DQG-DQ0", "DQG", "DQ0"),
    ("DQB-DQ0", "DQB", "DQ0"),
)
BREACH_TO_SEVERITY = {"DP0": "DQ0", "DPS": "DQS", "DPG": "DQG", "DPB": "DQB"}

BREACH_NAMES = {
    "DP0": "Direct promise baseline",
    "DPS": "Promise + seller profiles",
    "DPG": "Promise + geographic profiles",
    "DPB": "Promise + both profile blocks",
}
SEVERITY_NAMES = {
    "DQ0": "Direct promise severity baseline",
    "DQS": "Promise + seller profiles",
    "DQG": "Promise + state-OD profiles",
    "DQB": "Promise + both profile blocks",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the frozen extension config."""

    source = Path(path) if path is not None else CONFIG_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DIRECT_FROZEN_CONFIG.json must contain a JSON object")
    if payload.get("analysis_id") != "direct_promise_profile_extension_v1":
        raise AssertionError("unexpected direct-extension analysis_id")
    if payload.get("breach", {}).get("baseline_feature") != BASELINE_FEATURE:
        raise AssertionError("direct baseline differs from promised_delivery_days")
    configured_breach = {
        key: tuple(value) for key, value in payload.get("breach", {}).get("models", {}).items()
    }
    configured_severity = {
        key: tuple(value) for key, value in payload.get("severity", {}).get("models", {}).items()
    }
    if configured_breach != BREACH_MODEL_BLOCKS:
        raise AssertionError("configured breach ladder differs from frozen direct ladder")
    if configured_severity != SEVERITY_MODEL_BLOCKS:
        raise AssertionError("configured severity ladder differs from frozen direct ladder")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, features)) + "\n").encode("utf-8")).hexdigest()


def _coerce_bool(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    text = values.astype("string").str.strip().str.lower()
    valid = text.isin(["true", "false", "1", "0"])
    if not valid.all():
        bad = sorted(text.loc[~valid].dropna().unique().tolist())[:10]
        raise ValueError(f"{name} contains non-boolean values: {bad}")
    return text.isin(["true", "1"])


def profile_features(blocks: Sequence[str], representation: str = "full") -> list[str]:
    """Return the exact ordered profile feature payload for a representation."""

    unknown = sorted(set(blocks) - set(PROFILE_BLOCKS))
    if unknown:
        raise ValueError(f"unknown profile blocks: {unknown}")
    if representation == "full":
        suffixes = PROFILE_PAYLOAD_SUFFIXES
    elif representation == "score_only":
        suffixes = ("score",)
    elif representation == "metadata_only":
        suffixes = PROFILE_METADATA_SUFFIXES
    else:
        raise ValueError(f"unsupported profile representation: {representation}")
    return [f"{block}_{suffix}" for block in blocks for suffix in suffixes]


def breach_feature_map(representation: str = "full") -> dict[str, tuple[list[str], list[str]]]:
    """Return numeric/categorical lists for the four direct breach specifications."""

    result: dict[str, tuple[list[str], list[str]]] = {}
    for model_id, blocks in BREACH_MODEL_BLOCKS.items():
        # DP0 is representation-invariant and is emitted only as ``full`` by
        # evaluation routines to avoid duplicate baseline rows.
        features = [BASELINE_FEATURE]
        if blocks:
            features.extend(profile_features(blocks, representation))
        result[model_id] = (features, [])
    return result


def severity_feature_map(representation: str = "full") -> dict[str, tuple[list[str], list[str]]]:
    result: dict[str, tuple[list[str], list[str]]] = {}
    for model_id, blocks in SEVERITY_MODEL_BLOCKS.items():
        features = [BASELINE_FEATURE]
        if blocks:
            features.extend(profile_features(blocks, representation))
        result[model_id] = (features, [])
    return result


def _expected_model_columns() -> set[str]:
    columns = {
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "purchase_date",
        "late_delivery",
        "positive_late_days",
        BASELINE_FEATURE,
    }
    for block in PROFILE_BLOCKS:
        columns.update(f"{block}_{suffix}" for suffix in PROFILE_PAYLOAD_SUFFIXES)
        columns.update(f"{block}_{suffix}" for suffix in PROFILE_AUDIT_SUFFIXES)
    return columns


def load_and_validate_frame(
    config: Mapping[str, Any] | None = None,
    *,
    root: str | Path = ROOT,
) -> pd.DataFrame:
    """Load the exact protected Order V1 model frame and validate its contract."""

    frozen = dict(config) if config is not None else load_config()
    source_path_text, expected_sha256 = frozen["sources"]["order_model_frame"]
    source = Path(root) / str(source_path_text)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"protected Order V1 model-frame hash mismatch: {actual_sha256} != {expected_sha256}"
        )
    frame = pd.read_csv(source, low_memory=False)
    missing = sorted(_expected_model_columns() - set(frame.columns))
    if missing:
        raise KeyError(f"protected model frame missing columns: {missing}")

    for column in ("order_purchase_timestamp", "order_delivered_customer_date", "purchase_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for block in PROFILE_BLOCKS:
        frame[f"{block}_cold_start"] = _coerce_bool(
            frame[f"{block}_cold_start"], f"{block}_cold_start"
        )
        frame[f"{block}_mapping_status"] = frame[f"{block}_mapping_status"].astype("string")
        frame[f"{block}_last_mature_outcome_date"] = pd.to_datetime(
            frame[f"{block}_last_mature_outcome_date"], errors="coerce"
        )
        for suffix in (
            "score",
            "log1p_support",
            "posterior_se",
            "freshness_days",
            "support",
        ):
            frame[f"{block}_{suffix}"] = pd.to_numeric(
                frame[f"{block}_{suffix}"], errors="raise"
            )

    frame[BASELINE_FEATURE] = pd.to_numeric(frame[BASELINE_FEATURE], errors="raise")
    frame["late_delivery"] = pd.to_numeric(frame["late_delivery"], errors="raise").astype(int)
    frame["positive_late_days"] = pd.to_numeric(
        frame["positive_late_days"], errors="raise"
    )

    population = frozen["population"]
    expected_rows = int(population["model_frame_rows"])
    if len(frame) != expected_rows:
        raise AssertionError(f"model-frame row count {len(frame)} != {expected_rows}")
    if frame["order_id"].isna().any() or frame["order_id"].duplicated().any():
        raise AssertionError("model frame must contain one nonmissing row per order_id")
    order_hash = order_modeling.order_id_hash(frame["order_id"])
    if order_hash != str(population["model_frame_order_id_sha256"]):
        raise AssertionError("model-frame order-ID hash mismatch")
    if not frame["late_delivery"].isin([0, 1]).all():
        raise AssertionError("late_delivery must be fully observed binary")
    if frame["positive_late_days"].isna().any() or frame["positive_late_days"].lt(0).any():
        raise AssertionError("positive_late_days must be finite and nonnegative")
    if not frame["late_delivery"].eq(frame["positive_late_days"].gt(0).astype(int)).all():
        raise AssertionError("breach and positive-lateness conditions disagree")
    if frame[BASELINE_FEATURE].isna().any() or not np.isfinite(frame[BASELINE_FEATURE]).all():
        raise AssertionError("direct promise baseline must be complete and finite")
    purchase = frame["purchase_date"]
    if purchase.min() < pd.Timestamp(population["start"]):
        raise AssertionError("model frame starts before frozen direct population")
    if purchase.max() > pd.Timestamp(population["end_inclusive"]):
        raise AssertionError("model frame ends after frozen direct population")

    # Validate the exact feature namespace and strict as-of property for every
    # seen profile exposure.  Fallback rows may legitimately have no last
    # mature date or posterior/freshness value.
    for mapping in (breach_feature_map("full"), severity_feature_map("full")):
        for _, (numeric, categorical) in mapping.items():
            if categorical:
                raise AssertionError("direct ladder unexpectedly contains categorical predictors")
            if len(numeric) != len(set(numeric)):
                raise AssertionError("direct ladder contains duplicate predictors")
            if any(feature not in frame.columns for feature in numeric):
                raise AssertionError("direct ladder references an unavailable predictor")
    for block in PROFILE_BLOCKS:
        status = frame[f"{block}_mapping_status"]
        cold = frame[f"{block}_cold_start"]
        if not cold.eq(status.eq("mapped_cold_start")).all():
            raise AssertionError(f"{block} cold-start flag conflates mapping states")
        seen = status.eq("seen")
        mature = frame[f"{block}_last_mature_outcome_date"]
        if mature.loc[seen].isna().any() or not mature.loc[seen].lt(purchase.loc[seen]).all():
            raise AssertionError(f"{block} seen rows violate strict pre-snapshot maturity")
        if frame[f"{block}_score"].isna().any() or not np.isfinite(
            frame[f"{block}_score"]
        ).all():
            raise AssertionError(f"{block} score is not complete and finite")
    return frame.sort_values(["purchase_date", "order_id"], kind="mergesort").reset_index(
        drop=True
    )


def _normalised_folds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for row in config["periods"]["development_inner_folds"]:
        if isinstance(row, Mapping):
            fold = dict(row)
        else:
            if len(row) != 5:
                raise ValueError("each development fold must have five fields")
            fold = {
                "fold": int(row[0]),
                "train_start": row[1],
                "train_end_exclusive": row[2],
                "validation_start": row[3],
                "validation_end_exclusive": row[4],
            }
        folds.append(fold)
    return folds


def chronological_masks(
    frame: pd.DataFrame, fold: Mapping[str, Any]
) -> tuple[pd.Series, pd.Series]:
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
        raise AssertionError("chronological development fold contains overlapping IDs")
    if not train.any() or not validation.any():
        raise RuntimeError(f"empty development fold: {fold}")
    return train, validation


@dataclass(frozen=True)
class Cohort:
    period: str
    cohort: str
    origin: pd.Timestamp
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


def evaluation_cohorts(config: Mapping[str, Any] | None = None) -> list[Cohort]:
    frozen = dict(config) if config is not None else load_config()
    months = [pd.Timestamp(f"{value}-01") for value in frozen["periods"]["later_months"]]
    cohorts = [
        Cohort("later", start.strftime("%Y-%m"), start, start, start + pd.offsets.MonthBegin(1))
        for start in months
    ]
    terminal_start, terminal_end = frozen["periods"]["terminal"]
    start = pd.Timestamp(terminal_start)
    cohorts.append(Cohort("terminal", "2018-07_to_2018-08", start, start, pd.Timestamp(terminal_end)))
    return cohorts


def cohort_masks(
    frame: pd.DataFrame,
    cohort: Cohort,
    config: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, pd.Series]:
    frozen = dict(config) if config is not None else load_config()
    purchase = pd.to_datetime(frame["purchase_date"], errors="raise")
    availability = pd.to_datetime(frame["order_delivered_customer_date"], errors="raise")
    train = (
        purchase.ge(pd.Timestamp(frozen["population"]["start"]))
        & purchase.lt(cohort.origin)
        & availability.lt(cohort.origin)
    )
    test = purchase.ge(cohort.start) & purchase.lt(cohort.end_exclusive)
    if not train.any() or not test.any():
        raise RuntimeError(f"empty evaluation cohort: {cohort}")
    if set(frame.loc[train, "order_id"]) & set(frame.loc[test, "order_id"]):
        raise AssertionError("chronological evaluation cohort contains overlapping IDs")
    return train, test


def _classifier_parameter_grid(config: Mapping[str, Any], family: str) -> list[dict[str, Any]]:
    breach = config["breach"]
    if family == "logistic_l2":
        base = dict(breach["logistic"])
        grid = list(base.pop("C_grid"))
        return [{**base, "C": float(value)} for value in grid]
    if family == "xgboost":
        base = dict(breach["xgboost"])
        learning = list(base.pop("learning_rate_grid"))
        depths = list(base.pop("max_depth_grid"))
        child = list(base.pop("min_child_weight_grid"))
        base["n_estimators"] = int(base.pop("max_estimators"))
        return [
            {
                **base,
                "learning_rate": float(rate),
                "max_depth": int(depth),
                "min_child_weight": float(weight),
            }
            for rate, depth, weight in itertools.product(learning, depths, child)
        ]
    raise ValueError(f"unsupported classifier family: {family}")


def _manifest_row(
    *,
    task: str,
    stage: str,
    family: str,
    model_id: str,
    representation: str,
    fitted: object,
    parameters: Mapping[str, Any],
    numeric: Sequence[str],
    categorical: Sequence[str],
    cohort: str,
    origin: object,
    fold: int | None,
    quantile: float | None,
    n_train: int,
    n_evaluation: int,
    train_order_ids: Iterable[object] | None = None,
    evaluation_order_ids: Iterable[object] | None = None,
) -> dict[str, Any]:
    features = list(numeric) + list(categorical)
    return {
        "task": task,
        "stage": stage,
        "model_family": family,
        "family": family,
        "specification": model_id,
        "model_id": model_id,
        "representation": representation,
        "quantile": quantile,
        "cohort": cohort,
        "origin": origin,
        "fold": fold,
        "n_train": int(n_train),
        "n_evaluation": int(n_evaluation),
        "train_order_id_sha256": (
            order_modeling.order_id_hash(train_order_ids)
            if train_order_ids is not None
            else ""
        ),
        "evaluation_order_id_sha256": (
            order_modeling.order_id_hash(evaluation_order_ids)
            if evaluation_order_ids is not None
            else ""
        ),
        "test_order_id_sha256": (
            order_modeling.order_id_hash(evaluation_order_ids)
            if evaluation_order_ids is not None
            else ""
        ),
        "parameters_json": order_modeling.stable_json(dict(parameters)),
        "numeric_features_json": order_modeling.stable_json(list(numeric)),
        "categorical_features_json": order_modeling.stable_json(list(categorical)),
        "ordered_feature_sha256": _feature_hash(features),
        "fitted_model_sha256": order_modeling.fitted_model_sha256(fitted),
        "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
        "best_iteration": getattr(fitted, "best_iteration", None),
    }


def _calibrator_from_dict(payload: Mapping[str, Any]) -> order_modeling.FrozenCalibrator:
    return order_modeling.FrozenCalibrator(
        method=str(payload["method"]),
        platt_intercept=payload.get("platt_intercept"),
        platt_slope=payload.get("platt_slope"),
        isotonic_x=tuple(payload.get("isotonic_x", [])),
        isotonic_y=tuple(payload.get("isotonic_y", [])),
    )


def _metric_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if "family" in result:
        result.setdefault("model_family", result["family"])
    if "model_id" in result:
        result.setdefault("specification", result["model_id"])
    if "n_orders" in result:
        result.setdefault("n_obs", result["n_orders"])
    if "brier" in result:
        result.setdefault("brier_score", result["brier"])
    if "top_10pct_lift" in result:
        result.setdefault("top10_lift", result["top_10pct_lift"])
    if "empirical_coverage" in result:
        result.setdefault("coverage", result["empirical_coverage"])
    if "cohort" in result:
        result.setdefault("cohort_month", result["cohort"])
    return result


def _rows_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a frame without losing a useful empty-table contract."""

    return pd.DataFrame([dict(row) for row in rows])


def _development_classifier_selection(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    numeric, categorical = breach_feature_map("full")["DP0"]
    folds = _normalised_folds(config)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for family in config["breach"]["families"]:
        family_start = len(rows)
        for parameter_index, params in enumerate(_classifier_parameter_grid(config, family)):
            for fold in folds:
                train, valid = chronological_masks(frame, fold)
                fitted = order_modeling.fit_classifier(
                    frame.loc[train],
                    frame.loc[train, "late_delivery"],
                    numeric,
                    categorical,
                    str(family),
                    params,
                    validation_frame=frame.loc[valid] if family == "xgboost" else None,
                    validation_target=(
                        frame.loc[valid, "late_delivery"] if family == "xgboost" else None
                    ),
                )
                probability = fitted.predict_raw(frame.loc[valid])
                metrics, _ = order_modeling.classification_metrics(
                    frame.loc[valid, "order_id"],
                    frame.loc[valid, "late_delivery"],
                    probability,
                    int(config["breach"]["calibration_bins"]),
                )
                model_hash = order_modeling.fitted_model_sha256(fitted)
                row = {
                    "task": "breach",
                    "stage": "development_tuning",
                    "family": family,
                    "model_family": family,
                    "model_id": "DP0",
                    "specification": "DP0",
                    "representation": "full",
                    "quantile": np.nan,
                    "parameter_index": int(parameter_index),
                    "parameters_json": order_modeling.stable_json(params),
                    "fold": int(fold["fold"]),
                    "n_train": int(train.sum()),
                    "n_validation": int(valid.sum()),
                    "train_order_id_sha256": order_modeling.order_id_hash(
                        frame.loc[train, "order_id"]
                    ),
                    "validation_order_id_sha256": order_modeling.order_id_hash(
                        frame.loc[valid, "order_id"]
                    ),
                    "ordered_features_json": order_modeling.stable_json(numeric),
                    "ordered_feature_sha256": _feature_hash(numeric),
                    "fitted_model_sha256": model_hash,
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "brier_score": metrics["brier"],
                    "pinball_loss": np.nan,
                    "best_iteration": fitted.best_iteration,
                    "selected": False,
                    "invalid_reason": "",
                }
                rows.append(row)
                manifests.append(
                    _manifest_row(
                        task="breach",
                        stage="development_tuning",
                        family=str(family),
                        model_id="DP0",
                        representation="full",
                        fitted=fitted,
                        parameters=params,
                        numeric=numeric,
                        categorical=categorical,
                        cohort=f"development_fold_{int(fold['fold'])}",
                        origin=fold["validation_start"],
                        fold=int(fold["fold"]),
                        quantile=None,
                        n_train=int(train.sum()),
                        n_evaluation=int(valid.sum()),
                        train_order_ids=frame.loc[train, "order_id"],
                        evaluation_order_ids=frame.loc[valid, "order_id"],
                    )
                )
        family_rows = pd.DataFrame(rows[family_start:])
        aggregate = family_rows.groupby(
            ["parameter_index", "parameters_json"], as_index=False, sort=True
        ).agg(
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            valid_folds=("log_loss", "count"),
            median_best_iteration=("best_iteration", "median"),
        )
        eligible = aggregate.loc[aggregate["valid_folds"].eq(len(folds))]
        if eligible.empty:
            raise RuntimeError(f"no complete DP0 tuning candidate for {family}")
        best = eligible.sort_values(
            ["mean_log_loss", "mean_brier", "parameters_json"], kind="mergesort"
        ).iloc[0]
        params = json.loads(str(best["parameters_json"]))
        if family == "xgboost" and pd.notna(best["median_best_iteration"]):
            params["n_estimators"] = int(best["median_best_iteration"]) + 1
            params.pop("early_stopping_rounds", None)
        selected[str(family)] = params
        for row in rows[family_start:]:
            row["selected"] = row["parameter_index"] == int(best["parameter_index"])
            row["selected_final_parameters_json"] = order_modeling.stable_json(params)
    return selected, _rows_frame(rows), _rows_frame(manifests)


def _development_calibration_selection(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    classifier_parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibrators: dict[str, dict[str, dict[str, Any]]] = {}
    oof_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    folds = _normalised_folds(config)
    features = breach_feature_map("full")
    for family in config["breach"]["families"]:
        family = str(family)
        calibrators[family] = {}
        for model_id, (numeric, categorical) in features.items():
            parts: list[pd.DataFrame] = []
            for fold in folds:
                train, valid = chronological_masks(frame, fold)
                fitted = order_modeling.fit_classifier(
                    frame.loc[train],
                    frame.loc[train, "late_delivery"],
                    numeric,
                    categorical,
                    family,
                    classifier_parameters[family],
                )
                probability = fitted.predict_raw(frame.loc[valid])
                parts.append(
                    pd.DataFrame(
                        {
                            "order_id": frame.loc[valid, "order_id"].astype(str).to_numpy(),
                            "purchase_date": frame.loc[valid, "purchase_date"].to_numpy(),
                            "fold": int(fold["fold"]),
                            "target": frame.loc[valid, "late_delivery"].to_numpy(int),
                            "raw_probability": probability,
                            "family": family,
                            "model_id": model_id,
                            "representation": "full",
                            "fitted_model_sha256": order_modeling.fitted_model_sha256(fitted),
                        }
                    )
                )
                manifests.append(
                    _manifest_row(
                        task="breach",
                        stage="development_calibration_oof",
                        family=family,
                        model_id=model_id,
                        representation="full",
                        fitted=fitted,
                        parameters=classifier_parameters[family],
                        numeric=numeric,
                        categorical=categorical,
                        cohort=f"development_fold_{int(fold['fold'])}",
                        origin=fold["validation_start"],
                        fold=int(fold["fold"]),
                        quantile=None,
                        n_train=int(train.sum()),
                        n_evaluation=int(valid.sum()),
                        train_order_ids=frame.loc[train, "order_id"],
                        evaluation_order_ids=frame.loc[valid, "order_id"],
                    )
                )
            oof = pd.concat(parts, ignore_index=True)
            calibrator, audit = order_modeling.select_calibration_method(oof)
            oof["calibrated_probability"] = calibrator.predict(oof["raw_probability"])
            calibrators[family][model_id] = calibrator.as_dict()
            audit.insert(0, "representation", "full")
            audit.insert(0, "model_id", model_id)
            audit.insert(0, "specification", model_id)
            audit.insert(0, "family", family)
            audit.insert(0, "model_family", family)
            audit["calibrator_parameters_json"] = order_modeling.stable_json(
                calibrator.as_dict()
            )
            audit["oof_n_orders"] = len(oof)
            audit["oof_order_id_sha256"] = order_modeling.order_id_hash(oof["order_id"])
            audit_parts.append(audit)
            oof_parts.append(oof)
    return (
        calibrators,
        pd.concat(oof_parts, ignore_index=True),
        pd.concat(audit_parts, ignore_index=True),
        _rows_frame(manifests),
    )


def _development_severity_selection(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    classifier_parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame, pd.DataFrame]:
    numeric, categorical = severity_feature_map("full")["DQ0"]
    folds = _normalised_folds(config)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    selected: dict[str, dict[str, dict[str, Any]]] = {
        "linear_quantile": {},
        "xgboost_quantile": {},
    }
    for quantile_value in config["severity"]["quantiles"]:
        quantile = float(quantile_value)
        linear_grid = [
            {
                "alpha": float(alpha),
                "solver": str(config["severity"]["linear_solver"]),
            }
            for alpha in config["severity"]["linear_alpha_grid"]
        ]
        classifier_xgb = classifier_parameters["xgboost"]
        shared_keys = (
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
        xgb_params = {key: classifier_xgb[key] for key in shared_keys}
        xgb_params["n_estimators"] = max(50, int(xgb_params["n_estimators"]))
        xgb_params["early_stopping_rounds"] = int(
            config["breach"]["xgboost"]["early_stopping_rounds"]
        )
        xgb_params["objective"] = str(config["severity"]["xgboost_objective"])
        xgb_params["eval_metric"] = str(config["severity"]["xgboost_eval_metric"])
        xgb_params["quantile_alpha"] = quantile
        grids = {
            "linear_quantile": linear_grid,
            "xgboost_quantile": [xgb_params],
        }
        for family in config["severity"]["families"]:
            family = str(family)
            family_start = len(rows)
            for parameter_index, params in enumerate(grids[family]):
                for fold in folds:
                    train, valid = chronological_masks(frame, fold)
                    train &= frame["positive_late_days"].gt(0)
                    valid &= frame["positive_late_days"].gt(0)
                    if not train.any() or not valid.any():
                        raise RuntimeError(
                            f"empty positive-lateness development sample for fold {fold['fold']}"
                        )
                    fitted = order_modeling.fit_quantile_model(
                        frame.loc[train],
                        frame.loc[train, "positive_late_days"],
                        numeric,
                        categorical,
                        family,
                        quantile,
                        params,
                        validation_frame=(
                            frame.loc[valid] if family == "xgboost_quantile" else None
                        ),
                        validation_target=(
                            frame.loc[valid, "positive_late_days"]
                            if family == "xgboost_quantile"
                            else None
                        ),
                    )
                    prediction = fitted.predict(frame.loc[valid])
                    loss = order_modeling.pinball_loss(
                        frame.loc[valid, "positive_late_days"], prediction, quantile
                    )
                    rows.append(
                        {
                            "task": "severity",
                            "stage": "development_tuning",
                            "family": family,
                            "model_family": family,
                            "model_id": "DQ0",
                            "specification": "DQ0",
                            "representation": "full",
                            "quantile": quantile,
                            "parameter_index": int(parameter_index),
                            "parameters_json": order_modeling.stable_json(params),
                            "fold": int(fold["fold"]),
                            "n_train": int(train.sum()),
                            "n_validation": int(valid.sum()),
                            "train_order_id_sha256": order_modeling.order_id_hash(
                                frame.loc[train, "order_id"]
                            ),
                            "validation_order_id_sha256": order_modeling.order_id_hash(
                                frame.loc[valid, "order_id"]
                            ),
                            "ordered_features_json": order_modeling.stable_json(numeric),
                            "ordered_feature_sha256": _feature_hash(numeric),
                            "fitted_model_sha256": order_modeling.fitted_model_sha256(fitted),
                            "log_loss": np.nan,
                            "brier": np.nan,
                            "brier_score": np.nan,
                            "pinball_loss": loss,
                            "best_iteration": fitted.best_iteration,
                            "selected": False,
                            "invalid_reason": "",
                        }
                    )
                    manifests.append(
                        _manifest_row(
                            task="severity",
                            stage="development_tuning",
                            family=family,
                            model_id="DQ0",
                            representation="full",
                            fitted=fitted,
                            parameters=params,
                            numeric=numeric,
                            categorical=categorical,
                            cohort=f"development_fold_{int(fold['fold'])}",
                            origin=fold["validation_start"],
                            fold=int(fold["fold"]),
                            quantile=quantile,
                            n_train=int(train.sum()),
                            n_evaluation=int(valid.sum()),
                            train_order_ids=frame.loc[train, "order_id"],
                            evaluation_order_ids=frame.loc[valid, "order_id"],
                        )
                    )
            family_rows = pd.DataFrame(rows[family_start:])
            aggregate = family_rows.groupby(
                ["parameter_index", "parameters_json"], as_index=False, sort=True
            ).agg(
                mean_pinball_loss=("pinball_loss", "mean"),
                valid_folds=("pinball_loss", "count"),
                median_best_iteration=("best_iteration", "median"),
            )
            eligible = aggregate.loc[aggregate["valid_folds"].eq(len(folds))]
            if eligible.empty:
                raise RuntimeError(
                    f"no complete DQ0 tuning candidate for {family}/Q{quantile:g}"
                )
            best = eligible.sort_values(
                ["mean_pinball_loss", "parameters_json"], kind="mergesort"
            ).iloc[0]
            params = json.loads(str(best["parameters_json"]))
            if family == "xgboost_quantile" and pd.notna(best["median_best_iteration"]):
                params["n_estimators"] = int(best["median_best_iteration"]) + 1
                params.pop("early_stopping_rounds", None)
            selected[family][str(quantile)] = params
            for row in rows[family_start:]:
                row["selected"] = row["parameter_index"] == int(best["parameter_index"])
                row["selected_final_parameters_json"] = order_modeling.stable_json(params)
    return selected, _rows_frame(rows), _rows_frame(manifests)


def _selection_tables(selection: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for family, params in selection["classification_parameters"].items():
        rows.append(
            {
                "task": "breach",
                "family": family,
                "model_family": family,
                "model_id": "DP0",
                "specification": "DP0",
                "quantile": np.nan,
                "tuning_baseline": "DP0",
                "selection_metric": "mean_chronological_log_loss",
                "parameters_json": order_modeling.stable_json(params),
                "development_only": True,
                "later_or_terminal_outcomes_used": False,
            }
        )
    for family, specifications in selection["calibrators"].items():
        for model_id, calibrator in specifications.items():
            rows.append(
                {
                    "task": "breach_calibration",
                    "family": family,
                    "model_family": family,
                    "model_id": model_id,
                    "specification": model_id,
                    "quantile": np.nan,
                    "tuning_baseline": model_id,
                    "selection_metric": "chronological_development_oof_log_loss_then_brier_then_simplicity",
                    "calibration_method": calibrator["method"],
                    "parameters_json": order_modeling.stable_json(calibrator),
                    "development_only": True,
                    "later_or_terminal_outcomes_used": False,
                }
            )
    for family, quantiles in selection["severity_parameters"].items():
        for quantile, params in quantiles.items():
            rows.append(
                {
                    "task": "severity",
                    "family": family,
                    "model_family": family,
                    "model_id": "DQ0",
                    "specification": "DQ0",
                    "quantile": float(quantile),
                    "tuning_baseline": "DQ0",
                    "selection_metric": "mean_chronological_pinball_loss",
                    "parameters_json": order_modeling.stable_json(params),
                    "development_only": True,
                    "later_or_terminal_outcomes_used": False,
                }
            )
    table = _rows_frame(rows)
    parameters = table.copy()
    parameters.insert(0, "analysis_id", selection["analysis_id"])
    return table, parameters


def run_development_selection(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Tune DP0/DQ0 on development only and freeze full-spec calibrators.

    Returns ``(selection, tables)``.  ``selection`` is JSON serialisable and is
    the only object needed by :func:`evaluate_direct_extension`.
    """

    frozen = dict(config) if config is not None else load_config()
    classifier, breach_tuning, breach_manifests = _development_classifier_selection(
        frame, frozen
    )
    calibrators, oof, calibration, calibration_manifests = (
        _development_calibration_selection(frame, frozen, classifier)
    )
    severity, severity_tuning, severity_manifests = _development_severity_selection(
        frame, frozen, classifier
    )
    selection: dict[str, Any] = {
        "analysis_id": frozen["analysis_id"],
        "selected_on": "development_only_before_2018-01-01",
        "direct_config_sha256": _sha256_file(CONFIG_PATH),
        "source_model_frame_path": frozen["sources"]["order_model_frame"][0],
        "source_model_frame_sha256": frozen["sources"]["order_model_frame"][1],
        "development_inner_folds": frozen["periods"]["development_inner_folds"],
        "classification_tuning_baseline": "DP0",
        "severity_tuning_baseline": "DQ0",
        "later_or_terminal_outcomes_used": False,
        "classification_parameters": classifier,
        "calibrators": calibrators,
        "severity_parameters": severity,
    }
    model_selection, parameters = _selection_tables(selection)
    manifests = pd.concat(
        [breach_manifests, calibration_manifests, severity_manifests],
        ignore_index=True,
    )
    manifests["source_model_frame_sha256"] = frozen["sources"]["order_model_frame"][1]
    manifests["direct_config_sha256"] = _sha256_file(CONFIG_PATH)
    tuning = pd.concat([breach_tuning, severity_tuning], ignore_index=True, sort=False)
    return selection, {
        "development_tuning": tuning,
        "calibration_selection": calibration,
        "model_selection": model_selection,
        "selected_parameters": parameters,
        "development_model_manifests": manifests,
        "development_oof_predictions": oof,
    }


def _evaluate_breach_primary(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    bin_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    features = breach_feature_map("full")
    bins_count = int(config["breach"]["calibration_bins"])
    for cohort in evaluation_cohorts(config):
        train, test = cohort_masks(frame, cohort, config)
        test_hash = order_modeling.order_id_hash(frame.loc[test, "order_id"])
        for family_value in config["breach"]["families"]:
            family = str(family_value)
            params = selection["classification_parameters"][family]
            for model_id, (numeric, categorical) in features.items():
                fitted = order_modeling.fit_classifier(
                    frame.loc[train],
                    frame.loc[train, "late_delivery"],
                    numeric,
                    categorical,
                    family,
                    params,
                )
                model_hash = order_modeling.fitted_model_sha256(fitted)
                raw = fitted.predict_raw(frame.loc[test])
                calibrator = _calibrator_from_dict(
                    selection["calibrators"][family][model_id]
                )
                calibrated = calibrator.predict(raw)
                prediction = pd.DataFrame(
                    {
                        "order_id": frame.loc[test, "order_id"].astype(str).to_numpy(),
                        "purchase_date": frame.loc[test, "purchase_date"].to_numpy(),
                        "period": cohort.period,
                        "cohort": cohort.cohort,
                        "cohort_month": cohort.cohort,
                        "origin": cohort.origin,
                        "family": family,
                        "model_family": family,
                        "model_id": model_id,
                        "specification": model_id,
                        "model_name": BREACH_NAMES[model_id],
                        "representation": "full",
                        "target": frame.loc[test, "late_delivery"].to_numpy(int),
                        "raw_probability": raw,
                        "calibrated_probability": calibrated,
                        "calibration_method": calibrator.method,
                        "fitted_model_sha256": model_hash,
                        "order_id_sha256": test_hash,
                    }
                )
                for block in PROFILE_BLOCKS:
                    for suffix in (
                        "support",
                        "cold_start",
                        "mapping_status",
                        "score",
                    ):
                        column = f"{block}_{suffix}"
                        prediction[column] = frame.loc[test, column].to_numpy()
                prediction_parts.append(prediction)
                for probability_type, values in (
                    ("raw", raw),
                    ("calibrated", calibrated),
                ):
                    metrics, reliability = order_modeling.classification_metrics(
                        prediction["order_id"], prediction["target"], values, bins_count
                    )
                    metric_rows.append(
                        _metric_aliases(
                            {
                                "period": cohort.period,
                                "cohort": cohort.cohort,
                                "cohort_month": cohort.cohort,
                                "origin": cohort.origin,
                                "family": family,
                                "model_id": model_id,
                                "model_name": BREACH_NAMES[model_id],
                                "representation": "full",
                                "probability_type": probability_type,
                                "probability_variant": probability_type,
                                "calibration_method": (
                                    calibrator.method
                                    if probability_type == "calibrated"
                                    else "none"
                                ),
                                "n_train": int(train.sum()),
                                "train_order_id_sha256": order_modeling.order_id_hash(
                                    frame.loc[train, "order_id"]
                                ),
                                "fitted_model_sha256": model_hash,
                                "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                                **metrics,
                            }
                        )
                    )
                    reliability.insert(0, "probability_type", probability_type)
                    reliability.insert(0, "probability_variant", probability_type)
                    reliability.insert(0, "representation", "full")
                    reliability.insert(0, "model_id", model_id)
                    reliability.insert(0, "specification", model_id)
                    reliability.insert(0, "family", family)
                    reliability.insert(0, "model_family", family)
                    reliability.insert(0, "cohort", cohort.cohort)
                    reliability.insert(0, "cohort_month", cohort.cohort)
                    reliability.insert(0, "period", cohort.period)
                    reliability["fitted_model_sha256"] = model_hash
                    reliability["order_id_sha256"] = test_hash
                    bin_parts.append(reliability)
                manifests.append(
                    _manifest_row(
                        task="breach",
                        stage=(
                            "later_evaluation"
                            if cohort.period == "later"
                            else "terminal_stress"
                        ),
                        family=family,
                        model_id=model_id,
                        representation="full",
                        fitted=fitted,
                        parameters=params,
                        numeric=numeric,
                        categorical=categorical,
                        cohort=cohort.cohort,
                        origin=cohort.origin,
                        fold=None,
                        quantile=None,
                        n_train=int(train.sum()),
                        n_evaluation=int(test.sum()),
                        train_order_ids=frame.loc[train, "order_id"],
                        evaluation_order_ids=frame.loc[test, "order_id"],
                    )
                )
    return (
        _rows_frame(metric_rows),
        pd.concat(bin_parts, ignore_index=True),
        pd.concat(prediction_parts, ignore_index=True),
        _rows_frame(manifests),
    )


def _aggregate_breach_predictions(
    predictions: pd.DataFrame, *, period: str = "later", bins: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bin_parts: list[pd.DataFrame] = []
    selected = predictions.loc[predictions["period"].eq(period)]
    for (family, model_id), group in selected.groupby(
        ["family", "model_id"], sort=True, observed=True
    ):
        constituents = group[["cohort", "fitted_model_sha256"]].drop_duplicates()
        identifiers = [
            f"{row.cohort}:{row.fitted_model_sha256}"
            for row in constituents.sort_values(
                ["cohort", "fitted_model_sha256"], kind="mergesort"
            ).itertuples(index=False)
        ]
        composite = order_modeling.composite_fitted_model_sha256(identifiers)
        cohort_hash = order_modeling.order_id_hash(group["order_id"])
        for probability_type, column in (
            ("raw", "raw_probability"),
            ("calibrated", "calibrated_probability"),
        ):
            metrics, reliability = order_modeling.classification_metrics(
                group["order_id"], group["target"], group[column], bins
            )
            rows.append(
                _metric_aliases(
                    {
                        "period": "aggregate",
                        "cohort": f"{period}_pooled",
                        "cohort_month": f"{period}_pooled",
                        "origin": np.nan,
                        "family": family,
                        "model_id": model_id,
                        "model_name": BREACH_NAMES[str(model_id)],
                        "representation": "full",
                        "probability_type": probability_type,
                        "probability_variant": probability_type,
                        "calibration_method": (
                            str(group["calibration_method"].iloc[0])
                            if probability_type == "calibrated"
                            else "none"
                        ),
                        "n_train": np.nan,
                        "train_order_id_sha256": "multiple_expanding_training_sets",
                        "fitted_model_sha256": composite,
                        "model_hash_type": "composite_fitted_models",
                        "constituent_fitted_model_count": len(identifiers),
                        **metrics,
                    }
                )
            )
            reliability.insert(0, "probability_type", probability_type)
            reliability.insert(0, "probability_variant", probability_type)
            reliability.insert(0, "representation", "full")
            reliability.insert(0, "model_id", model_id)
            reliability.insert(0, "specification", model_id)
            reliability.insert(0, "family", family)
            reliability.insert(0, "model_family", family)
            reliability.insert(0, "cohort", f"{period}_pooled")
            reliability.insert(0, "cohort_month", f"{period}_pooled")
            reliability.insert(0, "period", "aggregate")
            reliability["fitted_model_sha256"] = composite
            reliability["order_id_sha256"] = cohort_hash
            bin_parts.append(reliability)
    return _rows_frame(rows), pd.concat(bin_parts, ignore_index=True)


def _paired_breach_differences(
    predictions: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_groups(source: pd.DataFrame, aggregate: bool = False) -> None:
        for (period, cohort, family), group in source.groupby(
            ["period", "cohort", "family"], sort=True, observed=True
        ):
            for comparison, candidate_id, reference_id in BREACH_COMPARISONS:
                candidate = group.loc[group["model_id"].eq(candidate_id)].copy()
                reference = group.loc[group["model_id"].eq(reference_id)].copy()
                paired = candidate.merge(
                    reference[
                        ["order_id", "target", "calibrated_probability"]
                    ],
                    on="order_id",
                    how="inner",
                    suffixes=("_candidate", "_reference"),
                    validate="one_to_one",
                )
                if len(paired) != len(candidate) or len(paired) != len(reference):
                    raise AssertionError(
                        f"nonidentical direct paired sample: {period}/{cohort}/{family}/{comparison}"
                    )
                if not paired["target_candidate"].eq(paired["target_reference"]).all():
                    raise AssertionError("direct paired breach target mismatch")
                y = paired["target_candidate"].to_numpy(int)
                candidate_probability = paired[
                    "calibrated_probability_candidate"
                ].to_numpy(float)
                reference_probability = paired[
                    "calibrated_probability_reference"
                ].to_numpy(float)
                candidate_metrics, _ = order_modeling.classification_metrics(
                    paired["order_id"], y, candidate_probability
                )
                reference_metrics, _ = order_modeling.classification_metrics(
                    paired["order_id"], y, reference_probability
                )
                bootstrap = order_modeling.paired_calendar_block_bootstrap(
                    paired["purchase_date"],
                    y,
                    candidate_probability,
                    reference_probability,
                    replicates=int(
                        config["determinism"]["paired_block_bootstrap_replicates"]
                    ),
                    seed=order_modeling.stable_seed(
                        int(config["determinism"]["seed"]),
                        period,
                        cohort,
                        family,
                        comparison,
                    ),
                )
                order_hash = order_modeling.order_id_hash(paired["order_id"])
                rows.append(
                    {
                        "period": period,
                        "cohort": cohort,
                        "cohort_month": cohort,
                        "family": family,
                        "model_family": family,
                        "comparison": comparison,
                        "candidate_model": candidate_id,
                        "candidate_specification": candidate_id,
                        "reference_model": reference_id,
                        "reference_specification": reference_id,
                        "representation": "full",
                        "probability_type": "calibrated",
                        "probability_variant": "calibrated",
                        "n_orders": len(paired),
                        "n_obs": len(paired),
                        "n_events": int(y.sum()),
                        "order_id_sha256": order_hash,
                        "paired_order_id_sha256": order_hash,
                        "delta_log_loss": candidate_metrics["log_loss"]
                        - reference_metrics["log_loss"],
                        "delta_brier": candidate_metrics["brier"]
                        - reference_metrics["brier"],
                        "delta_brier_score": candidate_metrics["brier"]
                        - reference_metrics["brier"],
                        "delta_average_precision": candidate_metrics["average_precision"]
                        - reference_metrics["average_precision"],
                        "delta_roc_auc": candidate_metrics["roc_auc"]
                        - reference_metrics["roc_auc"],
                        "delta_top_10pct_lift": candidate_metrics["top_10pct_lift"]
                        - reference_metrics["top_10pct_lift"],
                        "delta_top10_lift": candidate_metrics["top_10pct_lift"]
                        - reference_metrics["top_10pct_lift"],
                        "delta_calibration_intercept": candidate_metrics[
                            "calibration_intercept"
                        ]
                        - reference_metrics["calibration_intercept"],
                        "delta_calibration_slope": candidate_metrics[
                            "calibration_slope"
                        ]
                        - reference_metrics["calibration_slope"],
                        "is_pooled": bool(aggregate),
                        **bootstrap,
                    }
                )

    monthly = predictions.copy()
    add_groups(monthly, False)
    pooled = predictions.loc[predictions["period"].eq("later")].copy()
    if not pooled.empty:
        pooled["period"] = "aggregate"
        pooled["cohort"] = "later_pooled"
        add_groups(pooled, True)
    return _rows_frame(rows)


def _breach_ablations(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    primary_metrics: pd.DataFrame,
    primary_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    comparisons = {candidate: comparison for comparison, candidate, _ in BREACH_COMPARISONS}
    # The full/raw rows reuse the primary fits exactly.
    raw = primary_metrics.loc[primary_metrics["probability_type"].eq("raw")]
    for _, candidate in raw.loc[raw["model_id"].ne("DP0")].iterrows():
        reference = raw.loc[
            raw["period"].eq(candidate["period"])
            & raw["cohort"].eq(candidate["cohort"])
            & raw["family"].eq(candidate["family"])
            & raw["model_id"].eq("DP0")
        ]
        if len(reference) != 1:
            raise AssertionError("full breach ablation lacks unique DP0 reference")
        ref = reference.iloc[0]
        rows.append(
            _metric_aliases(
                {
                    **candidate.to_dict(),
                    "comparison": comparisons[str(candidate["model_id"])],
                    "candidate_model": candidate["model_id"],
                    "reference_model": "DP0",
                    "ablation_id": f"{candidate['model_id']}_full",
                    "representation": "full",
                    "reference_fitted_model_sha256": ref["fitted_model_sha256"],
                    "delta_log_loss": candidate["log_loss"] - ref["log_loss"],
                    "delta_brier": candidate["brier"] - ref["brier"],
                    "delta_brier_score": candidate["brier"] - ref["brier"],
                }
            )
        )
    for cohort in evaluation_cohorts(config):
        train, test = cohort_masks(frame, cohort, config)
        for family_value in config["breach"]["families"]:
            family = str(family_value)
            params = selection["classification_parameters"][family]
            baseline_prediction = primary_predictions.loc[
                primary_predictions["period"].eq(cohort.period)
                & primary_predictions["cohort"].eq(cohort.cohort)
                & primary_predictions["family"].eq(family)
                & primary_predictions["model_id"].eq("DP0")
            ]
            if len(baseline_prediction) != int(test.sum()):
                raise AssertionError("breach ablation baseline cohort mismatch")
            baseline_metrics, _ = order_modeling.classification_metrics(
                baseline_prediction["order_id"],
                baseline_prediction["target"],
                baseline_prediction["raw_probability"],
            )
            for representation in ("score_only", "metadata_only"):
                feature_map = breach_feature_map(representation)
                for model_id in ("DPS", "DPG", "DPB"):
                    numeric, categorical = feature_map[model_id]
                    fitted = order_modeling.fit_classifier(
                        frame.loc[train],
                        frame.loc[train, "late_delivery"],
                        numeric,
                        categorical,
                        family,
                        params,
                    )
                    probability = fitted.predict_raw(frame.loc[test])
                    metrics, _ = order_modeling.classification_metrics(
                        frame.loc[test, "order_id"],
                        frame.loc[test, "late_delivery"],
                        probability,
                    )
                    rows.append(
                        _metric_aliases(
                            {
                                "period": cohort.period,
                                "cohort": cohort.cohort,
                                "cohort_month": cohort.cohort,
                                "origin": cohort.origin,
                                "family": family,
                                "model_id": model_id,
                                "model_name": BREACH_NAMES[model_id],
                                "comparison": comparisons[model_id],
                                "candidate_model": model_id,
                                "reference_model": "DP0",
                                "ablation_id": f"{model_id}_{representation}",
                                "representation": representation,
                                "probability_type": "raw",
                                "probability_variant": "raw",
                                "n_train": int(train.sum()),
                                "train_order_id_sha256": order_modeling.order_id_hash(
                                    frame.loc[train, "order_id"]
                                ),
                                "fitted_model_sha256": order_modeling.fitted_model_sha256(
                                    fitted
                                ),
                                "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                                "reference_fitted_model_sha256": str(
                                    baseline_prediction["fitted_model_sha256"].iloc[0]
                                ),
                                "delta_log_loss": metrics["log_loss"]
                                - baseline_metrics["log_loss"],
                                "delta_brier": metrics["brier"]
                                - baseline_metrics["brier"],
                                "delta_brier_score": metrics["brier"]
                                - baseline_metrics["brier"],
                                **metrics,
                            }
                        )
                    )
                    manifests.append(
                        _manifest_row(
                            task="breach",
                            stage=f"{cohort.period}_{representation}_sensitivity",
                            family=family,
                            model_id=model_id,
                            representation=representation,
                            fitted=fitted,
                            parameters=params,
                            numeric=numeric,
                            categorical=categorical,
                            cohort=cohort.cohort,
                            origin=cohort.origin,
                            fold=None,
                            quantile=None,
                            n_train=int(train.sum()),
                            n_evaluation=int(test.sum()),
                            train_order_ids=frame.loc[train, "order_id"],
                            evaluation_order_ids=frame.loc[test, "order_id"],
                        )
                    )
    table = _rows_frame(rows)
    summaries: list[dict[str, Any]] = []
    for (family, model_id, representation), group in table.loc[
        table["period"].eq("later")
    ].groupby(["family", "model_id", "representation"], sort=True, observed=True):
        summaries.append(
            {
                "period": "later_aggregate",
                "cohort": "monthly_median",
                "cohort_month": "monthly_median",
                "family": family,
                "model_family": family,
                "model_id": model_id,
                "specification": model_id,
                "model_name": BREACH_NAMES[str(model_id)],
                "comparison": comparisons[str(model_id)],
                "candidate_model": model_id,
                "reference_model": "DP0",
                "ablation_id": f"{model_id}_{representation}",
                "representation": representation,
                "probability_type": "raw",
                "probability_variant": "raw",
                "n_orders": group["n_orders"].median(),
                "n_obs": group["n_orders"].median(),
                "n_events": group["n_events"].median(),
                "log_loss": group["log_loss"].median(),
                "brier": group["brier"].median(),
                "brier_score": group["brier"].median(),
                "delta_log_loss": group["delta_log_loss"].median(),
                "delta_brier": group["delta_brier"].median(),
                "delta_brier_score": group["delta_brier"].median(),
                "both_improved_month_count": int(
                    (group["delta_log_loss"].lt(0) & group["delta_brier"].lt(0)).sum()
                ),
                "order_id_sha256": "monthly_medians_no_single_order_hash",
            }
        )
    if summaries:
        table = pd.concat([table, _rows_frame(summaries)], ignore_index=True, sort=False)
    return table, _rows_frame(manifests)


def _support_stratum(values: pd.Series) -> pd.Series:
    support = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [
                support.lt(5),
                support.between(5, 9),
                support.between(10, 19),
                support.ge(20),
            ],
            ["0-4", "5-9", "10-19", "20+"],
            default="missing",
        ),
        index=values.index,
        dtype="string",
    )


def _breach_support_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    block_names = {"DPS": "seller", "DPG": "geographic", "DPB": "combined"}
    for comparison, candidate_id, reference_id in BREACH_COMPARISONS:
        blocks = BREACH_MODEL_BLOCKS[candidate_id]
        support_columns = [f"{block}_support" for block in blocks]
        cold_columns = [f"{block}_cold_start" for block in blocks]
        candidate = predictions.loc[predictions["model_id"].eq(candidate_id)].copy()
        reference = predictions.loc[predictions["model_id"].eq(reference_id)][
            [
                "period",
                "cohort",
                "family",
                "order_id",
                "target",
                "calibrated_probability",
            ]
        ]
        paired = candidate.merge(
            reference,
            on=["period", "cohort", "family", "order_id"],
            how="inner",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        if len(paired) != len(candidate):
            raise AssertionError(f"support-stratum paired sample mismatch for {comparison}")
        if not paired["target_candidate"].eq(paired["target_reference"]).all():
            raise AssertionError("support-stratum target mismatch")
        paired["minimum_support"] = paired[support_columns].apply(
            pd.to_numeric, errors="coerce"
        ).min(axis=1)
        paired["any_cold_start"] = (
            paired[cold_columns].fillna(False).astype(bool).any(axis=1)
        )
        paired["support_stratum"] = _support_stratum(paired["minimum_support"])
        paired.loc[paired["any_cold_start"], "support_stratum"] = "cold_start"
        for keys, group in paired.groupby(
            ["period", "cohort", "family", "support_stratum"],
            sort=True,
            observed=True,
        ):
            y = group["target_candidate"].to_numpy(int)
            candidate_metric, _ = order_modeling.classification_metrics(
                group["order_id"], y, group["calibrated_probability_candidate"]
            )
            reference_metric, _ = order_modeling.classification_metrics(
                group["order_id"], y, group["calibrated_probability_reference"]
            )
            order_hash = order_modeling.order_id_hash(group["order_id"])
            rows.append(
                {
                    "task": "breach",
                    "block": block_names[candidate_id],
                    "period": keys[0],
                    "cohort": keys[1],
                    "cohort_month": keys[1],
                    "family": keys[2],
                    "model_family": keys[2],
                    "comparison": comparison,
                    "candidate_model": candidate_id,
                    "model_id": candidate_id,
                    "specification": candidate_id,
                    "reference_model": reference_id,
                    "representation": "full",
                    "probability_type": "calibrated",
                    "support_stratum": keys[3],
                    "minimum_support_threshold": (
                        20 if keys[3] == "20+" else np.nan
                    ),
                    "median_support": group["minimum_support"].median(),
                    "cold_start_share": group["any_cold_start"].mean(),
                    "n_orders": len(group),
                    "n_obs": len(group),
                    "n_events": int(y.sum()),
                    "order_id_sha256": order_hash,
                    "delta_log_loss": candidate_metric["log_loss"]
                    - reference_metric["log_loss"],
                    "delta_brier": candidate_metric["brier"]
                    - reference_metric["brier"],
                    "delta_brier_score": candidate_metric["brier"]
                    - reference_metric["brier"],
                    "candidate_log_loss": candidate_metric["log_loss"],
                    "reference_log_loss": reference_metric["log_loss"],
                    "candidate_brier": candidate_metric["brier"],
                    "reference_brier": reference_metric["brier"],
                }
            )
    return _rows_frame(rows)


def _breach_evidence_summary(
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    support: pd.DataFrame,
    ablations: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    later_pairs = paired.loc[paired["period"].eq("later")]
    calibrated = metrics.loc[
        metrics["period"].eq("later")
        & metrics["probability_type"].eq("calibrated")
    ]
    thresholds = config["guards"]
    for comparison, candidate_id, reference_id in BREACH_COMPARISONS:
        for family_value in config["breach"]["families"]:
            family = str(family_value)
            primary = later_pairs.loc[
                later_pairs["family"].eq(family)
                & later_pairs["comparison"].eq(comparison)
            ].sort_values("cohort", kind="mergesort")
            primary_available = len(primary) == 6
            median_ll = primary["delta_log_loss"].median() if len(primary) else np.nan
            median_brier = primary["delta_brier"].median() if len(primary) else np.nan
            both_count = int(
                (primary["delta_log_loss"].lt(0) & primary["delta_brier"].lt(0)).sum()
            )

            candidate = calibrated.loc[
                calibrated["family"].eq(family)
                & calibrated["model_id"].eq(candidate_id)
            ]
            reference = calibrated.loc[
                calibrated["family"].eq(family)
                & calibrated["model_id"].eq(reference_id)
            ][["cohort", "wace", "calibration_slope"]]
            calibration_pair = candidate.merge(
                reference,
                on="cohort",
                how="inner",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            median_wace_delta = (
                calibration_pair["wace_candidate"]
                - calibration_pair["wace_reference"]
            ).median() if len(calibration_pair) else np.nan
            median_slope_error_delta = (
                (calibration_pair["calibration_slope_candidate"] - 1).abs()
                - (calibration_pair["calibration_slope_reference"] - 1).abs()
            ).median() if len(calibration_pair) else np.nan
            calibration_available = bool(
                len(calibration_pair) == 6
                and pd.notna(median_wace_delta)
                and pd.notna(median_slope_error_delta)
            )
            calibration_guard = bool(
                calibration_available
                and median_wace_delta
                <= float(thresholds["calibration_wace_worsening_tolerance"])
                and median_slope_error_delta
                <= float(
                    thresholds[
                        "calibration_absolute_slope_error_worsening_tolerance"
                    ]
                )
            )

            high = support.loc[
                support["period"].eq("later")
                & support["family"].eq(family)
                & support["comparison"].eq(comparison)
                & support["support_stratum"].eq("20+")
            ]
            high_available = len(high) == 6
            high_ll = high["delta_log_loss"].median() if len(high) else np.nan
            high_brier = high["delta_brier"].median() if len(high) else np.nan
            high_reversal = bool(
                high_available
                and pd.notna(high_ll)
                and pd.notna(high_brier)
                and high_ll > 0
                and high_brier > 0
            )
            high_guard = bool(high_available and not high_reversal)

            ablation = ablations.loc[
                ablations["period"].eq("later")
                & ablations["family"].eq(family)
                & ablations["model_id"].eq(candidate_id)
            ]
            full = ablation.loc[ablation["representation"].eq("full")]
            score = ablation.loc[ablation["representation"].eq("score_only")]
            metadata = ablation.loc[ablation["representation"].eq("metadata_only")]
            score_available = len(full) == len(score) == len(metadata) == 6
            score_only_median_delta = (
                score["delta_log_loss"].median() if len(score) else np.nan
            )
            full_beats_metadata_ll = bool(
                score_available
                and full["log_loss"].median() < metadata["log_loss"].median()
            )
            full_no_brier_worsening = bool(
                score_available
                and full["brier"].median() <= metadata["brier"].median()
            )
            score_guard = bool(
                score_available
                and (
                    score_only_median_delta < 0
                    or (full_beats_metadata_ll and full_no_brier_worsening)
                )
            )
            all_guards_available = bool(
                calibration_available and high_available and score_available
            )
            all_guards_pass = bool(
                calibration_guard and high_guard and score_guard
            )
            if not primary_available:
                evidence_status = "Blocked"
                reason = "missing_one_or_more_frozen_later_months"
            elif median_ll >= 0 or median_brier >= 0 or both_count <= 2:
                evidence_status = "Not-supported"
                reason = "nonimproving_median_or_at_most_two_both_improved_months"
            elif both_count == 3:
                evidence_status = "Mixed"
                reason = "exactly_three_both_improved_months"
            elif all_guards_available and all_guards_pass:
                evidence_status = "Supported"
                reason = "both_medians_improve_at_least_four_months_all_guards_pass"
            else:
                evidence_status = "Mixed"
                reason = "required_guard_unavailable_or_failed"
            rows.append(
                {
                    "task": "breach",
                    "family": family,
                    "model_family": family,
                    "comparison": comparison,
                    "candidate_model": candidate_id,
                    "candidate_specification": candidate_id,
                    "reference_model": reference_id,
                    "reference_specification": reference_id,
                    "representation": "full",
                    "later_month_count": len(primary),
                    "primary_evidence_available": primary_available,
                    "median_delta_log_loss": median_ll,
                    "median_delta_brier": median_brier,
                    "both_improved_month_count": both_count,
                    "high_support_guard_available": high_available,
                    "high_support_median_delta_log_loss": high_ll,
                    "high_support_median_delta_brier": high_brier,
                    "high_support_material_reversal": high_reversal,
                    "high_support_no_material_reversal": high_guard,
                    "calibration_guard_available": calibration_available,
                    "median_delta_wace": median_wace_delta,
                    "median_delta_absolute_calibration_slope_error": median_slope_error_delta,
                    "calibration_not_systematically_worse": calibration_guard,
                    "calibration_systematically_worse": (
                        calibration_available and not calibration_guard
                    ),
                    "score_guard_available": score_available,
                    "score_only_median_delta_log_loss": score_only_median_delta,
                    "full_beats_metadata_log_loss": full_beats_metadata_ll,
                    "full_no_metadata_brier_worsening": full_no_brier_worsening,
                    "score_contributes": score_guard,
                    "benefit_not_metadata_only": score_guard,
                    "all_guards_available": all_guards_available,
                    "all_guards_pass": all_guards_pass,
                    "evidence_status": evidence_status,
                    "evidence_label": evidence_status,
                    "evidence_reason": reason,
                }
            )
    return _rows_frame(rows)


def _evaluate_severity_primary(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    features = severity_feature_map("full")
    for cohort in evaluation_cohorts(config):
        train_all, test_all = cohort_masks(frame, cohort, config)
        train = train_all & frame["positive_late_days"].gt(0)
        test = test_all & frame["positive_late_days"].gt(0)
        if not train.any() or not test.any():
            raise RuntimeError(f"severity cohort has no breached orders: {cohort.cohort}")
        test_hash = order_modeling.order_id_hash(frame.loc[test, "order_id"])
        y_train = frame.loc[train, "positive_late_days"].to_numpy(float)
        y_test = frame.loc[test, "positive_late_days"].to_numpy(float)
        for quantile_value in config["severity"]["quantiles"]:
            quantile = float(quantile_value)
            unconditional = float(np.quantile(y_train, quantile, method="linear"))
            unconditional_prediction = np.full(len(y_test), unconditional)
            unconditional_loss = order_modeling.pinball_loss(
                y_test, unconditional_prediction, quantile
            )
            for family_value in config["severity"]["families"]:
                family = str(family_value)
                params = selection["severity_parameters"][family][str(quantile)]
                local: list[tuple[str, Any, np.ndarray]] = []
                for model_id, (numeric, categorical) in features.items():
                    fitted = order_modeling.fit_quantile_model(
                        frame.loc[train],
                        frame.loc[train, "positive_late_days"],
                        numeric,
                        categorical,
                        family,
                        quantile,
                        params,
                    )
                    prediction = fitted.predict(frame.loc[test])
                    local.append((model_id, fitted, prediction))
                    manifests.append(
                        _manifest_row(
                            task="severity",
                            stage=(
                                "later_evaluation"
                                if cohort.period == "later"
                                else "terminal_stress"
                            ),
                            family=family,
                            model_id=model_id,
                            representation="full",
                            fitted=fitted,
                            parameters=params,
                            numeric=numeric,
                            categorical=categorical,
                            cohort=cohort.cohort,
                            origin=cohort.origin,
                            fold=None,
                            quantile=quantile,
                            n_train=int(train.sum()),
                            n_evaluation=int(test.sum()),
                            train_order_ids=frame.loc[train, "order_id"],
                            evaluation_order_ids=frame.loc[test, "order_id"],
                        )
                    )
                baseline_candidates = [values for mid, _, values in local if mid == "DQ0"]
                if len(baseline_candidates) != 1:
                    raise AssertionError("severity evaluation lacks unique DQ0 prediction")
                baseline_prediction = baseline_candidates[0]
                baseline_loss = order_modeling.pinball_loss(
                    y_test, baseline_prediction, quantile
                )
                for model_id, fitted, prediction in local:
                    metric = order_modeling.quantile_metrics(y_test, prediction, quantile)
                    model_hash = order_modeling.fitted_model_sha256(fitted)
                    skill = (
                        1 - float(metric["pinball_loss"]) / baseline_loss
                        if baseline_loss > 0
                        else np.nan
                    )
                    row = _metric_aliases(
                        {
                            "period": cohort.period,
                            "cohort": cohort.cohort,
                            "cohort_month": cohort.cohort,
                            "origin": cohort.origin,
                            "family": family,
                            "model_id": model_id,
                            "model_name": SEVERITY_NAMES[model_id],
                            "quantile": quantile,
                            "representation": "full",
                            "n_train_breaches": int(train.sum()),
                            "train_order_id_sha256": order_modeling.order_id_hash(
                                frame.loc[train, "order_id"]
                            ),
                            "fitted_model_sha256": model_hash,
                            "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                            "unconditional_training_quantile": unconditional,
                            "unconditional_reference_loss": unconditional_loss,
                            "baseline_pinball_loss": baseline_loss,
                            "dq0_reference_loss": baseline_loss,
                            "skill": skill,
                            "skill_vs_dq0": skill,
                            "skill_vs_unconditional": (
                                1 - float(metric["pinball_loss"]) / unconditional_loss
                                if unconditional_loss > 0
                                else np.nan
                            ),
                            "nominal_coverage": quantile,
                            "order_id_sha256": test_hash,
                            **metric,
                        }
                    )
                    rows.append(row)
                    part = pd.DataFrame(
                        {
                            "order_id": frame.loc[test, "order_id"].astype(str).to_numpy(),
                            "purchase_date": frame.loc[test, "purchase_date"].to_numpy(),
                            "period": cohort.period,
                            "cohort": cohort.cohort,
                            "cohort_month": cohort.cohort,
                            "origin": cohort.origin,
                            "family": family,
                            "model_family": family,
                            "model_id": model_id,
                            "specification": model_id,
                            "model_name": SEVERITY_NAMES[model_id],
                            "quantile": quantile,
                            "representation": "full",
                            "actual_positive_late_days": y_test,
                            "prediction": prediction,
                            "dq0_prediction": baseline_prediction,
                            "unconditional_prediction": unconditional_prediction,
                            "fitted_model_sha256": model_hash,
                            "order_id_sha256": test_hash,
                        }
                    )
                    for block in PROFILE_BLOCKS:
                        for suffix in ("support", "cold_start", "mapping_status"):
                            column = f"{block}_{suffix}"
                            part[column] = frame.loc[test, column].to_numpy()
                    predictions.append(part)
    return _rows_frame(rows), pd.concat(predictions, ignore_index=True), _rows_frame(manifests)


def _aggregate_severity_predictions(
    predictions: pd.DataFrame, *, period: str = "later"
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = predictions.loc[predictions["period"].eq(period)]
    for (family, model_id, quantile_value), group in selected.groupby(
        ["family", "model_id", "quantile"], sort=True, observed=True
    ):
        quantile = float(quantile_value)
        constituents = group[["cohort", "fitted_model_sha256"]].drop_duplicates()
        identifiers = [
            f"{row.cohort}:{row.fitted_model_sha256}"
            for row in constituents.sort_values(
                ["cohort", "fitted_model_sha256"], kind="mergesort"
            ).itertuples(index=False)
        ]
        composite = order_modeling.composite_fitted_model_sha256(identifiers)
        metric = order_modeling.quantile_metrics(
            group["actual_positive_late_days"], group["prediction"], quantile
        )
        baseline_loss = order_modeling.pinball_loss(
            group["actual_positive_late_days"], group["dq0_prediction"], quantile
        )
        unconditional_loss = order_modeling.pinball_loss(
            group["actual_positive_late_days"],
            group["unconditional_prediction"],
            quantile,
        )
        skill = (
            1 - float(metric["pinball_loss"]) / baseline_loss
            if baseline_loss > 0
            else np.nan
        )
        rows.append(
            _metric_aliases(
                {
                    "period": "aggregate",
                    "cohort": f"{period}_pooled",
                    "cohort_month": f"{period}_pooled",
                    "origin": np.nan,
                    "family": family,
                    "model_id": model_id,
                    "model_name": SEVERITY_NAMES[str(model_id)],
                    "quantile": quantile,
                    "representation": "full",
                    "n_train_breaches": np.nan,
                    "train_order_id_sha256": "multiple_expanding_training_sets",
                    "fitted_model_sha256": composite,
                    "model_hash_type": "composite_fitted_models",
                    "constituent_fitted_model_count": len(identifiers),
                    "unconditional_training_quantile": np.nan,
                    "unconditional_reference_loss": unconditional_loss,
                    "baseline_pinball_loss": baseline_loss,
                    "dq0_reference_loss": baseline_loss,
                    "skill": skill,
                    "skill_vs_dq0": skill,
                    "skill_vs_unconditional": (
                        1 - float(metric["pinball_loss"]) / unconditional_loss
                        if unconditional_loss > 0
                        else np.nan
                    ),
                    "nominal_coverage": quantile,
                    "order_id_sha256": order_modeling.order_id_hash(group["order_id"]),
                    **metric,
                }
            )
        )
    return _rows_frame(rows)


def _severity_ablations(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    primary_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    comparisons = {
        candidate: comparison for comparison, candidate, _ in SEVERITY_COMPARISONS
    }
    for _, row in primary_metrics.loc[primary_metrics["model_id"].ne("DQ0")].iterrows():
        rows.append(
            {
                **row.to_dict(),
                "comparison": comparisons[str(row["model_id"])],
                "candidate_model": row["model_id"],
                "reference_model": "DQ0",
                "ablation_id": f"{row['model_id']}_full",
                "representation": "full",
            }
        )
    for cohort in evaluation_cohorts(config):
        train_all, test_all = cohort_masks(frame, cohort, config)
        train = train_all & frame["positive_late_days"].gt(0)
        test = test_all & frame["positive_late_days"].gt(0)
        for quantile_value in config["severity"]["quantiles"]:
            quantile = float(quantile_value)
            for family_value in config["severity"]["families"]:
                family = str(family_value)
                params = selection["severity_parameters"][family][str(quantile)]
                baseline = primary_metrics.loc[
                    primary_metrics["period"].eq(cohort.period)
                    & primary_metrics["cohort"].eq(cohort.cohort)
                    & primary_metrics["family"].eq(family)
                    & primary_metrics["model_id"].eq("DQ0")
                    & pd.to_numeric(primary_metrics["quantile"]).eq(quantile)
                ]
                if len(baseline) != 1:
                    raise AssertionError("severity ablation lacks unique DQ0 reference")
                baseline_row = baseline.iloc[0]
                for model_id in ("DQS", "DQG", "DQB"):
                    numeric, categorical = severity_feature_map("score_only")[model_id]
                    fitted = order_modeling.fit_quantile_model(
                        frame.loc[train],
                        frame.loc[train, "positive_late_days"],
                        numeric,
                        categorical,
                        family,
                        quantile,
                        params,
                    )
                    prediction = fitted.predict(frame.loc[test])
                    metric = order_modeling.quantile_metrics(
                        frame.loc[test, "positive_late_days"], prediction, quantile
                    )
                    baseline_loss = float(baseline_row["pinball_loss"])
                    skill = (
                        1 - float(metric["pinball_loss"]) / baseline_loss
                        if baseline_loss > 0
                        else np.nan
                    )
                    rows.append(
                        _metric_aliases(
                            {
                                "period": cohort.period,
                                "cohort": cohort.cohort,
                                "cohort_month": cohort.cohort,
                                "origin": cohort.origin,
                                "family": family,
                                "model_id": model_id,
                                "model_name": SEVERITY_NAMES[model_id],
                                "quantile": quantile,
                                "comparison": comparisons[model_id],
                                "candidate_model": model_id,
                                "reference_model": "DQ0",
                                "ablation_id": f"{model_id}_score_only",
                                "representation": "score_only",
                                "n_train_breaches": int(train.sum()),
                                "train_order_id_sha256": order_modeling.order_id_hash(
                                    frame.loc[train, "order_id"]
                                ),
                                "fitted_model_sha256": order_modeling.fitted_model_sha256(
                                    fitted
                                ),
                                "model_hash_type": "sha256_pickle_protocol_5_fitted_wrapper",
                                "baseline_pinball_loss": baseline_loss,
                                "dq0_reference_loss": baseline_loss,
                                "skill": skill,
                                "skill_vs_dq0": skill,
                                "nominal_coverage": quantile,
                                "order_id_sha256": order_modeling.order_id_hash(
                                    frame.loc[test, "order_id"]
                                ),
                                **metric,
                            }
                        )
                    )
                    manifests.append(
                        _manifest_row(
                            task="severity",
                            stage=f"{cohort.period}_score_only_sensitivity",
                            family=family,
                            model_id=model_id,
                            representation="score_only",
                            fitted=fitted,
                            parameters=params,
                            numeric=numeric,
                            categorical=categorical,
                            cohort=cohort.cohort,
                            origin=cohort.origin,
                            fold=None,
                            quantile=quantile,
                            n_train=int(train.sum()),
                            n_evaluation=int(test.sum()),
                            train_order_ids=frame.loc[train, "order_id"],
                            evaluation_order_ids=frame.loc[test, "order_id"],
                        )
                    )
    table = _rows_frame(rows)
    summaries: list[dict[str, Any]] = []
    for (family, model_id, quantile, representation), group in table.loc[
        table["period"].eq("later")
    ].groupby(
        ["family", "model_id", "quantile", "representation"],
        sort=True,
        observed=True,
    ):
        summaries.append(
            {
                "period": "later_aggregate",
                "cohort": "monthly_median",
                "cohort_month": "monthly_median",
                "family": family,
                "model_family": family,
                "model_id": model_id,
                "specification": model_id,
                "model_name": SEVERITY_NAMES[str(model_id)],
                "quantile": float(quantile),
                "comparison": comparisons[str(model_id)],
                "candidate_model": model_id,
                "reference_model": "DQ0",
                "ablation_id": f"{model_id}_{representation}",
                "representation": representation,
                "pinball_loss": group["pinball_loss"].median(),
                "baseline_pinball_loss": group["baseline_pinball_loss"].median(),
                "skill": group["skill"].median(),
                "skill_vs_dq0": group["skill"].median(),
                "empirical_coverage": group["empirical_coverage"].median(),
                "coverage": group["empirical_coverage"].median(),
                "coverage_error": group["coverage_error"].median(),
                "favourable_month_count": int(group["skill"].ge(0).sum()),
                "n_orders": group["n_orders"].median(),
                "n_obs": group["n_orders"].median(),
                "order_id_sha256": "monthly_medians_no_single_order_hash",
            }
        )
    if summaries:
        table = pd.concat([table, _rows_frame(summaries)], ignore_index=True, sort=False)
    return table, _rows_frame(manifests)


def _severity_support_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = {"DQS": "seller", "DQG": "geographic", "DQB": "combined"}
    for comparison, candidate_id, reference_id in SEVERITY_COMPARISONS:
        blocks = SEVERITY_MODEL_BLOCKS[candidate_id]
        support_columns = [f"{block}_support" for block in blocks]
        cold_columns = [f"{block}_cold_start" for block in blocks]
        subset = predictions.loc[predictions["model_id"].eq(candidate_id)].copy()
        subset["minimum_support"] = subset[support_columns].apply(
            pd.to_numeric, errors="coerce"
        ).min(axis=1)
        subset["any_cold_start"] = (
            subset[cold_columns].fillna(False).astype(bool).any(axis=1)
        )
        subset["support_stratum"] = _support_stratum(subset["minimum_support"])
        subset.loc[subset["any_cold_start"], "support_stratum"] = "cold_start"
        for keys, group in subset.groupby(
            ["period", "cohort", "family", "quantile", "support_stratum"],
            sort=True,
            observed=True,
        ):
            quantile = float(keys[3])
            metric = order_modeling.quantile_metrics(
                group["actual_positive_late_days"], group["prediction"], quantile
            )
            baseline_loss = order_modeling.pinball_loss(
                group["actual_positive_late_days"], group["dq0_prediction"], quantile
            )
            skill = (
                1 - float(metric["pinball_loss"]) / baseline_loss
                if baseline_loss > 0
                else np.nan
            )
            rows.append(
                _metric_aliases(
                    {
                        "task": "severity",
                        "block": names[candidate_id],
                        "period": keys[0],
                        "cohort": keys[1],
                        "cohort_month": keys[1],
                        "family": keys[2],
                        "quantile": quantile,
                        "comparison": comparison,
                        "candidate_model": candidate_id,
                        "model_id": candidate_id,
                        "specification": candidate_id,
                        "reference_model": reference_id,
                        "representation": "full",
                        "support_stratum": keys[4],
                        "median_support": group["minimum_support"].median(),
                        "cold_start_share": group["any_cold_start"].mean(),
                        "baseline_pinball_loss": baseline_loss,
                        "delta_pinball_loss": float(metric["pinball_loss"])
                        - baseline_loss,
                        "skill": skill,
                        "skill_vs_dq0": skill,
                        "order_id_sha256": order_modeling.order_id_hash(group["order_id"]),
                        **metric,
                    }
                )
            )
    result = _rows_frame(rows)
    result["high_support_guard_available"] = False
    result["support_ge20_gain_present"] = False
    result["high_support_guard"] = False
    result["gain_only_low_support"] = True
    high = result.loc[
        result["period"].eq("later") & result["support_stratum"].eq("20+")
    ]
    for (family, comparison, quantile), group in high.groupby(
        ["family", "comparison", "quantile"], sort=True, observed=True
    ):
        available = len(group) == 6
        guard = bool(
            available and group["skill"].median() > 0 and group["skill"].ge(0).sum() >= 4
        )
        mask = (
            result["family"].eq(family)
            & result["comparison"].eq(comparison)
            & pd.to_numeric(result["quantile"]).eq(float(quantile))
        )
        result.loc[mask, "high_support_guard_available"] = available
        result.loc[mask, "support_ge20_gain_present"] = guard
        result.loc[mask, "high_support_guard"] = guard
        result.loc[mask, "gain_only_low_support"] = available and not guard
    return result


def _severity_evidence_summary(
    monthly: pd.DataFrame,
    support: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    later = monthly.loc[monthly["period"].eq("later")]
    tolerance = float(config["guards"]["q90_absolute_coverage_error_worsening_tolerance"])
    for comparison, candidate_id, reference_id in SEVERITY_COMPARISONS:
        for family_value in config["severity"]["families"]:
            family = str(family_value)
            for quantile_value in config["severity"]["quantiles"]:
                quantile = float(quantile_value)
                candidate = later.loc[
                    later["family"].eq(family)
                    & later["model_id"].eq(candidate_id)
                    & pd.to_numeric(later["quantile"]).eq(quantile)
                ].sort_values("cohort", kind="mergesort")
                reference = later.loc[
                    later["family"].eq(family)
                    & later["model_id"].eq(reference_id)
                    & pd.to_numeric(later["quantile"]).eq(quantile)
                ][["cohort", "coverage_error"]]
                primary_available = len(candidate) == 6
                median_skill = candidate["skill"].median() if len(candidate) else np.nan
                favourable_count = int(candidate["skill"].ge(0).sum())
                high = support.loc[
                    support["period"].eq("later")
                    & support["family"].eq(family)
                    & support["comparison"].eq(comparison)
                    & pd.to_numeric(support["quantile"]).eq(quantile)
                    & support["support_stratum"].eq("20+")
                ]
                support_available = len(high) == 6
                high_median_skill = high["skill"].median() if len(high) else np.nan
                high_favourable_count = int(high["skill"].ge(0).sum())
                support_guard = bool(
                    support_available
                    and high_median_skill > 0
                    and high_favourable_count >= 4
                )
                coverage_pair = candidate.merge(
                    reference,
                    on="cohort",
                    how="inner",
                    suffixes=("_candidate", "_reference"),
                    validate="one_to_one",
                )
                coverage_deterioration = (
                    (
                        coverage_pair["coverage_error_candidate"].abs()
                        - coverage_pair["coverage_error_reference"].abs()
                    ).median()
                    if len(coverage_pair)
                    else np.nan
                )
                coverage_available = bool(
                    quantile != 0.9
                    or (len(coverage_pair) == 6 and pd.notna(coverage_deterioration))
                )
                coverage_guard = bool(
                    quantile != 0.9
                    or (
                        coverage_available
                        and coverage_deterioration <= tolerance
                    )
                )
                if not primary_available:
                    evidence_status = "Blocked"
                    reason = "missing_one_or_more_frozen_later_months"
                elif median_skill <= 0 or favourable_count < 4:
                    evidence_status = "Not-supported"
                    reason = "nonpositive_median_skill_or_fewer_than_four_favourable_months"
                elif support_available and not support_guard:
                    evidence_status = "Not-supported"
                    reason = "gain_confined_to_lower_support_orders"
                elif support_guard and coverage_guard:
                    evidence_status = "Supported"
                    reason = "positive_median_four_months_support_and_coverage_guards_pass"
                else:
                    evidence_status = "Mixed"
                    reason = "required_guard_unavailable_or_failed"
                rows.append(
                    {
                        "task": "severity",
                        "family": family,
                        "model_family": family,
                        "quantile": quantile,
                        "comparison": comparison,
                        "candidate_model": candidate_id,
                        "candidate_specification": candidate_id,
                        "reference_model": reference_id,
                        "reference_specification": reference_id,
                        "representation": "full",
                        "later_month_count": len(candidate),
                        "primary_evidence_available": primary_available,
                        "median_skill": median_skill,
                        "favourable_month_count": favourable_count,
                        "high_support_guard_available": support_available,
                        "high_support_median_skill": high_median_skill,
                        "high_support_favourable_month_count": high_favourable_count,
                        "support_ge20_gain_present": support_guard,
                        "high_support_guard": support_guard,
                        "gain_only_low_support": support_available and not support_guard,
                        "coverage_guard_available": coverage_available,
                        "median_absolute_coverage_error_deterioration": coverage_deterioration,
                        "coverage_not_materially_worse": coverage_guard,
                        "coverage_materially_worse": coverage_available and not coverage_guard,
                        "all_guards_available": support_available and coverage_available,
                        "all_guards_pass": support_guard and coverage_guard,
                        "evidence_status": evidence_status,
                        "evidence_label": evidence_status,
                        "evidence_reason": reason,
                    }
                )
    return _rows_frame(rows)


def _join_breach_evidence(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = table.copy()
    if "model_id" not in result.columns:
        return result
    evidence = summary.rename(columns={"candidate_model": "model_id"})
    evidence_columns = [
        column
        for column in evidence.columns
        if column not in {"task", "model_family", "candidate_specification"}
    ]
    result = result.merge(
        evidence[evidence_columns],
        on=["family", "model_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )
    if paired is not None and {"period", "cohort"}.issubset(result.columns):
        differences = paired.rename(columns={"candidate_model": "model_id"})
        delta_columns = [
            "period",
            "cohort",
            "family",
            "model_id",
            "delta_log_loss",
            "delta_brier",
            "delta_brier_score",
            "delta_average_precision",
            "delta_roc_auc",
            "delta_top_10pct_lift",
            "delta_top10_lift",
            "delta_calibration_intercept",
            "delta_calibration_slope",
            "delta_log_loss_lower",
            "delta_log_loss_upper",
            "delta_brier_lower",
            "delta_brier_upper",
        ]
        result = result.merge(
            differences[[column for column in delta_columns if column in differences]],
            on=["period", "cohort", "family", "model_id"],
            how="left",
            validate="many_to_one",
        )
    baseline = result["model_id"].eq("DP0")
    if "evidence_status" not in result:
        result["evidence_status"] = np.nan
    if "evidence_label" not in result:
        result["evidence_label"] = np.nan
    result.loc[baseline, ["evidence_status", "evidence_label"]] = "Reference"
    result["comparison"] = result.get("comparison", pd.Series(np.nan, index=result.index))
    result.loc[baseline, "comparison"] = "DP0_reference"
    return result


def _join_severity_evidence(
    table: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    result = table.copy()
    if "model_id" not in result.columns:
        return result
    evidence = summary.rename(columns={"candidate_model": "model_id"})
    evidence_columns = [
        column
        for column in evidence.columns
        if column
        not in {"task", "model_family", "candidate_specification", "representation"}
    ]
    result = result.merge(
        evidence[evidence_columns],
        on=["family", "model_id", "quantile"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )
    baseline = result["model_id"].eq("DQ0")
    if "evidence_status" not in result:
        result["evidence_status"] = np.nan
    if "evidence_label" not in result:
        result["evidence_label"] = np.nan
    result.loc[baseline, ["evidence_status", "evidence_label"]] = "Reference"
    result["comparison"] = result.get("comparison", pd.Series(np.nan, index=result.index))
    result.loc[baseline, "comparison"] = "DQ0_reference"
    return result


def _evidence_label_table(
    breach: pd.DataFrame, severity: pd.DataFrame
) -> pd.DataFrame:
    breach_rows = breach.copy()
    breach_rows["label_namespace"] = "RQ3_direct_order_evidence"
    breach_rows["evidence_role"] = "primary_direct_operational_estimand"
    breach_rows["outcome"] = "breach_probability"
    breach_rows["profile_block"] = breach_rows["candidate_model"].map(
        {"DPS": "seller", "DPG": "state_od", "DPB": "both"}
    )
    breach_rows["specification"] = breach_rows["candidate_model"]
    breach_rows["quantile"] = np.nan
    severity_rows = severity.copy()
    severity_rows["label_namespace"] = "RQ3_direct_order_evidence"
    severity_rows["evidence_role"] = "primary_direct_operational_estimand"
    severity_rows["outcome"] = "conditional_positive_lateness"
    severity_rows["profile_block"] = severity_rows["candidate_model"].map(
        {"DQS": "seller", "DQG": "state_od", "DQB": "both"}
    )
    severity_rows["specification"] = severity_rows["candidate_model"]
    return pd.concat([breach_rows, severity_rows], ignore_index=True, sort=False)


def _terminal_long_table(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    breach_metrics: pd.DataFrame,
    breach_pairs: pd.DataFrame,
    severity_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    terminal_breach = breach_metrics.loc[
        breach_metrics["period"].eq("terminal")
        & breach_metrics["probability_type"].eq("calibrated")
    ]
    terminal_pairs = breach_pairs.loc[breach_pairs["period"].eq("terminal")]
    for _, row in terminal_breach.iterrows():
        model_id = str(row["model_id"])
        pair = terminal_pairs.loc[
            terminal_pairs["family"].eq(row["family"])
            & terminal_pairs["candidate_model"].eq(model_id)
        ]
        metric_values = {
            "log_loss": row.get("log_loss"),
            "brier": row.get("brier"),
            "average_precision": row.get("average_precision"),
            "roc_auc": row.get("roc_auc"),
            "top_10pct_lift": row.get("top_10pct_lift"),
            "calibration_intercept": row.get("calibration_intercept"),
            "calibration_slope": row.get("calibration_slope"),
            "wace": row.get("wace"),
        }
        if len(pair) == 1:
            metric_values["delta_log_loss"] = pair["delta_log_loss"].iloc[0]
            metric_values["delta_brier"] = pair["delta_brier"].iloc[0]
        for metric, estimate in metric_values.items():
            rows.append(
                {
                    "task": "breach",
                    "outcome": "breach_probability",
                    "period": "terminal",
                    "cohort": row["cohort"],
                    "cohort_month": row["cohort"],
                    "family": row["family"],
                    "model_family": row["family"],
                    "model_id": model_id,
                    "specification": model_id,
                    "representation": "full",
                    "quantile": np.nan,
                    "probability_type": "calibrated",
                    "metric": metric,
                    "estimate": estimate,
                    "n_orders": row["n_orders"],
                    "n_obs": row["n_orders"],
                    "n_events": row["n_events"],
                    "order_id_sha256": row["order_id_sha256"],
                    "fitted_model_sha256": row["fitted_model_sha256"],
                    "evidence_status": row.get("evidence_status"),
                    "evidence_label": row.get("evidence_label"),
                }
            )
    terminal_severity = severity_metrics.loc[severity_metrics["period"].eq("terminal")]
    for _, row in terminal_severity.iterrows():
        metric_values = {
            "pinball_loss": row.get("pinball_loss"),
            "pinball_skill": row.get("skill"),
            "skill": row.get("skill"),
            "empirical_coverage": row.get("empirical_coverage"),
            "coverage": row.get("empirical_coverage"),
            "coverage_error": row.get("coverage_error"),
        }
        for metric, estimate in metric_values.items():
            rows.append(
                {
                    "task": "severity",
                    "outcome": "conditional_positive_lateness",
                    "period": "terminal",
                    "cohort": row["cohort"],
                    "cohort_month": row["cohort"],
                    "family": row["family"],
                    "model_family": row["family"],
                    "model_id": row["model_id"],
                    "specification": row["model_id"],
                    "representation": "full",
                    "quantile": row["quantile"],
                    "metric": metric,
                    "estimate": estimate,
                    "n_orders": row["n_orders"],
                    "n_obs": row["n_orders"],
                    "n_events": row["n_orders"],
                    "order_id_sha256": row["order_id_sha256"],
                    "fitted_model_sha256": row["fitted_model_sha256"],
                    "evidence_status": row.get("evidence_status"),
                    "evidence_label": row.get("evidence_label"),
                }
            )
    terminal_cohort = [
        cohort for cohort in evaluation_cohorts(config) if cohort.period == "terminal"
    ]
    if len(terminal_cohort) != 1:
        raise AssertionError("exactly one terminal cohort is required")
    _, terminal = cohort_masks(frame, terminal_cohort[0], config)
    terminal_frame = frame.loc[terminal]
    breach_frame = terminal_frame.loc[terminal_frame["positive_late_days"].gt(0)]
    terminal_hash = order_modeling.order_id_hash(terminal_frame["order_id"])
    context = {
        "n_orders": len(terminal_frame),
        "n_breaches": int(terminal_frame["late_delivery"].sum()),
        "breach_prevalence": float(terminal_frame["late_delivery"].mean()),
        "mean_positive_lateness_days": float(breach_frame["positive_late_days"].mean()),
        "median_positive_lateness_days": float(
            breach_frame["positive_late_days"].median()
        ),
        "q90_positive_lateness_days": float(
            breach_frame["positive_late_days"].quantile(0.9)
        ),
    }
    for metric, estimate in context.items():
        rows.append(
            {
                "task": "regime_context",
                "outcome": "terminal_regime_context",
                "period": "terminal",
                "cohort": terminal_cohort[0].cohort,
                "cohort_month": terminal_cohort[0].cohort,
                "family": "all",
                "model_family": "all",
                "model_id": "REGIME_CONTEXT",
                "specification": "REGIME_CONTEXT",
                "representation": "not_applicable",
                "quantile": np.nan,
                "metric": metric,
                "estimate": estimate,
                "n_orders": len(terminal_frame),
                "n_obs": len(terminal_frame),
                "n_events": int(terminal_frame["late_delivery"].sum()),
                "order_id_sha256": terminal_hash,
                "fitted_model_sha256": "not_applicable",
                "evidence_status": "Terminal stress only",
                "evidence_label": "Terminal stress only",
            }
        )
    for block in PROFILE_BLOCKS:
        for status, group in terminal_frame.groupby(
            f"{block}_mapping_status", sort=True, observed=True, dropna=False
        ):
            status_text = "missing" if pd.isna(status) else str(status)
            rows.append(
                {
                    "task": "profile_availability",
                    "outcome": "profile_label_availability",
                    "period": "terminal",
                    "cohort": terminal_cohort[0].cohort,
                    "cohort_month": terminal_cohort[0].cohort,
                    "family": "profile",
                    "model_family": "profile",
                    "model_id": block,
                    "specification": block,
                    "representation": "audit",
                    "quantile": np.nan,
                    "mapping_status": status_text,
                    "metric": f"mapping_share_{status_text}",
                    "estimate": len(group) / len(terminal_frame),
                    "n_orders": len(group),
                    "n_obs": len(group),
                    "n_events": int(group["late_delivery"].sum()),
                    "median_support": group[f"{block}_support"].median(),
                    "order_id_sha256": order_modeling.order_id_hash(group["order_id"]),
                    "fitted_model_sha256": "not_applicable",
                    "evidence_status": "Audit only",
                    "evidence_label": "Audit only",
                }
            )
    return _rows_frame(rows)


def _validate_selection(selection: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if selection.get("analysis_id") != config["analysis_id"]:
        raise AssertionError("selection belongs to a different analysis")
    if bool(selection.get("later_or_terminal_outcomes_used", True)):
        raise AssertionError("selection does not attest development-only tuning")
    if set(selection.get("classification_parameters", {})) != set(
        config["breach"]["families"]
    ):
        raise AssertionError("classification selection family mismatch")
    if set(selection.get("calibrators", {})) != set(config["breach"]["families"]):
        raise AssertionError("calibrator family mismatch")
    for family in config["breach"]["families"]:
        if set(selection["calibrators"][family]) != set(BREACH_MODEL_BLOCKS):
            raise AssertionError(f"calibrator specification mismatch for {family}")
    if set(selection.get("severity_parameters", {})) != set(
        config["severity"]["families"]
    ):
        raise AssertionError("severity selection family mismatch")
    expected_quantiles = {str(float(value)) for value in config["severity"]["quantiles"]}
    for family in config["severity"]["families"]:
        if set(selection["severity_parameters"][family]) != expected_quantiles:
            raise AssertionError(f"severity quantile selection mismatch for {family}")


def evaluate_direct_extension(
    frame: pd.DataFrame,
    selection: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit and evaluate the frozen direct ladders on later and terminal cohorts."""

    frozen = dict(config) if config is not None else load_config()
    _validate_selection(selection, frozen)
    breach_metrics, breach_bins, breach_predictions, breach_manifests = (
        _evaluate_breach_primary(frame, frozen, selection)
    )
    breach_pooled, pooled_bins = _aggregate_breach_predictions(
        breach_predictions,
        period="later",
        bins=int(frozen["breach"]["calibration_bins"]),
    )
    breach_pairs = _paired_breach_differences(breach_predictions, frozen)
    breach_ablations, breach_ablation_manifests = _breach_ablations(
        frame, frozen, selection, breach_metrics, breach_predictions
    )
    breach_support = _breach_support_strata(breach_predictions)
    breach_summary = _breach_evidence_summary(
        breach_pairs, breach_metrics, breach_support, breach_ablations, frozen
    )
    breach_metrics = _join_breach_evidence(
        breach_metrics, breach_summary, breach_pairs
    )
    breach_pooled = _join_breach_evidence(
        breach_pooled, breach_summary, breach_pairs
    )
    breach_pairs = breach_pairs.merge(
        breach_summary,
        on=["family", "comparison", "candidate_model", "reference_model"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )
    breach_support = breach_support.merge(
        breach_summary,
        on=["family", "comparison", "candidate_model", "reference_model"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )

    severity_metrics, severity_predictions, severity_manifests = (
        _evaluate_severity_primary(frame, frozen, selection)
    )
    severity_pooled = _aggregate_severity_predictions(
        severity_predictions, period="later"
    )
    severity_ablations, severity_ablation_manifests = _severity_ablations(
        frame, frozen, selection, severity_metrics
    )
    severity_support = _severity_support_strata(severity_predictions)
    severity_summary = _severity_evidence_summary(
        severity_metrics, severity_support, frozen
    )
    severity_metrics = _join_severity_evidence(severity_metrics, severity_summary)
    severity_pooled = _join_severity_evidence(severity_pooled, severity_summary)
    severity_support = severity_support.merge(
        severity_summary,
        on=["family", "quantile", "comparison", "candidate_model", "reference_model"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )

    terminal = _terminal_long_table(
        frame, frozen, breach_metrics, breach_pairs, severity_metrics
    )
    evaluation_manifests = pd.concat(
        [
            breach_manifests,
            breach_ablation_manifests,
            severity_manifests,
            severity_ablation_manifests,
        ],
        ignore_index=True,
        sort=False,
    )
    evaluation_manifests["source_model_frame_sha256"] = frozen["sources"][
        "order_model_frame"
    ][1]
    evaluation_manifests["direct_config_sha256"] = _sha256_file(CONFIG_PATH)
    reliability = pd.concat([breach_bins, pooled_bins], ignore_index=True, sort=False)
    calibration = pd.concat(
        [breach_metrics, breach_pooled], ignore_index=True, sort=False
    )
    severity_coverage = pd.concat(
        [severity_metrics, severity_pooled], ignore_index=True, sort=False
    )
    return {
        "breach_monthly": breach_metrics.loc[
            breach_metrics["period"].eq("later")
        ].reset_index(drop=True),
        "breach_pooled": breach_pooled.reset_index(drop=True),
        "breach_calibration": calibration.reset_index(drop=True),
        "breach_support": breach_support.reset_index(drop=True),
        "breach_paired_differences": breach_pairs.reset_index(drop=True),
        "breach_ablations": breach_ablations.reset_index(drop=True),
        "breach_reliability_bins": reliability.reset_index(drop=True),
        "severity_monthly": severity_metrics.loc[
            severity_metrics["period"].eq("later")
        ].reset_index(drop=True),
        "severity_pooled": severity_pooled.reset_index(drop=True),
        "severity_coverage": severity_coverage.reset_index(drop=True),
        "severity_support": severity_support.reset_index(drop=True),
        "severity_ablations": severity_ablations.reset_index(drop=True),
        "terminal": terminal.reset_index(drop=True),
        "evidence_labels": _evidence_label_table(
            breach_summary, severity_summary
        ).reset_index(drop=True),
        "evaluation_model_manifests": evaluation_manifests.reset_index(drop=True),
        "breach_predictions": breach_predictions.reset_index(drop=True),
        "severity_predictions": severity_predictions.reset_index(drop=True),
    }


OUTPUT_FILES: dict[str, str] = {
    "development_tuning": "DIRECT_DEVELOPMENT_TUNING.csv",
    "calibration_selection": "DIRECT_CALIBRATION_SELECTION.csv",
    "model_selection": "MODEL_SELECTION.csv",
    "selected_parameters": "DIRECT_MODEL_PARAMETERS.csv",
    "breach_monthly": "DIRECT_BREACH_MONTHLY.csv",
    "breach_pooled": "DIRECT_BREACH_POOLED.csv",
    "breach_calibration": "DIRECT_BREACH_CALIBRATION.csv",
    "breach_support": "DIRECT_BREACH_SUPPORT_STRATA.csv",
    "breach_paired_differences": "DIRECT_BREACH_PAIRED_DIFFERENCES.csv",
    "breach_ablations": "DIRECT_BREACH_ABLATIONS.csv",
    "breach_reliability_bins": "DIRECT_BREACH_RELIABILITY_BINS.csv",
    "severity_monthly": "DIRECT_SEVERITY_MONTHLY.csv",
    "severity_pooled": "DIRECT_SEVERITY_POOLED.csv",
    "severity_coverage": "DIRECT_SEVERITY_COVERAGE.csv",
    "severity_support": "DIRECT_SEVERITY_SUPPORT_STRATA.csv",
    "severity_ablations": "DIRECT_SEVERITY_ABLATIONS.csv",
    "terminal": "DIRECT_TERMINAL.csv",
    "evidence_labels": "EVIDENCE_LABELS.csv",
    "model_manifests": "DIRECT_MODEL_MANIFESTS.csv",
}

WORKING_OUTPUT_FILES: dict[str, str] = {
    "development_oof_predictions": "DIRECT_DEVELOPMENT_OOF_PREDICTIONS.csv.gz",
    "breach_predictions": "DIRECT_BREACH_PREDICTIONS.csv.gz",
    "severity_predictions": "DIRECT_SEVERITY_PREDICTIONS.csv.gz",
}


def _sorted_for_write(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "task",
        "stage",
        "period",
        "cohort",
        "family",
        "model_id",
        "quantile",
        "representation",
        "probability_type",
        "comparison",
        "support_stratum",
        "fold",
        "parameter_index",
        "bin",
        "order_id",
        "metric",
    ]
    keys = [column for column in preferred if column in frame.columns]
    if not keys or frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(
        keys, kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def _atomic_csv(frame: pd.DataFrame, path: Path, float_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        output = _sorted_for_write(frame)
        if path.suffix == ".gz":
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0
                ) as compressed:
                    with io.TextIOWrapper(
                        compressed,
                        encoding="utf-8",
                        newline="",
                        write_through=True,
                    ) as text_handle:
                        output.to_csv(
                            text_handle,
                            index=False,
                            float_format=float_format,
                            date_format="%Y-%m-%d",
                            na_rep="",
                        )
        else:
            output.to_csv(
                temporary,
                index=False,
                float_format=float_format,
                date_format="%Y-%m-%d",
                na_rep="",
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_modeling_outputs(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path = WORKSPACE,
    *,
    selection: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Atomically persist all supplied modelling tables below the extension root."""

    frozen = dict(config) if config is not None else load_config()
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise ValueError(
            f"direct-extension outputs must remain under {WORKSPACE}"
        ) from exc
    paths: dict[str, Path] = {}
    for key, filename in OUTPUT_FILES.items():
        if key not in tables:
            continue
        path = destination / filename
        _atomic_csv(
            tables[key], path, str(frozen["determinism"]["float_format"])
        )
        paths[key] = path
    for key, filename in WORKING_OUTPUT_FILES.items():
        if key not in tables:
            continue
        path = destination / "working" / filename
        _atomic_csv(
            tables[key], path, str(frozen["determinism"]["float_format"])
        )
        paths[key] = path
    if selection is not None:
        selection_path = destination / "DIRECT_MODEL_SELECTION_FREEZE.json"
        _atomic_json(selection, selection_path)
        paths["selection_freeze"] = selection_path
    return paths


def run_and_write_modeling(
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path = WORKSPACE,
    *,
    frame: pd.DataFrame | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run development selection, frozen evaluation, and deterministic writes.

    The return value has three keys: ``selection`` (JSON-serialisable),
    ``tables`` (all in-memory frames), and ``paths`` (persisted artifacts).
    Passing a selection is supported for an audited resume; it must carry the
    explicit development-only attestation validated by this module.
    """

    frozen = dict(config) if config is not None else load_config()
    model_frame = (
        load_and_validate_frame(frozen) if frame is None else frame.copy()
    )
    if selection is None:
        frozen_selection, development = run_development_selection(
            model_frame, frozen
        )
    else:
        frozen_selection = dict(selection)
        _validate_selection(frozen_selection, frozen)
        model_selection, selected_parameters = _selection_tables(frozen_selection)
        development = {
            "model_selection": model_selection,
            "selected_parameters": selected_parameters,
        }
    # Materialise the development-only freeze before any later/terminal fit is
    # launched.  If evaluation fails, the pre-evaluation selection receipt is
    # still available for the stage failure audit.
    selection_destination = Path(output_dir).resolve()
    try:
        selection_destination.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise ValueError(
            f"direct-extension outputs must remain under {WORKSPACE}"
        ) from exc
    selection_path = selection_destination / "DIRECT_MODEL_SELECTION_FREEZE.json"
    _atomic_json(frozen_selection, selection_path)
    evaluation = evaluate_direct_extension(
        model_frame, frozen_selection, frozen
    )
    tables: dict[str, pd.DataFrame] = {**development, **evaluation}
    manifest_parts = [
        tables[key]
        for key in ("development_model_manifests", "evaluation_model_manifests")
        if key in tables and not tables[key].empty
    ]
    if not manifest_parts:
        raise RuntimeError("no fitted-model manifests were generated")
    tables["model_manifests"] = pd.concat(
        manifest_parts, ignore_index=True, sort=False
    )
    required_manifest_columns = {
        "parameters_json",
        "numeric_features_json",
        "categorical_features_json",
        "train_order_id_sha256",
        "evaluation_order_id_sha256",
        "ordered_feature_sha256",
        "fitted_model_sha256",
    }
    missing_manifest = required_manifest_columns - set(tables["model_manifests"].columns)
    if missing_manifest:
        raise AssertionError(
            f"model manifests missing audit columns: {sorted(missing_manifest)}"
        )
    paths = write_modeling_outputs(
        tables,
        output_dir,
        selection=None,
        config=frozen,
    )
    paths["selection_freeze"] = selection_path
    return {"selection": frozen_selection, "tables": tables, "paths": paths}


__all__ = [
    "BASELINE_FEATURE",
    "BREACH_MODEL_BLOCKS",
    "CONFIG_PATH",
    "OUTPUT_FILES",
    "PROFILE_BLOCKS",
    "SEVERITY_MODEL_BLOCKS",
    "WORKING_OUTPUT_FILES",
    "WORKSPACE",
    "breach_feature_map",
    "chronological_masks",
    "cohort_masks",
    "evaluate_direct_extension",
    "evaluation_cohorts",
    "load_and_validate_frame",
    "load_config",
    "profile_features",
    "run_and_write_modeling",
    "run_development_selection",
    "severity_feature_map",
    "write_modeling_outputs",
]
