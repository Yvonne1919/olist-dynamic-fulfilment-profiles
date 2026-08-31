from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


EPS = 1e-6


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(map(str, (base_seed, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def order_id_hash(order_ids: Iterable[object]) -> str:
    values = sorted(map(str, order_ids))
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def fitted_model_sha256(fitted_model: object) -> str:
    """Fingerprint the complete fitted wrapper, including its feature order.

    Pickling the project wrapper captures the fitted preprocessor, estimator,
    selected feature order, family, and any frozen best iteration in one
    deterministic payload for the pinned runtime.
    """

    return hashlib.sha256(pickle.dumps(fitted_model, protocol=5)).hexdigest()


def composite_fitted_model_sha256(model_hashes: Iterable[object]) -> str:
    """Return an order-independent fingerprint of constituent fitted models."""

    hashes = sorted({str(value) for value in model_hashes if pd.notna(value) and str(value)})
    if not hashes:
        raise ValueError("at least one fitted-model hash is required")
    return hashlib.sha256(("\n".join(hashes) + "\n").encode("utf-8")).hexdigest()


def make_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str], *, rare_min_frequency: int = 20,
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=rare_min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, list(numeric)), ("categorical", categorical_pipe, list(categorical))],
        remainder="drop",
        sparse_threshold=1.0,
    )


@dataclass
class FittedClassifier:
    family: str
    preprocessor: ColumnTransformer
    model: object
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    best_iteration: int | None = None

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame)
        probability = np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)
        return np.clip(probability, EPS, 1 - EPS)


def fit_classifier(
    frame: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    numeric: Sequence[str],
    categorical: Sequence[str],
    family: str,
    params: Mapping[str, object],
    *,
    validation_frame: pd.DataFrame | None = None,
    validation_target: Sequence[int] | np.ndarray | pd.Series | None = None,
) -> FittedClassifier:
    y = np.asarray(target, dtype=int)
    if len(frame) != len(y) or len(np.unique(y)) < 2:
        raise ValueError("classifier training sample must be paired and contain both classes")
    preprocessor = make_preprocessor(numeric, categorical)
    train_matrix = preprocessor.fit_transform(frame)
    best_iteration: int | None = None
    if family == "logistic_l2":
        model = LogisticRegression(
            penalty="l2",
            C=float(params["C"]),
            solver=str(params.get("solver", "liblinear")),
            max_iter=int(params.get("max_iter", 1000)),
            class_weight=None,
            random_state=int(params.get("random_state", 20260823)),
        )
        model.fit(train_matrix, y)
    elif family == "xgboost":
        kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": str(params.get("tree_method", "hist")),
            "learning_rate": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "min_child_weight": float(params["min_child_weight"]),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "n_estimators": int(params["n_estimators"]),
            "n_jobs": int(params.get("n_jobs", 4)),
            "random_state": int(params.get("random_state", 20260823)),
        }
        if validation_frame is not None and validation_target is not None:
            kwargs["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 50))
        model = XGBClassifier(**kwargs)
        if validation_frame is not None and validation_target is not None:
            valid_matrix = preprocessor.transform(validation_frame)
            model.fit(train_matrix, y, eval_set=[(valid_matrix, np.asarray(validation_target, dtype=int))], verbose=False)
            best_iteration = int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None
        else:
            model.fit(train_matrix, y, verbose=False)
    else:
        raise ValueError(f"unsupported classifier family: {family}")
    return FittedClassifier(
        family=family,
        preprocessor=preprocessor,
        model=model,
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        best_iteration=best_iteration,
    )


@dataclass
class FrozenCalibrator:
    method: str
    platt_intercept: float | None = None
    platt_slope: float | None = None
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()

    def predict(self, probability: Sequence[float] | np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
        if self.method == "none":
            return p
        if self.method == "platt":
            if self.platt_intercept is None or self.platt_slope is None:
                raise RuntimeError("incomplete frozen Platt calibrator")
            eta = self.platt_intercept + self.platt_slope * np.log(p / (1 - p))
            return np.clip(1 / (1 + np.exp(-eta)), EPS, 1 - EPS)
        if self.method == "isotonic":
            if not self.isotonic_x:
                raise RuntimeError("incomplete frozen isotonic calibrator")
            return np.clip(np.interp(p, self.isotonic_x, self.isotonic_y), EPS, 1 - EPS)
        raise ValueError(self.method)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "platt_intercept": self.platt_intercept,
            "platt_slope": self.platt_slope,
            "isotonic_x": list(self.isotonic_x),
            "isotonic_y": list(self.isotonic_y),
        }


def fit_calibrator(method: str, probability: Sequence[float], target: Sequence[int]) -> FrozenCalibrator:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    y = np.asarray(target, dtype=int)
    if len(p) != len(y) or len(np.unique(y)) < 2:
        raise ValueError("calibration sample must be paired and contain both classes")
    if method == "none":
        return FrozenCalibrator("none")
    if method == "platt":
        x = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        return FrozenCalibrator("platt", float(model.intercept_[0]), float(model.coef_[0, 0]))
    if method == "isotonic":
        model = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
        model.fit(p, y)
        return FrozenCalibrator(
            "isotonic",
            isotonic_x=tuple(map(float, model.X_thresholds_)),
            isotonic_y=tuple(map(float, model.y_thresholds_)),
        )
    raise ValueError(method)


def select_calibration_method(oof: pd.DataFrame) -> tuple[FrozenCalibrator, pd.DataFrame]:
    required = {"fold", "target", "raw_probability"}
    if not required.issubset(oof.columns):
        raise KeyError(sorted(required - set(oof.columns)))
    folds = sorted(pd.to_numeric(oof["fold"], errors="raise").astype(int).unique())
    if len(folds) < 2:
        raise ValueError("calibration choice requires at least two chronological folds")
    rows: list[dict[str, object]] = []
    for method in ("none", "platt", "isotonic"):
        for fold in folds[1:]:
            previous = oof.loc[pd.to_numeric(oof["fold"]).lt(fold)]
            current = oof.loc[pd.to_numeric(oof["fold"]).eq(fold)]
            try:
                calibrator = fit_calibrator(method, previous["raw_probability"], previous["target"])
                probability = calibrator.predict(current["raw_probability"])
                valid = len(np.unique(current["target"])) == 2
                ll = float(log_loss(current["target"], probability, labels=[0, 1])) if valid else np.nan
                bs = float(brier_score_loss(current["target"], probability)) if len(current) else np.nan
                reason = ""
            except Exception as exc:  # explicit invalid receipt; never silent fallback
                ll = bs = np.nan
                reason = f"{type(exc).__name__}:{exc}"
            rows.append({"method": method, "evaluation_fold": fold, "log_loss": ll, "brier": bs, "invalid_reason": reason})
    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby("method", as_index=False)
        .agg(mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"), valid_folds=("log_loss", "count"))
    )
    complexity = {"none": 0, "platt": 1, "isotonic": 2}
    aggregate["complexity"] = aggregate["method"].map(complexity)
    eligible = aggregate.loc[aggregate["valid_folds"].eq(len(folds) - 1) & aggregate["mean_log_loss"].notna()].copy()
    if eligible.empty:
        raise RuntimeError("no valid development-only calibration method")
    chosen = eligible.sort_values(["mean_log_loss", "mean_brier", "complexity", "method"], kind="mergesort").iloc[0]
    final = fit_calibrator(str(chosen["method"]), oof["raw_probability"], oof["target"])
    detail = detail.merge(aggregate, on="method", how="left", validate="many_to_one")
    detail["selected"] = detail["method"].eq(final.method)
    return final, detail


def calibration_intercept_slope(target: Sequence[int], probability: Sequence[float]) -> tuple[float, float, str]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    if len(y) < 50 or len(np.unique(y)) < 2 or np.std(p) <= 0:
        return np.nan, np.nan, "insufficient_or_constant"
    x = sm.add_constant(np.log(p / (1 - p)), has_constant="add")
    try:
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, disp=0)
        return float(fit.params[0]), float(fit.params[1]), ""
    except Exception as exc:
        return np.nan, np.nan, f"{type(exc).__name__}:{exc}"


def reliability_bins(
    order_ids: Sequence[object], target: Sequence[int], probability: Sequence[float], bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"order_id": list(map(str, order_ids)), "target": np.asarray(target, dtype=int), "probability": np.asarray(probability, dtype=float)}
    ).sort_values(["probability", "order_id"], kind="mergesort").reset_index(drop=True)
    n = len(frame)
    if n == 0:
        return pd.DataFrame(columns=["bin", "n", "positives", "prevalence", "mean_probability", "min_probability", "max_probability", "absolute_calibration_error", "weight"])
    frame["bin"] = np.minimum((np.arange(n) * bins // n) + 1, bins)
    result = frame.groupby("bin", observed=True, as_index=False).agg(
        n=("target", "size"), positives=("target", "sum"), prevalence=("target", "mean"),
        mean_probability=("probability", "mean"), min_probability=("probability", "min"), max_probability=("probability", "max"),
    )
    result["absolute_calibration_error"] = (result["prevalence"] - result["mean_probability"]).abs()
    result["weight"] = result["n"] / result["n"].sum()
    return result


def _top_fraction_metrics(order_ids: Sequence[object], target: np.ndarray, probability: np.ndarray, fraction: float) -> tuple[float, float]:
    frame = pd.DataFrame({"order_id": list(map(str, order_ids)), "target": target, "probability": probability})
    frame = frame.sort_values(["probability", "order_id"], ascending=[False, True], kind="mergesort")
    k = max(1, int(math.ceil(len(frame) * fraction)))
    selected = frame.iloc[:k]
    prevalence = float(frame["target"].mean())
    lift = float(selected["target"].mean() / prevalence) if prevalence > 0 else np.nan
    recall = float(selected["target"].sum() / frame["target"].sum()) if frame["target"].sum() > 0 else np.nan
    return lift, recall


def classification_metrics(
    order_ids: Sequence[object], target: Sequence[int], probability: Sequence[float], bins: int = 10,
) -> tuple[dict[str, object], pd.DataFrame]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    if len(y) != len(p) or len(y) != len(order_ids):
        raise ValueError("classification metrics require identical paired rows")
    has_both = len(np.unique(y)) == 2
    intercept, slope, calibration_reason = calibration_intercept_slope(y, p)
    table = reliability_bins(order_ids, y, p, bins)
    top5_lift, _ = _top_fraction_metrics(order_ids, y, p, 0.05)
    top10_lift, top10_recall = _top_fraction_metrics(order_ids, y, p, 0.10)
    result: dict[str, object] = {
        "n_orders": len(y),
        "n_events": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "average_precision": float(average_precision_score(y, p)) if has_both else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if has_both else np.nan,
        "top_5pct_lift": top5_lift,
        "top_10pct_lift": top10_lift,
        "top_10pct_recall": top10_recall,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "wace": float((table["weight"] * table["absolute_calibration_error"]).sum()) if not table.empty else np.nan,
        "calibration_invalid_reason": calibration_reason,
        "order_id_sha256": order_id_hash(order_ids),
    }
    return result, table


def paired_loss_differences(target: Sequence[int], candidate: Sequence[float], reference: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=int)
    pc = np.clip(np.asarray(candidate, dtype=float), EPS, 1 - EPS)
    pr = np.clip(np.asarray(reference, dtype=float), EPS, 1 - EPS)
    if not (len(y) == len(pc) == len(pr)):
        raise ValueError("paired loss arrays differ in length")
    ll_c = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    ll_r = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    return ll_c - ll_r, (pc - y) ** 2 - (pr - y) ** 2


def paired_calendar_block_bootstrap(
    purchase_timestamp: Sequence[object],
    target: Sequence[int],
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    replicates: int = 500,
    seed: int = 20260823,
) -> dict[str, float | int | str]:
    ll_diff, brier_diff = paired_loss_differences(target, candidate, reference)
    dates = pd.to_datetime(pd.Series(purchase_timestamp), errors="raise")
    weeks = dates.dt.to_period("W-SUN").astype(str)
    grouped = pd.DataFrame({"week": weeks, "ll": ll_diff, "brier": brier_diff}).groupby("week", sort=True).agg(
        n=("ll", "size"), ll_sum=("ll", "sum"), brier_sum=("brier", "sum")
    )
    if len(grouped) < 2:
        return {
            "replicates": replicates, "valid_replicates": 0, "n_blocks": len(grouped),
            "delta_log_loss_lower": np.nan, "delta_log_loss_upper": np.nan,
            "delta_brier_lower": np.nan, "delta_brier_upper": np.nan,
            "invalid_reason": "fewer_than_two_calendar_week_blocks",
        }
    arrays = grouped[["n", "ll_sum", "brier_sum"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    ll_values = np.empty(replicates, dtype=float)
    bs_values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = arrays[rng.integers(0, len(arrays), size=len(arrays))]
        denominator = sampled[:, 0].sum()
        ll_values[index] = sampled[:, 1].sum() / denominator
        bs_values[index] = sampled[:, 2].sum() / denominator
    return {
        "replicates": replicates,
        "valid_replicates": replicates,
        "n_blocks": len(grouped),
        "delta_log_loss_lower": float(np.quantile(ll_values, 0.025)),
        "delta_log_loss_upper": float(np.quantile(ll_values, 0.975)),
        "delta_brier_lower": float(np.quantile(bs_values, 0.025)),
        "delta_brier_upper": float(np.quantile(bs_values, 0.975)),
        "invalid_reason": "",
    }


@dataclass
class FittedQuantileModel:
    family: str
    quantile: float
    preprocessor: ColumnTransformer
    model: object
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    best_iteration: int | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.model.predict(self.preprocessor.transform(frame)), dtype=float)
        return np.maximum(values, 0.0)


def fit_quantile_model(
    frame: pd.DataFrame,
    target: Sequence[float],
    numeric: Sequence[str],
    categorical: Sequence[str],
    family: str,
    quantile: float,
    params: Mapping[str, object],
    *,
    validation_frame: pd.DataFrame | None = None,
    validation_target: Sequence[float] | None = None,
) -> FittedQuantileModel:
    y = np.asarray(target, dtype=float)
    if len(frame) != len(y) or len(y) == 0 or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("quantile training requires paired finite positive-lateness rows")
    preprocessor = make_preprocessor(numeric, categorical)
    train_matrix = preprocessor.fit_transform(frame)
    best_iteration: int | None = None
    if family == "linear_quantile":
        model = QuantileRegressor(quantile=float(quantile), alpha=float(params["alpha"]), solver=str(params.get("solver", "highs")))
        model.fit(train_matrix, y)
    elif family == "xgboost_quantile":
        objective = str(params.get("objective", ""))
        eval_metric = str(params.get("eval_metric", ""))
        quantile_alpha = float(params.get("quantile_alpha", np.nan))
        if objective != "reg:quantileerror":
            raise ValueError(f"xgboost quantile objective mismatch: {objective!r}")
        if eval_metric != "quantile":
            raise ValueError(f"xgboost quantile eval_metric mismatch: {eval_metric!r}")
        if not np.isclose(quantile_alpha, float(quantile), rtol=0.0, atol=1e-12):
            raise ValueError(
                f"xgboost quantile_alpha mismatch: {quantile_alpha!r} != {float(quantile)!r}"
            )
        kwargs = {
            "objective": objective,
            "quantile_alpha": quantile_alpha,
            "eval_metric": eval_metric,
            "tree_method": str(params.get("tree_method", "hist")),
            "learning_rate": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "min_child_weight": float(params["min_child_weight"]),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "n_estimators": int(params["n_estimators"]),
            "n_jobs": int(params.get("n_jobs", 4)),
            "random_state": int(params.get("random_state", 20260823)),
        }
        if validation_frame is not None and validation_target is not None:
            kwargs["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 50))
        model = XGBRegressor(**kwargs)
        if validation_frame is not None and validation_target is not None:
            valid_matrix = preprocessor.transform(validation_frame)
            model.fit(train_matrix, y, eval_set=[(valid_matrix, np.asarray(validation_target, dtype=float))], verbose=False)
            best_iteration = int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None
        else:
            model.fit(train_matrix, y, verbose=False)
    else:
        raise ValueError(family)
    return FittedQuantileModel(family, float(quantile), preprocessor, model, tuple(numeric), tuple(categorical), best_iteration)


def pinball_loss(target: Sequence[float], prediction: Sequence[float], quantile: float) -> float:
    y = np.asarray(target, dtype=float)
    q = np.asarray(prediction, dtype=float)
    if len(y) != len(q) or len(y) == 0:
        return np.nan
    residual = y - q
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def quantile_metrics(target: Sequence[float], prediction: Sequence[float], quantile: float) -> dict[str, float | int]:
    y = np.asarray(target, dtype=float)
    q = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    if len(y) != len(q):
        raise ValueError("quantile metrics require paired rows")
    exceedance = y - q
    exceeded = exceedance > 0
    return {
        "n_orders": len(y),
        "pinball_loss": pinball_loss(y, q, quantile),
        "empirical_coverage": float(np.mean(y <= q)) if len(y) else np.nan,
        "coverage_error": float(np.mean(y <= q) - quantile) if len(y) else np.nan,
        "median_prediction": float(np.median(q)) if len(q) else np.nan,
        "mean_prediction": float(np.mean(q)) if len(q) else np.nan,
        "median_actual": float(np.median(y)) if len(y) else np.nan,
        "mean_actual": float(np.mean(y)) if len(y) else np.nan,
        "mean_exceedance": float(np.mean(exceedance[exceeded])) if exceeded.any() else 0.0,
        "p90_absolute_error": float(np.quantile(np.abs(y - q), 0.90)) if len(y) else np.nan,
    }


__all__ = [
    "EPS", "FittedClassifier", "FittedQuantileModel", "FrozenCalibrator",
    "calibration_intercept_slope", "classification_metrics", "composite_fitted_model_sha256",
    "fit_calibrator", "fit_classifier", "fit_quantile_model", "fitted_model_sha256",
    "make_preprocessor", "order_id_hash",
    "paired_calendar_block_bootstrap", "paired_loss_differences", "pinball_loss",
    "quantile_metrics", "reliability_bins", "select_calibration_method", "stable_json",
    "stable_seed",
]
