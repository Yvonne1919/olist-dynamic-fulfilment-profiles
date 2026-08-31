from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import OptimizeWarning
from scipy.stats import beta, rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/dynamic_profile_profile_validation_v1"
CONFIG_PATH = OUT / "PROFILE_FROZEN_CONFIG.json"
ASSEMBLER = ROOT / "analysis/profile_pivot_phase2a/scripts/data_pipeline.py"
ASSEMBLER_SHA = "0c4cad3c99db268292253abd26e2070ccf2a286337bcfe9fb76fb3768c44ab8d"
V11 = ROOT / "analysis/dynamic_profile_eda_v1_1"
PROTECTED = {
    "dynamic_profile_eda_v1": ROOT / "analysis/dynamic_profile_eda_v1",
    "dynamic_profile_eda_v1_1": V11,
    "profile_pivot_phase2a": ROOT / "analysis/profile_pivot_phase2a",
    "profile_pivot_phase1_audit": ROOT / "analysis/profile_pivot_phase1_audit",
    "docs_thesis": ROOT / "docs/thesis",
    "report_thesis": ROOT / "report/thesis",
    "results": ROOT / "results",
    "src": ROOT / "src",
}
CONTROL_FILES = (
    ROOT / "AGENTS.md",
    ROOT / "PROJECT_CONTEXT.md",
    ROOT / "RESULTS_REGISTRY.md",
    ROOT / "DECISION_LOG.md",
)

TARGET_SPECS: dict[str, dict[str, object]] = {
    "handling_level": {
        "kind": "continuous", "value": "handling_level_value",
        "raw_value": "handling_duration", "available": "handling_available_at",
        "granularities": ("seller_id",), "lags": (14, 21),
    },
    "handling_tail": {
        "kind": "binary", "value": "handling_tail",
        "raw_value": "handling_duration", "available": "handling_available_at",
        "granularities": ("seller_id",), "lags": (14, 21),
    },
    "transit_level": {
        "kind": "continuous", "value": "transit_level_value",
        "raw_value": "transit_duration", "available": "transit_available_at",
        "granularities": ("state_od", "region_od"), "lags": (30, 45),
    },
    "transit_tail": {
        "kind": "binary", "value": "transit_tail",
        "raw_value": "transit_duration", "available": "transit_available_at",
        "granularities": ("state_od", "region_od"), "lags": (30, 45),
    },
    "final_breach": {
        "kind": "binary", "value": "late_delivery",
        "raw_value": "promise_error_days", "available": "final_breach_available_at",
        "granularities": ("seller_id", "state_od", "region_od"), "lags": (30, 45),
    },
    "positive_late_severity": {
        "kind": "continuous", "value": "positive_late_severity_value",
        "raw_value": "positive_late_days", "available": "positive_late_days_available_at",
        "granularities": ("seller_id", "state_od", "region_od"), "lags": (30, 45),
    },
}

ENTITY_COLUMNS = {
    "seller_id": "seller_id",
    "state_od": "state_od",
    "region_od": "region_od",
    "seller_x_customer_region": "seller_x_customer_region",
    "zip2_od": "zip2_od",
}

STRUCTURAL_PARENT = {
    "seller_id": ("seller_state", "main_seller_state"),
    "state_od": ("region_od", "region_od"),
    "region_od": ("global", None),
}

EXPECTED_RAW_HASHES = {
    "orders": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "customers": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "geolocation": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    "items": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "products": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "sellers": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "categories": "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
}

PROFILE_BASE_COLUMNS = [
    "entity_id", "snapshot_date", "target", "granularity", "scheme",
    "window_days", "lag_days", "estimator", "parent_structure", "kappa",
    "base_candidate_id", "parent_id", "score", "raw_score", "support", "event_count",
    "parent_score", "global_score", "posterior_se", "lower_interval",
    "upper_interval", "cold_start", "profile_freshness_days", "active_days",
    "last_mature_outcome_date", "expected_rate", "observed_expected_ratio",
    "within_variance", "between_variance", "invalid_reason",
]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recursive_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        p.relative_to(path).as_posix(): sha256_file(p)
        for p in sorted(path.rglob("*")) if p.is_file()
    }


def control_file_hashes() -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in CONTROL_FILES}


def repository_state() -> dict:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True,
    )
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
    }


def preflight(data_dir: str | Path) -> dict:
    from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp

    assembler_hash = sha256_file(ASSEMBLER)
    raw_hashes = dp.raw_file_sha256s(data_dir)
    if assembler_hash != ASSEMBLER_SHA:
        raise RuntimeError(f"canonical assembler hash mismatch: {assembler_hash}")
    if raw_hashes != EXPECTED_RAW_HASHES:
        raise RuntimeError(f"raw Olist hash mismatch: {raw_hashes}")
    manifest = json.loads((V11 / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["raw_file_hashes"] != raw_hashes:
        raise RuntimeError("raw hashes do not match V1.1 manifest")
    if manifest["canonical_assembler"]["sha256"] != assembler_hash:
        raise RuntimeError("assembler hash does not match V1.1 manifest")
    return {
        "repository": repository_state(),
        "assembler_sha256": assembler_hash,
        "raw_file_hashes": raw_hashes,
        "protected_hashes": {name: recursive_hashes(path) for name, path in PROTECTED.items()},
        "control_file_hashes": control_file_hashes(),
        "v1_1_file_hashes": recursive_hashes(V11),
    }


def compare_hash_maps(before: Mapping[str, Mapping[str, str]]) -> tuple[bool, dict[str, object]]:
    after = {name: recursive_hashes(PROTECTED[name]) for name in before}
    detail: dict[str, object] = {}
    ok = True
    for name, old in before.items():
        new = after.get(name, {})
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
        detail[name] = {"added": added, "removed": removed, "changed": changed, "unchanged": not (added or removed or changed)}
        ok &= not (added or removed or changed)
    return ok, detail


def weekly_anchors(start: str | pd.Timestamp, end_exclusive: str | pd.Timestamp) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end_exclusive).normalize()
    if end_ts <= start_ts:
        return pd.DatetimeIndex([])
    return pd.date_range(start_ts, end_ts - pd.Timedelta(days=1), freq="7D")


def future_cohort(frame: pd.DataFrame, snapshot: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    t = pd.Timestamp(snapshot)
    return frame.loc[
        frame["order_purchase_timestamp"].ge(t)
        & frame["order_purchase_timestamp"].lt(t + pd.Timedelta(days=horizon_days))
    ].copy()


def build_analysis_frame(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the V1.1 all-placed frame and exact canonical delivered frame read-only."""
    from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp
    from analysis.dynamic_profile_eda_v1_1.scripts.core import build_all_placed

    raw = dp.read_raw_tables(data_dir)
    canonical = dp.assemble_order_base(raw)
    frame = build_all_placed(raw, canonical, dp.REGION)
    b0 = canonical[["order_id", *dp.B0_COLUMNS]].copy().rename(
        columns={column: f"b0__{column}" for column in dp.B0_COLUMNS}
    )
    frame = frame.merge(b0, on="order_id", how="left", validate="1:1")
    frame = frame.sort_values("order_id", kind="mergesort").reset_index(drop=True)

    purchase = frame["order_purchase_timestamp"]
    estimate = frame["order_estimated_delivery_date"]
    frame["promised_delivery_days"] = (estimate.dt.normalize() - purchase.dt.normalize()).dt.days
    frame["log1p_n_items"] = np.log1p(pd.to_numeric(frame["n_items"], errors="coerce").clip(lower=0))
    frame["log1p_total_price"] = np.log1p(pd.to_numeric(frame["total_price"], errors="coerce").clip(lower=0))
    frame["log1p_total_freight"] = np.log1p(pd.to_numeric(frame["total_freight_value"], errors="coerce").clip(lower=0))
    price = pd.to_numeric(frame["total_price"], errors="coerce")
    freight = pd.to_numeric(frame["total_freight_value"], errors="coerce")
    frame["freight_to_price_ratio"] = np.where(price.gt(0), freight / price, np.nan)
    frame["purchase_weekday"] = purchase.dt.weekday
    frame["purchase_hour"] = purchase.dt.hour
    frame["is_weekend_purchase"] = purchase.dt.weekday.ge(5).astype("Int64")
    frame["handling_level_value"] = np.log1p(frame["handling_duration"].where(frame["handling_duration"].ge(0)))
    frame["transit_level_value"] = np.log1p(frame["transit_duration"].where(frame["transit_duration"].ge(0)))
    frame["positive_late_severity_value"] = np.log1p(frame["positive_late_days"].where(frame["positive_late_days"].gt(0)))
    return frame, canonical, raw


def frozen_tail_thresholds(frame: pd.DataFrame, cutoff: str | pd.Timestamp = "2017-04-01") -> dict[str, float]:
    t = pd.Timestamp(cutoff)
    thresholds: dict[str, float] = {}
    for name, value, available in (
        ("handling_tail_threshold_days", "handling_duration", "handling_available_at"),
        ("transit_tail_threshold_days", "transit_duration", "transit_available_at"),
    ):
        eligible = frame.loc[
            frame["in_canonical"] & frame[available].notna() & frame[available].lt(t) & frame[value].ge(0), value
        ].dropna()
        if eligible.empty:
            raise RuntimeError(f"no eligible pre-development values for {name}")
        thresholds[name] = float(eligible.quantile(0.90, interpolation="linear"))
    return thresholds


def attach_tail_targets(frame: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    result = frame.copy()
    handling_valid = result["handling_duration"].ge(0) & result["handling_duration"].notna()
    transit_valid = result["transit_duration"].ge(0) & result["transit_duration"].notna()
    result["handling_tail"] = result["handling_duration"].gt(thresholds["handling_tail_threshold_days"]).where(handling_valid)
    result["transit_tail"] = result["transit_duration"].gt(thresholds["transit_tail_threshold_days"]).where(transit_valid)
    return result


def mask_locked_outcomes_for_development(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the only frame allowed in the development evaluation process."""
    result = frame.loc[frame["order_purchase_timestamp"].lt(pd.Timestamp("2018-01-01"))].copy()
    if result["order_purchase_timestamp"].ge(pd.Timestamp("2018-01-01")).any():
        raise AssertionError("locked confirmation purchase rows entered development view")
    result.attrs["locked_outcomes_masked"] = True
    return result


def target_valid_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    spec = TARGET_SPECS[target]
    value = str(spec["value"])
    available = str(spec["available"])
    mask = frame[available].notna() & frame[value].notna()
    if target.startswith("handling"):
        mask &= frame["handling_duration"].ge(0)
    if target.startswith("transit"):
        mask &= frame["transit_duration"].ge(0)
    if target == "positive_late_severity":
        mask &= frame["positive_late_days"].gt(0)
    return mask


def candidate_sources() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target, spec in TARGET_SPECS.items():
        for granularity in spec["granularities"]:
            for window in (30, 60, 90):
                rows.append({"target": target, "granularity": granularity, "scheme": "A", "window_days": window, "lag_days": 0})
                for lag in spec["lags"]:
                    rows.append({"target": target, "granularity": granularity, "scheme": "C", "window_days": window, "lag_days": int(lag)})
    return rows


def parent_options(granularity: str) -> tuple[str, ...]:
    structural = STRUCTURAL_PARENT[granularity][0]
    return ("global",) if structural == "global" else ("global", structural)


def base_candidate_id(source: Mapping[str, object], estimator: str, parent: str, kappa: float | None) -> str:
    kval = "na" if kappa is None or pd.isna(kappa) else str(int(kappa))
    return "|".join([
        str(source["target"]), str(source["granularity"]), str(source["scheme"]),
        f"w{int(source['window_days'])}", f"l{int(source['lag_days'])}", estimator,
        f"parent={parent}", f"kappa={kval}",
    ])


def candidate_variants(source: Mapping[str, object]) -> list[dict[str, object]]:
    target = str(source["target"])
    granularity = str(source["granularity"])
    kind = str(TARGET_SPECS[target]["kind"])
    result = [{"estimator": "P0", "parent_structure": "global", "kappa": None}]
    if kind == "binary":
        for parent in parent_options(granularity):
            for kappa in (10, 20, 50, 100):
                result.append({"estimator": "P1", "parent_structure": parent, "kappa": kappa})
    else:
        for parent in parent_options(granularity):
            result.append({"estimator": "P1", "parent_structure": parent, "kappa": None})
    if kind == "binary":
        for parent in parent_options(granularity):
            for kappa in (10, 20, 50, 100):
                result.append({"estimator": "P2", "parent_structure": parent, "kappa": kappa})
    else:
        for parent in parent_options(granularity):
            result.append({"estimator": "P2", "parent_structure": parent, "kappa": None})
    for item in result:
        item["base_candidate_id"] = base_candidate_id(source, str(item["estimator"]), str(item["parent_structure"]), item["kappa"])
    return result


def history_slice(frame: pd.DataFrame, source: Mapping[str, object], snapshot: pd.Timestamp) -> pd.DataFrame:
    target = str(source["target"])
    spec = TARGET_SPECS[target]
    t = pd.Timestamp(snapshot)
    window = int(source["window_days"])
    lag = int(source["lag_days"])
    valid = frame["in_canonical"] & target_valid_mask(frame, target) & frame[str(spec["available"])].lt(t)
    if source["scheme"] == "A":
        valid &= frame[str(spec["available"])].ge(t - pd.Timedelta(days=window))
    elif source["scheme"] == "C":
        valid &= frame["order_purchase_timestamp"].ge(t - pd.Timedelta(days=lag + window))
        valid &= frame["order_purchase_timestamp"].lt(t - pd.Timedelta(days=lag))
    else:
        raise ValueError(f"unsupported scheme {source['scheme']}")
    cols = list(dict.fromkeys([
        "order_id", "order_purchase_timestamp", str(spec["available"]), str(spec["value"]),
        ENTITY_COLUMNS[str(source["granularity"])], "main_seller_state", "region_od",
        f"expected_{target}",
    ]))
    return frame.loc[valid, [c for c in cols if c in frame.columns]].copy()


def _nuisance_preprocessor(config: Mapping[str, object]) -> ColumnTransformer:
    numeric = list(config["p2"]["features_numeric"])
    categorical = list(config["p2"]["features_categorical"])
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
        ]), categorical),
    ], remainder="drop")


def _nuisance_feature_frame(
    frame: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return sklearn-safe nuisance inputs without changing analytical values.

    The canonical assembler deliberately preserves pandas nullable dtypes.  In
    sklearn 1.6, ``SimpleImputer`` cannot evaluate ``pd.NA`` inside an object
    array (``pd.NA != pd.NA`` has an ambiguous truth value).  Convert only the
    modelling view: numeric columns become float64 and categorical missingness
    becomes ordinary ``np.nan``.  The source frame remains untouched.
    """
    numeric = list(config["p2"]["features_numeric"])
    categorical = list(config["p2"]["features_categorical"])
    result = pd.DataFrame(index=frame.index)
    for column in numeric:
        result[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    for column in categorical:
        values = frame[column].astype(object)
        result[column] = values.where(pd.notna(values), np.nan)
    return result


def generate_row_origin_expectations(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    stage_end_exclusive: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate strictly prior monthly nuisance predictions; never tune on their performance."""
    result = frame.copy()
    end = pd.Timestamp(stage_end_exclusive)
    origins = pd.date_range(result["order_purchase_timestamp"].min().to_period("M").start_time, end, freq="MS", inclusive="left")
    audit: list[dict[str, object]] = []
    min_rows = int(config["p2"]["minimum_training_rows"])
    numeric = list(config["p2"]["features_numeric"])
    categorical = list(config["p2"]["features_categorical"])
    features = numeric + categorical
    for target, spec in TARGET_SPECS.items():
        expected_col = f"expected_{target}"
        result[expected_col] = np.nan
        value_col = str(spec["value"])
        available_col = str(spec["available"])
        kind = str(spec["kind"])
        for origin in origins:
            next_origin = origin + pd.offsets.MonthBegin(1)
            score_mask = (
                result["in_canonical"]
                & result["order_purchase_timestamp"].ge(origin)
                & result["order_purchase_timestamp"].lt(next_origin)
            )
            if not score_mask.any():
                continue
            train_mask = result["in_canonical"] & target_valid_mask(result, target)
            train_mask &= result[available_col].lt(origin)
            train_mask &= result["order_purchase_timestamp"].lt(origin)
            train = result.loc[train_mask]
            y = pd.to_numeric(train[value_col], errors="coerce")
            fallback = float(y.mean()) if y.notna().any() else np.nan
            status = "invalid_insufficient_strict_prior_training"
            error_class = ""
            error_message = ""
            model: Pipeline | None = None
            binary_counts_ok = (
                kind != "binary" or (
                    y.nunique(dropna=True) == 2
                    and int(y.eq(1).sum()) >= int(config["p2"]["minimum_binary_class_count"])
                    and int(y.eq(0).sum()) >= int(config["p2"]["minimum_binary_class_count"])
                )
            )
            continuous_variance_ok = kind == "binary" or (np.isfinite(y.var()) and float(y.var()) > 0)
            if len(train) >= min_rows and binary_counts_ok and continuous_variance_ok:
                estimator = (
                    LogisticRegression(
                        penalty="l2", C=float(config["p2"]["binary_model"]["C"]),
                        solver=str(config["p2"]["binary_model"]["solver"]),
                        max_iter=int(config["p2"]["binary_model"]["max_iter"]),
                        class_weight=None,
                        random_state=int(config["p2"]["binary_model"]["random_state"]),
                    ) if kind == "binary" else Ridge(alpha=float(config["p2"]["continuous_model"]["alpha"]))
                )
                model = Pipeline([("prep", _nuisance_preprocessor(config)), ("model", estimator)])
                try:
                    model.fit(_nuisance_feature_frame(train[features], config), y)
                    scored = _nuisance_feature_frame(result.loc[score_mask, features], config)
                    predictions = model.predict_proba(scored)[:, 1] if kind == "binary" else model.predict(scored)
                    if not np.isfinite(predictions).all():
                        raise FloatingPointError("nuisance predictions contain nonfinite values")
                    result.loc[score_mask, expected_col] = predictions
                    status = "model"
                except Exception as exc:
                    # A failed P2 origin is an explicitly invalid branch.  It
                    # must never crash the P0/P1 search or receive a fallback
                    # prediction that could masquerade as case-mix adjustment.
                    status = "invalid_fit_or_score"
                    error_class = type(exc).__name__
                    error_message = str(exc)[:500]
            audit.append({
                "record_type": "nuisance_origin", "target": target,
                "origin": origin.strftime("%Y-%m-%d"), "train_rows": int(len(train)),
                "score_rows": int(score_mask.sum()), "status": status,
                "strict_prior_max_availability": train[available_col].max(),
                "fallback_value": fallback,
                "error_class": error_class,
                "error_message": error_message,
                "entity_features_present": False, "hrd_features_present": False,
            })
    return result, pd.DataFrame(audit)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cutoff = q * weights.sum()
    return float(values[min(np.searchsorted(np.cumsum(weights), cutoff, side="left"), len(values) - 1)])


def weighted_spearman(x: Iterable[float], y: Iterable[float], weights: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return np.nan
    rx = rankdata(x[mask], method="average")
    ry = rankdata(y[mask], method="average")
    w = w[mask]
    mx = np.average(rx, weights=w)
    my = np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    vx = np.average((rx - mx) ** 2, weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    return float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def _scalar_logistic_offset(y: np.ndarray, p: np.ndarray, tol: float = 1e-10, max_iter: int = 50) -> float:
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = np.clip(p[valid], 1e-8, 1 - 1e-8)
    if len(y) == 0:
        return 0.0
    offset = 0.0
    base = logit(p)
    for _ in range(max_iter):
        q = expit(base + offset)
        denom = np.sum(q * (1 - q))
        if denom <= 1e-12:
            break
        step = np.sum(y - q) / denom
        offset += step
        if abs(step) < tol:
            break
    return float(np.clip(offset, -20, 20))


def _group_logistic_offsets(
    y: np.ndarray,
    p: np.ndarray,
    codes: np.ndarray,
    groups: int,
    penalty_precision: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.zeros(groups, dtype=float)
    base = logit(np.clip(p, 1e-8, 1 - 1e-8))
    precision = float(penalty_precision)
    hessian = np.full(groups, precision, dtype=float)
    for _ in range(max_iter):
        q = expit(base + theta[codes])
        grad = np.bincount(codes, weights=y - q, minlength=groups) - precision * theta
        hessian = np.bincount(codes, weights=q * (1 - q), minlength=groups) + precision
        step = np.divide(grad, hessian, out=np.zeros_like(grad), where=hessian > 0)
        theta += step
        theta = np.clip(theta, -20, 20)
        if np.max(np.abs(step), initial=0.0) < tol:
            break
    return theta, hessian


def _continuous_components(history: pd.DataFrame, entity_col: str, parent_col: str, value_col: str) -> tuple[pd.DataFrame, float, float]:
    group = history.groupby([entity_col, parent_col], dropna=False, sort=True)[value_col]
    stats = group.agg(support="count", raw_score="mean", sample_variance="var").reset_index()
    mapped = history[[entity_col, parent_col, value_col]].merge(
        stats[[entity_col, parent_col, "raw_score"]], on=[entity_col, parent_col], how="left", validate="m:1",
    )
    within_df = int(np.maximum(stats["support"].to_numpy(dtype=int) - 1, 0).sum())
    within_ss = float(np.square(mapped[value_col] - mapped["raw_score"]).sum())
    within = within_ss / within_df if within_df > 0 else np.nan
    weights = stats["support"].to_numpy(dtype=float)
    means = stats["raw_score"].to_numpy(dtype=float)
    parent_means = history.groupby(parent_col, dropna=False)[value_col].mean()
    centers = stats[parent_col].map(parent_means).to_numpy(dtype=float)
    weighted_var = float(np.average(np.square(means - centers), weights=weights)) if weights.sum() > 0 else np.nan
    noise = float(within * np.average(1.0 / weights, weights=weights)) if np.isfinite(within) and weights.sum() > 0 else np.nan
    between = max(weighted_var - noise, 0.0) if np.isfinite(weighted_var) and np.isfinite(noise) else np.nan
    return stats, within, between


def _base_stats(history: pd.DataFrame, source: Mapping[str, object]) -> pd.DataFrame:
    target = str(source["target"])
    entity_col = ENTITY_COLUMNS[str(source["granularity"])]
    available_col = str(TARGET_SPECS[target]["available"])
    value_col = str(TARGET_SPECS[target]["value"])
    work = history.copy()
    work["_purchase_day"] = work["order_purchase_timestamp"].dt.normalize()
    grouped = work.groupby(entity_col, dropna=False, sort=True)
    stats = grouped.agg(
        support=(value_col, "count"),
        raw_score=(value_col, "mean"),
        event_count=(value_col, "sum"),
        active_days=("_purchase_day", "nunique"),
        last_mature_outcome_date=(available_col, "max"),
    ).reset_index().rename(columns={entity_col: "entity_id"})
    stats = stats.loc[stats["entity_id"].notna()].copy()
    stats["entity_id"] = stats["entity_id"].astype(str)
    return stats


def _parent_column(history: pd.DataFrame, granularity: str, parent_structure: str) -> pd.Series:
    if parent_structure == "global":
        return pd.Series("__GLOBAL__", index=history.index, dtype="string")
    structural_name, structural_col = STRUCTURAL_PARENT[granularity]
    if parent_structure != structural_name or structural_col is None:
        raise ValueError(f"invalid parent {parent_structure} for {granularity}")
    return history[structural_col].astype("string")


def _entity_parent_map(history: pd.DataFrame, granularity: str, parent_structure: str) -> pd.DataFrame:
    entity_col = ENTITY_COLUMNS[granularity]
    parents = _parent_column(history, granularity, parent_structure)
    work = pd.DataFrame({"entity_id": history[entity_col].astype("string"), "parent_id": parents})
    work = work.dropna(subset=["entity_id"]).drop_duplicates()
    counts = work.groupby("entity_id", sort=False)["parent_id"].nunique(dropna=False)
    if counts.gt(1).any():
        bad = counts[counts.gt(1)].index[:5].tolist()
        raise RuntimeError(f"nondeterministic parent mapping: {bad}")
    result = work.drop_duplicates("entity_id", keep="first")
    result["entity_id"] = result["entity_id"].astype(str)
    result["parent_id"] = result["parent_id"].fillna("__MISSING_PARENT__").astype(str)
    return result


def _decorate_profile(
    stats: pd.DataFrame,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
    variant: Mapping[str, object],
) -> pd.DataFrame:
    out = stats.copy()
    out["snapshot_date"] = pd.Timestamp(snapshot).normalize()
    out["target"] = str(source["target"])
    out["granularity"] = str(source["granularity"])
    out["scheme"] = str(source["scheme"])
    out["window_days"] = int(source["window_days"])
    out["lag_days"] = int(source["lag_days"])
    out["estimator"] = str(variant["estimator"])
    out["parent_structure"] = str(variant["parent_structure"])
    out["kappa"] = variant["kappa"]
    out["base_candidate_id"] = str(variant["base_candidate_id"])
    out["cold_start"] = 0
    out["profile_freshness_days"] = (
        pd.Timestamp(snapshot).normalize() - pd.to_datetime(out["last_mature_outcome_date"]).dt.normalize()
    ).dt.days
    for column in PROFILE_BASE_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[PROFILE_BASE_COLUMNS]


def _binary_profile_variant(
    history: pd.DataFrame,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
    variant: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = str(source["target"])
    granularity = str(source["granularity"])
    entity_col = ENTITY_COLUMNS[granularity]
    value_col = str(TARGET_SPECS[target]["value"])
    parent_structure = str(variant["parent_structure"])
    estimator = str(variant["estimator"])
    expected_col = f"expected_{target}"
    work = history.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").astype(float)
    work["parent_id"] = _parent_column(work, granularity, parent_structure).fillna("__MISSING_PARENT__")
    if estimator == "P2":
        work = work.loc[work[expected_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=PROFILE_BASE_COLUMNS), pd.DataFrame()
    stats = _base_stats(work, source)
    stats = stats.merge(_entity_parent_map(work, granularity, parent_structure), on="entity_id", how="left", validate="1:1")
    y = pd.to_numeric(work[value_col], errors="coerce").to_numpy(dtype=float)
    global_raw = float(np.mean(y))
    parent_raw = work.groupby("parent_id", dropna=False)[value_col].agg(["sum", "count", "mean"]).reset_index()
    parent_raw["parent_id"] = parent_raw["parent_id"].astype(str)
    parent_min = int(config["binary_eb"]["parent_min_support"])
    parent_table: pd.DataFrame

    if estimator == "P0":
        stats["score"] = stats["raw_score"]
        stats["parent_score"] = global_raw
        stats["global_score"] = global_raw
        n = stats["support"].to_numpy(dtype=float)
        p = stats["score"].to_numpy(dtype=float)
        se = np.sqrt(np.divide(p * (1 - p), n, out=np.full_like(p, np.nan), where=n > 0))
        stats["posterior_se"] = se
        stats["lower_interval"] = np.clip(p - 1.959963984540054 * se, 0, 1)
        stats["upper_interval"] = np.clip(p + 1.959963984540054 * se, 0, 1)
        stats["expected_rate"] = np.nan
        stats["observed_expected_ratio"] = np.nan
        stats["within_variance"] = np.nan
        stats["between_variance"] = np.nan
        stats["invalid_reason"] = ""
        parent_table = pd.DataFrame({
            "parent_id": ["__GLOBAL__"], "parent_score": [global_raw],
            "global_score": [global_raw], "parent_support": [len(work)],
        })
    elif estimator == "P1":
        kappa = float(variant["kappa"])
        if parent_structure == "global":
            parent_scores = pd.Series({"__GLOBAL__": global_raw})
            parent_raw = pd.DataFrame({"parent_id": ["__GLOBAL__"], "count": [len(work)], "mean": [global_raw]})
        else:
            parent_raw["parent_score"] = (parent_raw["sum"] + kappa * global_raw) / (parent_raw["count"] + kappa)
            unsupported_parent = parent_raw["count"].lt(parent_min)
            parent_raw.loc[unsupported_parent, "parent_score"] = global_raw
            parent_scores = parent_raw.set_index("parent_id")["parent_score"]
        stats["parent_score"] = stats["parent_id"].map(parent_scores).fillna(global_raw)
        stats["global_score"] = global_raw
        stats["score"] = (stats["event_count"] + kappa * stats["parent_score"]) / (stats["support"] + kappa)
        alpha = stats["event_count"] + kappa * stats["parent_score"]
        beta_param = stats["support"] - stats["event_count"] + kappa * (1 - stats["parent_score"])
        denom = alpha + beta_param
        stats["posterior_se"] = np.sqrt(alpha * beta_param / (np.square(denom) * (denom + 1)))
        stats["lower_interval"] = beta.ppf(0.025, alpha, beta_param)
        stats["upper_interval"] = beta.ppf(0.975, alpha, beta_param)
        stats["expected_rate"] = np.nan
        stats["observed_expected_ratio"] = np.nan
        stats["within_variance"] = np.nan
        stats["between_variance"] = np.nan
        stats["invalid_reason"] = ""
        if "parent_score" not in parent_raw:
            parent_raw["parent_score"] = global_raw
        parent_table = parent_raw.rename(columns={"count": "parent_support"})[
            ["parent_id", "parent_score", "parent_support"]
        ].copy()
        parent_table["global_score"] = global_raw
    elif estimator == "P2":
        p = np.clip(pd.to_numeric(work[expected_col], errors="coerce").to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        mu_ref = global_raw
        kappa = float(variant["kappa"])
        penalty = kappa * mu_ref * (1 - mu_ref)
        if not np.isfinite(penalty) or penalty <= 0 or not 0 < mu_ref < 1:
            return pd.DataFrame(columns=PROFILE_BASE_COLUMNS), pd.DataFrame()
        global_score = mu_ref
        if parent_structure == "global":
            parent_theta_by_id = {"__GLOBAL__": 0.0}
            parent_adjust = np.zeros(len(work))
        else:
            parent_codes, parent_levels = pd.factorize(work["parent_id"].astype(str), sort=True)
            pt, ph = _group_logistic_offsets(
                y, p, parent_codes, len(parent_levels),
                penalty_precision=penalty,
                tol=float(config["p2"]["binary_offset_tolerance"]),
                max_iter=int(config["p2"]["binary_offset_max_iter"]),
            )
            parent_counts = np.bincount(parent_codes, minlength=len(parent_levels))
            unsupported_parent = parent_counts < parent_min
            pt[unsupported_parent] = 0.0
            parent_theta_by_id = dict(zip(parent_levels.astype(str), pt))
            parent_adjust = pt[parent_codes]
        p_parent = expit(logit(p) + parent_adjust)
        entity_codes, entity_levels = pd.factorize(work[entity_col].astype(str), sort=True)
        et, eh = _group_logistic_offsets(
            y, p_parent, entity_codes, len(entity_levels),
            penalty_precision=penalty,
            tol=float(config["p2"]["binary_offset_tolerance"]),
            max_iter=int(config["p2"]["binary_offset_max_iter"]),
        )
        entity_theta = pd.Series(et, index=entity_levels.astype(str))
        entity_hessian = pd.Series(eh, index=entity_levels.astype(str))
        stats["_parent_theta"] = stats["parent_id"].map(parent_theta_by_id).fillna(0.0)
        stats["_entity_theta"] = stats["entity_id"].map(entity_theta).fillna(0.0)
        base = logit(mu_ref)
        stats["parent_score"] = expit(base + stats["_parent_theta"])
        stats["global_score"] = global_score
        eta = base + stats["_parent_theta"] + stats["_entity_theta"]
        stats["score"] = expit(eta)
        theta_se = 1.0 / np.sqrt(stats["entity_id"].map(entity_hessian).astype(float))
        stats["posterior_se"] = stats["score"] * (1 - stats["score"]) * theta_se
        stats["lower_interval"] = expit(eta - 1.959963984540054 * theta_se)
        stats["upper_interval"] = expit(eta + 1.959963984540054 * theta_se)
        exp_stats = work.groupby(entity_col, sort=True).agg(expected_rate=(expected_col, "mean"), expected_sum=(expected_col, "sum")).reset_index()
        exp_stats[entity_col] = exp_stats[entity_col].astype(str)
        stats = stats.merge(exp_stats.rename(columns={entity_col: "entity_id"}), on="entity_id", how="left", validate="1:1")
        stats["observed_expected_ratio"] = np.divide(
            stats["event_count"], stats["expected_sum"],
            out=np.full(len(stats), np.nan), where=stats["expected_sum"].to_numpy(dtype=float) > 0,
        )
        stats["within_variance"] = np.nan
        stats["between_variance"] = np.nan
        stats["invalid_reason"] = ""
        parent_rows = []
        for pid in sorted(set(stats["parent_id"].astype(str)) | {"__GLOBAL__"}):
            theta = parent_theta_by_id.get(pid, 0.0)
            parent_rows.append({
                "parent_id": pid,
                "parent_score": float(expit(base + theta)),
                "global_score": global_score,
                "parent_support": int((work["parent_id"].astype(str) == pid).sum()) if pid != "__GLOBAL__" else len(work),
            })
        parent_table = pd.DataFrame(parent_rows)
    else:
        raise ValueError(estimator)

    result = _decorate_profile(stats, source, snapshot, variant)
    parent_table["base_candidate_id"] = str(variant["base_candidate_id"])
    parent_table["target"] = target
    parent_table["snapshot_date"] = pd.Timestamp(snapshot).normalize()
    return result, parent_table


def _continuous_profile_variant(
    history: pd.DataFrame,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
    variant: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = str(source["target"])
    granularity = str(source["granularity"])
    entity_col = ENTITY_COLUMNS[granularity]
    value_col = str(TARGET_SPECS[target]["value"])
    available_col = str(TARGET_SPECS[target]["available"])
    expected_col = f"expected_{target}"
    parent_structure = str(variant["parent_structure"])
    estimator = str(variant["estimator"])
    work = history.copy()
    work["parent_id"] = _parent_column(work, granularity, parent_structure).fillna("__MISSING_PARENT__")
    if estimator == "P2":
        work = work.loc[work[expected_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=PROFILE_BASE_COLUMNS), pd.DataFrame()
    work["_model_value"] = pd.to_numeric(work[value_col], errors="coerce")
    if estimator == "P2":
        work["_model_value"] = work["_model_value"] - pd.to_numeric(work[expected_col], errors="coerce")
    work["_purchase_day"] = work["order_purchase_timestamp"].dt.normalize()
    grouped = work.groupby(entity_col, dropna=False, sort=True)
    stats = grouped.agg(
        support=("_model_value", "count"), raw_model_score=("_model_value", "mean"),
        raw_outcome_score=(value_col, "mean"), sample_variance=("_model_value", "var"),
        active_days=("_purchase_day", "nunique"), last_mature_outcome_date=(available_col, "max"),
    ).reset_index().rename(columns={entity_col: "entity_id"})
    stats = stats.loc[stats["entity_id"].notna()].copy()
    stats["entity_id"] = stats["entity_id"].astype(str)
    stats = stats.merge(_entity_parent_map(work, granularity, parent_structure), on="entity_id", how="left", validate="1:1")
    global_model_mean = float(work["_model_value"].mean())
    global_expected = float(pd.to_numeric(work[expected_col], errors="coerce").mean()) if estimator == "P2" else 0.0
    global_score = global_expected + global_model_mean
    parent_stats = work.groupby("parent_id", dropna=False)["_model_value"].agg(parent_support="count", parent_model_mean="mean").reset_index()
    parent_stats["parent_id"] = parent_stats["parent_id"].astype(str)
    parent_min = int(config["binary_eb"]["parent_min_support"])
    parent_stats["parent_model_mean"] = parent_stats["parent_model_mean"].where(parent_stats["parent_support"].ge(parent_min), global_model_mean)
    if parent_structure == "global":
        parent_stats = pd.DataFrame({"parent_id": ["__GLOBAL__"], "parent_support": [len(work)], "parent_model_mean": [global_model_mean]})
    parent_model = parent_stats.set_index("parent_id")["parent_model_mean"]
    stats["parent_model_mean"] = stats["parent_id"].map(parent_model).fillna(global_model_mean)
    stats["parent_score"] = global_expected + stats["parent_model_mean"]
    stats["global_score"] = global_score
    stats["raw_score"] = stats["raw_outcome_score"]
    stats["event_count"] = np.nan
    stats["expected_rate"] = np.nan
    stats["observed_expected_ratio"] = np.nan

    if estimator == "P0":
        stats["score"] = stats["raw_outcome_score"]
        stats["posterior_se"] = np.sqrt(stats["sample_variance"] / stats["support"])
        stats["lower_interval"] = stats["score"] - 1.959963984540054 * stats["posterior_se"]
        stats["upper_interval"] = stats["score"] + 1.959963984540054 * stats["posterior_se"]
        stats["within_variance"] = np.nan
        stats["between_variance"] = np.nan
        stats["invalid_reason"] = ""
    else:
        entity_means = stats.set_index("entity_id")["raw_model_score"]
        mapped = work[[entity_col, "_model_value"]].copy()
        mapped["entity_id"] = mapped[entity_col].astype(str)
        mapped["entity_mean"] = mapped["entity_id"].map(entity_means)
        within_df = int(np.maximum(stats["support"].to_numpy(dtype=int) - 1, 0).sum())
        within_ss = float(np.square(mapped["_model_value"] - mapped["entity_mean"]).sum())
        within = within_ss / within_df if within_df > 0 else np.nan
        weights = stats["support"].to_numpy(dtype=float)
        deviations = stats["raw_model_score"].to_numpy(dtype=float) - stats["parent_model_mean"].to_numpy(dtype=float)
        weighted_var = float(np.average(np.square(deviations), weights=weights)) if weights.sum() > 0 else np.nan
        noise = float(within * np.average(1.0 / weights, weights=weights)) if np.isfinite(within) else np.nan
        between = max(weighted_var - noise, 0.0) if np.isfinite(weighted_var) and np.isfinite(noise) else np.nan
        floor = float(config["continuous_eb"]["variance_floor"])
        stats["within_variance"] = within
        stats["between_variance"] = between
        if not np.isfinite(within) or not np.isfinite(between) or within <= floor or between <= floor:
            stats["score"] = stats["parent_score"]
            stats["posterior_se"] = np.sqrt(np.maximum(within, 0) / stats["support"]) if np.isfinite(within) else np.nan
            stats["invalid_reason"] = str(config["continuous_eb"]["degenerate_invalid_reason"])
        else:
            precision = stats["support"] / within + 1.0 / between
            posterior_model = (
                stats["support"] * stats["raw_model_score"] / within
                + stats["parent_model_mean"] / between
            ) / precision
            stats["score"] = global_expected + posterior_model
            stats["posterior_se"] = np.sqrt(1.0 / precision)
            stats["invalid_reason"] = ""
        stats["lower_interval"] = stats["score"] - 1.959963984540054 * stats["posterior_se"]
        stats["upper_interval"] = stats["score"] + 1.959963984540054 * stats["posterior_se"]
    parent_table = parent_stats.copy()
    parent_table["parent_score"] = global_expected + parent_table["parent_model_mean"]
    parent_table["global_score"] = global_score
    parent_table["base_candidate_id"] = str(variant["base_candidate_id"])
    parent_table["target"] = target
    parent_table["snapshot_date"] = pd.Timestamp(snapshot).normalize()
    result = _decorate_profile(stats, source, snapshot, variant)
    return result, parent_table


def build_profiles(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
    config: Mapping[str, object],
    allowed_base_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history_slice(frame, source, snapshot)
    profiles: list[pd.DataFrame] = []
    parents: list[pd.DataFrame] = []
    kind = str(TARGET_SPECS[str(source["target"])]["kind"])
    variants = candidate_variants(source)
    if allowed_base_ids is not None:
        variants = [variant for variant in variants if str(variant["base_candidate_id"]) in allowed_base_ids]
    for variant in variants:
        if kind == "binary":
            profile, parent = _binary_profile_variant(history, source, snapshot, variant, config)
        else:
            profile, parent = _continuous_profile_variant(history, source, snapshot, variant, config)
        profiles.append(profile)
        parents.append(parent)
    return (
        pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(columns=PROFILE_BASE_COLUMNS),
        pd.concat(parents, ignore_index=True) if parents else pd.DataFrame(),
    )


def anchor_schedule(config: Mapping[str, object]) -> pd.DataFrame:
    time_cfg = config["time"]
    origin = pd.Timestamp(time_cfg["anchor_origin"])
    terminal_end = pd.Timestamp(time_cfg["terminal"]["end_inclusive"])
    cadence = pd.date_range(origin, terminal_end, freq="7D")
    periods = {
        "development": (pd.Timestamp(time_cfg["development"]["start"]), pd.Timestamp(time_cfg["development"]["end_exclusive"])),
        "confirmation": (pd.Timestamp(time_cfg["confirmation"]["start"]), pd.Timestamp(time_cfg["confirmation"]["end_exclusive"])),
        "terminal": (pd.Timestamp(time_cfg["terminal"]["start"]), pd.Timestamp(time_cfg["terminal"]["end_inclusive"]) + pd.Timedelta(days=1)),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        for anchor in cadence[(cadence >= start) & (cadence < end)]:
            for horizon in (7, 30):
                contained = anchor + pd.Timedelta(days=horizon) <= end
                if contained:
                    rows.append({
                        "period": period, "anchor_date": anchor,
                        "horizon_days": horizon, "future_start": anchor,
                        "future_end_exclusive": anchor + pd.Timedelta(days=horizon),
                        "full_phase_containment": True,
                    })
    result = pd.DataFrame(rows).sort_values(["period", "horizon_days", "anchor_date"], kind="mergesort").reset_index(drop=True)
    expected = time_cfg["expected_valid_anchor_counts"]
    actual = {
        f"{period}_{horizon}d": int(len(group))
        for (period, horizon), group in result.groupby(["period", "horizon_days"])
    }
    if actual != {key: int(value) for key, value in expected.items()}:
        raise AssertionError(f"anchor schedule mismatch: {actual} != {expected}")
    return result


def _parent_ids_for_future(future: pd.DataFrame, granularity: str, parent_structure: str) -> pd.Series:
    if parent_structure == "global":
        return pd.Series("__GLOBAL__", index=future.index, dtype="string")
    structural, column = STRUCTURAL_PARENT[granularity]
    if structural != parent_structure or column is None:
        raise ValueError(f"invalid parent mapping {granularity}->{parent_structure}")
    return future[column].astype("string").fillna("__MISSING_PARENT__")


def map_future_orders(
    future: pd.DataFrame,
    profiles: pd.DataFrame,
    parents: pd.DataFrame,
    source: Mapping[str, object],
    base_id: str,
) -> pd.DataFrame:
    target = str(source["target"])
    granularity = str(source["granularity"])
    entity_col = ENTITY_COLUMNS[granularity]
    spec = TARGET_SPECS[target]
    profile = profiles.loc[profiles["base_candidate_id"].eq(base_id)].copy()
    if profile.empty:
        return pd.DataFrame()
    if profile["entity_id"].duplicated().any():
        raise AssertionError(f"duplicate profile entity rows for {base_id}")
    variant = profile.iloc[0]
    parent_structure = str(variant["parent_structure"])
    parent = parents.loc[parents["base_candidate_id"].eq(base_id)].drop_duplicates("parent_id", keep="first").copy()
    global_score = float(profile["global_score"].dropna().iloc[0]) if profile["global_score"].notna().any() else np.nan
    entity = future[entity_col].astype("string")
    mapping_valid = entity.notna()
    parent_id = _parent_ids_for_future(future, granularity, parent_structure)
    score_map = profile.set_index("entity_id")["score"]
    raw_map = profile.set_index("entity_id")["raw_score"]
    support_map = profile.set_index("entity_id")["support"]
    se_map = profile.set_index("entity_id")["posterior_se"]
    lower_map = profile.set_index("entity_id")["lower_interval"]
    upper_map = profile.set_index("entity_id")["upper_interval"]
    parent_map = parent.set_index("parent_id")["parent_score"] if not parent.empty else pd.Series(dtype=float)
    seen = mapping_valid & entity.astype(str).isin(score_map.index)
    parent_score = parent_id.astype(str).map(parent_map).fillna(global_score).astype(float)
    score = entity.astype(str).map(score_map).astype(float).where(seen, parent_score)
    # P0 is frozen as an entity raw estimate with a global cold-start
    # reference.  A hierarchical P1/P2 candidate may use its applicable
    # parent for its own cold-start score, but that must not silently change
    # the P0 comparator into a hierarchical reference.
    raw_score = entity.astype(str).map(raw_map).astype(float).where(seen, global_score)
    support = entity.astype(str).map(support_map).fillna(0).astype(int).where(mapping_valid, 0)
    posterior_se = entity.astype(str).map(se_map).astype(float).where(seen)
    lower = entity.astype(str).map(lower_map).astype(float).where(seen)
    upper = entity.astype(str).map(upper_map).astype(float).where(seen)
    mapping_status = np.select(
        [~mapping_valid.to_numpy(), seen.to_numpy()],
        ["missing_mapping", "seen"],
        default="mapped_cold_start",
    )
    observed = future["in_canonical"] & target_valid_mask(future, target)
    result = pd.DataFrame({
        "order_id": future["order_id"].to_numpy(),
        "purchase_timestamp": future["order_purchase_timestamp"].to_numpy(),
        "entity_id": entity.astype(object).to_numpy(),
        "parent_id": parent_id.astype(object).to_numpy(),
        "mapping_status": mapping_status,
        "history_support": support.to_numpy(),
        "cold_start": (mapping_status == "mapped_cold_start"),
        "profile_score": score.to_numpy(),
        "raw_score": raw_score.to_numpy(),
        "parent_score": parent_score.to_numpy(),
        "global_score": np.full(len(future), global_score),
        "posterior_se": posterior_se.to_numpy(),
        "lower_interval": lower.to_numpy(),
        "upper_interval": upper.to_numpy(),
        "target_observed": observed.to_numpy(dtype=bool),
        "target_value": pd.to_numeric(future[str(spec["value"])], errors="coerce").to_numpy(),
        "raw_target_value": pd.to_numeric(future[str(spec["raw_value"])], errors="coerce").to_numpy(),
        "label_available_at": future[str(spec["available"])].to_numpy(),
        "base_candidate_id": base_id,
        "target": target,
        "granularity": granularity,
    })
    result["eligible_for_metric"] = (
        result["target_observed"]
        & result["mapping_status"].ne("missing_mapping")
        & np.isfinite(result["profile_score"])
    )
    return result


def _calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 50 or len(np.unique(y)) < 2 or np.nanstd(p) <= 0:
        return np.nan
    predictor = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unknown solver options: iprint", category=OptimizeWarning,
            )
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(predictor, y)
        return float(model.coef_[0, 0])
    except Exception:
        return np.nan


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return {name: np.nan for name in ("log_loss", "brier", "citl", "calibration_slope", "average_precision", "roc_auc")}
    result = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "citl": float(_scalar_logistic_offset(y, p)),
        "calibration_slope": _calibration_slope(y, p),
        "average_precision": np.nan,
        "roc_auc": np.nan,
    }
    if len(np.unique(y)) == 2:
        result["average_precision"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
    return result


def _continuous_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(y) == 0:
        return {"log_mae": np.nan, "log_rmse": np.nan}
    return {
        "log_mae": float(mean_absolute_error(y, score)),
        "log_rmse": float(math.sqrt(mean_squared_error(y, score))),
    }


def _top_lift(entity: pd.DataFrame, kind: str, fraction: float = 0.20) -> float:
    if entity.empty:
        return np.nan
    ranked = entity.sort_values(["profile_score", "entity_id"], ascending=[False, True], kind="mergesort")
    top_n = max(1, int(math.ceil(len(ranked) * fraction)))
    top = ranked.iloc[:top_n]
    outcome_column = "future_mean" if kind == "binary" else "future_raw_mean"
    overall = float(np.average(ranked[outcome_column], weights=ranked["future_support"]))
    upper = float(np.average(top[outcome_column], weights=top["future_support"]))
    return upper / overall if np.isfinite(overall) and overall != 0 else np.nan


def evaluate_mapped_orders(
    mapped: pd.DataFrame,
    support_threshold: int,
    config: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if mapped.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    target = str(mapped["target"].iloc[0])
    kind = str(TARGET_SPECS[target]["kind"])
    valid = mapped.loc[mapped["eligible_for_metric"]].copy()
    y = valid["target_value"].to_numpy(dtype=float)
    p = valid["profile_score"].to_numpy(dtype=float)
    parent_p = valid["parent_score"].to_numpy(dtype=float)
    raw_p = valid["raw_score"].to_numpy(dtype=float)
    global_p = valid["global_score"].to_numpy(dtype=float)
    min_orders = int(config["validity"]["minimum_future_orders_for_primary_score"])
    class_min = int(config["validity"]["minimum_binary_class_count_per_anchor"])
    valid_anchor = len(valid) >= min_orders
    invalid_reason = ""
    if kind == "binary" and valid_anchor:
        counts = pd.Series(y).value_counts()
        valid_anchor = counts.get(0.0, 0) >= class_min and counts.get(1.0, 0) >= class_min
        if not valid_anchor:
            invalid_reason = "insufficient_binary_class_counts"
    if len(valid) < min_orders:
        invalid_reason = "insufficient_future_target_valid_orders"

    row: dict[str, object] = {
        "base_candidate_id": str(mapped["base_candidate_id"].iloc[0]),
        "support_threshold": int(support_threshold),
        "candidate_id": f"{mapped['base_candidate_id'].iloc[0]}|min_support={int(support_threshold)}",
        "target": target,
        "granularity": str(mapped["granularity"].iloc[0]),
        "future_orders_all_placed": int(len(mapped)),
        "future_mapping_valid_orders": int(mapped["mapping_status"].ne("missing_mapping").sum()),
        "future_seen_orders": int(mapped["mapping_status"].eq("seen").sum()),
        "future_cold_start_orders": int(mapped["mapping_status"].eq("mapped_cold_start").sum()),
        "future_missing_mapping_orders": int(mapped["mapping_status"].eq("missing_mapping").sum()),
        "future_target_valid_orders": int(len(valid)),
        "future_events": float(np.nansum(y)) if kind == "binary" else np.nan,
        "future_seen_coverage": float(mapped["mapping_status"].eq("seen").mean()) if len(mapped) else np.nan,
        "support_qualified_coverage": float((mapped["history_support"].ge(support_threshold) & mapped["mapping_status"].ne("missing_mapping")).mean()) if len(mapped) else np.nan,
        "valid": bool(valid_anchor), "invalid_reason": invalid_reason,
    }
    if kind == "binary":
        candidate_metrics = _binary_metrics(y, p)
        parent_metrics = _binary_metrics(y, parent_p)
        global_metrics = _binary_metrics(y, global_p)
        raw_metrics = _binary_metrics(y, raw_p)
        row.update(candidate_metrics)
        row.update({f"parent_{k}": v for k, v in parent_metrics.items()})
        row.update({f"global_{k}": v for k, v in global_metrics.items()})
        row.update({f"raw_{k}": v for k, v in raw_metrics.items()})
        best_log = np.nanmin([parent_metrics["log_loss"], global_metrics["log_loss"]])
        best_brier = np.nanmin([parent_metrics["brier"], global_metrics["brier"]])
        row["delta_log_loss"] = best_log - candidate_metrics["log_loss"]
        row["delta_brier"] = best_brier - candidate_metrics["brier"]
        if len(valid):
            ranked_orders = valid.sort_values(["profile_score", "order_id"], ascending=[False, True], kind="mergesort")
            top_n = max(1, int(math.ceil(len(ranked_orders) * 0.10)))
            overall = float(ranked_orders["target_value"].mean())
            row["top10_order_lift"] = float(ranked_orders.iloc[:top_n]["target_value"].mean() / overall) if overall > 0 else np.nan
    else:
        candidate_metrics = _continuous_metrics(y, p)
        parent_metrics = _continuous_metrics(y, parent_p)
        global_metrics = _continuous_metrics(y, global_p)
        raw_metrics = _continuous_metrics(y, raw_p)
        row.update(candidate_metrics)
        row.update({f"parent_{k}": v for k, v in parent_metrics.items()})
        row.update({f"global_{k}": v for k, v in global_metrics.items()})
        row.update({f"raw_{k}": v for k, v in raw_metrics.items()})
        best_mae = np.nanmin([parent_metrics["log_mae"], global_metrics["log_mae"]])
        row["log_mae_improvement"] = best_mae - candidate_metrics["log_mae"]
        raw_days = valid["raw_target_value"].to_numpy(dtype=float)
        row["future_mean_days"] = float(np.nanmean(raw_days)) if len(raw_days) else np.nan
        row["future_median_days"] = float(np.nanmedian(raw_days)) if len(raw_days) else np.nan

    entity = valid.groupby("entity_id", dropna=True, sort=True).agg(
        profile_score=("profile_score", "first"), future_mean=("target_value", "mean"),
        future_raw_mean=("raw_target_value", "mean"),
        future_support=("target_value", "count"), history_support=("history_support", "first"),
    ).reset_index()
    rank_valid = (
        len(entity) >= int(config["validity"]["minimum_common_entities_for_rank"])
        and entity["profile_score"].nunique() > 1 and entity["future_mean"].nunique() > 1
    )
    row["n_common_entities"] = int(len(entity))
    row["unweighted_spearman"] = float(spearmanr(entity["profile_score"], entity["future_mean"]).statistic) if rank_valid else np.nan
    row["weighted_spearman"] = weighted_spearman(entity["profile_score"], entity["future_mean"], entity["future_support"]) if rank_valid else np.nan
    row["top_quintile_lift"] = _top_lift(entity, kind) if rank_valid else np.nan
    entity["base_candidate_id"] = row["base_candidate_id"]
    entity["support_threshold"] = int(support_threshold)
    entity["candidate_id"] = row["candidate_id"]
    entity["rank_valid"] = bool(rank_valid)
    entity["invalid_reason"] = "" if rank_valid else "fewer_than_10_or_constant_common_entities"

    def stratum(value: int, status: str) -> str:
        if status == "missing_mapping": return "missing_mapping"
        if value == 0: return "support_0_cold_start"
        if value < 5: return "support_1_4"
        if value < 10: return "support_5_9"
        if value < 20: return "support_10_19"
        return "support_20_plus"

    mapped = mapped.copy()
    mapped["support_stratum"] = [stratum(int(n), str(s)) for n, s in zip(mapped["history_support"], mapped["mapping_status"])]
    strata_rows: list[dict[str, object]] = []
    for name, part in mapped.groupby("support_stratum", sort=True):
        eligible = part.loc[part["eligible_for_metric"]]
        record: dict[str, object] = {
            "base_candidate_id": row["base_candidate_id"], "candidate_id": row["candidate_id"],
            "support_threshold": int(support_threshold), "support_stratum": name,
            "n_orders": int(len(part)), "n_target_valid": int(len(eligible)),
            "n_entities": int(eligible["entity_id"].nunique()),
            "event_rate_or_mean": float(eligible["target_value"].mean()) if len(eligible) else np.nan,
        }
        if len(eligible):
            if kind == "binary":
                cm = _binary_metrics(eligible["target_value"].to_numpy(), eligible["profile_score"].to_numpy())
                pm = _binary_metrics(eligible["target_value"].to_numpy(), eligible["parent_score"].to_numpy())
                gm = _binary_metrics(eligible["target_value"].to_numpy(), eligible["global_score"].to_numpy())
                record["primary_metric"] = "delta_log_loss"
                record["primary_improvement"] = min(pm["log_loss"], gm["log_loss"]) - cm["log_loss"]
            else:
                cm = _continuous_metrics(eligible["target_value"].to_numpy(), eligible["profile_score"].to_numpy())
                pm = _continuous_metrics(eligible["target_value"].to_numpy(), eligible["parent_score"].to_numpy())
                gm = _continuous_metrics(eligible["target_value"].to_numpy(), eligible["global_score"].to_numpy())
                record["primary_metric"] = "log_mae_improvement"
                record["primary_improvement"] = min(pm["log_mae"], gm["log_mae"]) - cm["log_mae"]
        else:
            record.update({"primary_metric": "", "primary_improvement": np.nan})
        strata_rows.append(record)
    return row, entity, pd.DataFrame(strata_rows)


def stability_between_profiles(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    previous_date: pd.Timestamp,
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common_candidates = sorted(set(previous["base_candidate_id"]) & set(current["base_candidate_id"]))
    for candidate_id in common_candidates:
        left = previous.loc[previous["base_candidate_id"].eq(candidate_id), ["entity_id", "score"]].rename(columns={"score": "previous_score"})
        right = current.loc[current["base_candidate_id"].eq(candidate_id), ["entity_id", "score"]].rename(columns={"score": "current_score"})
        common = left.merge(right, on="entity_id", how="inner", validate="1:1").dropna()
        valid = len(common) >= 10 and common["previous_score"].nunique() > 1 and common["current_score"].nunique() > 1
        change = np.abs(common["current_score"] - common["previous_score"]) if len(common) else pd.Series(dtype=float)
        top_n = max(1, int(math.ceil(len(common) * 0.20))) if len(common) else 0
        top_left = set(common.sort_values(["previous_score", "entity_id"], ascending=[False, True], kind="mergesort").head(top_n)["entity_id"])
        top_right = set(common.sort_values(["current_score", "entity_id"], ascending=[False, True], kind="mergesort").head(top_n)["entity_id"])
        union = top_left | top_right
        sample = current.loc[current["base_candidate_id"].eq(candidate_id)].iloc[0]
        rows.append({
            "base_candidate_id": candidate_id, "target": sample["target"], "granularity": sample["granularity"],
            "previous_snapshot_date": pd.Timestamp(previous_date), "snapshot_date": pd.Timestamp(current_date),
            "n_common_entities": int(len(common)),
            "day_to_day_spearman": float(spearmanr(common["previous_score"], common["current_score"]).statistic) if valid else np.nan,
            "median_absolute_score_change": float(change.median()) if len(change) else np.nan,
            "p90_absolute_score_change": float(change.quantile(0.90)) if len(change) else np.nan,
            "top20_jaccard": float(len(top_left & top_right) / len(union)) if union else np.nan,
            "valid": bool(valid), "invalid_reason": "" if valid else "fewer_than_10_or_constant_common_entities",
        })
    return pd.DataFrame(rows)


def write_selection_freeze(
    freeze_path: Path,
    sidecar_path: Path,
    payload: Mapping[str, object],
    config: Mapping[str, object],
) -> str:
    """Atomically persist canonical JSON and its detached SHA-256 token."""
    body = dict(payload)
    body.setdefault("frozen_config_sha256", sha256_file(CONFIG_PATH))
    encoded = (json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode("utf-8")
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = freeze_path.with_name(freeze_path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, freeze_path)
    digest = sha256_file(freeze_path)
    side_tmp = sidecar_path.with_name(sidecar_path.name + f".tmp.{os.getpid()}")
    side_tmp.write_text(f"{digest}  {freeze_path.name}\n", encoding="utf-8")
    os.replace(side_tmp, sidecar_path)
    return digest


def verify_selection_freeze(
    freeze_path: Path,
    sidecar_path: Path,
    config: Mapping[str, object],
) -> dict:
    """Verify a detached freeze token without opening any outcome source."""
    if not freeze_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("selection freeze or SHA sidecar missing")
    expected = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(freeze_path)
    if actual != expected:
        raise RuntimeError("selection freeze SHA mismatch")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    config_hash = payload.get("frozen_config_sha256")
    if config_hash is not None and config_hash != sha256_file(CONFIG_PATH):
        raise RuntimeError("selection freeze references a different frozen config")
    if "promoted_candidates" not in payload:
        raise ValueError("selection freeze missing promoted_candidates")
    return payload


def assign_frozen_levels(
    rows: pd.DataFrame,
    minimum_support: int,
    low_medium_cutoff: float,
    medium_high_cutoff: float,
) -> pd.Series:
    """Apply the frozen Unknown/Low/Medium/High communication layer."""
    score = pd.to_numeric(rows["score"], errors="coerce")
    support = pd.to_numeric(rows["support"], errors="coerce").fillna(0)
    lower = pd.to_numeric(rows["lower_interval"], errors="coerce")
    upper = pd.to_numeric(rows["upper_interval"], errors="coerce")
    cold = rows["cold_start"].fillna(False).astype(bool)
    q33 = float(low_medium_cutoff)
    q67 = float(medium_high_cutoff)
    if not np.isfinite(q33) or not np.isfinite(q67) or q33 > q67:
        return pd.Series("Unknown", index=rows.index, dtype="string")
    unknown = (
        support.lt(int(minimum_support)) | cold | score.isna() | lower.isna() | upper.isna()
        | (lower.le(q33) & upper.ge(q67))
    )
    return pd.Series(
        np.select([unknown, score.le(q33), score.le(q67)], ["Unknown", "Low", "Medium"], default="High"),
        index=rows.index, dtype="string",
    )


def pareto_frontier(
    candidates: pd.DataFrame,
    maximize: Iterable[str],
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Generic deterministic all-maximise Pareto helper used by unit fixtures."""
    dimensions = list(maximize)
    missing = sorted(set(["candidate_id", *dimensions]) - set(candidates.columns))
    if missing:
        raise KeyError(missing)
    out = candidates.copy().reset_index(drop=True)
    values = out[dimensions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pareto = np.isfinite(values).all(axis=1)
    dominated_by: list[list[str]] = [[] for _ in range(len(out))]
    for i in range(len(out)):
        if not pareto[i]:
            continue
        for j in range(len(out)):
            if i == j or not np.isfinite(values[j]).all():
                continue
            no_worse = np.all(values[j] >= values[i] - tolerance)
            strict = np.any(values[j] > values[i] + tolerance)
            if no_worse and strict:
                pareto[i] = False
                dominated_by[i].append(str(out.loc[j, "candidate_id"]))
    out["is_pareto"] = pareto
    out["dominated"] = ~pareto
    out["dominated_by"] = [";".join(sorted(items)) for items in dominated_by]
    return out
