"""Comparable fixed-effects classification pipelines."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def estimators(seed: int, include_xgboost: bool = True):
    result = {
        "logistic_regression": LogisticRegression(
            penalty="l2", C=1.0, class_weight=None, solver="liblinear",
            max_iter=2000, random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=20, max_features="sqrt",
            class_weight=None, random_state=seed, n_jobs=-1,
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            result["xgboost"] = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            )
        except ImportError:
            pass
    return result


def pipeline(numeric: list[str], categorical: list[str], estimator):
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])
