"""Temporal split definitions with explicit integrity assertions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def development_final_holdout(frame: pd.DataFrame, fraction: float = 0.20):
    ordered = frame.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    cut = int(len(ordered) * (1 - fraction))
    development, final = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    assert development["order_purchase_timestamp"].max() <= final["order_purchase_timestamp"].min()
    assert set(development.order_id).isdisjoint(final.order_id)
    return development, final


def random_diagnostic_splits(frame: pd.DataFrame, target: str, n_splits: int, seed: int):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, frame[target].astype(int)))


def expanding_month_splits(frame: pd.DataFrame, min_train_months: int = 6):
    work = frame.copy()
    work["_month"] = pd.to_datetime(work["order_purchase_timestamp"]).dt.to_period("M")
    counts = work.groupby("_month")["order_id"].nunique()
    boundary = {work["_month"].min(), work["_month"].max()}
    months = list(counts[(counts >= 500) & ~counts.index.isin(boundary)].index.sort_values())
    folds = []
    for position in range(min_train_months, len(months)):
        train_months, test_month = months[:position], months[position]
        train_idx = np.flatnonzero(work["_month"].isin(train_months).to_numpy())
        test_idx = np.flatnonzero(work["_month"].eq(test_month).to_numpy())
        if len(test_idx) == 0 or work.iloc[test_idx]["late_delivery"].sum() < 20:
            continue
        train, test = work.iloc[train_idx], work.iloc[test_idx]
        assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
        assert set(train.order_id).isdisjoint(test.order_id)
        folds.append((len(folds) + 1, str(test_month), train_idx, test_idx))
    return folds
