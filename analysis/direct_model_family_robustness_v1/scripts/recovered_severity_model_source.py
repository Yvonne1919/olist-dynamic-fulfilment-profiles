"""Executable archive of the exact recovered severity-family definitions.

The source logic was recovered from ``SEV_run_alternative_models.py`` with
SHA-256 b328faf367fedcb2db239bbffd83dc63fd822efcb0f9a61ca819cab6419e65f2.
No historical result table is imported.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


SEED = 20260828


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * cumulative[-1]
    return float(
        sorted_values[
            min(np.searchsorted(cumulative, cutoff, side="left"), len(sorted_values) - 1)
        ]
    )


def random_forest_quantiles(
    model: RandomForestRegressor,
    x_train,
    y_train: np.ndarray,
    x_test,
    quantiles: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Exact recovered all-training-row leaf-weighted quantile approximation."""

    train_leaves = model.apply(x_train)
    test_leaves = model.apply(x_test)
    tree_maps: list[dict[int, np.ndarray]] = []
    for tree_index in range(train_leaves.shape[1]):
        mapping: dict[int, np.ndarray] = {}
        leaves = train_leaves[:, tree_index]
        for leaf in np.unique(leaves):
            mapping[int(leaf)] = y_train[leaves == leaf]
        tree_maps.append(mapping)
    result = {q: np.empty(len(test_leaves), dtype=float) for q in quantiles}
    tree_weight = 1.0 / train_leaves.shape[1]
    for row_index in range(len(test_leaves)):
        values_parts = []
        weight_parts = []
        for tree_index, mapping in enumerate(tree_maps):
            values = mapping[int(test_leaves[row_index, tree_index])]
            values_parts.append(values)
            weight_parts.append(np.full(len(values), tree_weight / len(values)))
        values = np.concatenate(values_parts)
        weights = np.concatenate(weight_parts)
        for quantile in quantiles:
            result[quantile][row_index] = weighted_quantile(values, weights, quantile)
    return result


def make_quantile_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10,
        max_features=0.7,
        n_jobs=4,
        random_state=SEED,
    )


def make_lognormal_ridge() -> Ridge:
    return Ridge(alpha=1.0, solver="lsqr")


def lognormal_quantiles(location: np.ndarray, sigma: float, quantiles: tuple[float, ...]) -> dict[float, np.ndarray]:
    return {
        q: np.maximum(np.exp(location + norm.ppf(q) * sigma), 0.0)
        for q in quantiles
    }


__all__ = [
    "SEED",
    "lognormal_quantiles",
    "make_lognormal_ridge",
    "make_quantile_forest",
    "random_forest_quantiles",
    "weighted_quantile",
]
