"""Deterministic statistical core for the supplementary RQ1 V1 study.

The authorised reader-facing specification writes ``cr(x, df=4)``.  Patsy
1.0.2's unconstrained ``cr`` basis contains a constant direction; together
with the GLM intercept this is rank deficient.  We therefore absorb a centring
constraint into the natural-cubic-regression-spline basis.  The intercept plus
the four centred spline columns spans the same identifiable natural-spline
model space, while retaining exactly four spline degrees of freedom.  This is
an identification parameterisation, not outcome-driven model selection.

This module performs no I/O.  Its public entry point accepts the already
reconciled reviewed-order frame and returns tables/dictionaries for the
reporting layer to persist.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import patsy
from scipy.special import expit
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf


SPLINE_DF = 4
SPLINE_CONSTRAINT = "center"
COVARIANCE_TYPE = "HC1"
MVN_METHOD = "cholesky"
MVN_QUANTILE_METHOD = "linear"
IRLS_MAXITER = 100
IRLS_TOL = 1e-8
MAX_DESIGN_CONDITION = 1e8
NORMAL_975 = float(norm.ppf(0.975))
SIMULATION_SEED = 20260824
SIMULATION_DRAWS = 10_000

# Public frozen formula templates used by tests/audits.  Runtime formulas only
# substitute the references from the frozen config; the statistical terms are
# identical to these templates.
MODEL_FORMULAS = {
    "A": (
        'low_review_2 ~ cr(actual_delivery_days, df=4, constraints="center") '
        '+ C(purchase_month_adjustment, Treatment(reference="2018-01"))'
    ),
    "B": (
        'low_review_2 ~ cr(actual_delivery_days, df=4, constraints="center") '
        '+ C(promise_error_group_label, Treatment(reference="on promised date")) '
        '+ C(purchase_month_adjustment, Treatment(reference="2018-01"))'
    ),
    "C": (
        'low_review_2 ~ cr(promised_lead_days, df=4, constraints="center") '
        '+ C(promise_error_group_label, Treatment(reference="on promised date")) '
        '+ C(purchase_month_adjustment, Treatment(reference="2018-01"))'
    ),
}

MODEL_RESULT_COLUMNS = (
    "variant", "model_id", "outcome", "formula", "term", "term_block",
    "coefficient_log_odds", "standard_error_hc1", "z_value", "p_value",
    "odds_ratio", "odds_ratio_ci_lower", "odds_ratio_ci_upper", "n_orders",
    "converged", "irls_iterations",
)
COVARIANCE_COLUMNS = (
    "variant", "model_id", "row_term", "column_term", "covariance_hc1",
)
WALD_COLUMNS = (
    "variant", "model_id", "outcome", "block", "term_name", "terms",
    "wald_chi2", "df", "p_value", "covariance", "n_orders", "passed",
)
PROBABILITY_COLUMNS = (
    "variant", "model_id", "outcome", "estimand_id", "estimand_type",
    "continuous_variable", "continuous_value", "error_group",
    "reference_error_group", "reference_estimate", "standardization_population",
    "n_standardization", "month_levels", "estimate", "standard_error",
    "ci_lower", "ci_upper", "ci_method", "support_rule", "support_lower",
    "support_upper", "support_n",
)
CONTRAST_COLUMNS = (
    "variant", "model_id", "outcome", "contrast", "contrast_id", "contrast_family",
    "high_setting", "low_setting", "estimate", "standard_error", "ci_lower",
    "ci_upper", "ci_method", "continuous_variable", "low_value", "high_value",
    "reference_value", "error_group", "reference_error_group",
    "standardization_population", "n_standardization", "support_rule",
    "support_lower", "support_upper", "support_n", "available", "blocker",
)
COMPARISON_COLUMNS = (
    "variant", "model_id", "outcome", "contrast_id", "estimate",
    "standard_error", "q025", "q975", "interval_excludes_zero", "seed",
    "n_draws", "n_valid", "mvn_method", "quantile_method", "covariance",
    "parameter_order_hash", "design_settings_hash", "fixed_settings_json",
    "draws_all_finite", "probability_min", "probability_max",
)
SENSITIVITY_COLUMNS = (
    "variant", "analysis_sample", "record_type", "model_id", "outcome",
    "record_id", "term", "group",
    "n_orders", "estimate", "standard_error", "ci_lower", "ci_upper",
    "p_value", "odds_ratio", "odds_ratio_ci_lower", "odds_ratio_ci_upper",
    "direction_positive", "support_rule", "support_lower", "support_upper",
    "support_n", "notes",
)


class StatisticalDesignError(RuntimeError):
    """Raised when a frozen statistical or numerical contract is violated."""


@dataclass
class FitBundle:
    variant: str
    model_id: str
    outcome: str
    formula: str
    data: pd.DataFrame
    result: Any
    covariance: np.ndarray
    design_info: patsy.DesignInfo
    month_audit: dict[str, Any]
    diagnostic: dict[str, Any]


@dataclass
class StandardizedSetting:
    estimate: float
    gradient: np.ndarray
    design: np.ndarray
    weights: np.ndarray
    settings: dict[str, Any]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise StatisticalDesignError(f"Frozen config section {name!r} is missing.")
    return value


def _quoted(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _resolve_column(frame: pd.DataFrame, canonical: str, aliases: Sequence[str]) -> str:
    for candidate in (canonical, *aliases):
        if candidate in frame.columns:
            return candidate
    raise StatisticalDesignError(
        f"Reviewed frame lacks required column {canonical!r}; aliases={list(aliases)!r}."
    )


def _numeric_integer(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise StatisticalDesignError(f"{name} contains missing/non-numeric values.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.round(values)).all():
        raise StatisticalDesignError(f"{name} must contain finite integer calendar days.")
    return pd.Series(values.astype(np.int64), index=series.index, name=name)


def _parse_edges(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, str) and value.lower() in {"-infinity", "-inf"}:
            result.append(-np.inf)
        elif isinstance(value, str) and value.lower() in {"infinity", "+infinity", "inf", "+inf"}:
            result.append(np.inf)
        else:
            result.append(float(value))
    return result


def _prepare_reviewed(reviewed: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(reviewed, pd.DataFrame):
        raise TypeError("reviewed must be a pandas DataFrame")
    data = reviewed.copy()
    order_col = _resolve_column(data, "order_id", ())
    score_col = next(
        (name for name in ("selected_review_score", "review_score") if name in data),
        None,
    )
    d_col = _resolve_column(data, "actual_delivery_days", ("D",))
    p_col = _resolve_column(data, "promised_lead_days", ("P",))
    e_col = _resolve_column(data, "promise_error_days", ("E",))

    data["order_id"] = data[order_col].astype("string")
    if data["order_id"].isna().any() or data["order_id"].duplicated().any():
        raise StatisticalDesignError("Reviewed input must contain one non-null row per order_id.")
    if score_col is not None:
        data["selected_review_score"] = _numeric_integer(data[score_col], "selected_review_score")
        if not data["selected_review_score"].between(1, 5).all():
            raise StatisticalDesignError("Selected review scores must be integers in 1..5.")
    elif {"low_review_2", "low_review_3"}.issubset(data.columns):
        low2 = _numeric_integer(data["low_review_2"], "low_review_2")
        low3 = _numeric_integer(data["low_review_3"], "low_review_3")
        if not low2.isin([0, 1]).all() or not low3.isin([0, 1]).all() or (low2 > low3).any():
            raise StatisticalDesignError("Supplied synthetic low-review targets are inconsistent.")
        # A deterministic compatible score is sufficient for the statistical
        # core; real runs always arrive with the selected observed score.
        data["selected_review_score"] = np.select(
            [low2.eq(1), low3.eq(1)], [2, 3], default=5,
        ).astype(np.int8)
    else:
        raise StatisticalDesignError(
            "Reviewed frame needs selected_review_score or both frozen binary outcomes."
        )
    data["actual_delivery_days"] = _numeric_integer(data[d_col], "actual_delivery_days")
    data["promised_lead_days"] = _numeric_integer(data[p_col], "promised_lead_days")
    data["promise_error_days"] = _numeric_integer(data[e_col], "promise_error_days")
    if data["actual_delivery_days"].lt(0).any():
        raise StatisticalDesignError("Negative actual delivery duration cannot enter a model.")
    if data["promised_lead_days"].lt(0).any():
        raise StatisticalDesignError("Negative promised lead time cannot enter a model.")
    identity = (
        data["actual_delivery_days"]
        - data["promised_lead_days"]
        - data["promise_error_days"]
    )
    if not identity.eq(0).all():
        raise StatisticalDesignError(
            f"D=P+E identity failed for {int(identity.ne(0).sum())} reviewed orders."
        )

    if "purchase_month" in data:
        month = data["purchase_month"].astype("string")
    elif "purchase_month_label" in data:
        month = data["purchase_month_label"].astype("string")
    else:
        purchase_col = _resolve_column(data, "order_purchase_timestamp", ())
        purchase = pd.to_datetime(data[purchase_col], errors="coerce")
        if purchase.isna().any():
            raise StatisticalDesignError("Purchase timestamp is missing for reviewed orders.")
        month = purchase.dt.to_period("M").astype("string")
    if month.isna().any():
        raise StatisticalDesignError("Purchase month is missing for reviewed orders.")
    data["purchase_month"] = month

    error_cfg = _config_section(config, "promise_error_groups")
    labels = [str(value) for value in error_cfg["labels"]]
    edges = _parse_edges(error_cfg["edges"])
    calculated = pd.cut(
        data["promise_error_days"], bins=edges, labels=labels,
        right=bool(error_cfg.get("right_closed", True)), include_lowest=True,
        ordered=True,
    )
    if calculated.isna().any():
        raise StatisticalDesignError("At least one promise-error value is outside frozen groups.")
    provided_col = next(
        (name for name in ("promise_error_group", "promise_error_group_label", "error_group") if name in data),
        None,
    )
    if provided_col is not None:
        provided = data[provided_col].astype("string")
        if not provided.eq(calculated.astype("string")).all():
            raise StatisticalDesignError("Provided promise-error groups differ from frozen grouping.")
    data["promise_error_group"] = pd.Categorical(
        calculated.astype("string"), categories=labels, ordered=True,
    )
    data["promise_error_group_label"] = pd.Categorical(
        calculated.astype("string"), categories=labels, ordered=True,
    )

    duration_cfg = _config_section(config, "actual_duration_groups")
    duration_labels = [str(value) for value in duration_cfg["labels"]]
    duration_edges = _parse_edges(duration_cfg["edges"])
    duration_bins = pd.cut(
        data["actual_delivery_days"], bins=duration_edges, labels=duration_labels,
        right=bool(duration_cfg.get("right_closed", True)), include_lowest=True,
        ordered=True,
    )
    if duration_bins.isna().any():
        raise StatisticalDesignError("At least one valid duration is outside frozen bins.")
    data["actual_duration_group"] = pd.Categorical(
        duration_bins.astype("string"), categories=duration_labels, ordered=True,
    )
    data["low_review_2"] = data["selected_review_score"].le(2).astype(np.int8)
    data["low_review_3"] = data["selected_review_score"].le(3).astype(np.int8)

    answer_col = next(
        (name for name in ("selected_review_answer_timestamp", "review_answer_timestamp") if name in data),
        None,
    )
    actual_ts_col = next(
        (name for name in ("order_delivered_customer_date", "actual_delivery_timestamp") if name in data),
        None,
    )
    if answer_col is not None and actual_ts_col is not None:
        data["_review_answer_timestamp"] = pd.to_datetime(data[answer_col], errors="coerce")
        data["_actual_delivery_timestamp"] = pd.to_datetime(data[actual_ts_col], errors="coerce")
        data["_review_at_or_after_delivery"] = (
            data["_review_answer_timestamp"].notna()
            & data["_actual_delivery_timestamp"].notna()
            & data["_review_answer_timestamp"].ge(data["_actual_delivery_timestamp"])
        )
    elif "review_at_or_after_delivery" in data:
        data["_review_at_or_after_delivery"] = data["review_at_or_after_delivery"].astype(bool)
        data["_review_answer_timestamp"] = pd.NaT
        data["_actual_delivery_timestamp"] = pd.NaT
    else:
        raise StatisticalDesignError(
            "Review-timing sensitivity needs timestamps or review_at_or_after_delivery."
        )
    observed_groups = set(data["promise_error_group"].astype("string"))
    if observed_groups != set(labels):
        raise StatisticalDesignError("All eight frozen promise-error groups must be present.")
    return data.reset_index(drop=True)


def _with_month_adjustment(
    data: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_cfg = _config_section(config, "models")
    minimum = int(model_cfg["month_min_orders"])
    pooled_label = str(model_cfg["sparse_month_label"])
    reference = str(model_cfg["month_reference"])
    output = data.copy()
    counts = output["purchase_month"].astype("string").value_counts()
    sparse = sorted(str(value) for value in counts[counts.lt(minimum)].index)
    adjustment = output["purchase_month"].astype("string").where(
        ~output["purchase_month"].astype("string").isin(sparse), pooled_label,
    )
    levels = sorted(str(value) for value in adjustment.unique() if str(value) != pooled_label)
    if sparse:
        levels.append(pooled_label)
    if reference not in levels:
        raise StatisticalDesignError(f"Frozen month reference {reference!r} is absent.")
    output["purchase_month_adjustment"] = pd.Categorical(
        adjustment, categories=levels, ordered=False,
    )
    adjusted_counts = output["purchase_month_adjustment"].value_counts(sort=False)
    audit = {
        "raw_levels": int(len(counts)),
        "adjustment_levels": int(len(levels)),
        "sparse_months": sparse,
        "minimum_orders": minimum,
        "reference": reference,
        "pooled_label": pooled_label,
        "adjusted_counts": {str(k): int(v) for k, v in adjusted_counts.items()},
    }
    return output, audit


def _formula(kind: str, outcome: str, config: Mapping[str, Any]) -> str:
    model_cfg = _config_section(config, "models")
    error_cfg = _config_section(config, "promise_error_groups")
    month_reference = _quoted(str(model_cfg["month_reference"]))
    error_reference = _quoted(str(error_cfg["reference"]))
    month_term = (
        f"C(purchase_month_adjustment, Treatment(reference={month_reference}))"
    )
    error_term = (
        f"C(promise_error_group_label, Treatment(reference={error_reference}))"
    )
    centered_actual = (
        f'cr(actual_delivery_days, df={SPLINE_DF}, constraints="{SPLINE_CONSTRAINT}")'
    )
    centered_promise = (
        f'cr(promised_lead_days, df={SPLINE_DF}, constraints="{SPLINE_CONSTRAINT}")'
    )
    if kind == "A":
        return f"{outcome} ~ {centered_actual} + {month_term}"
    if kind == "B":
        return f"{outcome} ~ {centered_actual} + {error_term} + {month_term}"
    if kind == "C":
        return f"{outcome} ~ {centered_promise} + {error_term} + {month_term}"
    if kind == "BIN_B":
        bin_reference = _quoted(str(_config_section(config, "actual_duration_groups")["labels"][0]))
        bin_term = f"C(actual_duration_group, Treatment(reference={bin_reference}))"
        return f"{outcome} ~ {bin_term} + {error_term} + {month_term}"
    raise ValueError(f"Unknown model kind: {kind}")


def model_formula(
    kind: str,
    outcome: str,
    config: Mapping[str, Any],
) -> str:
    """Return the executable frozen formula for model A, B, C, or BIN_B."""
    return _formula(str(kind).upper(), str(outcome), config)


def _term_for(design_info: patsy.DesignInfo, predicate: Callable[[str], bool]) -> str:
    matches = [str(name) for name in design_info.term_name_slices if predicate(str(name))]
    if len(matches) != 1:
        raise StatisticalDesignError(f"Expected one design term, found {matches!r}.")
    return matches[0]


def _fit(
    data: pd.DataFrame,
    *,
    variant: str,
    model_id: str,
    outcome: str,
    kind: str,
    config: Mapping[str, Any],
    month_audit: dict[str, Any],
) -> FitBundle:
    formula = _formula(kind, outcome, config)
    result = smf.glm(
        formula=formula, data=data, family=sm.families.Binomial(),
    ).fit(
        method="IRLS", maxiter=IRLS_MAXITER, tol=IRLS_TOL,
        cov_type=COVARIANCE_TYPE, use_t=False,
    )
    design = np.asarray(result.model.exog, dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    columns = int(design.shape[1])
    condition = float(np.linalg.cond(design))
    covariance = np.asarray(result.cov_params(), dtype=float)
    if not bool(result.converged):
        raise StatisticalDesignError(f"{variant}/{model_id} did not converge.")
    if str(result.cov_type).upper() != COVARIANCE_TYPE or bool(result.use_t):
        raise StatisticalDesignError(f"{variant}/{model_id} did not retain HC1 z inference.")
    if not np.isfinite(result.params.to_numpy(dtype=float)).all():
        raise StatisticalDesignError(f"{variant}/{model_id} has non-finite coefficients.")
    if not np.isfinite(result.bse.to_numpy(dtype=float)).all():
        raise StatisticalDesignError(f"{variant}/{model_id} has non-finite HC1 SEs.")
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance, covariance.T, rtol=1e-10, atol=1e-12,
    ):
        raise StatisticalDesignError(f"{variant}/{model_id} HC1 covariance is not finite/symmetric.")
    if rank != columns:
        raise StatisticalDesignError(
            f"{variant}/{model_id} exog is rank deficient ({rank}/{columns})."
        )
    if not np.isfinite(condition) or condition > MAX_DESIGN_CONDITION:
        raise StatisticalDesignError(
            f"{variant}/{model_id} exog condition number is unsafe: {condition:.6g}."
        )
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise StatisticalDesignError(
            f"{variant}/{model_id} HC1 covariance is not positive definite."
        ) from exc
    design_info = result.model.data.design_info
    spline_terms = [
        (str(name), sl) for name, sl in design_info.term_name_slices.items()
        if str(name).startswith("cr(")
    ]
    for term, sl in spline_terms:
        if sl.stop - sl.start != SPLINE_DF or "constraints=\"center\"" not in term:
            raise StatisticalDesignError(
                f"{variant}/{model_id} does not contain the identifiable four-df CR basis."
            )
    iterations = int(result.fit_history.get("iteration", -1))
    eigenvalues = np.linalg.eigvalsh(covariance)
    diagnostic = {
        "variant": variant,
        "model_id": model_id,
        "outcome": outcome,
        "formula": formula,
        "n_orders": int(result.nobs),
        "converged": bool(result.converged),
        "irls_iterations": iterations,
        "covariance": str(result.cov_type),
        "covariance_type": str(result.cov_type),
        "error_reference": str(
            _config_section(config, "promise_error_groups")["reference"]
        ) if kind in {"B", "C", "BIN_B"} else None,
        "use_t": bool(result.use_t),
        "design_rows": int(design.shape[0]),
        "design_columns": columns,
        "design_rank": rank,
        "design_full_rank": bool(rank == columns),
        "design_condition_number": condition,
        "covariance_min_eigenvalue": float(eigenvalues.min()),
        "covariance_max_eigenvalue": float(eigenvalues.max()),
        "covariance_positive_definite": True,
        "hc1_all_finite": True,
        "hc1_cholesky_passed": True,
        "hc1_min_eigenvalue": float(eigenvalues.min()),
        "spline_df": SPLINE_DF if spline_terms else None,
        "spline_columns": int(sum(sl.stop - sl.start for _, sl in spline_terms)),
        "spline_constraint": SPLINE_CONSTRAINT if spline_terms else None,
        "design_info_reused_for_prediction": True,
        "prediction_design_columns_match": True,
        "parameter_order": [str(value) for value in result.params.index],
        "parameter_order_hash": _stable_hash([str(value) for value in result.params.index]),
    }
    return FitBundle(
        variant=variant, model_id=model_id, outcome=outcome, formula=formula,
        data=data, result=result, covariance=covariance, design_info=design_info,
        month_audit=month_audit, diagnostic=diagnostic,
    )


def _block_for_term(bundle: FitBundle, position: int) -> str:
    for term, sl in bundle.design_info.term_name_slices.items():
        if sl.start <= position < sl.stop:
            return str(term)
    raise StatisticalDesignError("Coefficient position is absent from DesignInfo.")


def _model_rows(bundle: FitBundle) -> list[dict[str, Any]]:
    result = bundle.result
    params = result.params.to_numpy(dtype=float)
    se = result.bse.to_numpy(dtype=float)
    z_values = params / se
    p_values = result.pvalues.to_numpy(dtype=float)
    lower = params - NORMAL_975 * se
    upper = params + NORMAL_975 * se
    return [
        {
            "variant": bundle.variant,
            "model_id": bundle.model_id,
            "outcome": bundle.outcome,
            "formula": bundle.formula,
            "term": str(term),
            "term_block": _block_for_term(bundle, index),
            "coefficient_log_odds": float(params[index]),
            "standard_error_hc1": float(se[index]),
            "z_value": float(z_values[index]),
            "p_value": float(p_values[index]),
            "odds_ratio": float(np.exp(params[index])),
            "odds_ratio_ci_lower": float(np.exp(lower[index])),
            "odds_ratio_ci_upper": float(np.exp(upper[index])),
            "n_orders": int(result.nobs),
            "converged": bool(result.converged),
            "irls_iterations": int(result.fit_history.get("iteration", -1)),
        }
        for index, term in enumerate(result.params.index)
    ]


def _covariance_rows(bundle: FitBundle) -> list[dict[str, Any]]:
    terms = [str(value) for value in bundle.result.params.index]
    return [
        {
            "variant": bundle.variant,
            "model_id": bundle.model_id,
            "row_term": row_term,
            "column_term": column_term,
            "covariance_hc1": float(bundle.covariance[row, column]),
        }
        for row, row_term in enumerate(terms)
        for column, column_term in enumerate(terms)
    ]


def _wald(bundle: FitBundle, block: str) -> dict[str, Any]:
    if block == "duration_spline":
        term_name = _term_for(bundle.design_info, lambda name: name.startswith("cr("))
    elif block == "error_group":
        term_name = _term_for(
            bundle.design_info,
            lambda name: name.startswith("C(promise_error_group_label,"),
        )
    elif block == "duration_bin":
        term_name = _term_for(
            bundle.design_info,
            lambda name: name.startswith("C(actual_duration_group,"),
        )
    else:
        raise ValueError(block)
    sl = bundle.design_info.term_name_slices[term_name]
    restriction = np.eye(len(bundle.result.params), dtype=float)[sl]
    restricted_covariance = restriction @ bundle.covariance @ restriction.T
    block_df = int(restriction.shape[0])
    if np.linalg.matrix_rank(restricted_covariance) != block_df:
        raise StatisticalDesignError(
            f"{bundle.variant}/{bundle.model_id} {block} Wald covariance is singular."
        )
    test = bundle.result.wald_test(
        restriction, cov_p=bundle.covariance, use_f=False, scalar=True,
    )
    statistic = float(test.statistic)
    p_value = float(test.pvalue)
    if not np.isfinite(statistic) or statistic < 0 or not (0 <= p_value <= 1):
        raise StatisticalDesignError(
            f"{bundle.variant}/{bundle.model_id} {block} returned invalid Wald output."
        )
    terms = [str(value) for value in bundle.result.params.index[sl]]
    return {
        "variant": bundle.variant,
        "model_id": bundle.model_id,
        "outcome": bundle.outcome,
        "block": block,
        "term_name": term_name,
        "terms": json.dumps(terms, ensure_ascii=False),
        "wald_chi2": statistic,
        "df": block_df,
        "p_value": p_value,
        "covariance": COVARIANCE_TYPE,
        "n_orders": int(bundle.result.nobs),
        "passed": True,
    }


def _month_weights(bundle: FitBundle) -> pd.DataFrame:
    counts = (
        bundle.data["purchase_month_adjustment"].astype("string")
        .value_counts(sort=False).rename_axis("purchase_month_adjustment")
        .reset_index(name="n_orders")
    )
    counts = counts.loc[counts["n_orders"].gt(0)].copy()
    counts["weight"] = counts["n_orders"] / counts["n_orders"].sum()
    if not np.isclose(counts["weight"].sum(), 1.0, rtol=0, atol=1e-14):
        raise StatisticalDesignError("Purchase-month standardisation weights do not sum to one.")
    return counts.reset_index(drop=True)


def _standardized_setting(
    bundle: FitBundle,
    settings: Mapping[str, Any],
) -> StandardizedSetting:
    months = _month_weights(bundle)
    new_data = pd.DataFrame(
        {"purchase_month_adjustment": months["purchase_month_adjustment"].astype(str)}
    )
    for key, value in settings.items():
        new_data[key] = value
    try:
        matrix = patsy.build_design_matrices(
            [bundle.design_info], new_data, return_type="dataframe",
        )[0]
    except Exception as exc:
        raise StatisticalDesignError(
            f"Could not reuse fitted DesignInfo for {bundle.model_id}: {settings!r}."
        ) from exc
    design = np.asarray(matrix, dtype=float)
    if list(matrix.columns) != [str(value) for value in bundle.result.params.index]:
        raise StatisticalDesignError("Prediction design columns differ from fitted parameter order.")
    weights = months["weight"].to_numpy(dtype=float)
    probabilities = expit(design @ bundle.result.params.to_numpy(dtype=float))
    estimate = float(weights @ probabilities)
    gradient = (weights * probabilities * (1.0 - probabilities)) @ design
    if not np.isfinite(estimate) or not np.isfinite(gradient).all():
        raise StatisticalDesignError("Non-finite standardized probability or gradient.")
    return StandardizedSetting(
        estimate=estimate, gradient=np.asarray(gradient, dtype=float),
        design=design, weights=weights, settings=dict(settings),
    )


def _delta_interval(
    estimate: float, gradient: np.ndarray, covariance: np.ndarray,
) -> tuple[float, float, float]:
    variance = float(gradient @ covariance @ gradient)
    if variance < -1e-12 or not np.isfinite(variance):
        raise StatisticalDesignError(f"Invalid delta-method variance: {variance!r}.")
    standard_error = float(np.sqrt(max(0.0, variance)))
    return (
        standard_error,
        float(estimate - NORMAL_975 * standard_error),
        float(estimate + NORMAL_975 * standard_error),
    )


def _probability_row(
    bundle: FitBundle,
    setting: StandardizedSetting,
    *,
    estimand_id: str,
    estimand_type: str,
    continuous_variable: str,
    continuous_value: float,
    error_group: str = "",
    reference_error_group: str = "",
    reference_estimate: float | None = None,
    support_rule: str,
    support_lower: float,
    support_upper: float,
    support_n: int,
) -> dict[str, Any]:
    standard_error, lower, upper = _delta_interval(
        setting.estimate, setting.gradient, bundle.covariance,
    )
    return {
        "variant": bundle.variant,
        "model_id": bundle.model_id,
        "outcome": bundle.outcome,
        "estimand_id": estimand_id,
        "estimand_type": estimand_type,
        "continuous_variable": continuous_variable,
        "continuous_value": float(continuous_value),
        "error_group": error_group,
        "reference_error_group": reference_error_group,
        "reference_estimate": np.nan if reference_estimate is None else float(reference_estimate),
        "standardization_population": bundle.variant,
        "n_standardization": int(bundle.result.nobs),
        "month_levels": int(len(_month_weights(bundle))),
        "estimate": setting.estimate,
        "standard_error": standard_error,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_method": "HC1 delta-Wald",
        "support_rule": support_rule,
        "support_lower": float(support_lower),
        "support_upper": float(support_upper),
        "support_n": int(support_n),
    }


def _public_contrast_name(contrast_id: str) -> str:
    """Return the frozen short label used by reporting and audit tests."""
    return {
        "C_speed": "speed",
        "C_late_4_7": "late_4_7",
        "C_late_8_plus": "late_8_plus",
        "C_difference_4_7_minus_speed": "difference",
    }.get(contrast_id, contrast_id)


def _contrast_row(
    bundle: FitBundle,
    *,
    contrast_id: str,
    contrast_family: str,
    high: StandardizedSetting,
    low: StandardizedSetting,
    continuous_variable: str,
    low_value: float,
    high_value: float,
    reference_value: float,
    error_group: str,
    reference_error_group: str,
    support_rule: str,
    support_lower: float,
    support_upper: float,
    support_n: int,
    ci_method: str = "HC1 delta-Wald",
    interval: tuple[float, float] | None = None,
) -> dict[str, Any]:
    estimate = float(high.estimate - low.estimate)
    gradient = high.gradient - low.gradient
    standard_error, delta_lower, delta_upper = _delta_interval(
        estimate, gradient, bundle.covariance,
    )
    lower, upper = interval if interval is not None else (delta_lower, delta_upper)
    return {
        "variant": bundle.variant,
        "model_id": bundle.model_id,
        "outcome": bundle.outcome,
        "contrast": _public_contrast_name(contrast_id),
        "contrast_id": contrast_id,
        "contrast_family": contrast_family,
        "high_setting": json.dumps(high.settings, sort_keys=True, ensure_ascii=False),
        "low_setting": json.dumps(low.settings, sort_keys=True, ensure_ascii=False),
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_method": ci_method,
        "continuous_variable": continuous_variable,
        "low_value": float(low_value),
        "high_value": float(high_value),
        "reference_value": float(reference_value),
        "error_group": error_group,
        "reference_error_group": reference_error_group,
        "standardization_population": bundle.variant,
        "n_standardization": int(bundle.result.nobs),
        "support_rule": support_rule,
        "support_lower": float(support_lower),
        "support_upper": float(support_upper),
        "support_n": int(support_n),
        "available": True,
        "blocker": "",
    }


def _unavailable_contrast(
    bundle: FitBundle,
    *,
    contrast_id: str,
    contrast_family: str,
    continuous_variable: str,
    error_group: str,
    reference_error_group: str,
    blocker: str,
) -> dict[str, Any]:
    return {
        "variant": bundle.variant, "model_id": bundle.model_id,
        "outcome": bundle.outcome, "contrast": _public_contrast_name(contrast_id),
        "contrast_id": contrast_id,
        "contrast_family": contrast_family, "high_setting": "",
        "low_setting": "", "estimate": np.nan, "standard_error": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan, "ci_method": "not estimated",
        "continuous_variable": continuous_variable, "low_value": np.nan,
        "high_value": np.nan, "reference_value": np.nan,
        "error_group": error_group, "reference_error_group": reference_error_group,
        "standardization_population": bundle.variant,
        "n_standardization": int(bundle.result.nobs), "support_rule": "p05-p95 intersection",
        "support_lower": np.nan, "support_upper": np.nan, "support_n": 0,
        "available": False, "blocker": blocker,
    }


def _pair_support(
    data: pd.DataFrame,
    *,
    group: str,
    reference: str,
    continuous: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    support_cfg = _config_section(config, "support")
    q_low = float(support_cfg["common_support_quantile_min"])
    q_high = float(support_cfg["common_support_quantile_max"])
    group_values = data.loc[
        data["promise_error_group"].astype("string").eq(group), continuous,
    ]
    reference_values = data.loc[
        data["promise_error_group"].astype("string").eq(reference), continuous,
    ]
    if group_values.empty or reference_values.empty:
        return {"available": False, "blocker": "missing_error_group"}
    group_range = (
        float(group_values.quantile(q_low, interpolation="linear")),
        float(group_values.quantile(q_high, interpolation="linear")),
    )
    reference_range = (
        float(reference_values.quantile(q_low, interpolation="linear")),
        float(reference_values.quantile(q_high, interpolation="linear")),
    )
    lower = max(group_range[0], reference_range[0])
    upper = min(group_range[1], reference_range[1])
    if lower > upper:
        return {
            "available": False,
            "blocker": "insufficient_common_duration_support",
            "lower": float(lower),
            "upper": float(upper),
            "group_range": group_range,
            "reference_range": reference_range,
        }
    pooled = data.loc[
        data["promise_error_group"].astype("string").isin([group, reference])
        & data[continuous].between(lower, upper, inclusive="both"),
        continuous,
    ]
    if pooled.empty:
        return {"available": False, "blocker": "empty_common_support_intersection"}
    return {
        "available": True,
        "lower": float(lower),
        "upper": float(upper),
        # Durations are integer calendar days.  ``nearest`` makes the frozen
        # pooled-median reference an observed integer day even when the two
        # central order statistics differ.
        "reference_value": float(pooled.quantile(0.5, interpolation="nearest")),
        "support_n": int(len(pooled)),
        "group_range": group_range,
        "reference_range": reference_range,
        "rule": (
            f"intersection of group-specific p{int(100*q_low):02d}-p{int(100*q_high):02d}; "
            "pooled median inside intersection"
        ),
    }


def common_support_duration(
    data: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit-facing p05--p95 duration-support intersection.

    The returned reference duration is the nearest observed pooled median
    inside the pairwise intersection, hence an integer for calendar-day input.
    Unsupported comparisons fail closed and never return a reference duration.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    group_column = next(
        (
            name for name in
            ("promise_error_group_label", "promise_error_group", "error_group")
            if name in data.columns
        ),
        None,
    )
    if group_column is None or "actual_delivery_days" not in data.columns:
        raise StatisticalDesignError(
            "Common support requires actual_delivery_days and a promise-error group column."
        )
    working = data.copy()
    working["promise_error_group"] = working[group_column].astype("string")
    support = _pair_support(
        working,
        group=str(comparison_group),
        reference=str(reference_group),
        continuous="actual_delivery_days",
        config=config,
    )
    available = bool(support.get("available"))
    return {
        "status": "supported" if available else str(
            support.get("blocker", "insufficient_common_duration_support")
        ),
        "reference_group": str(reference_group),
        "comparison_group": str(comparison_group),
        "intersection_lower": support.get("lower"),
        "intersection_upper": support.get("upper"),
        "reference_duration": int(support["reference_value"]) if available else None,
        "support_n": int(support.get("support_n", 0)),
        "reference_range": support.get("reference_range"),
        "comparison_range": support.get("group_range"),
        "rule": support.get("rule", "p05-p95 pairwise intersection"),
    }


# Short compatibility alias for review/audit callers.
common_support = common_support_duration


def _duration_bin_label(value: float, config: Mapping[str, Any]) -> str:
    cfg = _config_section(config, "actual_duration_groups")
    result = pd.cut(
        pd.Series([value]), bins=_parse_edges(cfg["edges"]), labels=cfg["labels"],
        right=bool(cfg.get("right_closed", True)), include_lowest=True,
    )
    if result.isna().any():
        raise StatisticalDesignError(f"Duration {value} cannot be mapped to a frozen bin.")
    return str(result.iloc[0])


def _model_a_outputs(
    bundle: FitBundle, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support_cfg = _config_section(config, "support")
    q_low = float(support_cfg["curve_quantile_min"])
    q_high = float(support_cfg["curve_quantile_max"])
    values = bundle.data["actual_delivery_days"]
    lower = float(values.quantile(q_low, interpolation="linear"))
    upper = float(values.quantile(q_high, interpolation="linear"))
    grid_lower = int(np.ceil(lower))
    grid_upper = int(np.floor(upper))
    if grid_lower > grid_upper:
        raise StatisticalDesignError("The frozen p05--p95 integer curve grid is empty.")
    curve_grid = np.arange(grid_lower, grid_upper + 1, dtype=np.int64)
    support_n = int(values.between(lower, upper, inclusive="both").sum())
    probability_rows: list[dict[str, Any]] = []
    for index, value in enumerate(curve_grid):
        setting = _standardized_setting(bundle, {"actual_delivery_days": int(value)})
        probability_rows.append(_probability_row(
            bundle, setting, estimand_id=f"duration_curve_{index:03d}",
            estimand_type="duration_curve", continuous_variable="actual_delivery_days",
            continuous_value=int(value), support_rule=(
                f"inclusive integer days from ceil(p{100*q_low:g}) "
                f"to floor(p{100*q_high:g})"
            ),
            support_lower=lower, support_upper=upper, support_n=support_n,
        ))
    quantile_settings: dict[float, StandardizedSetting] = {}
    for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
        value = float(values.quantile(quantile, interpolation="linear"))
        setting = _standardized_setting(bundle, {"actual_delivery_days": value})
        quantile_settings[quantile] = setting
        probability_rows.append(_probability_row(
            bundle, setting, estimand_id=f"duration_p{int(100*quantile):02d}",
            estimand_type="duration_percentile", continuous_variable="actual_delivery_days",
            continuous_value=value, support_rule="full-sample empirical percentile",
            support_lower=lower, support_upper=upper, support_n=len(values),
        ))
    contrast = _contrast_row(
        bundle, contrast_id="model_a_full_sample_p75_minus_p25",
        contrast_family="absolute_speed", high=quantile_settings[0.75],
        low=quantile_settings[0.25], continuous_variable="actual_delivery_days",
        low_value=float(values.quantile(0.25, interpolation="linear")),
        high_value=float(values.quantile(0.75, interpolation="linear")),
        reference_value=np.nan, error_group="", reference_error_group="",
        support_rule="full-sample empirical p25/p75", support_lower=lower,
        support_upper=upper, support_n=len(values),
    )
    return probability_rows, [contrast]


def _group_outputs(
    bundle: FitBundle,
    config: Mapping[str, Any],
    *,
    support_continuous: str,
    model_setting_column: str,
    setting_transform: Callable[[float], Any] = lambda value: float(value),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    error_cfg = _config_section(config, "promise_error_groups")
    labels = [str(value) for value in error_cfg["labels"]]
    reference = str(error_cfg["reference"])
    probability_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    supports: dict[str, dict[str, Any]] = {}
    for group in labels:
        support = _pair_support(
            bundle.data, group=group, reference=reference,
            continuous=support_continuous, config=config,
        )
        supports[group] = support
        if not support.get("available"):
            contrast_rows.append(_unavailable_contrast(
                bundle, contrast_id=f"error_group_rd::{group}",
                contrast_family="error_group_vs_on_date",
                continuous_variable=support_continuous, error_group=group,
                reference_error_group=reference, blocker=str(support.get("blocker")),
            ))
            continue
        value = float(support["reference_value"])
        model_value = setting_transform(value)
        group_setting = _standardized_setting(
            bundle, {model_setting_column: model_value, "promise_error_group_label": group},
        )
        reference_setting = _standardized_setting(
            bundle, {model_setting_column: model_value, "promise_error_group_label": reference},
        )
        probability_rows.append(_probability_row(
            bundle, group_setting, estimand_id=f"error_group_probability::{group}",
            estimand_type="supported_error_group_probability",
            continuous_variable=support_continuous, continuous_value=value,
            error_group=group, reference_error_group=reference,
            reference_estimate=reference_setting.estimate,
            support_rule=str(support["rule"]), support_lower=float(support["lower"]),
            support_upper=float(support["upper"]), support_n=int(support["support_n"]),
        ))
        contrast_rows.append(_contrast_row(
            bundle, contrast_id=f"error_group_rd::{group}",
            contrast_family="error_group_vs_on_date", high=group_setting,
            low=reference_setting, continuous_variable=support_continuous,
            low_value=value, high_value=value, reference_value=value,
            error_group=group, reference_error_group=reference,
            support_rule=str(support["rule"]), support_lower=float(support["lower"]),
            support_upper=float(support["upper"]), support_n=int(support["support_n"]),
        ))
    return probability_rows, contrast_rows, supports


def _formal_settings(
    bundle: FitBundle,
    config: Mapping[str, Any],
    *,
    model_setting_column: str = "actual_delivery_days",
    setting_transform: Callable[[float], Any] = lambda value: float(value),
) -> dict[str, Any]:
    error_cfg = _config_section(config, "promise_error_groups")
    reference = str(error_cfg["reference"])
    support_cfg = _config_section(config, "support")
    q_low = float(support_cfg["common_support_quantile_min"])
    q_high = float(support_cfg["common_support_quantile_max"])
    on_date = bundle.data.loc[
        bundle.data["promise_error_group"].astype("string").eq(reference),
        "actual_delivery_days",
    ]
    on_support = (
        float(on_date.quantile(q_low, interpolation="linear")),
        float(on_date.quantile(q_high, interpolation="linear")),
    )
    d25 = float(on_date.quantile(0.25, interpolation="linear"))
    d75 = float(on_date.quantile(0.75, interpolation="linear"))
    speed_supported = on_support[0] <= d25 <= d75 <= on_support[1]
    result: dict[str, Any] = {
        "reference": reference,
        "d25": d25,
        "d75": d75,
        "on_support": on_support,
        "on_support_n": int(len(on_date)),
        "speed_supported": bool(speed_supported),
    }
    if speed_supported:
        result["speed_low"] = _standardized_setting(
            bundle, {
                model_setting_column: setting_transform(d25),
                "promise_error_group_label": reference,
            },
        )
        result["speed_high"] = _standardized_setting(
            bundle, {
                model_setting_column: setting_transform(d75),
                "promise_error_group_label": reference,
            },
        )
    for key, group in (("late_4_7", "4-7 days late"), ("late_8_plus", ">=8 days late")):
        support = _pair_support(
            bundle.data, group=group, reference=reference,
            continuous="actual_delivery_days", config=config,
        )
        result[f"{key}_support"] = support
        if support.get("available"):
            value = float(support["reference_value"])
            result[f"{key}_high"] = _standardized_setting(
                bundle, {
                    model_setting_column: setting_transform(value),
                    "promise_error_group_label": group,
                },
            )
            result[f"{key}_low"] = _standardized_setting(
                bundle, {
                    model_setting_column: setting_transform(value),
                    "promise_error_group_label": reference,
                },
            )
    return result


def _formal_delta_contrasts(
    bundle: FitBundle,
    config: Mapping[str, Any],
    *,
    model_setting_column: str = "actual_delivery_days",
    setting_transform: Callable[[float], Any] = lambda value: float(value),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = _formal_settings(
        bundle, config, model_setting_column=model_setting_column,
        setting_transform=setting_transform,
    )
    reference = str(settings["reference"])
    rows: list[dict[str, Any]] = []
    if settings["speed_supported"]:
        rows.append(_contrast_row(
            bundle, contrast_id="C_speed", contrast_family="absolute_speed",
            high=settings["speed_high"], low=settings["speed_low"],
            continuous_variable="actual_delivery_days", low_value=settings["d25"],
            high_value=settings["d75"], reference_value=np.nan,
            error_group=reference, reference_error_group=reference,
            support_rule="on-date empirical p25/p75 within on-date p05-p95",
            support_lower=settings["on_support"][0], support_upper=settings["on_support"][1],
            support_n=settings["on_support_n"],
        ))
    else:
        rows.append(_unavailable_contrast(
            bundle, contrast_id="C_speed", contrast_family="absolute_speed",
            continuous_variable="actual_delivery_days", error_group=reference,
            reference_error_group=reference, blocker="unsupported_on_date_p25_p75",
        ))
    for key, contrast_id, group in (
        ("late_4_7", "C_late_4_7", "4-7 days late"),
        ("late_8_plus", "C_late_8_plus", ">=8 days late"),
    ):
        support = settings[f"{key}_support"]
        if support.get("available"):
            value = float(support["reference_value"])
            rows.append(_contrast_row(
                bundle, contrast_id=contrast_id,
                contrast_family="promise_relative", high=settings[f"{key}_high"],
                low=settings[f"{key}_low"], continuous_variable="actual_delivery_days",
                low_value=value, high_value=value, reference_value=value,
                error_group=group, reference_error_group=reference,
                support_rule=str(support["rule"]), support_lower=float(support["lower"]),
                support_upper=float(support["upper"]), support_n=int(support["support_n"]),
            ))
        else:
            rows.append(_unavailable_contrast(
                bundle, contrast_id=contrast_id, contrast_family="promise_relative",
                continuous_variable="actual_delivery_days", error_group=group,
                reference_error_group=reference, blocker=str(support.get("blocker")),
            ))
    return rows, settings


def _draw_standardized_probability(
    setting: StandardizedSetting, coefficient_draws: np.ndarray,
) -> np.ndarray:
    probabilities = expit(setting.design @ coefficient_draws.T)
    return setting.weights @ probabilities


def draw_hc1_coefficients(
    beta: Sequence[float] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw the exact frozen Cholesky-MVN coefficient sample.

    There is no repair, eigenvalue clipping, retry, or discarded-draw path:
    non-finite, asymmetric, or non-positive-definite HC1 covariance fails
    before simulation.
    """
    location = np.asarray(beta, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if location.ndim != 1:
        raise StatisticalDesignError("Coefficient location must be one-dimensional.")
    if matrix.shape != (len(location), len(location)):
        raise StatisticalDesignError("HC1 covariance dimensions do not match coefficients.")
    if not np.isfinite(location).all() or not np.isfinite(matrix).all():
        raise StatisticalDesignError("Coefficient location and HC1 covariance must be finite.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise StatisticalDesignError("HC1 covariance must be symmetric.")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise StatisticalDesignError("HC1 covariance failed the Cholesky gate.") from exc
    contrast_cfg = _config_section(config, "contrasts")
    seed = int(contrast_cfg["simulation_seed"])
    draw_count = int(contrast_cfg["simulation_draws"])
    if seed != SIMULATION_SEED or draw_count != SIMULATION_DRAWS:
        raise StatisticalDesignError(
            "Simulation seed/draw count differ from the frozen 20260824/10000 contract."
        )
    draws = np.random.default_rng(seed).multivariate_normal(
        location,
        matrix,
        size=draw_count,
        check_valid="raise",
        method=MVN_METHOD,
    )
    if draws.shape != (draw_count, len(location)) or not np.isfinite(draws).all():
        raise StatisticalDesignError("Frozen Cholesky-MVN coefficient draws are invalid.")
    audit = {
        "seed": seed,
        "requested_draws": draw_count,
        "valid_draws": draw_count,
        "discarded_draws": 0,
        "mvn_method": MVN_METHOD,
        "draw_sha256": hashlib.sha256(draws.tobytes(order="C")).hexdigest(),
    }
    return draws, audit


def _primary_simulation(
    bundle: FitBundle,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    delta_rows, settings = _formal_delta_contrasts(bundle, config)
    contrast_cfg = _config_section(config, "contrasts")
    seed = int(contrast_cfg["simulation_seed"])
    draw_count = int(contrast_cfg["simulation_draws"])
    interval_quantiles = [float(value) for value in contrast_cfg["interval_quantiles"]]
    if interval_quantiles != [0.025, 0.975]:
        raise StatisticalDesignError("Frozen MVN interval quantiles must be [0.025, 0.975].")
    coefficient_draws, coefficient_draw_audit = draw_hc1_coefficients(
        bundle.result.params.to_numpy(dtype=float), bundle.covariance, config,
    )
    if coefficient_draws.shape != (draw_count, len(bundle.result.params)):
        raise StatisticalDesignError("MVN coefficient draw shape is incorrect.")
    if not np.isfinite(coefficient_draws).all():
        raise StatisticalDesignError("MVN produced a non-finite coefficient draw.")

    draw_values: dict[str, np.ndarray] = {}
    point_values: dict[str, float] = {}
    gradients: dict[str, np.ndarray] = {}
    fixed_settings: dict[str, Any] = {
        "C_speed": {"D_low": settings["d25"], "D_high": settings["d75"], "group": settings["reference"]},
    }
    all_probability_draws: list[np.ndarray] = []
    if settings["speed_supported"]:
        high = _draw_standardized_probability(settings["speed_high"], coefficient_draws)
        low = _draw_standardized_probability(settings["speed_low"], coefficient_draws)
        all_probability_draws.extend([high, low])
        draw_values["C_speed"] = high - low
        point_values["C_speed"] = settings["speed_high"].estimate - settings["speed_low"].estimate
        gradients["C_speed"] = settings["speed_high"].gradient - settings["speed_low"].gradient
    for key, contrast_id, group in (
        ("late_4_7", "C_late_4_7", "4-7 days late"),
        ("late_8_plus", "C_late_8_plus", ">=8 days late"),
    ):
        support = settings[f"{key}_support"]
        if support.get("available"):
            high = _draw_standardized_probability(settings[f"{key}_high"], coefficient_draws)
            low = _draw_standardized_probability(settings[f"{key}_low"], coefficient_draws)
            all_probability_draws.extend([high, low])
            draw_values[contrast_id] = high - low
            point_values[contrast_id] = settings[f"{key}_high"].estimate - settings[f"{key}_low"].estimate
            gradients[contrast_id] = settings[f"{key}_high"].gradient - settings[f"{key}_low"].gradient
            fixed_settings[contrast_id] = {
                "D_reference": support["reference_value"], "group": group,
                "reference_group": settings["reference"], "support_lower": support["lower"],
                "support_upper": support["upper"], "support_n": support["support_n"],
            }
    if "C_speed" in draw_values and "C_late_4_7" in draw_values:
        draw_values["C_difference_4_7_minus_speed"] = (
            draw_values["C_late_4_7"] - draw_values["C_speed"]
        )
        point_values["C_difference_4_7_minus_speed"] = (
            point_values["C_late_4_7"] - point_values["C_speed"]
        )
        gradients["C_difference_4_7_minus_speed"] = (
            gradients["C_late_4_7"] - gradients["C_speed"]
        )
        fixed_settings["C_difference_4_7_minus_speed"] = {
            "minuend": "C_late_4_7", "subtrahend": "C_speed",
        }
    if "C_speed" in draw_values and "C_late_8_plus" in draw_values:
        draw_values["C_difference_8_plus_minus_speed"] = (
            draw_values["C_late_8_plus"] - draw_values["C_speed"]
        )
        point_values["C_difference_8_plus_minus_speed"] = (
            point_values["C_late_8_plus"] - point_values["C_speed"]
        )
        gradients["C_difference_8_plus_minus_speed"] = (
            gradients["C_late_8_plus"] - gradients["C_speed"]
        )
        fixed_settings["C_difference_8_plus_minus_speed"] = {
            "minuend": "C_late_8_plus", "subtrahend": "C_speed",
        }
    parameter_order = [str(value) for value in bundle.result.params.index]
    parameter_hash = _stable_hash(parameter_order)
    design_hash = _stable_hash(fixed_settings)
    all_finite = all(np.isfinite(values).all() for values in draw_values.values())
    if not all_finite:
        raise StatisticalDesignError("At least one MVN contrast draw is non-finite.")
    probability_min = float(min(values.min() for values in all_probability_draws))
    probability_max = float(max(values.max() for values in all_probability_draws))
    if probability_min < 0 or probability_max > 1:
        raise StatisticalDesignError("Simulated probabilities fall outside [0, 1].")
    comparison_rows: list[dict[str, Any]] = []
    interval_by_id: dict[str, tuple[float, float]] = {}
    for contrast_id, values in draw_values.items():
        lower, upper = np.quantile(
            values, interval_quantiles, method=MVN_QUANTILE_METHOD,
        )
        interval_by_id[contrast_id] = (float(lower), float(upper))
        standard_error, _, _ = _delta_interval(
            point_values[contrast_id], gradients[contrast_id], bundle.covariance,
        )
        comparison_rows.append({
            "variant": bundle.variant,
            "model_id": bundle.model_id,
            "outcome": bundle.outcome,
            "contrast_id": contrast_id,
            "estimate": float(point_values[contrast_id]),
            "standard_error": standard_error,
            "q025": float(lower),
            "q975": float(upper),
            "interval_excludes_zero": bool(lower > 0 or upper < 0),
            "seed": seed,
            "n_draws": draw_count,
            "n_valid": draw_count,
            "mvn_method": MVN_METHOD,
            "quantile_method": MVN_QUANTILE_METHOD,
            "covariance": COVARIANCE_TYPE,
            "parameter_order_hash": parameter_hash,
            "design_settings_hash": design_hash,
            "fixed_settings_json": json.dumps(fixed_settings[contrast_id], sort_keys=True),
            "draws_all_finite": True,
            "probability_min": probability_min,
            "probability_max": probability_max,
        })
    simulated_contrast_rows: list[dict[str, Any]] = []
    for row in delta_rows:
        copied = dict(row)
        if row["available"] and row["contrast_id"] in interval_by_id:
            copied["ci_lower"], copied["ci_upper"] = interval_by_id[row["contrast_id"]]
            copied["ci_method"] = "HC1 coefficient MVN percentile"
        simulated_contrast_rows.append(copied)
    for contrast_id in ("C_difference_4_7_minus_speed", "C_difference_8_plus_minus_speed"):
        if contrast_id not in interval_by_id:
            continue
        standard_error, _, _ = _delta_interval(
            point_values[contrast_id], gradients[contrast_id], bundle.covariance,
        )
        simulated_contrast_rows.append({
            "variant": bundle.variant, "model_id": bundle.model_id,
            "outcome": bundle.outcome,
            "contrast": _public_contrast_name(contrast_id),
            "contrast_id": contrast_id,
            "contrast_family": "difference_of_contrasts", "high_setting": "",
            "low_setting": "", "estimate": float(point_values[contrast_id]),
            "standard_error": standard_error,
            "ci_lower": interval_by_id[contrast_id][0],
            "ci_upper": interval_by_id[contrast_id][1],
            "ci_method": "HC1 coefficient MVN percentile",
            "continuous_variable": "actual_delivery_days", "low_value": np.nan,
            "high_value": np.nan, "reference_value": np.nan,
            "error_group": "", "reference_error_group": settings["reference"],
            "standardization_population": bundle.variant,
            "n_standardization": int(bundle.result.nobs),
            "support_rule": "inherits component contrast support",
            "support_lower": np.nan, "support_upper": np.nan, "support_n": np.nan,
            "available": True, "blocker": "",
        })
    draw_audit = {
        "seed": seed, "requested_draws": draw_count, "created_draws": draw_count,
        "valid_draws": draw_count, "mvn_method": MVN_METHOD,
        "quantile_method": MVN_QUANTILE_METHOD, "draws_all_finite": True,
        "probability_min": probability_min, "probability_max": probability_max,
        "parameter_order_hash": parameter_hash, "design_settings_hash": design_hash,
        "coefficient_draw_shape": list(coefficient_draws.shape),
        "discarded_draws": coefficient_draw_audit["discarded_draws"],
        "draw_sha256": coefficient_draw_audit["draw_sha256"],
    }
    return simulated_contrast_rows, comparison_rows, draw_audit


def _group_coefficients(bundle: FitBundle, labels: Sequence[str], reference: str) -> dict[str, float]:
    coefficients = {reference: 0.0}
    for label in labels:
        if label == reference:
            continue
        suffix = f"[T.{label}]"
        matches = [
            str(term) for term in bundle.result.params.index if str(term).endswith(suffix)
        ]
        if len(matches) != 1:
            raise StatisticalDesignError(f"Cannot identify error-group coefficient for {label!r}.")
        coefficients[label] = float(bundle.result.params[matches[0]])
    return coefficients


def _pack_sensitivity(
    variant: str,
    model_rows: Sequence[dict[str, Any]],
    wald_rows: Sequence[dict[str, Any]],
    contrast_rows: Sequence[dict[str, Any]],
    *,
    n_orders: int,
    note: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in model_rows:
        rows.append({
            "variant": variant, "analysis_sample": variant,
            "record_type": "coefficient",
            "model_id": row["model_id"], "outcome": row["outcome"],
            "record_id": row["term"], "term": row["term"],
            "group": "", "n_orders": n_orders,
            "estimate": row["coefficient_log_odds"],
            "standard_error": row["standard_error_hc1"],
            "ci_lower": row["coefficient_log_odds"] - NORMAL_975 * row["standard_error_hc1"],
            "ci_upper": row["coefficient_log_odds"] + NORMAL_975 * row["standard_error_hc1"],
            "p_value": row["p_value"], "odds_ratio": row["odds_ratio"],
            "odds_ratio_ci_lower": row["odds_ratio_ci_lower"],
            "odds_ratio_ci_upper": row["odds_ratio_ci_upper"],
            "direction_positive": bool(row["coefficient_log_odds"] > 0),
            "support_rule": "", "support_lower": np.nan, "support_upper": np.nan,
            "support_n": np.nan, "notes": note,
        })
    for row in wald_rows:
        rows.append({
            "variant": variant, "analysis_sample": variant,
            "record_type": "wald", "model_id": row["model_id"],
            "outcome": row["outcome"], "record_id": row["block"], "group": "",
            "term": row["term_name"],
            "n_orders": n_orders, "estimate": row["wald_chi2"],
            "standard_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
            "p_value": row["p_value"], "odds_ratio": np.nan,
            "odds_ratio_ci_lower": np.nan, "odds_ratio_ci_upper": np.nan,
            "direction_positive": np.nan, "support_rule": "",
            "support_lower": np.nan, "support_upper": np.nan, "support_n": np.nan,
            "notes": note,
        })
    for row in contrast_rows:
        rows.append({
            "variant": variant, "analysis_sample": variant,
            "record_type": "contrast", "model_id": row["model_id"],
            "outcome": row["outcome"], "record_id": row["contrast_id"],
            "term": row["contrast_id"],
            "group": row["error_group"], "n_orders": n_orders,
            "estimate": row["estimate"], "standard_error": row["standard_error"],
            "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"],
            "p_value": np.nan, "odds_ratio": np.nan,
            "odds_ratio_ci_lower": np.nan, "odds_ratio_ci_upper": np.nan,
            "direction_positive": bool(row["estimate"] > 0) if row["available"] else np.nan,
            "support_rule": row["support_rule"], "support_lower": row["support_lower"],
            "support_upper": row["support_upper"], "support_n": row["support_n"],
            "notes": row["blocker"] or note,
        })
    return _as_frame(rows, SENSITIVITY_COLUMNS)


def _lookup_row(
    rows: Sequence[dict[str, Any]],
    *,
    variant: str,
    model_id: str,
    key: str,
    value: str,
) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row.get("variant") == variant and row.get("model_id") == model_id
        and row.get(key) == value
    ]
    if len(matches) != 1:
        raise StatisticalDesignError(
            f"Expected one {variant}/{model_id} row where {key}={value!r}; found {len(matches)}."
        )
    return matches[0]


def assign_extension_decision(
    duration_supported: bool,
    promise_relative_supported: bool,
    material_sensitivity_conflict: bool,
) -> str:
    """Apply the frozen, mechanical four-label extension decision rule."""
    if bool(material_sensitivity_conflict):
        return "INCONCLUSIVE_RQ1_EXTENSION"
    if bool(duration_supported) and bool(promise_relative_supported):
        return "EXPAND_RQ1_TO_SPEED_AND_RELIABILITY"
    if not bool(duration_supported) and bool(promise_relative_supported):
        return "RETAIN_SIGNED_ERROR_ONLY_RQ1"
    if bool(duration_supported) and not bool(promise_relative_supported):
        return "ACTUAL_DURATION_ASSOCIATION_WITHOUT_INCREMENTAL_PROMISE_ERROR"
    return "INCONCLUSIVE_RQ1_EXTENSION"


def _decision(
    *,
    bundles: Mapping[str, FitBundle],
    wald_rows: Sequence[dict[str, Any]],
    contrast_rows: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_a_wald = _lookup_row(
        wald_rows, variant="primary", model_id="A",
        key="block", value="duration_spline",
    )
    primary_b_wald = _lookup_row(
        wald_rows, variant="primary", model_id="B",
        key="block", value="error_group",
    )
    comparison_by_id = {
        row["contrast_id"]: row for row in comparison_rows
        if row["variant"] == "primary" and row["model_id"] == "B"
    }
    speed = comparison_by_id.get("C_speed")
    required_comparisons_present = speed is not None
    speed_positive = bool(speed and speed["estimate"] > 0)
    speed_ci_positive = bool(speed and speed["q025"] > 0)

    def contrast(variant: str, model_id: str, contrast_id: str) -> dict[str, Any]:
        return _lookup_row(
            contrast_rows, variant=variant, model_id=model_id,
            key="contrast_id", value=contrast_id,
        )

    low3_speed = contrast("low_review_3", "B", "C_speed")
    timing_speed = contrast("post_delivery_reviews", "B", "C_speed")
    actual_components = {
        "model_a_duration_wald_p_lt_0_05": bool(primary_a_wald["p_value"] < 0.05),
        "primary_speed_contrast_positive": speed_positive,
        "primary_speed_interval_excludes_zero_positive": speed_ci_positive,
        "low_review_3_speed_direction_positive": bool(low3_speed["available"] and low3_speed["estimate"] > 0),
        "post_delivery_speed_direction_positive": bool(timing_speed["available"] and timing_speed["estimate"] > 0),
    }
    actual_supported = all(actual_components.values())

    required_late_groups = ["2-3 days late", "4-7 days late", ">=8 days late"]
    primary_rd_rows = {
        group: contrast(
            "primary", "B", f"error_group_rd::{group}",
        )
        for group in required_late_groups
    }
    primary_rds_positive_significant = all(
        row["available"] and row["estimate"] > 0 and row["ci_lower"] > 0
        for row in primary_rd_rows.values()
    )
    error_cfg = _config_section(config, "promise_error_groups")
    labels = [str(value) for value in error_cfg["labels"]]
    reference = str(error_cfg["reference"])
    coefficients = _group_coefficients(
        bundles["primary_B"], labels=labels, reference=reference,
    )
    ordered_groups = ["1 day late", "2-3 days late", "4-7 days late", ">=8 days late"]
    ordered_values = [coefficients[group] for group in ordered_groups]
    monotone = all(left <= right for left, right in zip(ordered_values, ordered_values[1:]))
    sensitivity_directions: dict[str, bool] = {}
    for variant in ("low_review_3", "post_delivery_reviews", "fixed_duration_bins"):
        for contrast_id in ("C_late_4_7", "C_late_8_plus"):
            row = contrast(variant, "B", contrast_id)
            sensitivity_directions[f"{variant}_{contrast_id}_positive"] = bool(
                row["available"] and row["estimate"] > 0
            )
    promise_components = {
        "model_b_error_group_wald_p_lt_0_05": bool(primary_b_wald["p_value"] < 0.05),
        "required_late_rds_positive_with_ci_above_zero": primary_rds_positive_significant,
        "late_group_log_odds_point_estimates_non_decreasing": monotone,
        **sensitivity_directions,
    }
    promise_supported = all(promise_components.values())
    support_blockers = [
        row["blocker"] for row in contrast_rows
        if row["contrast_id"] in {"C_speed", "C_late_4_7", "C_late_8_plus"}
        and not row["available"]
    ]
    inconclusive_override = bool(support_blockers) or not required_comparisons_present
    label = assign_extension_decision(
        duration_supported=actual_supported,
        promise_relative_supported=promise_supported,
        material_sensitivity_conflict=inconclusive_override,
    )
    allowed = {
        "EXPAND_RQ1_TO_SPEED_AND_RELIABILITY": (
            "How are actual delivery duration and performance relative to the promised date "
            "associated with observed order reviews?"
        ),
        "RETAIN_SIGNED_ERROR_ONLY_RQ1": (
            "Retain the current narrowed signed-promise-error RQ1 wording."
        ),
        "ACTUAL_DURATION_ASSOCIATION_WITHOUT_INCREMENTAL_PROMISE_ERROR": (
            "Actual duration is associated with reviews, but the signed-error headline requires qualification."
        ),
        "INCONCLUSIVE_RQ1_EXTENSION": (
            "The supplementary analysis does not support expanding the current RQ1 wording."
        ),
    }
    return {
        "label": label,
        "actual_duration_association_supported": bool(actual_supported),
        "promise_relative_association_beyond_duration_supported": bool(promise_supported),
        "actual_duration_rule_components": actual_components,
        "promise_relative_rule_components": promise_components,
        "late_group_log_odds": {group: coefficients[group] for group in ordered_groups},
        "support_blockers": sorted(set(support_blockers)),
        "inconclusive_override": inconclusive_override,
        "future_rq_wording_recommendation": allowed[label],
        "causal_claim_authorised": False,
        "governance_update_authorised": False,
    }


def run_statistical_analysis(
    reviewed: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Fit all frozen RQ1 V1 models and return deterministic reporting objects.

    Required reviewed-frame fields (aliases accepted where noted) are one row
    per selected usable review: order ID, score, D, P, E, purchase month (or
    purchase timestamp), selected review-answer timestamp, and recorded actual
    delivery timestamp.  No object returned by this function mutates ``reviewed``.
    """
    data = _prepare_reviewed(reviewed, config)
    model_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    wald_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    month_diagnostics: dict[str, Any] = {}
    bundles: dict[str, FitBundle] = {}

    def register(bundle: FitBundle) -> None:
        model_rows.extend(_model_rows(bundle))
        covariance_rows.extend(_covariance_rows(bundle))
        diagnostic = dict(bundle.diagnostic)
        # Primary audit identifiers are exactly A/B/C.  Sensitivity fits retain
        # traceable identifiers without masquerading as additional primaries.
        if bundle.variant != "primary":
            diagnostic["model_id"] = f"{bundle.variant}:{bundle.model_id}"
        fit_diagnostics.append(diagnostic)

    primary, primary_month = _with_month_adjustment(data, config)
    month_diagnostics["primary"] = primary_month
    primary_a = _fit(
        primary, variant="primary", model_id="A",
        outcome="low_review_2", kind="A", config=config, month_audit=primary_month,
    )
    primary_b = _fit(
        primary, variant="primary", model_id="B",
        outcome="low_review_2", kind="B", config=config, month_audit=primary_month,
    )
    primary_c = _fit(
        primary, variant="primary", model_id="C",
        outcome="low_review_2", kind="C", config=config, month_audit=primary_month,
    )
    bundles.update({"primary_A": primary_a, "primary_B": primary_b, "primary_C": primary_c})
    for bundle in (primary_a, primary_b, primary_c):
        register(bundle)
    wald_rows.extend([
        _wald(primary_a, "duration_spline"),
        _wald(primary_b, "error_group"),
        _wald(primary_c, "error_group"),
    ])
    a_probabilities, a_contrasts = _model_a_outputs(primary_a, config)
    probability_rows.extend(a_probabilities)
    contrast_rows.extend(a_contrasts)
    b_probabilities, b_contrasts, b_supports = _group_outputs(
        primary_b, config, support_continuous="actual_delivery_days",
        model_setting_column="actual_delivery_days",
    )
    probability_rows.extend(b_probabilities)
    contrast_rows.extend(b_contrasts)
    c_probabilities, c_contrasts, c_supports = _group_outputs(
        primary_c, config, support_continuous="promised_lead_days",
        model_setting_column="promised_lead_days",
    )
    probability_rows.extend(c_probabilities)
    contrast_rows.extend(c_contrasts)
    simulated_contrasts, comparison, draw_audit = _primary_simulation(primary_b, config)
    contrast_rows.extend(simulated_contrasts)
    comparison_rows.extend(comparison)

    sensitivity_tables: dict[str, pd.DataFrame] = {}
    for variant, outcome, subset, note in (
        (
            "low_review_3", "low_review_3", data,
            "Alternative observed low-review threshold: selected score <=3.",
        ),
        (
            "post_delivery_reviews", "low_review_2",
            data.loc[data["_review_at_or_after_delivery"].astype(bool)].copy(),
            "Selected reviews at or after recorded customer-delivery timestamp.",
        ),
    ):
        adjusted, month_audit = _with_month_adjustment(subset, config)
        month_diagnostics[variant] = month_audit
        bundle_a = _fit(
            adjusted, variant=variant, model_id="A", outcome=outcome,
            kind="A", config=config, month_audit=month_audit,
        )
        bundle_b = _fit(
            adjusted, variant=variant, model_id="B", outcome=outcome,
            kind="B", config=config, month_audit=month_audit,
        )
        bundles[f"{variant}_A"] = bundle_a
        bundles[f"{variant}_B"] = bundle_b
        local_models: list[dict[str, Any]] = []
        local_wald = [_wald(bundle_a, "duration_spline"), _wald(bundle_b, "error_group")]
        local_probabilities, local_group_contrasts, _ = _group_outputs(
            bundle_b, config, support_continuous="actual_delivery_days",
            model_setting_column="actual_delivery_days",
        )
        local_formal_contrasts, _ = _formal_delta_contrasts(bundle_b, config)
        local_contrasts = [*local_group_contrasts, *local_formal_contrasts]
        for bundle in (bundle_a, bundle_b):
            rows = _model_rows(bundle)
            local_models.extend(rows)
            register(bundle)
        wald_rows.extend(local_wald)
        probability_rows.extend(local_probabilities)
        contrast_rows.extend(local_contrasts)
        sensitivity_tables[variant] = _pack_sensitivity(
            variant, local_models, local_wald, local_contrasts,
            n_orders=len(adjusted), note=note,
        )

    fixed, fixed_month = _with_month_adjustment(data, config)
    month_diagnostics["fixed_duration_bins"] = fixed_month
    fixed_b = _fit(
        fixed, variant="fixed_duration_bins", model_id="B",
        outcome="low_review_2", kind="BIN_B", config=config, month_audit=fixed_month,
    )
    bundles["fixed_duration_bins_B"] = fixed_b
    fixed_models = _model_rows(fixed_b)
    register(fixed_b)
    fixed_wald = [_wald(fixed_b, "duration_bin"), _wald(fixed_b, "error_group")]
    wald_rows.extend(fixed_wald)
    fixed_probabilities, fixed_group_contrasts, _ = _group_outputs(
        fixed_b, config, support_continuous="actual_delivery_days",
        model_setting_column="actual_duration_group",
        setting_transform=lambda value: _duration_bin_label(value, config),
    )
    fixed_formal_contrasts, _ = _formal_delta_contrasts(
        fixed_b, config, model_setting_column="actual_duration_group",
        setting_transform=lambda value: _duration_bin_label(value, config),
    )
    fixed_contrasts = [*fixed_group_contrasts, *fixed_formal_contrasts]
    probability_rows.extend(fixed_probabilities)
    contrast_rows.extend(fixed_contrasts)
    sensitivity_tables["fixed_duration_bins"] = _pack_sensitivity(
        "fixed_duration_bins", fixed_models, fixed_wald, fixed_contrasts,
        n_orders=len(fixed), note="Fixed pre-specified actual-duration bins replace the spline.",
    )

    decision = _decision(
        bundles=bundles, wald_rows=wald_rows, contrast_rows=contrast_rows,
        comparison_rows=comparison_rows, config=config,
    )
    sample_diagnostics = {
        "sample_reviewed_orders": int(len(data)),
        "sample_unique_order_ids": int(data["order_id"].nunique()),
        "sample_low_review_2_orders": int(data["low_review_2"].sum()),
        "sample_low_review_3_orders": int(data["low_review_3"].sum()),
        "sample_post_delivery_review_orders": int(
            data["_review_at_or_after_delivery"].astype(bool).sum()
        ),
        "sample_excluded_pre_delivery_reviews": int(
            (~data["_review_at_or_after_delivery"].astype(bool)).sum()
        ),
        "date_identity_failures": 0,
        "actual_delivery_min": int(data["actual_delivery_days"].min()),
        "actual_delivery_max": int(data["actual_delivery_days"].max()),
        "promised_lead_min": int(data["promised_lead_days"].min()),
        "promised_lead_max": int(data["promised_lead_days"].max()),
        "simulation_seed": int(draw_audit["seed"]),
        "simulation_draws": int(draw_audit["valid_draws"]),
        "simulation_draw_sha256": str(draw_audit["draw_sha256"]),
        "mvn_method": MVN_METHOD,
    }
    diagnostics_records: list[dict[str, Any]] = []
    for diagnostic in fit_diagnostics:
        record = {**diagnostic, **sample_diagnostics}
        variant = str(record["variant"])
        record["month_pooling_json"] = json.dumps(
            month_diagnostics[variant], sort_keys=True, ensure_ascii=False,
        )
        if variant == "primary" and record["model_id"] == "B":
            record["common_support_json"] = json.dumps(
                b_supports, sort_keys=True, ensure_ascii=False,
            )
        elif variant == "primary" and record["model_id"] == "C":
            record["common_support_json"] = json.dumps(
                c_supports, sort_keys=True, ensure_ascii=False,
            )
        else:
            record["common_support_json"] = ""
        diagnostics_records.append(record)
    diagnostics = pd.DataFrame.from_records(diagnostics_records)
    return {
        "model_results": _as_frame(model_rows, MODEL_RESULT_COLUMNS),
        "covariance": _as_frame(covariance_rows, COVARIANCE_COLUMNS),
        "wald": _as_frame(wald_rows, WALD_COLUMNS),
        "probabilities": _as_frame(probability_rows, PROBABILITY_COLUMNS),
        "contrasts": _as_frame(contrast_rows, CONTRAST_COLUMNS),
        "comparison": _as_frame(comparison_rows, COMPARISON_COLUMNS),
        "low_review_3_sensitivity": sensitivity_tables["low_review_3"],
        "review_timing_sensitivity": sensitivity_tables["post_delivery_reviews"],
        "duration_bin_sensitivity": sensitivity_tables["fixed_duration_bins"],
        "diagnostics": diagnostics,
        "decision": decision,
    }


__all__ = [
    "COVARIANCE_COLUMNS", "COVARIANCE_TYPE", "COMPARISON_COLUMNS",
    "CONTRAST_COLUMNS", "MODEL_FORMULAS", "MODEL_RESULT_COLUMNS", "MVN_METHOD",
    "PROBABILITY_COLUMNS", "SENSITIVITY_COLUMNS", "SPLINE_CONSTRAINT",
    "SIMULATION_DRAWS", "SIMULATION_SEED", "SPLINE_DF",
    "StatisticalDesignError", "WALD_COLUMNS", "assign_extension_decision",
    "common_support", "common_support_duration", "draw_hc1_coefficients",
    "model_formula", "run_statistical_analysis",
]
