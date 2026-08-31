"""Independent persisted-output validation for the direct extension.

This validator deliberately works from the frozen config and persisted tables,
not from in-memory experiment objects.  It is run after modelling/reporting and
before the final integrity receipt is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis" / "direct_promise_profile_extension_v1"
CONFIG = OUT / "DIRECT_FROZEN_CONFIG.json"
FLOAT_TOLERANCE = 1e-10

REQUIRED_OUTPUTS = (
    "RUN_MANIFEST.json",
    "README.md",
    "EXACT_FEATURE_MANIFEST.md",
    "MODEL_SELECTION.csv",
    "DIRECT_BREACH_MONTHLY.csv",
    "DIRECT_BREACH_POOLED.csv",
    "DIRECT_BREACH_CALIBRATION.csv",
    "DIRECT_BREACH_SUPPORT_STRATA.csv",
    "DIRECT_SEVERITY_MONTHLY.csv",
    "DIRECT_SEVERITY_POOLED.csv",
    "DIRECT_SEVERITY_COVERAGE.csv",
    "DIRECT_TERMINAL.csv",
    "DIRECT_VS_CURRENT_CONTEXT_ROBUSTNESS.csv",
    "RESULT_SUMMARY.md",
    "RESULT_SUMMARY_ZH.md",
    "FIGURE_DATA.csv",
    "HASH_INVENTORY.txt",
    "DIRECT_MODEL_MANIFESTS.csv",
    "EVIDENCE_LABELS.csv",
)

EXPECTED_MONTH_COUNTS = {
    "2018-01": (7069, 403),
    "2018-02": (6555, 926),
    "2018-03": (7003, 1328),
    "2018-04": (6798, 306),
    "2018-05": (6749, 443),
    "2018-06": (6096, 71),
}
EXPECTED_POOLED = (40270, 3477)
EXPECTED_TERMINAL = (12507, 601)

FORBIDDEN_FEATURE_TOKENS = {
    "n_items",
    "n_unique_products",
    "n_unique_sellers",
    "multi_item",
    "multi_product",
    "total_price",
    "total_freight_value",
    "freight_to_price_ratio",
    "distance_km",
    "customer_state",
    "main_seller_state",
    "main_product_category",
    "purchase_month_num",
    "purchase_weekday",
    "purchase_hour",
    "known_event_indicator",
    "review_score",
    "order_delivered_customer_date",
    "late_delivery",
    "positive_late_days",
}

PROFILE_SUFFIXES = (
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
SPECIFICATION_BLOCKS = {
    "DP0": (),
    "DPS": ("S1", "S2"),
    "DPG": ("R1", "R2"),
    "DPB": ("S1", "S2", "R1", "R2"),
    "DQ0": (),
    "DQS": ("S1", "S2"),
    "DQG": ("R1", "R2"),
    "DQB": ("S1", "S2", "R1", "R2"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.is_file():
        raise AssertionError(f"missing required output: {name}")
    return pd.read_csv(path, low_memory=False)


def _first_column(frame: pd.DataFrame, names: Iterable[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise AssertionError(f"none of the required columns exist: {list(names)}")
    return None


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=FLOAT_TOLERANCE, equal_nan=False):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def _validate_sources(config: dict[str, object], checks: list[dict[str, object]]) -> None:
    for key, value in dict(config["sources"]).items():
        path_text, expected = value
        path = ROOT / str(path_text)
        if not path.is_file():
            raise AssertionError(f"frozen source missing: {key}={path_text}")
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(f"frozen source hash drift: {key}: {actual} != {expected}")
    checks.append({"check": "all_frozen_source_hashes", "status": "PASS", "count": len(config["sources"])})


def _validate_features(checks: list[dict[str, object]]) -> None:
    manifest = _read_csv("DIRECT_MODEL_MANIFESTS.csv")
    feature_column = _first_column(
        manifest,
        ("features_json", "numeric_features_json", "feature_names_json", "numeric_features"),
    )
    specification_column = _first_column(manifest, ("specification", "model_id"))
    representation_column = _first_column(manifest, ("representation",))
    categorical_column = _first_column(
        manifest,
        ("categorical_features_json", "categorical_features"),
        required=False,
    )
    for row in manifest.itertuples(index=False):
        raw = getattr(row, feature_column)
        try:
            features = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception as exc:
            raise AssertionError(f"invalid feature JSON in model manifest: {raw!r}") from exc
        features = list(map(str, features))
        if not features or features[0] != "promised_delivery_days":
            raise AssertionError("every direct model must start with promised_delivery_days")
        forbidden = sorted(set(features) & FORBIDDEN_FEATURE_TOKENS)
        if forbidden:
            raise AssertionError(f"forbidden direct feature(s): {forbidden}")
        unexpected = [
            feature for feature in features[1:]
            if not any(feature.startswith(f"{prefix}_") for prefix in ("S1", "S2", "R1", "R2"))
        ]
        if unexpected:
            raise AssertionError(f"non-profile feature(s) in direct ladder: {unexpected}")
        spec = str(getattr(row, specification_column))
        representation = str(getattr(row, representation_column))
        if spec not in SPECIFICATION_BLOCKS:
            raise AssertionError(f"unexpected direct specification in manifest: {spec}")
        if not SPECIFICATION_BLOCKS[spec]:
            expected_features = ["promised_delivery_days"]
        else:
            suffixes = {
                "full": PROFILE_SUFFIXES,
                "score_only": ("score",),
                "metadata_only": PROFILE_METADATA_SUFFIXES,
            }.get(representation)
            if suffixes is None:
                raise AssertionError(
                    f"unexpected representation for {spec}: {representation}"
                )
            expected_features = ["promised_delivery_days"] + [
                f"{block}_{suffix}"
                for block in SPECIFICATION_BLOCKS[spec]
                for suffix in suffixes
            ]
        if features != expected_features:
            raise AssertionError(
                f"{spec}/{representation} feature order differs from the exact freeze"
            )
        if categorical_column:
            raw_categories = getattr(row, categorical_column)
            categories = json.loads(raw_categories) if isinstance(raw_categories, str) else list(raw_categories)
            if categories:
                raise AssertionError(f"{spec}/{representation} contains categorical predictors")
    checks.append({"check": "exact_direct_feature_contract", "status": "PASS", "rows": len(manifest)})


def _validate_monthly_samples(checks: list[dict[str, object]]) -> None:
    breach = _read_csv("DIRECT_BREACH_MONTHLY.csv")
    period = breach[_first_column(breach, ("period",))].astype(str)
    breach = breach.loc[period.eq("later")].copy()
    cohort_col = _first_column(breach, ("cohort", "cohort_month"))
    n_col = _first_column(breach, ("n_orders", "n_obs"))
    e_col = _first_column(breach, ("n_events", "n_breaches"))
    hash_col = _first_column(breach, ("order_id_sha256", "paired_order_id_sha256"))
    family_col = _first_column(breach, ("family", "model_family"))
    rep_col = _first_column(breach, ("representation",), required=False)
    variant_col = _first_column(breach, ("probability_variant", "probability_type"), required=False)
    for cohort, (expected_n, expected_events) in EXPECTED_MONTH_COUNTS.items():
        rows = breach.loc[breach[cohort_col].astype(str).eq(cohort)]
        if rows.empty:
            raise AssertionError(f"missing later cohort {cohort}")
        if set(pd.to_numeric(rows[n_col], errors="raise").astype(int)) != {expected_n}:
            raise AssertionError(f"breach sample count mismatch for {cohort}")
        if set(pd.to_numeric(rows[e_col], errors="raise").astype(int)) != {expected_events}:
            raise AssertionError(f"breach event count mismatch for {cohort}")
        grouping = [family_col]
        if rep_col:
            grouping.append(rep_col)
        if variant_col:
            grouping.append(variant_col)
        for _, group in rows.groupby(grouping, dropna=False, observed=True):
            if group[hash_col].dropna().astype(str).nunique() != 1:
                raise AssertionError(f"non-identical paired order IDs in {cohort}")
    if set(breach[cohort_col].astype(str).unique()) != set(EXPECTED_MONTH_COUNTS):
        raise AssertionError("later breach output contains missing or extra monthly cohorts")

    severity = _read_csv("DIRECT_SEVERITY_MONTHLY.csv")
    severity = severity.loc[severity[_first_column(severity, ("period",))].astype(str).eq("later")]
    cohort_col = _first_column(severity, ("cohort", "cohort_month"))
    n_col = _first_column(severity, ("n_orders", "n_obs"))
    hash_col = _first_column(severity, ("order_id_sha256",))
    family_col = _first_column(severity, ("family", "model_family"))
    quantile_col = _first_column(severity, ("quantile",))
    rep_col = _first_column(severity, ("representation",), required=False)
    for cohort, (_, expected_breaches) in EXPECTED_MONTH_COUNTS.items():
        rows = severity.loc[severity[cohort_col].astype(str).eq(cohort)]
        if rows.empty or set(pd.to_numeric(rows[n_col], errors="raise").astype(int)) != {expected_breaches}:
            raise AssertionError(f"severity breached-order sample mismatch for {cohort}")
        grouping = [family_col, quantile_col]
        if rep_col:
            grouping.append(rep_col)
        for _, group in rows.groupby(grouping, dropna=False, observed=True):
            if group[hash_col].dropna().astype(str).nunique() != 1:
                raise AssertionError(f"non-identical severity paired IDs in {cohort}")
    checks.append({"check": "exact_monthly_cohorts_and_paired_ids", "status": "PASS", "months": 6})


def _validate_pooled_terminal(checks: list[dict[str, object]]) -> None:
    pooled = _read_csv("DIRECT_BREACH_POOLED.csv")
    n_col = _first_column(pooled, ("n_orders", "n_obs"))
    e_col = _first_column(pooled, ("n_events", "n_breaches"))
    if set(pd.to_numeric(pooled[n_col], errors="raise").astype(int)) != {EXPECTED_POOLED[0]}:
        raise AssertionError("pooled breach order count mismatch")
    if set(pd.to_numeric(pooled[e_col], errors="raise").astype(int)) != {EXPECTED_POOLED[1]}:
        raise AssertionError("pooled breach event count mismatch")
    terminal = _read_csv("DIRECT_TERMINAL.csv")
    n_col = _first_column(terminal, ("n_orders", "n_obs"))
    task_col = _first_column(terminal, ("task",))
    breach_rows = terminal.loc[terminal[task_col].astype(str).str.contains("breach", case=False)]
    if breach_rows.empty or set(pd.to_numeric(breach_rows[n_col], errors="raise").astype(int)) != {EXPECTED_TERMINAL[0]}:
        raise AssertionError("terminal breach order count mismatch")
    event_col = _first_column(breach_rows, ("n_events", "n_breaches"), required=False)
    if event_col and set(pd.to_numeric(breach_rows[event_col], errors="raise").astype(int)) != {EXPECTED_TERMINAL[1]}:
        raise AssertionError("terminal breach event count mismatch")
    checks.append({"check": "pooled_and_terminal_isolation", "status": "PASS"})


def _breach_label(row: pd.Series) -> str:
    delta_ll = float(row[_first_column(row.to_frame().T, ("median_delta_log_loss",))])
    delta_bs = float(row[_first_column(row.to_frame().T, ("median_delta_brier", "median_delta_brier_score"))])
    count = int(row[_first_column(
        row.to_frame().T,
        ("months_both_improved", "both_improved_month_count", "favourable_months"),
    )])
    guards = []
    for candidates in (
        ("high_support_no_material_reversal", "no_high_support_reversal", "high_support_guard"),
        ("calibration_guard", "calibration_not_systematically_worse"),
        ("score_contribution_guard", "score_contributes", "score_not_metadata_only_guard"),
    ):
        column = next(
            (
                name
                for name in candidates
                if name in row.index and pd.notna(row[name])
            ),
            None,
        )
        if column is None:
            raise AssertionError(f"breach evidence row lacks guard: {candidates}")
        guards.append(_as_bool(row[column]))
    if delta_ll >= 0 or delta_bs >= 0 or count <= 2:
        return "Not-supported"
    if count == 3 or not all(guards):
        return "Mixed"
    return "Supported"


def _severity_label(row: pd.Series) -> str:
    skill = float(row[_first_column(row.to_frame().T, ("median_skill", "median_pinball_skill"))])
    count = int(row[_first_column(
        row.to_frame().T,
        ("favourable_months", "favourable_month_count", "months_nonnegative_skill"),
    )])
    support_col = _first_column(row.to_frame().T, ("high_support_guard", "support_ge20_gain_present"))
    support_available_col = _first_column(
        row.to_frame().T,
        ("high_support_guard_available", "support_guard_available"),
        required=False,
    )
    coverage_col = _first_column(row.to_frame().T, ("coverage_guard", "coverage_not_materially_worse"), required=False)
    q = float(row["quantile"])
    if skill <= 0 or count < 4:
        return "Not-supported"
    support_available = (
        _as_bool(row[support_available_col]) if support_available_col else True
    )
    if support_available and not _as_bool(row[support_col]):
        return "Not-supported"
    if not support_available:
        return "Mixed"
    if q == 0.9 and (coverage_col is None or not _as_bool(row[coverage_col])):
        return "Mixed"
    return "Supported"


def _validate_labels(checks: list[dict[str, object]]) -> None:
    labels = _read_csv("EVIDENCE_LABELS.csv")
    task_col = _first_column(labels, ("task",))
    status_col = _first_column(labels, ("evidence_status", "evidence_label"))
    representation_col = _first_column(labels, ("representation",), required=False)
    primary = labels.copy()
    if representation_col:
        primary = primary.loc[primary[representation_col].astype(str).isin({"full", "primary_full"})]
    for _, row in primary.iterrows():
        task = str(row[task_col]).lower()
        expected = _breach_label(row) if "breach" in task else _severity_label(row)
        actual = str(row[status_col])
        if actual != expected:
            identity = {
                key: row.get(key)
                for key in ("task", "family", "comparison", "quantile")
                if key in row.index
            }
            raise AssertionError(
                f"evidence label mismatch for {identity}: expected {expected}, "
                f"found {actual}"
            )
    expected_count = 6 + 12
    if len(primary) != expected_count:
        raise AssertionError(f"expected {expected_count} primary evidence labels, found {len(primary)}")
    checks.append({"check": "evidence_labels_recomputed", "status": "PASS", "rows": len(primary)})


def validate() -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    missing = [name for name in REQUIRED_OUTPUTS if not (OUT / name).is_file()]
    if missing:
        raise AssertionError(f"missing required persisted outputs: {missing}")
    checks.append({"check": "required_output_inventory", "status": "PASS", "count": len(REQUIRED_OUTPUTS)})
    _validate_sources(config, checks)
    _validate_features(checks)
    _validate_monthly_samples(checks)
    _validate_pooled_terminal(checks)
    _validate_labels(checks)
    result = {"status": "PASS", "checks": checks, "check_count": len(checks)}
    (OUT / "VALIDATION_REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(validate(), sort_keys=True))


if __name__ == "__main__":
    main()
