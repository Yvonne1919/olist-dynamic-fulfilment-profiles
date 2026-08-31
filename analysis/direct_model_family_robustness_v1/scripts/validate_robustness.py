"""Fail-closed validation of the persisted robustness workspace."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from analysis.direct_promise_profile_extension_v1.scripts import direct_experiment as direct

from .robustness_integrity import AFTER_PATH, BEFORE_PATH, MANIFEST_PATH, WORKSPACE, load_config, read_json, sha256_file, utc_now, write_json


REQUIRED_OUTPUTS = [
    "RUN_MANIFEST.json",
    "README.md",
    "ROBUSTNESS_FROZEN_CONFIG.json",
    "ROBUSTNESS_PROTOCOL.md",
    "MODEL_FAMILY_DEFINITIONS.md",
    "RECOVERED_MODEL_SOURCE_RECEIPT.md",
    "PREFLIGHT_ATTEMPT1_INCIDENT_RECEIPT.md",
    "VALIDATION_ATTEMPT1_INCIDENT_RECEIPT.md",
    "FINAL_QA_SUPERSESSION_RECEIPT.md",
    "ROBUSTNESS_MODEL_SELECTION_FREEZE.json",
    "MODEL_SELECTION_ALL_FAMILIES.csv",
    "BREACH_MODEL_FAMILY_MONTHLY.csv",
    "BREACH_MODEL_FAMILY_POOLED.csv",
    "BREACH_MODEL_FAMILY_CALIBRATION.csv",
    "BREACH_PROFILE_INCREMENT_BY_FAMILY.csv",
    "BREACH_MODEL_FAMILY_SUMMARY.csv",
    "SEVERITY_MODEL_FAMILY_MONTHLY.csv",
    "SEVERITY_MODEL_FAMILY_POOLED.csv",
    "SEVERITY_MODEL_FAMILY_COVERAGE.csv",
    "SEVERITY_PROFILE_INCREMENT_BY_FAMILY.csv",
    "SEVERITY_MODEL_FAMILY_SUMMARY.csv",
    "TERMINAL_MODEL_FAMILY_ROBUSTNESS.csv",
    "PRIMARY_RESULT_REPRODUCTION_AUDIT.csv",
    "ROBUSTNESS_EVIDENCE_LABELS.csv",
    "RESULT_SUMMARY.md",
    "RESULT_SUMMARY_ZH.md",
    "VALIDATION_REPORT.json",
    "HASH_INVENTORY.txt",
    "FIGURE_DATA_BREACH_MODEL_FAMILIES.csv",
    "FIGURE_DATA_SEVERITY_MODEL_FAMILIES.csv",
    "BREACH_RF_ABLATIONS.csv",
    "BREACH_SUPPORT_STRATA.csv",
    "SEVERITY_SUPPORT_STRATA.csv",
    "RF_CALIBRATION_SELECTION.csv",
    "MODEL_MANIFESTS.csv",
    "working/RF_DEVELOPMENT_OOF_PREDICTIONS.csv.gz",
    "working/ALTERNATIVE_BREACH_PREDICTIONS.csv.gz",
    "working/ALTERNATIVE_SEVERITY_PREDICTIONS.csv.gz",
    "working/MODEL_RUN_RECEIPT.json",
    "working/REPORT_RECEIPT.json",
    "working/CONTROL_HASHES_BEFORE_MODEL.json",
    "working/INTEGRITY_BEFORE.json",
    "working/INTEGRITY_AFTER.json",
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(WORKSPACE / name, low_memory=False)


def _json_safe_detail(value: Any) -> Any:
    """Convert diagnostic payloads, including MultiIndex dictionaries, to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_detail(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def validate() -> dict[str, Any]:
    config = load_config()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": _json_safe_detail(detail)})

    missing = [name for name in REQUIRED_OUTPUTS if name != "VALIDATION_REPORT.json" and not (WORKSPACE / name).is_file()]
    check("required_outputs_present_before_report_write", not missing, {"missing": missing})

    before = read_json(BEFORE_PATH)
    after = read_json(AFTER_PATH)
    check("protected_files_byte_identical", after.get("passed") is True, after.get("comparison"))
    check("robustness_controls_unchanged_after_model_freeze", after.get("control_comparison", {}).get("passed") is True, after.get("control_comparison"))
    check(
        "complete_direct_extension_hashed",
        "analysis/direct_promise_profile_extension_v1" in before["protection_boundary"]["roots"],
        before["protection_boundary"]["roots"].get("analysis/direct_promise_profile_extension_v1"),
    )
    check(
        "analysis_directory_not_enumerated",
        before["protection_boundary"].get("analysis_directory_enumerated") is False
        and after["protection_boundary"].get("analysis_directory_enumerated") is False,
    )

    manifest = read_json(MANIFEST_PATH)
    check("profile_history_variant_selected_90_day", manifest.get("profile_history_variant") == "selected_90_day")
    check("all_mature_workspace_not_consumed", manifest.get("all_mature_workspace_consumed") is False)
    check("isolation_incident_recorded", bool(manifest.get("isolation_incident", {}).get("occurred_before_execution_freeze")))

    reproduction = _read("PRIMARY_RESULT_REPRODUCTION_AUDIT.csv")
    direct_labels_for_keys = pd.read_csv(
        WORKSPACE.parent / "direct_promise_profile_extension_v1/EVIDENCE_LABELS.csv",
        low_memory=False,
    )
    expected_core_keys = {
        (
            str(row.task), str(row.family), str(row.comparison),
            -1.0 if pd.isna(row.quantile) else float(row.quantile),
        )
        for row in direct_labels_for_keys.itertuples(index=False)
    }
    actual_core = reproduction.loc[reproduction["record_type"].eq("core_comparison")]
    actual_core_keys = {
        (
            str(row.task), str(row.family), str(row.comparison),
            -1.0 if pd.isna(row.quantile) else float(row.quantile),
        )
        for row in actual_core.itertuples(index=False)
    }
    expected_artifacts = {
        "model_selection", "breach_monthly", "breach_pooled", "breach_calibration",
        "severity_monthly", "severity_pooled", "severity_coverage", "terminal", "evidence_labels",
    }
    actual_artifacts = set(
        reproduction.loc[reproduction["record_type"].eq("artifact_table"), "comparison"].astype(str)
    )
    selection_gate = reproduction.loc[
        reproduction["record_type"].eq("selection_freeze")
        & reproduction["comparison"].eq("selection_json")
    ]
    check(
        "primary_numeric_reproduction_gate_passed",
        len(reproduction) == 28
        and expected_core_keys == actual_core_keys
        and expected_artifacts == actual_artifacts
        and len(selection_gate) == 1
        and reproduction["passed"].fillna(False).astype(bool).all(),
        {"rows": len(reproduction), "failed": reproduction.loc[~reproduction["passed"].fillna(False).astype(bool), "comparison"].tolist()},
    )

    labels = _read("ROBUSTNESS_EVIDENCE_LABELS.csv")
    protected = labels.loc[labels["label_type"].eq("PROTECTED PRE-EXISTING DIRECT-EXTENSION LABEL")]
    original = pd.read_csv(WORKSPACE.parent / "direct_promise_profile_extension_v1/EVIDENCE_LABELS.csv", low_memory=False)
    key = ["task", "family", "comparison", "quantile"]
    protected_key_labels = protected[key + ["evidence_label"]].copy()
    original_key_labels = original[key + ["evidence_label"]].copy()
    for table in (protected_key_labels, original_key_labels):
        table["quantile_key"] = pd.to_numeric(table["quantile"], errors="coerce").fillna(-1.0)
    protected_key_labels = protected_key_labels.sort_values(
        ["task", "family", "comparison", "quantile_key"], kind="mergesort"
    ).reset_index(drop=True)
    original_key_labels = original_key_labels.sort_values(
        ["task", "family", "comparison", "quantile_key"], kind="mergesort"
    ).reset_index(drop=True)
    check(
        "protected_primary_labels_unchanged",
        len(protected) == 18
        and protected["protected_label_unchanged"].fillna(False).astype(bool).all()
        and protected_key_labels.equals(original_key_labels),
        {"protected_rows": len(protected), "source_sha256": sha256_file(WORKSPACE.parent / "direct_promise_profile_extension_v1/EVIDENCE_LABELS.csv")},
    )
    primary_severity = protected.loc[protected["task"].eq("severity")]
    check(
        "all_twelve_primary_severity_labels_not_supported",
        len(primary_severity) == 12 and primary_severity["evidence_label"].eq("Not-supported").all(),
    )
    rf_labels = labels.loc[
        labels["family"].eq("random_forest") & labels["label_type"].eq("ROBUSTNESS EVIDENCE LABEL")
    ]
    check("random_forest_has_three_robustness_labels", len(rf_labels) == 3)
    spline = labels.loc[labels["family"].eq("spline_logistic")]
    check(
        "spline_logistic_blocker_persisted",
        len(spline) == 3 and spline["evidence_label"].eq("Incomplete").all(),
    )
    new_severity = labels.loc[
        labels["task"].eq("severity") & labels["label_type"].eq("ROBUSTNESS EVIDENCE LABEL")
    ]
    check(
        "additional_severity_labels_complete",
        len(new_severity) == 12
        and set(new_severity["family"]) == {"random_forest_leaf_weighted_quantile", "lognormal_ridge"}
        and set(pd.to_numeric(new_severity["quantile"])) == {0.5, 0.9},
    )

    selection = _read("MODEL_SELECTION_ALL_FAMILIES.csv")
    selected_families = set(selection.loc[selection["record_type"].eq("selected_specification"), "family"].dropna())
    check(
        "all_evaluable_families_selected_on_development",
        {
            "logistic_l2", "xgboost", "random_forest", "linear_quantile",
            "xgboost_quantile", "random_forest_leaf_weighted_quantile", "lognormal_ridge",
        }.issubset(selected_families),
        sorted(selected_families),
    )
    additional_tuning = selection.loc[
        selection["record_type"].eq("development_tuning")
        & selection["family"].isin(["random_forest", "random_forest_leaf_weighted_quantile", "lognormal_ridge"])
    ]
    check(
        "additional_families_use_three_development_folds_only",
        set(pd.to_numeric(additional_tuning["fold"], errors="coerce").dropna().astype(int)) == {1, 2, 3}
        and additional_tuning["development_only"].fillna(False).astype(bool).all()
        and (~additional_tuning["later_or_terminal_outcomes_used"].fillna(True).astype(bool)).all(),
    )
    check(
        "additional_grids_are_recovered_singletons",
        additional_tuning["grid_type"].eq("recovered_singleton").all()
        and additional_tuning["parameter_index"].fillna(-1).astype(int).eq(0).all(),
    )
    rf_tuning_rows = additional_tuning.loc[
        additional_tuning["family"].eq("random_forest")
    ].sort_values("fold", kind="mergesort")
    expected_breach_dev = [(7904, 6695), (15659, 7062), (24106, 10309)]
    check(
        "rf_development_sample_counts_exact",
        list(zip(rf_tuning_rows["n_train"].astype(int), rf_tuning_rows["n_validation"].astype(int)))
        == expected_breach_dev,
    )
    expected_severity_dev = [(243, 96), (429, 176), (754, 628)]
    severity_dev_ok = True
    for family in ("random_forest_leaf_weighted_quantile", "lognormal_ridge"):
        for quantile in (0.5, 0.9):
            rows = additional_tuning.loc[
                additional_tuning["family"].eq(family)
                & pd.to_numeric(additional_tuning["quantile"], errors="coerce").eq(quantile)
            ].sort_values("fold", kind="mergesort")
            severity_dev_ok = severity_dev_ok and list(
                zip(rows["n_train"].astype(int), rows["n_validation"].astype(int))
            ) == expected_severity_dev
    check("additional_severity_development_sample_counts_exact", severity_dev_ok)

    breach_monthly = _read("BREACH_MODEL_FAMILY_MONTHLY.csv")
    breach_calibrated = breach_monthly.loc[breach_monthly["probability_type"].eq("calibrated")]
    breach_counts = breach_calibrated.groupby(["family", "model_id"], observed=True)["cohort"].nunique()
    check(
        "breach_six_months_every_evaluable_family_and_spec",
        len(breach_counts) == 12 and breach_counts.eq(6).all()
        and set(breach_calibrated["cohort"]) == set(config["periods"]["later_months"]),
        breach_counts.to_dict(),
    )
    expected_breach_counts = {
        "2018-01": 7069, "2018-02": 6555, "2018-03": 7003,
        "2018-04": 6798, "2018-05": 6749, "2018-06": 6096,
    }
    actual_breach_counts = (
        breach_calibrated.loc[breach_calibrated["family"].eq("random_forest") & breach_calibrated["model_id"].eq("DP0")]
        .set_index("cohort")["n_orders"].astype(int).to_dict()
    )
    check("breach_later_cohort_sample_counts_exact", actual_breach_counts == expected_breach_counts, actual_breach_counts)
    breach_hashes = breach_calibrated.groupby(["period", "cohort", "family"], observed=True)["order_id_sha256"].nunique()
    check("breach_identical_rows_within_family_month", breach_hashes.eq(1).all(), breach_hashes.to_dict())

    pairs = _read("BREACH_PROFILE_INCREMENT_BY_FAMILY.csv")
    later_pairs = pairs.loc[pairs["period"].eq("later")]
    pair_counts = later_pairs.groupby(["family", "comparison"], observed=True)["cohort"].nunique()
    check(
        "breach_month_based_increments_complete",
        len(pair_counts) == 9 and pair_counts.eq(6).all()
        and later_pairs["order_id_sha256"].eq(later_pairs["paired_order_id_sha256"]).all(),
        pair_counts.to_dict(),
    )
    rf_ablations = _read("BREACH_RF_ABLATIONS.csv")
    rf_later_ablations = rf_ablations.loc[rf_ablations["period"].eq("later")]
    ablation_counts = rf_later_ablations.groupby(
        ["model_id", "representation"], observed=True
    )["cohort"].nunique()
    check(
        "rf_score_contribution_ablations_persisted",
        len(ablation_counts) == 9 and ablation_counts.eq(6).all()
        and set(rf_later_ablations["representation"]) == {"full", "score_only", "metadata_only"},
        ablation_counts.to_dict(),
    )
    rf_oof = pd.read_csv(WORKSPACE / "working/RF_DEVELOPMENT_OOF_PREDICTIONS.csv.gz", low_memory=False)
    check(
        "rf_calibration_oof_predictions_persisted",
        set(rf_oof["model_id"]) == {"DP0", "DPS", "DPG", "DPB"}
        and set(pd.to_numeric(rf_oof["fold"]).astype(int)) == {1, 2, 3}
        and rf_oof["order_id"].notna().all(),
        {"rows": len(rf_oof)},
    )

    rf_support = _read("BREACH_SUPPORT_STRATA.csv")
    rf_support = rf_support.loc[rf_support["family"].eq("random_forest")]
    rf_high = rf_support.loc[
        rf_support["period"].eq("later") & rf_support["support_stratum"].eq("20+")
    ]
    rf_high_counts = rf_high.groupby("comparison", observed=True)["cohort"].nunique()
    check(
        "rf_high_support_guard_has_six_months",
        len(rf_high_counts) == 3 and rf_high_counts.eq(6).all(),
        rf_high_counts.to_dict(),
    )

    calibration = _read("BREACH_MODEL_FAMILY_CALIBRATION.csv")
    reliability = calibration.loc[calibration["record_type"].eq("reliability_bin")]
    check(
        "breach_reliability_bins_persisted",
        not reliability.empty and set(reliability["family"].dropna()) == {"logistic_l2", "random_forest", "xgboost"}
        and set(reliability["probability_type"].dropna()) == {"raw", "calibrated"},
    )
    rf_metrics_for_rubric = calibration.loc[
        calibration["record_type"].eq("model_metric")
        & calibration["family"].eq("random_forest")
        & calibration["period"].isin(["later", "terminal"])
    ].copy()
    rf_pairs_for_rubric = pairs.loc[pairs["family"].eq("random_forest")].copy()
    direct_config = direct.load_config()
    rf_config = json.loads(json.dumps(direct_config))
    rf_config["breach"]["families"] = ["random_forest"]
    recomputed_rf = direct._breach_evidence_summary(
        rf_pairs_for_rubric, rf_metrics_for_rubric, rf_support, rf_ablations, rf_config
    ).sort_values("comparison", kind="mergesort").reset_index(drop=True)
    published_rf = labels.loc[
        labels["family"].eq("random_forest")
        & labels["label_type"].eq("ROBUSTNESS EVIDENCE LABEL")
    ].sort_values("comparison", kind="mergesort").reset_index(drop=True)
    rubric_match = len(recomputed_rf) == len(published_rf) == 3
    if rubric_match:
        rubric_match = bool(
            recomputed_rf["evidence_label"].astype(str).equals(published_rf["evidence_label"].astype(str))
            and recomputed_rf["all_guards_pass"].astype(bool).equals(published_rf["all_guards_pass"].astype(bool))
            and np.allclose(
                recomputed_rf["median_delta_log_loss"], published_rf["median_delta_log_loss"],
                rtol=0.0, atol=1e-10, equal_nan=True,
            )
            and np.allclose(
                recomputed_rf["median_delta_brier"], published_rf["median_delta_brier"],
                rtol=0.0, atol=1e-10, equal_nan=True,
            )
        )
    check("rf_robustness_labels_recomputed_from_persisted_inputs", rubric_match)

    severity_monthly = _read("SEVERITY_MODEL_FAMILY_MONTHLY.csv")
    severity_counts = severity_monthly.groupby(["family", "model_id", "quantile"], observed=True)["cohort"].nunique()
    check(
        "severity_six_months_every_family_spec_quantile",
        len(severity_counts) == 32 and severity_counts.eq(6).all()
        and set(severity_monthly["cohort"]) == set(config["periods"]["later_months"]),
        severity_counts.to_dict(),
    )
    expected_severity_counts = {
        "2018-01": 403, "2018-02": 926, "2018-03": 1328,
        "2018-04": 306, "2018-05": 443, "2018-06": 71,
    }
    actual_severity_counts = (
        severity_monthly.loc[
            severity_monthly["family"].eq("random_forest_leaf_weighted_quantile")
            & severity_monthly["model_id"].eq("DQ0")
            & pd.to_numeric(severity_monthly["quantile"]).eq(0.5)
        ].set_index("cohort")["n_orders"].astype(int).to_dict()
    )
    check("severity_later_breached_sample_counts_exact", actual_severity_counts == expected_severity_counts, actual_severity_counts)
    severity_hashes = severity_monthly.groupby(["period", "cohort", "family", "quantile"], observed=True)["order_id_sha256"].nunique()
    check("severity_identical_rows_within_family_month_quantile", severity_hashes.eq(1).all())

    support = _read("SEVERITY_SUPPORT_STRATA.csv")
    high = support.loc[support["period"].eq("later") & support["support_stratum"].eq("20+")]
    high_counts = high.groupby(["family", "comparison", "quantile"], observed=True)["cohort"].nunique()
    check("severity_support_ge20_six_months", len(high_counts) == 24 and high_counts.eq(6).all(), high_counts.to_dict())

    severity_config = json.loads(json.dumps(direct_config))
    severity_config["severity"]["families"] = [
        "random_forest_leaf_weighted_quantile", "lognormal_ridge"
    ]
    recomputed_severity = direct._severity_evidence_summary(
        severity_monthly.loc[
            severity_monthly["family"].isin(severity_config["severity"]["families"])
        ],
        support.loc[support["family"].isin(severity_config["severity"]["families"])],
        severity_config,
    ).sort_values(["family", "comparison", "quantile"], kind="mergesort").reset_index(drop=True)
    published_severity = labels.loc[
        labels["task"].eq("severity")
        & labels["label_type"].eq("ROBUSTNESS EVIDENCE LABEL")
    ].sort_values(["family", "comparison", "quantile"], kind="mergesort").reset_index(drop=True)
    severity_rubric_match = len(recomputed_severity) == len(published_severity) == 12
    if severity_rubric_match:
        severity_rubric_match = bool(
            recomputed_severity["evidence_label"].astype(str).equals(
                published_severity["evidence_label"].astype(str)
            )
            and recomputed_severity["all_guards_pass"].astype(bool).equals(
                published_severity["all_guards_pass"].astype(bool)
            )
            and np.allclose(
                recomputed_severity["median_skill"], published_severity["median_skill"],
                rtol=0.0, atol=1e-10, equal_nan=True,
            )
            and recomputed_severity["favourable_month_count"].astype(int).equals(
                published_severity["favourable_month_count"].astype(int)
            )
            and np.allclose(
                recomputed_severity["median_absolute_coverage_error_deterioration"],
                published_severity["median_absolute_coverage_error_deterioration"],
                rtol=0.0, atol=1e-10, equal_nan=True,
            )
        )
    check("additional_severity_labels_recomputed_from_persisted_inputs", severity_rubric_match)

    coverage = _read("SEVERITY_MODEL_FAMILY_COVERAGE.csv")
    q90 = coverage.loc[pd.to_numeric(coverage["quantile"], errors="coerce").eq(0.9)]
    check(
        "q90_monthly_and_pooled_coverage_present",
        {"later", "aggregate"}.issubset(set(q90["period"]))
        and q90["empirical_coverage"].notna().all()
        and q90["coverage_error"].notna().all(),
    )

    terminal = _read("TERMINAL_MODEL_FAMILY_ROBUSTNESS.csv")
    check(
        "terminal_separate_and_unlabelled",
        set(terminal["period"]) == {"terminal"}
        and terminal["evidence_label"].fillna("").eq("").all()
        and terminal["label_type"].eq("none_terminal_stress").all(),
    )

    manifests = _read("MODEL_MANIFESTS.csv")
    allowed_features = {"promised_delivery_days"}
    for block in ("S1", "S2", "R1", "R2"):
        for suffix in config["profiles"]["payload_suffixes"]:
            allowed_features.add(f"{block}_{suffix}")
    feature_ok = True
    feature_failures: list[dict[str, Any]] = []
    for index, row in manifests.iterrows():
        try:
            numeric = json.loads(row["numeric_features_json"])
            categorical = json.loads(row["categorical_features_json"])
            task = str(row["task"])
            model_id = str(row["model_id"])
            representation = str(row["representation"])
            if task == "breach":
                expected_numeric, expected_categorical = direct.breach_feature_map(representation)[model_id]
            elif task == "severity":
                expected_numeric, expected_categorical = direct.severity_feature_map(representation)[model_id]
            else:
                raise KeyError(f"unexpected task {task}")
            expected_hash = __import__("hashlib").sha256(
                ("\n".join(expected_numeric + expected_categorical) + "\n").encode()
            ).hexdigest()
            row_ok = (
                numeric == expected_numeric
                and categorical == expected_categorical == []
                and set(numeric).issubset(allowed_features)
                and str(row["ordered_feature_sha256"]) == expected_hash
            )
        except Exception as error:
            row_ok = False
            expected_numeric = []
            expected_categorical = []
            error_text = f"{type(error).__name__}:{error}"
        else:
            error_text = ""
        if not row_ok:
            feature_ok = False
            feature_failures.append({
                "index": int(index), "task": row.get("task"), "family": row.get("family"),
                "model_id": row.get("model_id"), "representation": row.get("representation"),
                "error": error_text,
            })
    check("exact_ordered_direct_feature_vectors_only", feature_ok, feature_failures[:20])
    check(
        "source_model_frame_hash_pinned",
        manifests["source_model_frame_sha256"].notna().all()
        and manifests["source_model_frame_sha256"].eq(config["sources"]["order_model_frame"][1]).all(),
    )

    overall = all(row["passed"] for row in checks)
    report = {
        "analysis_id": config["analysis_id"],
        "validated_at_utc": utc_now(),
        "overall_passed": overall,
        "check_count": len(checks),
        "passed_count": sum(int(row["passed"]) for row in checks),
        "failed_count": sum(int(not row["passed"]) for row in checks),
        "checks": checks,
        "known_blockers": [
            "spline Logistic unavailable because no applicable exact predictive direct-feature implementation/grid was recovered",
            "additional breach and severity specifications use recovered singleton grids because no historical multi-point grids existed",
        ],
        "isolation_incident": config["isolation_incident"],
    }
    write_json(WORKSPACE / "VALIDATION_REPORT.json", report)
    if not overall:
        failed = [row["name"] for row in checks if not row["passed"]]
        raise RuntimeError(f"robustness validation failed: {failed}")
    return report


__all__ = ["REQUIRED_OUTPUTS", "validate"]
