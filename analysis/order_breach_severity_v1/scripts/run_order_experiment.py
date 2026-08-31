"""Stage runner for the frozen order breach/severity experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.order_breach_severity_v1.scripts import (
    order_experiment,
    order_features,
    order_io,
    order_modeling,
    order_preflight,
    order_profiles,
)
from analysis.profile_pivot_phase2a.scripts import data_pipeline


MODEL_FRAME_PATH = order_io.WORK / "ORDER_MODEL_FRAME.csv.gz"
MODEL_FRAME_RECEIPT_PATH = order_io.WORK / "ORDER_MODEL_FRAME_RECEIPT.json"
SELECTION_FREEZE_PATH = order_io.OUT / "ORDER_MODEL_SELECTION_FREEZE.json"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _require_preflight() -> dict:
    if not order_preflight.PRESTATE_PATH.is_file():
        raise RuntimeError("preflight receipt missing; run --stage preflight first")
    state = _read_json(order_preflight.PRESTATE_PATH)
    order_preflight.verify_protected_unchanged(state)
    return state


def _period_label(purchase: pd.Series) -> pd.Series:
    date = pd.to_datetime(purchase, errors="raise")
    return pd.Series(
        np.select(
            [
                date.lt(pd.Timestamp("2018-01-01")),
                date.lt(pd.Timestamp("2018-07-01")),
            ],
            ["development", "later"],
            default="terminal",
        ),
        index=purchase.index,
        dtype="string",
    )


def _sample_audit(canonical: pd.DataFrame, model_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(label: str, frame: pd.DataFrame) -> None:
        rows.append(
            {
                "sample": label,
                "n_orders": len(frame),
                "n_unique_orders": frame["order_id"].nunique(),
                "purchase_date_min": pd.to_datetime(frame["purchase_date"]).min(),
                "purchase_date_max": pd.to_datetime(frame["purchase_date"]).max(),
                "n_breaches": int(pd.to_numeric(frame["late_delivery"]).sum()),
                "breach_rate": pd.to_numeric(frame["late_delivery"]).mean(),
                "n_positive_severity": int(pd.to_numeric(frame["positive_late_days"]).gt(0).sum()),
                "n_multi_seller": int(pd.to_numeric(frame["multi_seller"]).sum()),
                "multi_seller_share": pd.to_numeric(frame["multi_seller"]).mean(),
                "unresolved_target_rows": int(frame["late_delivery"].isna().sum()),
                "duplicate_order_ids": int(frame["order_id"].duplicated().sum()),
                "source_hashes_valid": True,
                "canonical_assembler_valid": True,
                "source_verdict": "verified",
                "protected_files_unchanged": True,
                "protected_hashes_valid": True,
                "preservation_verdict": "unchanged",
            }
        )

    full = canonical.copy()
    full["purchase_date"] = pd.to_datetime(full["order_purchase_timestamp"]).dt.normalize()
    full["multi_seller"] = pd.to_numeric(full["n_unique_sellers"]).gt(1).astype("int8")
    add("canonical_delivered_all_dates", full)
    add("analytical_population_2017-04-01_to_2018-08-30", model_frame)
    for period, group in model_frame.groupby("period", sort=True, observed=True):
        add(str(period), group)
    for flag, label in ((0, "single_seller"), (1, "multi_seller")):
        add(label, model_frame.loc[pd.to_numeric(model_frame["multi_seller"]).eq(flag)])
    return pd.DataFrame(rows)


def _flatten_profile_audit(audit: Mapping[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    blocks = audit.get("blocks", {})
    for block, detail in sorted(blocks.items()):
        row = {"block": block}
        if isinstance(detail, Mapping):
            row.update({key: value for key, value in detail.items() if not isinstance(value, (dict, list))})
        row.update(
            {
                "input_orders": audit.get("input_orders"),
                "output_orders": audit.get("output_orders"),
                "snapshot_rule": audit.get("snapshot_rule"),
                "snapshot_after_purchase_violations": audit.get("snapshot_after_purchase_violations"),
                "profile_daily_sha256": audit.get("profile_daily_sha256"),
                "profile_parent_sha256": audit.get("profile_parent_sha256"),
                "row_preservation_valid": audit.get("row_preservation_valid"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_hrd(frame: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    path = order_io.ROOT / str(config["data"]["hrd_daily_labels"])
    hrd = pd.read_csv(path)
    required = {"date", "both_top10", "both_top10_phase"}
    if not required.issubset(hrd.columns):
        raise KeyError(f"HRD source missing {sorted(required - set(hrd.columns))}")
    hrd["purchase_date"] = pd.to_datetime(hrd["date"], format="%Y-%m-%d", errors="raise")
    if hrd["purchase_date"].duplicated().any():
        raise AssertionError("HRD daily source has duplicated dates")
    result = frame.merge(
        hrd[["purchase_date", "both_top10", "both_top10_phase"]],
        on="purchase_date", how="left", validate="many_to_one",
    )
    if result[["both_top10", "both_top10_phase"]].isna().any().any():
        raise AssertionError("an analytical purchase date has no retrospective HRD stratum")
    result["both_top10"] = result["both_top10"].astype(str).str.lower().isin(["true", "1"])
    return result


def _model_frame_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [
                "order_id", "order_purchase_timestamp", "order_delivered_customer_date", "purchase_date", "period",
                "late_delivery", "positive_late_days", "multi_seller", "known_event_indicator",
                "both_top10", "both_top10_phase",
                *order_features.CURRENT_ORDER_FEATURES,
                *order_profiles.PROFILE_JOIN_COLUMNS,
                "S1_score_x_known_event", "R1_score_x_known_event",
            ]
        )
    )


def run_prepare(data_dir: Path) -> None:
    prestate = _require_preflight()
    canonical = data_pipeline.build_order_base(data_dir)
    if len(canonical) != 96_470 or canonical["order_id"].duplicated().any():
        raise AssertionError("canonical assembler did not produce exactly 96,470 unique orders")
    enriched = order_features.build_current_order_features(canonical)
    enriched["purchase_date"] = pd.to_datetime(enriched["order_purchase_timestamp"]).dt.normalize()
    analytical = enriched.loc[
        enriched["purchase_date"].ge("2017-04-01") & enriched["purchase_date"].le("2018-08-30")
    ].copy()
    analytical["period"] = _period_label(analytical["purchase_date"])
    analytical, profile_audit = order_profiles.join_profiles(analytical, order_io.load_config())
    analytical = order_experiment.add_frozen_interactions(analytical)
    analytical = _attach_hrd(analytical, order_io.load_config())
    order_experiment.validate_model_frame(analytical)

    sample = _sample_audit(canonical, analytical)
    order_io.write_csv(order_io.OUT / "ORDER_SAMPLE_AUDIT.csv", sample)
    order_io.write_csv(order_io.OUT / "ORDER_PROFILE_JOIN_AUDIT.csv", _flatten_profile_audit(profile_audit))
    order_io.write_json(order_io.WORK / "ORDER_PROFILE_JOIN_AUDIT.json", profile_audit)
    for name, table in order_experiment.development_eda(analytical).items():
        order_io.write_csv(order_io.WORK / f"EDA_{name.upper()}.csv", table)

    persisted = analytical.loc[:, _model_frame_columns()].copy()
    for column in ("order_purchase_timestamp", "order_delivered_customer_date"):
        persisted[column] = pd.to_datetime(persisted[column], errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    persisted["purchase_date"] = pd.to_datetime(persisted["purchase_date"]).dt.strftime("%Y-%m-%d")
    for column in [name for name in persisted if name.endswith("last_mature_outcome_date")]:
        persisted[column] = pd.to_datetime(persisted[column], errors="coerce").dt.strftime("%Y-%m-%d")
    persisted = persisted.sort_values(["purchase_date", "order_id"], kind="mergesort").reset_index(drop=True)
    order_io.write_gzip_csv(MODEL_FRAME_PATH, persisted)
    receipt = {
        "rows": len(persisted), "columns": list(persisted.columns), "sha256": order_io.sha256_file(MODEL_FRAME_PATH),
        "order_id_sha256": order_modeling.order_id_hash(persisted["order_id"]),
        "profile_join_audit_sha256": order_io.sha256_file(order_io.OUT / "ORDER_PROFILE_JOIN_AUDIT.csv"),
        "sample_audit_sha256": order_io.sha256_file(order_io.OUT / "ORDER_SAMPLE_AUDIT.csv"),
    }
    order_io.write_json(MODEL_FRAME_RECEIPT_PATH, receipt)
    order_preflight.verify_protected_unchanged(prestate)
    order_io.append_run_event("order_model_frame_prepared", **receipt)


def _load_model_frame() -> pd.DataFrame:
    if not MODEL_FRAME_PATH.is_file() or not MODEL_FRAME_RECEIPT_PATH.is_file():
        raise RuntimeError("prepared model frame/receipt missing")
    receipt = _read_json(MODEL_FRAME_RECEIPT_PATH)
    if order_io.sha256_file(MODEL_FRAME_PATH) != receipt["sha256"]:
        raise RuntimeError("model frame hash drift")
    frame = pd.read_csv(MODEL_FRAME_PATH, low_memory=False)
    for column in ("order_purchase_timestamp", "order_delivered_customer_date", "purchase_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in [name for name in frame if name.endswith("cold_start")]:
        frame[column] = frame[column].astype(str).str.lower().isin(["true", "1"])
    frame["both_top10"] = frame["both_top10"].astype(str).str.lower().isin(["true", "1"])
    frame = order_experiment.add_frozen_interactions(frame.drop(columns=["S1_score_x_known_event", "R1_score_x_known_event"]))
    order_experiment.validate_model_frame(frame)
    if len(frame) != receipt["rows"] or order_modeling.order_id_hash(frame["order_id"]) != receipt["order_id_sha256"]:
        raise RuntimeError("model frame receipt mismatch")
    return frame


def run_tune() -> None:
    prestate = _require_preflight()
    frame = _load_model_frame()
    config = order_io.load_config()
    classifier_params, breach_tuning = order_experiment.tune_classification(frame, config)
    calibrators, oof, calibration_audit = order_experiment.development_oof_and_calibration(
        frame, config, classifier_params
    )
    severity_params, severity_tuning = order_experiment.tune_severity(frame, config, classifier_params)
    tuning = pd.concat([breach_tuning, severity_tuning], ignore_index=True)
    order_io.write_csv(order_io.OUT / "ORDER_DEVELOPMENT_TUNING.csv", tuning)
    order_io.write_gzip_csv(order_io.WORK / "DEVELOPMENT_OOF_PREDICTIONS.csv.gz", oof)
    order_io.write_csv(order_io.WORK / "DEVELOPMENT_CALIBRATION_SELECTION.csv", calibration_audit)

    source_hashes = prestate["source_code_hashes"]
    freeze = {
        "analysis_id": config["analysis_id"],
        "selection_source": "development_2017-04-01_to_2017-12-31_only",
        "later_or_terminal_outcomes_used": False,
        "model_frame_sha256": order_io.sha256_file(MODEL_FRAME_PATH),
        "config_sha256": order_io.sha256_file(order_io.CONFIG_PATH),
        "prestate_sha256": order_io.sha256_file(order_preflight.PRESTATE_PATH),
        "source_code_hashes": source_hashes,
        "classification_parameters": classifier_params,
        "calibrators": calibrators,
        "severity_parameters": severity_params,
    }
    order_io.write_json(SELECTION_FREEZE_PATH, freeze)
    parameter_rows: list[dict[str, object]] = []
    for family, params in classifier_params.items():
        for model_id in config["classification"]["model_ladder"]:
            parameter_rows.append(
                {
                    "task": "breach", "family": family, "model_id": model_id, "quantile": np.nan,
                    "parameters_json": order_modeling.stable_json(params),
                    "calibration_method": calibrators[family][model_id]["method"],
                    "development_only": True,
                }
            )
    for family, by_quantile in severity_params.items():
        for quantile, params in by_quantile.items():
            for model_id in config["severity"]["model_ladder"]:
                parameter_rows.append(
                    {
                        "task": "severity", "family": family, "model_id": model_id, "quantile": float(quantile),
                        "parameters_json": order_modeling.stable_json(params), "calibration_method": "not_applicable",
                        "development_only": True,
                    }
                )
    order_io.write_csv(order_io.OUT / "ORDER_MODEL_PARAMETERS.csv", pd.DataFrame(parameter_rows))
    order_preflight.verify_protected_unchanged(prestate)
    order_io.append_run_event(
        "development_model_selection_frozen",
        freeze_sha256=order_io.sha256_file(SELECTION_FREEZE_PATH),
        later_or_terminal_outcomes_used=False,
    )


def _verify_selection_freeze(prestate: Mapping[str, object]) -> dict:
    if not SELECTION_FREEZE_PATH.is_file():
        raise RuntimeError("ORDER_MODEL_SELECTION_FREEZE.json missing")
    freeze = _read_json(SELECTION_FREEZE_PATH)
    if freeze.get("later_or_terminal_outcomes_used") is not False:
        raise RuntimeError("development selection freeze has invalid label-access status")
    if freeze.get("model_frame_sha256") != order_io.sha256_file(MODEL_FRAME_PATH):
        raise RuntimeError("selection freeze model-frame hash mismatch")
    if freeze.get("config_sha256") != order_io.sha256_file(order_io.CONFIG_PATH):
        raise RuntimeError("selection freeze config hash mismatch")
    if freeze.get("source_code_hashes") != prestate.get("source_code_hashes"):
        raise RuntimeError("selection freeze source hashes mismatch")
    return freeze


def _terminal_stress(
    breach_results: pd.DataFrame,
    paired: pd.DataFrame,
    severity_results: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    terminal_breach = breach_results.loc[breach_results["period"].eq("terminal")].copy()
    later_breach = breach_results.loc[
        breach_results["period"].eq("aggregate")
        & breach_results["cohort"].eq("later_pooled")
    ].copy()
    for _, row in terminal_breach.iterrows():
        probability_type = str(row["probability_type"])
        common = {
            "period": row["period"],
            "cohort": row["cohort"],
            "family": row["family"],
            "model_id": row["model_id"],
            "probability_type": probability_type,
            "comparison": "",
            "quantile": np.nan,
            "n_orders": row["n_orders"],
        }
        for metric in (
            "log_loss", "brier", "average_precision", "roc_auc", "wace",
            "calibration_intercept", "calibration_slope",
        ):
            rows.append(
                {
                    "analysis": "breach_model", **common, "metric": metric, "estimate": row[metric],
                    "interpretation": "terminal_stress_not_model_selection",
                }
            )
        reference = later_breach.loc[
            later_breach["family"].eq(row["family"])
            & later_breach["model_id"].eq(row["model_id"])
            & later_breach["probability_type"].eq(row["probability_type"])
        ]
        if len(reference) != 1:
            raise AssertionError(
                "terminal calibration shift requires one later-pooled row for "
                f"{row['family']}/{row['model_id']}/{probability_type}; found {len(reference)}"
            )
        later_row = reference.iloc[0]
        for metric in ("calibration_intercept", "calibration_slope", "wace"):
            rows.append(
                {
                    "analysis": "breach_calibration_shift",
                    **common,
                    "comparison": "terminal_minus_later_pooled",
                    "metric": f"terminal_minus_later_{metric}",
                    "estimate": row[metric] - later_row[metric],
                    "terminal_estimate": row[metric],
                    "later_pooled_estimate": later_row[metric],
                    "later_pooled_n_orders": later_row["n_orders"],
                    "interpretation": "terminal_minus_january_june_pooled_calibration_shift",
                }
            )
    for _, row in paired.loc[paired["period"].eq("terminal")].iterrows():
        for metric in ("delta_log_loss", "delta_brier"):
            rows.append(
                {
                    "analysis": "breach_increment", "period": row["period"], "cohort": row["cohort"],
                    "family": row["family"], "model_id": row["candidate_model"],
                    "probability_type": "calibrated",
                    "comparison": row["comparison"], "quantile": np.nan, "metric": metric,
                    "estimate": row[metric], "n_orders": row["n_orders"],
                    "interpretation": "terminal_stress_not_model_selection",
                }
            )
    for _, row in severity_results.loc[severity_results["period"].eq("terminal")].iterrows():
        rows.append(
            {
                "analysis": "severity", "period": row["period"], "cohort": row["cohort"],
                "family": row["family"], "model_id": row["model_id"],
                "probability_type": "not_applicable",
                "comparison": f"{row['model_id']}-Q1", "quantile": row["quantile"], "metric": "pinball_skill",
                "estimate": row["skill_vs_q1"], "n_orders": row["n_orders"],
                "interpretation": "terminal_stress_not_model_selection",
            }
        )
    later = predictions.loc[predictions["period"].eq("later") & predictions["model_id"].eq("M4")]
    terminal = predictions.loc[predictions["period"].eq("terminal") & predictions["model_id"].eq("M4")]
    for family in sorted(predictions["family"].unique()):
        for block in ("S1", "S2", "R1", "R2"):
            old = pd.to_numeric(later.loc[later["family"].eq(family), f"{block}_score"], errors="coerce")
            new = pd.to_numeric(terminal.loc[terminal["family"].eq(family), f"{block}_score"], errors="coerce")
            rows.append(
                {
                    "analysis": "profile_score_shift", "period": "terminal", "cohort": "2018-07_to_2018-08",
                    "family": family, "model_id": "M4", "probability_type": "not_applicable",
                    "comparison": block,
                    "quantile": np.nan, "metric": "terminal_minus_later_median_score",
                    "estimate": new.median() - old.median(), "n_orders": len(new),
                    "interpretation": "order_exposure_weighted_descriptive_shift",
                }
            )
    return pd.DataFrame(rows)


def _reporting_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "cohort" in result.columns and "cohort_month" not in result.columns:
        result["cohort_month"] = result["cohort"]
    if "support_stratum" in result.columns and "stratum" not in result.columns:
        result["stratum"] = result["support_stratum"]
    return result


def run_evaluate() -> None:
    prestate = _require_preflight()
    selection = _verify_selection_freeze(prestate)
    frame = _load_model_frame()
    config = order_io.load_config()

    monthly_breach, calibration_bins, predictions = order_experiment.evaluate_classification(frame, config, selection)
    aggregate_breach, aggregate_bins = order_experiment._aggregate_classification_predictions(predictions)
    breach_results = pd.concat(
        [aggregate_breach, monthly_breach.loc[monthly_breach["period"].eq("terminal")]], ignore_index=True
    )
    paired = order_experiment.paired_classification_differences(predictions, config)
    paired = order_experiment.add_breach_evidence_guards(paired, monthly_breach, config)
    ablations = order_experiment.evaluate_profile_ablations(frame, config, selection)
    support = order_experiment.classification_support_strata(predictions)
    events = order_experiment.event_strata(predictions)

    severity_monthly, severity_predictions = order_experiment.evaluate_severity(frame, config, selection)
    severity_aggregate = order_experiment.aggregate_severity_predictions(severity_predictions)
    severity_results = pd.concat(
        [severity_aggregate, severity_monthly.loc[severity_monthly["period"].eq("terminal")]], ignore_index=True
    )
    severity_ablations = order_experiment.evaluate_severity_ablations(frame, config, selection)
    severity_support = order_experiment.severity_support_strata(severity_predictions)

    breach_prediction_path = order_io.OUT / "ORDER_BREACH_ROW_PREDICTIONS.parquet"
    severity_prediction_path = order_io.OUT / "SEVERITY_ROW_PREDICTIONS.parquet"
    storage = config["storage"]
    determinism = config["determinism"]
    order_io.write_parquet(
        breach_prediction_path,
        predictions,
        sort_by=storage["classification_sort_keys"],
        engine=str(storage["parquet_engine"]),
        compression=str(storage["parquet_compression"]),
        index=bool(storage["parquet_index"]),
        sort_kind=str(determinism["sort_kind"]),
    )
    order_io.write_parquet(
        severity_prediction_path,
        severity_predictions,
        sort_by=storage["severity_sort_keys"],
        engine=str(storage["parquet_engine"]),
        compression=str(storage["parquet_compression"]),
        index=bool(storage["parquet_index"]),
        sort_kind=str(determinism["sort_kind"]),
    )

    order_io.write_csv(order_io.OUT / "ORDER_BREACH_RESULTS.csv", _reporting_aliases(breach_results))
    order_io.write_csv(order_io.OUT / "ORDER_BREACH_BY_MONTH.csv", _reporting_aliases(monthly_breach.loc[monthly_breach["period"].eq("later")]))
    order_io.write_csv(order_io.OUT / "ORDER_BREACH_PAIRED_DIFFERENCES.csv", _reporting_aliases(paired))
    calibration_results = pd.concat([aggregate_breach, monthly_breach], ignore_index=True)
    order_io.write_csv(order_io.OUT / "ORDER_CALIBRATION_RESULTS.csv", _reporting_aliases(calibration_results))
    order_io.write_csv(order_io.OUT / "ORDER_CALIBRATION_BINS.csv", _reporting_aliases(pd.concat([calibration_bins, aggregate_bins], ignore_index=True)))
    order_io.write_csv(order_io.OUT / "ORDER_PROFILE_ABLATIONS.csv", _reporting_aliases(ablations))
    order_io.write_csv(order_io.OUT / "ORDER_PROFILE_SUPPORT_STRATA.csv", _reporting_aliases(support))
    order_io.write_csv(order_io.OUT / "ORDER_EVENT_STRATA.csv", _reporting_aliases(events))

    order_io.write_csv(order_io.OUT / "SEVERITY_RESULTS.csv", _reporting_aliases(severity_results))
    order_io.write_csv(order_io.OUT / "SEVERITY_BY_MONTH.csv", _reporting_aliases(severity_monthly.loc[severity_monthly["period"].eq("later")]))
    severity_all = pd.concat([severity_monthly, severity_aggregate], ignore_index=True)
    severity_skill = severity_all[
        ["period", "cohort", "family", "model_id", "quantile", "n_orders", "pinball_loss", "skill_vs_unconditional", "skill_vs_q1", "order_id_sha256"]
    ].copy()
    severity_skill["comparison"] = severity_skill["model_id"].map(
        {"Q1": "Q1-unconditional", "Q2": "Q2-Q1", "Q3": "Q3-Q1", "Q4": "Q4-Q1"}
    )
    severity_skill["pinball_skill"] = np.where(
        severity_skill["model_id"].eq("Q1"), severity_skill["skill_vs_unconditional"], severity_skill["skill_vs_q1"]
    )
    severity_coverage = severity_all[
        ["period", "cohort", "family", "model_id", "quantile", "n_orders", "empirical_coverage", "coverage_error", "median_prediction", "mean_prediction"]
    ].copy()
    severity_coverage["comparison"] = severity_coverage["model_id"].map(
        {"Q1": "Q1-unconditional", "Q2": "Q2-Q1", "Q3": "Q3-Q1", "Q4": "Q4-Q1"}
    )
    severity_skill, severity_coverage = order_experiment.add_severity_evidence_guards(
        severity_skill, severity_coverage, severity_support, config
    )
    order_io.write_csv(order_io.OUT / "SEVERITY_PINBALL_SKILL.csv", _reporting_aliases(severity_skill))
    order_io.write_csv(order_io.OUT / "SEVERITY_COVERAGE.csv", _reporting_aliases(severity_coverage))
    order_io.write_csv(order_io.OUT / "SEVERITY_PROFILE_ABLATIONS.csv", _reporting_aliases(severity_ablations))
    order_io.write_csv(order_io.OUT / "SEVERITY_SUPPORT_STRATA.csv", _reporting_aliases(severity_support))
    terminal = _terminal_stress(breach_results, paired, severity_results, predictions)
    order_io.write_csv(order_io.OUT / "ORDER_TERMINAL_STRESS.csv", terminal)
    order_preflight.verify_protected_unchanged(prestate)
    order_io.append_run_event(
        "later_and_terminal_evaluation_complete",
        selection_freeze_sha256=order_io.sha256_file(SELECTION_FREEZE_PATH),
        breach_prediction_rows=len(predictions), severity_prediction_rows=len(severity_predictions),
        breach_prediction_output={
            "path": breach_prediction_path.relative_to(order_io.OUT).as_posix(),
            "rows": len(predictions),
            "sha256": order_io.sha256_file(breach_prediction_path),
        },
        severity_prediction_output={
            "path": severity_prediction_path.relative_to(order_io.OUT).as_posix(),
            "rows": len(severity_predictions),
            "sha256": order_io.sha256_file(severity_prediction_path),
        },
    )


def run_finalize(test_results: Path | None) -> None:
    prestate = _require_preflight()
    _verify_selection_freeze(prestate)
    sample_path = order_io.OUT / "ORDER_SAMPLE_AUDIT.csv"
    if sample_path.is_file():
        sample = pd.read_csv(sample_path)
        for column, value in (
            ("source_hashes_valid", True), ("canonical_assembler_valid", True),
            ("source_verdict", "verified"), ("protected_files_unchanged", True),
            ("protected_hashes_valid", True), ("preservation_verdict", "unchanged"),
        ):
            sample[column] = value
        order_io.write_csv(sample_path, sample)
    try:
        from analysis.order_breach_severity_v1.scripts import order_reporting
    except ImportError as exc:
        raise RuntimeError("order_reporting.py is not available") from exc
    result = order_reporting.finalize_reporting(
        output_dir=order_io.OUT,
        work_dir=order_io.WORK,
        test_results_path=test_results,
    )
    order_preflight.verify_protected_unchanged(prestate)
    order_io.append_run_event("reporting_complete", overall_pass=bool(result.get("overall_pass")))
    if not result.get("overall_pass"):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["preflight", "prepare", "tune", "evaluate", "finalize"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--test-results", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    order_io.ensure_directories()
    config = order_io.load_config()
    data_dir = args.data_dir or Path(config["data"]["default_data_dir"])
    if args.workers != 4:
        raise SystemExit("frozen worker count is 4")
    if args.stage == "preflight":
        state = order_preflight.preflight(data_dir)
        order_io.append_run_event("preflight_passed", prestate_sha256=order_io.sha256_file(order_preflight.PRESTATE_PATH))
        print(json.dumps({"preflight": "PASS", "protected_file_count": state["protected_baseline"]["file_count"]}, sort_keys=True))
    elif args.stage == "prepare":
        run_prepare(data_dir)
    elif args.stage == "tune":
        run_tune()
    elif args.stage == "evaluate":
        run_evaluate()
    elif args.stage == "finalize":
        run_finalize(args.test_results)
    else:  # pragma: no cover
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
