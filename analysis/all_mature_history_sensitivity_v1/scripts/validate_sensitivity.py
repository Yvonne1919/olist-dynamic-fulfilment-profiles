from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/all_mature_history_sensitivity_v1"
WORK = OUT / "working"

REQUIRED = [
    "RUN_MANIFEST.json",
    "README.md",
    "EXACT_HISTORY_DEFINITIONS.md",
    "PROFILE_MATCH_AUDIT.csv",
    "STANDALONE_90D_VS_ALL_MATURE_ANCHOR.csv",
    "STANDALONE_90D_VS_ALL_MATURE_MONTHLY.csv",
    "STANDALONE_90D_VS_ALL_MATURE_SUMMARY.csv",
    "SUPPORT_COVERAGE_COLDSTART_COMPARISON.csv",
    "SUPPORT_GE5_ROBUSTNESS_COMPARISON.csv",
    "UNCERTAINTY_STABILITY_COMPARISON.csv",
    "TERMINAL_PROFILE_SENSITIVITY.csv",
    "RESULT_SUMMARY.md",
    "RESULT_SUMMARY_ZH.md",
    "FIGURE_SPEC_CH3_PROCESS_OBSERVABILITY.md",
    "FIGURE_SPEC_CH4_METHOD_FLOW.md",
    "FIGURE_SPEC_90D_VS_ALL_MATURE.md",
    "FIGURE_DATA_90D_VS_ALL_MATURE.csv",
    "ALL_MATURE_PROFILE_DAILY_SCORES.csv.gz",
    "ALL_MATURE_PROFILE_PARENT_STRUCTURE.csv.gz",
    "ALL_MATURE_PROFILE_STORE_INDEX.csv",
]

DIRECT_CONDITIONAL = [
    "DIRECT_BREACH_ALL_MATURE_MONTHLY.csv",
    "DIRECT_BREACH_90D_VS_ALL_MATURE.csv",
    "DIRECT_SEVERITY_ALL_MATURE_MONTHLY.csv",
    "DIRECT_SEVERITY_90D_VS_ALL_MATURE.csv",
    "DIRECT_ALL_MATURE_CALIBRATION_COVERAGE.csv",
    "DIRECT_ALL_MATURE_TERMINAL.csv",
]


def _direct_delta_contract(
    table: pd.DataFrame,
    *,
    grouping: list[str],
) -> tuple[bool, dict[str, object]]:
    """Verify literal paired deltas and the distinct monthly-median contract."""
    provenance = {
        "monthly": "paired_same_month_identical_orders",
        "monthly_median": "median_of_paired_monthly_differences",
        "pooled": "recomputed_on_concatenated_monthly_predictions",
    }
    provenance_ok = all(
        table.loc[table["row_type"].eq(row_type), "difference_aggregation"]
        .astype(str)
        .eq(label)
        .all()
        for row_type, label in provenance.items()
    )
    delta_columns = sorted(column for column in table.columns if column.startswith("delta_"))
    literal_ok = True
    for delta_column in delta_columns:
        metric = delta_column.removeprefix("delta_")
        candidate_column = f"all_mature_{metric}"
        reference_column = f"reference_{metric}"
        if candidate_column not in table or reference_column not in table:
            literal_ok = False
            continue
        literal = table["row_type"].isin(["monthly", "pooled"])
        observed = pd.to_numeric(table.loc[literal, delta_column], errors="coerce")
        expected = (
            pd.to_numeric(table.loc[literal, candidate_column], errors="coerce")
            - pd.to_numeric(table.loc[literal, reference_column], errors="coerce")
        )
        literal_ok &= bool(np.isclose(
            observed, expected, rtol=0, atol=2e-10, equal_nan=True
        ).all())

    median_ok = True
    monthly = table.loc[table["row_type"].eq("monthly")]
    medians = table.loc[table["row_type"].eq("monthly_median")]
    for _, row in medians.iterrows():
        mask = pd.Series(True, index=monthly.index)
        for column in grouping:
            if column == "quantile":
                mask &= np.isclose(
                    pd.to_numeric(monthly[column], errors="coerce"),
                    float(row[column]),
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                )
            else:
                mask &= monthly[column].astype(str).eq(str(row[column]))
        paired_months = monthly.loc[mask]
        median_ok &= len(paired_months) == 6
        for delta_column in delta_columns:
            expected = pd.to_numeric(
                paired_months[delta_column], errors="coerce"
            ).median()
            observed = pd.to_numeric(pd.Series([row[delta_column]]), errors="coerce").iloc[0]
            median_ok &= bool(np.isclose(
                observed, expected, rtol=0, atol=2e-10, equal_nan=True
            ))
    return provenance_ok and literal_ok and median_ok, {
        "provenance_ok": provenance_ok,
        "literal_monthly_and_pooled_ok": literal_ok,
        "median_of_six_paired_monthly_deltas_ok": median_ok,
        "delta_columns": delta_columns,
    }


def validate() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for name in REQUIRED:
        check(f"required_output:{name}", (OUT / name).is_file())
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    if missing:
        raise AssertionError(f"missing required outputs: {missing}")

    gate = json.loads((WORK / "DIRECT_EXTENSION_GATE.json").read_text(encoding="utf-8"))
    direct_available = gate.get("available") is True
    check("direct_gate_is_boolean", isinstance(gate.get("available"), bool), gate.get("available"))
    if direct_available:
        missing_direct = [name for name in DIRECT_CONDITIONAL if not (OUT / name).is_file()]
        check("conditional_direct_outputs_present_when_gate_true", not missing_direct, missing_direct)
        if missing_direct:
            raise AssertionError(f"missing conditional direct sensitivity outputs: {missing_direct}")
    else:
        unexpected_direct = [name for name in DIRECT_CONDITIONAL if (OUT / name).exists()]
        check("conditional_direct_outputs_absent_when_gate_false", not unexpected_direct, unexpected_direct)
        if unexpected_direct:
            raise AssertionError(f"unexpected direct sensitivity outputs: {unexpected_direct}")

    match = pd.read_csv(OUT / "PROFILE_MATCH_AUDIT.csv", low_memory=False)
    check("match_audit_row_count", len(match) == 528, len(match))
    check("four_profile_codes", set(match["profile_code"].astype(str)) == {"S1", "S2", "R1", "R2"})
    check("all_future_evidence_exactly_matched", match["exact_future_evidence_match"].astype(bool).all())
    check("all_unmatched_counts_zero", not pd.to_numeric(
        match[[
            "unmatched_90d_rows", "unmatched_all_mature_rows", "entity_mapping_mismatch_rows",
            "target_observed_mismatch_rows", "target_value_mismatch_rows",
            "raw_target_value_mismatch_rows", "label_available_at_mismatch_rows",
            "valid_future_outcome_id_symmetric_difference",
        ]].stack(), errors="coerce",
    ).fillna(0).ne(0).any())
    check("persisted_90d_reproduction_pass", match["persisted_90d_reproduction_pass"].astype(bool).all())
    check(
        "persisted_90d_reproduction_tolerance",
        pd.to_numeric(match["persisted_90d_max_absolute_difference"], errors="coerce").max() <= 1e-9,
        pd.to_numeric(match["persisted_90d_max_absolute_difference"], errors="coerce").max(),
    )

    anchor = pd.read_csv(OUT / "STANDALONE_90D_VS_ALL_MATURE_ANCHOR.csv", low_memory=False)
    expected_counts = {
        ("development", 7): 39,
        ("development", 30): 36,
        ("confirmation", 7): 25,
        ("confirmation", 30): 21,
        ("terminal", 7): 7,
        ("terminal", 30): 4,
    }
    observed = match.groupby(["profile_code", "period", "horizon_days"])["anchor_date"].nunique()
    anchors_ok = all(
        int(observed.loc[(code, period, horizon)]) == count
        for code in ("S1", "S2", "R1", "R2")
        for (period, horizon), count in expected_counts.items()
    )
    check(
        "exact_frozen_anchor_counts",
        anchors_ok,
        {"|".join(map(str, key)): int(value) for key, value in observed.to_dict().items()},
    )
    arithmetic = np.isclose(
        pd.to_numeric(anchor["all_mature_minus_90d"], errors="coerce"),
        pd.to_numeric(anchor["all_mature_value"], errors="coerce")
        - pd.to_numeric(anchor["selected_90d_value"], errors="coerce"),
        rtol=0, atol=1e-10, equal_nan=True,
    )
    check("literal_all_mature_minus_90d_arithmetic", bool(np.all(arithmetic)))
    check("no_replacement_confirmation_label", not any(
        "confirmation_label" in column.lower() for column in anchor.columns
    ))

    summary = pd.read_csv(OUT / "STANDALONE_90D_VS_ALL_MATURE_SUMMARY.csv", low_memory=False)
    check("summary_has_all_periods", set(summary["period"].astype(str)) == {"development", "confirmation", "terminal"})
    check("summary_has_both_horizons", set(pd.to_numeric(summary["horizon_days"], errors="coerce")) == {7, 30})
    allowed_assessments = {
        "within_frozen_tolerances_descriptive_only",
        "outside_frozen_tolerances_numeric_only",
    }
    check(
        "equivalence_language_is_descriptive_only",
        set(summary["practical_equivalence_assessment"].astype(str)).issubset(allowed_assessments),
        sorted(set(summary["practical_equivalence_assessment"].astype(str))),
    )
    aggregate_arithmetic = np.isclose(
        pd.to_numeric(summary["aggregate_median_difference"], errors="coerce"),
        pd.to_numeric(summary["all_mature_value"], errors="coerce")
        - pd.to_numeric(summary["selected_90d_value"], errors="coerce"),
        rtol=0, atol=1e-10, equal_nan=True,
    )
    check("summary_difference_of_medians_arithmetic", bool(np.all(aggregate_arithmetic)))
    check(
        "paired_delta_provenance_explicit",
        summary["all_mature_minus_90d_aggregation"].astype(str).eq(
            "median_of_paired_anchor_differences"
        ).all(),
    )
    check(
        "aggregate_delta_provenance_explicit",
        summary["aggregate_median_difference_aggregation"].astype(str).eq(
            "difference_of_separately_aggregated_medians"
        ).all(),
    )
    headline_equivalence = summary.loc[
        summary["period"].isin(["development", "confirmation"])
        & pd.to_numeric(summary["horizon_days"], errors="coerce").eq(7)
        & (
            (summary["target_kind"].eq("binary") & summary["metric"].eq("log_loss"))
            | (summary["target_kind"].eq("continuous") & summary["metric"].eq("log_mae"))
        )
    ]
    check(
        "frozen_equivalence_formula_reproduces_three_of_eight",
        len(headline_equivalence) == 8
        and int(headline_equivalence["practical_equivalence_assessment"].astype(str).str.startswith("within").sum()) == 3,
        headline_equivalence[["profile_code", "period", "practical_equivalence_assessment"]].to_dict("records"),
    )

    index = pd.read_csv(OUT / "ALL_MATURE_PROFILE_STORE_INDEX.csv")
    check("profile_store_four_specs", len(index) == 4 and set(index["profile_code"].astype(str)) == {"S1", "S2", "R1", "R2"})
    check("profile_store_636_snapshots_each", pd.to_numeric(index["snapshot_count"], errors="coerce").eq(636).all())
    check("profile_store_nonempty", pd.to_numeric(index["row_count"], errors="coerce").gt(0).all())
    check("profile_store_hash_single_consistent", index["store_sha256"].nunique() == 1)

    construction = pd.read_csv(WORK / "ALL_MATURE_CONSTRUCTION_AUDIT.csv", low_memory=False)
    check("construction_audit_4x636", len(construction) == 4 * 636, len(construction))
    check(
        "strict_availability_zero_violations",
        pd.to_numeric(construction["strict_availability_violations"], errors="coerce").fillna(0).eq(0).all(),
    )
    check(
        "scheme_c_purchase_cutoff_zero_violations",
        pd.to_numeric(construction["scheme_c_purchase_cutoff_violations"], errors="coerce").fillna(0).eq(0).all(),
    )
    support_robustness = pd.read_csv(OUT / "SUPPORT_GE5_ROBUSTNESS_COMPARISON.csv", low_memory=False)
    check("support_ge5_robustness_nonempty", not support_robustness.empty, len(support_robustness))
    check(
        "support_ge5_uses_exact_common_orders",
        support_robustness["exact_common_order_match"].astype(bool).all(),
    )
    check(
        "support_ge5_threshold_is_frozen_five",
        pd.to_numeric(support_robustness["support_threshold"], errors="coerce").eq(5).all(),
    )

    summary_text = (OUT / "RESULT_SUMMARY.md").read_text(encoding="utf-8")
    check("summary_states_selection_not_reopened", "did not rerun the candidate search" in summary_text)
    if direct_available:
        check("summary_states_direct_branch_run", "direct order-level branch was run" in summary_text)
    else:
        check("summary_states_direct_skip", "direct order-level branch was skipped" in summary_text)
    check("summary_distinguishes_representations", "bounded recent operational-state representation" in summary_text and "cumulative long-run historical-state representation" in summary_text)

    if direct_available:
        breach_monthly = pd.read_csv(OUT / "DIRECT_BREACH_ALL_MATURE_MONTHLY.csv", low_memory=False)
        severity_monthly = pd.read_csv(OUT / "DIRECT_SEVERITY_ALL_MATURE_MONTHLY.csv", low_memory=False)
        breach_compare = pd.read_csv(OUT / "DIRECT_BREACH_90D_VS_ALL_MATURE.csv", low_memory=False)
        severity_compare = pd.read_csv(OUT / "DIRECT_SEVERITY_90D_VS_ALL_MATURE.csv", low_memory=False)
        terminal_direct = pd.read_csv(OUT / "DIRECT_ALL_MATURE_TERMINAL.csv", low_memory=False)
        check("direct_breach_monthly_exact_96_rows", len(breach_monthly) == 96, len(breach_monthly))
        check("direct_severity_monthly_exact_96_rows", len(severity_monthly) == 96, len(severity_monthly))
        check(
            "direct_breach_comparison_counts",
            breach_compare["row_type"].value_counts().to_dict()
            == {"monthly": 144, "monthly_median": 24, "pooled": 24},
            breach_compare["row_type"].value_counts().to_dict(),
        )
        check(
            "direct_severity_comparison_counts",
            severity_compare["row_type"].value_counts().to_dict()
            == {"monthly": 288, "monthly_median": 48, "pooled": 48},
            severity_compare["row_type"].value_counts().to_dict(),
        )
        expected_breach_populations = {
            "all_orders": 96,
            "all_mature_support_ge20": 48,
            "common_support_ge20": 48,
        }
        expected_severity_populations = {
            "all_orders": 192,
            "all_mature_support_ge20": 96,
            "common_support_ge20": 96,
        }
        check(
            "direct_breach_high_support_rows_public",
            breach_compare["population"].value_counts().to_dict()
            == expected_breach_populations,
            breach_compare["population"].value_counts().to_dict(),
        )
        check(
            "direct_severity_q50_q90_high_support_rows_public",
            severity_compare["population"].value_counts().to_dict()
            == expected_severity_populations
            and severity_compare.loc[
                severity_compare["population"].ne("all_orders")
            ].groupby(["population", "quantile"], observed=True).size().to_dict()
            == {
                ("all_mature_support_ge20", 0.5): 48,
                ("all_mature_support_ge20", 0.9): 48,
                ("common_support_ge20", 0.5): 48,
                ("common_support_ge20", 0.9): 48,
            },
            {
                "|".join(map(str, key)): int(value)
                for key, value in severity_compare.loc[
                    severity_compare["population"].ne("all_orders")
                ].groupby(["population", "quantile"], observed=True).size().to_dict().items()
            },
        )
        for label, table in (("breach", breach_compare), ("severity", severity_compare)):
            support_reference_contract = (
                table.loc[table["population"].eq("all_mature_support_ge20"), "reference_kind"]
                .astype(str).eq("promise_only").all()
                and table.loc[table["population"].eq("common_support_ge20"), "reference_kind"]
                .astype(str).eq("selected_90d").all()
            )
            check(
                f"direct_{label}_high_support_reference_contract",
                support_reference_contract,
            )
        breach_delta_ok, breach_delta_detail = _direct_delta_contract(
            breach_compare,
            grouping=["family", "model_id", "reference_kind", "population"],
        )
        severity_delta_ok, severity_delta_detail = _direct_delta_contract(
            severity_compare,
            grouping=[
                "family", "quantile", "model_id", "reference_kind", "population",
            ],
        )
        check("direct_breach_delta_arithmetic_and_provenance", breach_delta_ok, breach_delta_detail)
        check("direct_severity_delta_arithmetic_and_provenance", severity_delta_ok, severity_delta_detail)
        terminal_delta_ok = bool(np.isclose(
            pd.to_numeric(terminal_direct["all_mature_minus_reference"], errors="coerce"),
            pd.to_numeric(terminal_direct["all_mature_value"], errors="coerce")
            - pd.to_numeric(terminal_direct["reference_value"], errors="coerce"),
            rtol=0,
            atol=2e-10,
            equal_nan=True,
        ).all())
        terminal_provenance_ok = terminal_direct["difference_aggregation"].astype(str).eq(
            "paired_same_terminal_orders"
        ).all()
        check(
            "direct_terminal_delta_arithmetic_and_provenance",
            terminal_delta_ok and terminal_provenance_ok,
            {
                "literal_candidate_minus_reference": terminal_delta_ok,
                "paired_terminal_provenance": terminal_provenance_ok,
            },
        )
        for label, table in (("breach", breach_compare), ("severity", severity_compare)):
            check(
                f"direct_{label}_reference_kind_preserved",
                set(table["reference_kind"].dropna().astype(str))
                == {"promise_only", "selected_90d"}
                and table["reference_kind"].notna().all(),
            )
            check(
                f"direct_{label}_reference_model_preserved",
                table["reference_model"].notna().all(),
            )
        check(
            "no_new_direct_evidence_labels",
            not any("evidence_label" in column or "evidence_status" in column for column in [*breach_compare.columns, *severity_compare.columns]),
        )
        reproduction = json.loads((WORK / "DIRECT_BASELINE_REPRODUCTION_AUDIT.json").read_text(encoding="utf-8"))
        check(
            "dp0_predictions_reproduced",
            reproduction["dp0_serialised_predictions_exact"] is True
            and reproduction["dp0_model_hashes_exact"] is True,
            reproduction,
        )
        check(
            "dq0_predictions_reproduced",
            reproduction["dq0_serialised_predictions_exact"] is True
            and reproduction["dq0_model_hashes_exact"] is True,
            reproduction,
        )
        pairing = pd.read_csv(WORK / "DIRECT_ALL_MATURE_PAIRING_AUDIT.csv", low_memory=False)
        check("direct_pairing_all_exact", pairing["exact_pair"].astype(bool).all())
        check("direct_pairing_zero_unmatched", pd.to_numeric(pairing["unmatched_rows"], errors="coerce").eq(0).all())
        check("direct_pairing_zero_target_mismatch", pd.to_numeric(pairing["target_mismatches"], errors="coerce").eq(0).all())
        check("direct_pairing_zero_origin_mismatch", pd.to_numeric(pairing["origin_mismatches"], errors="coerce").eq(0).all())
        population = json.loads((WORK / "DIRECT_COHORT_POPULATION_AUDIT.json").read_text(encoding="utf-8"))
        check("direct_frozen_cohort_population_audit", population.get("passed") is True, population)
        terminal_all = terminal_direct.loc[terminal_direct["population"].eq("all_orders")]
        breach_terminal_n = set(pd.to_numeric(terminal_all.loc[terminal_all["task"].eq("breach"), "n_orders"], errors="coerce").dropna().astype(int))
        severity_terminal_n = set(pd.to_numeric(terminal_all.loc[terminal_all["task"].eq("conditional_positive_lateness"), "n_orders"], errors="coerce").dropna().astype(int))
        check("direct_terminal_breach_exact_12507_orders", breach_terminal_n == {12507}, sorted(breach_terminal_n))
        check("direct_terminal_severity_exact_601_orders", severity_terminal_n == {601}, sorted(severity_terminal_n))

    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "status": "PASS" if passed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": sum(bool(item["passed"]) for item in checks),
        "checks": checks,
    }
    (OUT / "VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"{item['check']}: {'PASS' if item['passed'] else 'FAIL'}"
        + (f" — {item['detail']}" if item["detail"] not in ("", None) else "")
        for item in checks
    ]
    (OUT / "TEST_RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise AssertionError("artifact validation failed")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True, default=str))
