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


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _summary_value(summary: pd.DataFrame, code: str, period: str, horizon: int, metric: str) -> pd.Series:
    rows = summary.loc[
        summary["profile_code"].eq(code)
        & summary["period"].eq(period)
        & pd.to_numeric(summary["horizon_days"], errors="coerce").eq(horizon)
        & summary["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise AssertionError(f"missing summary row: {code}/{period}/{horizon}/{metric}")
    return rows.iloc[0]


def _profile_lines(summary: pd.DataFrame, period: str, horizon: int) -> list[str]:
    lines: list[str] = []
    for code in ("S1", "S2", "R1", "R2"):
        metric = "log_loss" if code in ("S2", "R2") else "log_mae"
        row = _summary_value(summary, code, period, horizon, metric)
        rank = _summary_value(summary, code, period, horizon, "weighted_spearman")
        lines.append(
            f"- {code}: {metric} {_fmt(row['selected_90d_value'])} → "
            f"{_fmt(row['all_mature_value'])} (difference of period medians "
            f"{_fmt(row['aggregate_median_difference'])}; median paired-anchor Δ "
            f"{_fmt(row['all_mature_minus_90d'])}); weighted Spearman "
            f"{_fmt(rank['selected_90d_value'])} → {_fmt(rank['all_mature_value'])} "
            f"(difference of period medians {_fmt(rank['aggregate_median_difference'])}; "
            f"median paired-anchor Δ {_fmt(rank['all_mature_minus_90d'])}); "
            f"{row['practical_equivalence_assessment']}."
        )
    return lines


def _profile_lines_zh(summary: pd.DataFrame, period: str, horizon: int) -> list[str]:
    lines: list[str] = []
    for code in ("S1", "S2", "R1", "R2"):
        metric = "log_loss" if code in ("S2", "R2") else "log_mae"
        row = _summary_value(summary, code, period, horizon, metric)
        rank = _summary_value(summary, code, period, horizon, "weighted_spearman")
        assessment = (
            "在冻结实用等价阈值内（仅作描述）"
            if str(row["practical_equivalence_assessment"]).startswith("within")
            else "超出冻结实用等价阈值（仅报告数值，不判定胜者）"
        )
        lines.append(
            f"- {code}：{metric} 从 {_fmt(row['selected_90d_value'])} 变为 "
            f"{_fmt(row['all_mature_value'])}（周期中位数之差 "
            f"{_fmt(row['aggregate_median_difference'])}；逐锚点配对差的中位数 "
            f"{_fmt(row['all_mature_minus_90d'])}）；加权 Spearman 从 "
            f"{_fmt(rank['selected_90d_value'])} 变为 {_fmt(rank['all_mature_value'])} "
            f"（周期中位数之差 {_fmt(rank['aggregate_median_difference'])}；逐锚点配对差的中位数 "
            f"{_fmt(rank['all_mature_minus_90d'])}）；{assessment}。"
        )
    return lines


def _tradeoff_table(support: pd.DataFrame, period: str, horizon: int = 7) -> list[str]:
    rows: list[str] = []
    metrics = ("future_seen_coverage", "support_qualified_coverage", "cold_start_share_all_placed", "support_median")
    for code in ("S1", "S2", "R1", "R2"):
        part = support.loc[
            support["profile_code"].eq(code)
            & support["period"].eq(period)
            & pd.to_numeric(support["horizon_days"], errors="coerce").eq(horizon)
            & support["population"].eq("future_orders")
            & support["metric"].isin(metrics)
        ]
        med = part.groupby("metric", sort=True).agg(
            selected_90d_value=("selected_90d_value", "median"),
            all_mature_value=("all_mature_value", "median"),
        )
        rows.append(
            f"- {code}: future-seen coverage {_fmt(med.loc['future_seen_coverage','selected_90d_value'])} → "
            f"{_fmt(med.loc['future_seen_coverage','all_mature_value'])}; support≥5 coverage "
            f"{_fmt(med.loc['support_qualified_coverage','selected_90d_value'])} → "
            f"{_fmt(med.loc['support_qualified_coverage','all_mature_value'])}; cold-start share "
            f"{_fmt(med.loc['cold_start_share_all_placed','selected_90d_value'])} → "
            f"{_fmt(med.loc['cold_start_share_all_placed','all_mature_value'])}; mapped-order median support "
            f"{_fmt(med.loc['support_median','selected_90d_value'], 1)} → "
            f"{_fmt(med.loc['support_median','all_mature_value'], 1)}. "
            "Each value is the median of weekly anchor-level statistics."
        )
    return rows


def _stability_lines(uncertainty: pd.DataFrame, period: str) -> list[str]:
    lines: list[str] = []
    for code in ("S1", "S2", "R1", "R2"):
        block = uncertainty.loc[
            uncertainty["profile_code"].eq(code)
            & uncertainty["period"].eq(period)
            & uncertainty["component"].eq("daily_stability")
        ].set_index("metric")
        spearman = block.loc["day_to_day_spearman"]
        churn = block.loc["pct_entities_changing_level"]
        lines.append(
            f"- {code}: median adjacent-day Spearman {_fmt(spearman['selected_90d_value'])} → "
            f"{_fmt(spearman['all_mature_value'])}; median communication-tier change share "
            f"{_fmt(churn['selected_90d_value'])} → {_fmt(churn['all_mature_value'])}."
        )
    return lines


def _substantive_interpretation(summary: pd.DataFrame) -> str:
    headline = summary.loc[
        summary["period"].isin(["development", "confirmation"])
        & pd.to_numeric(summary["horizon_days"], errors="coerce").eq(7)
        & (
            ((summary["target_kind"].eq("binary")) & summary["metric"].eq("log_loss"))
            | ((summary["target_kind"].eq("continuous")) & summary["metric"].eq("log_mae"))
        )
    ]
    paired_favourable = 0
    paired_adverse = 0
    aggregate_favourable = 0
    aggregate_adverse = 0
    for _, row in headline.iterrows():
        delta = float(row["all_mature_minus_90d"])
        if delta < -1e-12:
            paired_favourable += 1
        elif delta > 1e-12:
            paired_adverse += 1
        aggregate_delta = float(row["aggregate_median_difference"])
        if aggregate_delta < -1e-12:
            aggregate_favourable += 1
        elif aggregate_delta > 1e-12:
            aggregate_adverse += 1
    equivalent = int(
        headline["practical_equivalence_assessment"].astype(str).str.startswith("within").sum()
    )
    return (
        f"Across the eight development/confirmation 7-day primary-loss comparisons, "
        f"the median paired-anchor loss difference was favourable in {paired_favourable} and "
        f"adverse in {paired_adverse}; the separately aggregated period medians were favourable "
        f"in {aggregate_favourable} and adverse in {aggregate_adverse}; "
        f"{equivalent} profile-period comparisons met the reused frozen practical-"
        f"equivalence tolerances. This pattern is sensitivity evidence about adaptation "
        f"versus cumulative support; it does not reopen the selected 90-day definitions."
    )


def _direct_breach_lines(frame: pd.DataFrame) -> list[str]:
    labels = {"DPS": "seller", "DPG": "state-OD", "DPB": "combined"}
    medians = frame.loc[
        frame["row_type"].eq("monthly_median")
        & frame["population"].eq("all_orders")
    ]
    pooled = frame.loc[
        frame["row_type"].eq("pooled")
        & frame["population"].eq("all_orders")
    ]
    lines: list[str] = []
    for family in ("logistic_l2", "xgboost"):
        for model_id in ("DPS", "DPG", "DPB"):
            for reference in ("promise_only", "selected_90d"):
                row = medians.loc[
                    medians["family"].eq(family)
                    & medians["model_id"].eq(model_id)
                    & medians["reference_kind"].eq(reference)
                ]
                pool = pooled.loc[
                    pooled["family"].eq(family)
                    & pooled["model_id"].eq(model_id)
                    & pooled["reference_kind"].eq(reference)
                ]
                if len(row) != 1 or len(pool) != 1:
                    raise AssertionError(
                        f"missing direct breach comparison: {family}/{model_id}/{reference}"
                    )
                row = row.iloc[0]
                pool = pool.iloc[0]
                reference_text = "promise-only baseline" if reference == "promise_only" else "corresponding selected 90-day specification"
                lines.append(
                    f"- {family}, {labels[model_id]} versus {reference_text}: monthly-median "
                    f"Δ log loss {_fmt(row['delta_log_loss'], 5)}, Δ Brier "
                    f"{_fmt(row['delta_brier'], 5)}; both improved in "
                    f"{int(row['favourable_month_count'])}/6 months. Pooled Δ log loss "
                    f"{_fmt(pool['delta_log_loss'], 5)}, Δ Brier {_fmt(pool['delta_brier'], 5)}."
                )
    return lines


def _direct_severity_lines(frame: pd.DataFrame) -> list[str]:
    labels = {"DQS": "seller", "DQG": "state-OD", "DQB": "combined"}
    medians = frame.loc[
        frame["row_type"].eq("monthly_median")
        & frame["population"].eq("all_orders")
    ]
    pooled = frame.loc[
        frame["row_type"].eq("pooled")
        & frame["population"].eq("all_orders")
    ]
    lines: list[str] = []
    for family in ("linear_quantile", "xgboost_quantile"):
        for quantile in (0.5, 0.9):
            for model_id in ("DQS", "DQG", "DQB"):
                for reference in ("promise_only", "selected_90d"):
                    row = medians.loc[
                        medians["family"].eq(family)
                        & pd.to_numeric(medians["quantile"], errors="coerce").eq(quantile)
                        & medians["model_id"].eq(model_id)
                        & medians["reference_kind"].eq(reference)
                    ]
                    pool = pooled.loc[
                        pooled["family"].eq(family)
                        & pd.to_numeric(pooled["quantile"], errors="coerce").eq(quantile)
                        & pooled["model_id"].eq(model_id)
                        & pooled["reference_kind"].eq(reference)
                    ]
                    if len(row) != 1 or len(pool) != 1:
                        raise AssertionError(
                            f"missing direct severity comparison: {family}/{quantile}/{model_id}/{reference}"
                        )
                    row = row.iloc[0]
                    pool = pool.iloc[0]
                    reference_text = "promise-only baseline" if reference == "promise_only" else "corresponding selected 90-day specification"
                    lines.append(
                        f"- {family} Q{int(quantile * 100)}, {labels[model_id]} versus "
                        f"{reference_text}: monthly-median pinball skill "
                        f"{_fmt(row['all_mature_pinball_skill_vs_reference'], 5)}, favourable in "
                        f"{int(row['favourable_month_count'])}/6 months; pooled skill "
                        f"{_fmt(pool['all_mature_pinball_skill_vs_reference'], 5)}"
                        + (
                            f", pooled coverage {_fmt(pool['all_mature_empirical_coverage'])} "
                            f"(absolute error {_fmt(pool['all_mature_absolute_coverage_error'])})"
                            if quantile == 0.9 else ""
                        )
                        + "."
                    )
    return lines


def generate_reports() -> dict[str, object]:
    summary = pd.read_csv(OUT / "STANDALONE_90D_VS_ALL_MATURE_SUMMARY.csv", low_memory=False)
    support = pd.read_csv(OUT / "SUPPORT_COVERAGE_COLDSTART_COMPARISON.csv", low_memory=False)
    uncertainty = pd.read_csv(OUT / "UNCERTAINTY_STABILITY_COMPARISON.csv", low_memory=False)
    match = pd.read_csv(OUT / "PROFILE_MATCH_AUDIT.csv", low_memory=False)
    terminal = pd.read_csv(OUT / "TERMINAL_PROFILE_SENSITIVITY.csv", low_memory=False)
    gate = json.loads((WORK / "DIRECT_EXTENSION_GATE.json").read_text(encoding="utf-8"))
    direct_run = gate.get("available") is True
    if direct_run:
        breach_direct = pd.read_csv(
            OUT / "DIRECT_BREACH_90D_VS_ALL_MATURE.csv", low_memory=False
        )
        severity_direct = pd.read_csv(
            OUT / "DIRECT_SEVERITY_90D_VS_ALL_MATURE.csv", low_memory=False
        )
        calibration_direct = pd.read_csv(
            OUT / "DIRECT_ALL_MATURE_CALIBRATION_COVERAGE.csv", low_memory=False
        )
        terminal_direct = pd.read_csv(
            OUT / "DIRECT_ALL_MATURE_TERMINAL.csv", low_memory=False
        )
        gate_text = (
            "The direct order-level branch was run because the frozen preflight gate found a "
            f"completed, validation-PASS and integrity-PASS direct extension (manifest sequence "
            f"`{gate.get('manifest_sequence')}`). It reused the issued-promise-only baseline, "
            "monthly origins, preprocessing, hyperparameters, calibrators, quantiles and seeds; "
            "only the four profile payloads were replaced."
        )
        direct_sections = [
            "## January–June direct breach sensitivity",
            "",
            *_direct_breach_lines(breach_direct),
            "",
            "Negative loss deltas favour all-mature. These are numeric sensitivity comparisons "
            "only; no direct 90-day evidence label was recomputed.",
            "",
            "## January–June direct conditional-severity sensitivity",
            "",
            *_direct_severity_lines(severity_direct),
            "",
            "Severity is positive calendar-day lateness among breached orders. Calibration, Q90 "
            f"coverage and all-order/high-support Q50/Q90 comparisons are persisted in "
            f"`DIRECT_ALL_MATURE_CALIBRATION_COVERAGE.csv` ({len(calibration_direct):,} rows) "
            "and `DIRECT_SEVERITY_90D_VS_ALL_MATURE.csv`.",
            "",
            "## Direct terminal stress",
            "",
            f"July–August breach and conditional-severity sensitivity is kept separately in "
            f"`DIRECT_ALL_MATURE_TERMINAL.csv` ({len(terminal_direct):,} rows); it is not included "
            "in January–June medians, favourable-month counts or pooled results.",
            "",
        ]
    else:
        gate_text = (
            f"The direct order-level branch was skipped. At the frozen preflight gate the direct "
            f"extension had manifest status `{gate.get('manifest_status')}` and did not satisfy "
            "the completed validation/integrity contract."
        )
        direct_sections = []
    interpretation = _substantive_interpretation(summary)
    report = [
        "# Result summary",
        "",
        "## Scope and gate",
        "",
        "The analysis changed only the lower history bound for frozen S1, S2, R1 and R2. "
        "It did not rerun the candidate search, alter thresholds, assign sensitivity confirmation "
        "labels, or replace the selected 90-day profiles.",
        "",
        gate_text,
        "",
        "## Matched-future audit",
        "",
        f"All {len(match):,} profile-anchor-horizon comparisons used exactly matched future "
        f"orders, entity mappings, target-observation flags, target values and valid-outcome IDs. "
        f"The largest absolute difference when reproducing the persisted 90-day anchor metrics was "
        f"{_fmt(pd.to_numeric(match['persisted_90d_max_absolute_difference'], errors='coerce').max(), 12)}.",
        "",
        "## Development standalone comparison — 7-day horizon",
        "",
        *_profile_lines(summary, "development", 7),
        "",
        "## January–June standalone comparison — 7-day horizon",
        "",
        *_profile_lines(summary, "confirmation", 7),
        "",
        "The corresponding 30-day results are persisted in the summary, monthly and anchor tables; "
        "they are kept separate from the primary 7-day horizon.",
        "",
        "## January–June support, coverage and cold start — 7-day horizon, median across weekly anchors",
        "",
        *_tradeoff_table(support, "confirmation", 7),
        "",
        "## January–June daily stability",
        "",
        *_stability_lines(uncertainty, "confirmation"),
        "",
        "Uncertainty, interval width, freshness, support-stratum and full communication-tier "
        "transition comparisons are in `UNCERTAINTY_STABILITY_COMPARISON.csv`.",
        "",
        "Matched support≥5 future-loss and separation robustness is in "
        "`SUPPORT_GE5_ROBUSTNESS_COMPARISON.csv`; every comparison uses the intersection of "
        "orders meeting support≥5 under both histories.",
        "",
        *direct_sections,
        "## Terminal sensitivity",
        "",
        f"July–August terminal evidence is reported separately in "
        f"`TERMINAL_PROFILE_SENSITIVITY.csv` ({len(terminal):,} rows) and was not pooled with "
        f"January–June evidence or used to alter construction.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "The reused practical-equivalence check is based on separately aggregated period "
        "medians: log loss plus Brier for binary profiles, and log-MAE plus weighted Spearman "
        "for continuous profiles. It is descriptive only.",
        "",
        "The 90-day profile remains a bounded recent operational-state representation. The "
        "all-mature profile is a cumulative long-run historical-state representation. Higher "
        "support or lower uncertainty for all-mature does not by itself establish better future "
        "state tracking; future loss/separation and temporal adaptation must be read together.",
        "",
        "The original S1/S2/R1/R2 selection and confirmation labels remain unchanged. No thesis, "
        "Results Registry, ledger, Abstract, Discussion or Conclusion was edited.",
    ]
    (OUT / "RESULT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    report_zh = [
        "# 结果摘要",
        "",
        "## 范围与条件门",
        "",
        "本敏感性分析仅删除冻结 S1、S2、R1、R2 的历史下界；实体、过程目标、成熟规则、滞后、"
        "估计器、收缩强度、最低支持规则、阈值、快照与未来窗口均不变。没有重跑候选搜索，"
        "没有更改确认标签，也没有用全成熟版本替换已选 90 天版本。",
        "",
        (
            "订单层直接敏感性分支已运行：冻结预检确认直接承诺扩展已完成，验证与完整性均为 PASS。"
            "本分支仅替换四个画像字段组，承诺基线、月度起点、预处理、超参数、分位数与随机种子均保持不变。"
            if direct_run else
            f"订单层分支未运行；冻结预检时直接扩展状态为 `{gate.get('manifest_status')}`。"
        ),
        "",
        "## 未来样本匹配审计",
        "",
        f"共 {len(match):,} 个“画像—锚点—窗口”比较全部使用完全相同的未来订单、实体映射、"
        "目标观测标志、目标值和有效结果订单。对既有 90 天锚点指标的最大绝对复现差异为 "
        f"{_fmt(pd.to_numeric(match['persisted_90d_max_absolute_difference'], errors='coerce').max(), 12)}。",
        "",
        "## 开发期独立画像比较——7 天窗口",
        "",
        *_profile_lines_zh(summary, "development", 7),
        "",
        "## 1–6 月独立画像比较——7 天窗口",
        "",
        *_profile_lines_zh(summary, "confirmation", 7),
        "",
        "30 天窗口结果已分别保存在锚点、月度和汇总表中，不与主要 7 天窗口混合。",
        "",
        *(
            [
                "## 订单层敏感性",
                "",
                "1–6 月违约概率与条件正迟到 Q50/Q90 的逐月、中位数、有效月份数和合并结果已分别保存在 "
                "`DIRECT_BREACH_90D_VS_ALL_MATURE.csv` 与 "
                "`DIRECT_SEVERITY_90D_VS_ALL_MATURE.csv`。校准、Q90 覆盖率和高支持交集结果见 "
                "`DIRECT_ALL_MATURE_CALIBRATION_COVERAGE.csv`；Q50/Q90 高支持配对结果也保留在"
                "严重度比较表中；7–8 月终端压力结果单独保存。"
                "这些数值不会重算或替换原直接 90 天证据标签。",
                "",
            ]
            if direct_run else []
        ),
        "## 解释边界",
        "",
        "90 天画像表示有界的近期运营状态；全成熟画像表示累积的长期历史状态。全成熟版本若支持度"
        "更高或不确定性更低，并不自动说明其未来状态刻画更好，必须与未来损失、分离能力和时间适应"
        "共同解读。本结果仅为敏感性证据，不重开原始 90 天画像选择。",
        "",
        "终端期证据单独保存在 `TERMINAL_PROFILE_SENSITIVITY.csv`，没有与 1–6 月合并。原有"
        "S1/S2/R1/R2 选择和确认标签保持不变；论文、结果注册表、证据台账、摘要、讨论与结论均未修改。",
    ]
    (OUT / "RESULT_SUMMARY_ZH.md").write_text("\n".join(report_zh) + "\n", encoding="utf-8")

    process_spec = """# Figure brief — Chapter 3 process observability

## Purpose

Explain which parts of a generic e-commerce fulfilment process the public Olist extract observes, which operational subcomponents remain latent, and how mature prior outcomes can become bounded seller-associated or state-OD historical information.

## Reader-facing flow

`purchase / observed customer-facing promise` → `payment approval` → `carrier handoff` → `customer delivery`

Place `review (selected and variably timed)` on a separate side branch connected to the overall purchase experience, not on the physical fulfilment arrow. Some recorded reviews precede customer delivery.

Use solid nodes/arrows for recorded timestamps and a dashed enclosing band for unobserved internal preparation, packing, platform release, warehouse activity, carrier pickup waiting, hubs and carrier paths. Show the payment-approval-to-handoff interval as **seller-associated pre-handoff fulfilment duration**, never pure seller processing. Show handoff-to-delivery as transit duration, and the main-seller-state-to-customer-state key as a **state-OD geographic transit proxy**, never an observed route.

Below the process, add a strict as-of gate: an earlier outcome contributes only after its target-specific `label_available_at < snapshot`. Mature prior pre-handoff outcomes feed seller profiles; mature prior transit outcomes feed state-OD profiles. Current-order post-purchase timestamps and reviews do not feed ex-ante predictors.

## Visual rules

- Keep the observed black-box promise separate from any claim about how Olist generated it.
- Distinguish observed timestamp, derived interval, deterministic geographic proxy and unobserved mechanism by shape/line style.
- Do not depict review as a physical fulfilment stage or imply causal entity quality.
- Add a footnote that the public extract's platform sampling frame is undocumented.
"""
    (OUT / "FIGURE_SPEC_CH3_PROCESS_OBSERVABILITY.md").write_text(process_spec, encoding="utf-8")

    method_spec = """# Figure brief — Chapter 4 method flow

## Purpose

Show the evidence sequence from strictly mature prior outcomes to standalone profile validation and later order-level use, while positioning all-mature history as a sensitivity rather than a replacement candidate search.

## Reader-facing flow

`strictly mature prior outcomes` → `entity + process target` → `history representation` → `estimation` → `standalone future validation` → `development selection + immutable freeze` → `January–June confirmation` → `validated process profiles`

From the validated profiles, show two clearly separated order-level lanes:

- main current RQ3 lane: `recorded promise + purchase-time current context + profile block` → `breach probability and conditional positive-lateness Q50/Q90` → `later monthly transfer + separate terminal stress`;
- secondary direct sensitivity lane: `recorded promise + profile block` → `direct breach/Q50/Q90 sensitivity`, including the selected-90-day versus all-mature replacement check.

Split **history representation** into two visually unequal branches:

- main candidate search: 30 / 60 / 90 days under frozen Scheme A or C;
- post-selection sensitivity: selected 90 days versus all mature outcomes available strictly as of the same snapshot.

At **estimation**, show raw, shrinkage and case-mix-adjusted families as candidate families, while noting that the four promoted process representatives retain their exact frozen estimator. The all-mature sensitivity changes only the lower history bound and does not rerun the 3,024-candidate search, thresholds, promotion or confirmation labels.

## Visual rules

- Keep internal P0/P1/P2/M0/M1/Q1 labels out of the central reader flow.
- Use a lock icon at selection freeze and a side branch for the all-mature sensitivity.
- Mark future cohorts by purchase date and outcomes by later target-specific maturity.
- Label January–June as locked future-process confirmation / later-cohort evaluation, not external or untouched validation.
- Keep July–August terminal stress separate from headline aggregation.
"""
    (OUT / "FIGURE_SPEC_CH4_METHOD_FLOW.md").write_text(method_spec, encoding="utf-8")

    comparison_spec = """# Figure brief — selected 90-day versus all-mature history

## Purpose

Compare a bounded recent operational-state representation with a cumulative long-run historical-state representation without implying reselection or a categorical winner.

## Recommended design

Use a four-row small-multiple dumbbell or paired-dot figure (S1, S2, R1, R2). Columns should separate development and January–June confirmation; facet the 7-day primary horizon from the 30-day secondary horizon. Plot paired values for the primary proper loss and weighted Spearman. Because lower loss and higher rank association have opposite directions, use separate panels or transform only the displayed loss difference to a clearly labelled favourable-direction scale; never combine them into a composite.

Add a support panel with future-seen coverage, support≥5 coverage and cold-start share, and a stability panel with adjacent-day Spearman and communication-tier change share. Keep July–August in a separate terminal panel.

## Data

- `FIGURE_DATA_90D_VS_ALL_MATURE.csv`: compact development/confirmation profile metrics.
- `SUPPORT_COVERAGE_COLDSTART_COMPARISON.csv`: anchor-level support and cold-start trade-offs.
- `UNCERTAINTY_STABILITY_COMPARISON.csv`: uncertainty, freshness, adjacent-day stability and tier transitions.
- `TERMINAL_PROFILE_SENSITIVITY.csv`: separate terminal results.

## Annotation rules

- Label deltas literally as `all-mature minus selected 90-day`.
- Label the displayed profile values as medians. Distinguish the **median paired-anchor difference** from the **difference of separately aggregated medians**; do not place one beside arrows representing the other.
- State the favourable direction next to every metric.
- If the frozen practical-equivalence tolerances are shown, call them descriptive reused tolerances, not new confirmation criteria.
- Retain the original S1/S2/R1/R2 labels; do not assign all-mature confirmation labels.
- Caption: “Sensitivity only; the selected 90-day profile definitions were not reopened.”
"""
    (OUT / "FIGURE_SPEC_90D_VS_ALL_MATURE.md").write_text(comparison_spec, encoding="utf-8")

    readme = f"""# All-mature-history sensitivity V1

This isolated workspace compares the four frozen selected 90-day process profiles with otherwise identical cumulative all-mature counterparts. The selected 90-day profiles remain the main specifications. No candidate search, threshold selection, promotion, confirmation label, order model or thesis text was changed.

{("The direct order-level branch was run because its frozen preflight gate was satisfied. The six required direct sensitivity outputs compare all-mature seller, state-OD and combined blocks against both the issued-promise-only baseline and their corresponding selected-90-day specifications." if direct_run else "The direct order-level branch was skipped because its frozen preflight gate was not satisfied; the six conditional outputs are intentionally absent.")}

Core result tables are the anchor, monthly and period summaries; matched-future evidence is audited in `PROFILE_MATCH_AUDIT.csv`; support and stability trade-offs have dedicated tables; July–August is separate. `ALL_MATURE_PROFILE_DAILY_SCORES.csv.gz` is the reconstructed 636-snapshot store, with a compact index and parent store.

See `EXACT_HISTORY_DEFINITIONS.md`, `RESULT_SUMMARY.md`, `RESULT_SUMMARY_ZH.md` and the final `RUN_MANIFEST.json` for definitions, interpretation and integrity receipts.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    return {
        "matched_comparisons": len(match),
        "summary_rows": len(summary),
        "support_rows": len(support),
        "uncertainty_stability_rows": len(uncertainty),
        "terminal_rows": len(terminal),
        "direct_branch_skipped": not direct_run,
        "substantive_interpretation": interpretation,
    }
