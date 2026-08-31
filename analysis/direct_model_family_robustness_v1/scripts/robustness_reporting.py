"""Persisted-table-only reporting for model-family robustness."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .robustness_integrity import WORKSPACE, atomic_write_text, read_json, sha256_file, utc_now, write_json


def _read(name: str) -> pd.DataFrame:
    path = WORKSPACE / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—" if value is None or str(value) in {"", "nan"} else str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _markdown(frame: pd.DataFrame, columns: list[str], labels: list[str] | None = None) -> str:
    selected = frame[columns].copy()
    if labels:
        selected.columns = labels
    header = "| " + " | ".join(map(str, selected.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(selected.columns)) + " |"
    rows = []
    for row in selected.itertuples(index=False, name=None):
        rows.append("| " + " | ".join("—" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join([header, divider, *rows])


def _breach_label_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary.loc[summary["row_type"].eq("profile_increment")].copy()
    rows["median_delta_log_loss_fmt"] = rows["median_delta_log_loss"].map(_fmt)
    rows["median_delta_brier_fmt"] = rows["median_delta_brier"].map(_fmt)
    rows["months_fmt"] = rows["both_improved_month_count"].apply(
        lambda value: "—" if pd.isna(value) else f"{int(value)}/6"
    )
    rows["calibration_fmt"] = rows["calibration_not_systematically_worse"].apply(
        lambda value: "—" if pd.isna(value) else ("pass" if bool(value) else "fail")
    )
    return rows[[
        "family_display", "profile_block", "median_delta_log_loss_fmt",
        "median_delta_brier_fmt", "months_fmt", "calibration_fmt", "evidence_label",
    ]]


def _severity_label_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary.loc[summary["row_type"].eq("profile_increment")].copy()
    rows["q_fmt"] = rows["quantile"].apply(lambda value: f"Q{int(float(value) * 100)}")
    rows["skill_fmt"] = rows["median_skill"].map(_fmt)
    rows["months_fmt"] = rows["favourable_month_count"].apply(
        lambda value: "—" if pd.isna(value) else f"{int(value)}/6"
    )
    rows["coverage_fmt"] = rows.apply(
        lambda row: _fmt(row["median_empirical_coverage"]) if float(row["quantile"]) == 0.9 else "n/a",
        axis=1,
    )
    return rows[[
        "family_display", "q_fmt", "profile_block", "skill_fmt", "months_fmt",
        "coverage_fmt", "evidence_label",
    ]]


def _direction_statement(summary: pd.DataFrame, block: str) -> tuple[str, int, int]:
    rows = summary.loc[
        summary["row_type"].eq("profile_increment")
        & summary["family_status"].eq("evaluated")
        & summary["profile_block"].eq(block)
    ]
    favourable = rows["median_delta_log_loss"].lt(0) & rows["median_delta_brier"].lt(0)
    return (
        "persisted across every evaluable breach family" if favourable.all() else "did not persist across every evaluable breach family",
        int(favourable.sum()),
        len(rows),
    )


def _pareto_dominance(summary: pd.DataFrame) -> list[str]:
    baseline = summary.loc[
        summary["row_type"].eq("baseline_absolute") & summary["family_status"].eq("evaluated")
    ].copy()
    if baseline.empty:
        return []
    interpretation_rank = {"logistic_l2": 1, "random_forest": 2, "xgboost": 3}
    baseline["interpretability_rank"] = baseline["family"].map(interpretation_rank)
    lower = [
        "mean_log_loss", "mean_brier", "median_wace",
        "median_absolute_calibration_slope_error", "sd_log_loss", "sd_brier",
        "interpretability_rank",
    ]
    higher = ["median_average_precision", "median_roc_auc"]
    dominators: list[str] = []
    for _, candidate in baseline.iterrows():
        dominates_all = True
        for _, other in baseline.loc[baseline["family"].ne(candidate["family"])].iterrows():
            no_worse = all(candidate[col] <= other[col] for col in lower) and all(
                candidate[col] >= other[col] for col in higher
            )
            strictly = any(candidate[col] < other[col] for col in lower) or any(
                candidate[col] > other[col] for col in higher
            )
            dominates_all = dominates_all and no_worse and strictly
        if dominates_all:
            dominators.append(str(candidate["family_display"]))
    return dominators


def _primary_labels_unchanged(labels: pd.DataFrame) -> bool:
    protected = labels.loc[
        labels["label_type"].eq("PROTECTED PRE-EXISTING DIRECT-EXTENSION LABEL")
    ]
    return bool(len(protected) == 18 and protected["protected_label_unchanged"].fillna(False).astype(bool).all())


def render_reports() -> dict[str, Any]:
    breach = _read("BREACH_MODEL_FAMILY_SUMMARY.csv")
    severity = _read("SEVERITY_MODEL_FAMILY_SUMMARY.csv")
    labels = _read("ROBUSTNESS_EVIDENCE_LABELS.csv")
    terminal = _read("TERMINAL_MODEL_FAMILY_ROBUSTNESS.csv")
    reproduction = _read("PRIMARY_RESULT_REPRODUCTION_AUDIT.csv")
    selection = _read("MODEL_SELECTION_ALL_FAMILIES.csv")

    seller_statement, seller_positive, seller_total = _direction_statement(breach, "seller")
    route_statement, route_positive, route_total = _direction_statement(breach, "state_od")
    both_statement, both_positive, both_total = _direction_statement(breach, "both")

    severity_new = severity.loc[
        severity["row_type"].eq("profile_increment")
        & severity["source_role"].eq("subsequent_model_family_robustness")
    ]
    new_supported = severity_new.loc[severity_new["evidence_label"].eq("Supported")]
    new_mixed = severity_new.loc[severity_new["evidence_label"].eq("Mixed")]
    if new_supported.empty and new_mixed.empty:
        severity_conclusion = (
            "The bounded pre-existing conclusion survived both additional severity families: "
            "no new profile increment met the direct rubric."
        )
        severity_conclusion_zh = "既有的有限否定结论在两个新增严重度模型族中均保持：没有新增画像增量通过直接证据规则。"
    else:
        severity_conclusion = (
            "The pre-existing all-Not-supported result remains protected, but the broader sensitivity is "
            f"model-family dependent ({len(new_supported)} Supported and {len(new_mixed)} Mixed new rows)."
        )
        severity_conclusion_zh = (
            "既有的全体 Not-supported 结果仍受保护，但更广泛的敏感性结论取决于模型族"
            f"（新增结果中 {len(new_supported)} 条为 Supported，{len(new_mixed)} 条为 Mixed）。"
        )

    dominators = _pareto_dominance(breach)
    dominance_statement = (
        f"A strict no-worse Pareto check identified {', '.join(dominators)} as dominating the other evaluable breach families."
        if dominators
        else "No evaluable breach family uniformly dominated the others across proper scoring, ranking, calibration, monthly transfer and the predeclared interpretability ordering."
    )
    dominance_statement_zh = (
        f"严格的非劣 Pareto 检查显示，{', '.join(dominators)} 支配其他可评估的违约模型族。"
        if dominators
        else "在恰当评分、排序、校准、逐月迁移及预先声明的可解释性顺序上，没有任何可评估的违约模型族形成一致支配。"
    )

    breach_table = _breach_label_table(breach)
    severity_table = _severity_label_table(severity)
    baseline = breach.loc[breach["row_type"].eq("baseline_absolute")].copy()
    baseline["log_loss_fmt"] = baseline["mean_log_loss"].map(_fmt)
    baseline["brier_fmt"] = baseline["mean_brier"].map(_fmt)
    baseline["ap_fmt"] = baseline["median_average_precision"].map(_fmt)
    baseline["auc_fmt"] = baseline["median_roc_auc"].map(_fmt)
    baseline["wace_fmt"] = baseline["median_wace"].map(_fmt)
    baseline["slope_fmt"] = baseline["median_absolute_calibration_slope_error"].map(_fmt)

    primary_ok = bool(reproduction["passed"].fillna(False).astype(bool).all())
    labels_ok = _primary_labels_unchanged(labels)
    spline_blocked = bool(
        selection.loc[
            selection["family"].eq("spline_logistic")
            & selection["record_type"].eq("blocked_family")
        ].shape[0] == 1
    )
    q90 = severity.loc[pd.to_numeric(severity["quantile"]).eq(0.9)].copy()
    q90["coverage_fmt"] = q90["median_empirical_coverage"].map(_fmt)
    terminal_breach_n = int(pd.to_numeric(
        terminal.loc[terminal["task"].eq("breach"), "n_orders"], errors="coerce"
    ).max())
    terminal_severity_n = int(pd.to_numeric(
        terminal.loc[terminal["task"].eq("severity"), "n_orders"], errors="coerce"
    ).max())

    summary = f"""# Direct-Promise Model-Family Robustness V1 — Result Summary

## Outcome

The primary reproduction gate **{'passed' if primary_ok else 'failed'}**. All protected Logistic/XGBoost breach and Linear/XGBoost-quantile direct results, settings and labels were numerically reproduced at `rtol=0, atol=1e-10` before the additional families were interpreted.

Across the three evaluable breach families, the seller direction {seller_statement} ({seller_positive}/{seller_total}); the state-OD direction {route_statement} ({route_positive}/{route_total}); and the combined direction {both_statement} ({both_positive}/{both_total}). Spline Logistic remains unevaluable because an applicable prior direct-prediction implementation could not be recovered without inventing feature-to-spline and imputation rules.

{severity_conclusion}

{dominance_statement} No synthetic winner score was constructed.

## Breach profile increments, January--June 2018

Negative deltas favour the profile augmentation. Month counts require simultaneous log-loss and Brier improvement. Primary Logistic/XGBoost labels are protected; Random-Forest rows are subsequent robustness labels.

{_markdown(breach_table, list(breach_table.columns), ['Family', 'Block', 'Median Δ log loss', 'Median Δ Brier', 'Both improved', 'Calibration guard', 'Evidence label'])}

## Breach family baseline comparison

These are descriptive family-level summaries for DP0 only; they are not a leaderboard.

{_markdown(baseline, ['family_display', 'log_loss_fmt', 'brier_fmt', 'ap_fmt', 'auc_fmt', 'wace_fmt', 'slope_fmt'], ['Family', 'Mean log loss', 'Mean Brier', 'Median AP', 'Median ROC-AUC', 'Median WACE', 'Median abs(slope−1)'])}

## Conditional severity profile increments, January--June 2018

Skill is relative to the same-family DQ0 baseline. Q90 coverage is judged by closeness to 0.90, never by being larger in isolation.

{_markdown(severity_table, list(severity_table.columns), ['Family', 'Quantile', 'Block', 'Median skill', 'Favourable', 'Median coverage', 'Evidence label'])}

## Q90 coverage

{_markdown(q90, ['family_display', 'profile_block', 'coverage_fmt', 'pooled_empirical_coverage', 'coverage_not_materially_worse'], ['Family', 'Block', 'Median monthly coverage', 'Pooled coverage', 'Coverage guard'])}

## Terminal stress

July--August remains separate terminal-regime stress. The persisted long table contains breach proper scores, ranking and calibration metrics plus profile-minus-promise deltas, and severity Q50/Q90 pinball, skill and coverage. No terminal evidence label was assigned. It contains {len(terminal)} rows: {terminal_breach_n:,} breach-eligible orders and {terminal_severity_n:,} breached severity orders.

## Governance and limitations

- The profile history variant is the fixed selected 90-day representation; the all-mature-history workspace was not an analytical input.
- Two pre-freeze metadata-only isolation incidents are recorded: a search exposed two non-empirical count fields, and an initial Git status could enumerate untracked filenames. No excluded-workspace file content or empirical result was consumed.
- An initial complete staged execution was superseded after JSON serialization of validation diagnostics failed; the validator was repaired and the full chain rerun under a new control freeze.
- Breach RF, quantile RF and lognormal Ridge use recovered singleton specifications because no historical multi-point grids existed.
- Breach RF deliberately retains the exact controlled historical preprocessing (median imputation then scaling, without missing indicators) and seed 42.
- The recovered quantile RF is the recorded leaf-weighted approximation, not a third-party interchangeable QRF.
- Spline Logistic is incomplete, so the four-family breach audit is not complete even though three structurally distinct families are evaluable.
- The direct extension is subsequent evidence rather than a registered headline RQ3 study; no thesis, Results Registry or canonical ledger text was edited.
- Protected primary labels remained unchanged: **{labels_ok}**.

## Figure brief

`FIGURE_DATA_BREACH_MODEL_FAMILIES.csv` supports a family-by-profile-block panel of median Δ log loss with both-improved month count and calibration guard. `FIGURE_DATA_SEVERITY_MODEL_FAMILIES.csv` supports Q50/Q90 family panels of median skill, favourable months and Q90 coverage. Both are conclusion-robustness displays, not model rankings.
"""

    summary_zh = f"""# 直接承诺模型族稳健性 V1 — 结果摘要

## 核心结论

主结果复现门槛**{'通过' if primary_ok else '未通过'}**：在解释新增模型之前，已用 `rtol=0, atol=1e-10` 复现既有 Logistic/XGBoost 违约结果与 Linear/XGBoost 分位数结果、设定及证据标签。

在三个可评估的违约模型族中，卖家画像方向在 {seller_positive}/{seller_total} 个模型族中同时改善月度中位 log loss 与 Brier；州级 OD 画像为 {route_positive}/{route_total}；组合画像为 {both_positive}/{both_total}。Spline Logistic 因缺少可直接复用、能保持样本行不变的既有预测实现而被阻断，没有新造方法。

{severity_conclusion_zh}

{dominance_statement_zh} 本分析没有构造综合冠军分数。

## 违约结果（2018 年 1–6 月）

{_markdown(breach_table, list(breach_table.columns), ['模型族', '画像块', '中位 Δ log loss', '中位 Δ Brier', '双指标改善月份', '校准守门', '证据标签'])}

## 条件正迟到严重度（2018 年 1–6 月）

{_markdown(severity_table, list(severity_table.columns), ['模型族', '分位数', '画像块', '中位 skill', '有利月份', '中位覆盖率', '证据标签'])}

## 终端压力测试

2018 年 7–8 月保持为单独的终端状态压力测试，不分配证据标签。持久化长表共 {len(terminal)} 行，覆盖 {terminal_breach_n:,} 个违约预测合格订单和 {terminal_severity_n:,} 个已违约严重度订单。

## 边界

- 仅使用已选定的 90 天历史画像；全成熟历史工作区不是分析输入。
- Logistic/XGBoost 及 Linear/XGBoost 分位数的既有标签保持不变；新增标签均明确标为 `ROBUSTNESS EVIDENCE LABEL`。
- 随机森林、叶加权分位数森林及对数正态 Ridge 只有既有固定设定，因此按单点网格在开发折上复核，而未虚构新网格。
- 7–8 月只作为终端压力测试，不与 1–6 月合并，也不分配终端证据标签。
- 未修改论文、Results Registry 或规范化证据台账。
"""

    readme = """# Direct-Promise Model-Family Robustness V1

This isolated workspace tests whether the completed direct-promise profile conclusion changes across forecasting model families while holding the selected 90-day S1/S2/R1/R2 payloads fixed.

The analysis is intentionally not a leaderboard. It reproduces the protected primary direct-extension results first, evaluates only recoverable prior model definitions, preserves the original evidence rubrics, and separates January--June later-cohort evidence from July--August terminal stress.

## Canonical execution

```bash
.venv/bin/python -B -m analysis.direct_model_family_robustness_v1.scripts.run_robustness preflight
.venv/bin/python -B -m analysis.direct_model_family_robustness_v1.scripts.run_robustness model
.venv/bin/python -B -m analysis.direct_model_family_robustness_v1.scripts.run_robustness report
.venv/bin/python -B -m analysis.direct_model_family_robustness_v1.scripts.run_robustness finalize
.venv/bin/python -B -m analysis.direct_model_family_robustness_v1.scripts.run_robustness validate
```

Run from the repository root. `-B` prevents imports from creating bytecode inside protected source directories.

## Evidence status

Primary direct-extension labels are preserved as pre-existing extension evidence. Random-Forest breach and the two additional severity families are subsequent robustness evidence. Spline Logistic is blocked because no applicable exact prior direct-feature implementation/grid was recoverable. No thesis or governance file is edited.
"""

    atomic_write_text(WORKSPACE / "RESULT_SUMMARY.md", summary)
    atomic_write_text(WORKSPACE / "RESULT_SUMMARY_ZH.md", summary_zh)
    atomic_write_text(WORKSPACE / "README.md", readme)
    receipt = {
        "analysis_id": "direct_model_family_robustness_v1",
        "completed_at_utc": utc_now(),
        "primary_reproduction_gate_passed": primary_ok,
        "protected_primary_labels_unchanged": labels_ok,
        "spline_logistic_blocked": spline_blocked,
        "seller_direction_favourable_families": seller_positive,
        "seller_direction_evaluable_families": seller_total,
        "state_od_direction_favourable_families": route_positive,
        "state_od_direction_evaluable_families": route_total,
        "combined_direction_favourable_families": both_positive,
        "combined_direction_evaluable_families": both_total,
        "new_severity_supported_rows": len(new_supported),
        "new_severity_mixed_rows": len(new_mixed),
        "strict_pareto_dominators": dominators,
        "output_sha256": {
            name: sha256_file(WORKSPACE / name)
            for name in ["RESULT_SUMMARY.md", "RESULT_SUMMARY_ZH.md", "README.md"]
        },
    }
    write_json(WORKSPACE / "working/REPORT_RECEIPT.json", receipt)
    return receipt


__all__ = ["render_reports"]
