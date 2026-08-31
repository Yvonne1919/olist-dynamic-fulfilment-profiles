"""Deterministic compact reporting for the frozen order-level experiment.

This module consumes only compact analytical CSVs.  It never reconstructs
models or predictions.  Missing inputs/columns produce explicit blockers and
no-data receipts; they never produce invented metric values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FLOAT_FORMAT = "%.12g"
PNG_METADATA = {
    "Software": "order_breach_severity_v1",
    "Creation Time": "2026-08-23T00:00:00Z",
}

REQUIRED_COMPACT_CSVS = (
    "ORDER_SAMPLE_AUDIT.csv",
    "ORDER_PROFILE_JOIN_AUDIT.csv",
    "ORDER_DEVELOPMENT_TUNING.csv",
    "ORDER_MODEL_PARAMETERS.csv",
    "ORDER_BREACH_RESULTS.csv",
    "ORDER_BREACH_BY_MONTH.csv",
    "ORDER_BREACH_PAIRED_DIFFERENCES.csv",
    "ORDER_CALIBRATION_RESULTS.csv",
    "ORDER_CALIBRATION_BINS.csv",
    "ORDER_PROFILE_ABLATIONS.csv",
    "ORDER_PROFILE_SUPPORT_STRATA.csv",
    "ORDER_EVENT_STRATA.csv",
    "ORDER_TERMINAL_STRESS.csv",
    "SEVERITY_RESULTS.csv",
    "SEVERITY_BY_MONTH.csv",
    "SEVERITY_PINBALL_SKILL.csv",
    "SEVERITY_COVERAGE.csv",
    "SEVERITY_PROFILE_ABLATIONS.csv",
    "SEVERITY_SUPPORT_STRATA.csv",
)

REQUIRED_OUTPUTS = (
    "ORDER_PROTOCOL.md",
    "ORDER_FROZEN_CONFIG.json",
    "EVIDENCE_STATUS.md",
    "ORDER_FEATURE_DICTIONARY.md",
    *REQUIRED_COMPACT_CSVS,
    "ORDER_BREACH_ROW_PREDICTIONS.parquet",
    "SEVERITY_ROW_PREDICTIONS.parquet",
    "MODEL_COMPARISON_SUMMARY.csv",
    "ORDER_RESULTS_SUMMARY.md",
    "ORDER_RESULTS_SUMMARY_ZH.md",
    "BLOCKERS.md",
    "RUN_MANIFEST.json",
    "TEST_RESULTS.txt",
    "ARTIFACT_VALIDATION_REPORT.md",
)

MODEL_COMPARISON_COLUMNS = (
    "task",
    "family",
    "comparison",
    "probability_variant",
    "quantile",
    "n_months",
    "median_delta_log_loss",
    "median_delta_brier",
    "months_log_loss_improved",
    "months_brier_improved",
    "months_both_improved",
    "median_pinball_skill",
    "months_nonnegative_skill",
    "high_support_guard",
    "calibration_guard",
    "score_not_metadata_only_guard",
    "coverage_guard",
    "evidence_status",
    "evidence_reason",
    "source_file",
)


@dataclass(frozen=True)
class FigureSpec:
    number: int
    slug: str
    title: str
    source_files: tuple[str, ...]
    kind: str

    @property
    def stem(self) -> str:
        return f"{self.number:02d}_{self.slug}"


FIGURE_SPECS = (
    FigureSpec(1, "analytical_framework", "Frozen analytical framework", (), "framework"),
    FigureSpec(2, "model_ladder", "Frozen breach and severity model ladders", (), "ladder"),
    FigureSpec(3, "monthly_delta_log_loss", "Monthly paired delta log loss", ("ORDER_BREACH_PAIRED_DIFFERENCES.csv",), "delta_log_loss"),
    FigureSpec(4, "monthly_delta_brier", "Monthly paired delta Brier score", ("ORDER_BREACH_PAIRED_DIFFERENCES.csv",), "delta_brier"),
    FigureSpec(5, "ap_top10_comparison", "Average precision and top-decile lift", ("ORDER_BREACH_BY_MONTH.csv",), "ap_top10"),
    FigureSpec(6, "reliability", "Reliability: M1 versus profile-augmented models", ("ORDER_CALIBRATION_BINS.csv",), "reliability"),
    FigureSpec(7, "profile_block_ablation", "Score-only, metadata-only, and full profile blocks", ("ORDER_PROFILE_ABLATIONS.csv",), "ablation"),
    FigureSpec(8, "seller_increment", "Seller handling incremental contribution", ("ORDER_BREACH_PAIRED_DIFFERENCES.csv",), "seller"),
    FigureSpec(9, "route_increment", "Route transit incremental contribution", ("ORDER_BREACH_PAIRED_DIFFERENCES.csv",), "route"),
    FigureSpec(10, "combined_increment", "Combined seller and route contribution", ("ORDER_BREACH_PAIRED_DIFFERENCES.csv",), "combined"),
    FigureSpec(11, "q50_pinball_skill", "Q50 pinball skill by later cohort", ("SEVERITY_PINBALL_SKILL.csv",), "q50_skill"),
    FigureSpec(12, "q90_pinball_skill", "Q90 pinball skill by later cohort", ("SEVERITY_PINBALL_SKILL.csv",), "q90_skill"),
    FigureSpec(13, "q90_coverage", "Q90 empirical coverage by later cohort", ("SEVERITY_COVERAGE.csv",), "q90_coverage"),
    FigureSpec(14, "severity_profile_blocks", "Positive-severity results by profile block", ("SEVERITY_BY_MONTH.csv",), "severity_blocks"),
    FigureSpec(15, "bau_hrd_profile_gains", "BAU versus retrospective HRD profile gains", ("ORDER_EVENT_STRATA.csv",), "hrd"),
    FigureSpec(16, "later_vs_terminal", "January--June evidence versus terminal stress", ("ORDER_TERMINAL_STRESS.csv", "ORDER_BREACH_PAIRED_DIFFERENCES.csv"), "terminal"),
)

# The user's completion list contains 31 ordered items when the two compound
# provenance bullets are split into their constituent reports.
COMPLETION_REPORT_SECTIONS = (
    ("scripts", "Scripts executed", "执行的脚本"),
    ("commands", "Commands executed", "执行的命令"),
    ("source_verdict", "Source verdict", "源数据核验结论"),
    ("preservation_verdict", "Preservation verdict", "受保护文件保持结论"),
    ("evidence_status", "Evidence-status and non-untouched-holdout statement", "证据状态与非未触碰留出集声明"),
    ("sample_counts", "Analytical sample counts", "分析样本计数"),
    ("profile_join", "Profile join and cold-start coverage", "Profile 连接与冷启动覆盖"),
    ("feature_block", "Frozen current-order feature block", "冻结的当前订单特征块"),
    ("lr_tuning", "Logistic Regression tuning result", "逻辑回归调参结果"),
    ("boost_tuning", "Boosted-tree tuning result", "提升树调参结果"),
    ("m0_m1", "M0 to M1 breach result", "M0 到 M1 违约结果"),
    ("m1_m2", "M1 to M2 seller-handling incremental result", "M1 到 M2 卖家处理增量结果"),
    ("m1_m3", "M1 to M3 route-transit incremental result", "M1 到 M3 路线运输增量结果"),
    ("m1_m4", "M1 to M4 combined incremental result", "M1 到 M4 联合增量结果"),
    ("m4_m5", "M4 to M5 endpoint-profile incremental result", "M4 到 M5 终点 profile 增量结果"),
    ("m4_m4e", "M4 to M4E event-interaction result", "M4 到 M4E 事件交互结果"),
    ("calibration", "Calibration results", "概率校准结果"),
    ("profile_ablation", "Score-only versus metadata-only ablations", "仅分数与仅元数据消融"),
    ("support_cold", "High-support and cold-start results", "高支持度与冷启动结果"),
    ("q50", "Q50 severity result", "Q50 严重度结果"),
    ("q90", "Q90 severity result", "Q90 严重度结果"),
    ("coverage", "Quantile coverage", "分位数覆盖率"),
    ("seller_route_severity", "Seller versus route severity contribution", "卖家与路线严重度贡献"),
    ("bau_hrd", "BAU and HRD descriptive differences", "BAU 与 HRD 描述性差异"),
    ("terminal", "Terminal stress", "终端压力测试"),
    ("breach_labels", "Frozen breach-evidence labels", "冻结的违约证据标签"),
    ("severity_labels", "Frozen severity-evidence labels", "冻结的严重度证据标签"),
    ("blockers", "Blockers", "阻塞项"),
    ("tests", "Tests", "测试"),
    ("files", "Files created", "已创建文件"),
    ("scope_confirmation", "Scope confirmation", "范围确认"),
)

COLUMN_ALIASES = {
    "period": ("period", "evaluation_period", "split", "phase"),
    "cohort_month": ("cohort_month", "cohort", "calendar_month", "month", "test_month", "purchase_month"),
    "family": ("family", "model_family", "estimator_family"),
    "model_id": ("model_id", "model", "variant", "information_set", "ablation_id"),
    "probability_variant": ("probability_variant", "probability_type", "calibration_variant"),
    "comparison": ("comparison", "comparison_id", "contrast", "paired_comparison"),
    "candidate_model": ("candidate_model", "candidate", "model_candidate"),
    "reference_model": ("reference_model", "reference", "model_reference"),
    "metric": ("metric", "metric_name"),
    "estimate": ("estimate", "value", "metric_value"),
    "quantile": ("quantile", "tau", "quantile_level"),
    "representation": ("representation", "ablation", "ablation_id"),
    "stratum": ("stratum", "stratum_value", "regime", "support_stratum", "event_stratum"),
}

METRIC_ALIASES = {
    "delta_log_loss": ("delta_log_loss", "delta_ll", "log_loss_delta", "d_log_loss"),
    "delta_brier": ("delta_brier", "delta_brier_score", "brier_delta", "d_brier"),
    "average_precision": ("average_precision", "ap", "pr_auc", "precision_recall_auc"),
    "top10_lift": ("top10_lift", "top_10_lift", "top_decile_lift", "top_10pct_lift"),
    "pinball_skill": ("pinball_skill", "skill", "pinball_loss_skill", "skill_vs_q1"),
    "coverage": ("coverage", "empirical_coverage", "quantile_coverage"),
    "pinball_loss": ("pinball_loss", "quantile_loss"),
    "log_loss": ("log_loss", "cross_entropy"),
    "brier": ("brier", "brier_score"),
    "terminal_minus_later_calibration_intercept": ("terminal_minus_later_calibration_intercept",),
    "terminal_minus_later_calibration_slope": ("terminal_minus_later_calibration_slope",),
    "terminal_minus_later_wace": ("terminal_minus_later_wace",),
}

# Explicit, upstream-computed rule guards are carried through reshaping. The
# reporter deliberately does not infer these safeguards from unrelated metric
# columns: absence remains absence and therefore cannot yield ``Supported``.
GUARD_COLUMNS = (
    "high_support_ok",
    "high_support_no_material_reversal",
    "high_support_material_reversal",
    "calibration_not_systematically_worse",
    "calibration_ok",
    "calibration_systematically_worse",
    "benefit_not_metadata_only",
    "score_contributes",
    "benefit_metadata_only",
    "metadata_only_benefit",
    "coverage_not_materially_worse",
    "coverage_ok",
    "coverage_materially_worse",
    "coverage_deteriorated",
    "support_ge20_gain_present",
    "gain_only_low_support",
)

REQUESTED_COMPARISONS = (
    "M1-M0",
    "M2-M1",
    "M3-M1",
    "M4-M1",
    "M4-M2",
    "M4-M3",
    "M5-M4",
    "M4E-M4",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.rstrip() + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = handle.name
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str))


def _write_manifest_fixed_point(path: Path, manifest: dict[str, object]) -> None:
    """Persist a manifest whose self-reported byte size is exact (hash omitted)."""
    for _ in range(10):
        payload = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False, default=str
        ).rstrip() + "\n"
        size = len(payload.encode("utf-8"))
        changed = False
        for container_name in ("outputs", "artifact_inventory"):
            container = manifest.get(container_name)
            if container_name == "artifact_inventory" and isinstance(container, Mapping):
                container = container.get("required_outputs")
            if isinstance(container, dict) and "RUN_MANIFEST.json" in container:
                receipt = container["RUN_MANIFEST.json"]
                if isinstance(receipt, dict) and receipt.get("bytes") != size:
                    receipt["bytes"] = size
                    changed = True
        if not changed:
            _write_text(path, payload)
            if path.stat().st_size != size:
                raise AssertionError("RUN_MANIFEST self-size fixed point failed")
            return
    raise RuntimeError("RUN_MANIFEST self-size did not converge")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False, float_format=FLOAT_FORMAT, date_format="%Y-%m-%d", na_rep="")
        temporary = handle.name
    os.replace(temporary, path)


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lower_to_original = {str(column).lower(): column for column in result.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in result.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_to_original:
                result = result.rename(columns={lower_to_original[alias.lower()]: canonical})
                break
    if "comparison" not in result.columns and {"candidate_model", "reference_model"}.issubset(result.columns):
        result["comparison"] = (
            result["candidate_model"].astype(str) + "-" + result["reference_model"].astype(str)
        )
    return result


def _canonical_comparison(value: object) -> str:
    text = str(value).upper().replace(" ", "").replace("_MINUS_", "-").replace("_VS_", "-")
    text = text.replace("VS", "-").replace("→", "-").replace("_", "-")
    for comparison in REQUESTED_COMPARISONS:
        candidate, reference = comparison.split("-", 1)
        if candidate in text and reference in text:
            candidate_position = text.find(candidate)
            reference_position = text.find(reference)
            if candidate_position <= reference_position or "MINUS" in str(value).upper():
                return comparison
    return text


def _metric_long(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _standardize(frame)
    aliases = tuple(alias.lower() for alias in METRIC_ALIASES[metric])
    identifiers = [
        column
        for column in (
            "period",
            "cohort_month",
            "family",
            "model_id",
            "probability_variant",
            "comparison",
            "candidate_model",
            "reference_model",
            "quantile",
            "representation",
            "stratum",
            *GUARD_COLUMNS,
        )
        if column in data.columns
    ]
    if {"metric", "estimate"}.issubset(data.columns):
        names = data["metric"].astype(str).str.lower()
        selected = data.loc[names.isin(aliases), identifiers + ["estimate"]].copy()
        selected["metric"] = metric
        return selected
    lower = {str(column).lower(): column for column in data.columns}
    source_column = next((lower[alias] for alias in aliases if alias in lower), None)
    if source_column is None:
        return pd.DataFrame(columns=[*identifiers, "metric", "estimate"])
    selected = data.loc[:, identifiers + [source_column]].copy()
    selected = selected.rename(columns={source_column: "estimate"})
    selected["metric"] = metric
    return selected


def _later_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "period" not in frame.columns:
        return frame.iloc[0:0].copy()
    period = frame["period"].astype(str).str.lower()
    selected = period.str.contains("later|jan.jun|post.profile|confirmation", regex=True)
    selected &= ~period.str.contains("aggregate|pooled", regex=True)
    if "cohort_month" in frame.columns:
        cohort = frame["cohort_month"].astype(str).str.lower()
        selected &= ~cohort.str.contains("aggregate|pooled|monthly.median", regex=True)
    return frame.loc[selected].copy()


def _quantile_rows(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    if "quantile" not in frame.columns:
        return frame.iloc[0:0].copy()
    value = pd.to_numeric(frame["quantile"].astype(str).str.replace("Q", "", regex=False), errors="coerce")
    value = value.where(value.le(1), value / 100.0)
    return frame.loc[np.isclose(value, quantile, atol=1e-9, equal_nan=False)].copy()


def _receipt(spec: FigureSpec, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "no_data",
                "figure_id": spec.stem,
                "reason": reason,
                "required_input": "|".join(spec.source_files) if spec.source_files else "ORDER_FROZEN_CONFIG.json",
            }
        ]
    )


def _framework_source(config: Mapping[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "canonical_orders", "Canonical delivered orders", "", "population"),
            (2, "current_order", "Issued promise + current context", "canonical_orders", "purchase-time features"),
            (3, "daily_profiles", "Frozen daily seller/route profiles", "canonical_orders", "strictly as-of history"),
            (4, "breach", "Final promise breach probability", "current_order|daily_profiles", "part 1"),
            (5, "severity", "Positive lateness Q50/Q90", "current_order|daily_profiles", "part 2, breaches only"),
            (6, "later", "Jan--Jun later-cohort evaluation", "breach|severity", "primary chronological evidence"),
            (7, "terminal", "Jul--Aug terminal stress", "breach|severity", "stress test"),
        ],
        columns=["display_order", "node_id", "label", "parents", "role"],
    )


def _ladder_source(config: Mapping[str, object]) -> pd.DataFrame:
    rows = [
        ("breach", "M0", "", "issued promise only", "primary"),
        ("breach", "M1", "M0", "current ex-ante order/context", "primary"),
        ("breach", "M2", "M1", "seller handling block", "primary"),
        ("breach", "M3", "M1", "route transit block", "primary"),
        ("breach", "M4", "M2|M3", "seller + route blocks", "primary"),
        ("breach", "M5", "M4", "route historical final-breach", "secondary"),
        ("breach", "M4E", "M4", "two frozen profile x known-event interactions", "secondary"),
        ("severity", "Q1", "", "promise + current context", "primary"),
        ("severity", "Q2", "Q1", "seller handling block", "primary"),
        ("severity", "Q3", "Q1", "route transit block", "primary"),
        ("severity", "Q4", "Q2|Q3", "seller + route blocks", "primary"),
    ]
    return pd.DataFrame(rows, columns=["task", "model_id", "parent_model", "increment", "role"])


def _comparison_filter(frame: pd.DataFrame, comparisons: Sequence[str]) -> pd.DataFrame:
    if "comparison" not in frame.columns:
        return frame.iloc[0:0].copy()
    result = frame.copy()
    result["comparison"] = result["comparison"].map(_canonical_comparison)
    return result.loc[result["comparison"].isin(comparisons)].copy()


def _derive_model_deltas(
    frame: pd.DataFrame,
    comparisons: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    """Derive transparent candidate-minus-reference rows from compact scores."""
    data = _standardize(frame)
    if "model_id" not in data.columns:
        return pd.DataFrame()
    identifiers = [
        column
        for column in ("period", "cohort_month", "family", "probability_variant", "stratum", "quantile")
        if column in data.columns
    ]
    rows: list[pd.DataFrame] = []
    lower = {str(column).lower(): column for column in data.columns}
    for source_metric, output_metric in (("log_loss", "delta_log_loss"), ("brier", "delta_brier")):
        source_column = next(
            (lower[alias.lower()] for alias in METRIC_ALIASES[source_metric] if alias.lower() in lower),
            None,
        )
        if source_column is None:
            continue
        score = data.loc[:, [*identifiers, "model_id", source_column]].copy()
        score[source_column] = pd.to_numeric(score[source_column], errors="coerce")
        wide = score.pivot_table(
            index=identifiers,
            columns="model_id",
            values=source_column,
            aggfunc="mean",
            dropna=False,
        ).reset_index()
        for comparison, candidate, reference in comparisons:
            if candidate not in wide.columns or reference not in wide.columns:
                continue
            part = wide.loc[:, identifiers].copy()
            part["comparison"] = comparison
            part["metric"] = output_metric
            part["estimate"] = pd.to_numeric(wide[candidate], errors="coerce") - pd.to_numeric(
                wide[reference], errors="coerce"
            )
            rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _prepare_figure_source(
    spec: FigureSpec,
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, str | None]:
    if spec.kind == "framework":
        return _framework_source(config), None
    if spec.kind == "ladder":
        return _ladder_source(config), None
    missing = [name for name in spec.source_files if name not in tables]
    if missing:
        reason = f"missing compact input(s): {', '.join(missing)}"
        return _receipt(spec, reason), reason

    if spec.kind in {"delta_log_loss", "delta_brier", "seller", "route", "combined"}:
        source = tables["ORDER_BREACH_PAIRED_DIFFERENCES.csv"]
        metrics = ["delta_log_loss", "delta_brier"] if spec.kind in {"seller", "route", "combined"} else [spec.kind]
        parts = [_metric_long(source, metric) for metric in metrics]
        data = pd.concat(parts, ignore_index=True, sort=False)
        data = _later_rows(data)
        if spec.kind == "seller":
            data = _comparison_filter(data, ("M2-M1", "M4-M3"))
        elif spec.kind == "route":
            data = _comparison_filter(data, ("M3-M1", "M4-M2"))
        elif spec.kind == "combined":
            data = _comparison_filter(data, ("M4-M1",))
    elif spec.kind == "ap_top10":
        source = tables["ORDER_BREACH_BY_MONTH.csv"]
        data = pd.concat(
            [_metric_long(source, "average_precision"), _metric_long(source, "top10_lift")],
            ignore_index=True,
            sort=False,
        )
        data = _later_rows(data)
    elif spec.kind == "reliability":
        data = _standardize(tables["ORDER_CALIBRATION_BINS.csv"])
        aliases = {
            "mean_predicted_probability": ("mean_predicted_probability", "mean_probability", "predicted_rate", "bin_mean_probability"),
            "observed_rate": ("observed_rate", "event_rate", "prevalence", "observed_prevalence", "bin_observed_rate"),
            "n_orders": ("n_orders", "orders", "n", "bin_count", "count"),
        }
        lower = {str(column).lower(): column for column in data.columns}
        for canonical, candidates in aliases.items():
            if canonical not in data.columns:
                source_column = next((lower[candidate] for candidate in candidates if candidate in lower), None)
                if source_column is not None:
                    if canonical == "n_orders":
                        data[canonical] = data[source_column]
                    else:
                        data = data.rename(columns={source_column: canonical})
        required = {
            "period",
            "cohort_month",
            "family",
            "model_id",
            "probability_variant",
            "mean_predicted_probability",
            "observed_rate",
            "n_orders",
        }
        if not required.issubset(data.columns):
            data = data.iloc[0:0]
        else:
            period = data["period"].astype(str).str.lower()
            cohort = data["cohort_month"].astype(str).str.lower()
            probability = data["probability_variant"].astype(str).str.lower()
            data = data.loc[
                period.str.contains("aggregate|pooled", regex=True)
                & (period.str.contains("later", regex=False) | cohort.str.contains("later", regex=False))
                & cohort.str.contains("pooled|aggregate", regex=True)
                & data["model_id"].astype(str).str.upper().isin({"M1", "M2", "M3", "M4"})
                & probability.eq("calibrated")
            ].copy()
    elif spec.kind == "ablation":
        source = tables["ORDER_PROFILE_ABLATIONS.csv"]
        data = pd.concat(
            [_metric_long(source, "delta_log_loss"), _metric_long(source, "delta_brier")],
            ignore_index=True,
            sort=False,
        )
        if data.empty or pd.to_numeric(data.get("estimate"), errors="coerce").notna().sum() == 0:
            data = pd.concat(
                [_metric_long(source, "log_loss"), _metric_long(source, "brier")],
                ignore_index=True,
                sort=False,
            )
        data = _later_rows(data)
    elif spec.kind in {"q50_skill", "q90_skill"}:
        data = _metric_long(tables["SEVERITY_PINBALL_SKILL.csv"], "pinball_skill")
        data = _later_rows(_quantile_rows(data, 0.50 if spec.kind == "q50_skill" else 0.90))
    elif spec.kind == "q90_coverage":
        data = _metric_long(tables["SEVERITY_COVERAGE.csv"], "coverage")
        data = _later_rows(_quantile_rows(data, 0.90))
    elif spec.kind == "severity_blocks":
        source = tables["SEVERITY_BY_MONTH.csv"]
        data = pd.concat(
            [_metric_long(source, "pinball_loss"), _metric_long(source, "coverage")],
            ignore_index=True,
            sort=False,
        )
        data = _later_rows(data)
    elif spec.kind == "hrd":
        source = tables["ORDER_EVENT_STRATA.csv"]
        data = pd.concat(
            [_metric_long(source, "delta_log_loss"), _metric_long(source, "delta_brier")],
            ignore_index=True,
            sort=False,
        )
        if data.empty or pd.to_numeric(data.get("estimate"), errors="coerce").notna().sum() == 0:
            data = _derive_model_deltas(
                source,
                (("M4-M1", "M4", "M1"), ("M4E-M4", "M4E", "M4")),
            )
        data = _later_rows(data)
    elif spec.kind == "terminal":
        source = tables["ORDER_TERMINAL_STRESS.csv"]
        parts = []
        for metric in (
            "delta_log_loss",
            "delta_brier",
            "average_precision",
            "pinball_skill",
            "coverage",
            "terminal_minus_later_calibration_intercept",
            "terminal_minus_later_calibration_slope",
            "terminal_minus_later_wace",
        ):
            part = _metric_long(source, metric)
            if not part.empty:
                parts.append(part)
        data = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    else:  # pragma: no cover - guarded by the frozen FigureSpec table
        data = pd.DataFrame()

    if spec.kind == "reliability":
        required = ("mean_predicted_probability", "observed_rate")
        valid_rows = (
            not data.empty
            and all(column in data.columns for column in required)
            and data.loc[:, list(required)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).any()
        )
        if not valid_rows:
            reason = "required reliability rows/columns unavailable"
            return _receipt(spec, reason), reason
        data = data.copy()
        for column in required:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        return data.dropna(subset=list(required)).reset_index(drop=True), None

    if data.empty or "estimate" not in data.columns or pd.to_numeric(data["estimate"], errors="coerce").notna().sum() == 0:
        reason = f"required metric rows/columns unavailable for {spec.kind}"
        return _receipt(spec, reason), reason
    data = data.copy()
    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data = data.loc[data["estimate"].notna()].reset_index(drop=True)
    return data, None


def _figure_style():
    return plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "figure.dpi": 100,
            "savefig.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _series_label(frame: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in ("family", "comparison", "model_id", "probability_variant", "quantile", "representation", "stratum", "metric")
        if column in frame.columns
    ]
    if not columns:
        return pd.Series(["all"] * len(frame), index=frame.index)
    return frame[columns].fillna("").astype(str).agg(" | ".join, axis=1).str.strip(" |")


def _plot_receipt(ax: plt.Axes, source: pd.DataFrame) -> None:
    reason = str(source.iloc[0].get("reason", "No reportable data"))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "NO DATA RECEIPT",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.5, 0.40, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)


def _plot_diagram(ax: plt.Axes, source: pd.DataFrame, kind: str) -> None:
    ax.axis("off")
    if kind == "framework":
        y_positions = np.linspace(0.88, 0.12, len(source))
        for (_, row), y in zip(source.iterrows(), y_positions):
            ax.text(
                0.5,
                y,
                str(row["label"]),
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#E8F1FA", "edgecolor": "#315B7D"},
                transform=ax.transAxes,
            )
            if y != y_positions[-1]:
                ax.annotate("", xy=(0.5, y - 0.075), xytext=(0.5, y - 0.025), arrowprops={"arrowstyle": "->", "color": "#555555"}, xycoords=ax.transAxes)
    else:
        tasks = list(source["task"].drop_duplicates())
        for panel, task in enumerate(tasks):
            group = source.loc[source["task"].eq(task)].reset_index(drop=True)
            x0 = 0.08 if panel == 0 else 0.58
            ax.text(x0, 0.94, task.title(), weight="bold", transform=ax.transAxes)
            for index, row in group.iterrows():
                y = 0.84 - index * (0.72 / max(len(group) - 1, 1))
                ax.text(
                    x0,
                    y,
                    f"{row['model_id']}: {row['increment']}",
                    ha="left",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F4F7FA", "edgecolor": "#557A95"},
                    transform=ax.transAxes,
                )


def _plot_reliability(ax: plt.Axes, source: pd.DataFrame) -> None:
    source = source.copy()
    source["series"] = _series_label(source)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1, label="ideal")
    for label, group in source.groupby("series", sort=True):
        group = group.sort_values("mean_predicted_probability", kind="mergesort")
        line = ax.plot(
            group["mean_predicted_probability"], group["observed_rate"], linewidth=1.2, label=label
        )[0]
        counts = pd.to_numeric(group.get("n_orders"), errors="coerce").fillna(0).clip(lower=0)
        maximum = float(counts.max()) if len(counts) else 0.0
        sizes = 18.0 + (72.0 * counts / maximum if maximum > 0 else 0.0)
        ax.scatter(
            group["mean_predicted_probability"],
            group["observed_rate"],
            s=sizes,
            alpha=0.75,
            color=line.get_color(),
            linewidths=0.4,
            edgecolors="#333333",
            label="_nolegend_",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed breach rate")
    ax.text(0.99, 0.01, "Point area scales with bin n", ha="right", va="bottom", transform=ax.transAxes, fontsize=6.5)
    ax.legend(loc="best", frameon=False)


def _plot_metric(ax: plt.Axes, source: pd.DataFrame, kind: str) -> None:
    data = source.copy()
    data["series"] = _series_label(data)
    if "cohort_month" in data.columns:
        data["x"] = data["cohort_month"].astype(str)
        for label, group in data.groupby("series", sort=True):
            group = group.sort_values("x", kind="mergesort")
            ax.plot(group["x"], group["estimate"], marker="o", linewidth=1.1, label=label)
        ax.tick_params(axis="x", rotation=45)
    else:
        aggregate = data.groupby("series", sort=True, as_index=False)["estimate"].median()
        ax.bar(np.arange(len(aggregate)), aggregate["estimate"], color="#4C78A8")
        ax.set_xticks(np.arange(len(aggregate)), aggregate["series"], rotation=45, ha="right")
    if kind in {"delta_log_loss", "delta_brier", "seller", "route", "combined", "hrd", "q50_skill", "q90_skill"}:
        ax.axhline(0, color="#444444", linewidth=0.8)
    if kind == "q90_coverage":
        ax.axhline(0.90, color="#444444", linewidth=0.8, linestyle="--", label="nominal 0.90")
    ax.set_ylabel("Estimate")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=False, ncol=1)


def _render_figure(spec: FigureSpec, source: pd.DataFrame, destination: Path) -> None:
    with _figure_style():
        figure, ax = plt.subplots(figsize=(8.2, 4.8))
        if "status" in source.columns and source["status"].eq("no_data").all():
            _plot_receipt(ax, source)
        elif spec.kind in {"framework", "ladder"}:
            _plot_diagram(ax, source, spec.kind)
        elif spec.kind == "reliability":
            _plot_reliability(ax, source)
        else:
            _plot_metric(ax, source, spec.kind)
        ax.set_title(spec.title)
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="png", metadata=PNG_METADATA)
        plt.close(figure)


def _parse_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed", "supported", "ok", "valid", "verified", "unchanged"}:
        return True
    if text in {"false", "0", "no", "fail", "failed", "reversal", "worse", "invalid", "changed"}:
        return False
    return None


def _guard_from_group(
    group: pd.DataFrame,
    positive_columns: Sequence[str],
    negative_columns: Sequence[str],
) -> bool | None:
    lower = {str(column).lower(): column for column in group.columns}
    for name in positive_columns:
        if name.lower() in lower:
            values = [_parse_bool(value) for value in group[lower[name.lower()]]]
            values = [value for value in values if value is not None]
            return bool(all(values)) if values else None
    for name in negative_columns:
        if name.lower() in lower:
            values = [_parse_bool(value) for value in group[lower[name.lower()]]]
            values = [value for value in values if value is not None]
            return bool(not any(values)) if values else None
    return None


def _guard_from_table(
    tables: Mapping[str, pd.DataFrame],
    filename: str,
    *,
    family: object,
    comparison: object,
    positive_columns: Sequence[str],
    negative_columns: Sequence[str],
) -> bool | None:
    """Read an explicit rule guard from its dedicated compact audit table."""
    if filename not in tables:
        return None
    data = _standardize(tables[filename])
    if data.empty:
        return None
    if "family" in data.columns and pd.notna(family):
        data = data.loc[
            data["family"].astype(str).str.casefold().eq(str(family).casefold())
        ]
    if "comparison" in data.columns and pd.notna(comparison):
        wanted = _canonical_comparison(comparison)
        data = data.loc[data["comparison"].map(_canonical_comparison).eq(wanted)]
    elif "model_id" in data.columns and pd.notna(comparison):
        candidate = str(comparison).split("-", 1)[0]
        data = data.loc[
            data["model_id"].astype(str).str.casefold().eq(candidate.casefold())
        ]
    if data.empty:
        return None
    return _guard_from_group(data, positive_columns, negative_columns)


def _paired_wide(frame: pd.DataFrame) -> pd.DataFrame:
    data = _standardize(frame)
    if "comparison" in data.columns:
        data["comparison"] = data["comparison"].map(_canonical_comparison)
    ll = _metric_long(data, "delta_log_loss")
    brier = _metric_long(data, "delta_brier")
    keys = [
        column
        for column in ("period", "cohort_month", "family", "comparison", "probability_variant")
        if column in ll.columns and column in brier.columns
    ]
    keys.extend(column for column in GUARD_COLUMNS if column in ll.columns and column in brier.columns)
    if not keys:
        return pd.DataFrame()
    ll = ll.rename(columns={"estimate": "delta_log_loss"}).drop(columns="metric", errors="ignore")
    brier = brier.rename(columns={"estimate": "delta_brier"}).drop(columns="metric", errors="ignore")
    if ll.empty or brier.empty:
        return pd.DataFrame()
    merged = ll.merge(brier, on=keys, how="inner", validate="m:m")
    merged["delta_log_loss"] = pd.to_numeric(merged["delta_log_loss"], errors="coerce")
    merged["delta_brier"] = pd.to_numeric(merged["delta_brier"], errors="coerce")
    return merged.dropna(subset=["delta_log_loss", "delta_brier"])


def _breach_comparison_summary(
    tables: Mapping[str, pd.DataFrame], blockers: list[str]
) -> list[dict[str, object]]:
    if "ORDER_BREACH_PAIRED_DIFFERENCES.csv" not in tables:
        return []
    data = _paired_wide(tables["ORDER_BREACH_PAIRED_DIFFERENCES.csv"])
    data = _later_rows(data)
    if data.empty or not {"family", "comparison", "cohort_month"}.issubset(data.columns):
        blockers.append("breach_evidence_schema: paired later-cohort rows require family, comparison, cohort_month, delta_log_loss, delta_brier")
        return []
    rows: list[dict[str, object]] = []
    group_columns = ["family", "comparison"]
    if "probability_variant" in data.columns:
        group_columns.append("probability_variant")
    for keys, group in data.groupby(group_columns, dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_columns, values))
        comparison = str(meta["comparison"])
        month_rows = group.groupby("cohort_month", as_index=False).agg(
            delta_log_loss=("delta_log_loss", "mean"), delta_brier=("delta_brier", "mean")
        )
        n_months = int(month_rows["cohort_month"].nunique())
        median_ll = float(month_rows["delta_log_loss"].median())
        median_brier = float(month_rows["delta_brier"].median())
        months_ll = int(month_rows["delta_log_loss"].lt(0).sum())
        months_brier = int(month_rows["delta_brier"].lt(0).sum())
        months_both = int((month_rows["delta_log_loss"].lt(0) & month_rows["delta_brier"].lt(0)).sum())
        profile_comparison = comparison in {"M2-M1", "M3-M1", "M4-M1"}
        high_support = _guard_from_group(group, ("high_support_ok", "high_support_no_material_reversal"), ("high_support_material_reversal",))
        calibration = _guard_from_group(group, ("calibration_not_systematically_worse", "calibration_ok"), ("calibration_systematically_worse",))
        score_guard = _guard_from_group(group, ("benefit_not_metadata_only", "score_contributes"), ("benefit_metadata_only", "metadata_only_benefit"))
        if high_support is None:
            high_support = _guard_from_table(
                tables,
                "ORDER_PROFILE_SUPPORT_STRATA.csv",
                family=meta.get("family"),
                comparison=comparison,
                positive_columns=("high_support_ok", "high_support_no_material_reversal"),
                negative_columns=("high_support_material_reversal",),
            )
        if calibration is None:
            calibration = _guard_from_table(
                tables,
                "ORDER_CALIBRATION_RESULTS.csv",
                family=meta.get("family"),
                comparison=comparison,
                positive_columns=("calibration_not_systematically_worse", "calibration_ok"),
                negative_columns=("calibration_systematically_worse",),
            )
        if score_guard is None:
            score_guard = _guard_from_table(
                tables,
                "ORDER_PROFILE_ABLATIONS.csv",
                family=meta.get("family"),
                comparison=comparison,
                positive_columns=("benefit_not_metadata_only", "score_contributes"),
                negative_columns=("benefit_metadata_only", "metadata_only_benefit"),
            )
        if not profile_comparison:
            evidence_status = ""
            reason = "baseline_or_secondary_comparison_not_assigned_a_profile_block_label"
        elif n_months < 6:
            evidence_status = "Mixed"
            reason = f"incomplete_later_cohort_months:{n_months}/6"
        elif median_ll >= 0 or median_brier >= 0:
            evidence_status = "Not-supported"
            reason = "median_proper_score_non_improvement"
        elif months_both <= 2:
            evidence_status = "Not-supported"
            reason = "majority_of_months_reverse_on_one_or_both_proper_scores"
        elif high_support is False:
            evidence_status = "Mixed"
            reason = "high_support_results_inconsistent_or_materially_reversed"
        elif months_both < 4:
            evidence_status = "Mixed"
            reason = "proper_score_gain_concentrated_in_fewer_than_four_months"
        elif high_support is True and calibration is True and score_guard is True:
            evidence_status = "Supported"
            reason = "all_frozen_breach_rules_passed"
        else:
            evidence_status = "Mixed"
            missing_guards = [
                name
                for name, value in (("high_support", high_support), ("calibration", calibration), ("score_not_metadata_only", score_guard))
                if value is not True
            ]
            reason = "guard_failed_or_unavailable:" + "|".join(missing_guards)
            for guard_name, guard_value in (
                ("high_support", high_support),
                ("calibration", calibration),
                ("score_not_metadata_only", score_guard),
            ):
                if guard_value is None:
                    blockers.append(
                        f"breach_label_guard_missing:{meta.get('family', '')}:{comparison}:{guard_name}"
                    )
        rows.append(
            {
                "task": "breach",
                "family": meta.get("family", ""),
                "comparison": comparison,
                "probability_variant": meta.get("probability_variant", ""),
                "quantile": np.nan,
                "n_months": n_months,
                "median_delta_log_loss": median_ll,
                "median_delta_brier": median_brier,
                "months_log_loss_improved": months_ll,
                "months_brier_improved": months_brier,
                "months_both_improved": months_both,
                "median_pinball_skill": np.nan,
                "months_nonnegative_skill": np.nan,
                "high_support_guard": high_support,
                "calibration_guard": calibration,
                "score_not_metadata_only_guard": score_guard,
                "coverage_guard": np.nan,
                "evidence_status": evidence_status,
                "evidence_reason": reason,
                "source_file": "ORDER_BREACH_PAIRED_DIFFERENCES.csv",
            }
        )
    return rows


def _severity_comparison_summary(
    tables: Mapping[str, pd.DataFrame], blockers: list[str]
) -> list[dict[str, object]]:
    if "SEVERITY_PINBALL_SKILL.csv" not in tables:
        return []
    data = _metric_long(tables["SEVERITY_PINBALL_SKILL.csv"], "pinball_skill")
    data = _later_rows(data)
    if "comparison" not in data.columns and "model_id" in data.columns:
        data["comparison"] = data["model_id"].astype(str).map(
            {"Q2": "Q2-Q1", "Q3": "Q3-Q1", "Q4": "Q4-Q1"}
        )
    if "comparison" in data.columns:
        data["comparison"] = data["comparison"].astype(str).str.upper().str.replace("_", "-", regex=False)
        data = data.loc[data["comparison"].isin({"Q2-Q1", "Q3-Q1", "Q4-Q1"})].copy()
    required = {"family", "comparison", "cohort_month", "quantile", "estimate"}
    if data.empty or not required.issubset(data.columns):
        blockers.append("severity_evidence_schema: later pinball skill requires family, comparison/model_id, cohort_month, quantile, estimate")
        return []
    rows: list[dict[str, object]] = []
    for (family, comparison, quantile), group in data.groupby(
        ["family", "comparison", "quantile"], dropna=False, sort=True
    ):
        month_rows = group.groupby("cohort_month", as_index=False)["estimate"].mean()
        skill = pd.to_numeric(month_rows["estimate"], errors="coerce").dropna()
        if skill.empty:
            continue
        n_months = int(len(skill))
        median_skill = float(skill.median())
        months_nonnegative = int(skill.ge(0).sum())
        coverage_guard = _guard_from_group(
            group,
            ("coverage_not_materially_worse", "coverage_ok"),
            ("coverage_materially_worse", "coverage_deteriorated"),
        )
        high_support = _guard_from_group(
            group,
            ("support_ge20_gain_present", "high_support_ok"),
            ("gain_only_low_support", "high_support_material_reversal"),
        )
        if coverage_guard is None:
            coverage_guard = _guard_from_table(
                tables,
                "SEVERITY_COVERAGE.csv",
                family=family,
                comparison=comparison,
                positive_columns=("coverage_not_materially_worse", "coverage_ok"),
                negative_columns=("coverage_materially_worse", "coverage_deteriorated"),
            )
        if high_support is None:
            high_support = _guard_from_table(
                tables,
                "SEVERITY_SUPPORT_STRATA.csv",
                family=family,
                comparison=comparison,
                positive_columns=("support_ge20_gain_present", "high_support_ok"),
                negative_columns=("gain_only_low_support", "high_support_material_reversal"),
            )
        quantile_value = pd.to_numeric(
            pd.Series([str(quantile).upper().replace("Q", "")]), errors="coerce"
        ).iloc[0]
        if pd.notna(quantile_value) and quantile_value > 1:
            quantile_value /= 100.0
        q90_coverage_required = bool(
            pd.notna(quantile_value) and np.isclose(float(quantile_value), 0.90)
        )
        if n_months < 6:
            status = "Mixed"
            reason = f"incomplete_later_cohort_months:{n_months}/6"
        elif median_skill <= 0 or months_nonnegative < 4:
            status = "Not-supported"
            reason = "median_skill_nonpositive_or_fewer_than_four_nonnegative_months"
        elif high_support is False:
            status = "Not-supported"
            reason = "low_support_only_or_material_high_support_reversal"
        elif high_support is True and (not q90_coverage_required or coverage_guard is True):
            status = "Supported"
            reason = "all_frozen_severity_rules_passed"
        else:
            status = "Mixed"
            missing = [
                name
                for name, value, required_guard in (
                    ("coverage", coverage_guard, q90_coverage_required),
                    ("support_ge20", high_support, True),
                )
                if required_guard
                if value is not True
            ]
            reason = "guard_failed_or_unavailable:" + "|".join(missing)
            for guard_name, guard_value in (
                ("coverage", coverage_guard if q90_coverage_required else True),
                ("support_ge20", high_support),
            ):
                if guard_value is None:
                    blockers.append(
                        f"severity_label_guard_missing:{family}:{comparison}:{quantile}:{guard_name}"
                    )
        rows.append(
            {
                "task": "severity",
                "family": family,
                "comparison": comparison,
                "probability_variant": "",
                "quantile": quantile,
                "n_months": n_months,
                "median_delta_log_loss": np.nan,
                "median_delta_brier": np.nan,
                "months_log_loss_improved": np.nan,
                "months_brier_improved": np.nan,
                "months_both_improved": np.nan,
                "median_pinball_skill": median_skill,
                "months_nonnegative_skill": months_nonnegative,
                "high_support_guard": high_support,
                "calibration_guard": np.nan,
                "score_not_metadata_only_guard": np.nan,
                "coverage_guard": coverage_guard if q90_coverage_required else np.nan,
                "evidence_status": status,
                "evidence_reason": reason,
                "source_file": "SEVERITY_PINBALL_SKILL.csv",
            }
        )
    return rows


def _model_comparison_summary(
    tables: Mapping[str, pd.DataFrame], blockers: list[str]
) -> pd.DataFrame:
    rows = _breach_comparison_summary(tables, blockers) + _severity_comparison_summary(tables, blockers)
    if not rows:
        blockers.append("model_comparison_summary:no evaluable compact comparison rows")
        return pd.DataFrame(
            [
                {
                    **{column: np.nan for column in MODEL_COMPARISON_COLUMNS},
                    "task": "no_data",
                    "evidence_reason": "no evaluable compact comparison rows",
                }
            ],
            columns=MODEL_COMPARISON_COLUMNS,
        )
    return pd.DataFrame(rows).loc[:, MODEL_COMPARISON_COLUMNS].sort_values(
        ["task", "family", "comparison", "quantile", "probability_variant"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _markdown_excerpt(
    frame: pd.DataFrame | None,
    preferred: Sequence[str] = (),
    *,
    max_rows: int = 8,
) -> str:
    if frame is None or frame.empty:
        return "No reportable compact rows were available; see `BLOCKERS.md`."
    columns = [column for column in preferred if column in frame.columns]
    if not columns:
        columns = list(frame.columns[: min(8, len(frame.columns))])
    excerpt = frame.loc[:, columns].head(max_rows).copy()
    try:
        return excerpt.to_markdown(index=False)
    except ImportError:  # pragma: no cover - tabulate is present in the project env
        return "```text\n" + excerpt.to_string(index=False) + "\n```"


def _filter_text(frame: pd.DataFrame | None, column: str, token: str) -> pd.DataFrame | None:
    if frame is None or frame.empty or column not in frame.columns:
        return frame
    return frame.loc[frame[column].astype(str).str.contains(token, case=False, regex=False)].copy()


def _comparison_excerpt(summary: pd.DataFrame, comparison: str) -> str:
    selected = summary.loc[summary["comparison"].astype(str).eq(comparison)].copy()
    return _markdown_excerpt(
        selected,
        (
            "family",
            "probability_variant",
            "quantile",
            "n_months",
            "median_delta_log_loss",
            "median_delta_brier",
            "months_both_improved",
            "median_pinball_skill",
            "months_nonnegative_skill",
            "evidence_status",
            "evidence_reason",
        ),
    )


def _last_int(patterns: Sequence[str], text: str) -> int | None:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    return int(matches[-1]) if matches else None


def _parse_test_results(text: str) -> dict[str, object]:
    counts = {
        "collected": _last_int((r"collected\s+(\d+)\s+items?", r"(\d+)\s+tests?\s+collected"), text),
        "passed": _last_int((r"(\d+)\s+passed\b",), text),
        "failed": _last_int((r"(\d+)\s+failed\b",), text),
        "skipped": _last_int((r"(\d+)\s+skipped\b",), text),
        "deselected": _last_int((r"(\d+)\s+deselected\b",), text),
        "errors": _last_int((r"(\d+)\s+errors?\b",), text),
    }
    explicit_return_code = _last_int(
        (
            r"(?:pytest[_ -]?)?return[_ -]?code\s*[:=]\s*(-?\d+)",
            r"exit[_ -]?code\s*[:=]\s*(-?\d+)",
        ),
        text,
    )
    inferred_collected = False
    if counts["collected"] is None and any(counts[name] is not None for name in ("passed", "failed", "skipped", "deselected")):
        counts["collected"] = sum(int(counts[name] or 0) for name in ("passed", "failed", "skipped", "deselected"))
        inferred_collected = True
    return_code = explicit_return_code
    return_code_inferred = False
    if return_code is None:
        if int(counts["failed"] or 0) > 0 or int(counts["errors"] or 0) > 0:
            return_code = 1
            return_code_inferred = True
        elif int(counts["passed"] or 0) > 0:
            return_code = 0
            return_code_inferred = True
    collected = counts["collected"]
    return {
        **counts,
        "return_code": return_code,
        "required_minimum_collected": 30,
        "required_minimum_collected_met": bool(collected is not None and int(collected) >= 30),
        "collected_inferred_from_summary_counts": inferred_collected,
        "return_code_inferred_from_summary_counts": return_code_inferred,
    }


def _test_text(
    out_dir: Path,
    test_results: str | Path | None,
) -> tuple[str, dict[str, object], list[str]]:
    blockers: list[str] = []
    if test_results is None:
        candidate = out_dir / "TEST_RESULTS.txt"
        if not candidate.exists():
            text = "No persisted test log was supplied."
            blockers.append("tests:TEST_RESULTS.txt missing")
        else:
            text = candidate.read_text(encoding="utf-8", errors="replace")
    else:
        supplied = str(test_results)
        try:
            candidate = Path(supplied)
            is_file = len(supplied) < 4096 and candidate.is_file()
        except (OSError, ValueError):
            is_file = False
        text = candidate.read_text(encoding="utf-8", errors="replace") if is_file else supplied
    _write_text(out_dir / "TEST_RESULTS.txt", text)
    parsed = _parse_test_results(text)
    if int(parsed.get("failed") or 0) > 0 or int(parsed.get("errors") or 0) > 0:
        blockers.append("tests:failed_or_error_count_nonzero")
    if parsed.get("return_code") not in (0, None):
        blockers.append(f"tests:return_code={parsed['return_code']}")
    if int(parsed.get("passed") or 0) <= 0:
        blockers.append("tests:no passing tests parsed")
    if parsed.get("required_minimum_collected_met") is not True:
        blockers.append("tests:fewer_than_30_required_tests_collected")
    if parsed.get("return_code") is None:
        blockers.append("tests:return_code unavailable")
    return text, parsed, blockers


def _protected_baseline_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    roots = value.get("roots")
    return {
        "coverage_rule": value.get("coverage_rule"),
        "excluded_new_workspace": value.get("excluded_new_workspace"),
        "root_count": value.get("root_count"),
        "file_count": value.get("file_count"),
        "total_bytes": value.get("total_bytes"),
        "aggregate_sha256": value.get("aggregate_sha256"),
        "roots": dict(roots) if isinstance(roots, Mapping) else {},
    }


def _run_provenance(
    output: Path,
    work_dir: str | Path | None,
) -> tuple[dict[str, object], list[str]]:
    """Load and verify persisted preflight/selection receipts for the manifest."""
    result: dict[str, object] = {}
    blockers: list[str] = []
    if work_dir is None:
        return {"status": "not_supplied_to_reporting"}, blockers
    work = Path(work_dir)
    prestate_path = work / "PRE_EXECUTION_STATE.json"
    if not prestate_path.is_file():
        blockers.append("provenance:working/PRE_EXECUTION_STATE.json missing")
        return {"status": "missing_pre_execution_state"}, blockers
    try:
        prestate = json.loads(prestate_path.read_text(encoding="utf-8"))
        if not isinstance(prestate, dict):
            raise TypeError("top-level prestate must be an object")
    except (OSError, ValueError, TypeError) as exc:
        blockers.append(f"provenance:PRE_EXECUTION_STATE unreadable:{type(exc).__name__}")
        return {"status": "unreadable_pre_execution_state"}, blockers

    source_receipt = prestate.get("source_input_audit", {})
    source_verified = bool(
        prestate.get("status") == "passed"
        and isinstance(source_receipt, Mapping)
        and int(source_receipt.get("row_count", 0) or 0) > 0
        and int(source_receipt.get("row_count", 0) or 0)
        == int(source_receipt.get("verified_row_count", -1) or -1)
    )
    source_audit_path = output / "SOURCE_INPUT_AUDIT.csv"
    source_audit_current_hash = _sha256(source_audit_path) if source_audit_path.is_file() else None
    if isinstance(source_receipt, Mapping) and source_receipt.get("sha256") != source_audit_current_hash:
        source_verified = False
    if not source_verified:
        blockers.append("provenance:source_input_audit_not_fully_verified")

    result["pre_execution_state"] = {
        "path": str(prestate_path),
        "sha256": _sha256(prestate_path),
        "status": prestate.get("status"),
        "repository": prestate.get("repository"),
        "environment": prestate.get("environment"),
        "assembler_sha256": prestate.get("assembler_sha256"),
        "raw_file_hashes": prestate.get("raw_file_hashes"),
        "profile_input_hashes": prestate.get("profile_input_hashes"),
        "source_code_hashes": prestate.get("source_code_hashes"),
        "local_frozen_controls": prestate.get("local_frozen_controls"),
        "source_input_audit": source_receipt,
        "profile_block_audit": prestate.get("profile_block_audit"),
        "protected_baseline": _protected_baseline_summary(prestate.get("protected_baseline")),
    }
    result["source_verdict"] = {
        "passed": source_verified,
        "source_input_audit_current_sha256": source_audit_current_hash,
    }

    selection_path = output / "ORDER_MODEL_SELECTION_FREEZE.json"
    if selection_path.is_file():
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if not isinstance(selection, dict):
                raise TypeError("top-level selection freeze must be an object")
            result["model_selection_freeze"] = {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
                "content_summary": selection,
            }
        except (OSError, ValueError, TypeError) as exc:
            blockers.append(f"provenance:ORDER_MODEL_SELECTION_FREEZE unreadable:{type(exc).__name__}")
    else:
        blockers.append("provenance:ORDER_MODEL_SELECTION_FREEZE.json missing")

    try:
        from analysis.order_breach_severity_v1.scripts import order_preflight

        protected_verdict = order_preflight.verify_protected_unchanged(prestate_path)
        result["protected_before_after_verdict"] = protected_verdict
    except Exception as exc:  # preserve a receipt even when the mandatory hard gate fails
        result["protected_before_after_verdict"] = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        blockers.append(f"provenance:protected_before_after_verification_failed:{type(exc).__name__}")
    return result, blockers


def _runtime_commands(out_dir: Path) -> tuple[list[str], list[str]]:
    scripts = ["analysis/order_breach_severity_v1/scripts/order_reporting.py"]
    current_command = shlex.join(
        list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    )
    commands = [current_command]
    state_path = out_dir / "working/RUN_STATE.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            prior_commands = [str(value) for value in state.get("commands", [])]
            commands = list(dict.fromkeys([*prior_commands, current_command]))
            scripts = sorted(
                {
                    token
                    for command in commands
                    for token in command.split()
                    if token.endswith(".py") and "order_breach_severity_v1" in token
                }
                | set(scripts)
            )
        except Exception:
            pass
    return scripts, commands


def _section_payloads(
    out_dir: Path,
    config: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    comparison_summary: pd.DataFrame,
    blockers: Sequence[str],
    test_text: str,
    test_summary: Mapping[str, object],
    provenance: Mapping[str, object],
    created_files: Sequence[str],
) -> dict[str, str]:
    scripts, commands = _runtime_commands(out_dir)
    tuning = _standardize(tables.get("ORDER_DEVELOPMENT_TUNING.csv", pd.DataFrame()))
    calibration = _standardize(tables.get("ORDER_CALIBRATION_RESULTS.csv", pd.DataFrame()))
    severity_summary = comparison_summary.loc[comparison_summary["task"].eq("severity")]
    breach_labels = comparison_summary.loc[
        comparison_summary["task"].eq("breach") & comparison_summary["evidence_status"].isin(["Supported", "Mixed", "Not-supported"])
    ]
    severity_labels = severity_summary.loc[
        severity_summary["evidence_status"].isin(["Supported", "Mixed", "Not-supported"])
    ]
    feature_config = config.get("current_order_features", {})
    evidence_path = out_dir / "EVIDENCE_STATUS.md"
    evidence_text = (
        evidence_path.read_text(encoding="utf-8").strip()
        if evidence_path.exists()
        else "This is a frozen post-profile-selection chronological evaluation; no period is an untouched holdout."
    )
    source_verdict = "No verified source verdict was present in the compact audit tables."
    preservation_verdict = "No verified protected-file preservation verdict was present in the compact audit tables."
    sample = _standardize(tables.get("ORDER_SAMPLE_AUDIT.csv", pd.DataFrame()))
    for column in ("source_hashes_valid", "canonical_assembler_valid", "source_verdict"):
        if column in sample.columns:
            source_verdict = _markdown_excerpt(sample, (column, "audit_name", "value", "status"))
            break
    for column in ("protected_files_unchanged", "preservation_verdict", "protected_hashes_valid"):
        if column in sample.columns:
            preservation_verdict = _markdown_excerpt(sample, (column, "audit_name", "value", "status"))
            break
    if source_verdict.startswith("No verified") and "source_verdict" in provenance:
        source_verdict = "```json\n" + json.dumps(
            provenance["source_verdict"], indent=2, sort_keys=True, ensure_ascii=False, default=str
        ) + "\n```"
    if preservation_verdict.startswith("No verified") and "protected_before_after_verdict" in provenance:
        preservation_verdict = "```json\n" + json.dumps(
            provenance["protected_before_after_verdict"], indent=2, sort_keys=True, ensure_ascii=False, default=str
        ) + "\n```"
    payloads = {
        "scripts": "\n".join(f"- `{value}`" for value in scripts),
        "commands": "\n".join(f"- `{value}`" for value in commands),
        "source_verdict": source_verdict,
        "preservation_verdict": preservation_verdict,
        "evidence_status": evidence_text,
        "sample_counts": _markdown_excerpt(sample, ("sample", "metric", "count", "value", "n_orders", "n_events"), max_rows=14),
        "profile_join": _markdown_excerpt(_standardize(tables.get("ORDER_PROFILE_JOIN_AUDIT.csv", pd.DataFrame())), ("profile", "block", "metric", "value", "count", "rate", "n_orders"), max_rows=14),
        "feature_block": (
            f"Promise numeric ({len(feature_config.get('promise_numeric', []))}): "
            + ", ".join(f"`{value}`" for value in feature_config.get("promise_numeric", []))
            + f"\n\nContext numeric ({len(feature_config.get('context_numeric', []))}): "
            + ", ".join(f"`{value}`" for value in feature_config.get("context_numeric", []))
            + f"\n\nContext categorical ({len(feature_config.get('context_categorical', []))}): "
            + ", ".join(f"`{value}`" for value in feature_config.get("context_categorical", []))
        ),
        "lr_tuning": _markdown_excerpt(_filter_text(tuning, "family", "logistic"), ("family", "parameter_json", "params", "mean_log_loss", "mean_brier", "selected")),
        "boost_tuning": _markdown_excerpt(_filter_text(tuning, "family", "xgboost"), ("family", "parameter_json", "params", "mean_log_loss", "mean_brier", "selected")),
        "m0_m1": _comparison_excerpt(comparison_summary, "M1-M0"),
        "m1_m2": _comparison_excerpt(comparison_summary, "M2-M1"),
        "m1_m3": _comparison_excerpt(comparison_summary, "M3-M1"),
        "m1_m4": _comparison_excerpt(comparison_summary, "M4-M1"),
        "m4_m5": _comparison_excerpt(comparison_summary, "M5-M4"),
        "m4_m4e": _comparison_excerpt(comparison_summary, "M4E-M4"),
        "calibration": _markdown_excerpt(calibration, ("period", "cohort_month", "family", "model_id", "probability_variant", "log_loss", "brier", "calibration_intercept", "calibration_slope", "wace"), max_rows=12),
        "profile_ablation": _markdown_excerpt(_standardize(tables.get("ORDER_PROFILE_ABLATIONS.csv", pd.DataFrame())), ("period", "family", "model_id", "representation", "metric", "estimate"), max_rows=12),
        "support_cold": _markdown_excerpt(_standardize(tables.get("ORDER_PROFILE_SUPPORT_STRATA.csv", pd.DataFrame())), ("period", "family", "model_id", "stratum", "metric", "estimate", "n_orders"), max_rows=12),
        "q50": _markdown_excerpt(_quantile_rows(severity_summary, 0.50), ("family", "comparison", "n_months", "median_pinball_skill", "months_nonnegative_skill", "evidence_status", "evidence_reason")),
        "q90": _markdown_excerpt(_quantile_rows(severity_summary, 0.90), ("family", "comparison", "n_months", "median_pinball_skill", "months_nonnegative_skill", "evidence_status", "evidence_reason")),
        "coverage": _markdown_excerpt(_standardize(tables.get("SEVERITY_COVERAGE.csv", pd.DataFrame())), ("period", "cohort_month", "family", "model_id", "quantile", "coverage", "empirical_coverage", "coverage_error"), max_rows=12),
        "seller_route_severity": _markdown_excerpt(severity_summary.loc[severity_summary["comparison"].astype(str).isin(["Q2-Q1", "Q3-Q1", "Q4-Q1"])], ("family", "comparison", "quantile", "median_pinball_skill", "months_nonnegative_skill", "evidence_status", "evidence_reason"), max_rows=12),
        "bau_hrd": _markdown_excerpt(_standardize(tables.get("ORDER_EVENT_STRATA.csv", pd.DataFrame())), ("period", "family", "comparison", "stratum", "metric", "estimate", "prevalence", "n_orders"), max_rows=12),
        "terminal": _markdown_excerpt(
            _standardize(tables.get("ORDER_TERMINAL_STRESS.csv", pd.DataFrame())),
            (
                "analysis", "period", "family", "model_id", "probability_variant",
                "comparison", "metric", "estimate", "later_pooled_estimate",
                "terminal_estimate", "n_orders", "prevalence",
            ),
            max_rows=24,
        ),
        "breach_labels": _markdown_excerpt(breach_labels, ("family", "comparison", "probability_variant", "evidence_status", "evidence_reason"), max_rows=20),
        "severity_labels": _markdown_excerpt(severity_labels, ("family", "comparison", "quantile", "evidence_status", "evidence_reason"), max_rows=20),
        "blockers": "None." if not blockers else "\n".join(f"- {value}" for value in blockers),
        "tests": (
            "Parsed test summary:\n\n```json\n"
            + json.dumps(dict(test_summary), indent=2, sort_keys=True, ensure_ascii=False, default=str)
            + "\n```\n\nPersisted log excerpt (full log is in `TEST_RESULTS.txt`):\n\n```text\n"
            + ((test_text or "No test text available.").strip()[-3000:])
            + "\n```"
        ),
        "files": "\n".join(f"- `{value}`" for value in sorted(created_files)),
        "scope_confirmation": (
            "Confirmed: this reporting stage did not rewrite the thesis, edit prior registries, "
            "reselect profiles, optimise customer promises, or run a business-policy simulation."
        ),
    }
    return payloads


def _write_summary(
    path: Path,
    title: str,
    payloads: Mapping[str, str],
    *,
    chinese: bool,
) -> None:
    lines = [f"# {title}", "", "Sections below follow the frozen 31-item completion order." if not chinese else "以下章节严格遵循冻结的 31 项完成报告顺序。", ""]
    for index, (key, english, zh) in enumerate(COMPLETION_REPORT_SECTIONS, 1):
        lines.extend([f"## {index:02d}. {zh if chinese else english}", "", payloads[key], ""])
    _write_text(path, "\n".join(lines))


def _summary_section_order_valid(path: Path, *, chinese: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    positions = []
    for index, (_, english, zh) in enumerate(COMPLETION_REPORT_SECTIONS, 1):
        heading = f"## {index:02d}. {zh if chinese else english}"
        positions.append(text.find(heading))
    return all(position >= 0 for position in positions) and positions == sorted(positions)


def _blockers_markdown(blockers: Sequence[str]) -> str:
    lines = ["# Blockers", ""]
    if blockers:
        lines.extend(["Completion is blocked by:", ""])
        lines.extend(f"- {value}" for value in sorted(set(blockers)))
    else:
        lines.append("No reporting or evidence-completeness blockers were detected.")
    lines.extend(
        [
            "",
            "A no-data figure receipt is not empirical evidence and cannot be used to assign a Supported/Mixed/Not-supported label.",
        ]
    )
    return "\n".join(lines)


def _inventory(out_dir: Path, relative_paths: Iterable[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for relative in sorted(set(relative_paths)):
        path = out_dir / relative
        if path.is_file():
            result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _complete_inventory(
    out_dir: Path,
    relative_paths: Iterable[str],
    *,
    omit_self_hash: str | None = None,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for relative in sorted(set(relative_paths)):
        path = out_dir / relative
        if not path.is_file():
            result[relative] = {"exists": False, "bytes": None, "sha256": None}
        elif relative == omit_self_hash:
            result[relative] = {
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": None,
                "hash_omitted_reason": "self_referential_manifest_hash",
            }
        else:
            result[relative] = {
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return result


def _artifact_validation(
    out_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    blockers: Sequence[str],
) -> tuple[str, bool]:
    checks: list[tuple[str, bool, str]] = []
    for spec in FIGURE_SPECS:
        source_path = out_dir / "figure_sources" / f"{spec.stem}.csv"
        figure_path = out_dir / "figures" / f"{spec.stem}.png"
        checks.append((f"source:{spec.stem}", source_path.is_file(), source_path.as_posix()))
        png_ok = figure_path.is_file() and figure_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        checks.append((f"figure:{spec.stem}", png_ok, figure_path.as_posix()))
    for filename in REQUIRED_COMPACT_CSVS:
        checks.append((f"compact_input:{filename}", filename in tables, filename))
    for filename in REQUIRED_OUTPUTS:
        checks.append((f"output:{filename}", (out_dir / filename).is_file(), filename))
    checks.append(("summary_order:english", _summary_section_order_valid(out_dir / "ORDER_RESULTS_SUMMARY.md", chinese=False), "31 headings"))
    checks.append(("summary_order:chinese", _summary_section_order_valid(out_dir / "ORDER_RESULTS_SUMMARY_ZH.md", chinese=True), "31 headings"))
    comparison_path = out_dir / "MODEL_COMPARISON_SUMMARY.csv"
    comparison_columns_ok = False
    if comparison_path.exists():
        comparison_columns_ok = tuple(pd.read_csv(comparison_path, nrows=0).columns) == MODEL_COMPARISON_COLUMNS
    checks.append(("schema:MODEL_COMPARISON_SUMMARY.csv", comparison_columns_ok, "exact frozen reporting schema"))
    structural_pass = all(passed for _, passed, _ in checks)
    overall_pass = structural_pass and not blockers
    lines = [
        "# Artifact Validation Report",
        "",
        f"Structural artifact verdict: **{'PASS' if structural_pass else 'FAIL'}**.",
        f"Overall completion verdict: **{'PASS' if overall_pass else 'BLOCKED'}**.",
        "",
        "No-data receipts preserve traceability but do not satisfy empirical evidence completeness.",
        "",
        "| Check | Pass | Detail |",
        "|---|---:|---|",
    ]
    lines.extend(f"| `{name}` | {str(passed)} | {detail} |" for name, passed, detail in checks)
    if blockers:
        lines.extend(["", "## Active blockers", ""])
        lines.extend(f"- {value}" for value in sorted(set(blockers)))
    return "\n".join(lines), overall_pass


def finalize_reporting(
    out_dir: str | Path | None = None,
    config: Mapping[str, object] | str | Path | None = None,
    test_results: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    test_results_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate the deterministic compact report and validate its artifacts.

    Missing compact inputs or required columns are persisted as blockers.  The
    function still emits every frozen figure/source pair using an explicit
    no-data receipt so a reviewer can distinguish absent evidence from zero or
    neutral evidence.
    """

    if output_dir is not None:
        if out_dir is not None and Path(out_dir) != Path(output_dir):
            raise ValueError("out_dir and output_dir refer to different locations")
        out_dir = output_dir
    if out_dir is None:
        raise TypeError("finalize_reporting requires out_dir or output_dir")
    output = Path(out_dir)
    if test_results_path is not None:
        if test_results is not None and Path(test_results) != Path(test_results_path):
            raise ValueError("test_results and test_results_path refer to different locations")
        test_results = test_results_path

    config_blocker: str | None = None
    if isinstance(config, Mapping):
        frozen_config: Mapping[str, object] = config
    else:
        candidates: list[Path] = []
        if config is not None:
            candidates.append(Path(config))
        if work_dir is not None:
            candidates.append(Path(work_dir).parent / "ORDER_FROZEN_CONFIG.json")
        candidates.extend(
            [
                output / "ORDER_FROZEN_CONFIG.json",
                Path(__file__).resolve().parents[1] / "ORDER_FROZEN_CONFIG.json",
            ]
        )
        config_path = next((path for path in candidates if path.is_file()), None)
        if config_path is None:
            frozen_config = {}
            config_blocker = "frozen_config:ORDER_FROZEN_CONFIG.json unavailable"
        else:
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise TypeError("top-level config must be an object")
                frozen_config = loaded
            except (OSError, ValueError, TypeError) as exc:
                frozen_config = {}
                config_blocker = f"frozen_config:unreadable:{config_path}:{type(exc).__name__}"
    config = frozen_config

    figures = output / "figures"
    sources = output / "figure_sources"
    figures.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = [config_blocker] if config_blocker is not None else []
    provenance, provenance_blockers = _run_provenance(output, work_dir)
    blockers.extend(provenance_blockers)
    tables: dict[str, pd.DataFrame] = {}
    input_hashes: dict[str, str] = {}
    for filename in REQUIRED_COMPACT_CSVS:
        path = output / filename
        if not path.exists():
            blockers.append(f"missing_compact_input:{filename}")
            continue
        try:
            tables[filename] = pd.read_csv(path, low_memory=False)
            input_hashes[filename] = _sha256(path)
        except Exception as exc:
            blockers.append(f"unreadable_compact_input:{filename}:{type(exc).__name__}")

    figure_receipts: dict[str, str] = {}
    created_files: list[str] = []
    for spec in FIGURE_SPECS:
        source, reason = _prepare_figure_source(spec, tables, config)
        source_relative = f"figure_sources/{spec.stem}.csv"
        figure_relative = f"figures/{spec.stem}.png"
        _write_csv(output / source_relative, source)
        _render_figure(spec, source, output / figure_relative)
        created_files.extend([source_relative, figure_relative])
        if reason is not None:
            blocker = f"figure_no_data:{spec.stem}:{reason}"
            blockers.append(blocker)
            figure_receipts[spec.stem] = reason

    comparison_summary = _model_comparison_summary(tables, blockers)
    _write_csv(output / "MODEL_COMPARISON_SUMMARY.csv", comparison_summary)
    created_files.append("MODEL_COMPARISON_SUMMARY.csv")

    test_text, test_summary, test_blockers = _test_text(output, test_results)
    blockers.extend(test_blockers)
    created_files.append("TEST_RESULTS.txt")

    # Source/preservation verdicts must be evidenced, never assumed by reporting.
    sample = tables.get("ORDER_SAMPLE_AUDIT.csv", pd.DataFrame())
    source_columns = ("source_hashes_valid", "canonical_assembler_valid", "source_verdict")
    preservation_columns = ("protected_files_unchanged", "preservation_verdict", "protected_hashes_valid")
    runtime_source_passed = bool(
        isinstance(provenance.get("source_verdict"), Mapping)
        and provenance["source_verdict"].get("passed") is True
    )
    runtime_preservation_passed = bool(
        isinstance(provenance.get("protected_before_after_verdict"), Mapping)
        and provenance["protected_before_after_verdict"].get("passed") is True
    )
    if not any(column in sample.columns for column in source_columns):
        if not runtime_source_passed:
            blockers.append("source_verdict_missing_from_compact_or_preflight_receipt")
    else:
        source_verdict = _guard_from_group(sample, source_columns, ())
        if source_verdict is not True:
            blockers.append("source_verdict_failed_or_unparseable_from_ORDER_SAMPLE_AUDIT.csv")
    if not any(column in sample.columns for column in preservation_columns):
        if not runtime_preservation_passed:
            blockers.append("preservation_verdict_missing_from_compact_or_verified_preflight_baseline")
    else:
        preservation_verdict = _guard_from_group(sample, preservation_columns, ())
        if preservation_verdict is not True:
            blockers.append("preservation_verdict_failed_or_unparseable_from_ORDER_SAMPLE_AUDIT.csv")

    for filename in ("ORDER_PROTOCOL.md", "ORDER_FROZEN_CONFIG.json", "EVIDENCE_STATUS.md", "ORDER_FEATURE_DICTIONARY.md"):
        if not (output / filename).is_file():
            blockers.append(f"missing_required_control_output:{filename}")

    generated_core = [
        *REQUIRED_OUTPUTS,
        *(f"figures/{spec.stem}.png" for spec in FIGURE_SPECS),
        *(f"figure_sources/{spec.stem}.csv" for spec in FIGURE_SPECS),
    ]
    payloads = _section_payloads(
        output,
        config,
        tables,
        comparison_summary,
        sorted(set(blockers)),
        test_text,
        test_summary,
        provenance,
        generated_core,
    )
    _write_summary(output / "ORDER_RESULTS_SUMMARY.md", "Order Breach and Positive-Severity Results", payloads, chinese=False)
    _write_summary(output / "ORDER_RESULTS_SUMMARY_ZH.md", "订单违约与正向延迟严重度结果", payloads, chinese=True)
    _write_text(output / "BLOCKERS.md", _blockers_markdown(blockers))
    created_files.extend(["ORDER_RESULTS_SUMMARY.md", "ORDER_RESULTS_SUMMARY_ZH.md", "BLOCKERS.md"])

    scripts, commands = _runtime_commands(output)
    prestate_summary = provenance.get("pre_execution_state", {})
    selection_summary = provenance.get("model_selection_freeze", {})
    config_artifact = output / "ORDER_FROZEN_CONFIG.json"
    protocol_artifact = output / "ORDER_PROTOCOL.md"
    manifest = {
        "analysis_id": config.get("analysis_id", "order_breach_severity_v1"),
        "reporting_schema_version": "1.0",
        "frozen_at_utc": config.get("frozen_at_utc"),
        "config_sha256": _sha256(config_artifact) if config_artifact.is_file() else None,
        "config_semantic_hash": hashlib.sha256(_stable_json(config).encode("utf-8")).hexdigest(),
        "protocol_sha256": _sha256(protocol_artifact) if protocol_artifact.is_file() else None,
        "assembler_sha256": prestate_summary.get("assembler_sha256") if isinstance(prestate_summary, Mapping) else None,
        "raw_file_hashes": prestate_summary.get("raw_file_hashes") if isinstance(prestate_summary, Mapping) else None,
        "profile_input_hashes": prestate_summary.get("profile_input_hashes") if isinstance(prestate_summary, Mapping) else None,
        "source_code_hashes": prestate_summary.get("source_code_hashes") if isinstance(prestate_summary, Mapping) else None,
        "protected_baseline_summary": prestate_summary.get("protected_baseline") if isinstance(prestate_summary, Mapping) else None,
        "protected_before_after_verdict": provenance.get("protected_before_after_verdict"),
        "model_selection_freeze_sha256": selection_summary.get("sha256") if isinstance(selection_summary, Mapping) else None,
        "scripts": scripts,
        "commands": commands,
        "compact_inputs": input_hashes,
        "missing_compact_inputs": sorted(set(REQUIRED_COMPACT_CSVS) - set(tables)),
        "figure_slots": [spec.stem for spec in FIGURE_SPECS],
        "figure_no_data_receipts": figure_receipts,
        "completion_report_section_order": [key for key, _, _ in COMPLETION_REPORT_SECTIONS],
        "evidence_labels": comparison_summary.loc[
            comparison_summary["evidence_status"].isin(["Supported", "Mixed", "Not-supported"]),
            ["task", "family", "comparison", "quantile", "evidence_status", "evidence_reason"],
        ].to_dict(orient="records"),
        "test_results": test_summary,
        "provenance": provenance,
        "blockers": sorted(set(blockers)),
        "scope": {
            "thesis_rewrite_run": False,
            "prior_registry_edit_run": False,
            "profile_reselection_run": False,
            "business_policy_optimisation_run": False,
        },
    }
    _write_json(output / "RUN_MANIFEST.json", manifest)
    created_files.append("RUN_MANIFEST.json")

    _write_text(output / "ARTIFACT_VALIDATION_REPORT.md", "# Artifact Validation Report\n\nGeneration in progress.")
    validation_text, overall_pass = _artifact_validation(output, tables, sorted(set(blockers)))
    _write_text(output / "ARTIFACT_VALIDATION_REPORT.md", validation_text)
    created_files.append("ARTIFACT_VALIDATION_REPORT.md")

    figure_paths = [f"figures/{spec.stem}.png" for spec in FIGURE_SPECS]
    figure_source_paths = [f"figure_sources/{spec.stem}.csv" for spec in FIGURE_SPECS]
    manifest["artifact_inventory"] = {
        "compact_inputs": _complete_inventory(output, REQUIRED_COMPACT_CSVS),
        "figures": _complete_inventory(output, figure_paths),
        "figure_sources": _complete_inventory(output, figure_source_paths),
        "required_outputs": _complete_inventory(
            output, REQUIRED_OUTPUTS, omit_self_hash="RUN_MANIFEST.json"
        ),
    }
    manifest["outputs"] = _complete_inventory(
        output, created_files, omit_self_hash="RUN_MANIFEST.json"
    )
    manifest["artifact_validation_pass"] = overall_pass
    _write_manifest_fixed_point(output / "RUN_MANIFEST.json", manifest)

    return {
        "overall_pass": overall_pass,
        "blockers": sorted(set(blockers)),
        "figures": len(FIGURE_SPECS),
        "figure_sources": len(FIGURE_SPECS),
        "model_comparison_rows": int(len(comparison_summary)),
        "created_files": sorted(set(created_files)),
    }


__all__ = [
    "COMPLETION_REPORT_SECTIONS",
    "FIGURE_SPECS",
    "MODEL_COMPARISON_COLUMNS",
    "REQUIRED_COMPACT_CSVS",
    "finalize_reporting",
]
