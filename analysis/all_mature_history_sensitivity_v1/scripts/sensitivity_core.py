from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from analysis.dynamic_profile_profile_validation_v1.scripts import profile_core as pc
from analysis.dynamic_profile_profile_validation_v1.scripts import profile_reporting as pr


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/all_mature_history_sensitivity_v1"
WORK = OUT / "working"
CONFIG_PATH = OUT / "SENSITIVITY_FROZEN_CONFIG.json"
PROFILE_DIR = ROOT / "analysis/dynamic_profile_profile_validation_v1"
FLOAT_FORMAT = "%.12g"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def period_for_date(value: object) -> str:
    date = pd.Timestamp(value).normalize()
    if pd.Timestamp("2017-04-01") <= date < pd.Timestamp("2018-01-01"):
        return "development"
    if pd.Timestamp("2018-01-01") <= date < pd.Timestamp("2018-07-01"):
        return "confirmation"
    if pd.Timestamp("2018-07-01") <= date < pd.Timestamp("2018-08-31"):
        return "terminal"
    return "warmup_or_outside_evaluation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path, sort_by: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if sort_by and not out.empty:
        out = out.sort_values(sort_by, kind="mergesort", na_position="last").reset_index(drop=True)
    out.to_csv(path, index=False, float_format=FLOAT_FORMAT, date_format="%Y-%m-%d", na_rep="")


def write_gzip_csv(frame: pd.DataFrame, path: Path, sort_by: list[str] | None = None) -> None:
    out = frame.copy()
    if sort_by and not out.empty:
        out = out.sort_values(sort_by, kind="mergesort", na_position="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                out.to_csv(
                    text, index=False, float_format=FLOAT_FORMAT,
                    date_format="%Y-%m-%d", na_rep="",
                )


class IncrementalGzipCsv:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(
            filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=6,
        )
        self.text = io.TextIOWrapper(self.compressed, encoding="utf-8", newline="")
        self.header = True
        self.row_count = 0
        self.columns: list[str] | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if self.columns is None:
            self.columns = list(frame.columns)
        elif list(frame.columns) != self.columns:
            raise ValueError("incremental CSV schema changed")
        frame.to_csv(
            self.text, index=False, header=self.header, float_format=FLOAT_FORMAT,
            date_format="%Y-%m-%d", na_rep="",
        )
        self.header = False
        self.row_count += len(frame)

    def close(self) -> None:
        self.text.flush()
        self.text.close()
        self.raw.close()


def all_mature_history_slice(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    target = str(source["target"])
    spec = pc.TARGET_SPECS[target]
    t = pd.Timestamp(snapshot)
    lag = int(source["lag_days"])
    valid = (
        frame["in_canonical"]
        & pc.target_valid_mask(frame, target)
        & frame[str(spec["available"])].lt(t)
    )
    if source["scheme"] == "A":
        pass
    elif source["scheme"] == "C":
        valid &= frame["order_purchase_timestamp"].lt(t - pd.Timedelta(days=lag))
    else:
        raise ValueError(f"unsupported frozen scheme {source['scheme']}")
    columns = list(dict.fromkeys([
        "order_id", "order_purchase_timestamp", str(spec["available"]), str(spec["value"]),
        pc.ENTITY_COLUMNS[str(source["granularity"])], "main_seller_state", "region_od",
        f"expected_{target}",
    ]))
    return frame.loc[valid, [column for column in columns if column in frame.columns]].copy()


def source_and_variant(spec: Mapping[str, object]) -> tuple[dict, dict]:
    source = {
        "target": spec["target"],
        "granularity": spec["granularity"],
        "scheme": spec["scheme"],
        "window_days": int(spec["window_days"]),
        "lag_days": int(spec["lag_days"]),
    }
    variant = {
        "estimator": spec["estimator"],
        "parent_structure": spec["parent_structure"],
        "kappa": spec["kappa"],
        "base_candidate_id": spec["base_candidate_id"],
    }
    return source, variant


def build_all_mature_profile(
    frame: pd.DataFrame,
    spec: Mapping[str, object],
    snapshot: pd.Timestamp,
    profile_config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source, variant = source_and_variant(spec)
    history = all_mature_history_slice(frame, source, snapshot)
    kind = str(pc.TARGET_SPECS[str(spec["target"])]["kind"])
    if kind == "binary":
        profile, parent = pc._binary_profile_variant(history, source, snapshot, variant, profile_config)
    else:
        profile, parent = pc._continuous_profile_variant(history, source, snapshot, variant, profile_config)
    profile = profile.copy()
    profile["candidate_id"] = str(spec["candidate_id"])
    profile["profile_spec_id"] = str(spec["profile_spec_id"])
    profile["min_support"] = int(spec["minimum_support"])
    profile["low_medium_cutoff"] = float(spec["low_medium_cutoff"])
    profile["medium_high_cutoff"] = float(spec["medium_high_cutoff"])
    profile["level"] = pc.assign_frozen_levels(
        profile,
        int(spec["minimum_support"]),
        float(spec["low_medium_cutoff"]),
        float(spec["medium_high_cutoff"]),
    )
    profile["profile_code"] = str(spec["profile_code"])
    profile["history_mode"] = "all_mature"
    profile["effective_history_lower_bound"] = "none"
    profile["period"] = period_for_date(snapshot)
    parent = parent.copy()
    if not parent.empty:
        parent["profile_code"] = str(spec["profile_code"])
        parent["candidate_id"] = str(spec["candidate_id"])
        parent["history_mode"] = "all_mature"
        parent_columns = [
            "parent_id", "parent_score", "parent_support", "global_score",
            "base_candidate_id", "target", "snapshot_date", "parent_model_mean",
            "profile_code", "candidate_id", "history_mode",
        ]
        for column in parent_columns:
            if column not in parent:
                parent[column] = np.nan
        parent = parent[parent_columns]
    available = str(pc.TARGET_SPECS[str(spec["target"])]["available"])
    violations = int(history[available].ge(pd.Timestamp(snapshot)).sum()) if len(history) else 0
    purchase_violations = 0
    if spec["scheme"] == "C" and len(history):
        purchase_violations = int(
            history["order_purchase_timestamp"].ge(
                pd.Timestamp(snapshot) - pd.Timedelta(days=int(spec["lag_days"]))
            ).sum()
        )
    audit = pd.DataFrame([{
        "profile_code": spec["profile_code"],
        "candidate_id": spec["candidate_id"],
        "snapshot_date": pd.Timestamp(snapshot),
        "history_mode": "all_mature",
        "history_rows": len(history),
        "profile_entities": len(profile),
        "strict_availability_violations": violations,
        "scheme_c_purchase_cutoff_violations": purchase_violations,
        "earliest_purchase_timestamp": history["order_purchase_timestamp"].min() if len(history) else pd.NaT,
        "latest_purchase_timestamp": history["order_purchase_timestamp"].max() if len(history) else pd.NaT,
        "latest_label_available_at": history[available].max() if len(history) else pd.NaT,
        "degenerate_fallback_entities": int(
            profile["invalid_reason"].astype(str).eq("degenerate_variance_parent_fallback").sum()
        ) if len(profile) else 0,
    }])
    return profile, parent, audit


def load_selected_90d_anchor_profiles(
    selected: pd.DataFrame,
    anchor_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    wanted_ids = set(selected["candidate_id"].astype(str))
    usecols = [
        *pc.PROFILE_BASE_COLUMNS,
        "candidate_id", "profile_spec_id", "min_support",
        "low_medium_cutoff", "medium_high_cutoff", "level", "unknown_reason", "period",
    ]
    parts: list[pd.DataFrame] = []
    source = PROFILE_DIR / "PROFILE_DAILY_SCORES.csv.gz"
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=150_000, low_memory=False):
        ids = chunk["candidate_id"].astype(str).isin(wanted_ids)
        if not ids.any():
            continue
        dates = pd.to_datetime(chunk["snapshot_date"], errors="coerce").dt.normalize()
        keep = ids & dates.isin(anchor_dates)
        if keep.any():
            part = chunk.loc[keep].copy()
            part["snapshot_date"] = dates.loc[keep]
            parts.append(part)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    if result.duplicated(["candidate_id", "snapshot_date", "entity_id"]).any():
        raise AssertionError("selected 90-day anchor profile store has duplicate primary keys")
    observed_ids = set(result["candidate_id"].astype(str))
    if observed_ids != wanted_ids:
        raise AssertionError(f"selected anchor profiles missing IDs: {wanted_ids - observed_ids}")
    result["history_mode"] = "selected_90d"
    code = selected.set_index("candidate_id")["profile_code"].to_dict()
    result["profile_code"] = result["candidate_id"].map(code)
    return result.sort_values(
        ["profile_code", "snapshot_date", "entity_id"], kind="mergesort",
    ).reset_index(drop=True)


def _daily_stability_pair(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    pair = pd.concat([previous, current], ignore_index=True)
    result = pr.aggregate_daily_stability(pair)
    if len(result) != 1:
        raise AssertionError(f"expected one adjacent-day stability row, got {len(result)}")
    return result


def _tier_transition_rows(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    spec: Mapping[str, object],
) -> list[dict[str, object]]:
    joined = previous[["entity_id", "level"]].merge(
        current[["entity_id", "level"]], on="entity_id", how="inner",
        suffixes=("_from", "_to"), validate="one_to_one",
    )
    period = period_for_date(current["snapshot_date"].iloc[0])
    counts = joined.groupby(["level_from", "level_to"], sort=True).size()
    rows: list[dict[str, object]] = []
    for from_level in ("Unknown", "Low", "Medium", "High"):
        eligible = int(joined["level_from"].astype(str).eq(from_level).sum())
        for to_level in ("Unknown", "Low", "Medium", "High"):
            count = int(counts.get((from_level, to_level), 0))
            rows.append({
                "profile_code": spec["profile_code"],
                "candidate_id": spec["candidate_id"],
                "base_candidate_id": spec["base_candidate_id"],
                "target": spec["target"],
                "granularity": spec["granularity"],
                "period": period,
                "snapshot_date": pd.Timestamp(current["snapshot_date"].iloc[0]),
                "from_level": from_level,
                "to_level": to_level,
                "transition_count": count,
                "eligible_from_count": eligible,
            })
    return rows


def construct_all_mature_stores(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    schedule: pd.DataFrame,
    profile_config: Mapping[str, object],
    cfg: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    snapshots = pd.date_range(
        cfg["snapshots"]["first"], cfg["snapshots"]["last"], freq=cfg["snapshots"]["frequency"],
    )
    if len(snapshots) != int(cfg["snapshots"]["expected_count"]):
        raise AssertionError("daily snapshot count differs from frozen sensitivity config")
    anchor_dates = set(pd.to_datetime(schedule["anchor_date"]).dt.normalize())
    profile_writer = IncrementalGzipCsv(OUT / "ALL_MATURE_PROFILE_DAILY_SCORES.csv.gz")
    parent_writer = IncrementalGzipCsv(OUT / "ALL_MATURE_PROFILE_PARENT_STRUCTURE.csv.gz")
    anchor_profiles: list[pd.DataFrame] = []
    anchor_parents: list[pd.DataFrame] = []
    construction_audits: list[pd.DataFrame] = []
    stability_rows: list[pd.DataFrame] = []
    transition_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    try:
        for spec in cfg["profiles"]:
            print(f"building all-mature daily store for {spec['profile_code']}", flush=True)
            previous: pd.DataFrame | None = None
            candidate_rows = 0
            for index, snapshot in enumerate(snapshots):
                profile, parent, audit = build_all_mature_profile(
                    frame, spec, snapshot, profile_config,
                )
                profile = profile.sort_values("entity_id", kind="mergesort").reset_index(drop=True)
                parent = parent.sort_values("parent_id", kind="mergesort").reset_index(drop=True)
                profile_writer.write(profile)
                parent_writer.write(parent)
                candidate_rows += len(profile)
                construction_audits.append(audit)
                if snapshot in anchor_dates:
                    anchor_profiles.append(profile.copy())
                    anchor_parents.append(parent.copy())
                if previous is not None:
                    stability = _daily_stability_pair(previous, profile)
                    stability["profile_code"] = spec["profile_code"]
                    stability["history_mode"] = "all_mature"
                    stability_rows.append(stability)
                    transition_rows.extend(_tier_transition_rows(previous, profile, spec))
                previous = profile
                if (index + 1) % 100 == 0:
                    print(
                        f"  {spec['profile_code']}: {index + 1}/{len(snapshots)} snapshots",
                        flush=True,
                    )
            index_rows.append({
                "profile_code": spec["profile_code"],
                "candidate_id": spec["candidate_id"],
                "base_candidate_id": spec["base_candidate_id"],
                "snapshot_date_min": snapshots.min(),
                "snapshot_date_max": snapshots.max(),
                "snapshot_count": len(snapshots),
                "row_count": candidate_rows,
                "history_mode": "all_mature",
            })
    finally:
        profile_writer.close()
        parent_writer.close()
    index = pd.DataFrame(index_rows)
    index["store_path"] = "ALL_MATURE_PROFILE_DAILY_SCORES.csv.gz"
    index["store_sha256"] = sha256_file(OUT / "ALL_MATURE_PROFILE_DAILY_SCORES.csv.gz")
    write_csv(index, OUT / "ALL_MATURE_PROFILE_STORE_INDEX.csv", ["profile_code"])
    anchors = pd.concat(anchor_profiles, ignore_index=True)
    parents = pd.concat(anchor_parents, ignore_index=True)
    stability = pd.concat(stability_rows, ignore_index=True)
    transitions_daily = pd.DataFrame(transition_rows)
    transitions = transitions_daily.groupby(
        [
            "profile_code", "candidate_id", "base_candidate_id", "target", "granularity",
            "period", "from_level", "to_level",
        ], sort=True, as_index=False,
    ).agg(
        transition_count=("transition_count", "sum"),
        eligible_from_count=("eligible_from_count", "sum"),
    )
    transitions["transition_probability"] = np.divide(
        transitions["transition_count"], transitions["eligible_from_count"],
        out=np.full(len(transitions), np.nan),
        where=transitions["eligible_from_count"].to_numpy(dtype=float) > 0,
    )
    audit = pd.concat(construction_audits, ignore_index=True)
    write_gzip_csv(
        anchors, WORK / "ALL_MATURE_ANCHOR_PROFILES.csv.gz",
        ["profile_code", "snapshot_date", "entity_id"],
    )
    write_gzip_csv(
        parents, WORK / "ALL_MATURE_ANCHOR_PARENTS.csv.gz",
        ["profile_code", "snapshot_date", "parent_id"],
    )
    write_csv(
        stability, WORK / "ALL_MATURE_DAILY_STABILITY.csv",
        ["profile_code", "snapshot_date"],
    )
    write_csv(
        transitions, WORK / "ALL_MATURE_LEVEL_TRANSITIONS.csv",
        ["profile_code", "period", "from_level", "to_level"],
    )
    write_csv(
        audit, WORK / "ALL_MATURE_CONSTRUCTION_AUDIT.csv",
        ["profile_code", "snapshot_date"],
    )
    return {
        "anchor_profiles": anchors,
        "anchor_parents": parents,
        "daily_stability": stability,
        "level_transitions": transitions,
        "construction_audit": audit,
        "store_index": index,
    }


def _add_mapping_diagnostics(mapped: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    result = mapped.copy()
    entity_index = profile.set_index(profile["entity_id"].astype(str))
    entity = result["entity_id"].astype(str)
    seen = result["mapping_status"].eq("seen")
    freshness = entity.map(entity_index["profile_freshness_days"])
    result["profile_freshness_days"] = pd.to_numeric(freshness, errors="coerce").where(seen)
    result["interval_width"] = result["upper_interval"] - result["lower_interval"]
    return result


def _same_numeric(left: pd.Series, right: pd.Series, atol: float = 1e-12) -> pd.Series:
    x = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isclose(x, y, rtol=0, atol=atol, equal_nan=True), index=left.index)


def match_audit(
    mapped_90d: pd.DataFrame,
    mapped_all: pd.DataFrame,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    left = mapped_90d.sort_values("order_id", kind="mergesort").reset_index(drop=True)
    right = mapped_all.sort_values("order_id", kind="mergesort").reset_index(drop=True)
    joined = left.merge(
        right, on="order_id", how="outer", suffixes=("_90d", "_all"),
        indicator=True, validate="one_to_one",
    )
    both = joined["_merge"].eq("both")
    entity_match = (
        joined.loc[both, "entity_id_90d"].fillna("__NA__").astype(str)
        .eq(joined.loc[both, "entity_id_all"].fillna("__NA__").astype(str))
    )
    observed_match = joined.loc[both, "target_observed_90d"].astype(bool).eq(
        joined.loc[both, "target_observed_all"].astype(bool)
    )
    target_match = _same_numeric(
        joined.loc[both, "target_value_90d"], joined.loc[both, "target_value_all"],
    )
    raw_target_match = _same_numeric(
        joined.loc[both, "raw_target_value_90d"], joined.loc[both, "raw_target_value_all"],
    )
    availability_90d = pd.to_datetime(
        joined.loc[both, "label_available_at_90d"], errors="coerce"
    )
    availability_all = pd.to_datetime(
        joined.loc[both, "label_available_at_all"], errors="coerce"
    )
    availability_match = availability_90d.eq(availability_all) | (
        availability_90d.isna() & availability_all.isna()
    )
    valid_90 = set(left.loc[left["target_observed"], "order_id"].astype(str))
    valid_all = set(right.loc[right["target_observed"], "order_id"].astype(str))
    mapping_changed = int(
        joined.loc[both, "mapping_status_90d"].astype(str).ne(
            joined.loc[both, "mapping_status_all"].astype(str)
        ).sum()
    )
    reasons: list[str] = []
    values = {
        "unmatched_90d_rows": int(joined["_merge"].eq("left_only").sum()),
        "unmatched_all_mature_rows": int(joined["_merge"].eq("right_only").sum()),
        "entity_mapping_mismatch_rows": int((~entity_match).sum()),
        "target_observed_mismatch_rows": int((~observed_match).sum()),
        "target_value_mismatch_rows": int((~target_match).sum()),
        "raw_target_value_mismatch_rows": int((~raw_target_match).sum()),
        "label_available_at_mismatch_rows": int((~availability_match).sum()),
        "valid_future_outcome_id_symmetric_difference": len(valid_90 ^ valid_all),
    }
    for name, value in values.items():
        if value:
            reasons.append(name)
    return {
        **metadata,
        "future_rows_90d": len(left),
        "future_rows_all_mature": len(right),
        "matched_order_rows": int(both.sum()),
        **values,
        "mapping_status_changed_rows_expected_sensitivity": mapping_changed,
        "exact_future_evidence_match": not reasons,
        "unmatched_reason": ";".join(reasons) if reasons else "exact_match",
    }


def favourable_direction(metric: str) -> str:
    lower = {
        "log_loss", "brier", "log_mae", "log_rmse",
        "cold_start_share_all_placed", "cold_start_share_mapping_valid",
        "missing_mapping_share", "posterior_se_median", "posterior_se_p90",
        "interval_width_median", "interval_width_p90",
        "median_absolute_score_change", "p90_absolute_score_change",
        "pct_entities_changing_level", "cold_start_entry_rate", "cold_start_exit_rate",
    }
    higher = {
        "reference_minus_candidate_log_loss", "reference_minus_candidate_brier",
        "reference_minus_candidate_log_mae", "weighted_spearman", "top_quintile_lift",
        "top10_order_lift", "future_seen_coverage", "support_qualified_coverage",
        "support_p10", "support_median", "support_p90", "support_ge5_share",
        "entity_count", "day_to_day_spearman", "top20_jaccard",
        "diagonal_transition_probability",
    }
    if metric in lower:
        return "lower_is_favourable"
    if metric in higher:
        return "higher_is_favourable"
    return "descriptive_no_direction"


def paired_record(
    metadata: Mapping[str, object],
    metric: str,
    value_90d: object,
    value_all: object,
    **extra: object,
) -> dict[str, object]:
    try:
        left = float(value_90d)
    except (TypeError, ValueError):
        left = np.nan
    try:
        right = float(value_all)
    except (TypeError, ValueError):
        right = np.nan
    delta = right - left if np.isfinite(left) and np.isfinite(right) else np.nan
    direction = favourable_direction(metric)
    if not np.isfinite(delta) or direction == "descriptive_no_direction":
        favourable: object = np.nan
    elif abs(delta) <= 1e-12:
        favourable = False
    elif direction == "lower_is_favourable":
        favourable = delta < 0
    else:
        favourable = delta > 0
    return {
        **metadata,
        "metric": metric,
        "selected_90d_value": left,
        "all_mature_value": right,
        "all_mature_minus_90d": delta,
        "favourable_direction": direction,
        "all_mature_favourable": favourable,
        **extra,
    }


def _profile_population_metrics(profile: pd.DataFrame) -> dict[str, float]:
    support = pd.to_numeric(profile["support"], errors="coerce")
    return {
        "entity_count": float(profile["entity_id"].nunique()),
        "support_p10": float(support.quantile(0.10)) if support.notna().any() else np.nan,
        "support_median": float(support.median()) if support.notna().any() else np.nan,
        "support_p90": float(support.quantile(0.90)) if support.notna().any() else np.nan,
        "support_ge5_share": float(support.ge(5).mean()) if len(support) else np.nan,
    }


def _future_support_metrics(mapped: pd.DataFrame) -> dict[str, float]:
    support = pd.to_numeric(mapped.loc[mapped["mapping_status"].ne("missing_mapping"), "history_support"], errors="coerce")
    valid_mapping = int(mapped["mapping_status"].ne("missing_mapping").sum())
    cold = int(mapped["mapping_status"].eq("mapped_cold_start").sum())
    return {
        "future_seen_coverage": float(mapped["mapping_status"].eq("seen").mean()) if len(mapped) else np.nan,
        "support_qualified_coverage": float(
            (mapped["history_support"].ge(5) & mapped["mapping_status"].ne("missing_mapping")).mean()
        ) if len(mapped) else np.nan,
        "cold_start_share_all_placed": cold / len(mapped) if len(mapped) else np.nan,
        "cold_start_share_mapping_valid": cold / valid_mapping if valid_mapping else np.nan,
        "missing_mapping_share": float(mapped["mapping_status"].eq("missing_mapping").mean()) if len(mapped) else np.nan,
        "support_p10": float(support.quantile(0.10)) if support.notna().any() else np.nan,
        "support_median": float(support.median()) if support.notna().any() else np.nan,
        "support_p90": float(support.quantile(0.90)) if support.notna().any() else np.nan,
        "support_ge5_share": float(support.ge(5).mean()) if len(support) else np.nan,
    }


def _uncertainty_metrics(frame: pd.DataFrame) -> dict[str, float]:
    se = pd.to_numeric(frame["posterior_se"], errors="coerce")
    width = pd.to_numeric(frame["interval_width"], errors="coerce") if "interval_width" in frame else (
        pd.to_numeric(frame["upper_interval"], errors="coerce")
        - pd.to_numeric(frame["lower_interval"], errors="coerce")
    )
    freshness = pd.to_numeric(frame["profile_freshness_days"], errors="coerce")
    return {
        "posterior_se_median": float(se.median()) if se.notna().any() else np.nan,
        "posterior_se_p90": float(se.quantile(0.90)) if se.notna().any() else np.nan,
        "interval_width_median": float(width.median()) if width.notna().any() else np.nan,
        "interval_width_p90": float(width.quantile(0.90)) if width.notna().any() else np.nan,
        "freshness_median_days": float(freshness.median()) if freshness.notna().any() else np.nan,
        "freshness_p90_days": float(freshness.quantile(0.90)) if freshness.notna().any() else np.nan,
    }


def _support_stratum(value: object, status: object) -> str:
    if str(status) == "missing_mapping":
        return "missing_mapping"
    number = int(float(value)) if pd.notna(value) else 0
    if number == 0:
        return "support_0_cold_start"
    if number < 5:
        return "support_1_4"
    if number < 10:
        return "support_5_9"
    if number < 20:
        return "support_10_19"
    return "support_20_plus"


def evaluate_anchor_pairs(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    selected_profiles: pd.DataFrame,
    all_profiles: pd.DataFrame,
    all_parents: pd.DataFrame,
    schedule: pd.DataFrame,
    profile_config: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    parent_90d = pd.read_csv(PROFILE_DIR / "PROFILE_PARENT_STRUCTURE.csv", low_memory=False)
    parent_90d["snapshot_date"] = pd.to_datetime(parent_90d["snapshot_date"], errors="coerce").dt.normalize()
    wanted_base = set(selected["base_candidate_id"].astype(str))
    parent_90d = parent_90d.loc[parent_90d["base_candidate_id"].astype(str).isin(wanted_base)].copy()
    persisted = pd.read_csv(PROFILE_DIR / "working/SELECTED_ANCHOR_METRICS.csv", low_memory=False)
    persisted["anchor_date"] = pd.to_datetime(persisted["anchor_date"], errors="coerce").dt.normalize()
    persisted = persisted.loc[
        persisted["base_candidate_id"].astype(str).isin(wanted_base)
        & pd.to_numeric(persisted["support_threshold"], errors="coerce").eq(5)
    ].copy()
    match_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    support_robustness_rows: list[dict[str, object]] = []
    uncertainty_rows: list[dict[str, object]] = []
    reproduction_columns = [
        "future_orders_all_placed", "future_mapping_valid_orders", "future_seen_orders",
        "future_cold_start_orders", "future_missing_mapping_orders", "future_target_valid_orders",
        "log_loss", "brier", "delta_log_loss", "delta_brier", "log_mae", "log_rmse",
        "log_mae_improvement", "weighted_spearman", "top_quintile_lift",
        "future_seen_coverage", "support_qualified_coverage",
    ]
    for spec in selected.sort_values("profile_code", kind="mergesort").to_dict("records"):
        source, _ = source_and_variant(spec)
        base_id = str(spec["base_candidate_id"])
        code = str(spec["profile_code"])
        kind = str(pc.TARGET_SPECS[str(spec["target"])]["kind"])
        print(f"evaluating matched future evidence for {code}", flush=True)
        for row in schedule.to_dict("records"):
            anchor = pd.Timestamp(row["anchor_date"]).normalize()
            horizon = int(row["horizon_days"])
            p90 = selected_profiles.loc[
                selected_profiles["candidate_id"].astype(str).eq(str(spec["candidate_id"]))
                & selected_profiles["snapshot_date"].eq(anchor)
            ].copy()
            pall = all_profiles.loc[
                all_profiles["candidate_id"].astype(str).eq(str(spec["candidate_id"]))
                & pd.to_datetime(all_profiles["snapshot_date"]).eq(anchor)
            ].copy()
            par90 = parent_90d.loc[
                parent_90d["base_candidate_id"].astype(str).eq(base_id)
                & parent_90d["snapshot_date"].eq(anchor)
            ].copy()
            par_all = all_parents.loc[
                all_parents["base_candidate_id"].astype(str).eq(base_id)
                & pd.to_datetime(all_parents["snapshot_date"]).eq(anchor)
            ].copy()
            if p90.empty or pall.empty or par90.empty or par_all.empty:
                raise AssertionError(f"missing profile/parent at {code} {anchor.date()}")
            future = pc.future_cohort(frame, anchor, horizon)
            m90 = _add_mapping_diagnostics(pc.map_future_orders(future, p90, par90, source, base_id), p90)
            mall = _add_mapping_diagnostics(pc.map_future_orders(future, pall, par_all, source, base_id), pall)
            metadata = {
                "profile_code": code,
                "candidate_id": spec["candidate_id"],
                "base_candidate_id": base_id,
                "target": spec["target"],
                "target_kind": kind,
                "period": row["period"],
                "anchor_date": anchor,
                "calendar_month": anchor.to_period("M").strftime("%Y-%m"),
                "horizon_days": horizon,
            }
            audit = match_audit(m90, mall, metadata)
            match_rows.append(audit)
            if not audit["exact_future_evidence_match"]:
                continue
            eval90, _, _ = pc.evaluate_mapped_orders(m90, int(spec["minimum_support"]), profile_config)
            evalall, _, _ = pc.evaluate_mapped_orders(mall, int(spec["minimum_support"]), profile_config)
            threshold = int(spec["minimum_support"])
            common_support_ids = set(
                m90.loc[
                    m90["eligible_for_metric"]
                    & pd.to_numeric(m90["history_support"], errors="coerce").ge(threshold),
                    "order_id",
                ].astype(str)
            ) & set(
                mall.loc[
                    mall["eligible_for_metric"]
                    & pd.to_numeric(mall["history_support"], errors="coerce").ge(threshold),
                    "order_id",
                ].astype(str)
            )
            common90 = m90.loc[m90["order_id"].astype(str).isin(common_support_ids)].copy()
            commonall = mall.loc[mall["order_id"].astype(str).isin(common_support_ids)].copy()
            common90 = common90.sort_values("order_id", kind="mergesort").reset_index(drop=True)
            commonall = commonall.sort_values("order_id", kind="mergesort").reset_index(drop=True)
            if (
                len(common90) != len(commonall)
                or not common90["order_id"].astype(str).eq(commonall["order_id"].astype(str)).all()
                or not np.allclose(
                    pd.to_numeric(common90["target_value"], errors="coerce"),
                    pd.to_numeric(commonall["target_value"], errors="coerce"),
                    rtol=0, atol=0, equal_nan=True,
                )
            ):
                raise AssertionError(f"common-support matched sample failed: {code} {anchor} {horizon}")
            common_eval90, _, _ = pc.evaluate_mapped_orders(common90, threshold, profile_config)
            common_evalall, _, _ = pc.evaluate_mapped_orders(commonall, threshold, profile_config)
            audit["common_support_ge5_orders"] = len(common90)
            audit["common_support_ge5_order_id_match"] = True
            persisted_row = persisted.loc[
                persisted["base_candidate_id"].astype(str).eq(base_id)
                & persisted["period"].astype(str).eq(str(row["period"]))
                & persisted["anchor_date"].eq(anchor)
                & pd.to_numeric(persisted["horizon_days"], errors="coerce").eq(horizon)
            ]
            if len(persisted_row) != 1:
                raise AssertionError(f"persisted selected metric row mismatch for {code} {anchor} {horizon}")
            persisted_record = persisted_row.iloc[0]
            max_diff = 0.0
            compared = 0
            for column in reproduction_columns:
                if column not in eval90 or column not in persisted_record:
                    continue
                x = pd.to_numeric(pd.Series([eval90[column]]), errors="coerce").iloc[0]
                y = pd.to_numeric(pd.Series([persisted_record[column]]), errors="coerce").iloc[0]
                if pd.isna(x) and pd.isna(y):
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    raise AssertionError(f"90-day reproduction nonfinite mismatch: {code} {anchor} {column}")
                max_diff = max(max_diff, abs(float(x) - float(y)))
                compared += 1
            audit["persisted_90d_metrics_compared"] = compared
            audit["persisted_90d_max_absolute_difference"] = max_diff
            audit["persisted_90d_reproduction_pass"] = max_diff <= 1e-9
            if max_diff > 1e-9:
                raise AssertionError(f"90-day persisted metric reproduction failed: {code} {anchor} {horizon}: {max_diff}")
            if kind == "binary":
                metric_map = {
                    "log_loss": "log_loss",
                    "brier": "brier",
                    "reference_minus_candidate_log_loss": "delta_log_loss",
                    "reference_minus_candidate_brier": "delta_brier",
                    "weighted_spearman": "weighted_spearman",
                    "top_quintile_lift": "top_quintile_lift",
                    "top10_order_lift": "top10_order_lift",
                    "future_seen_coverage": "future_seen_coverage",
                    "support_qualified_coverage": "support_qualified_coverage",
                }
            else:
                metric_map = {
                    "log_mae": "log_mae",
                    "log_rmse": "log_rmse",
                    "reference_minus_candidate_log_mae": "log_mae_improvement",
                    "weighted_spearman": "weighted_spearman",
                    "top_quintile_lift": "top_quintile_lift",
                    "future_seen_coverage": "future_seen_coverage",
                    "support_qualified_coverage": "support_qualified_coverage",
                }
            for public_name, core_name in metric_map.items():
                anchor_rows.append(paired_record(
                    metadata, public_name, eval90.get(core_name), evalall.get(core_name),
                    n_future_rows=len(m90),
                    n_valid_future_outcomes_90d=eval90.get("future_target_valid_orders"),
                    n_valid_future_outcomes_all_mature=evalall.get("future_target_valid_orders"),
                    anchor_valid_90d=eval90.get("valid"),
                    anchor_valid_all_mature=evalall.get("valid"),
                ))
                support_robustness_rows.append(paired_record(
                    metadata, public_name,
                    common_eval90.get(core_name), common_evalall.get(core_name),
                    population="common_support_ge5",
                    support_threshold=threshold,
                    n_common_support_orders=len(common90),
                    exact_common_order_match=True,
                    n_valid_future_outcomes_90d=common_eval90.get("future_target_valid_orders"),
                    n_valid_future_outcomes_all_mature=common_evalall.get("future_target_valid_orders"),
                ))
            support_blocks = [
                ("profile_entities", _profile_population_metrics(p90), _profile_population_metrics(pall)),
                ("future_orders", _future_support_metrics(m90), _future_support_metrics(mall)),
            ]
            for population, left_metrics, right_metrics in support_blocks:
                for metric in sorted(left_metrics):
                    support_rows.append(paired_record(
                        metadata, metric, left_metrics[metric], right_metrics[metric],
                        population=population, n_future_rows=len(m90),
                    ))
            uncertainty_blocks = [
                ("profile_entities", p90, pall, "all_support"),
                (
                    "future_seen_orders",
                    m90.loc[m90["mapping_status"].eq("seen")],
                    mall.loc[mall["mapping_status"].eq("seen")],
                    "all_support",
                ),
            ]
            m90 = m90.copy()
            mall = mall.copy()
            m90["support_stratum"] = [
                _support_stratum(value, status)
                for value, status in zip(m90["history_support"], m90["mapping_status"])
            ]
            mall["support_stratum"] = [
                _support_stratum(value, status)
                for value, status in zip(mall["history_support"], mall["mapping_status"])
            ]
            for stratum in ("support_1_4", "support_5_9", "support_10_19", "support_20_plus"):
                uncertainty_blocks.append((
                    "future_seen_orders",
                    m90.loc[m90["support_stratum"].eq(stratum)],
                    mall.loc[mall["support_stratum"].eq(stratum)],
                    stratum,
                ))
            for population, left_frame, right_frame, stratum in uncertainty_blocks:
                left_metrics = _uncertainty_metrics(left_frame)
                right_metrics = _uncertainty_metrics(right_frame)
                for metric in sorted(left_metrics):
                    uncertainty_rows.append(paired_record(
                        metadata, metric, left_metrics[metric], right_metrics[metric],
                        component="anchor_uncertainty", population=population,
                        support_stratum=stratum, n_90d=len(left_frame), n_all_mature=len(right_frame),
                    ))
    match = pd.DataFrame(match_rows)
    if not match["exact_future_evidence_match"].all():
        write_csv(match, OUT / "PROFILE_MATCH_AUDIT.csv", ["profile_code", "anchor_date", "horizon_days"])
        raise AssertionError("matched future evidence audit failed")
    anchor = pd.DataFrame(anchor_rows)
    support = pd.DataFrame(support_rows)
    support_robustness = pd.DataFrame(support_robustness_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    return {
        "match": match,
        "anchor": anchor,
        "support": support,
        "support_robustness": support_robustness,
        "uncertainty_anchor": uncertainty,
    }


def aggregate_anchor_outputs(anchor: pd.DataFrame, cfg: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_month = [
        "profile_code", "candidate_id", "base_candidate_id", "target", "target_kind",
        "period", "calendar_month", "horizon_days", "metric", "favourable_direction",
    ]
    monthly = anchor.groupby(group_month, sort=True, dropna=False, as_index=False).agg(
        selected_90d_value=("selected_90d_value", "median"),
        all_mature_value=("all_mature_value", "median"),
        all_mature_minus_90d=("all_mature_minus_90d", "median"),
        scheduled_anchor_count=("anchor_date", "nunique"),
        paired_nonmissing_count=("all_mature_minus_90d", "count"),
        favourable_anchor_count=("all_mature_favourable", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
    )
    monthly["aggregate_median_difference"] = (
        monthly["all_mature_value"] - monthly["selected_90d_value"]
    )
    monthly["all_mature_minus_90d_aggregation"] = "median_of_paired_anchor_differences"
    monthly["aggregate_median_difference_aggregation"] = (
        "difference_of_separately_aggregated_medians"
    )
    group_summary = [
        "profile_code", "candidate_id", "base_candidate_id", "target", "target_kind",
        "period", "horizon_days", "metric", "favourable_direction",
    ]
    summary = anchor.groupby(group_summary, sort=True, dropna=False, as_index=False).agg(
        selected_90d_value=("selected_90d_value", "median"),
        all_mature_value=("all_mature_value", "median"),
        all_mature_minus_90d=("all_mature_minus_90d", "median"),
        scheduled_anchor_count=("anchor_date", "nunique"),
        paired_nonmissing_count=("all_mature_minus_90d", "count"),
        favourable_anchor_count=("all_mature_favourable", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
    )
    summary["aggregate_median_difference"] = (
        summary["all_mature_value"] - summary["selected_90d_value"]
    )
    summary["all_mature_minus_90d_aggregation"] = "median_of_paired_anchor_differences"
    summary["aggregate_median_difference_aggregation"] = (
        "difference_of_separately_aggregated_medians"
    )
    summary["practical_equivalence_assessment"] = "not_applicable_to_metric"
    for keys, part in summary.groupby(["profile_code", "period", "horizon_days"], sort=True):
        code, period, horizon = keys
        lookup = part.set_index("metric")
        target_kind = str(part["target_kind"].iloc[0])
        if target_kind == "binary":
            ll = abs(float(lookup.loc["log_loss", "aggregate_median_difference"]))
            br = abs(float(lookup.loc["brier", "aggregate_median_difference"]))
            equivalent = (
                ll < float(cfg["practical_equivalence"]["binary_absolute_log_loss"])
                and br < float(cfg["practical_equivalence"]["binary_absolute_brier"])
            )
        else:
            sp = abs(float(lookup.loc["weighted_spearman", "aggregate_median_difference"]))
            selected_mae = float(lookup.loc["log_mae", "selected_90d_value"])
            all_mature_mae = float(lookup.loc["log_mae", "all_mature_value"])
            denominator = max(
                abs(selected_mae), abs(all_mature_mae), np.finfo(float).eps,
            )
            mae = abs(all_mature_mae - selected_mae) / denominator
            equivalent = (
                sp < float(cfg["practical_equivalence"]["continuous_absolute_weighted_spearman"])
                and mae < float(cfg["practical_equivalence"]["continuous_relative_log_mae"])
            )
        mask = (
            summary["profile_code"].eq(code)
            & summary["period"].eq(period)
            & summary["horizon_days"].eq(horizon)
        )
        summary.loc[mask, "practical_equivalence_assessment"] = (
            "within_frozen_tolerances_descriptive_only"
            if equivalent else "outside_frozen_tolerances_numeric_only"
        )
    return monthly, summary


def build_uncertainty_stability_comparison(
    selected: pd.DataFrame,
    all_stability: pd.DataFrame,
    all_transitions: pd.DataFrame,
    anchor_uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    rows = anchor_uncertainty.to_dict("records")
    base_to_code = selected.set_index("base_candidate_id")["profile_code"].to_dict()
    original = pd.read_csv(PROFILE_DIR / "PROFILE_DAILY_STABILITY.csv", low_memory=False)
    original = original.loc[original["base_candidate_id"].astype(str).isin(base_to_code)].copy()
    original["profile_code"] = original["base_candidate_id"].map(base_to_code)
    metrics = [
        "day_to_day_spearman", "median_absolute_score_change", "p90_absolute_score_change",
        "top20_jaccard", "pct_entities_changing_level", "cold_start_entry_rate", "cold_start_exit_rate",
    ]
    for code in sorted(selected["profile_code"].astype(str)):
        spec = selected.loc[selected["profile_code"].astype(str).eq(code)].iloc[0]
        for period in ("development", "confirmation", "terminal"):
            left = original.loc[original["profile_code"].eq(code) & original["period"].eq(period)]
            right = all_stability.loc[all_stability["profile_code"].eq(code) & all_stability["period"].eq(period)]
            metadata = {
                "profile_code": code,
                "candidate_id": spec["candidate_id"],
                "base_candidate_id": spec["base_candidate_id"],
                "target": spec["target"],
                "target_kind": spec["target_kind"],
                "period": period,
                "anchor_date": pd.NaT,
                "calendar_month": "",
                "horizon_days": np.nan,
            }
            for metric in metrics:
                rows.append(paired_record(
                    metadata, metric,
                    pd.to_numeric(left[metric], errors="coerce").median(),
                    pd.to_numeric(right[metric], errors="coerce").median(),
                    component="daily_stability", population="common_entities",
                    support_stratum="all_support", n_90d=len(left), n_all_mature=len(right),
                ))
    original_transitions = pd.read_csv(PROFILE_DIR / "PROFILE_LEVEL_TRANSITIONS.csv", low_memory=False)
    original_transitions = original_transitions.loc[
        original_transitions["base_candidate_id"].astype(str).isin(base_to_code)
    ].copy()
    original_transitions["profile_code"] = original_transitions["base_candidate_id"].map(base_to_code)

    def diagonal_probability(table: pd.DataFrame) -> float:
        if table.empty:
            return np.nan
        numerator = pd.to_numeric(
            table.loc[table["from_level"].astype(str).eq(table["to_level"].astype(str)), "transition_count"],
            errors="coerce",
        ).sum()
        denominators = table.drop_duplicates("from_level")
        denominator = pd.to_numeric(denominators["eligible_from_count"], errors="coerce").sum()
        return float(numerator / denominator) if denominator > 0 else np.nan

    for code in sorted(selected["profile_code"].astype(str)):
        spec = selected.loc[selected["profile_code"].astype(str).eq(code)].iloc[0]
        for period in ("development", "confirmation", "terminal"):
            left = original_transitions.loc[
                original_transitions["profile_code"].eq(code)
                & original_transitions["period"].eq(period)
            ]
            right = all_transitions.loc[
                all_transitions["profile_code"].eq(code)
                & all_transitions["period"].eq(period)
            ]
            metadata = {
                "profile_code": code,
                "candidate_id": spec["candidate_id"],
                "base_candidate_id": spec["base_candidate_id"],
                "target": spec["target"],
                "target_kind": spec["target_kind"],
                "period": period,
                "anchor_date": pd.NaT,
                "calendar_month": "",
                "horizon_days": np.nan,
            }
            rows.append(paired_record(
                metadata, "diagonal_transition_probability",
                diagonal_probability(left), diagonal_probability(right),
                component="communication_tier_transition", population="common_entities",
                support_stratum="all_support", n_90d=len(left), n_all_mature=len(right),
            ))
    return pd.DataFrame(rows)


def terminal_table(
    anchor: pd.DataFrame,
    support: pd.DataFrame,
    uncertainty: pd.DataFrame,
    support_robustness: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = [
        ("standalone_future_metric", anchor),
        ("support_coverage_cold_start", support),
        ("uncertainty_stability", uncertainty),
    ]
    if support_robustness is not None:
        sources.append(("common_support_ge5_robustness", support_robustness))
    for source_name, table in sources:
        part = table.loc[table["period"].astype(str).eq("terminal")].copy()
        group = ["profile_code", "candidate_id", "base_candidate_id", "target", "metric", "favourable_direction"]
        if "horizon_days" in part:
            group.append("horizon_days")
        if "component" in part:
            group.append("component")
        if "population" in part:
            group.append("population")
        if "support_stratum" in part:
            group.append("support_stratum")
        if part.empty:
            continue
        agg = part.groupby(group, sort=True, dropna=False, as_index=False).agg(
            selected_90d_value=("selected_90d_value", "median"),
            all_mature_value=("all_mature_value", "median"),
            all_mature_minus_90d=("all_mature_minus_90d", "median"),
            paired_row_count=("all_mature_minus_90d", "count"),
        )
        agg["aggregate_median_difference"] = (
            agg["all_mature_value"] - agg["selected_90d_value"]
        )
        agg["all_mature_minus_90d_aggregation"] = "median_of_paired_anchor_differences"
        agg["aggregate_median_difference_aggregation"] = (
            "difference_of_separately_aggregated_medians"
        )
        agg["source"] = source_name
        rows.extend(agg.to_dict("records"))
    return pd.DataFrame(rows)


def run_standalone() -> dict[str, object]:
    cfg = load_config()
    profile_config = json.loads((PROFILE_DIR / "PROFILE_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    selected_source = pd.read_csv(PROFILE_DIR / "PROFILE_SELECTED_CANDIDATES.csv", low_memory=False)
    wanted = {item["candidate_id"] for item in cfg["profiles"]}
    selected = selected_source.loc[selected_source["candidate_id"].astype(str).isin(wanted)].copy()
    if len(selected) != 4 or set(selected["candidate_id"].astype(str)) != wanted:
        raise AssertionError("the four frozen selected profiles were not recovered exactly")
    configured = pd.DataFrame(cfg["profiles"])
    verification = selected.merge(
        configured, on="candidate_id", how="inner", validate="one_to_one", suffixes=("_frozen", "_sensitivity"),
    )
    for column in ("base_candidate_id", "profile_spec_id", "target", "granularity", "scheme", "window_days", "lag_days", "estimator", "parent_structure"):
        left = verification[f"{column}_frozen"].astype(str)
        right = verification[f"{column}_sensitivity"].astype(str)
        if not left.eq(right).all():
            raise AssertionError(f"sensitivity config disagrees with frozen selection: {column}")
    if not np.allclose(
        pd.to_numeric(verification["min_support"], errors="coerce"),
        pd.to_numeric(verification["minimum_support"], errors="coerce"),
        rtol=0, atol=0,
    ):
        raise AssertionError("sensitivity minimum support differs from frozen selection")
    for frozen_column, sensitivity_column in (
        ("kappa_frozen", "kappa_sensitivity"),
        ("low_medium_cutoff_frozen", "low_medium_cutoff_sensitivity"),
        ("medium_high_cutoff_frozen", "medium_high_cutoff_sensitivity"),
    ):
        if not np.allclose(
            pd.to_numeric(verification[frozen_column], errors="coerce"),
            pd.to_numeric(verification[sensitivity_column], errors="coerce"),
            rtol=0, atol=1e-12, equal_nan=True,
        ):
            raise AssertionError(f"sensitivity config disagrees with frozen selection: {frozen_column}")
    selected = configured.copy()
    selected["target_kind"] = selected["target"].map(
        lambda target: pc.TARGET_SPECS[str(target)]["kind"]
    )
    schedule = pc.anchor_schedule(profile_config)
    persisted_schedule = pd.read_csv(PROFILE_DIR / "ANCHOR_SCHEDULE.csv")
    for column in ("anchor_date", "future_start", "future_end_exclusive"):
        schedule[column] = pd.to_datetime(schedule[column])
        persisted_schedule[column] = pd.to_datetime(persisted_schedule[column])
    compare_columns = list(schedule.columns)
    if not schedule[compare_columns].reset_index(drop=True).equals(
        persisted_schedule[compare_columns].reset_index(drop=True)
    ):
        raise AssertionError("recomputed frozen anchor schedule differs from persisted schedule")
    print("assembling canonical all-placed and delivered frames", flush=True)
    frame, canonical, _ = pc.build_analysis_frame(cfg["raw_data_dir"])
    thresholds = pc.frozen_tail_thresholds(frame)
    expected_thresholds = {
        "handling_tail_threshold_days": cfg["tail_thresholds_days"]["handling_tail"],
        "transit_tail_threshold_days": cfg["tail_thresholds_days"]["transit_tail"],
    }
    for name, expected in expected_thresholds.items():
        if not math.isclose(float(thresholds[name]), float(expected), rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"tail threshold mismatch: {name}")
    frame = pc.attach_tail_targets(frame, thresholds)
    stores = construct_all_mature_stores(frame, selected, schedule, profile_config, cfg)
    anchor_dates = set(pd.to_datetime(schedule["anchor_date"]).dt.normalize())
    print("loading frozen selected 90-day anchor profiles", flush=True)
    selected_profiles = load_selected_90d_anchor_profiles(selected, anchor_dates)
    pairs = evaluate_anchor_pairs(
        frame, selected, selected_profiles, stores["anchor_profiles"], stores["anchor_parents"],
        schedule, profile_config,
    )
    monthly, summary = aggregate_anchor_outputs(pairs["anchor"], cfg)
    uncertainty = build_uncertainty_stability_comparison(
        selected, stores["daily_stability"], stores["level_transitions"], pairs["uncertainty_anchor"],
    )
    terminal = terminal_table(
        pairs["anchor"], pairs["support"], uncertainty, pairs["support_robustness"]
    )
    write_csv(pairs["match"], OUT / "PROFILE_MATCH_AUDIT.csv", ["profile_code", "anchor_date", "horizon_days"])
    write_csv(
        pairs["anchor"], OUT / "STANDALONE_90D_VS_ALL_MATURE_ANCHOR.csv",
        ["profile_code", "period", "horizon_days", "anchor_date", "metric"],
    )
    write_csv(
        monthly, OUT / "STANDALONE_90D_VS_ALL_MATURE_MONTHLY.csv",
        ["profile_code", "period", "horizon_days", "calendar_month", "metric"],
    )
    write_csv(
        summary, OUT / "STANDALONE_90D_VS_ALL_MATURE_SUMMARY.csv",
        ["profile_code", "period", "horizon_days", "metric"],
    )
    write_csv(
        pairs["support"], OUT / "SUPPORT_COVERAGE_COLDSTART_COMPARISON.csv",
        ["profile_code", "period", "horizon_days", "anchor_date", "population", "metric"],
    )
    write_csv(
        pairs["support_robustness"], OUT / "SUPPORT_GE5_ROBUSTNESS_COMPARISON.csv",
        ["profile_code", "period", "horizon_days", "anchor_date", "metric"],
    )
    write_csv(
        uncertainty, OUT / "UNCERTAINTY_STABILITY_COMPARISON.csv",
        ["profile_code", "period", "component", "horizon_days", "anchor_date", "population", "support_stratum", "metric"],
    )
    write_csv(
        terminal, OUT / "TERMINAL_PROFILE_SENSITIVITY.csv",
        ["profile_code", "source", "horizon_days", "component", "population", "support_stratum", "metric"],
    )
    figure = summary.loc[
        summary["period"].isin(["development", "confirmation"])
        & summary["metric"].isin(["log_loss", "brier", "log_mae", "weighted_spearman", "top_quintile_lift"])
    ].copy()
    figure["source"] = "standalone_summary"
    write_csv(
        figure, OUT / "FIGURE_DATA_90D_VS_ALL_MATURE.csv",
        ["profile_code", "period", "horizon_days", "metric"],
    )
    return {
        "all_placed_rows": len(frame),
        "canonical_delivered_rows": len(canonical),
        "snapshot_count": int(cfg["snapshots"]["expected_count"]),
        "anchor_schedule_rows": len(schedule),
        "matched_comparison_rows": len(pairs["match"]),
        "all_future_evidence_exactly_matched": bool(pairs["match"]["exact_future_evidence_match"].all()),
        "selected_90d_reproduction_max_abs_difference": float(
            pd.to_numeric(pairs["match"]["persisted_90d_max_absolute_difference"], errors="coerce").max()
        ),
        "all_mature_profile_store_rows": int(stores["store_index"]["row_count"].sum()),
        "direct_order_level_branch_executed": False,
    }
