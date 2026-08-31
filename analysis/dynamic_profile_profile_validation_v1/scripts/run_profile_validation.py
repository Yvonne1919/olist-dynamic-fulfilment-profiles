#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", ".cache/dynamic-profile-validation-v1-mpl")
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from analysis.dynamic_profile_profile_validation_v1.scripts import profile_selection
from analysis.dynamic_profile_profile_validation_v1.scripts.fast_stability import (
    compute_daily_score_stability,
)

from analysis.dynamic_profile_profile_validation_v1.scripts.profile_core import (
    ASSEMBLER,
    CONFIG_PATH,
    ENTITY_COLUMNS,
    OUT,
    PROTECTED,
    TARGET_SPECS,
    V11,
    anchor_schedule,
    attach_tail_targets,
    base_candidate_id,
    build_analysis_frame,
    build_profiles,
    candidate_sources,
    candidate_variants,
    compare_hash_maps,
    control_file_hashes,
    evaluate_mapped_orders,
    frozen_tail_thresholds,
    future_cohort,
    generate_row_origin_expectations,
    load_config,
    map_future_orders,
    mask_locked_outcomes_for_development,
    preflight,
    recursive_hashes,
    repository_state,
    sha256_file,
    stability_between_profiles,
    target_valid_mask,
)


WORK = OUT / "working"
FIGURES = OUT / "figures"
FIGURE_SOURCES = OUT / "figure_sources"
STATE_PATH = WORK / "RUN_STATE.json"
PRESTATE_PATH = WORK / "PRE_EXECUTION_STATE.json"
ANCHOR_PATH = OUT / "ANCHOR_SCHEDULE.csv"
TAIL_PATH = OUT / "FROZEN_TAIL_THRESHOLDS.json"
FREEZE_PATH = OUT / "PROFILE_SELECTION_FREEZE.json"
FREEZE_SHA_PATH = OUT / "PROFILE_SELECTION_FREEZE.sha256"
RICH_SCORING_PATH = WORK / "SELECTED_ORDER_SCORING_RICH.csv.gz"
FLOAT_FORMAT = "%.12g"
SCORING_WORK_COLUMNS = [
    "order_id", "purchase_timestamp", "entity_id", "mapping_status", "history_support",
    "cold_start", "profile_score", "shrinkage_score", "raw_score", "parent_score", "global_score", "level",
    "unknown_reason", "target_observed", "target_value", "raw_target_value",
    "label_available_at", "eligible_for_metric", "posterior_se", "lower_interval", "upper_interval",
    "candidate_id", "profile_spec_id", "target", "granularity", "period", "anchor_date",
    "horizon_days",
]
PARETO_PUBLIC_COLUMNS = (
    "candidate_id", "base_candidate_id", "target", "target_family",
    "granularity", "scheme", "window_days", "lag_days", "estimator",
    "parent_structure", "kappa", "support_threshold",
    "n_scheduled_anchors", "n_valid_anchors", "valid_anchor_fraction",
    "median_delta_logloss", "median_delta_brier",
    "median_parent_minus_candidate_mae", "median_weighted_spearman",
    "median_top_quintile_lift", "median_support_qualified_coverage",
    "median_daily_stability_spearman", "minimum_evidence_pass",
    "minimum_evidence_failure_reasons", "pareto_eligible",
    "pareto_nondominated", "dominated_by", "pareto_ineligible_reason",
    "selected_for_confirmation", "selection_rank", "selection_decision",
)
SELECTED_PUBLIC_COLUMNS = (
    "candidate_id", "base_candidate_id", "profile_spec_id", "target",
    "target_family", "granularity", "scheme", "window_days", "lag_days",
    "estimator", "parent_structure", "kappa", "min_support",
    "low_medium_cutoff", "medium_high_cutoff", "selection_rank",
    "selection_decision", "confirmation_label", "confirmation_label_reason",
)
SELECTED_METRIC_COLUMNS = (
    "base_candidate_id", "support_threshold", "candidate_id", "target",
    "granularity", "future_orders_all_placed", "future_mapping_valid_orders",
    "future_seen_orders", "future_cold_start_orders",
    "future_missing_mapping_orders", "future_target_valid_orders",
    "future_events", "future_seen_coverage", "support_qualified_coverage",
    "valid", "invalid_reason", "log_loss", "brier", "citl",
    "calibration_slope", "average_precision", "roc_auc",
    "parent_log_loss", "parent_brier", "parent_citl",
    "parent_calibration_slope", "parent_average_precision", "parent_roc_auc",
    "global_log_loss", "global_brier", "global_citl",
    "global_calibration_slope", "global_average_precision", "global_roc_auc",
    "raw_log_loss", "raw_brier", "raw_citl", "raw_calibration_slope",
    "raw_average_precision", "raw_roc_auc", "delta_log_loss", "delta_brier",
    "top10_order_lift", "log_mae", "log_rmse", "parent_log_mae",
    "parent_log_rmse", "global_log_mae", "global_log_rmse", "raw_log_mae",
    "raw_log_rmse", "log_mae_improvement", "future_mean_days",
    "future_median_days", "n_common_entities", "unweighted_spearman",
    "weighted_spearman", "top_quintile_lift", "profile_spec_id", "period",
    "anchor_date", "calendar_month", "horizon_days", "scheme",
    "window_days", "lag_days", "estimator", "parent_structure", "kappa",
    "target_kind", "information_time_violations", "required_fields_complete",
    "profile_invalid_reason",
)
SELECTED_ENTITY_COLUMNS = (
    "profile_score", "future_mean", "future_raw_mean", "future_support",
    "history_support", "level", "entity_id", "base_candidate_id", "support_threshold",
    "candidate_id", "rank_valid", "invalid_reason", "period", "anchor_date",
    "horizon_days", "profile_spec_id",
)
SELECTED_STRATA_COLUMNS = (
    "base_candidate_id", "candidate_id", "support_threshold",
    "support_stratum", "n_orders", "n_target_valid", "n_entities",
    "event_rate_or_mean", "primary_metric", "primary_improvement", "period",
    "anchor_date", "horizon_days", "profile_spec_id",
)
ORDER_SCORING_PUBLIC_COLUMNS = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "anchor_date", "horizon_days", "order_id", "purchase_timestamp",
    "entity_id", "mapping_status", "history_support", "cold_start",
    "profile_score", "parent_score", "level", "target_observed",
    "target_value", "label_available_at", "eligible_for_metric",
)
CONFIRMATION_MONTH_COLUMNS = (
    "candidate_id", "calendar_month", "target_family", "n_scheduled_anchors",
    "n_valid_anchors", "primary_improvement", "valid",
    "support_ge5_primary_improvement", "high_support_material_reversal",
    "advantage_only_low_support_or_cold",
)
CONFIRMATION_LABEL_COLUMNS = (
    "candidate_id", "confirmation_label", "confirmation_label_reason",
    "n_valid_confirmation_months", "n_favourable_confirmation_months",
    "strict_majority_favourable", "development_primary_improvement",
    "confirmation_aggregate_primary_improvement",
    "aggregate_direction_favourable", "aggregate_magnitude_within_50_percent",
    "high_support_material_reversal",
    "advantage_only_below_support5_or_cold", "failed_strong_conditions",
    "label_is_descriptive_not_stage_gate",
)

_DEVELOPMENT_SHARED_FRAME: pd.DataFrame | None = None
_DEVELOPMENT_SHARED_ANCHORS: pd.DataFrame | None = None
_DEVELOPMENT_SHARED_CONFIG: Mapping[str, object] | None = None
_STABILITY_SHARED_FRAME: pd.DataFrame | None = None
_STABILITY_SHARED_CONFIG: Mapping[str, object] | None = None
_SELECTED_DAILY_SHARED_FRAME: pd.DataFrame | None = None
_SELECTED_DAILY_SHARED_CONFIG: Mapping[str, object] | None = None
DEVELOPMENT_STABILITY_PATH = WORK / "DEVELOPMENT_DAILY_STABILITY.csv"
DEVELOPMENT_STABILITY_META_PATH = WORK / "DEVELOPMENT_DAILY_STABILITY_META.json"
METRIC_SCHEMA_COLUMNS = (
    "candidate_id", "base_candidate_id", "target", "granularity", "period",
    "anchor_date", "calendar_month", "horizon_days", "stratum_type",
    "stratum_value", "metric_name", "reference_id", "aggregation",
    "n_scheduled_anchors", "n_valid_anchors", "n_orders", "n_events",
    "n_entities", "n_common_entities", "estimate", "ci_lower", "ci_upper",
    "valid", "invalid_reason",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def invocation_command() -> str:
    original = getattr(sys, "orig_argv", None)
    return shlex.join(list(original) if original else [sys.executable, *sys.argv])


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value) if canonical else (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path, sort: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    if sort:
        keys = [key for key in sort if key in data.columns]
        if keys:
            data = data.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)
    data.to_csv(path, index=False, float_format=FLOAT_FORMAT, date_format="%Y-%m-%d", na_rep="", lineterminator="\n")


def exact_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with an explicit, ordered schema and no extra columns."""
    result = frame.copy()
    ordered = list(columns)
    for column in ordered:
        if column not in result:
            result[column] = np.nan
    return result.loc[:, ordered]


def write_deterministic_gzip_csv(
    frame: pd.DataFrame,
    path: Path,
    columns: Iterable[str],
    sort: Iterable[str] | None = None,
) -> None:
    """Write an exact-schema gzip CSV with the frozen gzip timestamp zero."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = exact_columns(frame, columns)
    keys = [key for key in (sort or ()) if key in data.columns]
    if keys:
        data = data.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                    data.to_csv(
                        text_handle,
                        index=False,
                        float_format=FLOAT_FORMAT,
                        date_format="%Y-%m-%d",
                        na_rep="",
                        lineterminator="\n",
                    )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def concatenate_csv_files_to_deterministic_gzip(
    paths: Iterable[Path],
    destination: Path,
    columns: Iterable[str],
) -> None:
    """Concatenate ordered exact-schema CSV parts into one deterministic gzip."""
    ordered_columns = list(columns)
    expected_header = (",".join(ordered_columns) + "\n").encode("utf-8")
    existing = [path for path in paths if path.exists() and path.stat().st_size > 0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(expected_header)
                for path in existing:
                    with path.open("rb") as source:
                        header = source.readline()
                        if header != expected_header:
                            raise RuntimeError(f"selected daily CSV header mismatch in {path}")
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            compressed.write(block)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def concatenate_csv_files(paths: Iterable[Path], destination: Path) -> None:
    paths = [path for path in paths if path.exists() and path.stat().st_size > 0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        header: bytes | None = None
        for path in paths:
            with path.open("rb") as source:
                current_header = source.readline()
                if header is None:
                    header = current_header
                    target.write(header)
                elif current_header != header:
                    raise RuntimeError(f"CSV header mismatch while combining {path}")
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"analysis_id": "dynamic_profile_profile_validation_v1", "started_at_utc": utc_now(), "commands": [], "stage_events": []}


def record_event(stage: str, event: str, detail: Mapping[str, object] | None = None) -> dict:
    state = load_state()
    command = invocation_command()
    if command not in state["commands"]:
        state["commands"].append(command)
    state["stage_events"].append({
        "sequence": len(state["stage_events"]) + 1,
        "at_utc": utc_now(), "pid": os.getpid(), "stage": stage, "event": event,
        "detail": dict(detail or {}),
    })
    state["last_stage"] = stage
    state["last_event"] = event
    write_json(STATE_PATH, state)
    return state


def profile_spec_id(base_id: str) -> str:
    return "ps_" + hashlib.sha256(base_id.encode("utf-8")).hexdigest()[:20]


def _validate_promoted_candidate_specs(
    promoted: Sequence[Mapping[str, object]],
    *,
    context: str,
) -> None:
    """Hard-stop on missing, duplicated, or non-deterministic profile IDs."""
    seen_candidate_ids: set[str] = set()
    for index, candidate in enumerate(promoted):
        required = {"candidate_id", "base_candidate_id", "profile_spec_id"}
        missing = sorted(required - set(candidate))
        if missing:
            raise RuntimeError(f"{context}: promoted candidate {index} missing {missing}")
        candidate_id = str(candidate["candidate_id"])
        base_id = str(candidate["base_candidate_id"])
        spec_id = str(candidate["profile_spec_id"])
        if not candidate_id or candidate_id in seen_candidate_ids:
            raise RuntimeError(f"{context}: promoted candidate IDs are blank or duplicated")
        seen_candidate_ids.add(candidate_id)
        if not base_id or spec_id != profile_spec_id(base_id):
            raise RuntimeError(
                f"{context}: promoted profile specification ID mismatch for {candidate_id}"
            )


def source_id(source: Mapping[str, object]) -> str:
    raw = json.dumps(dict(source), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def candidate_catalog() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in candidate_sources():
        for variant in candidate_variants(source):
            row = {**source, **variant}
            row["profile_spec_id"] = profile_spec_id(str(variant["base_candidate_id"]))
            row["target_kind"] = TARGET_SPECS[str(source["target"])]["kind"]
            rows.append(row)
    result = pd.DataFrame(rows)
    if result["base_candidate_id"].duplicated().any():
        raise AssertionError("duplicate base candidate IDs")
    return result.sort_values("base_candidate_id", kind="mergesort").reset_index(drop=True)


def _json_safe(value: object) -> object:
    """Convert pandas/numpy scalars to strict, deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _boolean_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    result = text.map({"true": True, "1": True, "false": False, "0": False})
    return result.fillna(default).astype(bool)


def _masked_development_frame(
    data_dir: Path,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build the raw frame, then discard every locked purchase before evaluation."""
    if not TAIL_PATH.exists():
        raise RuntimeError("selection requires the frozen development tail thresholds")
    frame, canonical, _ = build_analysis_frame(data_dir)
    if len(frame) != 99441 or len(canonical) != 96470:
        raise RuntimeError("authoritative sample counts failed in selection")
    # This is deliberately the first outcome-bearing transformation.  From this
    # point onward the selection process has no row purchased in confirmation.
    development = mask_locked_outcomes_for_development(frame)
    del frame, canonical
    if development["order_purchase_timestamp"].ge(pd.Timestamp("2018-01-01")).any():
        raise RuntimeError("locked confirmation purchase row entered selection")
    frozen_tail = json.loads(TAIL_PATH.read_text(encoding="utf-8"))
    if frozen_tail.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("selection tail threshold/config mismatch")
    development = attach_tail_targets(development, frozen_tail)
    development, nuisance_audit = generate_row_origin_expectations(
        development, config, "2018-01-01",
    )
    write_csv(
        nuisance_audit,
        WORK / "SELECTION_NUISANCE_AUDIT.csv",
        ["target", "origin"],
    )
    return development


def _stability_cache_signature(catalog: pd.DataFrame) -> dict[str, object]:
    scripts = OUT / "scripts"
    base_ids = "\n".join(sorted(catalog["base_candidate_id"].astype(str).unique()))
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "tail_threshold_sha256": sha256_file(TAIL_PATH),
        "profile_core_sha256": sha256_file(scripts / "profile_core.py"),
        "fast_stability_sha256": sha256_file(scripts / "fast_stability.py"),
        "base_candidate_catalog_sha256": hashlib.sha256(base_ids.encode("utf-8")).hexdigest(),
        "development_start": "2017-04-01",
        "development_end_exclusive": "2018-01-01",
    }


def _valid_stability_cache(
    stability: pd.DataFrame,
    catalog: pd.DataFrame,
) -> bool:
    required = {
        "base_candidate_id", "previous_snapshot_date", "snapshot_date",
        "day_to_day_spearman", "valid",
    }
    if not required.issubset(stability.columns):
        return False
    expected_ids = set(catalog["base_candidate_id"].astype(str))
    observed_ids = set(stability["base_candidate_id"].astype(str))
    if observed_ids != expected_ids:
        return False
    if stability.duplicated(["base_candidate_id", "snapshot_date"]).any():
        return False
    counts = stability.groupby("base_candidate_id", sort=False)["snapshot_date"].nunique()
    if len(counts) != len(expected_ids) or not counts.eq(274).all():
        return False
    snapshots = pd.to_datetime(stability["snapshot_date"], errors="coerce")
    return bool(
        snapshots.notna().all()
        and snapshots.min() == pd.Timestamp("2017-04-02")
        and snapshots.max() == pd.Timestamp("2017-12-31")
    )


def _stability_chunk_worker(
    payload: tuple[int, list[dict[str, object]], list[pd.Timestamp], str],
) -> dict[str, object]:
    index, sources, dates, destination_text = payload
    if _STABILITY_SHARED_FRAME is None or _STABILITY_SHARED_CONFIG is None:
        raise RuntimeError("stability worker started without frozen shared inputs")
    result = compute_daily_score_stability(
        _STABILITY_SHARED_FRAME,
        _STABILITY_SHARED_CONFIG,
        dates=pd.DatetimeIndex(dates),
        sources=sources,
        allowed_base_ids=None,
        include_binary_p2=True,
        include_continuous_p2=True,
        max_exact_binary_p2_calls=20_000,
    )
    destination = Path(destination_text)
    write_csv(result, destination, ["base_candidate_id", "snapshot_date"])
    metadata = _json_safe(dict(result.attrs))
    write_json(destination.with_suffix(".json"), metadata, canonical=True)
    return {
        "chunk": index,
        "source_count": len(sources),
        "rows": len(result),
        "path": destination_text,
        "metadata": metadata,
    }


def _load_or_compute_development_stability(
    development: pd.DataFrame,
    config: Mapping[str, object],
    catalog: pd.DataFrame,
    *,
    resume: bool,
    workers: int,
) -> pd.DataFrame:
    signature = _stability_cache_signature(catalog)
    if resume and DEVELOPMENT_STABILITY_PATH.exists() and DEVELOPMENT_STABILITY_META_PATH.exists():
        try:
            metadata = json.loads(DEVELOPMENT_STABILITY_META_PATH.read_text(encoding="utf-8"))
            cached = pd.read_csv(
                DEVELOPMENT_STABILITY_PATH,
                parse_dates=["previous_snapshot_date", "snapshot_date"],
                low_memory=False,
            )
            cache_matches = (
                metadata.get("signature") == signature
                and metadata.get("csv_sha256") == sha256_file(DEVELOPMENT_STABILITY_PATH)
                and _valid_stability_cache(cached, catalog)
            )
            if cache_matches:
                record_event("selection", "development_daily_stability_cache_validated", {
                    "rows": len(cached), "sha256": metadata["csv_sha256"],
                })
                return cached
        except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
            pass

    dates = pd.date_range("2017-04-01", "2017-12-31", freq="D")
    sources = candidate_sources()
    if workers == 1:
        stability = compute_daily_score_stability(
            development,
            config,
            dates=dates,
            sources=sources,
            allowed_base_ids=None,
            include_binary_p2=True,
            include_continuous_p2=True,
            max_exact_binary_p2_calls=20_000,
        )
        engine_metadata = dict(stability.attrs)
    else:
        global _STABILITY_SHARED_FRAME, _STABILITY_SHARED_CONFIG
        _STABILITY_SHARED_FRAME = development
        _STABILITY_SHARED_CONFIG = config
        part_dir = WORK / "development_stability_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        chunks = [sources[index::workers] for index in range(workers)]
        payloads = [
            (
                index,
                chunk,
                list(dates),
                str(part_dir / f"chunk_{index:02d}.csv"),
            )
            for index, chunk in enumerate(chunks) if chunk
        ]
        context = mp.get_context("fork")
        try:
            with context.Pool(processes=min(workers, len(payloads))) as pool:
                records = list(pool.imap_unordered(_stability_chunk_worker, payloads, chunksize=1))
        finally:
            _STABILITY_SHARED_FRAME = None
            _STABILITY_SHARED_CONFIG = None
        records = sorted(records, key=lambda row: int(row["chunk"]))
        stability = pd.concat(
            [pd.read_csv(row["path"], low_memory=False) for row in records],
            ignore_index=True,
        )
        engine_metadata = {
            "engine": "interval_sufficient_statistics_parallel_exact",
            "worker_count": workers,
            "chunks": records,
            "exact_binary_p2_calls": int(sum(
                int(dict(row.get("metadata") or {}).get("exact_binary_p2_calls", 0) or 0)
                for row in records
            )),
        }
    if not _valid_stability_cache(stability, catalog):
        raise RuntimeError("daily stability engine did not return the frozen candidate x 274-day-pair grid")
    write_csv(
        stability,
        DEVELOPMENT_STABILITY_PATH,
        ["base_candidate_id", "snapshot_date"],
    )
    metadata = {
        "signature": signature,
        "csv_sha256": sha256_file(DEVELOPMENT_STABILITY_PATH),
        "row_count": int(len(stability)),
        "engine": _json_safe(engine_metadata),
    }
    write_json(DEVELOPMENT_STABILITY_META_PATH, metadata, canonical=True)
    record_event("selection", "development_daily_stability_computed", {
        "rows": len(stability), "sha256": metadata["csv_sha256"],
        "exact_binary_p2_calls": engine_metadata.get("exact_binary_p2_calls"),
    })
    return stability


def _stability_summary(stability: pd.DataFrame) -> pd.DataFrame:
    work = stability.copy()
    work["valid"] = _boolean_series(work["valid"])
    work["day_to_day_spearman"] = pd.to_numeric(
        work["day_to_day_spearman"], errors="coerce",
    )
    work["stability_usable"] = work["valid"] & np.isfinite(work["day_to_day_spearman"])
    rows: list[dict[str, object]] = []
    for base_id, group in work.groupby("base_candidate_id", sort=True):
        usable = group.loc[group["stability_usable"], "day_to_day_spearman"]
        rows.append({
            "base_candidate_id": str(base_id),
            "daily_stability_spearman": float(usable.median()) if len(usable) else np.nan,
            "n_scheduled_daily_pairs": 274,
            "n_valid_daily_pairs": int(len(usable)),
            "daily_stability_valid_fraction": float(len(usable) / 274.0),
        })
    return pd.DataFrame(rows)


def _complete_development_anchor_grid(
    raw_metrics: pd.DataFrame,
    catalog: pd.DataFrame,
    schedule: pd.DataFrame,
    stability_summary: pd.DataFrame,
) -> pd.DataFrame:
    required = {"candidate_id", "anchor_date", "period", "horizon_days"}
    missing = sorted(required - set(raw_metrics.columns))
    if missing:
        raise RuntimeError(f"development anchor metrics missing {missing}")
    metrics = raw_metrics.copy()
    metrics["anchor_date"] = pd.to_datetime(metrics["anchor_date"], errors="raise")
    metrics = metrics.loc[
        metrics["period"].astype(str).eq("development")
        & pd.to_numeric(metrics["horizon_days"], errors="coerce").eq(7)
    ].copy()
    if metrics.duplicated(["candidate_id", "anchor_date"]).any():
        raise RuntimeError("duplicate candidate/development-7d anchor metric")
    unexpected = sorted(set(metrics["candidate_id"].astype(str)) - set(catalog["candidate_id"].astype(str)))
    if unexpected:
        raise RuntimeError(f"anchor metrics contain candidates outside frozen catalog: {unexpected[:3]}")

    dev_anchors = schedule.loc[
        schedule["period"].astype(str).eq("development")
        & pd.to_numeric(schedule["horizon_days"], errors="coerce").eq(7),
        ["anchor_date"],
    ].drop_duplicates().sort_values("anchor_date", kind="mergesort")
    dev_anchors["anchor_date"] = pd.to_datetime(dev_anchors["anchor_date"], errors="raise")
    if len(dev_anchors) != 39:
        raise RuntimeError(f"selection requires exactly 39 development 7-day anchors, found {len(dev_anchors)}")

    grid = catalog.merge(dev_anchors, how="cross")
    metadata_columns = set(catalog.columns) | {"period", "calendar_month", "horizon_days"}
    payload_columns = [
        column for column in metrics.columns
        if column in {"candidate_id", "anchor_date"} or column not in metadata_columns
    ]
    payload = metrics[payload_columns].copy()
    completed = grid.merge(
        payload,
        on=["candidate_id", "anchor_date"],
        how="left",
        validate="1:1",
    )
    completed["period"] = "development"
    completed["calendar_month"] = completed["anchor_date"].dt.strftime("%Y-%m")
    completed["horizon_days"] = 7
    absent = completed["valid"].isna() if "valid" in completed else pd.Series(True, index=completed.index)
    completed["valid"] = _boolean_series(
        completed["valid"] if "valid" in completed else pd.Series(False, index=completed.index),
    )
    if "invalid_reason" not in completed:
        completed["invalid_reason"] = ""
    completed["invalid_reason"] = completed["invalid_reason"].fillna("").astype(str)
    completed.loc[absent, "invalid_reason"] = "candidate_profile_unavailable"
    completed.loc[~completed["valid"] & completed["invalid_reason"].eq(""), "invalid_reason"] = "invalid_anchor"
    if "information_time_violations" not in completed:
        completed["information_time_violations"] = 0
    completed["information_time_violations"] = pd.to_numeric(
        completed["information_time_violations"], errors="coerce",
    ).fillna(0).astype(int)
    if "required_fields_complete" not in completed:
        completed["required_fields_complete"] = False
    completed["required_fields_complete"] = _boolean_series(completed["required_fields_complete"])
    completed.loc[absent, "required_fields_complete"] = False
    completed = completed.merge(
        stability_summary,
        on="base_candidate_id",
        how="left",
        validate="many_to_one",
    )
    counts = completed.groupby("candidate_id", sort=False)["anchor_date"].nunique()
    if len(counts) != len(catalog) or not counts.eq(39).all():
        raise RuntimeError("completed selection grid is not frozen catalog x 39 anchors")
    return completed.sort_values(["candidate_id", "anchor_date"], kind="mergesort").reset_index(drop=True)


def _selection_results_long(
    wide: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    by_month: bool,
) -> pd.DataFrame:
    metric_definitions = (
        ("median_delta_logloss", "delta_log_loss_candidate_minus_reference", "best_parent_or_global"),
        ("median_delta_brier", "delta_brier_candidate_minus_reference", "best_parent_or_global"),
        ("median_log_loss", "candidate_log_loss", "candidate"),
        ("median_brier", "candidate_brier", "candidate"),
        ("median_candidate_mae", "candidate_log_mae", "candidate"),
        ("median_parent_minus_candidate_mae", "parent_minus_candidate_log_mae", "best_parent_or_global"),
        ("median_primary_improvement", "primary_improvement", "best_parent_or_global"),
        ("median_weighted_spearman", "weighted_future_spearman", "future_entity_outcome"),
        ("median_top_quintile_lift", "top_quintile_future_lift", "all_future_entities"),
        ("median_support_qualified_coverage", "support_qualified_coverage", "all_placed_orders"),
        ("median_seen_coverage", "seen_coverage", "all_placed_orders"),
        ("median_daily_stability_spearman", "daily_stability_spearman", "previous_daily_snapshot"),
        ("valid_anchor_fraction", "valid_anchor_fraction", "39_scheduled_anchors"),
        ("maximum_single_month_positive_share", "maximum_single_month_positive_improvement_share", "positive_primary_improvement"),
        ("high_support_positive_improvement", "high_support_positive_improvement", "support_at_least_5"),
        ("minimum_evidence_pass", "minimum_evidence_pass", "frozen_protocol_gate"),
    )
    anchor_work = anchors.copy()
    anchor_work["calendar_month"] = pd.to_datetime(anchor_work["anchor_date"], errors="raise").dt.strftime("%Y-%m")
    grouping = ["candidate_id", "calendar_month"] if by_month else ["candidate_id"]
    count_rows: list[dict[str, object]] = []
    for key, group in anchor_work.groupby(grouping, sort=True, dropna=False):
        if len(grouping) == 1:
            scalar_key = key[0] if isinstance(key, tuple) and len(key) == 1 else key
            keys = (scalar_key,)
        else:
            keys = tuple(key)
        valid_group = group.loc[_boolean_series(group["valid"])]
        count_rows.append({
            **dict(zip(grouping, keys)),
            "n_orders_count": int(pd.to_numeric(group.get("future_orders_all_placed"), errors="coerce").fillna(0).sum()),
            "n_events_count": float(pd.to_numeric(valid_group.get("future_events"), errors="coerce").fillna(0).sum()) if "future_events" in valid_group else np.nan,
            "n_entities_count": int(pd.to_numeric(valid_group.get("n_common_entities"), errors="coerce").fillna(0).sum()) if "n_common_entities" in valid_group else 0,
            "n_common_entities_count": int(pd.to_numeric(valid_group.get("n_common_entities"), errors="coerce").fillna(0).sum()) if "n_common_entities" in valid_group else 0,
        })
    counts = pd.DataFrame(count_rows)
    table = wide.copy()
    if by_month:
        table = table.rename(columns={"anchor_month": "calendar_month"})
    else:
        table["calendar_month"] = ""
    table = table.merge(counts, on=grouping, how="left", validate="1:1")

    rows: list[dict[str, object]] = []
    for record in table.to_dict("records"):
        for source, metric_name, reference_id in metric_definitions:
            if source not in table.columns:
                continue
            estimate = pd.to_numeric(pd.Series([record.get(source)]), errors="coerce").iloc[0]
            finite = bool(pd.notna(estimate) and np.isfinite(float(estimate)))
            rows.append({
                "candidate_id": record["candidate_id"],
                "base_candidate_id": record["base_candidate_id"],
                "target": record["target"],
                "granularity": record["granularity"],
                "period": "development",
                "anchor_date": "",
                "calendar_month": record.get("calendar_month", ""),
                "horizon_days": 7,
                "stratum_type": "overall",
                "stratum_value": "all",
                "metric_name": metric_name,
                "reference_id": reference_id,
                "aggregation": "median_across_scheduled_anchors_in_month" if by_month else "median_across_39_scheduled_anchors",
                "n_scheduled_anchors": int(record.get("n_anchor_rows", 0) if by_month else record.get("n_scheduled_anchors", 39)),
                "n_valid_anchors": int(record.get("n_valid_anchors", 0)),
                "n_orders": int(record.get("n_orders_count", 0)),
                "n_events": (
                    record.get("n_events_count", np.nan)
                    if profile_selection.target_family(str(record["target"])) == "binary"
                    else np.nan
                ),
                "n_entities": int(record.get("n_entities_count", 0)),
                "n_common_entities": int(record.get("n_common_entities_count", 0)),
                "estimate": float(estimate) if finite else np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "valid": finite,
                "invalid_reason": "" if finite else "no_finite_valid_anchor_estimate",
            })
    return pd.DataFrame(rows, columns=METRIC_SCHEMA_COLUMNS)


def _complete_selection_support_strata(
    strata: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Persist explicit zero rows when a valid anchor has no low/high support bin."""
    required = {
        "candidate_id", "anchor_date", "support_stratum", "primary_improvement",
    }
    missing = sorted(required - set(strata.columns))
    if missing:
        raise RuntimeError(f"development support strata missing {missing}")
    result = strata.copy()
    result["anchor_date"] = pd.to_datetime(result["anchor_date"], errors="raise")
    low_names = {"support_0_cold_start", "support_1_4"}
    high_names = {"support_5_9", "support_10_19", "support_20_plus"}
    valid_anchors = anchors.loc[_boolean_series(anchors["valid"])].copy()
    additions: list[dict[str, object]] = []
    indexed = {
        (str(candidate_id), pd.Timestamp(anchor)): set(group["support_stratum"].astype(str))
        for (candidate_id, anchor), group in result.groupby(
            ["candidate_id", "anchor_date"], sort=False,
        )
    }
    for row in valid_anchors.to_dict("records"):
        key = (str(row["candidate_id"]), pd.Timestamp(row["anchor_date"]))
        present = indexed.get(key, set())
        missing_bins: list[str] = []
        if not (present & low_names):
            missing_bins.append("support_1_4")
        if not (present & high_names):
            missing_bins.append("support_5_9")
        for support_stratum in missing_bins:
            additions.append({
                "base_candidate_id": row["base_candidate_id"],
                "candidate_id": row["candidate_id"],
                "support_threshold": int(row["support_threshold"]),
                "support_stratum": support_stratum,
                "n_orders": 0,
                "n_target_valid": 0,
                "n_entities": 0,
                "event_rate_or_mean": np.nan,
                "primary_metric": (
                    "delta_log_loss" if row["target_family"] == "binary"
                    else "log_mae_improvement"
                ),
                "primary_improvement": 0.0,
                "profile_spec_id": row["profile_spec_id"],
                "period": "development",
                "anchor_date": row["anchor_date"],
                "calendar_month": pd.Timestamp(row["anchor_date"]).strftime("%Y-%m"),
                "horizon_days": 7,
                "explicit_zero_count_stratum": True,
            })
    if "explicit_zero_count_stratum" not in result:
        result["explicit_zero_count_stratum"] = False
    else:
        result["explicit_zero_count_stratum"] = _boolean_series(
            result["explicit_zero_count_stratum"],
        )
    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True, sort=False)
    duplicates = result.duplicated(
        ["candidate_id", "anchor_date", "support_stratum"], keep=False,
    )
    if duplicates.any():
        raise RuntimeError("duplicate completed candidate/anchor/support stratum")
    return result.sort_values(
        ["candidate_id", "anchor_date", "support_stratum"], kind="mergesort",
    ).reset_index(drop=True)


def run_preflight(data_dir: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if FREEZE_PATH.exists() or FREEZE_SHA_PATH.exists():
        raise RuntimeError(
            "a new preflight cannot reset run provenance after a selection freeze exists"
        )
    previous_state_sha = sha256_file(STATE_PATH) if STATE_PATH.exists() else ""
    write_json(
        STATE_PATH,
        {
            "analysis_id": "dynamic_profile_profile_validation_v1",
            "started_at_utc": utc_now(),
            "commands": [],
            "stage_events": [],
            "preflight_reset": {
                "previous_state_sha256": previous_state_sha,
                "reason": "formal_run_start_before_selection_freeze",
            },
        },
    )
    record_event(
        "preflight", "started", {"previous_state_sha256": previous_state_sha}
    )
    state = preflight(data_dir)
    state.update({
        "captured_at_utc": utc_now(),
        "command": invocation_command(),
        "command_working_directory": str(Path.cwd()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(OUT / "PROFILE_PROTOCOL.md"),
        "source_code_hashes": recursive_hashes(OUT / "scripts"),
        "charter": {
            "path": "docs/omitted-private-controls/OLIST_PROFILE_PIVOT_PROJECT_CHARTER_2026-08-21.md",
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        },
    })
    charter = Path(state["charter"]["path"])
    state["charter"]["sha256"] = sha256_file(charter)
    write_json(PRESTATE_PATH, state)
    config = load_config()
    anchors = anchor_schedule(config)
    write_csv(anchors, ANCHOR_PATH, ["period", "horizon_days", "anchor_date"])
    record_event("preflight", "passed", {"prestate_sha256": sha256_file(PRESTATE_PATH), "anchor_schedule_sha256": sha256_file(ANCHOR_PATH)})
    print(json.dumps({"stage": "preflight", "status": "passed", "anchor_rows": len(anchors)}, sort_keys=True), flush=True)


def require_preflight() -> dict:
    if not PRESTATE_PATH.exists() or not ANCHOR_PATH.exists():
        raise RuntimeError("preflight artifacts missing")
    before = json.loads(PRESTATE_PATH.read_text(encoding="utf-8"))
    ok, detail = compare_hash_maps(before["protected_hashes"])
    if not ok:
        raise RuntimeError(f"protected paths changed since preflight: {detail}")
    if before.get("control_file_hashes") != control_file_hashes():
        raise RuntimeError("project control files changed since preflight")
    if sha256_file(CONFIG_PATH) != before["config_sha256"]:
        raise RuntimeError("frozen config changed after preflight")
    if sha256_file(OUT / "PROFILE_PROTOCOL.md") != before["protocol_sha256"]:
        raise RuntimeError("profile protocol changed after preflight")
    if recursive_hashes(OUT / "scripts") != before.get("source_code_hashes"):
        raise RuntimeError("V1 profile source code changed after preflight")
    return before


def _add_metric_context(
    record: dict[str, object],
    source: Mapping[str, object], profile: pd.Series,
    anchor: pd.Timestamp, horizon: int, period: str,
) -> dict[str, object]:
    raw_profile_reason = profile.get("invalid_reason", "")
    profile_invalid_reason = "" if pd.isna(raw_profile_reason) else str(raw_profile_reason)
    # A continuous EB degeneracy is an auditable parent fallback, not a valid
    # EB candidate-anchor for selection.  Keep the scores, but exclude the
    # anchor from the 30/39 evidence count.
    if profile_invalid_reason == "degenerate_variance_parent_fallback":
        record["valid"] = False
        existing = str(record.get("invalid_reason", "") or "")
        record["invalid_reason"] = ";".join(
            item for item in (existing, profile_invalid_reason) if item
        )
    record.update({
        "profile_spec_id": profile_spec_id(str(profile["base_candidate_id"])),
        "period": period, "anchor_date": anchor, "calendar_month": anchor.strftime("%Y-%m"),
        "horizon_days": int(horizon), "scheme": source["scheme"],
        "window_days": int(source["window_days"]), "lag_days": int(source["lag_days"]),
        "estimator": profile["estimator"], "parent_structure": profile["parent_structure"],
        "kappa": profile["kappa"], "target_kind": TARGET_SPECS[str(source["target"])]["kind"],
        "information_time_violations": 0,
        "required_fields_complete": bool(
            pd.notna(profile["score"]) and pd.notna(profile["support"])
            and pd.notna(profile["parent_score"])
        ),
        "profile_invalid_reason": profile_invalid_reason,
    })
    return record


def _construction_source_audit(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
) -> dict[str, object]:
    """Audit source counts and next-seven-day entity coverage around exclusion."""
    target = str(source["target"])
    granularity = str(source["granularity"])
    spec = TARGET_SPECS[target]
    available_column = str(spec["available"])
    raw_value_column = str(spec["raw_value"])
    entity_column = ENTITY_COLUMNS[granularity]
    t = pd.Timestamp(snapshot)
    window = int(source["window_days"])
    lag = int(source["lag_days"])
    if str(source["scheme"]) == "A":
        interval_axis = "label_available_at"
        interval_end = t
        interval_start = t - pd.Timedelta(days=window)
        cohort = (
            frame["in_canonical"]
            & frame[available_column].ge(interval_start)
            & frame[available_column].lt(interval_end)
        )
    elif str(source["scheme"]) == "C":
        interval_axis = "purchase_timestamp"
        interval_end = t - pd.Timedelta(days=lag)
        interval_start = interval_end - pd.Timedelta(days=window)
        cohort = (
            frame["in_canonical"]
            & frame["order_purchase_timestamp"].ge(interval_start)
            & frame["order_purchase_timestamp"].lt(interval_end)
        )
    else:
        raise ValueError(f"unsupported construction scheme {source['scheme']}")
    available_before_snapshot = frame[available_column].notna() & frame[available_column].lt(t)
    observed = cohort & available_before_snapshot & frame[raw_value_column].notna()
    negative = pd.Series(False, index=frame.index)
    if target.startswith("handling"):
        negative = observed & pd.to_numeric(frame["handling_duration"], errors="coerce").lt(0)
    elif target.startswith("transit"):
        negative = observed & pd.to_numeric(frame["transit_duration"], errors="coerce").lt(0)
    valid = cohort & available_before_snapshot & target_valid_mask(frame, target)
    observed_count = int(observed.sum())
    valid_count = int(valid.sum())
    negative_count = int(negative.sum())
    observed_entities = set(
        frame.loc[observed, entity_column].dropna().astype(str)
    )
    valid_entities = set(
        frame.loc[valid, entity_column].dropna().astype(str)
    )
    if not valid_entities.issubset(observed_entities):
        raise RuntimeError("valid historical entity domain is not a subset of observed history")
    future = future_cohort(frame, t, 7)
    future_entities = future[entity_column]
    future_mapping_valid = future_entities.notna()
    future_entity_strings = future_entities.astype("string")
    coverage_before = (
        float(
            (
                future_mapping_valid
                & future_entity_strings.isin(observed_entities)
            ).sum()
            / len(future)
        )
        if len(future)
        else np.nan
    )
    coverage_after = (
        float(
            (
                future_mapping_valid
                & future_entity_strings.isin(valid_entities)
            ).sum()
            / len(future)
        )
        if len(future)
        else np.nan
    )
    if np.isfinite(coverage_before) and np.isfinite(coverage_after) and coverage_after > coverage_before:
        raise RuntimeError("post-exclusion future entity coverage exceeds pre-exclusion coverage")
    affected_entities = int(frame.loc[negative, entity_column].dropna().astype(str).nunique())
    maximum_available = frame.loc[observed, available_column].max()
    maximum_purchase = frame.loc[cohort, "order_purchase_timestamp"].max()
    last_mature = frame.loc[valid, available_column].max()
    strict_asof = bool(pd.isna(maximum_available) or pd.Timestamp(maximum_available) < t)
    return {
        **source,
        "snapshot_date": t,
        "period": "development",
        "history_sample": "canonical_delivered_96470_target_valid",
        "future_denominator_sample": "all_placed_99441_next_7d_purchase_cohort",
        "source_interval_axis": interval_axis,
        "source_interval_start": interval_start,
        "source_interval_end": interval_end,
        "availability_cutoff": t,
        "entity_domain_count": int(len(valid_entities)),
        "source_orders_observed": observed_count,
        "source_orders_valid": valid_count,
        "source_orders_excluded_negative": negative_count,
        "affected_entities_negative": affected_entities,
        "coverage_before_negative_exclusion": coverage_before,
        "coverage_after_negative_exclusion": coverage_after,
        "max_source_purchase_at": maximum_purchase,
        "max_source_label_available_at": maximum_available,
        "last_mature_outcome_date": last_mature,
        "strict_asof_pass": strict_asof,
        "window_pass": True,
        "valid": strict_asof,
        "invalid_reason": "" if strict_asof else "source_label_at_or_after_snapshot",
    }


def evaluate_source_development(
    frame: pd.DataFrame,
    source: Mapping[str, object],
    anchors: pd.DataFrame,
    config: Mapping[str, object],
    destination: Path,
) -> None:
    metric_rows: list[dict[str, object]] = []
    strata_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for index, anchor in enumerate(sorted(anchors["anchor_date"].unique())):
        t = pd.Timestamp(anchor)
        profiles, parents = build_profiles(frame, source, t, config)
        if profiles.empty:
            continue
        construction_audit = _construction_source_audit(frame, source, t)
        construction_audit.update({
            "profile_spec_count": int(profiles["base_candidate_id"].nunique()),
            "profile_rows": int(len(profiles)),
            "parent_rows": int(len(parents)),
            "cold_start_rows": int(pd.to_numeric(profiles["cold_start"], errors="coerce").fillna(0).sum()),
        })
        audit_rows.append(construction_audit)
        scheduled = anchors.loc[anchors["anchor_date"].eq(t)]
        for horizon in sorted(scheduled["horizon_days"].unique()):
            future = future_cohort(frame, t, int(horizon))
            for base_id in sorted(profiles["base_candidate_id"].unique()):
                sub = profiles.loc[profiles["base_candidate_id"].eq(base_id)]
                profile_row = sub.iloc[0]
                mapped = map_future_orders(future, profiles, parents, source, base_id)
                if mapped.empty:
                    continue
                thresholds = [int(value) for value in config["levels"]["support_candidates"]]
                base_metrics, _, base_strata = evaluate_mapped_orders(mapped, thresholds[0], config)
                if not base_metrics:
                    continue
                estimator_reasons = sorted({
                    str(value) for value in sub["invalid_reason"].dropna().astype(str)
                    if str(value).strip()
                })
                for threshold in thresholds:
                    metrics = dict(base_metrics)
                    metrics["support_threshold"] = threshold
                    metrics["candidate_id"] = f"{base_id}|min_support={threshold}"
                    metrics["support_qualified_coverage"] = float(
                        (
                            mapped["history_support"].ge(threshold)
                            & mapped["mapping_status"].ne("missing_mapping")
                        ).mean()
                    ) if len(mapped) else np.nan
                    record = _add_metric_context(
                        metrics, source, profile_row, t, int(horizon), "development",
                    )
                    if estimator_reasons:
                        record["valid"] = False
                        existing = str(record.get("invalid_reason", "")).strip()
                        reason = "profile_estimator_invalid:" + ";".join(estimator_reasons)
                        record["invalid_reason"] = ";".join(value for value in (existing, reason) if value)
                        record["required_fields_complete"] = False
                    metric_rows.append(record)
                    if not base_strata.empty:
                        strata = base_strata.copy()
                        strata["support_threshold"] = threshold
                        strata["candidate_id"] = f"{base_id}|min_support={threshold}"
                        strata["profile_spec_id"] = profile_spec_id(base_id)
                        strata["period"] = "development"
                        strata["anchor_date"] = t
                        strata["calendar_month"] = t.strftime("%Y-%m")
                        strata["horizon_days"] = int(horizon)
                        strata_rows.extend(strata.to_dict("records"))
        if (index + 1) % 5 == 0:
            print(json.dumps({"source": source_id(source), "anchors_done": index + 1, "anchors_total": anchors["anchor_date"].nunique()}, sort_keys=True), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(metric_rows), destination / "metrics.csv", ["candidate_id", "horizon_days", "anchor_date"])
    write_csv(pd.DataFrame(strata_rows), destination / "strata.csv", ["candidate_id", "horizon_days", "anchor_date", "support_stratum"])
    write_csv(pd.DataFrame(audit_rows), destination / "audit.csv", ["target", "granularity", "scheme", "window_days", "lag_days", "snapshot_date"])
    write_json(destination / "complete.json", {"source": dict(source), "completed_at_utc": utc_now(), "metrics_sha256": sha256_file(destination / "metrics.csv")})


def _development_source_worker(
    payload: tuple[int, int, dict[str, object], str],
) -> dict[str, object]:
    number, total, source, destination_text = payload
    if (
        _DEVELOPMENT_SHARED_FRAME is None
        or _DEVELOPMENT_SHARED_ANCHORS is None
        or _DEVELOPMENT_SHARED_CONFIG is None
    ):
        raise RuntimeError("development worker started without frozen shared inputs")
    started = time.monotonic()
    print(json.dumps({
        "stage": "development", "source_number": number,
        "source_total": total, "source": source, "worker_pid": os.getpid(),
    }, sort_keys=True), flush=True)
    evaluate_source_development(
        _DEVELOPMENT_SHARED_FRAME,
        source,
        _DEVELOPMENT_SHARED_ANCHORS,
        _DEVELOPMENT_SHARED_CONFIG,
        Path(destination_text),
    )
    return {
        "source_number": number,
        "source_id": source_id(source),
        "seconds": round(time.monotonic() - started, 3),
        "worker_pid": os.getpid(),
    }


def combine_source_parts(base: Path, filename: str) -> pd.DataFrame:
    files = sorted(base.glob(f"*/{filename}"))
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def verify_selection_freeze(token: str | None) -> dict:
    """Validate the immutable token before any confirmation/terminal frame can be built."""
    if token is None:
        raise RuntimeError("confirmation access denied: --freeze-token is required")
    if not FREEZE_PATH.exists() or not FREEZE_SHA_PATH.exists():
        raise RuntimeError("confirmation access denied: selection freeze or SHA sidecar missing")
    calculated = sha256_file(FREEZE_PATH)
    sidecar = FREEZE_SHA_PATH.read_text(encoding="utf-8").strip().split()[0]
    if calculated != sidecar or calculated != token:
        raise RuntimeError("confirmation access denied: selection freeze hash mismatch")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    required = {
        "frozen_config_sha256", "development_artifact_hashes", "development_anchor_schedule_sha256",
        "promoted_candidates", "confirmation_access_guard", "development_pid",
    }
    missing = sorted(required - set(freeze))
    if missing:
        raise RuntimeError(f"confirmation access denied: malformed freeze missing {missing}")
    _validate_promoted_candidate_specs(
        list(freeze["promoted_candidates"]),
        context="confirmation access denied",
    )
    if freeze["frozen_config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("confirmation access denied: frozen config changed")
    if freeze["development_anchor_schedule_sha256"] != sha256_file(ANCHOR_PATH):
        raise RuntimeError("confirmation access denied: anchor schedule changed")
    source_hashes = dict(freeze.get("source_hashes") or {})
    if source_hashes.get("scripts") != recursive_hashes(OUT / "scripts"):
        raise RuntimeError("confirmation access denied: source scripts changed after selection freeze")
    expected_assembler = dict(source_hashes.get("assembler") or {}).get(str(ASSEMBLER))
    if expected_assembler != sha256_file(ASSEMBLER):
        raise RuntimeError("confirmation access denied: canonical assembler changed after selection freeze")
    if freeze.get("protocol_sha256") != sha256_file(OUT / "PROFILE_PROTOCOL.md"):
        raise RuntimeError("confirmation access denied: profile protocol changed after selection freeze")
    if int(freeze["development_pid"]) == os.getpid():
        raise RuntimeError("confirmation access denied: fresh process required after selection freeze")
    for relative, expected in freeze["development_artifact_hashes"].items():
        path = OUT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"confirmation access denied: development artifact mismatch {relative}")
    state = load_state()
    events = [event["event"] for event in state.get("stage_events", [])]
    if "selection_freeze_written" not in events or "selection_freeze_hashed" not in events:
        raise RuntimeError("confirmation access denied: ordered freeze events absent")
    record_event("confirmation", "freeze_token_validated_before_label_open", {"freeze_sha256": calculated})
    return freeze


def assign_levels(mapped: pd.DataFrame, candidate: Mapping[str, object]) -> pd.DataFrame:
    result = mapped.copy()
    threshold = int(candidate["min_support"])
    q33 = pd.to_numeric(pd.Series([candidate.get("low_medium_cutoff")]), errors="coerce").iloc[0]
    q67 = pd.to_numeric(pd.Series([candidate.get("medium_high_cutoff")]), errors="coerce").iloc[0]
    if not np.isfinite(q33) or not np.isfinite(q67) or q33 > q67:
        result["level"] = "Unknown"
        result["unknown_reason"] = "invalid_frozen_cutoffs"
        return result
    missing_mapping = result["mapping_status"].eq("missing_mapping")
    low_support = result["history_support"].lt(threshold)
    cold = result["cold_start"].astype(bool)
    score_nonfinite = ~np.isfinite(pd.to_numeric(result["profile_score"], errors="coerce"))
    interval_missing = ~np.isfinite(result["lower_interval"]) | ~np.isfinite(result["upper_interval"])
    spans_both = result["lower_interval"].le(q33) & result["upper_interval"].ge(q67)
    unknown = missing_mapping | low_support | cold | score_nonfinite | interval_missing | spans_both
    score = result["profile_score"]
    result["level"] = np.select(
        [unknown, score.le(q33), score.le(q67)], ["Unknown", "Low", "Medium"], default="High",
    )
    result["unknown_reason"] = np.select(
        [missing_mapping, cold, low_support, score_nonfinite, interval_missing, spans_both],
        ["missing_mapping", "cold_start", "below_min_support", "nonfinite_score", "nonfinite_interval", "interval_spans_both_cutoffs"],
        default="",
    )
    result["min_support"] = threshold
    result["low_medium_cutoff"] = q33
    result["medium_high_cutoff"] = q67
    return result


def source_from_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "target": candidate["target"], "granularity": candidate["granularity"],
        "scheme": candidate["scheme"], "window_days": int(candidate["window_days"]),
        "lag_days": int(candidate["lag_days"]),
    }


def matched_p1_base_id(
    source: Mapping[str, object],
    candidate: Mapping[str, object],
) -> str | None:
    """Return the frozen P1 comparator for a selected P2 profile."""

    estimator = str(candidate.get("estimator", ""))
    if estimator != "P2":
        return None
    return base_candidate_id(
        source,
        "P1",
        str(candidate["parent_structure"]),
        candidate.get("kappa"),
    )


def attach_shrinkage_score(
    mapped: pd.DataFrame,
    matched_p1: pd.DataFrame | None,
    estimator: str,
) -> pd.DataFrame:
    """Attach the P1-only comparator without changing candidate selection."""

    result = mapped.copy()
    if estimator == "P0":
        result["shrinkage_score"] = np.nan
        return result
    if estimator == "P1":
        result["shrinkage_score"] = pd.to_numeric(
            result["profile_score"], errors="coerce"
        )
        return result
    if estimator != "P2":
        raise RuntimeError(f"unrecognised selected estimator: {estimator}")
    if matched_p1 is None or matched_p1.empty:
        raise RuntimeError("selected P2 evaluation is missing its matched P1 mapping")
    if result["order_id"].isna().any() or result["order_id"].duplicated().any():
        raise RuntimeError("selected P2 mapping has missing or duplicate order_id values")
    if matched_p1["order_id"].isna().any() or matched_p1["order_id"].duplicated().any():
        raise RuntimeError("matched P1 mapping has missing or duplicate order_id values")
    order_before = result["order_id"].astype(str).tolist()
    comparator = matched_p1[["order_id", "profile_score"]].rename(
        columns={"profile_score": "shrinkage_score"}
    )
    result = result.merge(
        comparator,
        on="order_id",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator="_matched_p1_merge",
    )
    if not result["_matched_p1_merge"].eq("both").all():
        raise RuntimeError("selected P2 and matched P1 future order mappings differ")
    result = result.drop(columns="_matched_p1_merge")
    if result["order_id"].astype(str).tolist() != order_before:
        raise RuntimeError("matched P1 merge changed selected P2 future order order")
    return result


def attach_frozen_entity_levels(
    entities: pd.DataFrame,
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the order-level frozen level to one level per future entity."""

    level_rows = mapped.loc[
        mapped["entity_id"].notna(), ["entity_id", "level"]
    ].copy()
    if not level_rows.empty:
        level_counts = level_rows.groupby(
            "entity_id", sort=True, dropna=False
        )["level"].nunique(dropna=False)
        conflicts = level_counts.loc[level_counts.ne(1)]
        if not conflicts.empty:
            examples = conflicts.index.astype(str).tolist()[:5]
            raise RuntimeError(
                "future entity has non-unique frozen levels: " + ",".join(examples)
            )
        level_rows = level_rows.drop_duplicates("entity_id", keep="first")
    if entities.empty:
        result = entities.copy()
        result["level"] = pd.Series(dtype="object")
        return result
    if entities["entity_id"].isna().any() or entities["entity_id"].duplicated().any():
        raise RuntimeError("future entity evaluation rows have missing or duplicate entity_id values")
    result = entities.merge(
        level_rows,
        on="entity_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result["level"].isna().any():
        missing = result.loc[result["level"].isna(), "entity_id"].astype(str).tolist()[:5]
        raise RuntimeError(
            "future entity evaluation is missing its frozen level: " + ",".join(missing)
        )
    return result


def evaluate_selected_source(
    frame: pd.DataFrame,
    candidates: list[dict[str, object]],
    schedule: pd.DataFrame,
    config: Mapping[str, object],
    destination: Path,
    completion_signature: Mapping[str, object] | None = None,
) -> None:
    source = source_from_candidate(candidates[0])
    allowed = {str(candidate["base_candidate_id"]) for candidate in candidates}
    allowed.update(
        matched
        for candidate in candidates
        if (matched := matched_p1_base_id(source, candidate)) is not None
    )
    metrics_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    strata_rows: list[dict[str, object]] = []
    scoring_parts: list[Path] = []
    for anchor in sorted(schedule["anchor_date"].unique()):
        t = pd.Timestamp(anchor)
        anchor_scoring_rows: list[dict[str, object]] = []
        profiles, parents = build_profiles(frame, source, t, config, allowed_base_ids=allowed)
        profile_base_ids = set(profiles["base_candidate_id"].astype(str))
        mapped_by_base_id: dict[str, pd.DataFrame] = {}
        scheduled = schedule.loc[schedule["anchor_date"].eq(t)]
        for horizon in sorted(scheduled["horizon_days"].unique()):
            period = str(scheduled.loc[scheduled["horizon_days"].eq(horizon), "period"].iloc[0])
            future = future_cohort(frame, t, int(horizon))
            mapped_by_base_id.clear()

            def mapped_for(base_id: str) -> pd.DataFrame:
                if base_id not in profile_base_ids:
                    raise RuntimeError(
                        f"required selected-evaluation profile is absent at {t.date()}: {base_id}"
                    )
                if base_id not in mapped_by_base_id:
                    mapped_by_base_id[base_id] = map_future_orders(
                        future, profiles, parents, source, base_id
                    )
                return mapped_by_base_id[base_id]

            for candidate in candidates:
                base_id = str(candidate["base_candidate_id"])
                mapped = mapped_for(base_id).copy()
                estimator = str(candidate.get("estimator", ""))
                matched_base_id = matched_p1_base_id(source, candidate)
                matched_mapping = (
                    mapped_for(matched_base_id) if matched_base_id is not None else None
                )
                mapped = attach_shrinkage_score(mapped, matched_mapping, estimator)
                mapped = assign_levels(mapped, candidate)
                metrics, entities, strata = evaluate_mapped_orders(mapped, int(candidate["min_support"]), config)
                entities = attach_frozen_entity_levels(entities, mapped)
                profile_row = profiles.loc[profiles["base_candidate_id"].eq(base_id)].iloc[0]
                metrics_rows.append(_add_metric_context(metrics, source, profile_row, t, int(horizon), period))
                if not entities.empty:
                    entities["period"] = period
                    entities["anchor_date"] = t
                    entities["horizon_days"] = int(horizon)
                    entities["profile_spec_id"] = candidate["profile_spec_id"]
                    entity_rows.extend(entities.to_dict("records"))
                if not strata.empty:
                    strata["period"] = period
                    strata["anchor_date"] = t
                    strata["horizon_days"] = int(horizon)
                    strata["profile_spec_id"] = candidate["profile_spec_id"]
                    strata_rows.extend(strata.to_dict("records"))
                keep = [
                    "order_id", "purchase_timestamp", "entity_id", "mapping_status", "history_support",
                    "cold_start", "profile_score", "shrinkage_score", "raw_score", "parent_score", "global_score", "level",
                    "unknown_reason", "target_observed", "target_value", "raw_target_value",
                    "label_available_at", "eligible_for_metric", "posterior_se", "lower_interval", "upper_interval",
                ]
                scored = mapped[keep].copy()
                scored["candidate_id"] = candidate["candidate_id"]
                scored["profile_spec_id"] = candidate["profile_spec_id"]
                scored["target"] = candidate["target"]
                scored["granularity"] = candidate["granularity"]
                scored["period"] = period
                scored["anchor_date"] = t
                scored["horizon_days"] = int(horizon)
                anchor_scoring_rows.extend(scored.to_dict("records"))
        scoring_part = destination / "scoring_parts" / f"{t.strftime('%Y%m%d')}.csv"
        write_csv(
            pd.DataFrame.from_records(anchor_scoring_rows, columns=SCORING_WORK_COLUMNS),
            scoring_part,
            ["candidate_id", "period", "horizon_days", "anchor_date", "order_id"],
        )
        scoring_parts.append(scoring_part)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(metrics_rows), destination / "metrics.csv", ["candidate_id", "period", "horizon_days", "anchor_date"])
    write_csv(pd.DataFrame(entity_rows), destination / "entities.csv", ["candidate_id", "period", "horizon_days", "anchor_date", "entity_id"])
    write_csv(pd.DataFrame(strata_rows), destination / "strata.csv", ["candidate_id", "period", "horizon_days", "anchor_date", "support_stratum"])
    concatenate_csv_files(scoring_parts, destination / "scoring.csv")
    artifact_names = ("metrics.csv", "entities.csv", "strata.csv", "scoring.csv")
    write_json(destination / "complete.json", {
        "source": source,
        "candidate_count": len(candidates),
        "completed_at_utc": utc_now(),
        "signature": _json_safe(dict(completion_signature or {})),
        "artifact_hashes": {
            name: sha256_file(destination / name) for name in artifact_names
        },
    })


def _selected_evaluation_signature(
    source_key: str,
    candidates: list[dict[str, object]],
    freeze_token: str,
) -> dict[str, object]:
    candidate_payload = _json_safe(sorted(candidates, key=lambda row: str(row["candidate_id"])))
    source = source_from_candidate(candidates[0])
    matched_p1_base_ids = sorted(
        matched
        for candidate in candidates
        if (matched := matched_p1_base_id(source, candidate)) is not None
    )
    return {
        "source_id": source_key,
        "freeze_sha256": freeze_token,
        "config_sha256": sha256_file(CONFIG_PATH),
        "anchor_schedule_sha256": sha256_file(ANCHOR_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "profile_core_sha256": sha256_file(OUT / "scripts" / "profile_core.py"),
        "matched_p1_base_ids": matched_p1_base_ids,
        "scoring_work_columns_sha256": hashlib.sha256(
            canonical_json_bytes(SCORING_WORK_COLUMNS)
        ).hexdigest(),
        "selected_entity_columns_sha256": hashlib.sha256(
            canonical_json_bytes(SELECTED_ENTITY_COLUMNS)
        ).hexdigest(),
        "candidate_payload_sha256": hashlib.sha256(
            canonical_json_bytes(candidate_payload)
        ).hexdigest(),
    }


def _valid_selected_evaluation_part(
    destination: Path,
    expected_signature: Mapping[str, object],
) -> bool:
    complete_path = destination / "complete.json"
    if not complete_path.exists():
        return False
    try:
        record = json.loads(complete_path.read_text(encoding="utf-8"))
        if record.get("signature") != dict(expected_signature):
            return False
        hashes = dict(record.get("artifact_hashes") or {})
        required = ("metrics.csv", "entities.csv", "strata.csv", "scoring.csv")
        return all(
            (destination / name).exists()
            and hashes.get(name) == sha256_file(destination / name)
            for name in required
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _prepare_hrd_daily_labels(config: Mapping[str, object]) -> pd.DataFrame:
    """Persist all six V1.1 HRD definitions with deterministic per-definition phases."""
    definitions = [str(value) for value in config["hrd"]["all_definitions"]]
    if len(definitions) != 6 or len(set(definitions)) != 6:
        raise RuntimeError("the frozen HRD audit requires exactly six unique definitions")
    daily_path = V11 / "DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv"
    phase_path = V11 / "HRD_EVENT_PHASES.csv"
    if not daily_path.exists() or not phase_path.exists():
        raise RuntimeError("V1.1 HRD source tables are missing")
    daily = pd.read_csv(daily_path, low_memory=False)
    phases = pd.read_csv(phase_path, low_memory=False)
    required_daily = {"date", *definitions}
    required_phase = {"definition", "cluster_id", "date", "phase", "phase_day"}
    if not required_daily.issubset(daily.columns):
        raise RuntimeError(f"V1.1 HRD daily table missing {sorted(required_daily - set(daily.columns))}")
    if not required_phase.issubset(phases.columns):
        raise RuntimeError(f"V1.1 HRD phase table missing {sorted(required_phase - set(phases.columns))}")
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    if daily["date"].duplicated().any():
        raise RuntimeError("V1.1 HRD daily table has duplicate dates")
    for definition in definitions:
        daily[definition] = _boolean_series(daily[definition])
    phases = phases.loc[phases["definition"].astype(str).isin(definitions)].copy()
    if set(phases["definition"].astype(str).unique()) != set(definitions):
        raise RuntimeError("V1.1 HRD phase table does not cover all frozen definitions")
    phases["date"] = pd.to_datetime(phases["date"], errors="raise").dt.normalize()
    phase_text = phases["phase"].astype(str)
    phases["_phase_priority"] = np.select(
        [phase_text.eq("event"), phase_text.eq("pre_event"), phase_text.str.startswith("post_event")],
        [0, 1, 2],
        default=3,
    )
    phases["_phase_day_priority"] = pd.to_numeric(
        phases["phase_day"], errors="coerce",
    ).abs().fillna(np.inf)
    overlap_rows = int(phases.duplicated(["definition", "date"], keep=False).sum())
    overlap_dates = int(
        phases.loc[phases.duplicated(["definition", "date"], keep=False), ["definition", "date"]]
        .drop_duplicates().shape[0]
    )
    resolved = (
        phases.sort_values(
            ["definition", "date", "_phase_priority", "_phase_day_priority", "cluster_id", "phase"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(["definition", "date"], keep="first")
    )
    for definition in definitions:
        phase_map = (
            resolved.loc[resolved["definition"].astype(str).eq(definition), ["date", "phase"]]
            .set_index("date")["phase"]
        )
        phase = daily["date"].map(phase_map)
        daily[f"{definition}_phase"] = phase.where(
            phase.notna(),
            np.where(daily[definition], "event_unclustered", "BAU"),
        )
    daily["phase_overlap_resolution_rule"] = (
        "event_then_pre_event_then_nearest_post_event_then_cluster_id"
    )
    write_csv(daily, WORK / "HRD_DAILY_LABELS.csv", ["date"])
    audit = {
        "definitions": definitions,
        "primary_definition": str(config["hrd"]["primary_definition"]),
        "daily_source": str(daily_path.relative_to(ROOT)),
        "daily_source_sha256": sha256_file(daily_path),
        "phase_source": str(phase_path.relative_to(ROOT)),
        "phase_source_sha256": sha256_file(phase_path),
        "daily_rows": int(len(daily)),
        "source_phase_rows": int(len(phases)),
        "resolved_definition_date_rows": int(len(resolved)),
        "overlap_source_rows": overlap_rows,
        "overlap_definition_dates": overlap_dates,
        "priority": [
            "event", "pre_event", "nearest_post_event", "lowest_cluster_id", "lexical_phase",
        ],
        "used_as_profile_predictor": False,
    }
    write_json(WORK / "HRD_PHASE_RESOLUTION_AUDIT.json", audit, canonical=True)
    record_event("confirmation", "hrd_daily_labels_persisted", {
        "definitions": definitions,
        "rows": len(daily),
        "overlap_definition_dates": overlap_dates,
        "priority": audit["priority"],
    })
    return daily


def _confirmation_monthly_label_inputs(
    metrics: pd.DataFrame,
    strata: pd.DataFrame,
    candidates: list[dict[str, object]],
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Build one frozen seven-day primary-improvement row per confirmation month."""
    if not candidates:
        return pd.DataFrame(columns=CONFIRMATION_MONTH_COLUMNS)
    candidate_frame = pd.DataFrame(candidates)
    if "target_family" not in candidate_frame:
        candidate_frame["target_family"] = candidate_frame["target"].map(
            lambda value: profile_selection.target_family(str(value))
        )
    candidate_frame = candidate_frame[["candidate_id", "target_family"]].drop_duplicates()
    if candidate_frame["candidate_id"].duplicated().any():
        raise RuntimeError("frozen promoted candidates have inconsistent target families")
    scheduled = schedule.loc[
        schedule["period"].astype(str).eq("confirmation")
        & pd.to_numeric(schedule["horizon_days"], errors="coerce").eq(7)
    ].copy()
    scheduled["calendar_month"] = pd.to_datetime(
        scheduled["anchor_date"], errors="raise",
    ).dt.strftime("%Y-%m")
    scheduled_counts = (
        scheduled.groupby("calendar_month", sort=True)["anchor_date"]
        .nunique()
        .rename("n_scheduled_anchors")
        .reset_index()
    )
    if int(scheduled_counts["n_scheduled_anchors"].sum()) != 25:
        raise RuntimeError("confirmation label audit requires exactly 25 scheduled 7-day anchors")
    grid = candidate_frame.merge(scheduled_counts, how="cross")

    metric_work = exact_columns(metrics, SELECTED_METRIC_COLUMNS)
    if not metric_work.empty:
        metric_work = metric_work.loc[
            metric_work["period"].astype(str).eq("confirmation")
            & pd.to_numeric(metric_work["horizon_days"], errors="coerce").eq(7)
        ].copy()
        metric_work["anchor_date"] = pd.to_datetime(metric_work["anchor_date"], errors="raise")
        metric_work["calendar_month"] = metric_work["anchor_date"].dt.strftime("%Y-%m")
        metric_work["valid"] = _boolean_series(metric_work["valid"])
        family_map = candidate_frame.set_index("candidate_id")["target_family"]
        metric_work["target_family"] = metric_work["candidate_id"].map(family_map)
        metric_work["primary_improvement"] = np.where(
            metric_work["target_family"].eq("binary"),
            pd.to_numeric(metric_work["delta_log_loss"], errors="coerce"),
            pd.to_numeric(metric_work["log_mae_improvement"], errors="coerce"),
        )
    valid_anchor_keys: set[tuple[str, pd.Timestamp]] = set()
    primary_rows: list[dict[str, object]] = []
    if not metric_work.empty:
        for (candidate_id, month), group in metric_work.groupby(
            ["candidate_id", "calendar_month"], sort=True, dropna=False,
        ):
            valid = group.loc[
                group["valid"] & np.isfinite(pd.to_numeric(group["primary_improvement"], errors="coerce"))
            ].copy()
            valid_anchor_keys.update(
                (str(candidate_id), pd.Timestamp(value)) for value in valid["anchor_date"].unique()
            )
            values = pd.to_numeric(valid["primary_improvement"], errors="coerce")
            primary_rows.append({
                "candidate_id": str(candidate_id),
                "calendar_month": str(month),
                "n_valid_anchors": int(valid["anchor_date"].nunique()),
                "primary_improvement": float(values.median()) if len(values) else np.nan,
                "valid": bool(len(values)),
            })
    primary = pd.DataFrame(primary_rows)
    result = grid.merge(
        primary,
        on=["candidate_id", "calendar_month"],
        how="left",
        validate="1:1",
    )
    result["n_valid_anchors"] = pd.to_numeric(
        result.get("n_valid_anchors"), errors="coerce",
    ).fillna(0).astype(int)
    result["valid"] = _boolean_series(result.get("valid", pd.Series(False, index=result.index)))

    high_names = {"support_5_9", "support_10_19", "support_20_plus"}
    high_rows: list[dict[str, object]] = []
    if not strata.empty:
        stratum_work = exact_columns(strata, SELECTED_STRATA_COLUMNS)
        stratum_work = stratum_work.loc[
            stratum_work["period"].astype(str).eq("confirmation")
            & pd.to_numeric(stratum_work["horizon_days"], errors="coerce").eq(7)
            & stratum_work["support_stratum"].astype(str).isin(high_names)
        ].copy()
        stratum_work["anchor_date"] = pd.to_datetime(stratum_work["anchor_date"], errors="raise")
        stratum_work = stratum_work.loc[[
            (str(candidate_id), pd.Timestamp(anchor)) in valid_anchor_keys
            for candidate_id, anchor in zip(stratum_work["candidate_id"], stratum_work["anchor_date"])
        ]].copy()
        stratum_work["calendar_month"] = stratum_work["anchor_date"].dt.strftime("%Y-%m")
        stratum_work["primary_improvement"] = pd.to_numeric(
            stratum_work["primary_improvement"], errors="coerce",
        )
        for (candidate_id, month), group in stratum_work.groupby(
            ["candidate_id", "calendar_month"], sort=True, dropna=False,
        ):
            values = group.loc[
                np.isfinite(group["primary_improvement"]), "primary_improvement"
            ]
            high_rows.append({
                "candidate_id": str(candidate_id),
                "calendar_month": str(month),
                "support_ge5_primary_improvement": float(values.median()) if len(values) else np.nan,
                "high_support_material_reversal": bool((values < -1e-12).any()) if len(values) else False,
                "high_support_positive": bool((values > 1e-12).any()) if len(values) else False,
            })
    high = pd.DataFrame(high_rows)
    if not high.empty:
        result = result.merge(
            high,
            on=["candidate_id", "calendar_month"],
            how="left",
            validate="1:1",
        )
    for column, default in (
        ("support_ge5_primary_improvement", np.nan),
        ("high_support_material_reversal", False),
        ("high_support_positive", False),
    ):
        if column not in result:
            result[column] = default
    result["high_support_material_reversal"] = _boolean_series(
        result["high_support_material_reversal"],
    )
    result["high_support_positive"] = _boolean_series(result["high_support_positive"])
    result["advantage_only_low_support_or_cold"] = (
        pd.to_numeric(result["primary_improvement"], errors="coerce").gt(1e-12)
        & ~result["high_support_positive"]
    )
    return exact_columns(result, CONFIRMATION_MONTH_COLUMNS).sort_values(
        ["candidate_id", "calendar_month"], kind="mergesort",
    ).reset_index(drop=True)


def _persist_confirmation_labels(
    metrics: pd.DataFrame,
    strata: pd.DataFrame,
    candidates: list[dict[str, object]],
    schedule: pd.DataFrame,
    freeze: Mapping[str, object],
) -> pd.DataFrame:
    """Apply and persist the frozen descriptive confirmation-label rubric."""
    months = _confirmation_monthly_label_inputs(metrics, strata, candidates, schedule)
    month_path = WORK / "CONFIRMATION_BY_MONTH_FOR_LABELS.csv"
    write_csv(months, month_path, ["candidate_id", "calendar_month"])
    development_path = WORK / "DEVELOPMENT_SELECTION_AGGREGATE.csv"
    if not development_path.exists():
        raise RuntimeError("frozen development selection summary is missing")
    relative = str(development_path.relative_to(OUT))
    expected = dict(freeze["development_artifact_hashes"]).get(relative)
    if expected is None or sha256_file(development_path) != expected:
        raise RuntimeError("confirmation label audit development summary is not freeze-matched")
    development = pd.read_csv(development_path, low_memory=False)
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    development = development.loc[
        development["candidate_id"].astype(str).isin(candidate_ids)
    ].copy()
    if len(development) != len(candidate_ids):
        raise RuntimeError("confirmation label audit cannot match every frozen candidate to development")
    if candidates:
        labels = profile_selection.confirmation_labels(development, months)
        labels["confirmation_label_reason"] = labels["failed_strong_conditions"].replace(
            "", "all_strong_conditions_passed",
        )
    else:
        labels = pd.DataFrame(columns=CONFIRMATION_LABEL_COLUMNS)
    labels = exact_columns(labels, CONFIRMATION_LABEL_COLUMNS)
    label_path = WORK / "CONFIRMATION_LABELS.csv"
    write_csv(labels, label_path, ["candidate_id"])
    summary = pd.DataFrame(candidates).merge(
        labels,
        on="candidate_id",
        how="left",
        validate="1:1",
    ) if candidates else pd.DataFrame(columns=["candidate_id", *CONFIRMATION_LABEL_COLUMNS[1:]])
    write_csv(
        summary,
        WORK / "SELECTED_CANDIDATE_CONFIRMATION_SUMMARY.csv",
        ["target", "granularity", "selection_rank", "candidate_id"],
    )
    counts = {
        str(key): int(value)
        for key, value in labels["confirmation_label"].value_counts(dropna=False).sort_index().items()
    } if not labels.empty else {}
    audit = {
        "candidate_count": int(len(candidates)),
        "month_rows": int(len(months)),
        "labels": counts,
        "development_summary": relative,
        "development_summary_sha256": sha256_file(development_path),
        "confirmation_month_input_sha256": sha256_file(month_path),
        "confirmation_labels_sha256": sha256_file(label_path),
        "seven_day_confirmation_only": True,
        "label_is_descriptive_not_stage_gate": True,
        "frozen_selected_candidates_unchanged": True,
    }
    write_json(WORK / "CONFIRMATION_LABEL_AUDIT.json", audit, canonical=True)
    record_event("confirmation", "descriptive_confirmation_labels_persisted", audit)
    return labels


def _selected_daily_worker(
    payload: tuple[int, list[dict[str, object]], list[str], str, str],
) -> dict[str, object]:
    """Build one bounded consecutive-date chunk in a forked worker."""
    index, candidates, dates, daily_path_text, parent_path_text = payload
    if _SELECTED_DAILY_SHARED_FRAME is None or _SELECTED_DAILY_SHARED_CONFIG is None:
        raise RuntimeError("selected daily worker started without frozen shared inputs")
    from analysis.dynamic_profile_profile_validation_v1.scripts import selected_daily

    daily, parents = selected_daily._generate_on_dates(
        _SELECTED_DAILY_SHARED_FRAME,
        _SELECTED_DAILY_SHARED_CONFIG,
        candidates,
        dates,
    )
    daily = exact_columns(daily, selected_daily.SELECTED_DAILY_COLUMNS)
    parents = exact_columns(parents, selected_daily.SELECTED_PARENT_COLUMNS)
    daily_path = Path(daily_path_text)
    parent_path = Path(parent_path_text)
    write_csv(daily, daily_path, ["candidate_id", "snapshot_date", "entity_id"])
    write_csv(parents, parent_path, ["base_candidate_id", "snapshot_date", "parent_id"])
    return {
        "chunk": index,
        "candidate_count": len(candidates),
        "date_start": dates[0],
        "date_end": dates[-1],
        "date_count": len(dates),
        "daily_rows": int(len(daily)),
        "parent_rows": int(len(parents)),
        "daily_path": daily_path_text,
        "parent_path": parent_path_text,
        "worker_pid": os.getpid(),
    }


def _persist_selected_daily_profiles(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    candidates: list[dict[str, object]],
    freeze_token: str,
    *,
    workers: int,
    resume: bool,
) -> dict[str, object]:
    """Persist complete selected daily/parent rows with deterministic gzip bytes."""
    from analysis.dynamic_profile_profile_validation_v1.scripts import selected_daily

    daily_path = WORK / "SELECTED_DAILY_PROFILE_ROWS.csv.gz"
    parent_path = WORK / "SELECTED_PARENT_ROWS.csv"
    meta_path = WORK / "SELECTED_DAILY_BUILD_META.json"
    signature = {
        "freeze_sha256": freeze_token,
        "runner_sha256": sha256_file(Path(__file__)),
        "selected_daily_source_sha256": sha256_file(OUT / "scripts" / "selected_daily.py"),
        "profile_core_sha256": sha256_file(OUT / "scripts" / "profile_core.py"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "candidate_ids": sorted(str(candidate["candidate_id"]) for candidate in candidates),
    }
    if resume and daily_path.exists() and parent_path.exists() and meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                cached.get("signature") == signature
                and cached.get("daily_sha256") == sha256_file(daily_path)
                and cached.get("parent_sha256") == sha256_file(parent_path)
            ):
                record_event("confirmation", "selected_daily_profile_cache_validated", {
                    "daily_rows": cached.get("daily_rows", 0),
                    "parent_rows": cached.get("parent_rows", 0),
                })
                return cached
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    plan = selected_daily.build_construction_plan(candidates, config)
    write_csv(
        plan,
        WORK / "SELECTED_DAILY_CONSTRUCTION_PLAN.csv",
        ["candidate_id", "component"],
    )
    if not candidates:
        write_deterministic_gzip_csv(
            pd.DataFrame(columns=selected_daily.SELECTED_DAILY_COLUMNS),
            daily_path,
            selected_daily.SELECTED_DAILY_COLUMNS,
            ["candidate_id", "snapshot_date", "entity_id"],
        )
        write_csv(
            pd.DataFrame(columns=selected_daily.SELECTED_PARENT_COLUMNS),
            parent_path,
            ["base_candidate_id", "snapshot_date", "parent_id"],
        )
        records: list[dict[str, object]] = []
        daily_rows = 0
        parent_rows = 0
    else:
        ordered_candidates = sorted(candidates, key=lambda row: str(row["candidate_id"]))
        frozen_dates = [pd.Timestamp(value) for value in selected_daily.FROZEN_DATES]
        date_chunks = [frozen_dates[index:index + 3] for index in range(0, len(frozen_dates), 3)]
        part_dir = WORK / "selected_daily_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        payloads = [
            (
                index,
                ordered_candidates,
                [value.strftime("%Y-%m-%d") for value in date_chunk],
                str(part_dir / f"daily_{index:02d}.csv"),
                str(part_dir / f"parents_{index:02d}.csv"),
            )
            for index, date_chunk in enumerate(date_chunks)
        ]
        global _SELECTED_DAILY_SHARED_FRAME, _SELECTED_DAILY_SHARED_CONFIG
        _SELECTED_DAILY_SHARED_FRAME = frame
        _SELECTED_DAILY_SHARED_CONFIG = config
        try:
            if len(payloads) == 1:
                records = [_selected_daily_worker(payloads[0])]
            else:
                context = mp.get_context("fork")
                with context.Pool(processes=min(workers, len(payloads))) as pool:
                    records = list(pool.imap_unordered(_selected_daily_worker, payloads, chunksize=1))
        finally:
            _SELECTED_DAILY_SHARED_FRAME = None
            _SELECTED_DAILY_SHARED_CONFIG = None
        records = sorted(records, key=lambda row: int(row["chunk"]))
        concatenate_csv_files_to_deterministic_gzip(
            [Path(str(row["daily_path"])) for row in records],
            daily_path,
            selected_daily.SELECTED_DAILY_COLUMNS,
        )
        concatenate_csv_files(
            [Path(str(row["parent_path"])) for row in records],
            parent_path,
        )
        daily_rows = int(sum(int(row["daily_rows"]) for row in records))
        parent_rows = int(sum(int(row["parent_rows"]) for row in records))
    metadata = {
        "signature": signature,
        "daily_rows": daily_rows,
        "parent_rows": parent_rows,
        "daily_sha256": sha256_file(daily_path),
        "parent_sha256": sha256_file(parent_path),
        "gzip_mtime": 0,
        "workers": int(min(workers, max(1, len(records)))),
        "date_chunk_days": 3,
        "working_row_order": "date_chunk_then_candidate_then_snapshot_then_entity",
        "chunks": records,
    }
    write_json(meta_path, metadata, canonical=True)
    record_event("confirmation", "selected_daily_profiles_persisted", {
        "candidate_count": len(candidates),
        "daily_rows": daily_rows,
        "parent_rows": parent_rows,
        "daily_sha256": metadata["daily_sha256"],
        "parent_sha256": metadata["parent_sha256"],
        "gzip_mtime": 0,
    })
    return metadata


def run_development(data_dir: Path, resume: bool = True, workers: int = 1) -> None:
    before = require_preflight()
    record_event("development", "started", {"locked_purchase_end_exclusive": "2018-01-01"})
    config = load_config()
    frame, canonical, raw = build_analysis_frame(data_dir)
    if len(frame) != 99441 or len(canonical) != 96470 or canonical["order_id"].nunique() != 96470:
        raise RuntimeError("authoritative sample counts failed")
    thresholds = frozen_tail_thresholds(frame)
    threshold_record = {
        **thresholds, "sample": "canonical_delivered", "quantile": 0.90,
        "method": "linear", "availability_end_exclusive": "2017-04-01",
        "event_operator": ">", "frozen_before_development_evaluation": True,
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    write_json(TAIL_PATH, threshold_record, canonical=True)
    record_event("development", "tail_thresholds_written_and_hashed", {"sha256": sha256_file(TAIL_PATH), **thresholds})
    frame = attach_tail_targets(frame, thresholds)
    development = mask_locked_outcomes_for_development(frame)
    development, nuisance_audit = generate_row_origin_expectations(development, config, "2018-01-01")
    if development["order_purchase_timestamp"].ge(pd.Timestamp("2018-01-01")).any():
        raise RuntimeError("confirmation purchase row entered development process")
    write_csv(nuisance_audit, WORK / "DEVELOPMENT_NUISANCE_AUDIT.csv", ["target", "origin"])
    schedule = pd.read_csv(ANCHOR_PATH, parse_dates=["anchor_date", "future_start", "future_end_exclusive"])
    dev_schedule = schedule.loc[schedule["period"].eq("development")].copy()
    parts = WORK / "development_parts"
    sources = candidate_sources()
    payloads: list[tuple[int, int, dict[str, object], str]] = []
    for number, source in enumerate(sources, 1):
        destination = parts / source_id(source)
        if resume and (destination / "complete.json").exists():
            continue
        payloads.append((number, len(sources), source, str(destination)))
    records: list[dict[str, object]] = []
    if not payloads:
        pass
    elif workers == 1:
        global _DEVELOPMENT_SHARED_FRAME, _DEVELOPMENT_SHARED_ANCHORS, _DEVELOPMENT_SHARED_CONFIG
        _DEVELOPMENT_SHARED_FRAME = development
        _DEVELOPMENT_SHARED_ANCHORS = dev_schedule
        _DEVELOPMENT_SHARED_CONFIG = config
        try:
            records = [_development_source_worker(payload) for payload in payloads]
        finally:
            _DEVELOPMENT_SHARED_FRAME = None
            _DEVELOPMENT_SHARED_ANCHORS = None
            _DEVELOPMENT_SHARED_CONFIG = None
    else:
        _DEVELOPMENT_SHARED_FRAME = development
        _DEVELOPMENT_SHARED_ANCHORS = dev_schedule
        _DEVELOPMENT_SHARED_CONFIG = config
        context = mp.get_context("fork")
        try:
            with context.Pool(processes=min(workers, len(payloads))) as pool:
                records = list(pool.imap_unordered(_development_source_worker, payloads, chunksize=1))
        finally:
            _DEVELOPMENT_SHARED_FRAME = None
            _DEVELOPMENT_SHARED_ANCHORS = None
            _DEVELOPMENT_SHARED_CONFIG = None
    if payloads:
        write_csv(
            pd.DataFrame(records), WORK / "DEVELOPMENT_WORKER_TIMINGS.csv",
            ["source_number"],
        )
    metrics = combine_source_parts(parts, "metrics.csv")
    strata = combine_source_parts(parts, "strata.csv")
    audit = combine_source_parts(parts, "audit.csv")
    write_csv(metrics, WORK / "DEVELOPMENT_ANCHOR_METRICS.csv", ["candidate_id", "horizon_days", "anchor_date"])
    write_csv(strata, WORK / "DEVELOPMENT_ANCHOR_STRATA.csv", ["candidate_id", "horizon_days", "anchor_date", "support_stratum"])
    write_csv(audit, WORK / "DEVELOPMENT_CONSTRUCTION_AUDIT.csv", ["target", "granularity", "scheme", "window_days", "lag_days", "snapshot_date"])
    write_csv(candidate_catalog(), WORK / "CANDIDATE_CATALOG.csv", ["target", "granularity", "scheme", "window_days", "lag_days", "estimator", "parent_structure", "kappa"])
    negative_audit = pd.DataFrame([
        {
            "record_type": "negative_process_exclusion", "target": "handling",
            "all_placed_excluded": int(frame["handling_duration"].lt(0).sum()),
            "canonical_excluded": int((frame["in_canonical"] & frame["handling_duration"].lt(0)).sum()),
            "affected_sellers_all_placed": int(frame.loc[frame["handling_duration"].lt(0), "seller_id"].nunique()),
            "affected_sellers_canonical": int(frame.loc[frame["in_canonical"] & frame["handling_duration"].lt(0), "seller_id"].nunique()),
            "clipped_to_zero": False,
        },
        {
            "record_type": "negative_process_exclusion", "target": "transit",
            "all_placed_excluded": int(frame["transit_duration"].lt(0).sum()),
            "canonical_excluded": int((frame["in_canonical"] & frame["transit_duration"].lt(0)).sum()),
            "affected_routes_all_placed": int(frame.loc[frame["transit_duration"].lt(0), "state_od"].nunique()),
            "affected_routes_canonical": int(frame.loc[frame["in_canonical"] & frame["transit_duration"].lt(0), "state_od"].nunique()),
            "clipped_to_zero": False,
        },
    ])
    write_csv(negative_audit, WORK / "NEGATIVE_DURATION_AUDIT.csv", ["target"])
    record_event("development", "candidate_anchor_evaluation_complete", {
        "metric_rows": len(metrics), "strata_rows": len(strata), "source_count": len(sources),
        "locked_rows_present": False,
    })
    print(json.dumps({"stage": "development", "status": "anchor_evaluation_complete", "metric_rows": len(metrics)}, sort_keys=True), flush=True)


def _development_level_score_rows(
    development: pd.DataFrame,
    selected: pd.DataFrame,
    schedule: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    columns = [
        "candidate_id", "base_candidate_id", "support_threshold", "anchor_date",
        "entity_id", "score", "support", "cold_start", "missing_mapping",
        "lower_interval", "upper_interval", "future_7d_all_placed_exposure",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    dev_schedule = schedule.loc[
        schedule["period"].astype(str).eq("development")
        & pd.to_numeric(schedule["horizon_days"], errors="coerce").eq(7)
    ].copy()
    if len(dev_schedule) != 39:
        raise RuntimeError("level-threshold derivation requires exactly 39 development 7-day anchors")
    future_intervals = dev_schedule[["future_start", "future_end_exclusive"]].sort_values("future_start")
    starts = pd.to_datetime(future_intervals["future_start"], errors="raise").to_numpy()
    ends = pd.to_datetime(future_intervals["future_end_exclusive"], errors="raise").to_numpy()
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
        raise RuntimeError("development 7-day exposure cohorts overlap")

    rows: list[pd.DataFrame] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for candidate in selected.to_dict("records"):
        grouped.setdefault(source_id(source_from_candidate(candidate)), []).append(candidate)
    for sid, candidates in sorted(grouped.items()):
        source = source_from_candidate(candidates[0])
        allowed = {str(candidate["base_candidate_id"]) for candidate in candidates}
        by_base: dict[str, list[dict[str, object]]] = {}
        for candidate in candidates:
            by_base.setdefault(str(candidate["base_candidate_id"]), []).append(candidate)
        for anchor in sorted(dev_schedule["anchor_date"].unique()):
            snapshot = pd.Timestamp(anchor)
            profiles, parents = build_profiles(
                development, source, snapshot, config, allowed_base_ids=allowed,
            )
            future = future_cohort(development, snapshot, 7)
            for base_id, members in sorted(by_base.items()):
                profile = profiles.loc[profiles["base_candidate_id"].astype(str).eq(base_id)].copy()
                if profile.empty:
                    continue
                mapped = map_future_orders(future, profiles, parents, source, base_id)
                exposure = (
                    mapped.loc[mapped["mapping_status"].eq("seen")]
                    .groupby("entity_id", sort=True)["order_id"]
                    .size()
                    .rename("future_7d_all_placed_exposure")
                )
                score_rows = profile[[
                    "entity_id", "score", "support", "lower_interval", "upper_interval",
                ]].merge(exposure, left_on="entity_id", right_index=True, how="left", validate="1:1")
                score_rows["future_7d_all_placed_exposure"] = score_rows[
                    "future_7d_all_placed_exposure"
                ].fillna(0).astype(int)
                score_rows = score_rows.loc[score_rows["future_7d_all_placed_exposure"].gt(0)].copy()
                score_rows["base_candidate_id"] = base_id
                score_rows["anchor_date"] = snapshot
                score_rows["cold_start"] = False
                score_rows["missing_mapping"] = False
                for member in members:
                    candidate_rows = score_rows.copy()
                    candidate_rows["candidate_id"] = str(member["candidate_id"])
                    candidate_rows["support_threshold"] = int(member["support_threshold"])
                    rows.append(candidate_rows[columns])
        print(json.dumps({
            "stage": "selection", "event": "level_threshold_source_complete",
            "source": sid, "candidate_count": len(candidates),
        }, sort_keys=True), flush=True)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["candidate_id", "anchor_date", "entity_id"], kind="mergesort")
        .reset_index(drop=True)
        if rows else pd.DataFrame(columns=columns)
    )


def _derive_selected_level_thresholds(
    development: pd.DataFrame,
    selected: pd.DataFrame,
    schedule: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = _development_level_score_rows(
        development, selected, schedule, config,
    )
    write_csv(
        score_rows,
        WORK / "DEVELOPMENT_LEVEL_SCORE_INPUTS.csv",
        ["candidate_id", "anchor_date", "entity_id"],
    )
    threshold_columns = [
        "candidate_id", "support_threshold", "q33", "q67", "threshold_rows",
        "threshold_total_weight", "thresholds_valid", "threshold_invalid_reason",
    ]
    if score_rows.empty:
        thresholds = pd.DataFrame(columns=threshold_columns)
    else:
        thresholds = profile_selection.derive_weighted_level_thresholds(score_rows)
    keys = selected[["candidate_id", "support_threshold"]].drop_duplicates()
    thresholds = keys.merge(
        thresholds,
        on=["candidate_id", "support_threshold"],
        how="left",
        validate="1:1",
    )
    if not thresholds.empty:
        missing = thresholds["thresholds_valid"].isna()
        thresholds.loc[missing, "thresholds_valid"] = False
        thresholds.loc[missing, "threshold_invalid_reason"] = "no_development_profile_exposure"
        thresholds["thresholds_valid"] = _boolean_series(thresholds["thresholds_valid"])
    write_csv(
        thresholds,
        WORK / "DEVELOPMENT_LEVEL_THRESHOLDS.csv",
        ["candidate_id", "support_threshold"],
    )
    return score_rows, thresholds


def _write_selection_freeze_ordered(payload: Mapping[str, object]) -> str:
    if FREEZE_PATH.exists() or FREEZE_SHA_PATH.exists():
        raise RuntimeError("immutable selection freeze already exists; refusing to overwrite")
    safe_payload = _json_safe(payload)
    encoded = canonical_json_bytes(safe_payload)
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FREEZE_PATH.with_name(FREEZE_PATH.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, FREEZE_PATH)
    record_event("selection", "selection_freeze_written", {
        "path": FREEZE_PATH.name,
        "promoted_candidate_count": len(payload.get("promoted_candidates", [])),
    })
    digest = sha256_file(FREEZE_PATH)
    side_tmp = FREEZE_SHA_PATH.with_name(FREEZE_SHA_PATH.name + f".tmp.{os.getpid()}")
    side_tmp.write_text(f"{digest}  {FREEZE_PATH.name}\n", encoding="utf-8")
    os.replace(side_tmp, FREEZE_SHA_PATH)
    record_event("selection", "selection_freeze_hashed", {
        "sha256": digest, "sidecar": FREEZE_SHA_PATH.name,
    })
    record_event("selection", "selection_freeze_recorded", {
        "sha256": digest, "immutable": True,
    })
    return digest


def run_selection(data_dir: Path, resume: bool = True, workers: int = 1) -> None:
    before = require_preflight()
    if FREEZE_PATH.exists() or FREEZE_SHA_PATH.exists():
        raise RuntimeError("selection freeze already exists; selection is immutable once written")
    record_event("selection", "started", {
        "development_only": True,
        "locked_purchase_end_exclusive": "2018-01-01",
        "development_pid": os.getpid(),
    })
    config = load_config()
    schedule = pd.read_csv(
        ANCHOR_PATH,
        parse_dates=["anchor_date", "future_start", "future_end_exclusive"],
    )
    catalog = profile_selection.build_candidate_catalog(config)
    if len(catalog) != 3024 or catalog["base_candidate_id"].nunique() != 1008:
        raise RuntimeError("frozen selection catalog cardinality mismatch")
    write_csv(
        catalog,
        WORK / "SELECTION_CANDIDATE_CATALOG.csv",
        ["candidate_id"],
    )

    development = _masked_development_frame(data_dir, config)
    record_event("selection", "locked_rows_removed_before_selection_evaluation", {
        "remaining_rows": len(development),
        "locked_rows_present": False,
    })
    stability = _load_or_compute_development_stability(
        development, config, catalog, resume=resume, workers=workers,
    )
    stability_summary = _stability_summary(stability)
    write_csv(
        stability_summary,
        WORK / "DEVELOPMENT_DAILY_STABILITY_SUMMARY.csv",
        ["base_candidate_id"],
    )

    metrics_path = WORK / "DEVELOPMENT_ANCHOR_METRICS.csv"
    strata_path = WORK / "DEVELOPMENT_ANCHOR_STRATA.csv"
    if not metrics_path.exists() or not strata_path.exists():
        raise RuntimeError("selection requires completed development anchor metrics and strata")
    raw_metrics = pd.read_csv(metrics_path, low_memory=False)
    raw_strata = pd.read_csv(strata_path, low_memory=False)
    anchors = _complete_development_anchor_grid(
        raw_metrics, catalog, schedule, stability_summary,
    )
    strata = raw_strata.copy()
    if not strata.empty:
        strata["anchor_date"] = pd.to_datetime(strata["anchor_date"], errors="raise")
        strata = strata.loc[
            strata["period"].astype(str).eq("development")
            & pd.to_numeric(strata["horizon_days"], errors="coerce").eq(7)
        ].copy()
    strata = _complete_selection_support_strata(strata, anchors)
    write_csv(
        strata,
        WORK / "DEVELOPMENT_SELECTION_STRATA_GRID.csv",
        ["candidate_id", "anchor_date", "support_stratum"],
    )
    write_csv(
        anchors,
        WORK / "DEVELOPMENT_SELECTION_ANCHOR_GRID.csv",
        ["candidate_id", "anchor_date"],
    )
    aggregate, by_month = profile_selection.aggregate_anchor_metrics(
        anchors,
        scheduled_anchors=39,
        source_delta_convention="reference_minus_candidate",
    )
    gates = profile_selection.minimum_evidence_gates(
        anchors,
        support_strata=strata,
        scheduled_anchors=39,
        minimum_valid_fraction=float(config["validity"]["minimum_valid_anchor_fraction"]),
        maximum_month_share=float(config["validity"]["maximum_single_month_positive_improvement_share"]),
        source_delta_convention="reference_minus_candidate",
        tolerance=float(config["pareto"]["numeric_equality_tolerance"]),
    )
    gate_columns = ["candidate_id"] + [
        column for column in gates.columns
        if column != "candidate_id" and column not in aggregate.columns
    ]
    candidate_summary = aggregate.merge(
        gates[gate_columns], on="candidate_id", how="left", validate="1:1",
    )
    write_csv(
        candidate_summary,
        WORK / "DEVELOPMENT_SELECTION_AGGREGATE.csv",
        ["candidate_id"],
    )
    write_csv(
        by_month,
        WORK / "DEVELOPMENT_SELECTION_BY_MONTH.csv",
        ["candidate_id", "anchor_month"],
    )

    development_long = _selection_results_long(
        candidate_summary, anchors, by_month=False,
    )
    development_month_long = _selection_results_long(
        by_month, anchors, by_month=True,
    )
    write_csv(
        development_long,
        OUT / "PROFILE_DEVELOPMENT_RESULTS.csv",
        ["candidate_id", "metric_name"],
    )
    write_csv(
        development_month_long,
        OUT / "PROFILE_DEVELOPMENT_BY_MONTH.csv",
        ["candidate_id", "calendar_month", "metric_name"],
    )

    pareto = profile_selection.pareto_frontier(
        candidate_summary,
        tolerance=float(config["pareto"]["numeric_equality_tolerance"]),
    )
    decisions = profile_selection.select_confirmation_candidates(
        pareto,
        max_candidates=int(config["pareto"]["max_confirmation_candidates_per_target_granularity"]),
    )
    write_csv(
        decisions,
        WORK / "DEVELOPMENT_PARETO_DECISIONS_WIDE.csv",
        ["target", "granularity", "selected_for_confirmation", "selection_rank", "candidate_id"],
    )
    write_csv(
        exact_columns(decisions, PARETO_PUBLIC_COLUMNS),
        OUT / "PROFILE_PARETO_FRONTIER.csv",
        ["target", "granularity", "selected_for_confirmation", "selection_rank", "candidate_id"],
    )
    selected = decisions.loc[_boolean_series(decisions["selected_for_confirmation"])].copy()
    _, thresholds = _derive_selected_level_thresholds(
        development, selected, schedule, config,
    )
    selected = selected.merge(
        thresholds,
        on=["candidate_id", "support_threshold"],
        how="left",
        validate="1:1",
    )
    selected["min_support"] = selected["support_threshold"].astype(int)
    selected["low_medium_cutoff"] = selected.get("q33", pd.Series(np.nan, index=selected.index))
    selected["medium_high_cutoff"] = selected.get("q67", pd.Series(np.nan, index=selected.index))
    selected["level_thresholds_valid"] = _boolean_series(
        selected.get("thresholds_valid", pd.Series(False, index=selected.index)),
    )
    _validate_promoted_candidate_specs(
        selected.to_dict("records"),
        context="selection freeze denied",
    )
    write_csv(
        selected,
        WORK / "SELECTED_CANDIDATES_WIDE.csv",
        ["target", "granularity", "selection_rank", "candidate_id"],
    )
    selected_public = selected.copy()
    selected_public["confirmation_label"] = ""
    selected_public["confirmation_label_reason"] = ""
    write_csv(
        exact_columns(selected_public, SELECTED_PUBLIC_COLUMNS),
        OUT / "PROFILE_SELECTED_CANDIDATES.csv",
        ["target", "granularity", "selection_rank", "candidate_id"],
    )

    protected_ok, protected_detail = compare_hash_maps(before["protected_hashes"])
    if not protected_ok:
        raise RuntimeError(f"protected paths changed before selection freeze: {protected_detail}")
    development_artifacts = [
        OUT / "PROFILE_DEVELOPMENT_RESULTS.csv",
        OUT / "PROFILE_DEVELOPMENT_BY_MONTH.csv",
        OUT / "PROFILE_PARETO_FRONTIER.csv",
        OUT / "PROFILE_SELECTED_CANDIDATES.csv",
        WORK / "DEVELOPMENT_SELECTION_ANCHOR_GRID.csv",
        WORK / "DEVELOPMENT_SELECTION_AGGREGATE.csv",
        WORK / "DEVELOPMENT_SELECTION_BY_MONTH.csv",
        WORK / "DEVELOPMENT_SELECTION_STRATA_GRID.csv",
        WORK / "DEVELOPMENT_LEVEL_SCORE_INPUTS.csv",
        WORK / "DEVELOPMENT_LEVEL_THRESHOLDS.csv",
        WORK / "SELECTION_CANDIDATE_CATALOG.csv",
        WORK / "DEVELOPMENT_PARETO_DECISIONS_WIDE.csv",
        WORK / "SELECTED_CANDIDATES_WIDE.csv",
        DEVELOPMENT_STABILITY_PATH,
        DEVELOPMENT_STABILITY_META_PATH,
        metrics_path,
        strata_path,
        TAIL_PATH,
    ]
    artifact_hashes = {
        str(path.relative_to(OUT)): sha256_file(path) for path in development_artifacts
    }
    decision_records = decisions[[
        "candidate_id", "pareto_eligible", "pareto_nondominated", "dominated_by",
        "pareto_ineligible_reason", "selected_for_confirmation", "selection_rank",
        "selection_decision",
    ]].to_dict("records")
    freeze_payload = {
        "schema_version": "dynamic_profile_selection_freeze_v1",
        "analysis_id": "dynamic_profile_profile_validation_v1",
        "created_at_utc": utc_now(),
        "development_pid": os.getpid(),
        "development_purchase_end_exclusive": "2018-01-01",
        "confirmation_outcomes_accessed": False,
        "frozen_config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(OUT / "PROFILE_PROTOCOL.md"),
        "development_anchor_schedule_sha256": sha256_file(ANCHOR_PATH),
        "development_artifact_hashes": artifact_hashes,
        "source_hashes": {
            "scripts": recursive_hashes(OUT / "scripts"),
            "assembler": {str(ASSEMBLER): sha256_file(ASSEMBLER)},
            "pre_execution_state_sha256": sha256_file(PRESTATE_PATH),
            "charter": before.get("charter", {}),
            "raw_inputs": before.get("raw_file_hashes", {}),
        },
        "development_evidence": {
            "scheduled_7d_anchors": 39,
            "candidate_support_rules": int(len(catalog)),
            "base_candidates": int(catalog["base_candidate_id"].nunique()),
            "minimum_evidence_pass_count": int(gates["minimum_evidence_pass"].sum()),
            "pareto_nondominated_count": int(decisions["pareto_nondominated"].sum()),
            "promoted_candidate_count": int(len(selected)),
            "negative_selection_valid": bool(selected.empty),
        },
        "selection_rule": {
            "method": "minimum_evidence_then_pareto_without_composite",
            "binary_delta_convention": "candidate_minus_best_parent_or_global_reference",
            "numeric_equality_tolerance": config["pareto"]["numeric_equality_tolerance"],
            "minimum_valid_anchor_fraction": config["validity"]["minimum_valid_anchor_fraction"],
            "maximum_single_month_positive_improvement_share": config["validity"]["maximum_single_month_positive_improvement_share"],
            "simplicity_order": config["pareto"]["simplicity_order"],
            "max_candidates_per_target_granularity": config["pareto"]["max_confirmation_candidates_per_target_granularity"],
            "level_quantiles": config["levels"]["weighted_quantiles"],
            "level_weight": config["levels"]["weight"],
        },
        "tie_break_choices": decision_records,
        "promoted_candidates": selected.to_dict("records"),
        "confirmation_access_guard": {
            "fresh_process_required": True,
            "same_pid_forbidden": True,
            "freeze_token_required": True,
            "confirmation_start_inclusive": "2018-01-01",
            "mutable_candidate_options_allowed": False,
        },
        "protected_hash_audit": {"passed": True, "detail": protected_detail},
    }
    freeze_digest = _write_selection_freeze_ordered(freeze_payload)
    print(json.dumps({
        "stage": "selection", "status": "selection_frozen",
        "promoted_candidates": len(selected), "freeze_token": freeze_digest,
    }, sort_keys=True), flush=True)


def run_confirmation(
    data_dir: Path,
    token: str,
    resume: bool = True,
    workers: int = 1,
) -> None:
    freeze = verify_selection_freeze(token)
    require_preflight()
    record_event("confirmation", "label_frame_open_started_after_freeze")
    config = load_config()
    frame, canonical, raw = build_analysis_frame(data_dir)
    frozen_tail = json.loads(TAIL_PATH.read_text(encoding="utf-8"))
    if frozen_tail["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("tail threshold/config mismatch")
    frame = attach_tail_targets(frame, frozen_tail)
    frame, nuisance_audit = generate_row_origin_expectations(frame, config, "2018-09-01")
    write_csv(nuisance_audit, WORK / "FULL_NUISANCE_AUDIT_AFTER_FREEZE.csv", ["target", "origin"])
    record_event("confirmation", "confirmation_labels_opened", {"rows": len(frame), "canonical_rows": len(canonical)})
    candidates = [dict(candidate) for candidate in freeze["promoted_candidates"]]
    schedule = pd.read_csv(ANCHOR_PATH, parse_dates=["anchor_date", "future_start", "future_end_exclusive"])
    _prepare_hrd_daily_labels(config)
    if not candidates:
        metrics = pd.DataFrame(columns=SELECTED_METRIC_COLUMNS)
        entities = pd.DataFrame(columns=SELECTED_ENTITY_COLUMNS)
        strata = pd.DataFrame(columns=SELECTED_STRATA_COLUMNS)
        write_csv(metrics, WORK / "SELECTED_ANCHOR_METRICS.csv", ["candidate_id", "period", "horizon_days", "anchor_date"])
        write_csv(entities, WORK / "SELECTED_ENTITY_ROWS.csv", ["candidate_id", "period", "horizon_days", "anchor_date", "entity_id"])
        write_csv(strata, WORK / "SELECTED_ANCHOR_STRATA.csv", ["candidate_id", "period", "horizon_days", "anchor_date", "support_stratum"])
        write_csv(
            pd.DataFrame(columns=ORDER_SCORING_PUBLIC_COLUMNS),
            OUT / "PROFILE_FUTURE_ORDER_SCORING.csv",
            ["candidate_id", "period", "horizon_days", "anchor_date", "order_id"],
        )
        write_deterministic_gzip_csv(
            pd.DataFrame(columns=SCORING_WORK_COLUMNS),
            RICH_SCORING_PATH,
            SCORING_WORK_COLUMNS,
            ["candidate_id", "period", "horizon_days", "anchor_date", "order_id"],
        )
        daily_metadata = _persist_selected_daily_profiles(
            frame, config, candidates, token, workers=workers, resume=resume,
        )
        _persist_confirmation_labels(metrics, strata, candidates, schedule, freeze)
        protected_ok, protected_detail = compare_hash_maps(
            json.loads(PRESTATE_PATH.read_text(encoding="utf-8"))["protected_hashes"]
        )
        if not protected_ok:
            raise RuntimeError(f"protected paths changed during negative confirmation: {protected_detail}")
        record_event("confirmation", "negative_selection_evaluation_complete", {
            "candidate_count": 0,
            "metric_rows": 0,
            "daily_rows": daily_metadata["daily_rows"],
            "order_scoring_sha256": sha256_file(OUT / "PROFILE_FUTURE_ORDER_SCORING.csv"),
            "rich_order_scoring_sha256": sha256_file(RICH_SCORING_PATH),
            "zero_candidates_is_valid_negative_result": True,
        })
        print(json.dumps({
            "stage": "confirmation",
            "status": "negative_selection_evaluation_complete",
            "candidate_count": 0,
            "metric_rows": 0,
        }, sort_keys=True), flush=True)
        return

    parts = WORK / "selected_evaluation_parts"
    grouped: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        sid = source_id(source_from_candidate(candidate))
        grouped.setdefault(sid, []).append(candidate)
    for number, (sid, members) in enumerate(sorted(grouped.items()), 1):
        destination = parts / sid
        completion_signature = _selected_evaluation_signature(sid, members, token)
        if resume and _valid_selected_evaluation_part(destination, completion_signature):
            record_event("confirmation", "selected_evaluation_part_cache_validated", {
                "source_id": sid,
                "candidate_count": len(members),
            })
            continue
        print(json.dumps({"stage": "confirmation", "selected_source_number": number, "selected_source_total": len(grouped), "source": source_from_candidate(members[0]), "candidate_count": len(members)}, sort_keys=True), flush=True)
        evaluate_selected_source(
            frame,
            members,
            schedule,
            config,
            destination,
            completion_signature=completion_signature,
        )
    selected_destinations = [parts / sid for sid in sorted(grouped)]
    def combine_selected(filename: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for destination in selected_destinations:
            try:
                frames.append(pd.read_csv(destination / filename, low_memory=False))
            except pd.errors.EmptyDataError:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    metrics = combine_selected("metrics.csv")
    entities = combine_selected("entities.csv")
    strata = combine_selected("strata.csv")
    metrics = exact_columns(metrics, SELECTED_METRIC_COLUMNS)
    entities = exact_columns(entities, SELECTED_ENTITY_COLUMNS)
    strata = exact_columns(strata, SELECTED_STRATA_COLUMNS)
    write_csv(metrics, WORK / "SELECTED_ANCHOR_METRICS.csv", ["candidate_id", "period", "horizon_days", "anchor_date"])
    write_csv(entities, WORK / "SELECTED_ENTITY_ROWS.csv", ["candidate_id", "period", "horizon_days", "anchor_date", "entity_id"])
    write_csv(strata, WORK / "SELECTED_ANCHOR_STRATA.csv", ["candidate_id", "period", "horizon_days", "anchor_date", "support_stratum"])
    concatenate_csv_files(
        [destination / "scoring.csv" for destination in selected_destinations],
        OUT / "PROFILE_FUTURE_ORDER_SCORING.csv",
    )
    concatenate_csv_files_to_deterministic_gzip(
        [destination / "scoring.csv" for destination in selected_destinations],
        RICH_SCORING_PATH,
        SCORING_WORK_COLUMNS,
    )
    daily_metadata = _persist_selected_daily_profiles(
        frame, config, candidates, token, workers=workers, resume=resume,
    )
    labels = _persist_confirmation_labels(metrics, strata, candidates, schedule, freeze)
    protected_ok, protected_detail = compare_hash_maps(
        json.loads(PRESTATE_PATH.read_text(encoding="utf-8"))["protected_hashes"]
    )
    if not protected_ok:
        raise RuntimeError(f"protected paths changed during confirmation: {protected_detail}")
    record_event("confirmation", "selected_future_evaluation_complete", {
        "candidate_count": len(candidates), "metric_rows": len(metrics),
        "daily_rows": daily_metadata["daily_rows"],
        "confirmation_label_count": len(labels),
        "order_scoring_sha256": sha256_file(OUT / "PROFILE_FUTURE_ORDER_SCORING.csv"),
        "rich_order_scoring_sha256": sha256_file(RICH_SCORING_PATH),
    })
    print(json.dumps({"stage": "confirmation", "status": "selected_future_evaluation_complete", "candidate_count": len(candidates), "metric_rows": len(metrics)}, sort_keys=True), flush=True)


def run_finalize(test_results_path: Path | None = None) -> None:
    require_preflight()
    if not FREEZE_PATH.exists() or not FREEZE_SHA_PATH.exists():
        raise RuntimeError("finalize requires an immutable selection freeze")
    state = load_state()
    events = [str(event.get("event", "")) for event in state.get("stage_events", [])]
    if not any(
        event in events
        for event in ("selected_future_evaluation_complete", "negative_selection_evaluation_complete")
    ):
        raise RuntimeError("finalize requires completed post-freeze evaluation")
    detail: dict[str, object] = {
        "test_results_path": str(test_results_path) if test_results_path is not None else None,
    }
    if test_results_path is not None and test_results_path.exists():
        for line in test_results_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("COMMAND:"):
                detail["test_command"] = line.removeprefix("COMMAND:").strip()
                break
    record_event("finalize", "reporting_started", detail)
    from analysis.dynamic_profile_profile_validation_v1.scripts.profile_reporting import finalize_reporting

    result = finalize_reporting(
        output_dir=OUT,
        work_dir=WORK,
        test_results_path=test_results_path,
    )
    print(json.dumps({"stage": "finalize", **result}, sort_keys=True), flush=True)
    if not bool(result.get("overall_pass", False)):
        raise RuntimeError(
            f"final artifact validation failed with {result.get('checks_failed', 'unknown')} failed checks"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "development", "selection", "confirmation", "finalize"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--freeze-token", default=None)
    parser.add_argument("--test-results", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.workers) <= 8:
        raise ValueError("--workers must be between 1 and 8")
    if args.stage == "preflight":
        run_preflight(args.data_dir)
    elif args.stage == "development":
        run_development(args.data_dir, resume=not args.no_resume, workers=int(args.workers))
    elif args.stage == "selection":
        run_selection(args.data_dir, resume=not args.no_resume, workers=int(args.workers))
    elif args.stage == "confirmation":
        run_confirmation(
            args.data_dir,
            str(args.freeze_token) if args.freeze_token is not None else "",
            resume=not args.no_resume,
            workers=int(args.workers),
        )
    else:
        run_finalize(args.test_results)


if __name__ == "__main__":
    main()
