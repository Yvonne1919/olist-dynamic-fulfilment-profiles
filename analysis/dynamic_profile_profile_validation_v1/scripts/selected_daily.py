"""Exact selected-candidate daily profile construction.

This module is deliberately side-effect free.  It accepts the already enriched
analysis frame, the frozen configuration, and the immutable promoted-candidate
records, then returns deterministic in-memory tables.  It never reads or writes
artifacts and never implements a second profile estimator: every profile and
parent estimate comes from :func:`profile_core.build_profiles`.

The public entry point constructs all calendar snapshots from 2016-12-03
through 2018-08-30 inclusive.  For each promoted candidate it also constructs
the otherwise identical 30-day and 90-day variants so that
``short_long_trend = score_30d - score_90d`` is available on the estimator's
native scale.  Missing component profiles remain missing; no approximation,
forward fill, or future-outcome lookup is permitted.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Package import in production and pytest.
    from . import profile_core as core
except ImportError:  # Direct import during a local smoke check.
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_core as core


FROZEN_START_DATE = pd.Timestamp("2016-12-03")
FROZEN_END_DATE = pd.Timestamp("2018-08-30")
FROZEN_DATES = pd.date_range(FROZEN_START_DATE, FROZEN_END_DATE, freq="D")

SOURCE_COLUMNS = (
    "target",
    "granularity",
    "scheme",
    "window_days",
    "lag_days",
)

CANDIDATE_COLUMNS = (
    "candidate_id",
    "base_candidate_id",
    "profile_spec_id",
    "target",
    "granularity",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent_structure",
    "kappa",
    "min_support",
    "q33",
    "q67",
)

CONSTRUCTION_PLAN_COLUMNS = (
    "candidate_id",
    "profile_spec_id",
    "component",
    "component_base_candidate_id",
    "target",
    "granularity",
    "scheme",
    "window_days",
    "lag_days",
    "estimator",
    "parent_structure",
    "kappa",
)

DAILY_EXTRA_COLUMNS = (
    "candidate_id",
    "profile_spec_id",
    "min_support",
    "q33",
    "q67",
    "level",
    "unknown_reason",
    "period",
    "score_30d",
    "support_30d",
    "score_90d",
    "support_90d",
    "short_long_trend",
)

# The shared base columns must remain first so the returned frame can be passed
# directly to downstream reporting without silently changing estimator fields.
SELECTED_DAILY_COLUMNS = tuple(core.PROFILE_BASE_COLUMNS) + DAILY_EXTRA_COLUMNS

# Parent estimates are support-rule independent, so they are returned once per
# selected base specification rather than duplicated for multiple min-support
# communication rules.  This matches the frozen parent-artifact primary key.
SELECTED_PARENT_COLUMNS = (
    "base_candidate_id",
    "snapshot_date",
    "target",
    "granularity",
    "parent_structure",
    "parent_id",
    "parent_support",
    "parent_event_count",
    "parent_score",
    "global_score",
    "parent_within_variance",
    "parent_between_variance",
    "parent_posterior_se",
    "parent_interval_lower",
    "parent_interval_upper",
    "fallback_child_count",
    "parent_supported",
    "valid",
    "invalid_reason",
)


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    """Return an empty frame with an exact, ordered schema."""

    return pd.DataFrame(columns=list(columns))


def _is_missing(value: object) -> bool:
    """Return a scalar missingness decision without ambiguous array truth."""

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _normalise_kappa(value: object) -> int | None:
    """Normalise a frozen kappa value to an integer or ``None``."""

    if value is None or _is_missing(value) or str(value).lower() in {"", "na", "none"}:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"kappa must be an integer or missing, got {value!r}")
    return int(numeric)


def _stable_profile_spec_id(base_candidate_id: str) -> str:
    """Return the selection module's deterministic base-specification token."""

    digest = hashlib.sha256(str(base_candidate_id).encode("utf-8")).hexdigest()[:20]
    return f"ps_{digest}"


def _period_for_snapshot(value: object) -> str:
    """Map a frozen snapshot to its non-overlapping protocol period."""

    snapshot = pd.Timestamp(value).normalize()
    if pd.Timestamp("2017-04-01") <= snapshot < pd.Timestamp("2018-01-01"):
        return "development"
    if pd.Timestamp("2018-01-01") <= snapshot < pd.Timestamp("2018-07-01"):
        return "confirmation"
    if pd.Timestamp("2018-07-01") <= snapshot <= FROZEN_END_DATE:
        return "terminal"
    return "warmup_or_outside_evaluation"


def _validate_frozen_date_contract(config: Mapping[str, object]) -> None:
    """Hard-stop if the supplied configuration changes the frozen date span."""

    time_config = config["time"]
    configured_start = pd.Timestamp(time_config["warmup_first_snapshot"]).normalize()
    configured_end = pd.Timestamp(time_config["terminal"]["end_inclusive"]).normalize()
    if configured_start != FROZEN_START_DATE or configured_end != FROZEN_END_DATE:
        raise ValueError(
            "selected daily date contract mismatch: "
            f"{configured_start.date()}..{configured_end.date()} != "
            f"{FROZEN_START_DATE.date()}..{FROZEN_END_DATE.date()}"
        )


def _source_from_row(row: Mapping[str, object], *, window_days: int | None = None) -> dict[str, object]:
    """Build one canonical source mapping accepted by ``build_profiles``."""

    return {
        "target": str(row["target"]),
        "granularity": str(row["granularity"]),
        "scheme": str(row["scheme"]),
        "window_days": int(row["window_days"] if window_days is None else window_days),
        "lag_days": int(row["lag_days"]),
    }


def _source_key(source: Mapping[str, object]) -> tuple[object, ...]:
    """Return the deterministic grouping key for one construction source."""

    return tuple(source[column] for column in SOURCE_COLUMNS)


def _normalise_promoted_candidates(
    promoted_candidates: pd.DataFrame | Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Validate and canonicalise immutable promoted-candidate records.

    The function accepts the field aliases used by the selection freeze
    (``support_threshold``, ``low_medium_cutoff`` and
    ``medium_high_cutoff``), but emits only the canonical schema used here.
    Supplied identifiers are checked rather than silently replaced.
    """

    if isinstance(promoted_candidates, pd.DataFrame):
        frame = promoted_candidates.copy()
    else:
        frame = pd.DataFrame.from_records(list(promoted_candidates))
    if frame.empty:
        return _empty(CANDIDATE_COLUMNS)

    aliases = {
        "support_threshold": "min_support",
        "low_medium_cutoff": "q33",
        "medium_high_cutoff": "q67",
    }
    for source_name, destination in aliases.items():
        if destination in frame and source_name in frame:
            left = pd.to_numeric(frame[destination], errors="coerce")
            right = pd.to_numeric(frame[source_name], errors="coerce")
            equal = left.eq(right) | (left.isna() & right.isna())
            if not equal.all():
                raise ValueError(
                    f"promoted candidate aliases disagree: {source_name} and {destination}"
                )
        elif destination not in frame and source_name in frame:
            frame[destination] = frame[source_name]

    required = {
        "target",
        "granularity",
        "scheme",
        "window_days",
        "lag_days",
        "estimator",
        "parent_structure",
        "min_support",
        "q33",
        "q67",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"promoted candidates are missing fields: {missing}")

    support_options = {int(value) for value in config["levels"]["support_candidates"]}
    frozen_sources = {_source_key(source) for source in core.candidate_sources()}
    normalised: list[dict[str, object]] = []
    for raw in frame.to_dict("records"):
        target = str(raw["target"])
        granularity = str(raw["granularity"])
        scheme = str(raw["scheme"])
        window_days = int(raw["window_days"])
        lag_days = int(raw["lag_days"])
        estimator = str(raw["estimator"])
        parent_structure = str(raw["parent_structure"])
        kappa = _normalise_kappa(raw.get("kappa"))
        min_support = int(raw["min_support"])
        raw_q33 = raw.get("q33")
        raw_q67 = raw.get("q67")
        q33 = np.nan if _is_missing(raw_q33) else float(raw_q33)
        q67 = np.nan if _is_missing(raw_q67) else float(raw_q67)

        if target not in core.TARGET_SPECS:
            raise ValueError(f"unknown target in promoted candidate: {target}")
        if granularity not in core.TARGET_SPECS[target]["granularities"]:
            raise ValueError(f"invalid target/granularity pair: {target}/{granularity}")
        if scheme not in {"A", "C"}:
            raise ValueError(f"unsupported construction scheme: {scheme}")
        if window_days not in {30, 60, 90}:
            raise ValueError(f"window_days is outside the frozen catalog: {window_days}")
        if min_support not in support_options:
            raise ValueError(f"min_support is outside the frozen catalog: {min_support}")
        # A promoted continuous profile may have an invalid communication-level
        # layer without invalidating the continuous score itself.  Preserve
        # missing cutoffs so every daily row is deterministically Unknown with
        # ``invalid_frozen_cutoffs``; only a finite reversed pair is malformed.
        if np.isfinite(q33) and np.isfinite(q67) and q33 > q67:
            raise ValueError(f"invalid frozen q33/q67 for {target}/{granularity}")

        source = {
            "target": target,
            "granularity": granularity,
            "scheme": scheme,
            "window_days": window_days,
            "lag_days": lag_days,
        }
        if _source_key(source) not in frozen_sources:
            raise ValueError(f"candidate source is outside the frozen catalog: {source}")
        expected_base_id = core.base_candidate_id(
            source, estimator, parent_structure, kappa,
        )
        available_variants = {
            str(variant["base_candidate_id"])
            for variant in core.candidate_variants(source)
        }
        if expected_base_id not in available_variants:
            raise ValueError(f"candidate is outside the frozen estimator catalog: {expected_base_id}")

        supplied_base = raw.get("base_candidate_id")
        if supplied_base is not None and not _is_missing(supplied_base):
            if str(supplied_base) != expected_base_id:
                raise ValueError(
                    f"base_candidate_id disagrees with frozen fields: {supplied_base!r} != {expected_base_id!r}"
                )
        expected_candidate_id = f"{expected_base_id}|min_support={min_support}"
        supplied_candidate = raw.get("candidate_id")
        if supplied_candidate is not None and not _is_missing(supplied_candidate):
            if str(supplied_candidate) != expected_candidate_id:
                raise ValueError(
                    f"candidate_id disagrees with frozen fields: {supplied_candidate!r} != {expected_candidate_id!r}"
                )
        expected_spec_id = _stable_profile_spec_id(expected_base_id)
        supplied_spec = raw.get("profile_spec_id")
        if supplied_spec is not None and not _is_missing(supplied_spec):
            if str(supplied_spec) != expected_spec_id:
                raise ValueError(
                    f"profile_spec_id disagrees with base candidate: {supplied_spec!r} != {expected_spec_id!r}"
                )

        normalised.append(
            {
                "candidate_id": expected_candidate_id,
                "base_candidate_id": expected_base_id,
                "profile_spec_id": expected_spec_id,
                **source,
                "estimator": estimator,
                "parent_structure": parent_structure,
                "kappa": kappa,
                "min_support": min_support,
                "q33": q33,
                "q67": q67,
            }
        )

    result = pd.DataFrame.from_records(normalised, columns=CANDIDATE_COLUMNS)
    if result["candidate_id"].duplicated().any():
        duplicate = result.loc[result["candidate_id"].duplicated(), "candidate_id"].iloc[0]
        raise ValueError(f"duplicate promoted candidate_id: {duplicate}")
    return result.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def build_construction_plan(
    promoted_candidates: pd.DataFrame | Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return the exact selected/30d/90d construction plan.

    The plan is useful for runner-side progress and audit inputs.  It contains
    no observed outcome values and does not construct profiles.  Repeated
    component sources are intentionally visible at candidate level; the
    execution path deduplicates them by source and base-candidate ID.
    """

    candidates = _normalise_promoted_candidates(promoted_candidates, config)
    if candidates.empty:
        return _empty(CONSTRUCTION_PLAN_COLUMNS)
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        for component, window_days in (
            ("selected", int(candidate["window_days"])),
            ("30d", 30),
            ("90d", 90),
        ):
            source = _source_from_row(candidate, window_days=window_days)
            component_base_id = core.base_candidate_id(
                source,
                str(candidate["estimator"]),
                str(candidate["parent_structure"]),
                _normalise_kappa(candidate.get("kappa")),
            )
            available = {
                str(variant["base_candidate_id"])
                for variant in core.candidate_variants(source)
            }
            if component_base_id not in available:
                raise ValueError(f"missing frozen {component} counterpart: {component_base_id}")
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "profile_spec_id": candidate["profile_spec_id"],
                    "component": component,
                    "component_base_candidate_id": component_base_id,
                    **source,
                    "estimator": candidate["estimator"],
                    "parent_structure": candidate["parent_structure"],
                    "kappa": candidate["kappa"],
                }
            )
    return (
        pd.DataFrame.from_records(rows, columns=CONSTRUCTION_PLAN_COLUMNS)
        .sort_values(["candidate_id", "component"], kind="mergesort")
        .reset_index(drop=True)
    )


def _assign_levels(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the immutable Unknown/Low/Medium/High communication rule."""

    result = rows.copy()
    score = pd.to_numeric(result["score"], errors="coerce")
    support = pd.to_numeric(result["support"], errors="coerce")
    lower = pd.to_numeric(result["lower_interval"], errors="coerce")
    upper = pd.to_numeric(result["upper_interval"], errors="coerce")
    q33 = pd.to_numeric(result["q33"], errors="coerce")
    q67 = pd.to_numeric(result["q67"], errors="coerce")
    minimum = pd.to_numeric(result["min_support"], errors="coerce")
    cold = result["cold_start"].fillna(False).astype(bool) | support.fillna(0).eq(0)
    result["cold_start"] = cold

    masks_and_reasons = (
        (cold, "cold_start"),
        (support.isna() | support.lt(minimum), "below_min_support"),
        (~np.isfinite(score), "nonfinite_score"),
        (~np.isfinite(lower) | ~np.isfinite(upper), "nonfinite_interval"),
        (~np.isfinite(q33) | ~np.isfinite(q67) | q33.gt(q67), "invalid_frozen_cutoffs"),
        (lower.le(q33) & upper.ge(q67), "interval_spans_both_cutoffs"),
    )
    unknown = pd.Series(False, index=result.index)
    reason = pd.Series("", index=result.index, dtype="object")
    for mask, label in masks_and_reasons:
        assign = ~unknown & mask.fillna(True)
        reason.loc[assign] = label
        unknown |= assign

    result["level"] = np.select(
        [unknown, score.le(q33), score.le(q67)],
        ["Unknown", "Low", "Medium"],
        default="High",
    )
    result["unknown_reason"] = reason
    return result


def _validate_profiles(
    profiles: pd.DataFrame,
    parents: pd.DataFrame,
    *,
    source: Mapping[str, object],
    snapshot: pd.Timestamp,
    allowed_base_ids: set[str],
) -> None:
    """Hard-stop on malformed or out-of-source shared-builder output."""

    missing = sorted(set(core.PROFILE_BASE_COLUMNS) - set(profiles.columns))
    if missing:
        raise RuntimeError(f"shared build_profiles omitted base columns: {missing}")
    if not profiles.empty:
        unexpected = sorted(set(profiles["base_candidate_id"].astype(str)) - allowed_base_ids)
        if unexpected:
            raise RuntimeError(f"shared build_profiles returned unrequested candidates: {unexpected[:3]}")
        snapshots = pd.to_datetime(profiles["snapshot_date"], errors="raise").dt.normalize()
        if not snapshots.eq(pd.Timestamp(snapshot).normalize()).all():
            raise RuntimeError("shared build_profiles returned a non-requested snapshot")
        for field in SOURCE_COLUMNS:
            expected = source[field]
            observed = profiles[field]
            if field in {"window_days", "lag_days"}:
                matches = pd.to_numeric(observed, errors="coerce").eq(int(expected))
            else:
                matches = observed.astype(str).eq(str(expected))
            if not matches.all():
                raise RuntimeError(f"shared build_profiles returned mismatched {field}")
        if profiles.duplicated(["base_candidate_id", "entity_id"]).any():
            raise RuntimeError("shared build_profiles returned duplicate entity rows")
    if not parents.empty:
        if "base_candidate_id" not in parents or "parent_id" not in parents:
            raise RuntimeError("shared parent output lacks base_candidate_id or parent_id")
        parent_unexpected = sorted(set(parents["base_candidate_id"].astype(str)) - allowed_base_ids)
        if parent_unexpected:
            raise RuntimeError(f"shared build_profiles returned unrequested parents: {parent_unexpected[:3]}")
        if "snapshot_date" not in parents:
            raise RuntimeError("shared parent output lacks snapshot_date")
        parent_snapshots = pd.to_datetime(
            parents["snapshot_date"], errors="raise",
        ).dt.normalize()
        if not parent_snapshots.eq(pd.Timestamp(snapshot).normalize()).all():
            raise RuntimeError("shared build_profiles returned parents for a non-requested snapshot")
        if parents.duplicated(["base_candidate_id", "parent_id"]).any():
            raise RuntimeError("shared build_profiles returned duplicate parent rows")


def _parent_rows_for_base(
    raw_parents: pd.DataFrame,
    profiles: pd.DataFrame,
    candidate: Mapping[str, object],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Normalise one shared-builder parent table to the fixed parent schema."""

    if raw_parents.empty:
        if profiles.empty:
            return _empty(SELECTED_PARENT_COLUMNS)
        raise RuntimeError(f"profile rows exist without parent rows for {candidate['base_candidate_id']}")
    parents = raw_parents.copy()
    parents["target"] = str(candidate["target"])
    parents["granularity"] = str(candidate["granularity"])
    parents["parent_structure"] = str(candidate["parent_structure"])

    aliases = {
        "support": "parent_support",
        "event_count": "parent_event_count",
        "score": "parent_score",
        "posterior_se": "parent_posterior_se",
        "lower_interval": "parent_interval_lower",
        "upper_interval": "parent_interval_upper",
        "within_variance": "parent_within_variance",
        "between_variance": "parent_between_variance",
    }
    for source_name, destination in aliases.items():
        if destination not in parents and source_name in parents:
            parents[destination] = parents[source_name]
    for column in SELECTED_PARENT_COLUMNS:
        if column not in parents:
            parents[column] = np.nan

    parent_minimum = int(config["binary_eb"]["parent_min_support"])
    parent_support = pd.to_numeric(parents["parent_support"], errors="coerce")
    parents["parent_supported"] = parent_support.ge(parent_minimum)
    parents["valid"] = np.isfinite(pd.to_numeric(parents["parent_score"], errors="coerce"))
    parents["invalid_reason"] = np.where(
        parents["valid"], "", "nonfinite_parent_score",
    )
    parents["fallback_child_count"] = 0
    if str(candidate["parent_structure"]) != "global" and not profiles.empty:
        child_parent = profiles["parent_id"].astype(str)
        for index, parent in parents.iterrows():
            unsupported = not bool(parents.at[index, "parent_supported"])
            if unsupported:
                parents.at[index, "fallback_child_count"] = int(
                    child_parent.eq(str(parent["parent_id"])).sum()
                )
    return parents.loc[:, list(SELECTED_PARENT_COLUMNS)]


def _generate_on_dates(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    promoted_candidates: pd.DataFrame | Sequence[Mapping[str, object]],
    dates: Iterable[str | pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Internal exact constructor for an explicit, unique daily date grid.

    This helper exists so a tiny synthetic smoke test can exercise the same
    implementation without performing the full 636-day production run.  The
    public function below always supplies the complete frozen grid.
    """

    candidates = _normalise_promoted_candidates(promoted_candidates, config)
    if candidates.empty:
        return _empty(SELECTED_DAILY_COLUMNS), _empty(SELECTED_PARENT_COLUMNS)
    snapshots = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise")).normalize()
    snapshots = snapshots.sort_values()
    if snapshots.empty:
        return _empty(SELECTED_DAILY_COLUMNS), _empty(SELECTED_PARENT_COLUMNS)
    if snapshots.has_duplicates:
        raise ValueError("snapshot dates must be unique")
    if snapshots.min() < FROZEN_START_DATE or snapshots.max() > FROZEN_END_DATE:
        raise ValueError("selected daily snapshots must remain inside the frozen date span")
    if len(snapshots) > 1 and not np.all(np.diff(snapshots.asi8) == pd.Timedelta(days=1).value):
        raise ValueError("selected daily construction requires consecutive calendar dates")

    plan = build_construction_plan(candidates, config)
    construction_sources: dict[tuple[object, ...], dict[str, object]] = {}
    allowed_by_source: dict[tuple[object, ...], set[str]] = defaultdict(set)
    plan_lookup: dict[tuple[str, str], str] = {}
    for record in plan.to_dict("records"):
        source = _source_from_row(record)
        key = _source_key(source)
        construction_sources[key] = source
        allowed_by_source[key].add(str(record["component_base_candidate_id"]))
        plan_lookup[(str(record["candidate_id"]), str(record["component"]))] = str(
            record["component_base_candidate_id"]
        )

    daily_parts: list[pd.DataFrame] = []
    parent_parts: list[pd.DataFrame] = []
    unique_main = candidates.drop_duplicates("base_candidate_id", keep="first")
    for snapshot in snapshots:
        profiles_by_base: dict[str, pd.DataFrame] = {}
        parents_by_base: dict[str, pd.DataFrame] = {}
        for key in sorted(construction_sources, key=lambda item: tuple(map(str, item))):
            source = construction_sources[key]
            allowed = allowed_by_source[key]
            profiles, parents = core.build_profiles(
                frame,
                source,
                pd.Timestamp(snapshot),
                config,
                allowed_base_ids=set(allowed),
            )
            _validate_profiles(
                profiles,
                parents,
                source=source,
                snapshot=pd.Timestamp(snapshot),
                allowed_base_ids=allowed,
            )
            if not profiles.empty:
                for base_id, group in profiles.groupby("base_candidate_id", sort=True):
                    profiles_by_base[str(base_id)] = group.copy()
            if not parents.empty:
                for base_id, group in parents.groupby("base_candidate_id", sort=True):
                    parents_by_base[str(base_id)] = group.copy()

        for candidate in candidates.to_dict("records"):
            candidate_id = str(candidate["candidate_id"])
            main_id = str(candidate["base_candidate_id"])
            main = profiles_by_base.get(main_id)
            if main is None or main.empty:
                continue
            rows = main.copy()
            rows["candidate_id"] = candidate_id
            rows["profile_spec_id"] = str(candidate["profile_spec_id"])
            rows["min_support"] = int(candidate["min_support"])
            rows["q33"] = float(candidate["q33"])
            rows["q67"] = float(candidate["q67"])
            rows["period"] = _period_for_snapshot(snapshot)

            for component, score_column, support_column in (
                ("30d", "score_30d", "support_30d"),
                ("90d", "score_90d", "support_90d"),
            ):
                component_id = plan_lookup[(candidate_id, component)]
                component_rows = profiles_by_base.get(component_id)
                if component_rows is None or component_rows.empty:
                    rows[score_column] = np.nan
                    rows[support_column] = np.nan
                    continue
                lookup = component_rows[["entity_id", "score", "support"]].rename(
                    columns={"score": score_column, "support": support_column},
                )
                rows = rows.merge(
                    lookup,
                    on="entity_id",
                    how="left",
                    validate="one_to_one",
                    sort=False,
                )

            score_30 = pd.to_numeric(rows["score_30d"], errors="coerce")
            score_90 = pd.to_numeric(rows["score_90d"], errors="coerce")
            rows["short_long_trend"] = (score_30 - score_90).where(
                np.isfinite(score_30) & np.isfinite(score_90),
            )
            rows = _assign_levels(rows)
            daily_parts.append(rows.loc[:, list(SELECTED_DAILY_COLUMNS)])

        for candidate in unique_main.to_dict("records"):
            base_id = str(candidate["base_candidate_id"])
            profiles = profiles_by_base.get(base_id)
            parents = parents_by_base.get(base_id)
            if profiles is None or profiles.empty:
                continue
            parent_parts.append(
                _parent_rows_for_base(
                    _empty(()) if parents is None else parents,
                    profiles,
                    candidate,
                    config,
                )
            )

    daily = (
        pd.concat(daily_parts, ignore_index=True)
        if daily_parts else _empty(SELECTED_DAILY_COLUMNS)
    )
    parents = (
        pd.concat(parent_parts, ignore_index=True)
        if parent_parts else _empty(SELECTED_PARENT_COLUMNS)
    )
    daily = daily.loc[:, list(SELECTED_DAILY_COLUMNS)].sort_values(
        ["candidate_id", "snapshot_date", "entity_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    parents = parents.loc[:, list(SELECTED_PARENT_COLUMNS)].sort_values(
        ["base_candidate_id", "snapshot_date", "parent_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    if daily.duplicated(["candidate_id", "snapshot_date", "entity_id"]).any():
        raise RuntimeError("selected daily profile primary key is duplicated")
    if parents.duplicated(["base_candidate_id", "snapshot_date", "parent_id"]).any():
        raise RuntimeError("selected parent profile primary key is duplicated")
    return daily, parents


def generate_selected_daily_profiles(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    promoted_candidates: pd.DataFrame | Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the complete frozen selected-profile daily tables.

    Parameters
    ----------
    frame:
        The already enriched full analysis frame.  This function does not read
        outcome columns directly; the shared builder applies the strict
        ``label_available_at < snapshot`` history rule.
    config:
        The frozen profile-validation configuration.
    promoted_candidates:
        Immutable records from ``PROFILE_SELECTION_FREEZE.json`` (or an
        equivalent DataFrame).  An empty promotion set is a valid negative
        selection and returns two exact-schema empty frames.

    Returns
    -------
    (daily_profiles, parent_profiles):
        Deterministically sorted DataFrames.  No artifacts are written.  Daily
        rows span the inclusive 2016-12-03..2018-08-30 grid wherever the shared
        estimator produces an entity profile.  Parent rows are unique by base
        specification, snapshot and parent because communication support rules
        do not change parent estimates.
    """

    _validate_frozen_date_contract(config)
    return _generate_on_dates(
        frame,
        config,
        promoted_candidates,
        FROZEN_DATES,
    )


# Concise integration alias.
build_selected_daily_profiles = generate_selected_daily_profiles


__all__ = [
    "CANDIDATE_COLUMNS",
    "CONSTRUCTION_PLAN_COLUMNS",
    "FROZEN_DATES",
    "FROZEN_END_DATE",
    "FROZEN_START_DATE",
    "SELECTED_DAILY_COLUMNS",
    "SELECTED_PARENT_COLUMNS",
    "build_construction_plan",
    "build_selected_daily_profiles",
    "generate_selected_daily_profiles",
]
