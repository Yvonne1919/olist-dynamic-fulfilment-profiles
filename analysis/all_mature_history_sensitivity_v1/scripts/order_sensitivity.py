from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from analysis.all_mature_history_sensitivity_v1.scripts import sensitivity_core as sc
from analysis.direct_promise_profile_extension_v1.scripts import direct_experiment as de
from analysis.order_breach_severity_v1.scripts import order_modeling
from analysis.order_breach_severity_v1.scripts import order_profiles
from analysis.profile_pivot_phase2a.scripts import data_pipeline


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/all_mature_history_sensitivity_v1"
WORK = OUT / "working"
DIRECT = ROOT / "analysis/direct_promise_profile_extension_v1"
RAW_DIR = Path("data/olist_data")

PROFILE_BLOCKS = ("S1", "S2", "R1", "R2")
SWAP_SUFFIXES = (
    "score",
    "log1p_support",
    "cold_start",
    "posterior_se",
    "freshness_days",
    "mapping_status",
    "last_mature_outcome_date",
    "support",
)
BREACH_CANDIDATES = {
    "DPS": ("S1", "S2"),
    "DPG": ("R1", "R2"),
    "DPB": ("S1", "S2", "R1", "R2"),
}
SEVERITY_CANDIDATES = {
    "DQS": ("S1", "S2"),
    "DQG": ("R1", "R2"),
    "DQB": ("S1", "S2", "R1", "R2"),
}
PROFILE_BLOCK_NAMES = {
    "DPS": "all-mature seller pre-handoff profiles",
    "DPG": "all-mature state-OD transit profiles",
    "DPB": "all-mature seller pre-handoff and state-OD transit profiles",
    "DQS": "all-mature seller pre-handoff profiles",
    "DQG": "all-mature state-OD transit profiles",
    "DQB": "all-mature seller pre-handoff and state-OD transit profiles",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_id_sha256(order_ids: Iterable[object]) -> str:
    values = [str(value) for value in order_ids]
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _string_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.astype("string").fillna("__MISSING__").eq(
        right.astype("string").fillna("__MISSING__")
    )


def _support_stratum(support: pd.Series, cold: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(support, errors="coerce").fillna(0)
    result = pd.Series(
        np.select(
            [numeric.lt(5), numeric.between(5, 9), numeric.between(10, 19), numeric.ge(20)],
            ["0-4", "5-9", "10-19", "20+"],
            default="0-4",
        ),
        index=support.index,
        dtype="string",
    )
    result.loc[cold.fillna(False).astype(bool)] = "cold_start"
    return result


def _load_direct_inputs() -> tuple[dict, dict, pd.DataFrame, dict[str, object]]:
    gate = _json(WORK / "DIRECT_EXTENSION_GATE.json")
    if gate.get("available") is not True:
        raise RuntimeError("frozen direct-extension gate is not true")
    current_manifest_sha = _sha256(DIRECT / "RUN_MANIFEST.json")
    if current_manifest_sha != gate.get("manifest_sha256"):
        raise RuntimeError("direct-extension manifest changed after the frozen preflight gate")
    config = de.load_config()
    selection_path = DIRECT / "DIRECT_MODEL_SELECTION_FREEZE.json"
    selection = _json(selection_path)
    de._validate_selection(selection, config)
    frame = de.load_and_validate_frame(config)
    receipt = {
        "gate": gate,
        "direct_config_sha256": _sha256(DIRECT / "DIRECT_FROZEN_CONFIG.json"),
        "direct_selection_sha256": _sha256(selection_path),
        "direct_experiment_sha256": _sha256(DIRECT / "scripts/direct_experiment.py"),
        "order_profiles_sha256": _sha256(
            ROOT / "analysis/order_breach_severity_v1/scripts/order_profiles.py"
        ),
        "order_modeling_sha256": _sha256(
            ROOT / "analysis/order_breach_severity_v1/scripts/order_modeling.py"
        ),
        "source_model_frame_sha256": config["sources"]["order_model_frame"][1],
        "rows": len(frame),
        "columns": len(frame.columns),
        "order_id_sha256": order_modeling.order_id_hash(frame["order_id"]),
    }
    return config, selection, frame, receipt


def _canonical_entity_keys(frame: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, object]]:
    canonical = data_pipeline.build_order_base(RAW_DIR)
    required = [
        "order_id", "order_purchase_timestamp", "main_seller_id",
        "main_seller_state", "customer_state",
    ]
    if canonical["order_id"].duplicated().any():
        raise AssertionError("canonical assembler returned duplicate order IDs")
    indexed = canonical.set_index("order_id", verify_integrity=True)
    missing = sorted(set(frame["order_id"].astype(str)) - set(indexed.index.astype(str)))
    if missing:
        raise AssertionError(f"direct frame has canonical-key misses: {missing[:5]}")
    keys = indexed.loc[frame["order_id"].astype(str), required[1:]].reset_index(drop=True)
    purchase = pd.to_datetime(keys["order_purchase_timestamp"], errors="raise")
    if not purchase.eq(pd.to_datetime(frame["order_purchase_timestamp"], errors="raise")).all():
        raise AssertionError("canonical and direct purchase timestamps disagree")
    for column in ("main_seller_state", "customer_state"):
        if not _string_equal(keys[column], frame[column]).all():
            raise AssertionError(f"canonical and direct {column} disagree")
    join_frame = pd.DataFrame(
        {
            "order_id": frame["order_id"].astype(str).to_numpy(),
            "order_purchase_timestamp": frame["order_purchase_timestamp"].to_numpy(),
            "main_seller_id": keys["main_seller_id"].astype("string").to_numpy(),
            "main_seller_state": frame["main_seller_state"].astype("string").to_numpy(),
            "customer_state": frame["customer_state"].astype("string").to_numpy(),
        },
        index=frame.index,
    )
    entities = order_profiles._deterministic_entities(join_frame)
    if any(values.isna().any() for values in entities.values()):
        raise AssertionError("direct sensitivity unexpectedly has missing deterministic entity keys")
    audit = {
        "canonical_rows": len(canonical),
        "direct_rows_matched": len(keys),
        "direct_order_id_sha256": order_modeling.order_id_hash(frame["order_id"]),
        "purchase_timestamp_mismatches": 0,
        "main_seller_state_mismatches": 0,
        "customer_state_mismatches": 0,
        "seller_levels": int(entities["seller_id"].nunique()),
        "state_od_levels": int(entities["state_od"].nunique()),
    }
    return entities, audit


def _load_all_mature_join_stores(
    snapshots: pd.Series,
    entities: Mapping[str, pd.Series],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, object]]:
    store_path = OUT / "ALL_MATURE_PROFILE_DAILY_SCORES.csv.gz"
    parent_path = OUT / "ALL_MATURE_PROFILE_PARENT_STRUCTURE.csv.gz"
    index = pd.read_csv(OUT / "ALL_MATURE_PROFILE_STORE_INDEX.csv", low_memory=False)
    if len(index) != 4 or set(index["profile_code"].astype(str)) != set(PROFILE_BLOCKS):
        raise AssertionError("all-mature profile-store index does not contain the four frozen blocks")
    expected_hashes = set(index["store_sha256"].astype(str))
    if expected_hashes != {_sha256(store_path)}:
        raise AssertionError("all-mature profile store differs from its persisted index")
    code_to_spec = {str(row["profile_code"]): row for _, row in index.iterrows()}
    key_sets: dict[str, pd.MultiIndex] = {}
    for code in PROFILE_BLOCKS:
        entity_name = "seller_id" if code.startswith("S") else "state_od"
        key_sets[code] = pd.MultiIndex.from_arrays(
            [snapshots.to_numpy(), entities[entity_name].astype(str).to_numpy()],
            names=["snapshot_date", "entity_id"],
        ).unique()
    usecols = [
        "profile_code", "candidate_id", "snapshot_date", "entity_id", "score", "support",
        "posterior_se", "profile_freshness_days", "last_mature_outcome_date",
        "history_mode", "effective_history_lower_bound",
    ]
    retained: dict[str, list[pd.DataFrame]] = {code: [] for code in PROFILE_BLOCKS}
    row_counts = {code: 0 for code in PROFILE_BLOCKS}
    date_sets: dict[str, set[str]] = {code: set() for code in PROFILE_BLOCKS}
    for chunk in pd.read_csv(store_path, usecols=usecols, chunksize=180_000, low_memory=False):
        if not chunk["history_mode"].astype(str).eq("all_mature").all():
            raise AssertionError("all-mature store contains a different history mode")
        if not chunk["effective_history_lower_bound"].astype(str).eq("none").all():
            raise AssertionError("all-mature store contains a nonempty lower history bound")
        chunk["snapshot_date"] = pd.to_datetime(chunk["snapshot_date"], errors="raise")
        for code in PROFILE_BLOCKS:
            block = chunk.loc[chunk["profile_code"].astype(str).eq(code)].copy()
            if block.empty:
                continue
            row_counts[code] += len(block)
            date_sets[code].update(block["snapshot_date"].dt.strftime("%Y-%m-%d").unique())
            expected_candidate = str(code_to_spec[code]["candidate_id"])
            if not block["candidate_id"].astype(str).eq(expected_candidate).all():
                raise AssertionError(f"{code} candidate differs from frozen store index")
            block_keys = pd.MultiIndex.from_frame(block[["snapshot_date", "entity_id"]])
            keep = block_keys.isin(key_sets[code])
            if keep.any():
                retained[code].append(block.loc[keep])
    profiles: dict[str, pd.DataFrame] = {}
    expected_dates = pd.date_range("2016-12-03", "2018-08-30", freq="D")
    for code in PROFILE_BLOCKS:
        expected_rows = int(code_to_spec[code]["row_count"])
        if row_counts[code] != expected_rows or len(date_sets[code]) != 636:
            raise AssertionError(
                f"{code} all-mature store coverage mismatch: rows={row_counts[code]}, "
                f"dates={len(date_sets[code])}"
            )
        if set(date_sets[code]) != set(expected_dates.strftime("%Y-%m-%d")):
            raise AssertionError(f"{code} does not contain the exact 636 snapshot dates")
        profile = pd.concat(retained[code], ignore_index=True)
        profile["snapshot_date"] = pd.to_datetime(profile["snapshot_date"], errors="raise")
        profile["last_mature_outcome_date"] = pd.to_datetime(
            profile["last_mature_outcome_date"], errors="raise"
        )
        if profile.duplicated(["snapshot_date", "entity_id"]).any():
            raise AssertionError(f"{code} retained order-key store contains duplicates")
        if not profile["last_mature_outcome_date"].lt(profile["snapshot_date"]).all():
            raise AssertionError(f"{code} retained store violates strict maturity")
        profiles[code] = profile
    parent = pd.read_csv(parent_path, low_memory=False)
    if len(parent) != 4 * 636 or not parent["parent_id"].astype(str).eq("__GLOBAL__").all():
        raise AssertionError("all-mature parent store has the wrong row or parent-key count")
    parent["snapshot_date"] = pd.to_datetime(parent["snapshot_date"], errors="raise")
    parent_scores: dict[str, pd.Series] = {}
    for code in PROFILE_BLOCKS:
        rows = parent.loc[parent["profile_code"].astype(str).eq(code)].copy()
        if len(rows) != 636 or rows["snapshot_date"].duplicated().any():
            raise AssertionError(f"{code} parent store does not contain one row per date")
        score = pd.to_numeric(rows["parent_score"], errors="raise")
        global_score = pd.to_numeric(rows["global_score"], errors="raise")
        if not np.isfinite(score).all() or not np.allclose(score, global_score, rtol=0, atol=1e-12):
            raise AssertionError(f"{code} all-mature global-parent score is invalid")
        parent_scores[code] = pd.Series(
            score.to_numpy(float), index=rows["snapshot_date"].to_numpy(), name="parent_score"
        ).sort_index()
    receipt = {
        "daily_store_sha256": _sha256(store_path),
        "daily_store_rows": int(sum(row_counts.values())),
        "daily_rows_by_profile": row_counts,
        "retained_rows_by_profile": {code: len(profiles[code]) for code in PROFILE_BLOCKS},
        "parent_store_sha256": _sha256(parent_path),
        "parent_store_rows": len(parent),
        "snapshot_count_each": 636,
    }
    return profiles, parent_scores, receipt


def construct_all_mature_frame(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    entities, canonical_audit = _canonical_entity_keys(base)
    snapshots = pd.to_datetime(base["order_purchase_timestamp"], errors="raise").dt.normalize()
    if not snapshots.eq(pd.to_datetime(base["purchase_date"], errors="raise")).all():
        raise AssertionError("normalised purchase timestamp differs from persisted purchase_date")
    profiles, parent_scores, store_receipt = _load_all_mature_join_stores(
        snapshots, entities
    )
    result = base.copy()
    audit_rows: list[dict[str, object]] = []
    for code in PROFILE_BLOCKS:
        entity_name = "seller_id" if code.startswith("S") else "state_od"
        block_audit = order_profiles._join_one_block(
            result,
            snapshots,
            entities[entity_name],
            profiles[code],
            parent_scores[code],
            code,
        )
        all_support = pd.to_numeric(result[f"{code}_support"], errors="raise")
        selected_support = pd.to_numeric(base[f"{code}_support"], errors="raise")
        support_decreases = int(all_support.lt(selected_support).sum())
        if support_decreases:
            raise AssertionError(f"{code} all-mature support decreased on {support_decreases} orders")
        audit_rows.append(
            {
                "profile_code": code,
                "entity_key": entity_name,
                **block_audit,
                "selected_90d_orders_seen": int(base[f"{code}_mapping_status"].eq("seen").sum()),
                "all_mature_support_decrease_count": support_decreases,
                "selected_90d_median_support": float(selected_support.median()),
                "all_mature_median_support": float(all_support.median()),
            }
        )
    swap = [f"{code}_{suffix}" for code in PROFILE_BLOCKS for suffix in SWAP_SUFFIXES]
    non_swap = [column for column in base.columns if column not in swap]
    if len(swap) != 32 or len(non_swap) != len(base.columns) - 32:
        raise AssertionError("profile swap is not exactly 32 columns")
    if not base[non_swap].equals(result[non_swap]):
        raise AssertionError("a non-profile column changed during the all-mature swap")
    if len(result) != 91_254 or list(result.columns) != list(base.columns):
        raise AssertionError("all-mature model frame changed the frozen row/column contract")
    if not result.index.equals(base.index) or not np.array_equal(
        result["order_id"].astype(str).to_numpy(), base["order_id"].astype(str).to_numpy()
    ):
        raise AssertionError("all-mature model frame changed row order or index")
    if order_modeling.order_id_hash(result["order_id"]) != de.load_config()["population"][
        "model_frame_order_id_sha256"
    ]:
        raise AssertionError("all-mature model frame changed the frozen order population")
    de.load_and_validate_frame  # Explicitly document that validation semantics are reused below.
    for code in PROFILE_BLOCKS:
        status = result[f"{code}_mapping_status"].astype("string")
        cold = result[f"{code}_cold_start"].astype(bool)
        if not cold.eq(status.eq("mapped_cold_start")).all():
            raise AssertionError(f"{code} cold-start/mapping status mismatch")
        seen = status.eq("seen")
        mature = pd.to_datetime(result[f"{code}_last_mature_outcome_date"], errors="coerce")
        if mature.loc[seen].isna().any() or not mature.loc[seen].lt(snapshots.loc[seen]).all():
            raise AssertionError(f"{code} order exposures violate strict as-of maturity")
        if not np.isfinite(pd.to_numeric(result[f"{code}_score"], errors="raise")).all():
            raise AssertionError(f"{code} joined scores are incomplete")
    receipt = {
        **canonical_audit,
        **store_receipt,
        "swapped_columns": swap,
        "swapped_column_count": len(swap),
        "non_swapped_column_count": len(non_swap),
        "non_swapped_columns_exactly_equal": True,
        "rows": len(result),
        "columns": len(result.columns),
        "order_id_sha256": order_modeling.order_id_hash(result["order_id"]),
        "ordered_order_id_sha256": _ordered_id_sha256(result["order_id"]),
        "strict_maturity_violations": 0,
    }
    return result, pd.DataFrame(audit_rows), receipt


def _reproduction_audit(
    all_breach: pd.DataFrame,
    all_severity: pd.DataFrame,
) -> dict[str, object]:
    protected_breach = pd.read_csv(
        DIRECT / "working/DIRECT_BREACH_PREDICTIONS.csv.gz", low_memory=False
    )
    protected_severity = pd.read_csv(
        DIRECT / "working/DIRECT_SEVERITY_PREDICTIONS.csv.gz", low_memory=False
    )
    breach_keys = ["period", "cohort", "family", "model_id", "representation", "order_id"]
    severity_keys = [
        "period", "cohort", "family", "model_id", "quantile", "representation", "order_id"
    ]
    if protected_breach.duplicated(breach_keys).any() or all_breach.duplicated(breach_keys).any():
        raise AssertionError("breach prediction primary key is duplicated")
    if protected_severity.duplicated(severity_keys).any() or all_severity.duplicated(severity_keys).any():
        raise AssertionError("severity prediction primary key is duplicated")
    left = all_breach.loc[all_breach["model_id"].eq("DP0")].merge(
        protected_breach.loc[protected_breach["model_id"].eq("DP0")][
            breach_keys + ["target", "raw_probability", "calibrated_probability", "fitted_model_sha256"]
        ],
        on=breach_keys,
        how="outer",
        indicator=True,
        suffixes=("_all_mature", "_protected"),
        validate="one_to_one",
    )
    if not left["_merge"].eq("both").all() or not left["target_all_mature"].eq(
        left["target_protected"]
    ).all():
        raise AssertionError("DP0 paired prediction population/target mismatch")
    def serialised(values: pd.Series) -> np.ndarray:
        array = pd.to_numeric(values, errors="raise").to_numpy(float)
        return np.char.mod("%.12g", array).astype(float)

    breach_raw_diff = float(
        np.max(np.abs(left["raw_probability_all_mature"] - left["raw_probability_protected"]))
    )
    breach_cal_diff = float(
        np.max(
            np.abs(
                left["calibrated_probability_all_mature"]
                - left["calibrated_probability_protected"]
            )
        )
    )
    breach_serialised_exact = bool(
        np.array_equal(
            serialised(left["raw_probability_all_mature"]),
            left["raw_probability_protected"].to_numpy(float),
        )
        and np.array_equal(
            serialised(left["calibrated_probability_all_mature"]),
            left["calibrated_probability_protected"].to_numpy(float),
        )
    )
    if not breach_serialised_exact or not left[
        "fitted_model_sha256_all_mature"
    ].eq(left["fitted_model_sha256_protected"]).all():
        raise AssertionError("all-mature DP0 did not reproduce the frozen direct baseline")
    right = all_severity.loc[all_severity["model_id"].eq("DQ0")].merge(
        protected_severity.loc[protected_severity["model_id"].eq("DQ0")][
            severity_keys + [
                "actual_positive_late_days", "prediction", "fitted_model_sha256"
            ]
        ],
        on=severity_keys,
        how="outer",
        indicator=True,
        suffixes=("_all_mature", "_protected"),
        validate="one_to_one",
    )
    if not right["_merge"].eq("both").all() or not right[
        "actual_positive_late_days_all_mature"
    ].eq(right["actual_positive_late_days_protected"]).all():
        raise AssertionError("DQ0 paired prediction population/target mismatch")
    severity_diff = float(
        np.max(np.abs(right["prediction_all_mature"] - right["prediction_protected"]))
    )
    severity_serialised_exact = bool(
        np.array_equal(
            serialised(right["prediction_all_mature"]),
            right["prediction_protected"].to_numpy(float),
        )
    )
    if not severity_serialised_exact or not right["fitted_model_sha256_all_mature"].eq(
        right["fitted_model_sha256_protected"]
    ).all():
        raise AssertionError("all-mature DQ0 did not reproduce the frozen direct baseline")
    return {
        "direct_breach_predictions_sha256": _sha256(
            DIRECT / "working/DIRECT_BREACH_PREDICTIONS.csv.gz"
        ),
        "direct_severity_predictions_sha256": _sha256(
            DIRECT / "working/DIRECT_SEVERITY_PREDICTIONS.csv.gz"
        ),
        "dp0_rows": len(left),
        "dp0_max_raw_probability_difference": breach_raw_diff,
        "dp0_max_calibrated_probability_difference": breach_cal_diff,
        "dp0_serialised_predictions_exact": breach_serialised_exact,
        "dp0_model_hashes_exact": True,
        "dq0_rows": len(right),
        "dq0_max_prediction_difference": severity_diff,
        "dq0_serialised_predictions_exact": severity_serialised_exact,
        "dq0_model_hashes_exact": True,
    }


def _classification_values(frame: pd.DataFrame) -> dict[str, float]:
    metrics, _ = order_modeling.classification_metrics(
        frame["order_id"], frame["target_candidate"], frame["prediction_candidate"]
    )
    metrics["absolute_calibration_intercept_error"] = abs(
        float(metrics["calibration_intercept"])
    )
    metrics["absolute_calibration_slope_error"] = abs(
        float(metrics["calibration_slope"]) - 1.0
    )
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.number))}


def _classification_pair_values(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    candidate, _ = order_modeling.classification_metrics(
        frame["order_id"], frame["target_candidate"], frame["prediction_candidate"]
    )
    reference, _ = order_modeling.classification_metrics(
        frame["order_id"], frame["target_reference"], frame["prediction_reference"]
    )
    for values in (candidate, reference):
        values["absolute_calibration_intercept_error"] = abs(
            float(values["calibration_intercept"])
        )
        values["absolute_calibration_slope_error"] = abs(
            float(values["calibration_slope"]) - 1.0
        )
    return candidate, reference


def _breach_comparison_rows(
    all_predictions: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    *,
    period: str,
    include_high_support: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    pair_audit: list[dict[str, object]] = []
    cohorts: list[str]
    if period == "later":
        cohorts = sorted(all_predictions.loc[all_predictions["period"].eq(period), "cohort"].unique())
    elif period == "pooled_source":
        cohorts = ["later_pooled"]
    else:
        cohorts = ["2018-07_to_2018-08"]
    for cohort in cohorts:
        cohort_mask = (
            all_predictions["period"].eq(period)
            if period == "pooled_source"
            else all_predictions["period"].eq(period) & all_predictions["cohort"].eq(cohort)
        )
        for family in sorted(all_predictions["family"].astype(str).unique()):
            for model_id, blocks in BREACH_CANDIDATES.items():
                candidate = all_predictions.loc[
                    cohort_mask
                    & all_predictions["family"].eq(family)
                    & all_predictions["model_id"].eq(model_id)
                ].copy()
                for reference_kind, source, reference_model in (
                    ("promise_only", all_predictions, "DP0"),
                    ("selected_90d", selected_predictions, model_id),
                ):
                    source_mask = (
                        source["period"].eq(period)
                        if period == "pooled_source"
                        else source["period"].eq(period) & source["cohort"].eq(cohort)
                    )
                    reference = source.loc[
                        source_mask
                        & source["family"].eq(family)
                        & source["model_id"].eq(reference_model)
                    ].copy()
                    keep_reference = [
                        "order_id", "purchase_date", "origin", "target", "calibrated_probability",
                        "fitted_model_sha256",
                    ] + [f"{block}_{suffix}" for block in blocks for suffix in ("support", "cold_start")]
                    paired = candidate.merge(
                        reference[keep_reference],
                        on="order_id",
                        how="outer",
                        indicator=True,
                        suffixes=("_candidate", "_reference"),
                        validate="one_to_one",
                    )
                    exact = (
                        len(paired) == len(candidate) == len(reference)
                        and paired["_merge"].eq("both").all()
                        and paired["target_candidate"].eq(paired["target_reference"]).all()
                        and pd.to_datetime(paired["purchase_date_candidate"]).eq(
                            pd.to_datetime(paired["purchase_date_reference"])
                        ).all()
                        and pd.to_datetime(paired["origin_candidate"]).eq(
                            pd.to_datetime(paired["origin_reference"])
                        ).all()
                    )
                    pair_audit.append(
                        {
                            "task": "breach", "period": period, "cohort": cohort,
                            "family": family, "model_id": model_id,
                            "reference_kind": reference_kind,
                            "candidate_rows": len(candidate), "reference_rows": len(reference),
                            "paired_rows": int(paired["_merge"].eq("both").sum()),
                            "unmatched_rows": int(paired["_merge"].ne("both").sum()),
                            "target_mismatches": int(
                                (~paired["target_candidate"].eq(paired["target_reference"])).sum()
                            ),
                            "origin_mismatches": int(
                                (~pd.to_datetime(paired["origin_candidate"]).eq(
                                    pd.to_datetime(paired["origin_reference"])
                                )).sum()
                            ),
                            "exact_pair": exact,
                        }
                    )
                    if not exact:
                        raise AssertionError(
                            f"breach pair mismatch: {period}/{cohort}/{family}/{model_id}/{reference_kind}"
                        )
                    paired = paired.loc[paired["_merge"].eq("both")].copy()
                    paired["prediction_candidate"] = paired["calibrated_probability_candidate"]
                    paired["prediction_reference"] = paired["calibrated_probability_reference"]
                    all_support = paired[[f"{block}_support_candidate" for block in blocks]].apply(
                        pd.to_numeric, errors="coerce"
                    ).min(axis=1)
                    all_cold = paired[[f"{block}_cold_start_candidate" for block in blocks]].fillna(
                        False
                    ).astype(bool).any(axis=1)
                    selected_support = paired[
                        [f"{block}_support_reference" for block in blocks]
                    ].apply(pd.to_numeric, errors="coerce").min(axis=1)
                    selected_cold = paired[
                        [f"{block}_cold_start_reference" for block in blocks]
                    ].fillna(False).astype(bool).any(axis=1)
                    paired["all_mature_minimum_support"] = all_support
                    paired["all_mature_any_cold_start"] = all_cold
                    paired["selected_90d_minimum_support"] = selected_support
                    paired["selected_90d_any_cold_start"] = selected_cold
                    paired["support_stratum"] = _support_stratum(all_support, all_cold)
                    scopes: list[tuple[str, pd.Series]] = [("all_orders", pd.Series(True, index=paired.index))]
                    if include_high_support:
                        high = all_support.ge(20) & ~all_cold
                        if reference_kind == "selected_90d":
                            high &= selected_support.ge(20) & ~selected_cold
                            scope_name = "common_support_ge20"
                        else:
                            scope_name = "all_mature_support_ge20"
                        scopes.append((scope_name, high))
                    for scope, mask in scopes:
                        sample = paired.loc[mask].copy()
                        if sample.empty:
                            continue
                        cand, ref = _classification_pair_values(sample)
                        record: dict[str, object] = {
                            "task": "breach", "period": period,
                            "row_type": "monthly" if period == "later" else ("pooled" if period == "pooled_source" else "terminal"),
                            "cohort": cohort, "family": family, "model_id": model_id,
                            "profile_block": PROFILE_BLOCK_NAMES[model_id],
                            "history_mode": "all_mature", "reference_kind": reference_kind,
                            "reference_model": reference_model, "population": scope,
                            "n_orders": len(sample), "n_events": int(sample["target_candidate"].sum()),
                            "order_id_sha256": order_modeling.order_id_hash(sample["order_id"]),
                            "paired_exact": True,
                        }
                        metrics = (
                            "log_loss", "brier", "average_precision", "roc_auc", "top_5pct_lift",
                            "top_10pct_lift", "calibration_intercept", "calibration_slope", "wace",
                            "absolute_calibration_intercept_error", "absolute_calibration_slope_error",
                        )
                        for metric in metrics:
                            record[f"all_mature_{metric}"] = cand.get(metric, np.nan)
                            record[f"reference_{metric}"] = ref.get(metric, np.nan)
                            record[f"delta_{metric}"] = cand.get(metric, np.nan) - ref.get(metric, np.nan)
                        record["favourable_primary"] = bool(
                            record["delta_log_loss"] < 0 and record["delta_brier"] < 0
                        )
                        rows.append(record)
                    for stratum, sample in paired.groupby("support_stratum", sort=True, observed=True):
                        cand, ref = _classification_pair_values(sample)
                        support_rows.append(
                            {
                                "task": "breach", "period": period, "cohort": cohort,
                                "family": family, "model_id": model_id,
                                "reference_kind": reference_kind,
                                "support_stratum_definition": "all_mature_minimum_applicable_support",
                                "support_stratum": stratum, "n_orders": len(sample),
                                "median_all_mature_support": float(sample["all_mature_minimum_support"].median()),
                                "selected_90d_cold_start_share": float(sample["selected_90d_any_cold_start"].mean()),
                                "all_mature_cold_start_share": float(sample["all_mature_any_cold_start"].mean()),
                                "delta_log_loss": cand["log_loss"] - ref["log_loss"],
                                "delta_brier": cand["brier"] - ref["brier"],
                                "order_id_sha256": order_modeling.order_id_hash(sample["order_id"]),
                            }
                        )
    frame = pd.DataFrame(rows)
    if period == "later":
        all_order_monthly = frame.loc[frame["population"].eq("all_orders")]
        medians: list[dict[str, object]] = []
        group = ["family", "model_id", "profile_block", "reference_kind", "reference_model", "population"]
        for keys, part in frame.groupby(group, sort=True, observed=True):
            row = {column: value for column, value in zip(group, keys)}
            row.update({"task": "breach", "period": "later", "row_type": "monthly_median", "cohort": "2018-01_to_2018-06", "history_mode": "all_mature", "n_months": len(part), "favourable_month_count": int(part["favourable_primary"].sum()), "paired_exact": True})
            for column in frame.columns:
                if (
                    column not in {"reference_kind", "reference_model"}
                    and column.startswith(("all_mature_", "reference_", "delta_"))
                ):
                    row[column] = pd.to_numeric(part[column], errors="coerce").median()
            for metric in ("log_loss", "brier", "wace", "absolute_calibration_intercept_error", "absolute_calibration_slope_error"):
                row[f"favourable_month_count_{metric}"] = int(
                    pd.to_numeric(part[f"delta_{metric}"], errors="coerce").lt(0).sum()
                )
            for metric in ("average_precision", "roc_auc", "top_5pct_lift", "top_10pct_lift"):
                row[f"favourable_month_count_{metric}"] = int(
                    pd.to_numeric(part[f"delta_{metric}"], errors="coerce").gt(0).sum()
                )
            row["difference_aggregation"] = "median_of_paired_monthly_differences"
            medians.append(row)
        frame = pd.concat([frame, pd.DataFrame(medians)], ignore_index=True, sort=False)
        # Pooled values are recomputed from concatenated monthly predictions, never from monthly metrics.
        pooled, pooled_support, pooled_audit = _breach_comparison_rows(
            all_predictions.assign(period=np.where(all_predictions["period"].eq("later"), "pooled_source", all_predictions["period"])),
            selected_predictions.assign(period=np.where(selected_predictions["period"].eq("later"), "pooled_source", selected_predictions["period"])),
            period="pooled_source",
            include_high_support=include_high_support,
        )
        if not pooled.empty:
            pooled["period"] = "aggregate"
            pooled["row_type"] = "pooled"
            pooled["cohort"] = "later_pooled"
            frame = pd.concat([frame, pooled], ignore_index=True, sort=False)
            support_rows.extend(pooled_support.to_dict("records"))
            pair_audit.extend(pooled_audit.to_dict("records"))
        frame.loc[frame["row_type"].eq("monthly"), "difference_aggregation"] = (
            "paired_same_month_identical_orders"
        )
        frame.loc[frame["row_type"].eq("pooled"), "difference_aggregation"] = (
            "recomputed_on_concatenated_monthly_predictions"
        )
    return frame.reset_index(drop=True), pd.DataFrame(support_rows), pd.DataFrame(pair_audit)


def _severity_pair_metrics(frame: pd.DataFrame, quantile: float) -> tuple[dict[str, float], dict[str, float]]:
    candidate = order_modeling.quantile_metrics(
        frame["target_candidate"], frame["prediction_candidate"], quantile
    )
    reference = order_modeling.quantile_metrics(
        frame["target_reference"], frame["prediction_reference"], quantile
    )
    candidate["absolute_coverage_error"] = abs(float(candidate["coverage_error"]))
    reference["absolute_coverage_error"] = abs(float(reference["coverage_error"]))
    reference_loss = float(reference["pinball_loss"])
    candidate["pinball_skill_vs_reference"] = (
        1.0 - float(candidate["pinball_loss"]) / reference_loss
        if reference_loss > 0 else np.nan
    )
    reference["pinball_skill_vs_reference"] = 0.0
    return candidate, reference


def _severity_comparison_rows(
    all_predictions: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    *,
    period: str,
    include_high_support: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    pair_audit: list[dict[str, object]] = []
    if period == "later":
        cohorts = sorted(all_predictions.loc[all_predictions["period"].eq(period), "cohort"].unique())
    elif period == "pooled_source":
        cohorts = ["later_pooled"]
    else:
        cohorts = ["2018-07_to_2018-08"]
    for cohort in cohorts:
        cohort_mask = (
            all_predictions["period"].eq(period)
            if period == "pooled_source"
            else all_predictions["period"].eq(period) & all_predictions["cohort"].eq(cohort)
        )
        for family in sorted(all_predictions["family"].astype(str).unique()):
            for quantile in (0.5, 0.9):
                for model_id, blocks in SEVERITY_CANDIDATES.items():
                    candidate = all_predictions.loc[
                        cohort_mask
                        & all_predictions["family"].eq(family)
                        & pd.to_numeric(all_predictions["quantile"]).eq(quantile)
                        & all_predictions["model_id"].eq(model_id)
                    ].copy()
                    for reference_kind, source, reference_model in (
                        ("promise_only", all_predictions, "DQ0"),
                        ("selected_90d", selected_predictions, model_id),
                    ):
                        source_mask = (
                            source["period"].eq(period)
                            if period == "pooled_source"
                            else source["period"].eq(period) & source["cohort"].eq(cohort)
                        )
                        reference = source.loc[
                            source_mask
                            & source["family"].eq(family)
                            & pd.to_numeric(source["quantile"]).eq(quantile)
                            & source["model_id"].eq(reference_model)
                        ].copy()
                        keep_reference = [
                            "order_id", "purchase_date", "origin", "actual_positive_late_days", "prediction",
                            "fitted_model_sha256",
                        ] + [f"{block}_{suffix}" for block in blocks for suffix in ("support", "cold_start")]
                        paired = candidate.merge(
                            reference[keep_reference], on="order_id", how="outer", indicator=True,
                            suffixes=("_candidate", "_reference"), validate="one_to_one",
                        )
                        exact = (
                            len(paired) == len(candidate) == len(reference)
                            and paired["_merge"].eq("both").all()
                            and paired["actual_positive_late_days_candidate"].eq(
                                paired["actual_positive_late_days_reference"]
                            ).all()
                            and pd.to_datetime(paired["purchase_date_candidate"]).eq(
                                pd.to_datetime(paired["purchase_date_reference"])
                            ).all()
                            and pd.to_datetime(paired["origin_candidate"]).eq(
                                pd.to_datetime(paired["origin_reference"])
                            ).all()
                        )
                        pair_audit.append(
                            {
                                "task": "severity", "period": period, "cohort": cohort,
                                "family": family, "quantile": quantile, "model_id": model_id,
                                "reference_kind": reference_kind, "candidate_rows": len(candidate),
                                "reference_rows": len(reference),
                                "paired_rows": int(paired["_merge"].eq("both").sum()),
                                "unmatched_rows": int(paired["_merge"].ne("both").sum()),
                                "target_mismatches": int((~paired["actual_positive_late_days_candidate"].eq(paired["actual_positive_late_days_reference"])).sum()),
                                "origin_mismatches": int(
                                    (~pd.to_datetime(paired["origin_candidate"]).eq(
                                        pd.to_datetime(paired["origin_reference"])
                                    )).sum()
                                ),
                                "exact_pair": exact,
                            }
                        )
                        if not exact:
                            raise AssertionError(
                                f"severity pair mismatch: {period}/{cohort}/{family}/{quantile}/{model_id}/{reference_kind}"
                            )
                        paired = paired.loc[paired["_merge"].eq("both")].copy()
                        paired["target_candidate"] = paired["actual_positive_late_days_candidate"]
                        paired["target_reference"] = paired["actual_positive_late_days_reference"]
                        all_support = paired[[f"{block}_support_candidate" for block in blocks]].apply(
                            pd.to_numeric, errors="coerce"
                        ).min(axis=1)
                        all_cold = paired[[f"{block}_cold_start_candidate" for block in blocks]].fillna(False).astype(bool).any(axis=1)
                        selected_support = paired[[f"{block}_support_reference" for block in blocks]].apply(
                            pd.to_numeric, errors="coerce"
                        ).min(axis=1)
                        selected_cold = paired[[f"{block}_cold_start_reference" for block in blocks]].fillna(False).astype(bool).any(axis=1)
                        paired["all_mature_minimum_support"] = all_support
                        paired["all_mature_any_cold_start"] = all_cold
                        paired["selected_90d_minimum_support"] = selected_support
                        paired["selected_90d_any_cold_start"] = selected_cold
                        paired["support_stratum"] = _support_stratum(all_support, all_cold)
                        scopes: list[tuple[str, pd.Series]] = [("all_orders", pd.Series(True, index=paired.index))]
                        if include_high_support:
                            high = all_support.ge(20) & ~all_cold
                            if reference_kind == "selected_90d":
                                high &= selected_support.ge(20) & ~selected_cold
                                scope_name = "common_support_ge20"
                            else:
                                scope_name = "all_mature_support_ge20"
                            scopes.append((scope_name, high))
                        for population, mask in scopes:
                            sample = paired.loc[mask].copy()
                            if sample.empty:
                                continue
                            cand, ref = _severity_pair_metrics(sample, quantile)
                            record: dict[str, object] = {
                                "task": "severity", "period": period,
                                "row_type": "monthly" if period == "later" else ("pooled" if period == "pooled_source" else "terminal"),
                                "cohort": cohort, "family": family, "quantile": quantile,
                                "model_id": model_id, "profile_block": PROFILE_BLOCK_NAMES[model_id],
                                "history_mode": "all_mature", "reference_kind": reference_kind,
                                "reference_model": reference_model, "population": population,
                                "n_orders": len(sample),
                                "order_id_sha256": order_modeling.order_id_hash(sample["order_id"]),
                                "paired_exact": True,
                            }
                            for metric in (
                                "pinball_loss", "pinball_skill_vs_reference", "empirical_coverage",
                                "coverage_error", "absolute_coverage_error", "p90_absolute_error",
                            ):
                                record[f"all_mature_{metric}"] = cand.get(metric, np.nan)
                                record[f"reference_{metric}"] = ref.get(metric, np.nan)
                                record[f"delta_{metric}"] = cand.get(metric, np.nan) - ref.get(metric, np.nan)
                            record["favourable_primary"] = bool(cand["pinball_skill_vs_reference"] >= 0)
                            rows.append(record)
                        for stratum, sample in paired.groupby("support_stratum", sort=True, observed=True):
                            cand, ref = _severity_pair_metrics(sample, quantile)
                            support_rows.append(
                                {
                                    "task": "severity", "period": period, "cohort": cohort,
                                    "family": family, "quantile": quantile, "model_id": model_id,
                                    "reference_kind": reference_kind,
                                    "support_stratum_definition": "all_mature_minimum_applicable_support",
                                    "support_stratum": stratum, "n_orders": len(sample),
                                    "median_all_mature_support": float(sample["all_mature_minimum_support"].median()),
                                    "selected_90d_cold_start_share": float(sample["selected_90d_any_cold_start"].mean()),
                                    "all_mature_cold_start_share": float(sample["all_mature_any_cold_start"].mean()),
                                    "delta_pinball_loss": cand["pinball_loss"] - ref["pinball_loss"],
                                    "pinball_skill_vs_reference": cand["pinball_skill_vs_reference"],
                                    "empirical_coverage": cand["empirical_coverage"],
                                    "absolute_coverage_error": cand["absolute_coverage_error"],
                                    "order_id_sha256": order_modeling.order_id_hash(sample["order_id"]),
                                }
                            )
    frame = pd.DataFrame(rows)
    if period == "later":
        medians: list[dict[str, object]] = []
        group = ["family", "quantile", "model_id", "profile_block", "reference_kind", "reference_model", "population"]
        for keys, part in frame.groupby(group, sort=True, observed=True):
            row = {column: value for column, value in zip(group, keys)}
            row.update({"task": "severity", "period": "later", "row_type": "monthly_median", "cohort": "2018-01_to_2018-06", "history_mode": "all_mature", "n_months": len(part), "favourable_month_count": int(part["favourable_primary"].sum()), "paired_exact": True})
            for column in frame.columns:
                if (
                    column not in {"reference_kind", "reference_model"}
                    and column.startswith(("all_mature_", "reference_", "delta_"))
                ):
                    row[column] = pd.to_numeric(part[column], errors="coerce").median()
            row["favourable_month_count_pinball_loss"] = int(
                pd.to_numeric(part["delta_pinball_loss"], errors="coerce").lt(0).sum()
            )
            row["favourable_month_count_absolute_coverage_error"] = int(
                pd.to_numeric(part["delta_absolute_coverage_error"], errors="coerce").lt(0).sum()
            )
            row["difference_aggregation"] = "median_of_paired_monthly_differences"
            medians.append(row)
        frame = pd.concat([frame, pd.DataFrame(medians)], ignore_index=True, sort=False)
        all_pool = all_predictions.loc[all_predictions["period"].eq("later")].copy()
        selected_pool = selected_predictions.loc[selected_predictions["period"].eq("later")].copy()
        all_pool["period"] = "pooled_source"
        selected_pool["period"] = "pooled_source"
        pooled, pooled_support, pooled_audit = _severity_comparison_rows(
            all_pool, selected_pool, period="pooled_source", include_high_support=include_high_support
        )
        pooled["period"] = "aggregate"
        pooled["row_type"] = "pooled"
        pooled["cohort"] = "later_pooled"
        frame = pd.concat([frame, pooled], ignore_index=True, sort=False)
        support_rows.extend(pooled_support.to_dict("records"))
        pair_audit.extend(pooled_audit.to_dict("records"))
        frame.loc[frame["row_type"].eq("monthly"), "difference_aggregation"] = (
            "paired_same_month_identical_orders"
        )
        frame.loc[frame["row_type"].eq("pooled"), "difference_aggregation"] = (
            "recomputed_on_concatenated_monthly_predictions"
        )
    return frame.reset_index(drop=True), pd.DataFrame(support_rows), pd.DataFrame(pair_audit)


def _calibration_coverage_table(
    breach: pd.DataFrame,
    severity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    breach_metrics = (
        "calibration_intercept", "absolute_calibration_intercept_error",
        "calibration_slope", "absolute_calibration_slope_error", "wace",
    )
    for _, source in breach.iterrows():
        for metric in breach_metrics:
            rows.append(
                {
                    "task": "breach", "period": source["period"], "row_type": source["row_type"],
                    "cohort": source["cohort"], "family": source["family"],
                    "model_id": source["model_id"], "quantile": np.nan,
                    "reference_kind": source["reference_kind"], "population": source["population"],
                    "metric": metric, "all_mature_value": source.get(f"all_mature_{metric}"),
                    "reference_value": source.get(f"reference_{metric}"),
                    "all_mature_minus_reference": source.get(f"delta_{metric}"),
                    "n_orders": source.get("n_orders"), "order_id_sha256": source.get("order_id_sha256"),
                    "difference_aggregation": source.get("difference_aggregation"),
                }
            )
    q90 = severity.loc[pd.to_numeric(severity["quantile"], errors="coerce").eq(0.9)]
    for _, source in q90.iterrows():
        for metric in ("empirical_coverage", "absolute_coverage_error", "pinball_loss"):
            rows.append(
                {
                    "task": "conditional_positive_lateness", "period": source["period"],
                    "row_type": source["row_type"], "cohort": source["cohort"],
                    "family": source["family"], "model_id": source["model_id"],
                    "quantile": 0.9, "reference_kind": source["reference_kind"],
                    "population": source["population"], "metric": metric,
                    "all_mature_value": source.get(f"all_mature_{metric}"),
                    "reference_value": source.get(f"reference_{metric}"),
                    "all_mature_minus_reference": source.get(f"delta_{metric}"),
                    "n_orders": source.get("n_orders"), "order_id_sha256": source.get("order_id_sha256"),
                    "difference_aggregation": source.get("difference_aggregation"),
                }
            )
    return pd.DataFrame(rows)


def _terminal_long(breach: pd.DataFrame, severity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, source in breach.iterrows():
        for metric in ("log_loss", "brier", "wace"):
            rows.append(
                {
                    "task": "breach", "period": "terminal", "cohort": source["cohort"],
                    "family": source["family"], "model_id": source["model_id"],
                    "quantile": np.nan, "reference_kind": source["reference_kind"],
                    "population": source["population"], "metric": metric,
                    "all_mature_value": source.get(f"all_mature_{metric}"),
                    "reference_value": source.get(f"reference_{metric}"),
                    "all_mature_minus_reference": source.get(f"delta_{metric}"),
                    "n_orders": source.get("n_orders"), "n_events": source.get("n_events"),
                    "order_id_sha256": source.get("order_id_sha256"),
                    "difference_aggregation": "paired_same_terminal_orders",
                }
            )
    for _, source in severity.iterrows():
        metrics = ["pinball_loss", "pinball_skill_vs_reference"]
        if float(source["quantile"]) == 0.9:
            metrics.extend(["empirical_coverage", "absolute_coverage_error"])
        for metric in metrics:
            rows.append(
                {
                    "task": "conditional_positive_lateness", "period": "terminal",
                    "cohort": source["cohort"], "family": source["family"],
                    "model_id": source["model_id"], "quantile": source["quantile"],
                    "reference_kind": source["reference_kind"], "population": source["population"],
                    "metric": metric, "all_mature_value": source.get(f"all_mature_{metric}"),
                    "reference_value": source.get(f"reference_{metric}"),
                    "all_mature_minus_reference": source.get(f"delta_{metric}"),
                    "n_orders": source.get("n_orders"), "n_events": np.nan,
                    "order_id_sha256": source.get("order_id_sha256"),
                    "difference_aggregation": "paired_same_terminal_orders",
                }
            )
    return pd.DataFrame(rows)


def _recompute_monthly_metrics_from_persisted_predictions(
    breach_template: pd.DataFrame,
    breach_predictions: pd.DataFrame,
    severity_template: pd.DataFrame,
    severity_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    breach = breach_template.copy()
    for index, row in breach.iterrows():
        group = breach_predictions.loc[
            breach_predictions["period"].eq(row["period"])
            & breach_predictions["cohort"].eq(row["cohort"])
            & breach_predictions["family"].eq(row["family"])
            & breach_predictions["model_id"].eq(row["model_id"])
        ]
        column = "raw_probability" if row["probability_type"] == "raw" else "calibrated_probability"
        metrics, _ = order_modeling.classification_metrics(
            group["order_id"], group["target"], group[column]
        )
        for name, value in metrics.items():
            if name in breach.columns:
                breach.at[index, name] = value
        if "brier_score" in breach:
            breach.at[index, "brier_score"] = metrics["brier"]
        if "top10_lift" in breach:
            breach.at[index, "top10_lift"] = metrics["top_10pct_lift"]
    severity = severity_template.copy()
    for index, row in severity.iterrows():
        group = severity_predictions.loc[
            severity_predictions["period"].eq(row["period"])
            & severity_predictions["cohort"].eq(row["cohort"])
            & severity_predictions["family"].eq(row["family"])
            & severity_predictions["model_id"].eq(row["model_id"])
            & pd.to_numeric(severity_predictions["quantile"], errors="coerce").eq(
                float(row["quantile"])
            )
        ]
        metrics = order_modeling.quantile_metrics(
            group["actual_positive_late_days"], group["prediction"], float(row["quantile"])
        )
        baseline_loss = order_modeling.pinball_loss(
            group["actual_positive_late_days"], group["dq0_prediction"], float(row["quantile"])
        )
        skill = 1.0 - float(metrics["pinball_loss"]) / baseline_loss if baseline_loss > 0 else np.nan
        for name, value in metrics.items():
            if name in severity.columns:
                severity.at[index, name] = value
        for name in ("baseline_pinball_loss", "dq0_reference_loss"):
            if name in severity:
                severity.at[index, name] = baseline_loss
        for name in ("skill", "skill_vs_dq0"):
            if name in severity:
                severity.at[index, name] = skill
        if "coverage" in severity:
            severity.at[index, "coverage"] = metrics["empirical_coverage"]
    return breach, severity


def _population_audit(
    breach_predictions: pd.DataFrame,
    severity_predictions: pd.DataFrame,
) -> dict[str, object]:
    expected_breach = {
        "2018-01": 7069, "2018-02": 6555, "2018-03": 7003,
        "2018-04": 6798, "2018-05": 6749, "2018-06": 6096,
    }
    expected_severity = {
        "2018-01": 403, "2018-02": 926, "2018-03": 1328,
        "2018-04": 306, "2018-05": 443, "2018-06": 71,
    }
    breach_base = breach_predictions.loc[
        breach_predictions["period"].eq("later")
        & breach_predictions["family"].eq("logistic_l2")
        & breach_predictions["model_id"].eq("DP0")
    ]
    severity_base = severity_predictions.loc[
        severity_predictions["period"].eq("later")
        & severity_predictions["family"].eq("linear_quantile")
        & severity_predictions["model_id"].eq("DQ0")
        & pd.to_numeric(severity_predictions["quantile"], errors="coerce").eq(0.5)
    ]
    breach_counts = breach_base.groupby("cohort", sort=True)["order_id"].nunique().to_dict()
    severity_counts = severity_base.groupby("cohort", sort=True)["order_id"].nunique().to_dict()
    breach_hash = order_modeling.order_id_hash(breach_base["order_id"])
    severity_hash = order_modeling.order_id_hash(severity_base["order_id"])
    terminal_breach = breach_predictions.loc[
        breach_predictions["period"].eq("terminal")
        & breach_predictions["family"].eq("logistic_l2")
        & breach_predictions["model_id"].eq("DP0")
    ]
    terminal_severity = severity_predictions.loc[
        severity_predictions["period"].eq("terminal")
        & severity_predictions["family"].eq("linear_quantile")
        & severity_predictions["model_id"].eq("DQ0")
        & pd.to_numeric(severity_predictions["quantile"], errors="coerce").eq(0.5)
    ]
    receipt = {
        "later_breach_counts": breach_counts,
        "later_breach_expected_counts": expected_breach,
        "later_breach_pooled_rows": len(breach_base),
        "later_breach_pooled_order_id_sha256": breach_hash,
        "later_severity_counts": severity_counts,
        "later_severity_expected_counts": expected_severity,
        "later_severity_pooled_rows": len(severity_base),
        "later_severity_pooled_order_id_sha256": severity_hash,
        "terminal_breach_rows": len(terminal_breach),
        "terminal_severity_rows": len(terminal_severity),
    }
    receipt["passed"] = bool(
        breach_counts == expected_breach
        and severity_counts == expected_severity
        and len(breach_base) == 40_270
        and len(severity_base) == 3_477
        and breach_hash == "72cfdc8f6be208328a97da86166024c4a11b6099541993e6da4d9e100c5d76d2"
        and severity_hash == "118782cfc610a9028878d750598e5fe92f9ee67f521716c506bc37135f89b805"
        and len(terminal_breach) == 12_507
        and len(terminal_severity) == 601
    )
    if not receipt["passed"]:
        raise AssertionError(f"direct cohort population audit failed: {receipt}")
    return receipt


def run_order_sensitivity() -> dict[str, object]:
    config, selection, base, source_receipt = _load_direct_inputs()
    frame, join_audit, frame_receipt = construct_all_mature_frame(base)
    model_frame_path = WORK / "DIRECT_ALL_MATURE_MODEL_FRAME.parquet"
    frame.to_parquet(model_frame_path, index=False, compression="zstd")
    reloaded_frame = pd.read_parquet(model_frame_path)
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True),
        reloaded_frame.reset_index(drop=True),
        check_dtype=False,
        check_categorical=False,
        check_exact=True,
    )
    model_frame_sha256 = _sha256(model_frame_path)
    exposure_columns = [
        "order_id", "purchase_date",
        *[f"{code}_{suffix}" for code in PROFILE_BLOCKS for suffix in SWAP_SUFFIXES],
    ]
    sc.write_gzip_csv(
        frame[exposure_columns],
        WORK / "DIRECT_ALL_MATURE_PROFILE_EXPOSURES.csv.gz",
        ["purchase_date", "order_id"],
    )
    exposure_sha256 = _sha256(WORK / "DIRECT_ALL_MATURE_PROFILE_EXPOSURES.csv.gz")
    sc.write_csv(join_audit, WORK / "DIRECT_ALL_MATURE_PROFILE_JOIN_AUDIT.csv", ["profile_code"])
    _write_json(
        WORK / "DIRECT_ALL_MATURE_MODEL_FRAME_RECEIPT.json",
        {
            **source_receipt,
            **frame_receipt,
            "persisted_model_frame_path": model_frame_path.relative_to(ROOT).as_posix(),
            "persisted_model_frame_sha256": model_frame_sha256,
            "persisted_model_frame_roundtrip_exact_ignoring_dtype_container": True,
            "profile_exposure_sha256": exposure_sha256,
        },
    )

    breach_metrics, breach_bins, breach_predictions, breach_manifests = de._evaluate_breach_primary(
        frame, config, selection
    )
    severity_metrics, severity_predictions, severity_manifests = de._evaluate_severity_primary(
        frame, config, selection
    )
    if len(breach_predictions) != 422_216 or len(severity_predictions) != 65_248:
        raise AssertionError(
            f"unexpected prediction counts: breach={len(breach_predictions)}, "
            f"severity={len(severity_predictions)}"
        )
    breach_predictions["history_mode"] = "all_mature"
    severity_predictions["history_mode"] = "all_mature"
    sc.write_gzip_csv(
        breach_predictions, WORK / "DIRECT_ALL_MATURE_BREACH_PREDICTIONS.csv.gz",
        ["period", "cohort", "family", "model_id", "order_id"],
    )
    sc.write_gzip_csv(
        severity_predictions, WORK / "DIRECT_ALL_MATURE_SEVERITY_PREDICTIONS.csv.gz",
        ["period", "cohort", "family", "quantile", "model_id", "order_id"],
    )
    breach_predictions = pd.read_csv(
        WORK / "DIRECT_ALL_MATURE_BREACH_PREDICTIONS.csv.gz", low_memory=False
    )
    severity_predictions = pd.read_csv(
        WORK / "DIRECT_ALL_MATURE_SEVERITY_PREDICTIONS.csv.gz", low_memory=False
    )
    breach_metrics, severity_metrics = _recompute_monthly_metrics_from_persisted_predictions(
        breach_metrics, breach_predictions, severity_metrics, severity_predictions
    )
    population_receipt = _population_audit(breach_predictions, severity_predictions)
    _write_json(WORK / "DIRECT_COHORT_POPULATION_AUDIT.json", population_receipt)
    reproduction = _reproduction_audit(breach_predictions, severity_predictions)
    _write_json(WORK / "DIRECT_BASELINE_REPRODUCTION_AUDIT.json", reproduction)
    manifests = pd.concat([breach_manifests, severity_manifests], ignore_index=True, sort=False)
    manifests["history_mode"] = "all_mature"
    manifests["direct_selection_sha256"] = source_receipt["direct_selection_sha256"]
    manifests["source_model_frame_sha256"] = model_frame_sha256
    manifests["source_model_frame_hash_type"] = "sha256_parquet_all_mature_sensitivity_frame"
    manifests["source_selected_90d_model_frame_sha256"] = source_receipt[
        "source_model_frame_sha256"
    ]
    manifests["source_profile_exposure_sha256"] = exposure_sha256
    protected_manifests = pd.read_csv(
        DIRECT / "DIRECT_MODEL_MANIFESTS.csv", low_memory=False
    )
    protected_primary = protected_manifests.loc[
        protected_manifests["stage"].isin(["later_evaluation", "terminal_stress"])
        & protected_manifests["representation"].eq("full")
    ].copy()
    manifest_keys = [
        "task", "stage", "cohort", "family", "model_id", "quantile", "representation"
    ]
    contract_columns = [
        "n_train", "n_evaluation", "train_order_id_sha256",
        "evaluation_order_id_sha256", "parameters_json", "numeric_features_json",
        "categorical_features_json", "ordered_feature_sha256",
    ]
    manifest_pair = manifests.merge(
        protected_primary[manifest_keys + contract_columns],
        on=manifest_keys,
        how="outer",
        indicator=True,
        suffixes=("_all_mature", "_protected"),
        validate="one_to_one",
    )
    contract_mismatches: dict[str, int] = {}
    for column in contract_columns:
        left = manifest_pair[f"{column}_all_mature"]
        right = manifest_pair[f"{column}_protected"]
        if column in ("n_train", "n_evaluation"):
            equal = pd.to_numeric(left, errors="coerce").eq(
                pd.to_numeric(right, errors="coerce")
            )
        else:
            equal = _string_equal(left, right)
        contract_mismatches[column] = int((~equal).sum())
    cohort_contract_ok = (
        len(manifest_pair) == 168
        and manifest_pair["_merge"].eq("both").all()
        and not any(contract_mismatches.values())
    )
    cohort_receipt = {
        "rows": len(manifest_pair),
        "all_primary_manifests_paired": bool(manifest_pair["_merge"].eq("both").all()),
        "contract_mismatches": contract_mismatches,
        "passed": cohort_contract_ok,
    }
    _write_json(WORK / "DIRECT_MODEL_COHORT_CONTRACT_AUDIT.json", cohort_receipt)
    if not cohort_contract_ok:
        raise AssertionError(f"direct frozen cohort/model contract mismatch: {cohort_receipt}")
    sc.write_csv(
        manifests, WORK / "DIRECT_ALL_MATURE_MODEL_MANIFESTS.csv",
        ["task", "stage", "cohort", "family", "quantile", "model_id"],
    )
    sc.write_csv(breach_bins, WORK / "DIRECT_ALL_MATURE_RELIABILITY_BINS.csv")

    protected_breach = pd.read_csv(
        DIRECT / "working/DIRECT_BREACH_PREDICTIONS.csv.gz", low_memory=False
    )
    protected_severity = pd.read_csv(
        DIRECT / "working/DIRECT_SEVERITY_PREDICTIONS.csv.gz", low_memory=False
    )
    breach_comparison, breach_support, breach_pairing = _breach_comparison_rows(
        breach_predictions, protected_breach, period="later", include_high_support=True
    )
    severity_comparison, severity_support, severity_pairing = _severity_comparison_rows(
        severity_predictions, protected_severity, period="later", include_high_support=True
    )
    breach_terminal, breach_support_terminal, breach_pairing_terminal = _breach_comparison_rows(
        breach_predictions, protected_breach, period="terminal", include_high_support=True
    )
    severity_terminal, severity_support_terminal, severity_pairing_terminal = _severity_comparison_rows(
        severity_predictions, protected_severity, period="terminal", include_high_support=True
    )
    pairing = pd.concat(
        [breach_pairing, severity_pairing, breach_pairing_terminal, severity_pairing_terminal],
        ignore_index=True, sort=False,
    )
    if not pairing["exact_pair"].astype(bool).all() or pd.to_numeric(
        pairing["unmatched_rows"], errors="coerce"
    ).ne(0).any():
        raise AssertionError("direct all-mature pair audit failed")
    sc.write_csv(
        pairing, WORK / "DIRECT_ALL_MATURE_PAIRING_AUDIT.csv",
        ["task", "period", "cohort", "family", "quantile", "model_id", "reference_kind"],
    )
    support = pd.concat(
        [breach_support, severity_support, breach_support_terminal, severity_support_terminal],
        ignore_index=True, sort=False,
    )
    sc.write_csv(
        support, WORK / "DIRECT_ALL_MATURE_SUPPORT_STRATA.csv",
        ["task", "period", "cohort", "family", "quantile", "model_id", "reference_kind", "support_stratum"],
    )

    breach_monthly = breach_metrics.loc[
        breach_metrics["period"].eq("later")
    ].copy()
    breach_monthly["history_mode"] = "all_mature"
    severity_monthly = severity_metrics.loc[severity_metrics["period"].eq("later")].copy()
    severity_monthly["history_mode"] = "all_mature"
    sc.write_csv(
        breach_monthly, OUT / "DIRECT_BREACH_ALL_MATURE_MONTHLY.csv",
        ["cohort", "family", "model_id", "probability_type"],
    )
    sc.write_csv(
        breach_comparison,
        OUT / "DIRECT_BREACH_90D_VS_ALL_MATURE.csv",
        ["row_type", "cohort", "family", "model_id", "reference_kind", "population"],
    )
    sc.write_csv(
        severity_monthly, OUT / "DIRECT_SEVERITY_ALL_MATURE_MONTHLY.csv",
        ["cohort", "family", "quantile", "model_id"],
    )
    sc.write_csv(
        severity_comparison,
        OUT / "DIRECT_SEVERITY_90D_VS_ALL_MATURE.csv",
        [
            "row_type", "cohort", "family", "quantile", "model_id",
            "reference_kind", "population",
        ],
    )
    calibration = _calibration_coverage_table(breach_comparison, severity_comparison)
    sc.write_csv(
        calibration, OUT / "DIRECT_ALL_MATURE_CALIBRATION_COVERAGE.csv",
        ["task", "row_type", "cohort", "family", "quantile", "model_id", "reference_kind", "population", "metric"],
    )
    terminal = _terminal_long(breach_terminal, severity_terminal)
    sc.write_csv(
        terminal, OUT / "DIRECT_ALL_MATURE_TERMINAL.csv",
        ["task", "cohort", "family", "quantile", "model_id", "reference_kind", "population", "metric"],
    )
    return {
        "model_frame_rows": len(frame),
        "model_frame_order_id_sha256": frame_receipt["order_id_sha256"],
        "breach_prediction_rows": len(breach_predictions),
        "severity_prediction_rows": len(severity_predictions),
        "evaluation_model_manifest_rows": len(manifests),
        "breach_monthly_rows": len(breach_monthly),
        "breach_comparison_rows": len(breach_comparison),
        "severity_monthly_rows": len(severity_monthly),
        "severity_comparison_rows": len(severity_comparison),
        "calibration_coverage_rows": len(calibration),
        "terminal_rows": len(terminal),
        "pairing_audit_rows": len(pairing),
        "support_strata_rows": len(support),
        "baseline_reproduction": reproduction,
        "cohort_population_audit": population_receipt,
    }


__all__ = ["construct_all_mature_frame", "run_order_sensitivity"]
