"""Review-record loading and deterministic order-level selection for RQ1.

The Phase 1 loader historically averages multiple review scores.  RQ1 instead
retains one observed integer score per order and persists the multiplicity audit
that justifies the selection.  Review variables remain outcomes/diagnostics and
must never enter the ex-ante M0--M2 feature sets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.olist import resolve_data_dir

REVIEW_COLUMNS = (
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
)


def prepare_review_records(records: pd.DataFrame) -> pd.DataFrame:
    """Validate scores and parse timestamps without collapsing orders."""
    missing = set(REVIEW_COLUMNS) - set(records.columns)
    if missing:
        raise KeyError(f"Missing review columns: {sorted(missing)}")

    result = records.loc[:, REVIEW_COLUMNS].copy()
    result["_source_row"] = np.arange(len(result), dtype=int)
    result["review_creation_date"] = pd.to_datetime(
        result["review_creation_date"], errors="coerce"
    )
    result["review_answer_timestamp"] = pd.to_datetime(
        result["review_answer_timestamp"], errors="coerce"
    )
    numeric_score = pd.to_numeric(result["review_score"], errors="coerce")
    usable = (
        numeric_score.notna()
        & numeric_score.between(1, 5)
        & numeric_score.eq(numeric_score.round())
    )
    result["review_score"] = numeric_score.where(usable).astype("Int64")
    return result


def load_review_records(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load only review fields required for the observational RQ1 analysis."""
    root = resolve_data_dir(data_dir)
    records = pd.read_csv(
        root / "olist_order_reviews_dataset.csv", usecols=list(REVIEW_COLUMNS)
    )
    return prepare_review_records(records)


def select_latest_usable_review(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one valid review per order and return a multiplicity audit.

    Selection is deterministic: latest answer timestamp, then latest review
    creation timestamp, then lexicographically smallest review ID, then source
    row.  Review IDs are not treated as globally unique and never replace the
    order ID as the join key.
    """
    work = prepare_review_records(records) if "_source_row" not in records else records.copy()
    linked = work.loc[work["order_id"].notna()].copy()
    usable = linked.loc[linked["review_score"].notna()].copy()
    usable["_review_id_sort"] = usable["review_id"].astype("string").fillna("")

    order_audit = linked.groupby("order_id", as_index=False).agg(
        raw_review_records=("order_id", "size"),
        usable_review_records=("review_score", "count"),
        distinct_review_ids=("review_id", "nunique"),
        distinct_review_scores=("review_score", "nunique"),
        minimum_review_score=("review_score", "min"),
        maximum_review_score=("review_score", "max"),
    )
    order_audit["conflicting_scores"] = order_audit["distinct_review_scores"].gt(1)

    chosen = usable.sort_values(
        [
            "order_id",
            "review_answer_timestamp",
            "review_creation_date",
            "_review_id_sort",
            "_source_row",
        ],
        ascending=[True, False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).drop_duplicates("order_id", keep="first")
    chosen = chosen.drop(columns="_review_id_sort")
    chosen = chosen.merge(order_audit, on="order_id", how="left", validate="one_to_one")
    chosen = chosen.rename(columns={
        "review_id": "selected_review_id",
        "review_score": "selected_review_score",
        "review_creation_date": "selected_review_creation_date",
        "review_answer_timestamp": "selected_review_answer_timestamp",
    })
    if chosen["order_id"].duplicated().any():
        raise AssertionError("Review selection did not produce one row per order.")

    selected_columns = [
        "order_id",
        "selected_review_id",
        "selected_review_score",
        "selected_review_creation_date",
        "selected_review_answer_timestamp",
        "raw_review_records",
        "usable_review_records",
        "distinct_review_ids",
        "distinct_review_scores",
        "minimum_review_score",
        "maximum_review_score",
        "conflicting_scores",
        "_source_row",
    ]
    return chosen[selected_columns].reset_index(drop=True), order_audit
