from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/dynamic_profile_eda_v1_1"
PROTECTED = {
    "v1": ROOT / "analysis/dynamic_profile_eda_v1",
    "phase2a": ROOT / "analysis/profile_pivot_phase2a",
}
ASSEMBLER = PROTECTED["phase2a"] / "scripts/data_pipeline.py"
ASSEMBLER_SHA = "0c4cad3c99db268292253abd26e2070ccf2a286337bcfe9fb76fb3768c44ab8d"
WINDOWS = (30, 60, 90)
LAGS = (7, 14, 21, 30, 45, 60)
SUPPORTS = (1, 5, 10, 20, 50)
HORIZONS = (7, 30)
AGES = (0, 3, 7, 14, 21, 30, 45, 60, 90)
MATURITY_THRESHOLDS = (0.90, 0.95, 0.975, 0.99)
HRD_DEFS = ("order_top10", "order_top5", "gmv_top10", "gmv_top5", "both_top10", "both_top5")
ENTITIES = {
    "seller_id": "main_seller_id",
    "seller_x_customer_region": "seller_x_customer_region",
    "seller_x_customer_state": "seller_x_customer_state",
    "seller_x_state_od": "seller_x_state_od",
    "region_od": "region_od",
    "state_od": "state_od",
    "zip2_od": "zip2_od",
}
TARGETS = {
    "final_breach": {"value": "late_delivery", "available": "final_breach_available_at", "kind": "binary"},
    "positive_late_days": {"value": "positive_late_days", "available": "positive_late_days_available_at", "kind": "severity"},
    "handling": {"value": "handling_duration", "available": "handling_available_at", "kind": "process"},
    "transit": {"value": "transit_duration", "available": "transit_available_at", "kind": "process"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recursive_hashes(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def quantile(s: pd.Series, p: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.quantile(p)) if len(x) else np.nan


def deterministic_mode(s: pd.Series):
    x = s.dropna().astype(str)
    if x.empty:
        return pd.NA
    c = x.value_counts()
    return sorted(c[c.eq(c.max())].index)[0]


def strict_max(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    return frame[cols].max(axis=1, skipna=False)


def build_all_placed(raw: dict[str, pd.DataFrame], canonical: pd.DataFrame, region_map: dict) -> pd.DataFrame:
    """One row per raw placed order with deterministic entity mapping and raw outcomes."""
    orders = raw["orders"].copy()
    for c in [
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    if orders.order_id.isna().any() or orders.order_id.duplicated().any():
        raise AssertionError("raw orders must be one row per nonmissing order_id")

    items = raw["items"].copy()
    item = items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        n_unique_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        main_seller_id=("seller_id", deterministic_mode),
    )
    item["gmv_observed"] = True
    sellers = raw["sellers"][["seller_id", "seller_state", "seller_zip_code_prefix"]].rename(
        columns={"seller_id": "main_seller_id", "seller_state": "main_seller_state", "seller_zip_code_prefix": "main_seller_zip"}
    )
    item = item.merge(sellers, on="main_seller_id", how="left", validate="m:1")
    customers = raw["customers"][["customer_id", "customer_state", "customer_zip_code_prefix"]]
    frame = orders.merge(customers, on="customer_id", how="left", validate="m:1", indicator="customer_join")
    frame = frame.merge(item, on="order_id", how="left", validate="1:1", indicator="item_join")
    frame["gmv_observed"] = frame.gmv_observed.fillna(False).astype(bool)
    frame["purchase_date"] = frame.order_purchase_timestamp.dt.normalize()
    frame["seller_region"] = frame.main_seller_state.map(region_map)
    frame["customer_region"] = frame.customer_state.map(region_map)
    frame["seller_id"] = frame.main_seller_id.astype("string")
    frame["seller_x_customer_region"] = frame.main_seller_id.astype("string") + " -> " + frame.customer_region.astype("string")
    frame["seller_x_customer_state"] = frame.main_seller_id.astype("string") + " -> " + frame.customer_state.astype("string")
    state_od = frame.main_seller_state.astype("string") + " -> " + frame.customer_state.astype("string")
    frame["state_od"] = state_od
    frame["seller_x_state_od"] = frame.main_seller_id.astype("string") + " -> " + state_od
    frame["region_od"] = frame.seller_region.astype("string") + " -> " + frame.customer_region.astype("string")
    z1 = pd.to_numeric(frame.main_seller_zip, errors="coerce").astype("Int64").astype("string").str.zfill(5).str[:2]
    z2 = pd.to_numeric(frame.customer_zip_code_prefix, errors="coerce").astype("Int64").astype("string").str.zfill(5).str[:2]
    frame["zip2_od"] = z1 + " -> " + z2
    for c in ENTITIES.values():
        frame[c] = frame[c].mask(frame[c].str.contains("<NA>", na=True), pd.NA)

    actual = frame.order_delivered_customer_date
    estimate = frame.order_estimated_delivery_date
    frame["final_breach_available_at"] = actual
    frame["positive_late_days_available_at"] = actual
    err = (actual.dt.normalize() - estimate.dt.normalize()).dt.days
    frame["promise_error_days"] = err
    frame["late_delivery"] = err.gt(0).where(err.notna())
    frame["positive_late_days"] = err.clip(lower=0)
    frame["handling_available_at"] = strict_max(frame, ["order_approved_at", "order_delivered_carrier_date"])
    frame["handling_duration"] = (frame.order_delivered_carrier_date - frame.order_approved_at).dt.total_seconds() / 86400
    frame["transit_available_at"] = strict_max(frame, ["order_delivered_carrier_date", "order_delivered_customer_date"])
    frame["transit_duration"] = (frame.order_delivered_customer_date - frame.order_delivered_carrier_date).dt.total_seconds() / 86400
    frame["purchase_to_carrier"] = (frame.order_delivered_carrier_date - frame.order_purchase_timestamp).dt.total_seconds() / 86400
    frame["is_multi_seller"] = frame.n_unique_sellers.gt(1).fillna(False)
    frame["in_canonical"] = frame.order_id.isin(set(canonical.order_id))
    if len(frame) != 99441 or frame.order_id.nunique() != 99441:
        raise AssertionError(f"all-placed sample mismatch: {len(frame)}")
    return frame


def availability_mask(frame: pd.DataFrame, target: str, endpoint: pd.Timestamp | None = None) -> pd.Series:
    spec = TARGETS[target]
    m = frame[spec["available"]].notna()
    if endpoint is not None:
        m &= frame[spec["available"]].le(endpoint)
    return m


def target_observed_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    return frame[TARGETS[target]["value"]].notna()


def target_valid_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    m = availability_mask(frame, target) & target_observed_mask(frame, target)
    if TARGETS[target]["kind"] == "process":
        m &= frame[TARGETS[target]["value"]].ge(0)
    return m


def reconcile_sample(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame[["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date", "customer_join", "item_join", "in_canonical"]].copy()
    r["status_delivered"] = r.order_status.eq("delivered")
    r["customer_delivery_observed"] = r.order_delivered_customer_date.notna()
    r["estimate_observed"] = r.order_estimated_delivery_date.notna()
    r["purchase_observed"] = r.order_purchase_timestamp.notna()
    reasons = []
    for x in r.itertuples(index=False):
        z = []
        if not x.status_delivered:
            if pd.notna(x.order_delivered_customer_date):
                z.append(f"status_inconsistency:order_status={x.order_status};assembler_requires_delivered")
            else:
                z.append(f"status_not_delivered:{x.order_status}")
        if pd.isna(x.order_delivered_customer_date): z.append("missing_customer_delivery")
        if pd.isna(x.order_estimated_delivery_date): z.append("missing_estimate")
        if pd.isna(x.order_purchase_timestamp): z.append("missing_purchase_timestamp")
        if x.customer_join != "both": z.append("invalid_customer_join")
        if x.item_join != "both": z.append("invalid_item_join")
        reasons.append("included_canonical" if x.in_canonical else ";".join(z) or "other_assembler_exclusion")
    r["deterministic_reconciliation_reason"] = reasons
    return r


def maturity_outputs(frame: pd.DataFrame, audit_endpoint: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves, quants, components = [], [], []
    for target, spec in TARGETS.items():
        avail = frame[spec["available"]]
        age = (avail - frame.order_purchase_timestamp).dt.total_seconds() / 86400
        eventual = avail.notna() & avail.le(audit_endpoint)
        observed = frame[spec["value"]].notna()
        valid = target_valid_mask(frame, target)
        neg = frame[spec["value"]].lt(0) if spec["kind"] == "process" else pd.Series(False, index=frame.index)
        required_components = {
            "final_breach": ["order_delivered_customer_date", "order_estimated_delivery_date"],
            "positive_late_days": ["order_delivered_customer_date", "order_estimated_delivery_date"],
            "handling": ["order_approved_at", "order_delivered_carrier_date"],
            "transit": ["order_delivered_carrier_date", "order_delivered_customer_date"],
        }[target]
        components.append({
            "target": target, "all_placed_orders": len(frame),
            "availability_timestamp_observed": int(avail.notna().sum()),
            "availability_by_audit_endpoint": int(eventual.sum()),
            "target_value_observed": int(observed.sum()),
            "target_value_valid_for_descriptive_summary": int(valid.sum()),
            "negative_duration_count": int(neg.sum()),
            "missing_component_count": int(frame[required_components].isna().any(axis=1).sum()),
            "eventual_observation_plateau": float(eventual.mean()),
        })
        for a in AGES:
            available_by_age = age.le(a) & eventual
            curves.append({
                "target": target, "conditioning": "all_placed_orders", "age_days": a,
                "denominator_orders": len(frame), "available_by_age_orders": int(available_by_age.sum()),
                "unconditional_cumulative_availability": float(available_by_age.mean()),
                "eventual_observation_plateau": float(eventual.mean()),
            })
        valid_age = np.sort(age[eventual].dropna().to_numpy())
        plateau = float(eventual.mean())
        for p in MATURITY_THRESHOLDS:
            threshold_reached = plateau >= p
            all_order_age = np.nan
            if threshold_reached:
                # Earliest observed availability age whose all-order cumulative share reaches p.
                rank = int(np.ceil(p * len(frame))) - 1
                all_order_age = float(valid_age[rank]) if 0 <= rank < len(valid_age) else np.nan
            conditional_age = float(np.quantile(valid_age, p, method="higher")) if len(valid_age) else np.nan
            quants.append({
                "target": target, "conditioning": "all_placed_orders", "threshold": p,
                "threshold_reached": bool(threshold_reached), "age_days_required": all_order_age,
                "eventual_observation_plateau": plateau,
                "reason": "reached" if threshold_reached else "eventual_observation_plateau_below_threshold",
            })
            quants.append({
                "target": target, "conditioning": "conditional_on_eventual_observation", "threshold": p,
                "threshold_reached": True if len(valid_age) else False, "age_days_required": conditional_age,
                "eventual_observation_plateau": plateau,
                "reason": "conditional_quantile" if len(valid_age) else "no_eventually_observed_outcomes",
            })
    return pd.DataFrame(curves), pd.DataFrame(quants), pd.DataFrame(components)


def derive_intervals(frame: pd.DataFrame, audit_endpoint: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    raw_start = frame.order_purchase_timestamp.min().normalize()
    first = raw_start + pd.Timedelta(days=90)
    dense = frame.loc[frame.order_delivered_customer_date.notna()].groupby(frame.order_delivered_customer_date.dt.normalize()).size()
    completion_last = dense[dense.ge(100)].index.max()
    candidates = pd.date_range(first, completion_last, freq="D")
    rows, ends = [], {}
    for target, spec in TARGETS.items():
        vals = []
        for t in candidates:
            c = frame.order_purchase_timestamp.ge(t - pd.Timedelta(days=60)) & frame.order_purchase_timestamp.lt(t)
            denom = int(c.sum())
            mature = c & frame[spec["available"]].notna() & frame[spec["available"]].le(audit_endpoint)
            frac = float(mature.sum() / denom) if denom else np.nan
            vals.append((t, denom, int(mature.sum()), frac))
        eligible = [x for x in vals if x[1] > 0 and x[3] >= .95]
        last = max(x[0] for x in eligible) if eligible else pd.NaT
        ends[target] = last
        rec = next((x for x in vals if x[0] == last), (last, 0, 0, np.nan))
        rows.append({"boundary_scope": "target_purchase_comparison", "target": target, "raw_data_start": raw_start,
                     "first_90d_warmup_snapshot": first, "last_eligible_snapshot": last,
                     "candidate_snapshot_domain_start": first, "candidate_snapshot_domain_end": completion_last,
                     "audit_endpoint_proxy": audit_endpoint, "cohort_total_orders": rec[1],
                     "available_by_audit_endpoint": rec[2], "unconditional_availability": rec[3],
                     "maturity_threshold": .95, "maturity_achieved": bool(pd.notna(rec[3]) and rec[3] >= .95),
                     "reason": "latest within frozen broad snapshot grid whose [t-60d,t) cohort has unconditional availability >=0.95"})
    missing_endpoints=[target for target,value in ends.items() if pd.isna(value)]
    if missing_endpoints:
        raise AssertionError(f"no >=0.95 eligible purchase-comparison endpoint for targets: {missing_endpoints}")
    common = min(ends.values())
    rows.append({"boundary_scope": "common_purchase_comparison", "target": "all_targets", "raw_data_start": raw_start,
                 "first_90d_warmup_snapshot": first, "last_eligible_snapshot": common,
                 "candidate_snapshot_domain_start": first, "candidate_snapshot_domain_end": completion_last,
                 "audit_endpoint_proxy": audit_endpoint, "cohort_total_orders": np.nan,
                 "available_by_audit_endpoint": np.nan, "unconditional_availability": np.nan,
                 "maturity_threshold": .95, "maturity_achieved": bool(pd.notna(common)),
                 "reason": "minimum target-specific eligible endpoint"})
    rows.append({"boundary_scope": "completion_availability", "target": "all_targets", "raw_data_start": raw_start,
                 "first_90d_warmup_snapshot": first, "last_eligible_snapshot": completion_last,
                 "candidate_snapshot_domain_start": first, "candidate_snapshot_domain_end": completion_last,
                 "audit_endpoint_proxy": audit_endpoint, "cohort_total_orders": np.nan,
                 "available_by_audit_endpoint": np.nan, "unconditional_availability": np.nan,
                 "maturity_threshold": np.nan, "maturity_achieved": np.nan,
                 "reason": "last delivery date with at least 100 observed deliveries; endpoint proxy only"})
    return pd.DataFrame(rows), first, completion_last, common


def target_summary(frame: pd.DataFrame, target: str, prefix: str) -> dict:
    spec = TARGETS[target]; value = spec["value"]; kind = spec["kind"]
    x = pd.to_numeric(frame[value], errors="coerce")
    out = {f"{prefix}_target_value_count": int(x.notna().sum())}
    if kind == "binary":
        out.update({f"{prefix}_breach_count": int(x.eq(1).sum()), f"{prefix}_breach_rate": float(x.mean()) if x.notna().any() else np.nan})
    elif kind == "severity":
        observed=x.dropna(); pos = observed[observed.gt(0)]
        out.update({f"{prefix}_zero_severity_share": float(observed.eq(0).mean()) if len(observed) else np.nan,
                    f"{prefix}_mean": float(x.mean()), f"{prefix}_median": quantile(x,.5), f"{prefix}_p75": quantile(x,.75),
                    f"{prefix}_p90": quantile(x,.9), f"{prefix}_p95": quantile(x,.95),
                    f"{prefix}_positive_only_count": int(len(pos)), f"{prefix}_positive_only_mean": float(pos.mean()),
                    f"{prefix}_positive_only_median": quantile(pos,.5), f"{prefix}_positive_only_p90": quantile(pos,.9), f"{prefix}_positive_only_p95": quantile(pos,.95)})
    else:
        nonneg = x[x.ge(0)]
        out.update({f"{prefix}_raw_duration_count": int(x.notna().sum()), f"{prefix}_negative_duration_count": int(x.lt(0).sum()),
                    f"{prefix}_nonnegative_duration_count": int(x.ge(0).sum()),
                    f"{prefix}_raw_mean": float(x.mean()), f"{prefix}_raw_median": quantile(x,.5), f"{prefix}_raw_p75": quantile(x,.75),
                    f"{prefix}_raw_p90": quantile(x,.9), f"{prefix}_raw_p95": quantile(x,.95),
                    f"{prefix}_nonnegative_mean": float(nonneg.mean()), f"{prefix}_nonnegative_median": quantile(nonneg,.5),
                    f"{prefix}_nonnegative_p75": quantile(nonneg,.75), f"{prefix}_nonnegative_p90": quantile(nonneg,.9), f"{prefix}_nonnegative_p95": quantile(nonneg,.95)})
    return out


def scheme_cohort(frame: pd.DataFrame, target: str, t: pd.Timestamp, w: int, scheme: str, lag: int, audit_endpoint: pd.Timestamp):
    spec = TARGETS[target]
    if scheme == "A":
        full = frame[frame[spec["available"]].ge(t-pd.Timedelta(days=w)) & frame[spec["available"]].lt(t)]
        mature = full
        eventual = full
    elif scheme == "B":
        full = frame[frame.order_purchase_timestamp.ge(t-pd.Timedelta(days=w)) & frame.order_purchase_timestamp.lt(t)]
        mature = full[full[spec["available"]].lt(t)]
        eventual = full[full[spec["available"]].notna() & full[spec["available"]].le(audit_endpoint)]
    else:
        full = frame[frame.order_purchase_timestamp.ge(t-pd.Timedelta(days=lag+w)) & frame.order_purchase_timestamp.lt(t-pd.Timedelta(days=lag))]
        mature = full[full[spec["available"]].lt(t)]
        eventual = full[full[spec["available"]].notna() & full[spec["available"]].le(audit_endpoint)]
    return full, mature, eventual


def historical_valid(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame[target_valid_mask(frame, target)]


def rank_stats(entity_summary: pd.DataFrame, kind: str) -> list[str]:
    return ["rate"] if kind == "binary" else ["mean", "median", "p90"]


def entity_table(
    mature: pd.DataFrame,
    target: str,
    entity: str,
    t: pd.Timestamp,
    *,
    valid_frame: pd.DataFrame | None = None,
    include_rank_stats: bool = True,
) -> pd.DataFrame:
    spec = TARGETS[target]; value=spec["value"]
    x = (historical_valid(mature, target) if valid_frame is None else valid_frame).copy()
    x = x[x[entity].notna()]
    if x.empty:
        return pd.DataFrame(columns=[entity,"support","mean","median","p90","rate","last_available","active_days","volume"])
    # Pandas/Numpy no longer linearly interpolates boolean quantiles.  Numeric
    # target summaries are equivalent and keep binary breach rates explicit.
    x[value] = pd.to_numeric(x[value], errors="coerce").astype(float)
    grouped=x.groupby(entity, observed=True)
    g = grouped.agg(
        support=(value,"count"), last_available=(spec["available"],"max"), active_days=("purchase_date","nunique")
    )
    g["volume"]=g["support"]
    if include_rank_stats:
        g["mean"]=grouped[value].mean(); g["median"]=grouped[value].median()
        g["p90"]=grouped[value].quantile(.9,interpolation="linear"); g["rate"]=g["mean"]
    else:
        g[["mean","median","p90","rate"]]=np.nan
    g=g.reset_index()
    return g


def spearman_row(a: pd.DataFrame, b: pd.DataFrame, entity: str, stat: str, threshold: int) -> dict:
    aa=a[a.support.ge(threshold)][[entity,"support",stat]].rename(columns={"support":"support_a",stat:"value_a"})
    bb=b[b.support.ge(threshold)][[entity,"support",stat]].rename(columns={"support":"support_b",stat:"value_b"})
    j=aa.merge(bb,on=entity,how="inner")
    constant_a = bool(j.value_a.nunique(dropna=True) <= 1); constant_b = bool(j.value_b.nunique(dropna=True) <= 1)
    reason=""; valid=True; rho=np.nan; p=np.nan
    if len(j)<10: valid=False; reason="fewer_than_10_common_entities"
    elif constant_a or constant_b: valid=False; reason="constant_vector"
    elif j[["value_a","value_b"]].dropna().shape[0]<10: valid=False; reason="fewer_than_10_complete_pairs"
    else:
        z=j[["value_a","value_b"]].dropna(); result=spearmanr(z.value_a,z.value_b); rho=float(result.statistic); p=float(result.pvalue)
    return {"n_entities_source_a":len(a),"n_entities_source_b":len(b),"n_common_entities":len(j),"support_threshold":threshold,
            "spearman_correlation":rho,"p_value":p,"constant_vector_a":constant_a,"constant_vector_b":constant_b,
            "valid":valid,"invalid_reason":reason}


def data_quality(frame: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    def add(scope, check, count, denominator, detail=""):
        rows.append({"scope":scope,"check":check,"count":int(count),"denominator":int(denominator),"percentage":100*count/denominator if denominator else np.nan,"detail":detail})
    add("all_placed","duplicate_order_ids",frame.order_id.duplicated().sum(),len(frame))
    add("all_placed","multi_seller_orders",frame.is_multi_seller.sum(),len(frame))
    for name,col in ENTITIES.items(): add("all_placed",f"missing_entity_{name}",frame[col].isna().sum(),len(frame))
    for c in ["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]:
        add("all_placed",f"missing_{c}",frame[c].isna().sum(),len(frame))
    add("all_placed","negative_purchase_to_carrier",frame.purchase_to_carrier.lt(0).sum(),len(frame),"retained")
    add("all_placed","negative_handling",frame.handling_duration.lt(0).sum(),len(frame),"retained; excluded only from nonnegative diagnostic")
    add("all_placed","negative_transit",frame.transit_duration.lt(0).sum(),len(frame),"retained; excluded only from nonnegative diagnostic")
    add("all_placed","unresolved_non_delivered",frame.order_status.ne("delivered").sum(),len(frame),"never coded as non-breach or zero severity")
    add("all_placed","delivered_status_missing_customer_delivery",(frame.order_status.eq("delivered")&frame.order_delivered_customer_date.isna()).sum(),len(frame))
    for c in ["purchase_to_carrier","handling_duration","transit_duration","promise_error_days"]:
        threshold=quantile(frame[c],.999); add("all_placed",f"extreme_{c}_above_p999",frame[c].gt(threshold).sum(),len(frame),f"raw p99.9={threshold}; retained")
    add("canonical_delivered","duplicate_order_ids",canonical.order_id.duplicated().sum(),len(canonical))
    add("canonical_delivered","multi_seller_orders",canonical.n_unique_sellers.gt(1).sum(),len(canonical))
    add("canonical_delivered","missing_seller_id",canonical.main_seller_id.isna().sum(),len(canonical))
    add("canonical_delivered","missing_route_state",canonical.route_state.isna().sum(),len(canonical))
    add("canonical_delivered","missing_distance",canonical.distance_km.isna().sum(),len(canonical),"retained V1 canonical distance diagnostic")
    add("canonical_delivered","negative_purchase_to_carrier",canonical.purchase_to_carrier.lt(0).sum(),len(canonical),"retained")
    add("canonical_delivered","negative_handling",canonical.post_approval_handling.lt(0).sum(),len(canonical),"retained")
    add("canonical_delivered","negative_transit",canonical.transit_time.lt(0).sum(),len(canonical),"retained")
    for c in ["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]:
        add("canonical_delivered",f"missing_{c}",canonical[c].isna().sum(),len(canonical))
    for name,c in [("purchase_to_carrier",canonical.purchase_to_carrier),("post_approval_handling",canonical.post_approval_handling),("transit_time",canonical.transit_time),("promise_error_days",canonical.promise_error_days)]:
        threshold=quantile(c,.999); add("canonical_delivered",f"extreme_{name}_above_p999",c.gt(threshold).sum(),len(canonical),f"raw p99.9={threshold}; retained")
    return pd.DataFrame(rows)


def duration_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for target in ["handling","transit"]:
        v=TARGETS[target]["value"]; x=frame[v]; non=x[x.ge(0)]
        for scope,s in [("raw",x),("nonnegative_only",non)]:
            rows.append({"target":target,"scope":scope,"count":int(s.notna().sum()),"mean":float(s.mean()),"p01":quantile(s,.01),"median":quantile(s,.5),"p90":quantile(s,.9),"p95":quantile(s,.95),"p99":quantile(s,.99)})
    return pd.DataFrame(rows)


def calendar_clusters(daily: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    cluster_rows=[]; phase_rows=[]
    dates_all=pd.date_range(daily.date.min(),daily.date.max(),freq="D")
    lookup=daily.set_index("date")
    for flag in HRD_DEFS:
        dates=set(daily.loc[daily[flag],"date"])
        clusters=[]; current=[]
        for d in sorted(dates):
            if current and (d-current[-1]).days != 1:
                clusters.append(current); current=[]
            current.append(d)
        if current: clusters.append(current)
        for i,c in enumerate(clusters,1):
            cluster_rows.append({"definition":flag,"cluster_id":i,"start_date":c[0],"end_date":c[-1],"duration_days":len(c),"n_hrd_days":len(c)})
            phase_dates=[(c[0]-pd.Timedelta(days=1),"pre_event",0)]
            phase_dates += [(d,"event",(d-c[0]).days+1) for d in c]
            phase_dates += [(c[-1]+pd.Timedelta(days=k),f"post_event_day_{k}",k) for k in [1,2,3]]
            for date,phase,k in phase_dates:
                row={"definition":flag,"cluster_id":i,"date":date,"phase":phase,"phase_day":k,"date_present_in_marketplace_table":date in lookup.index}
                phase_rows.append(row)
    return pd.DataFrame(cluster_rows),pd.DataFrame(phase_rows)


def md_table(frame: pd.DataFrame) -> str:
    cols=[str(x) for x in frame.columns]
    esc=lambda x: str(x).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]+["| "+" | ".join(esc(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])
