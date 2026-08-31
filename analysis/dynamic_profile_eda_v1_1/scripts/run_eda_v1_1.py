#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", ".cache/dynamic-profile-v1-1-mpl")
# Allow both the recorded direct-script command and ``python -m`` execution.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy

from analysis.dynamic_profile_eda_v1_1.scripts.core import (
    AGES, ASSEMBLER, ASSEMBLER_SHA, ENTITIES, HORIZONS, HRD_DEFS, LAGS,
    MATURITY_THRESHOLDS, OUT, PROTECTED, ROOT, SUPPORTS, TARGETS, WINDOWS,
    build_all_placed, calendar_clusters, data_quality, derive_intervals,
    duration_diagnostics, entity_table, historical_valid, maturity_outputs,
    md_table, quantile, rank_stats, reconcile_sample, recursive_hashes,
    scheme_cohort, sha256_file, spearman_row, target_summary, target_valid_mask,
)

EXPECTED_RAW_HASHES = {
    "orders": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "customers": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "geolocation": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    "items": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "products": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "sellers": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "categories": "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
}
CHARTER = ROOT / "docs/omitted-private-controls/OLIST_PROFILE_PIVOT_PROJECT_CHARTER_2026-08-21.md"

G_FRAME: pd.DataFrame | None = None
G_AUDIT_ENDPOINT: pd.Timestamp | None = None
G_RANK_DAYS: set[pd.Timestamp] = set()


def invocation_command() -> str:
    """Return the literal Python invocation, including interpreter flags such as ``-B``."""
    original = getattr(sys, "orig_argv", None)
    if original:
        return shlex.join(original)
    return shlex.join([sys.executable, *sys.argv])


def repository_state() -> dict:
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True)
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
    }


def workspace_hashes_excluding_v1_1() -> dict[str, str]:
    """Hash every tracked or nonignored untracked file outside the authorised workspace."""
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    prefix = "analysis/dynamic_profile_eda_v1_1/"
    result = {}
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        rel = os.fsdecode(encoded)
        if rel == "analysis/dynamic_profile_eda_v1_1" or rel.startswith(prefix):
            continue
        path = ROOT / rel
        if path.is_file():
            result[rel] = sha256_file(path)
    return result


def snapshot_worker_init(frame: pd.DataFrame, audit_endpoint: pd.Timestamp, rank_days: set[pd.Timestamp]):
    global G_FRAME, G_AUDIT_ENDPOINT, G_RANK_DAYS
    G_FRAME = frame
    G_AUDIT_ENDPOINT = audit_endpoint
    G_RANK_DAYS = rank_days


def _selection_differences(row: dict) -> None:
    for k in list(row):
        if k.startswith("asof_"):
            suffix = k[len("asof_"):]
            ek = "eventual_" + suffix
            if ek in row and isinstance(row[k], (int,float,np.integer,np.floating)) and isinstance(row[ek], (int,float,np.integer,np.floating)):
                row["eventual_minus_asof_"+suffix] = row[ek] - row[k] if pd.notna(row[k]) and pd.notna(row[ek]) else np.nan


def process_snapshot(t_value: str) -> tuple[list[dict],list[dict],list[dict],list[dict]]:
    assert G_FRAME is not None and G_AUDIT_ENDPOINT is not None
    frame=G_FRAME; audit_endpoint=G_AUDIT_ENDPOINT; t=pd.Timestamp(t_value)
    slicing=[]; support_rows=[]; coverage_rows=[]; ranks=[]
    future={h:frame[frame.order_purchase_timestamp.ge(t)&frame.order_purchase_timestamp.lt(t+pd.Timedelta(days=h))] for h in HORIZONS}
    scheme_defs=[("A",0),("B",0)]+[("C",l) for l in LAGS]
    for target,spec in TARGETS.items():
        for w in WINDOWS:
            entity_cache={}
            cohort_cache={}
            for scheme,lag in scheme_defs:
                full,mature,eventual=scheme_cohort(frame,target,t,w,scheme,lag,audit_endpoint)
                cohort_cache[(scheme,lag)]=(full,mature,eventual)
                mature_valid=historical_valid(mature,target)
                eventual_valid=historical_valid(eventual,target)
                summaries={}; summaries.update(target_summary(mature,target,"asof")); summaries.update(target_summary(eventual,target,"eventual")); _selection_differences(summaries)
                for granularity,entity in ENTITIES.items():
                    et=entity_table(mature,target,entity,t,valid_frame=mature_valid,include_rank_stats=t in G_RANK_DAYS); entity_cache[(scheme,lag,granularity)]=et
                    row={"snapshot_date":t,"sample":"all_placed","target":target,"target_kind":spec["kind"],"window_days":w,"scheme":scheme,"lag_days":lag,"granularity":granularity,
                         "source_records":len(full),"entity_id_available_orders":int(full[entity].notna().sum()),"entity_id_nonmissing_rate":float(full[entity].notna().mean()) if len(full) else np.nan,
                         "timestamp_observed_asof_orders":len(mature),"valid_outcomes_asof":len(mature_valid),"invalid_or_anomalous_outcomes_asof":len(mature)-len(mature_valid)}
                    if scheme=="A":
                        row.update({"cohort_total_orders_all_placed":np.nan,"eventually_available_orders_by_audit_end":np.nan,"mature_asof_orders":np.nan,"unresolved_asof_orders":np.nan,
                                    "never_observed_by_audit_end_orders":np.nan,"unconditional_maturity_fraction":np.nan,"conditional_maturity_fraction":np.nan,"eventual_observed_fraction":np.nan,
                                    "purchase_cohort_maturity_reason":"not_applicable_for_completion_window"})
                    else:
                        eventual_n=len(eventual); mature_n=len(mature); denom=len(full)
                        row.update({"cohort_total_orders_all_placed":denom,"eventually_available_orders_by_audit_end":eventual_n,"mature_asof_orders":mature_n,
                                    "unresolved_asof_orders":denom-mature_n,"never_observed_by_audit_end_orders":denom-eventual_n,
                                    "unconditional_maturity_fraction":mature_n/denom if denom else np.nan,"conditional_maturity_fraction":mature_n/eventual_n if eventual_n else np.nan,
                                    "eventual_observed_fraction":eventual_n/denom if denom else np.nan,"purchase_cohort_maturity_reason":"full_purchase_cohort_denominator"})
                    row.update(summaries)
                    slicing.append(row)

                    if len(et):
                        vol=et.volume.sort_values(ascending=False); total=vol.sum()
                        sr={"snapshot_date":t,"sample":"all_placed","target":target,"window_days":w,"scheme":scheme,"lag_days":lag,"granularity":granularity,
                            "active_entities":len(et),"support_p10":quantile(et.support,.1),"support_p25":quantile(et.support,.25),"support_median":quantile(et.support,.5),"support_p75":quantile(et.support,.75),"support_p90":quantile(et.support,.9),
                            "entity_id_nonmissing_rate":float(full[entity].notna().mean()) if len(full) else np.nan,"pct_entities_active_one_day":float(et.active_days.eq(1).mean()),"median_active_days":quantile(et.active_days,.5),
                            "profile_freshness_median_days":quantile((t-et.last_available).dt.total_seconds()/86400,.5)}
                        for k in SUPPORTS: sr[f"entities_support_ge_{k}"]=int(et.support.ge(k).sum()); sr[f"pct_entities_support_ge_{k}"]=float(et.support.ge(k).mean())
                        for p in [1,5,10]: sr[f"top_{p}pct_order_concentration"]=float(vol.head(max(1,int(np.ceil(len(vol)*p/100)))).sum()/total) if total else np.nan
                    else:
                        sr={"snapshot_date":t,"sample":"all_placed","target":target,"window_days":w,"scheme":scheme,"lag_days":lag,"granularity":granularity,"active_entities":0,"entity_id_nonmissing_rate":float(full[entity].notna().mean()) if len(full) else np.nan}
                    sup_map=et.set_index(entity).support if len(et) else pd.Series(dtype=float)
                    for h,fut in future.items():
                        mapped=fut[entity].notna(); hist=fut[entity].map(sup_map).fillna(0).astype(float)
                        seen=mapped & hist.ge(1); cold=mapped & hist.eq(0)
                        mapped_support=hist[mapped]
                        cr={"snapshot_date":t,"sample":"all_placed","target":target,"window_days":w,"scheme":scheme,"lag_days":lag,"granularity":granularity,"future_horizon_days":h,
                            "total_future_placed_orders":len(fut),"orders_with_valid_entity_mapping":int(mapped.sum()),"entity_id_nonmissing_rate":float(mapped.mean()) if len(fut) else np.nan,
                            "historical_seen_orders":int(seen.sum()),"historical_seen_rate":float(seen.mean()) if len(fut) else np.nan,
                            "mapped_cold_start_orders":int(cold.sum()),"cold_start_rate":float(cold.mean()) if len(fut) else np.nan,
                            "cold_start_rate_among_mapped":float(cold.sum()/mapped.sum()) if mapped.sum() else np.nan,
                            "missing_mapping_count":int((~mapped).sum()),"multi_seller_count":int(fut.is_multi_seller.sum()),
                            "support_quantile_denominator":"mapped_future_orders_including_seen_and_unseen",
                            "median_historical_support":quantile(mapped_support,.5),"support_p10":quantile(mapped_support,.1),"support_p25":quantile(mapped_support,.25),"support_p75":quantile(mapped_support,.75),"support_p90":quantile(mapped_support,.9)}
                        for k in SUPPORTS:
                            qualified=mapped & hist.ge(k)
                            cr[f"orders_support_ge_{k}"]=int(qualified.sum()); cr[f"order_weighted_support_ge_{k}_rate"]=float(qualified.mean()) if len(fut) else np.nan
                        coverage_rows.append(cr)
                        sr[f"future_{h}d_seen_rate"]=cr["historical_seen_rate"]
                        sr[f"future_{h}d_entity_id_nonmissing_rate"]=cr["entity_id_nonmissing_rate"]
                        for k in SUPPORTS: sr[f"future_{h}d_support_ge_{k}_rate"]=cr[f"order_weighted_support_ge_{k}_rate"]
                    support_rows.append(sr)

            # All-granularity rank agreement on frozen month-start audit snapshots.
            if t in G_RANK_DAYS:
                pairs=[(("A",0),("B",0))]
                pairs += [(("A",0),("C",l)) for l in LAGS]
                pairs += [(("B",0),("C",l)) for l in LAGS]
                pairs += [(('C',a),('C',b)) for a,b in zip(LAGS[:-1],LAGS[1:])]
                for granularity,entity in ENTITIES.items():
                    stats=rank_stats(pd.DataFrame(),spec["kind"])
                    for a,b in pairs:
                        ea=entity_cache[(a[0],a[1],granularity)]; eb=entity_cache[(b[0],b[1],granularity)]
                        for stat in stats:
                            for threshold in [1,5,10,20]:
                                rr=spearman_row(ea,eb,entity,stat,threshold)
                                rr.update({"snapshot_date":t,"target":target,"window_days":w,"granularity":granularity,"entity_statistic":stat,
                                           "scheme_a":a[0],"lag_a_days":a[1],"scheme_b":b[0],"lag_b_days":b[1],"interpretation":"descriptive_historical_rank_agreement_not_predictive_validation"})
                                ranks.append(rr)
    return slicing,support_rows,coverage_rows,ranks


def chunk_worker(args):
    idx,dates,part_dir=args
    buckets=[[],[],[],[]]
    for d in dates:
        result=process_snapshot(d)
        for i,x in enumerate(result): buckets[i].extend(x)
    names=["slicing","support","coverage","rank"]
    paths=[]
    for name,rows in zip(names,buckets):
        if not rows:
            continue
        p=Path(part_dir)/f"{idx:03d}_{name}.csv"; pd.DataFrame(rows).to_csv(p,index=False); paths.append(str(p))
    return paths


def fast_chunk_worker(args):
    """Bounded worker for parity-gated slicing/support/coverage only."""
    assert G_FRAME is not None and G_AUDIT_ENDPOINT is not None
    from analysis.dynamic_profile_eda_v1_1.scripts.fast_engine import compute_snapshot_outputs
    idx,dates,part_dir=args
    outputs=compute_snapshot_outputs(G_FRAME,G_AUDIT_ENDPOINT,dates,set())[:3]
    names=["slicing","support","coverage"]
    paths=[]
    for name,frame in zip(names,outputs):
        if frame.empty:
            continue
        p=Path(part_dir)/f"{idx:03d}_{name}.csv"; frame.to_csv(p,index=False); paths.append(str(p))
    return paths


def reference_rank_worker(args):
    """Persist rank rows from the audited pandas reference path."""
    idx,dates,part_dir=args
    rows=[]
    for date in dates:
        rows.extend(process_snapshot(date)[3])
    if not rows:
        return []
    p=Path(part_dir)/f"{idx:03d}_rank.csv"; pd.DataFrame(rows).to_csv(p,index=False); return [str(p)]


PARITY_SORT_COLUMNS={
    "slicing":["snapshot_date","target","window_days","scheme","lag_days","granularity"],
    "support":["snapshot_date","target","window_days","scheme","lag_days","granularity"],
    "coverage":["snapshot_date","target","window_days","scheme","lag_days","granularity","future_horizon_days"],
    "rank":["snapshot_date","target","window_days","granularity","scheme_a","lag_a_days","scheme_b","lag_b_days","entity_statistic","support_threshold"],
}


def validate_fast_engine_parity(frame: pd.DataFrame,audit_endpoint: pd.Timestamp) -> dict:
    """Hard-gate the production engine against reference ordinary/rank snapshots."""
    from analysis.dynamic_profile_eda_v1_1.scripts.fast_engine import compute_snapshot_outputs
    snapshots=[pd.Timestamp("2018-02-01"),pd.Timestamp("2018-02-02")]
    rank_days={snapshots[0]}
    snapshot_worker_init(frame,audit_endpoint,rank_days)
    started=time.time()
    reference_buckets=[[],[],[],[]]
    for snapshot in snapshots:
        result=process_snapshot(str(snapshot.date()))
        for idx,rows in enumerate(result): reference_buckets[idx].extend(rows)
    reference_seconds=time.time()-started
    if len(reference_buckets[3]) != 15120:
        raise AssertionError(f"reference rank parity snapshot row count differs: {len(reference_buckets[3])}")
    started=time.time(); fast=compute_snapshot_outputs(frame,audit_endpoint,snapshots,set())[:3]; fast_seconds=time.time()-started
    names=["slicing","support","coverage"]
    row_counts={}
    for name,reference_rows,fast_frame in zip(names,reference_buckets[:3],fast):
        reference=pd.DataFrame(reference_rows)
        if set(reference.columns)!=set(fast_frame.columns):
            raise AssertionError(f"fast engine {name} schema differs: reference_only={set(reference)-set(fast_frame)}, fast_only={set(fast_frame)-set(reference)}")
        columns=sorted(reference.columns)
        sort_columns=[c for c in PARITY_SORT_COLUMNS[name] if c in columns]
        reference=reference.sort_values(sort_columns,kind="mergesort").reset_index(drop=True)[columns]
        candidate=fast_frame.sort_values(sort_columns,kind="mergesort").reset_index(drop=True)[columns]
        pd.testing.assert_frame_equal(reference,candidate,check_dtype=False,check_exact=False,rtol=1e-11,atol=1e-11,obj=f"fast engine parity: {name}")
        row_counts[name]=len(reference)
    row_counts["reference_rank"]=len(reference_buckets[3])
    return {"status":"passed_for_slicing_support_coverage","snapshots":[str(x) for x in snapshots],"reference_rank_snapshot":str(snapshots[0]),"reference_seconds":reference_seconds,"fast_seconds":fast_seconds,"speedup":reference_seconds/fast_seconds if fast_seconds else np.nan,"row_counts":row_counts,"float_tolerance":{"rtol":1e-11,"atol":1e-11},"rank_engine":"pandas reference; candidate fast rank rejected after real-data parity failure"}


def combine_parts(part_dir: Path, kind: str, output: Path, sort_cols: list[str]) -> pd.DataFrame:
    files=sorted(part_dir.glob(f"*_{kind}.csv")); wrote=False
    for p in files:
        if p.stat().st_size==0:
            continue
        part=pd.read_csv(p,low_memory=False)
        if len(part):
            part=part.sort_values([c for c in sort_cols if c in part.columns],kind="mergesort")
        part.to_csv(output,index=False,mode="a" if wrote else "w",header=not wrote)
        wrote=True
    if not wrote:
        pd.DataFrame().to_csv(output,index=False)
        return pd.DataFrame()
    return pd.read_csv(output,low_memory=False)


def all_placed_daily(frame: pd.DataFrame) -> pd.DataFrame:
    statuses=frame.pivot_table(index="purchase_date",columns="order_status",values="order_id",aggfunc="count",fill_value=0).add_prefix("status_")
    g=frame.groupby("purchase_date",observed=True).agg(order_count=("order_id","size"),delivered_count=("order_status",lambda s:int(s.eq("delivered").sum())),
        unresolved_or_cancelled_count=("order_status",lambda s:int(s.ne("delivered").sum())),active_sellers=("main_seller_id","nunique"),active_routes=("state_od","nunique"),
        orders_with_gmv=("gmv_observed","sum"),total_gmv=("total_price","sum"),freight_value=("total_freight_value","sum")).join(statuses)
    dates=pd.date_range(frame.purchase_date.min(),frame.purchase_date.max(),freq="D")
    g=g.reindex(dates,fill_value=0); g.index.name="date"; d=g.reset_index(); d["orders_missing_gmv"]=d.order_count-d.orders_with_gmv; d["gmv_join_coverage"]=d.orders_with_gmv/d.order_count.replace(0,np.nan)
    for metric,prefix in [("order_count","order"),("total_gmv","gmv")]:
        for q,label in [(.9,"top10"),(.95,"top5")]: d[f"{prefix}_{label}"]=d[metric].ge(d[metric].quantile(q))
    d["both_top10"]=d.order_top10&d.gmv_top10; d["both_top5"]=d.order_top5&d.gmv_top5
    return d


def hrd_outputs(frame: pd.DataFrame, daily: pd.DataFrame, clusters: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    tagged=frame.merge(daily[["date"]+list(HRD_DEFS)],left_on="purchase_date",right_on="date",how="left")
    rows=[]
    for flag in HRD_DEFS:
        sub=tagged[tagged[flag].fillna(False)]
        for gran,entity in ENTITIES.items():
            support=sub.groupby(entity,observed=True).size(); cluster_count=clusters.loc[clusters.definition.eq(flag),"cluster_id"].nunique()
            r={"definition":flag,"granularity":gran,"n_hrd_days":int(daily[flag].sum()),"n_event_clusters":int(cluster_count),"n_orders":len(sub),"pct_all_placed_orders":len(sub)/len(frame),
               "active_entities":len(support),"entity_id_nonmissing_rate":float(sub[entity].notna().mean()) if len(sub) else np.nan,"historical_hrd_clusters_available":int(cluster_count)}
            for k in SUPPORTS:
                r[f"entities_support_ge_{k}"]=int(support.ge(k).sum()); r[f"pct_entities_support_ge_{k}"]=float(support.ge(k).mean()) if len(support) else np.nan
                r[f"order_weighted_support_ge_{k}_rate"]=float(sub[entity].map(support).ge(k).mean()) if len(sub) else np.nan
            rows.append(r)
    overlap=[]
    for a in HRD_DEFS:
        for b in HRD_DEFS:
            inter=int((daily[a]&daily[b]).sum()); union=int((daily[a]|daily[b]).sum())
            overlap.append({"definition_a":a,"definition_b":b,"days_intersection":inter,"days_union":union,"jaccard":inter/union if union else np.nan})
    return pd.DataFrame(rows),pd.DataFrame(overlap)


def scheme_b_maturity_by_purchase_age(frame: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """All-placed Scheme-B availability by calendar purchase-age at the common endpoint."""
    cohort=frame[
        frame.order_purchase_timestamp.ge(snapshot-pd.Timedelta(days=60))
        & frame.order_purchase_timestamp.lt(snapshot)
    ].copy()
    cohort["purchase_age_days"]=(snapshot.normalize()-cohort.order_purchase_timestamp.dt.normalize()).dt.days
    rows=[]
    for target,spec in TARGETS.items():
        for age,g in cohort.groupby("purchase_age_days",observed=True):
            available=g[spec["available"]].notna() & g[spec["available"]].lt(snapshot)
            rows.append({
                "snapshot_date":snapshot,"target":target,"purchase_age_days":int(age),
                "all_placed_orders":len(g),"available_asof_orders":int(available.sum()),
                "unconditional_availability_asof":float(available.mean()) if len(g) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["target","purchase_age_days"]).reset_index(drop=True)


def make_figures() -> dict[str,dict]:
    figdir=OUT/"figures"; figdir.mkdir(exist_ok=True); srcdir=OUT/"figure_sources"; srcdir.mkdir(exist_ok=True); plt.style.use("seaborn-v0_8-whitegrid")
    records={}
    def source(name,frame):
        p=srcdir/name; frame.to_csv(p,index=False); return p
    def save(name,source_path):
        p=figdir/name; plt.tight_layout(); plt.savefig(p,dpi=150); plt.close(); records[name]={"source_csv":str(source_path.relative_to(OUT)),"source_sha256":sha256_file(source_path),"source_rows":len(pd.read_csv(source_path)),"sha256":sha256_file(p),"bytes":p.stat().st_size}
    m=pd.read_csv(OUT/"MATURITY_CURVES_UNCONDITIONAL.csv")
    for target,g in m.groupby("target"): plt.plot(g.age_days,g.unconditional_cumulative_availability,label=target)
    plt.axhline(1,color="k",lw=.5); plt.legend(fontsize=7); plt.xlabel("days since purchase"); plt.ylabel("all-order availability"); save("01_unconditional_availability.png",source("01_unconditional_availability.csv",m))
    q=pd.read_csv(OUT/"MATURITY_QUANTILES_CONDITIONAL.csv"); z=q[q.conditioning.eq("conditional_on_eventual_observation")]
    zp=z.pivot(index="target",columns="threshold",values="age_days_required").reset_index(); zp.set_index("target").plot.bar(); plt.ylabel("conditional maturity age (days)"); save("02_conditional_maturity_quantiles.png",source("02_conditional_maturity_quantiles.csv",zp))
    c=pd.read_csv(OUT/"MATURITY_COMPONENT_COUNTS.csv"); cq=q[(q.conditioning.eq("conditional_on_eventual_observation"))&(q.threshold.eq(.95))][["target","age_days_required"]].rename(columns={"age_days_required":"conditional_p95_age_days"}); z=c[["target","eventual_observation_plateau"]].merge(cq,on="target"); ax=z.plot.bar(x="target",y="eventual_observation_plateau",legend=False,color="#4C78A8"); ax.set_ylim(.9,1); ax.set_ylabel("all-order eventual-observation plateau"); ax2=ax.twinx(); ax2.plot(np.arange(len(z)),z.conditional_p95_age_days,color="#E45756",marker="o"); ax2.set_ylabel("conditional P95 maturity age (days)"); save("03_plateau_vs_conditional.png",source("03_plateau_vs_conditional.csv",z))
    s=pd.read_csv(OUT/"SLICING_SCHEME_COMPARISON_V1_1.csv",low_memory=False)
    z=pd.read_csv(OUT/"SCHEME_B_MATURITY_BY_PURCHASE_AGE.csv"); [plt.plot(g.purchase_age_days,g.unconditional_availability_asof,label=target) for target,g in z.groupby("target")]; plt.axhline(.95,color="k",lw=.7,ls="--"); plt.xlabel("calendar days since purchase at common snapshot"); plt.ylabel("Scheme B all-order availability as of snapshot"); plt.legend(fontsize=7); save("04_scheme_b_maturity.png",source("04_scheme_b_maturity.csv",z))
    z=s[(s.scheme.eq("C"))&(s.target.eq("final_breach"))&(s.granularity.eq("seller_id"))&(s.window_days.eq(60))].groupby("lag_days",as_index=False).unconditional_maturity_fraction.mean(); z.plot(x="lag_days",y="unconditional_maturity_fraction",marker="o",legend=False); plt.ylabel("mean unconditional maturity"); save("05_scheme_c_maturity_by_lag.png",source("05_scheme_c_maturity_by_lag.csv",z))
    metric_cols={"final_breach":("asof_breach_rate","eventual_breach_rate","breach rate"),"positive_late_days":("asof_mean","eventual_mean","mean nonnegative lateness days"),"handling":("asof_raw_mean","eventual_raw_mean","raw mean handling days"),"transit":("asof_raw_mean","eventual_raw_mean","raw mean transit days")}; rows=[]
    for target,(ac,ec,label) in metric_cols.items():
        g=s[(s.scheme.isin(["B","C"]))&(s.target.eq(target))&(s.granularity.eq("seller_id"))].groupby("scheme")[[ac,ec]].mean().reset_index()
        for row in g.itertuples(index=False): rows.append({"target":target,"scheme":row.scheme,"metric":label,"asof":getattr(row,ac),"eventual":getattr(row,ec)})
    z=pd.DataFrame(rows); fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(target,g) in zip(axes.flat,z.groupby("target",sort=False)):
        g.set_index("scheme")[["asof","eventual"]].plot.bar(ax=ax); ax.set_title(f"{target}: {g.metric.iloc[0]}"); ax.set_xlabel("")
    save("06_asof_vs_eventual.png",source("06_asof_vs_eventual.csv",z))
    cv=pd.read_csv(OUT/"FUTURE_PROFILE_COVERAGE.csv"); z=cv[(cv.future_horizon_days.eq(7))&(cv.target.eq("final_breach"))&(cv.scheme.eq("A"))&(cv.window_days.eq(90))].groupby("granularity",as_index=False).historical_seen_rate.mean().sort_values("historical_seen_rate"); z.plot.barh(x="granularity",y="historical_seen_rate",legend=False); plt.xlabel("7-day future seen rate"); save("07_future_7d_coverage.png",source("07_future_7d_coverage.csv",z))
    z=cv[(cv.future_horizon_days.eq(30))&(cv.target.eq("final_breach"))&(cv.scheme.eq("A"))&(cv.window_days.eq(90))].groupby("granularity",as_index=False).historical_seen_rate.mean().sort_values("historical_seen_rate"); z.plot.barh(x="granularity",y="historical_seen_rate",legend=False); plt.xlabel("30-day future seen rate"); save("08_future_30d_coverage.png",source("08_future_30d_coverage.csv",z))
    e=pd.read_csv(OUT/"ENTITY_GRANULARITY_SUPPORT_V1_1.csv"); z=e[(e.target.eq("final_breach"))&(e.scheme.eq("A"))&(e.window_days.eq(90))].groupby("granularity").agg(support=("support_median","mean"),coverage=("future_7d_seen_rate","mean")).reset_index(); plt.scatter(z.support,z.coverage); [plt.text(row.support,row.coverage,row.granularity,fontsize=6) for row in z.itertuples()]; plt.xlabel("median historical support"); plt.ylabel("future seen coverage"); save("09_coverage_specificity.png",source("09_coverage_specificity.csv",z))
    r=pd.read_csv(OUT/"ENTITY_RANK_AGREEMENT_ALL_GRANULARITIES.csv",low_memory=False); z=r[(r.valid.eq(True))&(r.support_threshold.eq(5))].groupby("granularity",as_index=False).spearman_correlation.mean().sort_values("spearman_correlation"); z.plot.barh(x="granularity",y="spearman_correlation",legend=False); plt.xlabel("mean descriptive Spearman"); save("10_rank_agreement.png",source("10_rank_agreement.csv",z))
    d=pd.read_csv(OUT/"DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv",parse_dates=["date"]); z=d[["date","order_count","total_gmv","both_top10"]]; ax=z.plot(x="date",y="order_count",legend=False,color="#4C78A8"); ax.set_ylabel("all-placed order count"); flagged=z.loc[z.both_top10]; ax.scatter(mdates.date2num(flagged.date.dt.to_pydatetime()),flagged.order_count.to_numpy(dtype=float),s=8,c="#E45756"); ax2=ax.twinx(); ax2.plot(z.date,z.total_gmv,color="#72B7B2",alpha=.65); ax2.set_ylabel("GMV (item price)"); save("11_all_placed_hrd.png",source("11_all_placed_hrd.csv",z))
    p=pd.read_csv(OUT/"HRD_EVENT_PHASES.csv",parse_dates=["date"]); z=p[p.definition.eq("both_top10")][["definition","cluster_id","date","phase","phase_day","date_present_in_marketplace_table"]].copy(); phase_order={"pre_event":0,"event":1,"post_event_day_1":2,"post_event_day_2":3,"post_event_day_3":4}; z["phase_code"]=z.phase.map(phase_order); sc=plt.scatter(mdates.date2num(z.date.dt.to_pydatetime()),z.cluster_id.to_numpy(dtype=float),c=z.phase_code.to_numpy(dtype=float),cmap="viridis",marker="s",s=18); plt.gca().xaxis_date(); plt.ylabel("both-top10 HRD cluster"); plt.xlabel("calendar date"); cb=plt.colorbar(sc); cb.set_ticks(list(phase_order.values()),labels=list(phase_order)); save("12_hrd_clusters_phases.png",source("12_hrd_clusters_phases.csv",z))
    ia=pd.read_csv(OUT/"SNAPSHOT_INTERVAL_AUDIT.csv"); z=ia[ia.boundary_scope.eq("target_purchase_comparison")][["target","unconditional_availability"]]; z.plot.bar(x="target",y="unconditional_availability",legend=False); plt.axhline(.95,color="r"); plt.ylabel("terminal prior-60d availability"); save("13_terminal_cohort_availability.png",source("13_terminal_cohort_availability.csv",z))
    dur=pd.read_csv(OUT/"PROCESS_DURATION_DIAGNOSTICS.csv"); z=dur[["target","scope","count","mean","median","p90","p95"]]; z.pivot(index="target",columns="scope",values="mean").plot.bar(); plt.ylabel("duration mean (days)"); save("14_process_duration_diagnostics.png",source("14_process_duration_diagnostics.csv",z))
    return records


def write_reports(reconciliation: pd.DataFrame):
    maturity=pd.read_csv(OUT/"MATURITY_QUANTILES_CONDITIONAL.csv"); comp=pd.read_csv(OUT/"MATURITY_COMPONENT_COUNTS.csv")
    interval=pd.read_csv(OUT/"SNAPSHOT_INTERVAL_AUDIT.csv"); quality=pd.read_csv(OUT/"DATA_QUALITY_AUDIT_V1_1.csv")
    six=reconciliation[reconciliation.customer_delivery_observed & ~reconciliation.in_canonical]
    lines=["# Dynamic profile EDA V1.1 corrected results","","> Correction-only descriptive EDA. These outputs do not establish future predictive validity or select a profile design.","",
        "## Sample reconciliation",f"Raw orders: **{len(reconciliation):,}**; customer-delivery timestamp observed: **{reconciliation.customer_delivery_observed.sum():,}**; canonical delivered: **{reconciliation.in_canonical.sum():,}**. The six-order difference consists entirely of timestamp-observed orders marked `canceled`.","",md_table(six[["order_id","order_status","deterministic_reconciliation_reason"]]),"",
        "## Corrected maturity",md_table(comp[["target","all_placed_orders","availability_by_audit_endpoint","eventual_observation_plateau","target_value_valid_for_descriptive_summary","negative_duration_count"]]),"",
        "Conditional quantiles are reported separately and are not all-order maturity ages.",md_table(maturity[maturity.conditioning.eq("conditional_on_eventual_observation")]),"",
        "## Interval audit",md_table(interval),"","## Data quality",md_table(quality),"",
        "## Interpretation boundary","Scheme A is recent-completion evidence. Schemes B/C retain complete purchase-cohort denominators and quantify as-of versus eventual maturity selection. Entity-ID completeness is separate from future historical-support coverage. Rank agreement is descriptive, not predictive validation. HRD labels are retrospective and built from all placed orders.","",
        "No estimator, model, risk level, window, lag, granularity, HRD definition, thesis branch, or business policy was selected."]
    (OUT/"EDA_RESULTS_SUMMARY.md").write_text("\n".join(lines))
    zh=["# 动态画像 EDA V1.1 修正结果","","> 本结果仅为修正后的描述性 EDA，不证明未来预测有效性，也不选择最终画像方案。","",
        f"原始订单 **{len(reconciliation):,}**；有客户签收时间的订单 **{reconciliation.customer_delivery_observed.sum():,}**；规范已交付订单 **{reconciliation.in_canonical.sum():,}**。六单差异全部来自已有签收时间但状态为 `canceled` 的订单。","",
        "V1.1 已严格区分：全部下单订单的无条件标签可用率，以及仅在最终可观察结果中的条件成熟分位数。条件 99 分位不再表述为 99% 全部订单已经成熟。","",
        "B/C 方案先保留完整购买队列，因此未解决订单仍在分母中；A 方案不再报告不可比的购买队列成熟率。实体 ID 完整率与未来历史支持覆盖率分开报告。","",
        "HRD 已按全部 99,441 个下单订单重建，未使用 25% 实体达到 support≥20 的未经批准通过阈值。","",
        "未拟合任何预测模型，未选择估计器、风险等级、窗口、滞后、粒度、HRD 定义、论文分支或业务策略。"]
    (OUT/"EDA_RESULTS_SUMMARY_ZH.md").write_text("\n".join(zh))
    (OUT/"BLOCKERS.md").write_text("# Remaining blockers and unresolved design choices\n\n- The audit endpoint is a proxy, not an authoritative administrative censoring date.\n- EDA cannot choose a final window, lag, entity granularity, estimator, fallback hierarchy, or risk level.\n- Fine seller-route granularities remain sparse; any pooling/fallback rule needs a separately frozen validation protocol.\n- Negative handling/transit durations remain anomalies whose later modelling treatment must be pre-specified.\n- Retrospective HRD definitions are not start-of-day predictors and have not been predictively validated.\n- Descriptive historical rank agreement and support coverage do not establish future outcome prediction.\n")
    (OUT/"V1_V1_1_DIFF_SUMMARY.md").write_text("# V1 versus V1.1\n\n## Approved V1 to V1.1 design correction\n\n- The purchase-comparison interval audit uses the recent all-placed purchase cohort `[t-60 days, t)`.\n- `2018-10-17 13:22:46`, the raw maximum customer-delivery timestamp, is used only as a fixed retrospective audit-endpoint proxy. It is not an authoritative administrative censoring date or a true dataset-closure date.\n- Each target's latest eligible snapshot is the latest `t` whose full recent cohort has at least 95% unconditional label availability at that proxy endpoint; cross-target comparisons use the minimum target-specific endpoint.\n\n## Unchanged conclusions\n\n- Handling labels become available earlier than final customer-delivery outcomes.\n- Fine seller-route entities are sparse relative to coarse routes.\n- HRD labels are retrospective and many entity-specific HRD histories are limited.\n- EDA does not establish predictive validity or select a profile design.\n\n## Corrected or withdrawn statements\n\n- V1's 23.1/29.3/46.1-day delivery figures are relabelled as conditional-on-eventual-observation quantiles; they are withdrawn as all-order 90/95/99% maturity ages.\n- V1 Scheme B/C `n_orders` and maturity percentages were incorrect because mature filtering preceded denominator construction; replaced by full-cohort denominators.\n- V1 `order_weighted_coverage=1.0` was entity-ID completeness, not historical profile coverage; the coverage interpretation and figure are withdrawn.\n- V1 seller-only rank table is replaced by all-granularity descriptive agreement.\n- V1 delivered-only HRD demand table and 25% feasibility pass/fail rule are withdrawn.\n- V1 copied purchase-comparison endpoint is replaced by the approved recent-cohort, target-specific maturity-derived endpoints above.\n\nNo directional support conclusion is promoted to predictive evidence. Final window, lag, granularity, estimator, HRD definition and thesis architecture remain unresolved.\n")


def write_final_reports(
    manifest: dict,
    reconciliation: pd.DataFrame,
    slicing: pd.DataFrame,
    support: pd.DataFrame,
    coverage: pd.DataFrame,
    rank: pd.DataFrame,
    daily: pd.DataFrame,
    hrd: pd.DataFrame,
    interval: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    """Overwrite the provisional summaries with evidence-backed completion reports."""
    comp=pd.read_csv(OUT/"MATURITY_COMPONENT_COUNTS.csv")
    maturity=pd.read_csv(OUT/"MATURITY_QUANTILES_CONDITIONAL.csv")
    cond=maturity[maturity.conditioning.eq("conditional_on_eventual_observation")].pivot(index="target",columns="threshold",values="age_days_required").reset_index()
    cond.columns=["target"]+[f"conditional_age_p{str(c).replace('.','_')}" for c in cond.columns[1:]]
    maturity_summary=comp.merge(cond,on="target",how="left")

    # Global summaries are duplicated across granularity rows; remove that broadcast first.
    global_cells=slicing.drop_duplicates(["snapshot_date","target","window_days","scheme","lag_days"])
    selection_rows=[]
    diff_cols={"final_breach":"eventual_minus_asof_breach_rate","positive_late_days":"eventual_minus_asof_mean","handling":"eventual_minus_asof_raw_mean","transit":"eventual_minus_asof_raw_mean"}
    for target,col in diff_cols.items():
        for scheme in ["B","C"]:
            g=global_cells[(global_cells.target.eq(target))&(global_cells.scheme.eq(scheme))]
            selection_rows.append({
                "target":target,"scheme":scheme,"design_cells":len(g),
                "median_unconditional_maturity":quantile(g.unconditional_maturity_fraction,.5),
                "p10_unconditional_maturity":quantile(g.unconditional_maturity_fraction,.1),
                "p90_unconditional_maturity":quantile(g.unconditional_maturity_fraction,.9),
                "median_eventual_minus_asof_target_summary":quantile(g[col],.5) if col in g else np.nan,
            })
    selection=pd.DataFrame(selection_rows)

    support_summary=support.groupby("granularity",as_index=False).agg(
        design_cells=("snapshot_date","size"),median_active_entities=("active_entities","median"),
        median_entity_support=("support_median","median"),median_future_7d_seen=("future_7d_seen_rate","median"),
        median_future_30d_seen=("future_30d_seen_rate","median"),median_entity_id_nonmissing=("entity_id_nonmissing_rate","median"),
    )
    coverage_check=coverage.groupby(["granularity","future_horizon_days"],as_index=False).agg(
        median_seen_rate=("historical_seen_rate","median"),median_cold_start_rate=("cold_start_rate","median"),
        median_missing_mapping=("missing_mapping_count","median"),
    )
    rank_rows=[]
    for gran,g in rank.groupby("granularity"):
        valid=g[g.valid.astype(str).str.lower().eq("true")]
        rank_rows.append({"granularity":gran,"comparisons":len(g),"valid_comparisons":len(valid),"valid_fraction":len(valid)/len(g) if len(g) else np.nan,"median_valid_spearman":quantile(valid.spearman_correlation,.5)})
    rank_summary=pd.DataFrame(rank_rows)
    hrd_summary=hrd.groupby("definition",as_index=False).agg(n_hrd_days=("n_hrd_days","first"),n_event_clusters=("n_event_clusters","first"),n_orders=("n_orders","first"),median_active_entities=("active_entities","median"),median_entities_support_ge_20=("entities_support_ge_20","median"))
    dq=quality[quality.check.isin(["negative_purchase_to_carrier","negative_handling","negative_transit","missing_distance","delivered_status_missing_customer_delivery","unresolved_non_delivered"])].copy()
    six=reconciliation[reconciliation.customer_delivery_observed & ~reconciliation.in_canonical]
    tests=manifest.get("tests",{})
    inventory=pd.DataFrame([{"artifact":name,"rows":meta.get("rows"),"sha256":meta.get("sha256")} for name,meta in manifest.get("csv_artifacts",{}).items()])

    lines=[
        "# Dynamic profile EDA V1.1 corrected results","",
        "> Correction-only descriptive EDA. It does not establish future predictive validity and does not select a profile design.","",
        "## Reproducibility verdict",
        f"Canonical assembler hash matched the frozen SHA-256; protected V1/Phase 2A preservation: **{manifest.get('protected_byte_preservation_passed')}**; files outside the authorised V1.1 workspace unchanged during execution: **{manifest.get('outside_v1_1_workspace_unchanged')}**; pytest: **{tests.get('passed',0)}/{tests.get('collected',0)} passed**.","",
        "## Sample reconciliation",
        f"All **{len(reconciliation):,}** raw placed orders are reconciled. Customer-delivery timestamps are observed for **{int(reconciliation.customer_delivery_observed.sum()):,}** orders and the frozen canonical delivered sample contains **{int(reconciliation.in_canonical.sum()):,}** orders. The six timestamp-observed but noncanonical orders are all marked `canceled`; none is attributed to a missing estimate, missing timestamp, invalid join, or duplicate aggregation.","",md_table(six[["order_id","order_status","deterministic_reconciliation_reason"]]),"",
        "## Unconditional availability and conditional maturity",
        "The plateau is computed over all placed orders. Conditional ages use only eventually observed availability timestamps and are not all-order maturity ages.","",md_table(maturity_summary),"",
        "## Scheme B/C maturity and selection",
        "B/C denominators retain the full purchase cohort. The last column is the retrospective-minus-as-of change in the target-appropriate rate or mean; it is descriptive evidence of maturity selection, not a bias correction.","",md_table(selection),"",
        "## Seller and route support versus genuine future coverage",
        "Entity-ID nonmissingness, mapped cold start, missing mapping and historical seen coverage are separate quantities. Values below are medians over the complete frozen design grid and are not a design ranking.","",md_table(support_summary),"",md_table(coverage_check),"",
        "## Descriptive rank agreement",md_table(rank_summary),"",
        "## All-placed HRD evidence",
        f"Daily demand and GMV use all **{int(daily.order_count.sum()):,}** placed orders. No HRD feasibility pass/fail threshold is applied.","",md_table(hrd_summary),"",
        "## Snapshot interval audit",
        "The purchase-comparison audit is logically distinct from the broad grid. A coincident endpoint is reported as an empirical coincidence, not as a copied interval or authoritative closure date.","",md_table(interval),"",
        "## Data quality and retained anomalies",md_table(dq),"",
        "## Artifact inventory",md_table(inventory),"",
        "## Interpretation boundary",
        "No profile estimator, predictive model, risk level, window, lag, granularity, HRD definition, thesis branch, or business policy was selected. Phase 2A was not reclassified. Future support coverage and historical rank agreement are not future outcome-prediction evidence.",
    ]
    (OUT/"EDA_RESULTS_SUMMARY.md").write_text("\n".join(lines))

    zh=[
        "# 动态画像 EDA V1.1 修正结果","",
        "> 本结果仅为修正后的描述性 EDA，不证明未来预测有效性，也不选择最终画像方案。","",
        "## 可复现性与样本",
        f"规范汇编器哈希匹配；V1/Phase 2A 保持不变：**{manifest.get('protected_byte_preservation_passed')}**；授权目录外文件在运行期间保持不变：**{manifest.get('outside_v1_1_workspace_unchanged')}**；测试 **{tests.get('passed',0)}/{tests.get('collected',0)}** 通过。原始订单 {len(reconciliation):,}，有客户签收时间 {int(reconciliation.customer_delivery_observed.sum()):,}，规范已交付 {int(reconciliation.in_canonical.sum()):,}；六单差异均为已有签收时间但状态为 `canceled`。","",
        "## 成熟度",
        "全部下单订单的无条件可用率与最终可观察订单中的条件成熟分位数已分开；条件 P99 不再表述为全部订单的 99% 成熟年龄。","",md_table(maturity_summary),"",
        "## B/C 方案与选择效应",
        "B/C 保留完整购买队列分母，未解决订单不再被提前删除；下表的最终值与 as-of 值之差仅是描述性成熟选择证据。","",md_table(selection),"",
        "## 卖家、路线、覆盖与排序",
        "实体 ID 完整率、未来缺失映射、已映射冷启动及真实历史支持覆盖已分别报告。下列全网格中位数不构成方案排名。","",md_table(support_summary),"",md_table(rank_summary),"",
        "## HRD、区间与数据质量",
        f"HRD 使用全部 {int(daily.order_count.sum()):,} 个下单订单，不设置通过/失败阈值。购买比较端点在冻结的宽区间网格内按 `[t-60,t)` 与 95% 无条件可用率独立推导；审计端点仅为代理。","",md_table(interval),"",md_table(dq),"",
        "## 边界",
        "未拟合预测模型，未选择画像估计器、风险等级、窗口、滞后、粒度、HRD 定义、论文分支或业务策略；Phase 2A 未被重新分类。",
    ]
    (OUT/"EDA_RESULTS_SUMMARY_ZH.md").write_text("\n".join(zh))

    (OUT/"V1_V1_1_DIFF_SUMMARY.md").write_text(
        "# V1 versus V1.1\n\n"
        "## Approved design correction\n\n"
        "V1.1 uses the recent all-placed purchase cohort `[t-60 days,t)` on the frozen broad daily snapshot grid. `2018-10-17 13:22:46` is only a retrospective audit-endpoint proxy, not an administrative censoring or dataset-closure date. Target-specific 95% endpoints are independently calculated; empirical coincidence with the broad endpoint is disclosed.\n\n"
        "## Conclusions retained\n\n"
        "Handling labels become available earlier than final-delivery labels; fine seller-route entities remain sparser than coarse routes; HRD evidence remains retrospective; descriptive EDA does not establish predictive validity or choose a final design.\n\n"
        "## Numerical/table corrections and withdrawals\n\n"
        "- V1 conditional 23.1/29.3/46.1-day customer-delivery quantiles are relabelled and withdrawn as all-order 90/95/99% maturity claims.\n"
        "- V1 `SLICING_SCHEME_COMPARISON.csv` B/C denominators and maturity percentages are replaced by full-cohort calculations in `SLICING_SCHEME_COMPARISON_V1_1.csv`.\n"
        "- V1 `DAILY_SNAPSHOT_COVERAGE.csv` ID completeness is withdrawn as profile coverage and replaced by `FUTURE_PROFILE_COVERAGE.csv`.\n"
        "- V1 seller-only `ENTITY_RANK_AGREEMENT.csv` is replaced by all-granularity agreement with validity reasons and p-values.\n"
        "- V1 delivered-only HRD tables and the unapproved 25% pass/fail rule are withdrawn; all-placed raw evidence replaces them. The qualitative sparsity warning remains, but no directional feasibility pass/fail verdict survives.\n"
        "- The copied purchase endpoint is withdrawn and independently audited under the approved recent-cohort rule.\n\n"
        "## Figure replacements\n\n"
        "All 14 V1.1 figures are newly generated from exact persisted source CSVs: unconditional/conditional maturity (01–03), corrected Scheme B/C and as-of/eventual comparisons (04–06), genuine future coverage/support/rank evidence (07–10), all-placed HRD/calendar phases (11–12), interval audit (13), and raw/nonnegative process diagnostics (14).\n\n"
        "## Still unresolved\n\n"
        "No window, lag, entity granularity, estimator, fallback hierarchy, risk level, HRD definition, thesis architecture, or business policy is selected.\n"
    )

    blockers=list(dict.fromkeys(manifest.get("blockers",[])+[
        "the retrospective audit endpoint is a proxy rather than an authoritative censoring date",
        "future predictive validity has not been tested",
        "fine seller-route sparsity requires a separately frozen pooling/fallback and validation protocol",
        "negative process-duration treatment for any later model remains unresolved",
        "retrospective HRD labels are not deployable start-of-day predictors",
    ]))
    (OUT/"BLOCKERS.md").write_text("# Remaining blockers and unresolved design choices\n\n"+"\n".join(f"- {x}" for x in blockers)+"\n")


def validate_artifacts(required: list[str]) -> tuple[bool,str]:
    lines=["# Artifact validation report",""] ; ok=True
    for name in required:
        p=OUT/name; exists=p.is_file(); ok &= exists
        if exists and p.suffix==".csv":
            try: rows=len(pd.read_csv(p,low_memory=False)); detail=f"rows={rows}, sha256={sha256_file(p)}"
            except Exception as e: ok=False; detail=f"ERROR {e}"
        elif exists: detail=f"sha256={sha256_file(p)}"
        else: detail="MISSING"
        lines.append(f"- `{name}`: {'PASS' if exists else 'FAIL'}; {detail}")
    lines += ["",f"Overall: **{'PASS' if ok else 'FAIL'}**"]
    text="\n".join(lines); (OUT/"ARTIFACT_VALIDATION_REPORT.md").write_text(text); return ok,text


def run_pipeline():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True); ap.add_argument("--workers",type=int,default=4); args=ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    test_argv=[sys.executable,"-B","-m","pytest","analysis/dynamic_profile_eda_v1_1/scripts/test_v1_1.py","-q","-p","no:cacheprovider"]
    test_command=shlex.join(test_argv)
    started=time.time(); commands=[invocation_command()]; command_cwd=str(Path.cwd().resolve())
    pre_repo=repository_state(); pre_hash={k:recursive_hashes(v) for k,v in PROTECTED.items()}; workspace_before=workspace_hashes_excluding_v1_1()
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps({
        "analysis_id":"dynamic_profile_eda_v1_1","task_status":"running_preflight",
        "started_at_utc":datetime.fromtimestamp(started,timezone.utc).isoformat(),"commands":commands,
        "command_working_directory":command_cwd,
        "repository_pre":pre_repo,"protected_hashes_before":pre_hash,
        "outside_v1_1_workspace_hashes_before":workspace_before,
    },indent=2,default=str))
    if not CHARTER.is_file():
        raise SystemExit(f"HARD STOP: required project charter missing: {CHARTER}")
    if sha256_file(ASSEMBLER)!=ASSEMBLER_SHA: raise SystemExit("HARD STOP: canonical assembler hash mismatch")

    from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp
    raw_hashes=dp.raw_file_sha256s(args.data_dir)
    if raw_hashes!=EXPECTED_RAW_HASHES: raise SystemExit(f"HARD STOP: raw hashes differ: {raw_hashes}")
    raw=dp.read_raw_tables(args.data_dir); canonical=dp.assemble_order_base(raw)
    if len(raw["orders"])!=99441 or len(canonical)!=96470: raise SystemExit("HARD STOP: source sample count/schema differs")
    frame=build_all_placed(raw,canonical,dp.REGION)
    audit_endpoint=frame.order_delivered_customer_date.max()
    if audit_endpoint != pd.Timestamp("2018-10-17 13:22:46"):
        raise SystemExit(f"HARD STOP: audit endpoint proxy differs: {audit_endpoint}")
    if frame.order_delivered_customer_date.notna().sum()!=96476: raise SystemExit("HARD STOP: eventual-delivery count differs")
    mapped=frame[frame.in_canonical].set_index("order_id")
    reference=canonical.set_index("order_id")
    for all_col,canonical_col in [("main_seller_id","main_seller_id"),("n_unique_sellers","n_unique_sellers"),("total_price","total_price"),("total_freight_value","total_freight_value"),("main_seller_state","main_seller_state"),("customer_state","customer_state")]:
        left=mapped.loc[reference.index,all_col]; right=reference[canonical_col]
        if pd.api.types.is_numeric_dtype(right):
            equal=np.isclose(pd.to_numeric(left),pd.to_numeric(right),equal_nan=True)
        else:
            equal=left.astype("string").fillna("<MISSING>").eq(right.astype("string").fillna("<MISSING>"))
        if not bool(np.all(equal)):
            raise SystemExit(f"HARD STOP: all-placed audit mapping differs from canonical assembler for {all_col}")

    engine_validation=validate_fast_engine_parity(frame,audit_endpoint)

    rec=reconcile_sample(frame)
    six=rec[rec.customer_delivery_observed&~rec.in_canonical]
    if len(six)!=6 or not six.order_status.eq("canceled").all(): raise SystemExit("HARD STOP: six-order reconciliation unexplained")
    rec.to_csv(OUT/"CANONICAL_SAMPLE_RECONCILIATION.csv",index=False)
    curves,quants,components=maturity_outputs(frame,audit_endpoint); curves.to_csv(OUT/"MATURITY_CURVES_UNCONDITIONAL.csv",index=False); quants.to_csv(OUT/"MATURITY_QUANTILES_CONDITIONAL.csv",index=False); components.to_csv(OUT/"MATURITY_COMPONENT_COUNTS.csv",index=False)
    quality=data_quality(frame,canonical); quality.to_csv(OUT/"DATA_QUALITY_AUDIT_V1_1.csv",index=False); duration_diagnostics(frame).to_csv(OUT/"PROCESS_DURATION_DIAGNOSTICS.csv",index=False)
    interval,first,completion_last,common_last=derive_intervals(frame,audit_endpoint); interval.to_csv(OUT/"SNAPSHOT_INTERVAL_AUDIT.csv",index=False)
    scheme_b_maturity_by_purchase_age(frame,common_last).to_csv(OUT/"SCHEME_B_MATURITY_BY_PURCHASE_AGE.csv",index=False)
    daily=all_placed_daily(frame); daily.to_csv(OUT/"DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv",index=False)
    clusters,phases=calendar_clusters(daily); clusters.to_csv(OUT/"HRD_EVENT_CLUSTERS.csv",index=False); phases.to_csv(OUT/"HRD_EVENT_PHASES.csv",index=False)
    hrd,overlap=hrd_outputs(frame,daily,clusters); hrd.to_csv(OUT/"HRD_FEASIBILITY_V1_1.csv",index=False); overlap.to_csv(OUT/"HRD_DEFINITION_OVERLAP.csv",index=False)

    snapshots=pd.date_range(first,completion_last,freq="D"); rank_days=set(snapshots[snapshots.day == 1])
    part_dir=Path(tempfile.mkdtemp(prefix="dynamic_profile_v1_1_",dir=None))
    snapshot_values=snapshots.astype(str).to_numpy(); chunk_days=14
    chunks=[snapshot_values[i:i+chunk_days] for i in range(0,len(snapshot_values),chunk_days)]
    jobs=[(i,list(x),str(part_dir)) for i,x in enumerate(chunks) if len(x)]
    ctx=mp.get_context("fork")
    with ctx.Pool(processes=args.workers,initializer=snapshot_worker_init,initargs=(frame,audit_endpoint,rank_days)) as pool:
        pool.map(fast_chunk_worker,jobs)
        rank_jobs=[(1000+i,[str(day.date())],str(part_dir)) for i,day in enumerate(sorted(rank_days))]
        pool.map(reference_rank_worker,rank_jobs)
    slicing=combine_parts(part_dir,"slicing",OUT/"SLICING_SCHEME_COMPARISON_V1_1.csv",["snapshot_date","target","window_days","scheme","lag_days","granularity"])
    support=combine_parts(part_dir,"support",OUT/"ENTITY_GRANULARITY_SUPPORT_V1_1.csv",["snapshot_date","target","window_days","scheme","lag_days","granularity"])
    coverage=combine_parts(part_dir,"coverage",OUT/"FUTURE_PROFILE_COVERAGE.csv",["snapshot_date","target","window_days","scheme","lag_days","granularity","future_horizon_days"])
    rank=combine_parts(part_dir,"rank",OUT/"ENTITY_RANK_AGREEMENT_ALL_GRANULARITIES.csv",["snapshot_date","target","window_days","granularity","scheme_a","lag_a_days","scheme_b","lag_b_days","entity_statistic","support_threshold"])
    shutil.rmtree(part_dir)

    # Hard semantic validations before reporting.
    bc=slicing[slicing.scheme.isin(["B","C"])]
    if not (bc.cohort_total_orders_all_placed >= bc.mature_asof_orders).all(): raise SystemExit("HARD STOP: B/C denominator excludes unresolved orders")
    if not curves.conditioning.eq("all_placed_orders").all() or not set(quants.conditioning)=={"all_placed_orders","conditional_on_eventual_observation"}: raise SystemExit("HARD STOP: maturity concepts not separated")
    if coverage.total_future_placed_orders.max()<=0: raise SystemExit("HARD STOP: future coverage unavailable")
    if daily.order_count.sum()!=99441: raise SystemExit("HARD STOP: HRD does not use all placed orders")

    figures=make_figures(); write_reports(rec)
    required=["EDA_V1_1_PROTOCOL.md","EDA_V1_1_FROZEN_CONFIG.json","CORRECTION_LOG.md","V1_V1_1_DIFF_SUMMARY.md","DATA_DICTIONARY_V1_1.md","CANONICAL_SAMPLE_RECONCILIATION.csv","DATA_QUALITY_AUDIT_V1_1.csv","MATURITY_CURVES_UNCONDITIONAL.csv","MATURITY_QUANTILES_CONDITIONAL.csv","MATURITY_COMPONENT_COUNTS.csv","PROCESS_DURATION_DIAGNOSTICS.csv","SCHEME_B_MATURITY_BY_PURCHASE_AGE.csv","SLICING_SCHEME_COMPARISON_V1_1.csv","ENTITY_GRANULARITY_SUPPORT_V1_1.csv","FUTURE_PROFILE_COVERAGE.csv","ENTITY_RANK_AGREEMENT_ALL_GRANULARITIES.csv","DAILY_MARKETPLACE_SUMMARY_ALL_PLACED.csv","HRD_FEASIBILITY_V1_1.csv","HRD_EVENT_CLUSTERS.csv","HRD_EVENT_PHASES.csv","HRD_DEFINITION_OVERLAP.csv","SNAPSHOT_INTERVAL_AUDIT.csv","EDA_RESULTS_SUMMARY.md","EDA_RESULTS_SUMMARY_ZH.md","BLOCKERS.md"]
    required += sorted(str(Path("figure_sources")/Path(v["source_csv"]).name) for v in figures.values())
    required += sorted(str(Path("figures")/name) for name in figures)
    artifact_ok,_=validate_artifacts(required)
    post_hash={k:recursive_hashes(v) for k,v in PROTECTED.items()}; protected_ok=pre_hash==post_hash
    if not protected_ok: raise SystemExit("HARD STOP: protected V1 or Phase2A files changed")
    post_repo=repository_state()
    csvs={str(p.relative_to(OUT)):{"rows":len(pd.read_csv(p,low_memory=False)),"sha256":sha256_file(p)} for p in sorted(OUT.rglob("*.csv"))}
    sources={str(p.relative_to(OUT)):sha256_file(p) for p in sorted((OUT/"scripts").glob("*.py"))}
    manifest={"analysis_id":"dynamic_profile_eda_v1_1","task_status":"pipeline_generated_pending_tests" if artifact_ok else "blocked","started_at_utc":datetime.fromtimestamp(started,timezone.utc).isoformat(),"pipeline_generated_at_utc":datetime.now(timezone.utc).isoformat(),"runtime_seconds":time.time()-started,
        "repository_pre":pre_repo,"repository_post":post_repo,"outside_v1_1_workspace_hashes_before":workspace_before,"project_charter":{"path":str(CHARTER),"sha256":sha256_file(CHARTER),"read_only":True},"raw_file_paths":{k:str(Path(args.data_dir)/v) for k,v in dp.RAW_FILES.items()},"raw_file_hashes":raw_hashes,"canonical_assembler":{"path":str(ASSEMBLER),"sha256":sha256_file(ASSEMBLER),"matched":True},
        "protected_hashes_before":pre_hash,"protected_hashes_after":post_hash,"protected_byte_preservation_passed":protected_ok,"source_code_hashes":sources,"config_hashes":{"EDA_V1_1_FROZEN_CONFIG.json":sha256_file(OUT/"EDA_V1_1_FROZEN_CONFIG.json")},"csv_artifacts":csvs,"figures":figures,"commands":commands,"command_working_directory":command_cwd,
        "environment":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":plt.matplotlib.__version__,"workers":args.workers,"python_dont_write_bytecode":True,"mplconfigdir":os.environ["MPLCONFIGDIR"]},"snapshot_engine":{"production":{"slicing_support_coverage":"fast_engine.compute_snapshot_outputs","rank":"run_eda_v1_1.process_snapshot reference path"},"chunk_days":chunk_days,"reference":"run_eda_v1_1.process_snapshot","parity_validation":engine_validation,"rejected_candidate":"fast rank output failed frozen 1e-11 real-data parity; tolerance not relaxed"},
        "snapshot_intervals":{"completion":{"first":str(first),"last":str(completion_last),"days":len(snapshots)},"purchase_comparison":{"first":str(first),"common_last":str(common_last)},"rank_audit_cadence":"month-start daily snapshots"},
        "sample_counts":{"raw_orders":len(frame),"customer_delivery_observed":int(frame.order_delivered_customer_date.notna().sum()),"canonical_delivered":len(canonical),"six_status_inconsistent":len(six)},
        "reconciliation_counts":rec.deterministic_reconciliation_reason.value_counts().to_dict(),"warnings":["audit endpoint is a proxy","rank agreement evaluated on frozen month-start snapshots to bound artifact size; daily support/coverage retained"],"blockers":["no authoritative censoring date","no predictive validation","negative process durations unresolved for later modelling"],
        "predictive_models_fitted":False,"final_profile_choice_made":False,"risk_level_selected":False,"thesis_modified":False,"phase2a_reclassified":False,"git_commit_created":False,
        "tests":{"command":test_command,"status":"pending_post_pipeline"}}
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2,default=str))

    # The task cannot be marked complete until the full deterministic suite passes.
    test_env=os.environ.copy(); test_env["PYTHONDONTWRITEBYTECODE"]="1"; test_env.setdefault("MPLCONFIGDIR",".cache/dynamic-profile-v1-1-mpl")
    test_run=subprocess.run(test_argv,cwd=ROOT,env=test_env,text=True,capture_output=True)
    test_log=(f"COMMAND: {test_command}\nRETURN_CODE: {test_run.returncode}\n\nSTDOUT\n{test_run.stdout}\nSTDERR\n{test_run.stderr}")
    (OUT/"TEST_RESULTS.txt").write_text(test_log)
    commands.append(test_command)
    def pytest_count(label: str) -> int:
        match=re.search(rf"(\d+) {label}",test_run.stdout)
        return int(match.group(1)) if match else 0
    passed_count=pytest_count("passed"); failed_count=pytest_count("failed"); skipped_count=pytest_count("skipped")
    xfailed_count=pytest_count("xfailed"); xpassed_count=pytest_count("xpassed"); error_count=pytest_count("error(?:s)?")
    duration_matches=re.findall(r"in ([0-9.]+)s",test_run.stdout)
    final_hash={k:recursive_hashes(v) for k,v in PROTECTED.items()}
    workspace_after=workspace_hashes_excluding_v1_1()
    tests_passed=test_run.returncode==0 and passed_count>0 and bool(duration_matches)
    preservation_passed=pre_hash==final_hash
    outside_workspace_unchanged=workspace_before==workspace_after
    manifest["commands"]=commands
    manifest["protected_hashes_after_tests"]=final_hash
    manifest["protected_byte_preservation_passed"]=preservation_passed
    manifest["outside_v1_1_workspace_hashes_after"]=workspace_after
    manifest["outside_v1_1_workspace_unchanged"]=outside_workspace_unchanged
    manifest["tests"]={
        "command":test_command,
        "status":"passed" if tests_passed else "failed",
        "return_code":test_run.returncode,
        "collected":passed_count+failed_count+skipped_count+xfailed_count+xpassed_count,
        "passed":passed_count,
        "failed":failed_count,
        "skipped":skipped_count,
        "xfailed":xfailed_count,
        "xpassed":xpassed_count,
        "errors":error_count,
        "duration_seconds":float(duration_matches[-1]) if duration_matches else None,
        "full_log":"TEST_RESULTS.txt",
        "full_log_sha256":sha256_file(OUT/"TEST_RESULTS.txt"),
    }
    write_final_reports(manifest,rec,slicing,support,coverage,rank,daily,hrd,interval,quality)
    final_artifact_ok,_=validate_artifacts(required+["TEST_RESULTS.txt"])
    manifest["artifact_validation"]={"status":"passed" if final_artifact_ok else "failed","report":"ARTIFACT_VALIDATION_REPORT.md","report_sha256":sha256_file(OUT/"ARTIFACT_VALIDATION_REPORT.md")}
    manifest["task_status"]="correction_only_completed" if tests_passed and preservation_passed and outside_workspace_unchanged and final_artifact_ok else "blocked"
    if not tests_passed:
        manifest["blockers"].append("full deterministic pytest suite failed; see TEST_RESULTS.txt")
    if not preservation_passed:
        manifest["blockers"].append("protected V1 or Phase2A bytes changed during tests")
    if not outside_workspace_unchanged:
        manifest["blockers"].append("a tracked or nonignored untracked file outside the authorised V1.1 workspace changed during execution")
    if not final_artifact_ok:
        manifest["blockers"].append("artifact validation failed")
    if not (tests_passed and preservation_passed and outside_workspace_unchanged and final_artifact_ok):
        write_final_reports(manifest,rec,slicing,support,coverage,rank,daily,hrd,interval,quality)
        final_artifact_ok,_=validate_artifacts(required+["TEST_RESULTS.txt"])
        manifest["artifact_validation"]={"status":"passed" if final_artifact_ok else "failed","report":"ARTIFACT_VALIDATION_REPORT.md","report_sha256":sha256_file(OUT/"ARTIFACT_VALIDATION_REPORT.md")}
    manifest["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
    manifest["runtime_seconds"]=time.time()-started
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2,default=str))
    print(json.dumps({"status":manifest["task_status"],"runtime_seconds":manifest["runtime_seconds"],"snapshots":len(snapshots),"slicing_rows":len(slicing),"support_rows":len(support),"coverage_rows":len(coverage),"rank_rows":len(rank)}))
    if manifest["task_status"] != "correction_only_completed":
        raise SystemExit(1)


def _write_runtime_blocked_manifest(exc: BaseException) -> None:
    manifest_path=OUT/"RUN_MANIFEST.json"
    try:
        manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    except Exception:
        manifest={}
    # Preserve richer evidence already written by a post-test hard stop.
    manifest.setdefault("analysis_id","dynamic_profile_eda_v1_1")
    manifest["task_status"]="blocked"
    manifest["blocking_condition"]=f"{type(exc).__name__}: {exc}"
    manifest["blocked_at_utc"]=datetime.now(timezone.utc).isoformat()
    manifest.setdefault("commands",[invocation_command()])
    manifest.setdefault("command_working_directory",str(Path.cwd().resolve()))
    try:
        manifest["repository_post_failure"]=repository_state()
    except Exception as state_error:
        manifest["repository_post_failure_error"]=repr(state_error)
    try:
        protected_after={k:recursive_hashes(v) for k,v in PROTECTED.items()}
        manifest["protected_hashes_after_failure"]=protected_after
        if "protected_hashes_before" in manifest:
            manifest["protected_byte_preservation_passed"]=manifest["protected_hashes_before"]==protected_after
    except Exception as hash_error:
        manifest["protected_hash_failure_error"]=repr(hash_error)
    try:
        outside_after=workspace_hashes_excluding_v1_1()
        manifest["outside_v1_1_workspace_hashes_after_failure"]=outside_after
        if "outside_v1_1_workspace_hashes_before" in manifest:
            manifest["outside_v1_1_workspace_unchanged"]=manifest["outside_v1_1_workspace_hashes_before"]==outside_after
    except Exception as workspace_error:
        manifest["outside_workspace_hash_failure_error"]=repr(workspace_error)
    manifest["partial_artifacts"]={
        str(p.relative_to(OUT)):{"bytes":p.stat().st_size,"sha256":sha256_file(p)}
        for p in sorted(OUT.rglob("*")) if p.is_file() and p != manifest_path
    }
    manifest["predictive_models_fitted"]=False
    manifest["final_profile_choice_made"]=False
    manifest["thesis_modified"]=False
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str))
    failure_text="# V1.1 blocked\n\nThe deterministic pipeline stopped without a completed interpretive summary.\n\n- Exact failure: `"+str(manifest["blocking_condition"]).replace("`","'")+"`\n- Evidence: `RUN_MANIFEST.json` and any persisted `TEST_RESULTS.txt`.\n"
    (OUT/"BLOCKERS.md").write_text(failure_text)
    if (OUT/"EDA_RESULTS_SUMMARY.md").exists():
        (OUT/"EDA_RESULTS_SUMMARY.md").write_text(failure_text)
    if (OUT/"EDA_RESULTS_SUMMARY_ZH.md").exists():
        (OUT/"EDA_RESULTS_SUMMARY_ZH.md").write_text("# V1.1 已阻塞\n\n确定性流水线已停止；未生成完成版解释性摘要。详见 `RUN_MANIFEST.json` 与 `TEST_RESULTS.txt`（如存在）。\n")


def main():
    try:
        run_pipeline()
    except SystemExit as exc:
        if exc.code in (None,0):
            raise
        _write_runtime_blocked_manifest(exc)
        raise
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        _write_runtime_blocked_manifest(exc)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__=="__main__":
    main()
