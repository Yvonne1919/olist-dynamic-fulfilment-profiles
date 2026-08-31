#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/dynamic_profile_eda_v1"
FIG = OUT / "figures"
sys.path.insert(0, str(ROOT))
from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp

TARGETS = {
 "final_breach": ("late_delivery", "final_breach_available_at"),
 "positive_late_days": ("positive_late_days", "positive_late_days_available_at"),
 "handling": ("post_approval_handling", "seller_handling_available_at"),
 "transit": ("transit_time", "transit_available_at"),
}
WINDOWS=[30,60,90]; LAGS=[7,14,21,30,45,60]; SUPPORT=[5,10,20,50]

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def q(s,p):
 x=pd.to_numeric(s,errors="coerce").dropna(); return float(x.quantile(p)) if len(x) else np.nan

def mdtable(frame):
 f=frame.copy()
 cols=[str(c) for c in f.columns]
 def esc(v): return str(v).replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(esc(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])

def add_semantics(d):
 d=d.copy()
 d["final_breach_available_at"]=d.order_delivered_customer_date
 d["positive_late_days_available_at"]=d.order_delivered_customer_date
 d["transit_available_at"]=d[["order_delivered_carrier_date","order_delivered_customer_date"]].max(axis=1,skipna=False)
 d["seller_handling_available_at"]=d[["order_approved_at","order_delivered_carrier_date"]].max(axis=1,skipna=False)
 d["carrier_handoff_available_at"]=d.order_delivered_carrier_date
 d["customer_delivery_available_at"]=d.order_delivered_customer_date
 d["purchase_date"]=d.order_purchase_timestamp.dt.normalize()
 d["seller_customer_region"]=d.main_seller_id.astype("string")+" -> "+d.customer_region.astype("string")
 d["seller_customer_state"]=d.main_seller_id.astype("string")+" -> "+d.customer_state.astype("string")
 d["seller_state_od"]=d.main_seller_id.astype("string")+" -> "+d.route_state.astype("string")
 d["route_region_entity"]=d.seller_region.astype("string")+" -> "+d.customer_region.astype("string")
 d["route_state_entity"]=d.main_seller_state.astype("string")+" -> "+d.customer_state.astype("string")
 z1=d.main_seller_zip.astype("Int64").astype("string").str.zfill(5).str[:2]
 z2=d.customer_zip_code_prefix.astype("Int64").astype("string").str.zfill(5).str[:2]
 d["route_zip2_entity"]=z1+" -> "+z2
 return d

ENTITIES={"seller_id":"main_seller_id","seller_x_customer_region":"seller_customer_region","seller_x_customer_state":"seller_customer_state","seller_x_state_od":"seller_state_od","region_od":"route_region_entity","state_od":"route_state_entity","zip2_od":"route_zip2_entity"}

def maturity(raw_orders,d,daily):
 ages=[0,3,7,14,21,30,45,60,90]; rows=[]; lagrows=[]
 full=raw_orders[["order_id","order_status","order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date"]].copy()
 for c in full.columns[2:]: full[c]=pd.to_datetime(full[c],errors="coerce")
 extra=d[["order_id","is_single_seller"]].copy(); full=full.merge(extra,on="order_id",how="left")
 full["is_eventually_delivered"]=full.order_delivered_customer_date.notna()
 full["seller_handling_available_at"]=full[["order_approved_at","order_delivered_carrier_date"]].max(axis=1,skipna=False)
 full["transit_available_at"]=full[["order_delivered_carrier_date","order_delivered_customer_date"]].max(axis=1,skipna=False)
 avail={"carrier_handoff":"order_delivered_carrier_date","customer_delivery":"order_delivered_customer_date","final_breach":"order_delivered_customer_date","positive_late_days":"order_delivered_customer_date","handling_duration":"seller_handling_available_at","transit_duration":"transit_available_at"}
 hrd=daily.set_index("date")["both_top10"].to_dict(); full["regime"]=full.order_purchase_timestamp.dt.normalize().map(hrd).map({True:"candidate_HRD",False:"BAU"}).fillna("BAU")
 groups={"all_placed":pd.Series(True,index=full.index),"eventually_delivered":full.is_eventually_delivered,"single_seller":full.is_single_seller.eq(True),"multi_seller":full.is_single_seller.eq(False),"BAU":full.regime.eq("BAU"),"candidate_HRD":full.regime.eq("candidate_HRD")}
 for target,col in avail.items():
  age=(full[col]-full.order_purchase_timestamp).dt.total_seconds()/86400
  for gn,gm in groups.items():
   denom=int(gm.sum()); a=age[gm]
   for x in ages: rows.append({"target":target,"sample":gn,"age_days":x,"n_denominator":denom,"n_available":int(a.le(x).sum()),"cumulative_availability":float(a.le(x).sum()/denom) if denom else np.nan,"n_right_censored_or_unavailable":int(a.isna().sum())})
   valid=np.sort(a.dropna().to_numpy())
   for p in [.9,.95,.975,.99]: lagrows.append({"target":target,"sample":gn,"availability_fraction":p,"age_days_required":float(np.quantile(valid,p,method="higher")) if len(valid) else np.nan,"n_denominator":denom,"n_eventually_observed":len(valid),"eventual_observed_fraction":len(valid)/denom if denom else np.nan})
 pd.DataFrame(rows).to_csv(OUT/"MATURITY_CURVES_BY_AGE.csv",index=False); pd.DataFrame(lagrows).to_csv(OUT/"MATURITY_LAG_SUMMARY.csv",index=False)
 pd.DataFrame({"target":np.repeat(list(avail),len(full)),"purchase_to_availability_age_days":np.concatenate([((full[c]-full.order_purchase_timestamp).dt.total_seconds()/86400).to_numpy() for c in avail.values()])}).to_csv(OUT/"FIGURE_SOURCE_PURCHASE_TO_AVAILABILITY_AGE.csv",index=False)

def daily_market(d):
 g=d.groupby("purchase_date",observed=True)
 x=g.agg(order_count=("order_id","size"),gmv=("total_price","sum"),freight_value=("total_freight_value","sum"),active_sellers=("main_seller_id","nunique"),active_routes=("route_state","nunique"),breach_rate=("late_delivery","mean"),positive_late_mean=("positive_late_days","mean"),positive_late_p90=("positive_late_days",lambda s:q(s,.9)),handling_mean=("post_approval_handling","mean"),handling_p90=("post_approval_handling",lambda s:q(s,.9)),transit_mean=("transit_time","mean"),transit_p90=("transit_time",lambda s:q(s,.9))).reset_index(names="date")
 for metric in ["order_count","gmv"]:
  for pct,label in [(.9,"top10"),(.95,"top5")]: x[f"{metric.split('_')[0]}_{label}"]=x[metric].ge(x[metric].quantile(pct))
 x["both_top10"]=x.order_top10 & x.gmv_top10; x["both_top5"]=x.order_top5 & x.gmv_top5
 x["high_volume_cluster"]=((x.order_top10 != x.order_top10.shift()).cumsum().where(x.order_top10)).astype("Int64")
 phase=np.full(len(x),"BAU",object)
 for i in np.flatnonzero(x.order_top10):
  phase[max(0,i-1):i]="pre_event"; phase[i]="event"; phase[i+1:min(len(x),i+4)]="post_1_3d"
 x["event_phase"]=phase; x.to_csv(OUT/"DAILY_MARKETPLACE_SUMMARY.csv",index=False); return x

def describe_slice(s,entity,target,value,avail,t,w,scheme,lag):
 n=len(s); valid=s[value].notna(); late=s.late_delivery.dropna(); pos=s.positive_late_days
 sup=s.loc[valid].groupby(entity,observed=True).size() if entity in s else pd.Series(dtype=float)
 return {"snapshot_date":t,"window_days":w,"scheme":scheme,"lag_days":lag,"target":target,"granularity":entity,"n_orders":n,"n_valid_outcomes":int(valid.sum()),"retrospective_maturity_pct":float(valid.mean()) if n else np.nan,"event_rate":float(late.mean()) if len(late) else np.nan,"positive_late_mean":float(pos.mean()),"positive_late_p50":q(pos,.5),"positive_late_p90":q(pos,.9),"positive_late_p95":q(pos,.95),"handling_mean":float(s.post_approval_handling.mean()),"handling_p50":q(s.post_approval_handling,.5),"handling_p90":q(s.post_approval_handling,.9),"transit_mean":float(s.transit_time.mean()),"transit_p50":q(s.transit_time,.5),"transit_p90":q(s.transit_time,.9),"availability_age_median":q((s[avail]-s.order_purchase_timestamp).dt.total_seconds()/86400,.5),"availability_age_p90":q((s[avail]-s.order_purchase_timestamp).dt.total_seconds()/86400,.9),"n_entities":int(s[entity].nunique()),"seen_entity_rate":np.nan,"cold_start_rate":np.nan,**{f"fraction_orders_support_ge_{k}":float(s[entity].map(sup).ge(k).mean()) if n else np.nan for k in SUPPORT}}

def slicing_and_support(d,snaps):
 comp=[]; supp=[]; cover=[]; rank=[]
 for t in snaps:
  for target,(value,avail) in TARGETS.items():
   eligible=d[avail].lt(t)&d[value].notna()
   for w in WINDOWS:
    schemes=[("A",0,d.loc[eligible & d[avail].ge(t-pd.Timedelta(days=w))]) ,("B",0,d.loc[eligible & d.order_purchase_timestamp.ge(t-pd.Timedelta(days=w)) & d.order_purchase_timestamp.lt(t)])]
    schemes += [("C",L,d.loc[eligible & d.order_purchase_timestamp.ge(t-pd.Timedelta(days=L+w)) & d.order_purchase_timestamp.lt(t-pd.Timedelta(days=L))]) for L in LAGS]
    scheme_rates={}
    for scheme,L,s in schemes:
     scheme_rates[(scheme,L)]=s.groupby("main_seller_id")[value].agg(["mean","size"])
     for gn,e in ENTITIES.items():
      r=describe_slice(s,e,target,value,avail,t,w,scheme,L); r["granularity"]=gn
      future=d[d.order_purchase_timestamp.ge(t)&d.order_purchase_timestamp.lt(t+pd.Timedelta(days=7))]
      hist=set(s[e].dropna()); seen=future[e].isin(hist)
      r["seen_entity_rate"]=float(seen.mean()) if len(future) else np.nan; r["cold_start_rate"]=1-r["seen_entity_rate"] if len(future) else np.nan; comp.append(r)
      gs=s.groupby(e,observed=True).agg(support=(value,"count"),events=("late_delivery","sum"),last_available=(avail,"max"),active_days=("purchase_date","nunique"),volume=("order_id","size")).reset_index()
      if len(gs):
       total=gs.support.sum(); vs=gs.volume.sort_values(ascending=False)
       rr={"snapshot_date":t,"window_days":w,"scheme":scheme,"lag_days":L,"target":target,"granularity":gn,"active_entities":len(gs),"support_p10":q(gs.support,.1),"support_p25":q(gs.support,.25),"support_median":q(gs.support,.5),"support_p75":q(gs.support,.75),"support_p90":q(gs.support,.9),"order_weighted_coverage":float(s[e].notna().mean()),"event_count_mean":float(gs.events.mean()),"event_count_p90":q(gs.events,.9),"pct_entities_active_one_day":float(gs.active_days.eq(1).mean()),"median_active_days":q(gs.active_days,.5),"profile_freshness_days":q((t-gs.last_available).dt.total_seconds()/86400,.5)}
       for k in SUPPORT: rr[f"entities_ge_{k}"]=int(gs.support.ge(k).sum()); rr[f"pct_entities_ge_{k}"]=float(gs.support.ge(k).mean())
       for pct in [1,5,10]: rr[f"top_{pct}pct_volume_share"]=float(vs.head(max(1,int(np.ceil(len(vs)*pct/100)))).sum()/vs.sum())
       for horizon in [7,30]:
        fut=d[d.order_purchase_timestamp.ge(t)&d.order_purchase_timestamp.lt(t+pd.Timedelta(days=horizon))]; rr[f"cold_start_{horizon}d"]=float((~fut[e].isin(set(gs[e]))).mean()) if len(fut) else np.nan
       supp.append(rr)
    a=scheme_rates.get(("A",0));
    for scheme,L in [("B",0)]+[("C",x) for x in LAGS]:
     b=scheme_rates.get((scheme,L)); j=a.join(b,lsuffix="_a",rsuffix="_b",how="inner") if a is not None and b is not None else pd.DataFrame()
     j=j[(j.size_a>=5)&(j.size_b>=5)] if len(j) else j
     if len(j)>=3 and j.mean_a.nunique()>1 and j.mean_b.nunique()>1:
      ra=j.mean_a.rank(method="average").to_numpy(); rb=j.mean_b.rank(method="average").to_numpy(); rho=float(np.corrcoef(ra,rb)[0,1])
     else: rho=np.nan
     rank.append({"snapshot_date":t,"target":target,"window_days":w,"comparison_scheme":scheme,"lag_days":L,"granularity":"seller_id","n_common_entities_support_ge_5":len(j),"spearman_raw_rate_rank":rho})
  future=d[d.order_purchase_timestamp.ge(t)&d.order_purchase_timestamp.lt(t+pd.Timedelta(days=7))]
  cover.append({"snapshot_date":t,"future_7d_orders":len(future),**{f"{gn}_seen_rate_90d_A":float(future[e].isin(set(d.loc[d.final_breach_available_at.lt(t)&d.final_breach_available_at.ge(t-pd.Timedelta(days=90)),e])).mean()) if len(future) else np.nan for gn,e in ENTITIES.items()}})
 pd.DataFrame(comp).to_csv(OUT/"SLICING_SCHEME_COMPARISON.csv",index=False); pd.DataFrame(supp).to_csv(OUT/"ENTITY_GRANULARITY_SUPPORT.csv",index=False); pd.DataFrame(cover).to_csv(OUT/"DAILY_SNAPSHOT_COVERAGE.csv",index=False); pd.DataFrame(rank).to_csv(OUT/"ENTITY_RANK_AGREEMENT.csv",index=False)

def quality(d,raw):
 rows=[]
 def add(check,n,den=len(d),scope="canonical_delivered",detail=""): rows.append({"scope":scope,"check":check,"count":int(n),"denominator":int(den),"percentage":100*n/den if den else np.nan,"detail":detail})
 add("duplicate_order_ids",d.order_id.duplicated().sum()); add("multi_seller_orders",d.n_unique_sellers.gt(1).sum()); add("missing_seller_id",d.main_seller_id.isna().sum()); add("missing_route_state",d.route_state.isna().sum()); add("missing_distance",d.distance_km.isna().sum()); add("negative_purchase_to_carrier",d.purchase_to_carrier.lt(0).sum()); add("negative_handling",d.post_approval_handling.lt(0).sum()); add("negative_transit",d.transit_time.lt(0).sum())
 for c in ["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]: add("missing_"+c,d[c].isna().sum())
 for c in ["purchase_to_carrier","post_approval_handling","transit_time","promise_error_days"]: add("extreme_"+c+"_above_p999",d[c].gt(d[c].quantile(.999)).sum(),detail=f"raw p99.9={d[c].quantile(.999):.3f}; retained")
 add("non_delivered_or_unresolved",raw.order_status.ne("delivered").sum(),len(raw),"all_raw_orders","never coded as non-breach/zero severity")
 add("delivered_status_missing_customer_delivery",(raw.order_status.eq("delivered")&raw.order_delivered_customer_date.isna()).sum(),len(raw),"all_raw_orders")
 pd.DataFrame(rows).to_csv(OUT/"DATA_QUALITY_AUDIT.csv",index=False)
 cols=["promise_error_days","late_delivery","positive_late_days","purchase_to_carrier","post_approval_handling","transit_time","promised_delivery_days","distance_km","total_price","total_freight_value","n_items","n_unique_products","n_unique_sellers"]
 rr=[]
 for month,g in [("ALL",d)]+list(d.groupby("purchase_month",observed=True)):
  for c in cols: rr.append({"purchase_month":month,"variable":c,"n":g[c].count(),"missing":g[c].isna().sum(),"mean":g[c].mean(),"sd":g[c].std(),"p01":q(g[c],.01),"p10":q(g[c],.1),"p50":q(g[c],.5),"p90":q(g[c],.9),"p99":q(g[c],.99)})
 pd.DataFrame(rr).to_csv(OUT/"OUTCOME_FEATURE_DISTRIBUTIONS.csv",index=False)

def hrd(d,daily):
 rows=[]; defs=["order_top10","order_top5","gmv_top10","gmv_top5","both_top10","both_top5"]
 dm=d.merge(daily[["date"]+defs],left_on="purchase_date",right_on="date",how="left")
 for flag in defs:
  hd=daily[daily[flag]]; sub=dm[dm[flag]]; bau=dm[~dm[flag]]
  for gn,e in ENTITIES.items():
   sp=sub.groupby(e).size(); rows.append({"definition":flag,"granularity":gn,"n_hrd_days":len(hd),"n_orders":len(sub),"pct_orders":len(sub)/len(dm),"n_event_clusters":int(((hd.date.diff().dt.days.ne(1)).sum())) if len(hd) else 0,"active_entities":sp.size,"pct_entities_support_ge_5":float(sp.ge(5).mean()) if len(sp) else np.nan,"pct_entities_support_ge_10":float(sp.ge(10).mean()) if len(sp) else np.nan,"pct_entities_support_ge_20":float(sp.ge(20).mean()) if len(sp) else np.nan,"breach_rate_hrd":sub.late_delivery.mean(),"breach_rate_bau":bau.late_delivery.mean(),"positive_late_mean_hrd":sub.positive_late_days.mean(),"positive_late_mean_bau":bau.positive_late_days.mean(),"handling_mean_hrd":sub.post_approval_handling.mean(),"handling_mean_bau":bau.post_approval_handling.mean(),"transit_mean_hrd":sub.transit_time.mean(),"transit_mean_bau":bau.transit_time.mean(),"separate_profile_support_feasible_ge20":bool(sp.ge(20).mean()>=.25) if len(sp) else False})
 pd.DataFrame(rows).to_csv(OUT/"HRD_FEASIBILITY.csv",index=False)
 ov=[]
 for a in defs:
  for b in defs: ov.append({"definition_a":a,"definition_b":b,"days_intersection":int((daily[a]&daily[b]).sum()),"days_union":int((daily[a]|daily[b]).sum()),"jaccard":float((daily[a]&daily[b]).sum()/(daily[a]|daily[b]).sum())})
 pd.DataFrame(ov).to_csv(OUT/"HRD_OVERLAP.csv",index=False)

def figures():
 plt.style.use("seaborn-v0_8-whitegrid"); FIG.mkdir(exist_ok=True)
 m=pd.read_csv(OUT/"MATURITY_CURVES_BY_AGE.csv"); a=pd.read_csv(OUT/"FIGURE_SOURCE_PURCHASE_TO_AVAILABILITY_AGE.csv")
 for target,g in m[m['sample'].eq('all_placed')].groupby('target'): plt.plot(g.age_days,g.cumulative_availability,label=target)
 plt.legend(fontsize=7,ncol=2); plt.xlabel('days since purchase'); plt.ylabel('cumulative availability'); plt.tight_layout(); plt.savefig(FIG/'01_target_maturity_curves.png',dpi=160); plt.close()
 a.boxplot(column='purchase_to_availability_age_days',by='target',rot=30,showfliers=False); plt.suptitle(''); plt.title('Purchase-to-availability age'); plt.tight_layout(); plt.savefig(FIG/'02_purchase_to_availability_age.png',dpi=160); plt.close()
 s=pd.read_csv(OUT/'ENTITY_GRANULARITY_SUPPORT.csv'); ss=s[(s.target=='final_breach')&(s.scheme=='A')&(s.window_days==90)]
 ss.groupby('granularity').order_weighted_coverage.mean().sort_values().plot.barh(); plt.xlabel('mean order-weighted coverage'); plt.tight_layout(); plt.savefig(FIG/'03_coverage_by_granularity.png',dpi=160); plt.close()
 s.groupby(['granularity','window_days']).support_median.mean().unstack().plot.bar(); plt.ylabel('median support (snapshot mean)'); plt.tight_layout(); plt.savefig(FIG/'04_support_by_window.png',dpi=160); plt.close()
 s[s.scheme=='A'].groupby(['granularity','window_days']).cold_start_7d.mean().unstack().plot.bar(); plt.ylabel('7-day cold-start rate'); plt.tight_layout(); plt.savefig(FIG/'05_cold_start.png',dpi=160); plt.close()
 c=pd.read_csv(OUT/'SLICING_SCHEME_COMPARISON.csv'); z=c[(c.target=='final_breach')&(c.granularity=='seller_id')&(c.window_days==90)&(((c.scheme=='A')&(c.lag_days==0))|((c.scheme=='C')&(c.lag_days==30)))]
 z.pivot(index='snapshot_date',columns='scheme',values='event_rate').plot(); plt.ylabel('breach rate'); plt.tight_layout(); plt.savefig(FIG/'06_completion_vs_lagged_rates.png',dpi=160); plt.close()
 r=pd.read_csv(OUT/'ENTITY_RANK_AGREEMENT.csv'); r[r.comparison_scheme=='C'].groupby('lag_days').spearman_raw_rate_rank.mean().plot(marker='o'); plt.ylabel('mean Spearman'); plt.tight_layout(); plt.savefig(FIG/'07_entity_rank_agreement.png',dpi=160); plt.close()
 d=pd.read_csv(OUT/'DAILY_MARKETPLACE_SUMMARY.csv',parse_dates=['date']); ax=d.plot(x='date',y='order_count',legend=False); ax.scatter(d.loc[d.both_top10,'date'],d.loc[d.both_top10,'order_count'],s=8,c='r'); plt.tight_layout(); plt.savefig(FIG/'08_daily_volume_hrd.png',dpi=160); plt.close()
 h=pd.read_csv(OUT/'HRD_FEASIBILITY.csv'); h[h.granularity=='seller_id'].set_index('definition')[['breach_rate_hrd','breach_rate_bau']].plot.bar(); plt.tight_layout(); plt.savefig(FIG/'09_bau_vs_hrd.png',dpi=160); plt.close()
 ql=pd.read_csv(OUT/'DATA_QUALITY_AUDIT.csv'); ql[ql.scope=='all_raw_orders'].plot.barh(x='check',y='count',legend=False); plt.tight_layout(); plt.savefig(FIG/'10_terminal_unresolved.png',dpi=160); plt.close()

def reports(d,raw,snaps,daily,commands):
 m=pd.read_csv(OUT/'MATURITY_LAG_SUMMARY.csv'); s=pd.read_csv(OUT/'ENTITY_GRANULARITY_SUPPORT.csv'); h=pd.read_csv(OUT/'HRD_FEASIBILITY.csv'); ql=pd.read_csv(OUT/'DATA_QUALITY_AUDIT.csv')
 lines=["# EDA results summary","","> Exploratory data-design evidence only. No profile predictive validity, method selection, or thesis claim is established.","",f"Canonical delivered orders: **{len(d):,}**. Full raw orders: **{len(raw):,}**. Snapshot interval: **{snaps.min().date()} to {snaps.max().date()}** ({len(snaps):,} daily snapshots).", "","## Maturity",mdtable(m[(m['sample']=='all_placed')&(m.availability_fraction.isin([.9,.95,.99]))]),"","Customer-delivery date is a correct availability cut for final breach/severity and transit completion, but not the least-delayed cut for handling; handling becomes available at carrier handoff once approval is known.","","## Slicing and support","Scheme A describes recent completions, Scheme B is selectively observed among recent purchases, and Scheme C trades freshness for maturity. All candidate lags/windows remain unresolved design choices.","",mdtable(s[(s.target=='final_breach')&(s.scheme=='A')].groupby(['granularity','window_days']).agg(median_support=('support_median','median'),cold_start_7d=('cold_start_7d','mean'),coverage=('order_weighted_coverage','mean')).reset_index()),"","## BAU / candidate HRD",mdtable(h[h.granularity.isin(['seller_id','state_od'])][['definition','granularity','n_hrd_days','pct_orders','pct_entities_support_ge_20','separate_profile_support_feasible_ge20']]),"","HRD labels are retrospective and cannot be used as start-of-day predictors. Separate entity-level HRD histories are sparse under many definitions.","","## Data quality and censoring",mdtable(ql),"","No unresolved order was assigned a breach or severity outcome. Negative and extreme durations remain in the raw audit and were not silently removed.","","## Boundaries","No predictive models were fitted; no profile estimator, risk level, final scheme, lag, granularity, or HRD definition was selected. No thesis, Phase 1, Results Registry, approved amendment, Phase 2A output, or decision label was modified."]
 (OUT/'EDA_RESULTS_SUMMARY.md').write_text('\n'.join(lines))
 zh=["# EDA 结果摘要","","> 本结果仅用于探索性数据设计，不证明卖家或路线画像具有预测效度。","",f"规范已交付订单：**{len(d):,}**；原始全部订单：**{len(raw):,}**；每日快照区间：**{snaps.min().date()} 至 {snaps.max().date()}**。","","客户签收时间可作为最终违约、正向迟到天数和运输阶段的保守成熟截止；卖家处理阶段应在承运交接且审批时间已知时成熟，不必等待客户签收。","","A 方案描述近期完成结果；B 方案会偏向较快成熟订单；C 方案以滞后换取更高成熟度。本任务不选择最终窗口、滞后或粒度。","","候选 HRD 标签基于事后整日订单量和 GMV，不能当作当日开始时可用的线上特征。许多实体在 HRD 子样本中的支持度不足。","","未解决订单没有被编码为准时或零严重度；负持续时间和极端值均先报告、未静默删除。","","本任务没有拟合预测模型，没有选择画像估计器或风险等级，也没有修改论文、Phase 1、结果注册表、Phase 2A 文件或冻结决策。"]
 (OUT/'EDA_RESULTS_SUMMARY_ZH.md').write_text('\n'.join(zh))
 (OUT/'BLOCKERS.md').write_text("# Blockers and unresolved design choices\n\n- The raw data provide no authoritative administrative censoring date; the last dense delivery date is an auditable proxy, not ground truth.\n- Scheme B is maturity-selected; Scheme C requires a future design choice among six lags.\n- Fine seller-route and ZIP-group entities are sparse and require an explicit later fallback/pooling policy.\n- Candidate HRD definitions are retrospective, overlap, and often provide insufficient entity-specific history.\n- Negative process durations remain data-quality anomalies; a later modelling protocol must pre-specify treatment.\n- EDA cannot establish future predictive validity or choose a profile estimator.\n")
 manifest={"analysis_id":"dynamic_profile_eda_v1","status":"completed_exploratory_eda_only","completed_at_utc":datetime.now(timezone.utc).isoformat(),"canonical_assembler":{"path":str(ROOT/'analysis/profile_pivot_phase2a/scripts/data_pipeline.py'),"sha256":sha(ROOT/'analysis/profile_pivot_phase2a/scripts/data_pipeline.py'),"phase2a_manifest_expected_sha256":"0c4cad3c99db268292253abd26e2070ccf2a286337bcfe9fb76fb3768c44ab8d","matched":True},"raw_input_paths":{k:str(Path(args.data_dir)/v) for k,v in dp.RAW_FILES.items()},"raw_input_sha256":dp.raw_file_sha256s(args.data_dir),"repository":{"commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"branch":subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),"dirty":bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())},"commands_executed":commands,"environment":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":plt.matplotlib.__version__},"samples":{"canonical_delivered":len(d),"raw_all_orders":len(raw),"raw_statuses":raw.order_status.value_counts(dropna=False).to_dict()},"snapshot_intervals":{"completion":{"first":str(snaps.min().date()),"last":str(snaps.max().date()),"n_days":len(snaps)},"purchase_comparison":{"first":str(snaps.min().date()),"last":str(snaps.max().date())}},"files_created":sorted(str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()),"phase2a_decision_unchanged":"STOP_PIVOT_AND_RETAIN_EXISTING_THESIS","predictive_models_fitted":False,"thesis_modified":False}
 (OUT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2,default=str))

def main():
 global args
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); args=ap.parse_args()
 expected=json.loads((OUT/'EDA_FROZEN_CONFIG.json').read_text())['canonical_assembler_sha256']; actual=sha(ROOT/'analysis/profile_pivot_phase2a/scripts/data_pipeline.py')
 if actual!=expected: raise SystemExit(f"STOP: assembler hash discrepancy expected={expected} actual={actual}")
 raw=dp.read_raw_tables(args.data_dir); d=add_semantics(dp.assemble_order_base(raw)); orders=raw['orders'].copy()
 for c in orders.columns:
  if c.endswith('_timestamp') or c.endswith('_at') or c.endswith('_date'): orders[c]=pd.to_datetime(orders[c],errors='coerce')
 daily=daily_market(d); maturity(orders,d,daily); quality(d,orders); hrd(d,daily)
 first=d.order_purchase_timestamp.min().normalize()+pd.Timedelta(days=90)
 counts=d.groupby(d.order_delivered_customer_date.dt.normalize()).size(); last=counts[counts.ge(100)].index.max()
 snaps=pd.date_range(first,last,freq='D'); slicing_and_support(d,snaps); figures(); reports(d,orders,snaps,daily,[f"{sys.executable} {Path(__file__).relative_to(ROOT)} --data-dir {args.data_dir}"])
 print(json.dumps({'status':'complete','canonical_orders':len(d),'raw_orders':len(orders),'snapshot_first':str(snaps.min().date()),'snapshot_last':str(snaps.max().date()),'snapshot_days':len(snaps)}))
if __name__=='__main__': main()
