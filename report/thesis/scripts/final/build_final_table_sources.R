#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
})

command_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", command_args[grep("^--file=", command_args)][1])
script_dir <- dirname(normalizePath(file_arg))
root <- normalizePath(file.path(script_dir, "../../../.."))
out <- file.path(root, "report/thesis/table_sources/final")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write_source <- function(data, name) write_csv(data, file.path(out, name), na = "")

t1 <- tribble(
  ~literature_stream, ~established_idea, ~olist_boundary, ~thesis_response, ~verified_keys,
  "Delivery forecasting and promise setting", "Conditional delivery distributions require a service rule, cost or reliability constraint before becoming a customer promise.", "Live inventory, queue, carrier and network state and Olist's original promise objective are undocumented.", "Evaluate historical process profiles as a post-promise risk-assessment layer around the recorded estimate.", "salari2022; preethi2021; raj2024",
  "Breach incidence and delay magnitude", "Late probability and delay length are related but non-equivalent outcomes.", "Final breach and severity are observable only after customer delivery.", "Model missed-date probability and breached-order positive-lateness Q50/Q90 separately.", "mueller2025; awaysheh2021; steinberg2023",
  "Delivery performance and customer outcomes", "Longer and promise-relative delivery failures are associated with ratings and later behaviour in the cited settings.", "Olist reviews are selected, timing-limited and not pure logistics-satisfaction labels.", "Use observed low reviews as observational customer-relevance evidence, with timing sensitivity.", "akturk2022; ravula2023; nguyen2018; rao2011; rao2014",
  "Provider and entity profiling", "Sparse entity rates require support, uncertainty and shrinkage-aware interpretation.", "Seller and state-OD histories have uneven support; case mix remains unresolved.", "Construct point-in-time raw/shrinkage candidates and retain support, cold start, uncertainty and freshness.", "normand1997; spiegelhalter2012",
  "Temporal validation and reliability", "Later-origin evaluation, proper scores, reliability and quantile coverage answer different questions.", "Profile definitions and terminal regimes are not external or prospective holdouts.", "Use development selection, subsequent process confirmation, six later cohorts and separate terminal stress.", "tashman2000; cerqueira2020; gneiting2007scoring; gneiting2011; ovadia2019; yao2022"
)
write_source(t1, "T1_literature_positioning.csv")

t2 <- tribble(
  ~population_or_object, ~n, ~period_or_rule, ~role_and_timing,
  "All placed orders", 99441, "2016-2018 release", "Unconditional availability and status denominator; includes non-delivered statuses.",
  "Canonical delivered orders", 96470, "Observed final delivery", "Calendar-day D, P, E and delivered-outcome descriptive population.",
  "Selected reviewed delivered orders", 95824, "One deterministic usable review", "Primary RQ1 observational population; 4,653 reviews precede delivery.",
  "Reviews at/after delivery", 91171, "Timing sensitivity", "RQ1c sensitivity; changes the estimand and may itself be selected.",
  "Frozen final order frame", 91254, "Purchase-time predictors plus observed final target", "Development/later/terminal order experiment population.",
  "Order development", 38477, "Earlier purchases", "Model and calibration selection only.",
  "Later monthly cohorts", 40270, "2018-01 to 2018-06", "Six chronological evaluations; 3,477 breached orders.",
  "Terminal stress", 12507, "2018-07-01 to 2018-08-29", "Separate stress evidence; 601 breached orders."
)
write_source(t2, "T2_populations_feature_timing.csv")

t3 <- tribble(
  ~stage_or_profile, ~reader_definition, ~entity_scheme_estimator, ~window_lag_kappa, ~confirmation_or_role, ~order_payload,
  "Selection chain", "3,024 candidates to 2,031 evidence-eligible to 97 non-dominated to 16 selected to 14 next-stage eligible to 5 representatives", "Development-only Pareto and tie-break rules", "30/60/90-day candidates; all 16 selected used 90 days", "Five frozen representatives; four process profiles in main ladders", "Not applicable",
  "S1", "Seller pre-handoff duration level", "seller_id; Scheme C; P1", "90d; lag 14d; no binary kappa", "Strongly confirmed; 6/6 favourable months", "score, log1p support, cold start, posterior SE, freshness",
  "S2", "Long pre-handoff-duration risk above 7.1645d", "seller_id; Scheme A; P1", "90d; lag 0; kappa 10", "Strongly confirmed; 6/6", "same metadata payload",
  "R1", "State-OD transit duration level", "state_od; Scheme A; P0", "90d; lag 0; no kappa", "Partially confirmed; 6/6", "same metadata payload",
  "R2", "State-OD transit tail risk above 14.1079d", "state_od; Scheme A; P1", "90d; lag 0; kappa 10", "Partially confirmed; 6/6", "same metadata payload",
  "M5 endpoint", "Secondary state-OD historical final-breach profile", "state_od; Scheme A; P1", "90d; lag 0; kappa 100", "Secondary comparison only", "same metadata payload"
)
write_source(t3, "T3_profile_selection_frozen_blocks.csv")

t4 <- tribble(
  ~layer, ~data_business_challenge, ~candidate_methods, ~evaluation_selection, ~fitted_role_settings,
  "Dynamic profiles", "Target-specific maturity, sparse history, granularity/coverage, cold start, temporal change", "Schemes A/C; 30/60/90d; seller and geographic entities; level/tail targets; P0/P1/P2", "Support, uncertainty, coverage, proper loss/log-MAE, rank/lift, daily stability, Pareto selection, subsequent process confirmation", "S1/S2/R1/R2 main process representatives; M5 endpoint secondary; continuous score plus metadata",
  "Missed-date probability", "Rare target, probability reliability and temporal shift", "Promise-only plus seller, state-OD or both; L2 Logistic Regression, Random Forest and XGBoost", "Six paired months; log loss/Brier primary; calibration, high-support and score-contribution guards", "Logistic C=10; Random Forest 250 trees/minimum leaf 20; XGBoost rate .03/depth 3/4 trees",
  "Conditional positive lateness", "Breached-only, right-tailed integer-day severity", "Promise-only plus seller, state-OD or both; Linear Quantile, leaf-weighted Quantile Random Forest, XGBoost Quantile and Lognormal Ridge", "Q50/Q90 pinball skill, high-support guards, pooled coverage and terminal robustness", "Linear alpha .0001/.01; QRF 100 trees/minimum leaf 10; XGBoost Quantile 1/5 trees; Lognormal Ridge alpha 1"
)
write_source(t4, "T4_method_selection_fitted_specifications.csv")

t5 <- tribble(
  ~profile, ~process_target, ~primary_metric, ~development_primary, ~confirmation_month_median, ~favourable_months, ~future_separation, ~label, ~strongest_limitation,
  "S1", "Seller pre-handoff level", "Reference minus candidate log-MAE", 0.067132006471, 0.073534165507, "6/6", "25-anchor log-MAE improvement 0.072964; weighted Spearman 0.514; lift 1.79", "Strongly confirmed", "Sparse seller support and deterministic seller attribution",
  "S2", "Seller pre-handoff tail risk", "Reference minus candidate log loss", 0.043056941648, 0.063627153350, "6/6", "25-anchor log-loss improvement 0.067064; Brier 0.014831; Spearman 0.337; lift 3.55", "Strongly confirmed", "Historical association, not intrinsic seller quality",
  "R1", "State-OD transit level", "Reference minus candidate log-MAE", 0.107441123503, 0.130329098265, "6/6", "25-anchor log-MAE improvement 0.138549; Spearman 0.897; lift 2.13", "Partially confirmed", "High-support guard failed; geographic proxy only",
  "R2", "State-OD transit tail risk", "Reference minus candidate log loss", 0.061095361605, 0.101947847841, "6/6", "25-anchor log-loss improvement 0.104064; Brier 0.036984; Spearman 0.675; lift 2.15", "Partially confirmed", "Magnitude/high-support guards failed; not a carrier route"
)
write_source(t5, "T5_standalone_profile_transfer.csv")

t6 <- tribble(
  ~comparison, ~logistic_monthly_delta, ~logistic_months, ~logistic_label, ~logistic_pooled_delta, ~xgb_monthly_delta, ~xgb_months, ~xgb_label, ~xgb_pooled_delta,
  "Issued promise to current context (M0 to M1)", "-0.002676 / -0.000574", "3/6", "No profile label", "+0.010735 / +0.000221", "-0.000880 / -0.000123", "4/6", "No profile label", "+0.000765 / +0.000012",
  "Seller pre-handoff block (M1 to M2)", "-0.003630 / -0.000410", "6/6", "Supported", "-0.006315 / -0.000642", "-0.000141 / -0.000012", "3/6", "Mixed", "+0.000573 / +0.000058",
  "State-OD transit block (M1 to M3)", "-0.009341 / -0.000967", "3/6", "Mixed", "-0.015668 / -0.000992", "-0.000041 / +0.000018", "3/6", "Not-supported", "-0.001775 / -0.000181",
  "Combined process block (M1 to M4)", "-0.011808 / -0.001245", "4/6", "Supported", "-0.021884 / -0.001865", "-0.000906 / -0.000128", "4/6", "Supported", "-0.003044 / -0.000303",
  "Secondary endpoint (M4 to M5)", "+0.001339 / +0.000177", "2/6", "No formal label", "+0.001943 / +0.000253", "+0.000734 / +0.000111", "2/6", "No formal label", "+0.001409 / +0.000233",
  "Two event interactions (M4 to M4E)", "+0.000430 / +0.000140", "1/6", "No formal label", "+0.000813 / +0.000137", "+0.000085 / -0.000001", "3/6", "No formal label", "-0.000356 / -0.000034"
)
write_source(t6, "T6_missed_date_probability_results.csv")

t6_direct <- tribble(
  ~family, ~profile_block, ~median_delta_log_loss, ~median_delta_brier, ~both_improved_months, ~calibration_guard, ~evidence_label,
  "L2 Logistic Regression", "Seller", -0.00415653020117, -0.000474113022350, "6/6", "Pass", "Supported",
  "L2 Logistic Regression", "State-OD", -0.0126256584508, -0.000678203167734, "3/6", "Pass", "Mixed",
  "L2 Logistic Regression", "Both", -0.0188355042404, -0.00146902299902, "5/6", "Pass", "Supported",
  "Random Forest", "Seller", 0.00193643020825, 0.000363597265665, "2/6", "Pass", "Not-supported",
  "Random Forest", "State-OD", -0.0110668832408, -0.00105165332621, "5/6", "Pass", "Supported",
  "Random Forest", "Both", -0.00761967829778, 0.0000580000102071, "3/6", "Fail", "Not-supported",
  "XGBoost", "Seller", -0.000144657998041, -0.0000196016143433, "5/6", "Pass", "Supported",
  "XGBoost", "State-OD", -0.00197620878246, -0.000261235940820, "5/6", "Fail", "Mixed",
  "XGBoost", "Both", -0.00229967244447, -0.000293696109202, "5/6", "Fail", "Mixed"
)
write_source(t6_direct, "T6_direct_breach_model_families.csv")

t7 <- tribble(
  ~family_quantile, ~q2_seller_skill_months_label, ~q3_state_od_skill_months_label, ~q4_combined_skill_months_label, ~pooled_coverage_or_boundary,
  "Linear Q50", "+0.001055; 4/6; Not-supported (high-support guard)", "+0.002392; 4/6; Supported", "+0.003554; 5/6; Supported", "Pooled skill vs Q1: -0.000807 / +0.004403 / +0.004068",
  "XGBoost Q50", "-0.001241; 1/6; Not-supported", "-0.001726; 3/6; Not-supported", "+0.000032; 3/6; Not-supported", "No XGBoost Q50 block supported",
  "Linear Q90", "-0.000072; 3/6; Not-supported", "-0.000160; 2/6; Not-supported", "-0.001426; 3/6; Not-supported", "Later-pooled coverage Q1-Q4: 0.8760-0.8735",
  "XGBoost Q90", "+0.025481; 4/6; Supported", "+0.026592; 5/6; Supported", "+0.008561; 3/6; Not-supported", "Coverage 0.8654-0.8772; supported blocks do not beat unconditional Q90 pooled"
)
write_source(t7, "T7_severity_skill_coverage.csv")

t8 <- tribble(
  ~rq, ~evidence_level, ~answer, ~numerical_anchor, ~design, ~interpretation,
  "RQ1a", "Observational association", "Longer actual delivery is associated with higher observed low-review probability", "Model A D7 to D16: +5.85 percentage points (95% CI 5.53 to 6.18)", "95,824 selected reviewed orders", "Observed customer association",
  "RQ1b-c", "Observational plus timing sensitivity", "Promise-relative lateness is strongly associated in the full sample but attenuates after pre-delivery reviews are excluded", "4-7 days late: +51.58 points full; +3.75 points (CI -0.33 to 7.83) at/after delivery", "95,824 full; 91,171 timing sensitivity", "Timing-sensitive observational relationship",
  "RQ2", "Standalone process evidence", "Seller profiles are Strongly confirmed and state-OD profiles Partially confirmed", "All four favourable 6/6; weighted rank correlations 0.337 to 0.897", "Development selection; Jan-Jun process confirmation", "Future-process evidence within Olist",
  "RQ2 sensitivity", "History-memory sensitivity", "All-mature histories improve support and smoothness but not future tracking", "Seller level support 31 to 71 and seller tail support 34 to 77; state-OD level/tail support 850 to 2921; all-mature favourable 1/8", "Fixed representatives; paired development and confirmation losses", "Fixed-profile history-memory sensitivity",
  "RQ3a", "Later-cohort prediction", "Profiles add direct missed-date information beyond the recorded promise but no block is Supported in all three families", "LR seller/both Supported; RF state-OD Supported; XGB seller Supported", "40,270 orders; six monthly cohorts; pooled separate", "Later-cohort predictive evidence across model families",
  "RQ3b-c", "Severity and terminal stress", "Severity is model-specific and upper-tail reliability remains incomplete", "QRF seller/both Supported at Q50/Q90; Lognormal state-OD Q90 Supported; pooled Q90 coverage 83.1%-88.6%", "3,477 later and 601 terminal breached orders", "Model-dependent severity signal; nominal Q90 coverage unmet"
)
write_source(t8, "T8_rq_evidence_summary.csv")

cat(sprintf("Created 9 final table-source CSVs in %s\n", out))
