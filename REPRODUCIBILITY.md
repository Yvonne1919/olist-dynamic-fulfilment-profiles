# Reproduction scope and limitations

This source-and-aggregate-evidence export preserves the research algorithms.
It does not claim that the private execution environment, governance history,
or multi-gigabyte intermediate stores have been published.

## What can be checked now

| Check | Inputs included? | Release preparation |
|---|---|---|
| Existing synthetic/contract unit tests | Yes | Executed on the exported code |
| Chapter 4/5/6 figure rendering | Yes, aggregate inputs | Executed without model fitting |
| Curated table-source emission | Yes | Executed; not statistical re-estimation |
| Canonical order assembly/audit | Raw CSVs must be downloaded | Read-only audit passed on local original data; no artifacts written |
| RQ1 statistical tables | Raw CSVs must be downloaded | Public wrapper checked; no full-data fit |
| Selected daily profiles | Raw CSVs must be downloaded | Public wrapper/import checked; full reconstruction not run |
| Complete candidate selection, confirmation and all order models | No, original orchestration has additional dependencies | Source included, not end-to-end validated here |

## Scientific entry-point inventory

All paths below are relative to the release root. The study directories retain
their original names; the public wrapper is `scripts/reproduce.py`.

- `analysis/profile_pivot_phase2a/scripts/data_pipeline.py`: canonical delivered
  order assembly and target availability. Retained as a shared dependency;
  no Phase 2A gate model, STOP/PASS decision or obsolete experiment is rerun.
- `analysis/dynamic_profile_eda_v1_1/scripts/core.py` and `fast_engine.py`:
  all-placed observability, support, maturity and historical-window mechanics.
  `run_eda_v1_1.py` is the original full driver. Its private preflight metadata
  is absent. The predecessor `dynamic_profile_eda_v1/scripts/run_eda.py` is
  retained only because the corrected driver imports its helpers.
- `analysis/rq1_speed_reliability_review_v1/scripts/rq1_data.py` and
  `rq1_stats.py`: exact sample, review selection, HC1 observational models,
  contrasts and sensitivities. The public RQ1 wrapper reuses them without
  pretending to regenerate the historical governance receipt. Existing source
  and raw-input hash checks remain active.
- `analysis/dynamic_profile_profile_validation_v1/scripts/profile_core.py`,
  `profile_selection.py`, `selected_daily.py`, `fast_stability.py`, and
  `run_profile_validation.py`: original candidate estimators, selection,
  frozen daily construction, future-process validation and stability logic.
  The public profile wrapper reuses `generate_selected_daily_profiles` only;
  it neither searches candidates nor fabricates a new confirmation.
- `analysis/order_breach_severity_v1/`: **historical current-context
  implementation / dependency; not the main Chapter 6 estimand**.
  `order_modeling.py` is imported by direct-promise and model-family code;
  `order_profiles.py` supports the all-mature sensitivity. Existing contract
  tests also import `order_features.py`, `order_experiment.py`, `order_io.py`,
  `order_preflight.py`, `order_reporting.py` and `run_order_experiment.py`.
  These modules, the test file, package initializer and two exact exported
  configuration/selection dependencies are retained unchanged. The historical
  full runner is not a public reproduction target. Eight aggregate CSVs are
  retained as secondary inputs to `direct_reporting.py`, and
  `ORDER_BREACH_PAIRED_DIFFERENCES.csv` remains an exact upstream dependency
  named in `DIRECT_FROZEN_CONFIG.json`; other current-context result CSVs
  are excluded.
- `analysis/direct_promise_profile_extension_v1/scripts/direct_experiment.py`:
  direct-promise LR/XGBoost breach and linear/XGBoost Q50/Q90 implementation.
  The public `direct` wrapper uses its already frozen model selection and
  exact-frame validator. It does not re-tune against later outcomes.
- `analysis/direct_model_family_robustness_v1/scripts/robustness_experiment.py`
  and `recovered_severity_model_source.py`: executed Random Forest, leaf-weighted
  Quantile Random Forest and Lognormal Ridge comparison. The shared historical
  RF factory in `src/models/classification.py` is a necessary dependency,
  not an invitation to rerun the historical classifier leaderboard.
- `analysis/all_mature_history_sensitivity_v1/scripts/sensitivity_core.py`
  and `order_sensitivity.py`: executed all-mature-history sensitivity;
  not a replacement for the selected 90-day specification.

## Why the complete raw-data pipeline is not a quick-start command

The original full runners require preflight receipts and protected-source
inventories referring to private controls and historical outputs. These are
deliberately excluded. Their checks are not disabled or replaced with fake
success receipts. Some exported configuration files consequently differ in
byte hash from their original receipts even though their scientific settings
are unchanged.

The missing large-artifact chain is:

```text
Original raw CSVs
  -> selected profile daily store and parent table
  -> exact profile join + current-order features + retrospective audit strata
  -> Order V1 working/ORDER_MODEL_FRAME.csv.gz
  -> direct-promise predictions
  -> model-family and all-mature-history sensitivity results
```

The profile join validates original profile-store hashes and selection
receipts. Simply renaming the public wrapper's Parquet files will not satisfy
that contract. A complete public replay would need a separately verified
portable orchestration layer covering serialization, profile joins, audit
strata and historical receipt replacement. That layer has **not** been
implemented or validated in this packaging task. Source code and aggregate
outputs for inspection are included instead of advertising broken full-run
commands.

If the exact Order V1 frame is already available locally, the direct wrapper
accepts it with `--model-frame`; its SHA-256 must match the value in
`DIRECT_FROZEN_CONFIG.json`. Output is written to a fresh `outputs/direct/`
directory. The wrapper's `--help` documents this advanced option. No row-level
artifact should be committed to the public repository.

## Source and environment provenance

`RELEASE_FILES.csv` records source and export SHA-256 values for each public
file. Release-only changes are identified separately from byte-identical
copies. Original hashes embedded in study files remain historical references;
they are not claims that a path-redacted release copy has its original hash.
No algorithm, target, split, selected hyperparameter, numerical aggregate or
evidence label was changed for packaging.

Python pins describe the inspected research environment. A fresh online
installation, another OS, parallel numerical reproducibility, a complete model
rerun and exact PDF byte equality are not certified. R package versions are
observed versions, not a dependency lock; font/rendering changes can alter PDF
bytes without changing numerical inputs. The data provider's original release
must be obtained and attributed separately.

The original research Git history was not copied. Publish only this export,
never the research repository or its history.
