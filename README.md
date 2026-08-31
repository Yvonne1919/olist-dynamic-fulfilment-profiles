# Dynamic fulfilment profiles for delivery-promise risk

Code and selected aggregate evidence for **Predicting Missed Delivery Dates
and Delay Severity in E-Commerce Using Dynamic Seller and Origin–Destination
Profiles**, an Imperial College London MLDS MSc research project using Olist.

## Motivation

A promised delivery date conveys an estimate, but does not describe all the
uncertainty around fulfilment. This project constructs maturity-aware,
point-in-time seller pre-handoff and state origin–destination (OD) transit
profiles from previously observable outcomes. It evaluates future process
separation and asks whether these profiles add information about missed dates
and positive delay severity. Observed reviews provide separate, observational
evidence about customer relevance.

Olist's recorded estimate is treated as a black-box customer-facing promise;
purchase time is the available proxy for issuance. The project does not
reconstruct Olist's undocumented promise-generation system or establish an
optimal promise policy.

## Research questions

1. **Customer relevance of speed and reliability.** How are actual delivery
   duration and performance relative to the promised date associated with
   observed order reviews?
2. **Dynamic profile construction and quality.** Among the seller and
   origin–destination representations observable in the data, which profile
   definitions—across process target, entity granularity, history window and
   estimator—provide sufficient support, future process separation and
   temporal stability?
3. **Order-level prediction and incremental profile value.** How accurately and reliably can later orders be forecast to miss their recorded delivery date and, conditional on a miss, how large may the delay be? For the profiles retained from RQ2, what additional information do they provide beyond the recorded delivery estimate?

RQ1 distinguishes absolute speed, duration-conditioned promise-relative
performance, and review-timing sensitivity. RQ3 distinguishes breach
probabilities, conditional positive-lateness Q50/Q90, and temporal robustness.

## Repository map

```text
scripts/                        Public reproduction and lightweight-test entry points
src/                            Shared order/review, feature, metric and RF utilities
analysis/
  profile_pivot_phase2a/         Canonical order assembler only (historical path)
  dynamic_profile_eda_v1/        Required predecessor utility, not current EDA evidence
  dynamic_profile_eda_v1_1/      Corrected maturity and entity-support analysis
  rq1_speed_reliability_review_v1/  Review construction and observational models
  dynamic_profile_profile_validation_v1/  Profiles, selection and future-process tests
  order_breach_severity_v1/      Historical current-context implementation / dependency
  direct_promise_profile_extension_v1/    Direct-promise/profile comparison
  direct_model_family_robustness_v1/      Executed RF/QRF/lognormal family comparison
  all_mature_history_sensitivity_v1/      Executed history-window sensitivity
report/thesis/
  scripts/final/                Final Chapter 4/5/6 figure and table-source builders
  figure_sources/               Compact inputs used by the final figures
  table_sources/                Persisted, curated table-source CSVs
  tables/final/                 Final curated Chapter 4/5 LaTeX tables
tests/                          Shared synthetic unit tests
data/README.md                  Original data source and local placement
RELEASE_MANIFEST.md             Chapter-to-code/evidence map and release boundaries
RELEASE_FILES.csv               Per-file release hashes and source provenance
```

Historical directory names are retained so imports and scientific provenance
remain traceable. They do not revive superseded experiments.
`analysis/order_breach_severity_v1/` is a **historical current-context
implementation / dependency; not the main Chapter 6 estimand**. Its retained
modules support shared modelling, profile joins and existing tests; nine
aggregate tables preserve secondary reporting and frozen-input dependencies
of the direct-promise implementation. Current-context results are not a main
reproduction target.
Chapter 6 evaluates the recorded-estimate baseline and retained-profile
increments directly, without the historical current-context feature block.

## Data

Download the original
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from Olist's Kaggle page and extract it into `data/olist_data/`. See
[data/README.md](data/README.md) for the nine required filenames and input-version
checks. Raw data are **not redistributed**. Neither row-level predictions nor
the large daily entity-profile stores are included.

## Setup

Run all commands from the root of this exported repository. The recorded
Python environment is CPython 3.12.13; dependency pins describe that observed
environment, not a newly validated cross-platform lockfile.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B scripts/verify_release.py
.venv/bin/python -B scripts/run_lightweight_tests.py
```

The lightweight suite explicitly selects existing synthetic/contract tests.
It excludes tests that require raw data, omitted large artifacts or private
historical execution receipts. It is not a new full experiment validation.
On systems requiring an OpenMP runtime for XGBoost, install the appropriate
system runtime before importing XGBoost.

## Rebuild final figures without raw data

The figure builders read included aggregate tables and do not fit models.
R 4.4.1 and Ghostscript were used for the release smoke check. Required R
packages and observed versions are in [environment/R-packages.csv](environment/R-packages.csv).
Install Ghostscript through your system package manager; `gs` must be on PATH.

```bash
Rscript -e 'install.packages(c("dplyr", "tidyr", "readr", "ggplot2", "patchwork", "scales", "stringr", "tibble"), repos="https://cloud.r-project.org")'
Rscript report/thesis/scripts/final/plot_rq1_chapter4_traceability.R
Rscript report/thesis/scripts/final/plot_ch5_ch8_integration.R
Rscript report/thesis/scripts/final/build_final_table_sources.R
```

PDFs appear under `report/thesis/images/` (ignored by Git). The last command
re-emits **curated CSV table sources**. Any historical current-context
comparison is secondary and separate from the main direct-promise estimand.
It does not recompute metrics
or generate the hand-authored LaTeX tables. The three current Chapter 4/5
LaTeX tables are supplied as source; no automatic `.tex` regeneration is claimed.
The complete thesis, literature PDFs and private drafts are not part of this
code release.

## Data-dependent reproduction

After downloading the exact data version, the canonical order audit is read-only:

```bash
.venv/bin/python -B analysis/profile_pivot_phase2a/scripts/data_pipeline.py --audit-only --data-dir data/olist_data
```

The following opt-in commands call the existing scientific functions. They
write only to new subdirectories of ignored `outputs/` and refuse overwrites.
They were **not run on the full dataset during release preparation**.

```bash
.venv/bin/python -B scripts/reproduce.py rq1 --data-dir data/olist_data --output outputs/rq1
.venv/bin/python -B scripts/reproduce.py profiles --data-dir data/olist_data --output outputs/profiles
```

`rq1` fits the frozen observational models and exports their statistical tables.
`profiles` reconstructs the previously selected daily profiles on the original
date grid; it does **not** repeat candidate selection or the full standalone
validation programme. It can require substantial memory and disk space. Its
Parquet serialization is not the historical compressed CSV format and must not
be relabelled as the historical hash-verified profile store.

The direct-promise evaluation wrapper additionally needs the exact original
Order V1 model-frame artifact. It verifies its frozen hash before evaluation.
That row-level artifact is deliberately absent from this release:

```bash
.venv/bin/python -B scripts/reproduce.py direct --help
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the missing-artifact chain,
original entry points and limits. **This is not a turnkey raw-data-to-all-results
release.** The private preflight/finalization commands retained in study source
files are historical orchestration, not working public quick-start commands.

## Interpretation boundaries

- Breach means actual customer delivery **calendar date** later than the
  promised calendar date. Q50/Q90 severity concerns positive lateness among
  breached orders, not the all-delivered signed-error distribution.
- Seller pre-handoff duration spans payment approval to carrier handoff. It
  is not pure seller processing or intrinsic seller quality. State-OD is a
  geographic transit proxy, not an observed carrier route.
- Historical labels enter profiles only when observable before the snapshot;
  current-order delivery, review and payment outcomes are not predictors.
- Hyperparameters and calibration choices are selected in earlier development
  folds. January–June 2018 contains six sequential out-of-time monthly
  evaluations. At each origin, the model is refitted on all eligible earlier
  data using the frozen settings; this is not a static fitted-model holdout.
  The same calendar period also contributes profile confirmation, so it is
  not a thesis-wide independent external validation stage. July–August is
  separate terminal-regime stress evidence.
- Review associations are observational and materially sensitive to review
  timing. Neither these associations nor model scores establish causal
  customer, commercial, policy or production benefits.
- Direct-promise/profile and current-context/profile comparisons are distinct
  information sets. Their metrics and evidence labels must not be pooled or
  treated as interchangeable.

No code license is supplied: the author has not yet selected one. Dataset
terms remain those of the original publisher. Publication does not itself
grant a separate code license.
