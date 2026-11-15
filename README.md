# Lifestyle and Cause of Death in Medicolegal Deaths (NMDID Metadata)

**Separating biological signal from investigation-driven recording artifacts**

**Course:** CSE 499A Capstone Project — North South University

## Overview
Does knowing how a person lived (drinking, tobacco, substance use, medical history, marital status, socioeconomic background, body mass index) help estimate how they died in medicolegal investigations? And how much of that apparent signal reflects behaviour and biology, versus **which fields investigators chose to record** for different suspected causes of death?

This repository implements a leak-free, fully reproducible statistical and machine-learning pipeline on structured decedent metadata from the **New Mexico Decedent Image Database (NMDID)**: 1,450 adult decedents, de-duplicated from 1,500 records across three NMDID data carts.

## Research Questions
- **RQ1, incremental value:** how much discrimination do lifestyle (L) and medical-history (H) features add beyond age and sex?
- **RQ2, factors and stability:** which factors carry the signal after adjustment for age and sex, and is it stable across sex and age bands?
- **RQ3, signal versus recording artifact:** how much of the gain comes from *whether* a field was recorded rather than *what* it says?

## Three-Level Analytic Hierarchy
1. **Level 1, broad cause (n = 1,380):** `natural_cardiac` (373), `drug_alcohol` (500), `injury` (507). Excluded: `cardiac_conflict` (39, cardiac cause with non-natural manner) and `other` causes (31).
2. **Level 2, intent within injury deaths (n = 487):** `accident` (215), `homicide` (157), `suicide` (115).
3. **Level 3, weak negative control within homicides (n = 137):** stab wound (107) versus gunshot wound (30).

## Main Findings (HistGB, 5×5 repeated stratified CV, 95% bootstrap CIs)
| Question | Result |
|---|---|
| RQ1 | Level 1 macro AUC 0.688 (age + sex), 0.754 (+ lifestyle), 0.776 (+ medical history). Level 2 stays weak (0.524 to 0.594). |
| RQ3 | Indicators of *which fields were recorded* alone give about 60% of the lifestyle gain. Recorded values, after the blank pattern is destroyed, give about 35% (a lower bound). In the 243 complete cases the gain is +0.010 (−0.039 to +0.062). Family-interview-only fields add +0.001; investigator-recorded fields add +0.061. |
| Controls | Labels-permuted models sit at chance (see `permutation_summary.csv`). Lifestyle adds nothing detectable to gunshot-versus-stab discrimination. Results hold with zip code added, alternative BMI definitions, class weights, and deaths after 2017 excluded. |

Full numbers, interpretation and limitations: [`results/methods_draft.md`](results/methods_draft.md). Reporting checklist: [`results/reporting_checklist.md`](results/reporting_checklist.md).

## Design Safeguards
- **Leakage control.** Seventeen administrative and post-mortem variables (`FORBIDDEN_INPUTS` in `src/config.py`) are blocked from every model and checked at build time. Their association with the outcome is reported in `leakage_check.csv`.
- **Cadaver BMI.** The main analysis uses autopsy weight and length with an implausibility filter only. Blanking BMI for decomposition would copy the forbidden `decomposition_score` into the missing flag (Cramér's V 0.74), so that version is a sensitivity analysis (`bmi_flag_audit.csv`).
- **Cart design.** Each cart was extracted for one cause of death, so cart is nearly collinear with the Level-1 outcome. The 43 decedents found in two carts have identical records in all eight linked tables (`cart_invariance_check.csv`), so extraction selects people without altering their metadata. This limitation is discussed explicitly.
- **Out-of-fold everything.** Imputation and scaling are fitted inside training folds. Permutation importance and TreeSHAP are computed on held-out folds only. No hyper-parameters are tuned.
- **Transparent outcome parsing.** Every manner-of-death string and the rule that classified it is listed in `intent_parsing_audit.csv`.
- **Privacy.** Only aggregate tables are published. Crosstab cells below 5 are suppressed. Person-level data and trained models are git-ignored.

## Repository Structure
```text
nmdid-cause-of-death/
├── src/
│   ├── config.py                     # Paths, seed (42), outcome definitions, leakage blocklist, fast mode
│   ├── model_utils.py                # Shared feature sets, models, repeated CV, AUC, paired bootstrap
│   ├── 01_build_dataset.py           # De-duplication, cohort, 87 model features + RQ3/sensitivity extras
│   ├── 02_data_audit.py              # Cohort flow, Table 1, coverage, leakage, BMI and cart audits
│   ├── 03_baseline_models.py         # Level-1 baseline incl. missing-flags-only model
│   ├── 04_rq1_nested_models.py       # RQ1: M0 → M1 → M2, three models, calibration, reverse entry order
│   ├── 05_rq2_factors.py             # RQ2: univariate + age/sex-adjusted tests, OOF importance, OOF SHAP, age bands
│   ├── 06_rq3_missingness.py         # RQ3: encoding regimes, complete cases, family vs shared, NOK, medications
│   ├── 07_level3_negative_control.py # Level 3 control, permutation test, robustness
│   ├── 08_paper_outputs.py           # Paper tables 1-4 and auto-filled methods/results draft
│   └── 09_train_demo_model.py        # Demo models for the app (git-ignored output)
├── app/app.py                        # Streamlit demo and results dashboard
├── results/
│   ├── tables/                       # Aggregate CSV tables (safe to publish)
│   ├── figures/                      # 9 PNG figures
│   ├── methods_draft.md              # Methods, results and limitations draft (generated)
│   └── reporting_checklist.md        # TRIPOD+AI mapping
├── data/                             # raw/ and processed/ (git-ignored, NMDID data-use agreement)
└── requirements.txt
```

## Data Access
Individual-level NMDID records are governed by the NMDID Data Use Agreement and must never be committed or uploaded to public services. After obtaining authorised access from [NMDID](https://nmdid.unm.edu/), place the eight CSV tables of each cart in `data/raw/cart2960/`, `data/raw/cart2966/` and `data/raw/cart2975/`.

## Reproduction
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional quick smoke test of the whole pipeline (1 CV repeat, 50 bootstrap draws)
$env:NMDID_FAST = "1"; python -W ignore -u src/03_baseline_models.py; Remove-Item Env:NMDID_FAST

python -W ignore -u src/01_build_dataset.py
python -W ignore -u src/02_data_audit.py
python -W ignore -u src/03_baseline_models.py
python -W ignore -u src/04_rq1_nested_models.py
python -W ignore -u src/05_rq2_factors.py
python -W ignore -u src/06_rq3_missingness.py
python -W ignore -u src/07_level3_negative_control.py
python -W ignore -u src/08_paper_outputs.py
python -W ignore -u src/09_train_demo_model.py
streamlit run app/app.py
```
The full run takes about 25 minutes on a dual-core laptop when scripts 03/05/06 and 04/07 run as two parallel streams (07 needs 04's output; 08 needs all). Every model uses one core (`n_jobs=1`).

## Key Output Tables (`results/tables/`)
| File | Content |
|---|---|
| `cohort_flow.csv`, `table1_by_class.csv` | Sample flow and Level-1 characteristics |
| `coverage_by_class.csv`, `missingness_tests.csv` | Field coverage by class and chi-square tests |
| `leakage_check.csv`, `bmi_flag_audit.csv`, `cart_invariance_check.csv` | Leakage and design audits |
| `intent_by_cause.csv`, `intent_parsing_audit.csv` | Outcome parsing (small cells suppressed) |
| `rq1_metrics.csv`, `rq1_gains.csv`, `rq1_per_class_auc.csv`, `rq1_reverse_order.csv` | RQ1 |
| `rq2_univariate_level{1,2}.csv`, `rq2_adjusted_level{1,2}.csv` | RQ2 unadjusted and age/sex-adjusted tests |
| `rq2_rf_importance_by_sex.csv`, `rq2_shap_importance.csv`, `rq2_age_bands.csv` | RQ2 out-of-fold importance and subgroup models |
| `rq3_artifact.csv`, `rq3_gain_shares.csv` | RQ3 |
| `level3_negative_control.csv`, `permutation_summary.csv`, `permutation_check.csv`, `robustness.csv` | Controls and robustness |
| `paper_table1_cohort.csv` … `paper_table4_rq3_and_controls.csv` | Formatted manuscript tables |
