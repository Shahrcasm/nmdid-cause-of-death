# Lifestyle and Cause of Death in Medicolegal Deaths (NMDID Metadata)

**Course:** CSE 499A Capstone Project — North South University  

## Overview
Does knowing how a person lived—drinking habits, tobacco use, substance use, medical history, marital status, socioeconomic background, and body mass index (BMI)—help estimate how they died in medicolegal investigations? Furthermore, how much of that predictive signal reflects true biological/behavioral risk versus **investigation-driven recording artifacts** (i.e., which fields medicolegal investigators choose to record depending on the suspected cause of death)?

This repository implements a reproducible statistical and machine learning pipeline using structured decedent metadata from the **New Mexico Decedent Image Database (NMDID)** across $N = 1,450$ adult decedents (deduplicated from 1,500 records across 3 NMDID data carts).

---

## Research Questions & Three-Level Study Design

### Research Questions
- **RQ1 (Incremental Predictive Value):** How much discriminative power do lifestyle ($L$) and medical history ($H$) features add beyond baseline demographics (`age_years` and `sex`)?
- **RQ2 (Factor-Level Associations & Subgroup Stability):** Which specific lifestyle and clinical factors drive predictions, and do these associations remain consistent across sex and age strata (`18–39`, `40–59`, `60+`) after Benjamini–Hochberg FDR correction?
- **RQ3 (Biological Signal vs. Recording Artifact):** How much of the apparent predictive performance is driven by non-random missingness (`_missing` indicator flags and next-of-kin interview availability) rather than recorded lifestyle values?

### Three-Level Analytic Hierarchy
1. **Level 1 — Broad Cause of Death ($n = 1,380$):**
   - `natural_cardiac` ($n = 373$, median age 59 [IQR 51–68], 69.4% male)
   - `drug_alcohol` ($n = 500$, median age 43 [IQR 34–52], 64.8% male)
   - `injury` ($n = 507$, median age 42 [IQR 29–56], 77.3% male)
   - *Excluded:* `cardiac_conflict` ($n = 39$, cardiac cause with non-natural manner) and `other` causes ($n = 31$).
2. **Level 2 — Manner / Intent Within Injury Deaths ($n = 487$):**
   - `accident` ($n = 215$), `homicide` ($n = 157$), and `suicide` ($n = 115$).
3. **Level 3 — Negative Control Within Homicides ($n = 137$):**
   - `Stab wound` ($n = 107$) vs. `Gunshot wound` ($n = 30$), where pre-mortem lifestyle should have minimal biological effect.

---

## Repository Structure

```text
nmdid-cause-of-death/
├── src/
│   ├── config.py                  # Paths, random seed (42), outcome definitions, leakage blocklist
│   ├── 01_build_dataset.py        # Deduplication, cohort construction, 91-feature matrix generation
│   ├── 02_data_audit.py           # Cohort flow, Table 1, missingness chi-square tests, Cramér's V audit
│   ├── 03_baseline_models.py      # 5-fold CV baseline & missing-flag-only comparison (Level 1)
│   ├── 04_rq1_nested_models.py    # 5x5 repeated stratified CV for nested sets M0, M1, M2 + calibration
│   └── 05_rq2_factors.py          # Univariate BH-FDR tests, sex-stratified RF importance, SHAP, age bands
├── results/
│   ├── tables/                    # Aggregate CSV summary tables (safe for version control)
│   └── figures/                   # Publication-quality plots (PNG)
├── data/
│   ├── raw/                       # Raw NMDID carts (git-ignored per NMDID data-use agreement)
│   └── processed/                 # Person-level cohort.csv and features.csv (git-ignored)
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## Data Access & Privacy Notice
Individual-level NMDID records are governed by the **New Mexico Decedent Image Database (NMDID) Data Use Agreement** and must **never** be committed to public version control.  
To reproduce this pipeline after obtaining authorized access from [NMDID](https://nmdid.unm.edu/), place the 8 CSV tables for each cart into:

```text
data/raw/cart2960/
data/raw/cart2966/
data/raw/cart2975/
```

The `.gitignore` strictly blocks `data/`, `*.csv`, `*.parquet`, and `*.zip` outside of `results/`, ensuring only aggregate summary statistics and plots are tracked.

---

## Environment Setup & Reproduction

```powershell
# 1. Create and activate Python 3.11 virtual environment (Windows PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline scripts in order
python src/config.py
python src/01_build_dataset.py
python src/02_data_audit.py
python src/03_baseline_models.py
python src/04_rq1_nested_models.py
python src/05_rq2_factors.py
```

---

## Key Summary Outputs (`results/`)

| Output File | Description |
| :--- | :--- |
| `results/tables/cohort_flow.csv` | Step-by-step sample counts from 1,500 raw rows to Levels 1, 2, and 3 |
| `results/tables/table1_by_class.csv` | Age (median, IQR) and sex distribution across Level-1 classes |
| `results/tables/intent_by_cause.csv` | Cross-tabulation of primary cause of death by parsed manner/intent |
| `results/tables/coverage_by_class.csv` | Metadata completeness (%) across Level-1 cause-of-death classes |
| `results/tables/missingness_tests.csv` | $\chi^2$ tests confirming non-random missingness across classes ($p < 10^{-7}$) |
| `results/tables/leakage_check.csv` | Cramér's $V$ audit of excluded administrative and post-mortem variables |
| `results/tables/level1_baseline.csv` | 5-fold CV comparison of `M0`, `M_missing_flags_only`, `M1`, and `M2` |
| `results/tables/rq1_metrics.csv` | 5×5 repeated CV metrics (Macro AUC, Macro F1, Balanced Accuracy, Brier) |
| `results/tables/rq1_gains.csv` | Bootstrap $\Delta\text{AUC}$ (95% CI) for `M1 - M0`, `M2 - M1`, and `M2 - M0` |
| `results/tables/rq1_per_class_auc.csv` | One-vs-rest ROC AUC for each class at Levels 1 and 2 |
| `results/tables/rq2_univariate_level1.csv` | Univariate tests + BH-FDR ($q < 0.05$) + age Kruskal–Wallis check (Level 1) |
| `results/tables/rq2_univariate_level2.csv` | Univariate tests + BH-FDR ($q < 0.05$) + age Kruskal–Wallis check (Level 2) |
| `results/tables/rq2_rf_importance_by_sex.csv` | Sex-stratified Random Forest permutation importance (Male vs. Female) |
| `results/tables/rq2_shap_importance.csv` | Mean absolute TreeSHAP values per class and overall (`HistGB` on `M2`) |
| `results/tables/rq2_age_bands.csv` | Stratified Macro AUC across age bands (`18–39`, `40–59`, `60+`) |
