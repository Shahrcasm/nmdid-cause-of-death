# Lifestyle and Cause of Death in Medicolegal Deaths (NMDID Metadata)

**Estimating the broad cause of death from incomplete records**

**Course:** CSE 499A Capstone Project, North South University

**Status:** analysis complete; manuscript in preparation.

## Research statement
From a person's recorded lifestyle and medical history, how well can we estimate the broad cause of death, and how much does the estimate suffer when parts of the record are missing?

The study uses structured decedent metadata from the **New Mexico Decedent Image Database (NMDID)**: 1,450 adult decedents, de-duplicated from 1,500 records across three NMDID data carts.

## Questions
1. **Accuracy.** How often does a model name the correct broad cause from the record, and what is the most these records can support?
2. **Missing information.** How much accuracy is lost when sections of the record are missing, and does a model trained with randomly withheld sections cope better?
3. **Reliability.** When can a single estimate be trusted? Calibration, answering only when confident, and shortlists that contain the true cause 90%, 95% or 99% of the time.
4. **Explanation.** Why does the estimate survive missing information? How much comes from what the record says, and how much from which sections investigators filled in?

## Outcomes
1. **Primary, broad cause (n = 1,380):** natural cardiac (373), drug or alcohol (500), injury (507).
2. **Secondary, intent within injury deaths (n = 487):** accident (215), homicide (157), suicide (115).
3. **Negative control (n = 137):** stab (107) versus gunshot (30) homicides. Lifestyle records should say little about the weapon an attacker used.

## Main findings (held-out people, 5 x 5 cross-validation, 95% intervals)
| Question | Result |
|---|---|
| Accuracy | All inputs: 58.6% correct (55.8 to 61.2), macro AUC 0.776. Age and sex alone: 51.3%. Always naming the largest group: 36.7%. The true cause is among the top two for 87.7%. The best honest configuration reached 62.0% (59.4 to 64.5). Intent within injury deaths stays hard: 46.0%. |
| Validation | Trained on deaths up to 2014 and tested on 2015 onwards: 58.0% (53.7 to 62.3), macro AUC 0.770. Trained on four county groups and tested on the fifth: 57.2% (54.5 to 59.9), macro AUC 0.756. The 90% shortlist covers 93.5% of later-year deaths. On complete records in later years the withholding-trained model is less accurate than the standard model (52.1% versus 58.0%), so the standard model is the choice when a record is complete. |
| Missing information | Hiding any one section from the withholding-trained model costs at most 3.4 points. With every section hidden, a model that reads the gaps as investigator blanks falls to 42.9%, multiple imputation from similar people keeps 49.4%, and the withholding-trained model keeps 52.8% (9.9 points, 6.5 to 13.3, above the naive model; 3.3 points, 1.2 to 5.4, above multiple imputation). On complete records both models perform the same (58.2% versus 58.6%). |
| Next question | Starting from age and sex, letting the model choose which section to ask about next beats one fixed order by 3.4 points after two questions (1.6 to 5.2), but is not clearly better than a random order (+0.9, -1.1 to +3.3). The model most often asks first about body size. |
| Reliability | Shortlists reach 90.0%, 95.0% and 99.2% coverage with average lengths of 1.99, 2.31 and 2.69 of three causes. With one threshold, 90% coverage varies from 86.4% to 92.4% across levels of missingness; adjusting the threshold for the number of blank sections keeps it between 90.0% and 91.1%. Answering only when the top probability is at least 0.7 covers 32.7% of people at 72.1% accuracy. Temperature scaling of the withholding-trained model overcorrects (calibration slope 0.58 to 0.68 before, 1.22 to 1.41 after). |
| Explanation | Which lifestyle sections are filled in carries 61% of the lifestyle gain; what they say carries at least 35%. Among 243 people with drinking, tobacco and diagnoses all recorded, the gain over age and sex is +0.010 AUC (-0.039 to +0.062). |
| Controls | Negative control: no gain (-0.028 AUC, -0.128 to +0.067). Shuffled labels score 0.498. Results hold under five robustness checks (macro AUC 0.772 to 0.778), and adding zip code changes macro AUC by only +0.007. |

Full numbers, interpretation and limitations: [`results/methods_draft.md`](results/methods_draft.md). Reporting checklist: [`results/reporting_checklist.md`](results/reporting_checklist.md).

## How missing information is handled
A section of the record can be **recorded**, **left blank by the investigator**, or **not available to the model**. These are different: investigators leave different sections blank for different causes of death, so a blank is itself informative. We compare five ways of handling a section that is not available:
1. present it as left blank (the naive approach);
2. fill in typical values from the training data;
3. multiple imputation: three donors drawn from the ten most similar training people supply the hidden sections, and the predictions are averaged;
4. a single **withholding-trained model**, trained on the records plus copies in which sections are hidden at random, with a separate "not available" indicator so it never confuses a withheld section with an investigator blank (its probabilities are also recalibrated by temperature scaling);
5. a dedicated model refitted without the section (a reference, not practical when any combination of sections can be missing).

Two further tools build on this: **shortlists adjusted for missingness**, which keep the 90%, 95% or 99% target for complete and nearly empty records alike, and a **"which question to ask next"** ranking, in which the model orders the missing sections by how much the answer is expected to reduce its uncertainty.

## Design safeguards
- **Leakage control.** Seventeen administrative and post-mortem columns (`FORBIDDEN_INPUTS` in `src/config.py`) are blocked from every model and checked at build time. The source cart alone would score 98.2% accuracy (`controls_leakage_demo.csv`), which is why it is blocked.
- **Cadaver BMI.** The main analysis uses autopsy weight and length with an implausibility filter only. Blanking it for decomposition would copy a blocked post-mortem variable into its flag (`bmi_flag_audit.csv`).
- **Cart design.** The 43 decedents found in two carts have identical records in all eight linked tables (`cart_invariance_check.csv`). The cause-specific cart design is discussed as a limitation.
- **Held-out everything.** Imputation, scaling, SHAP values and conformal thresholds use training data only. Model settings are fixed; a small nested search appears only in the accuracy-ceiling analysis.
- **Fair comparison.** Every comparison uses the same folds and the same bootstrap resamples (paired).
- **Subgroups.** Accuracy is reported by sex, age, race and ethnicity. Race and ethnicity are not model inputs.
- **Privacy.** Only aggregate tables are published. Crosstab cells below 5 are suppressed. Person-level data and trained models are git-ignored.

## Repository structure
```text
nmdid-cause-of-death/
├── src/
│   ├── config.py                    # Paths, seed (42), outcome definitions, blocked columns, fast mode
│   ├── model_utils.py               # Cohorts, input sets, record sections, models, CV, metrics, bootstrap
│   ├── 01_build_dataset.py          # De-duplication, cohort, 87 model inputs + extra columns
│   ├── 02_data_audit.py             # Cohort flow, Table 1, coverage, leakage, BMI and cart audits
│   ├── 03_accuracy.py               # Q1: accuracy by inputs and model, ceiling, subgroups
│   ├── 04_missing_information.py    # Q2: records as they are; sections hidden; five handlings; recalibration
│   ├── 04b_next_question.py         # Q2: which question to ask next (model-chosen vs fixed vs random order)
│   ├── 05_explanation.py            # Q4: content vs blank pattern, complete cases, factor tests, SHAP
│   ├── 06_reliability.py            # Q3: calibration, confident answers, conformal shortlists
│   ├── 07_controls.py               # Negative control, permutation test, robustness, leakage demo
│   ├── 07b_validation.py            # Validation on later years and on held-out county groups
│   ├── 08_paper_outputs.py          # Paper tables and the auto-filled methods/results draft
│   └── 09_train_demo_model.py       # Demo models for the app (git-ignored output)
├── app/app.py                       # Streamlit demo and results dashboard
├── results/
│   ├── tables/                      # Aggregate CSV tables (safe to publish)
│   ├── figures/                     # PNG figures
│   ├── methods_draft.md             # Methods, results and limitations draft (generated)
│   └── reporting_checklist.md       # TRIPOD+AI mapping
├── data/                            # raw/ and processed/ (git-ignored, NMDID data-use agreement)
└── requirements.txt
```

## Data access
Individual-level NMDID records are governed by the NMDID Data Use Agreement and must never be committed or uploaded to public services. After obtaining authorised access from [NMDID](https://nmdid.unm.edu/), place the eight CSV tables of each cart in `data/raw/cart2960/`, `data/raw/cart2966/` and `data/raw/cart2975/`.

Data source: Edgar HJH, Daneshvari Berry S, Moes E, Adolphi NL, Bridges P, Nolte KB. *New Mexico Decedent Image Database.* Office of the Medical Investigator, University of New Mexico; 2020. doi:[10.25827/5s8c-n515](https://doi.org/10.25827/5s8c-n515).

## Reproduction
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OMP_NUM_THREADS = "1"      # one thread per process; avoids slowdowns when two scripts run in parallel

python -W ignore -u src/01_build_dataset.py
python -W ignore -u src/02_data_audit.py
python -W ignore -u src/03_accuracy.py
python -W ignore -u src/04_missing_information.py
python -W ignore -u src/04b_next_question.py
python -W ignore -u src/05_explanation.py
python -W ignore -u src/06_reliability.py      # needs 04
python -W ignore -u src/07_controls.py
python -W ignore -u src/07b_validation.py      # needs 03
python -W ignore -u src/08_paper_outputs.py    # needs 02-07b
python -W ignore -u src/09_train_demo_model.py # needs 06
streamlit run app/app.py
```
On a dual-core laptop the full run takes about 45 minutes with two parallel streams (03, 04b then 05; and 04, 06 then 07). Setting `NMDID_FAST=1` runs a quick smoke test with one CV repeat and 50 bootstrap draws; its numbers are not for reporting.

## Demo app and deployment
`app/app.py` is a Streamlit research demonstration, not a forensic tool. It has two tabs:
- **Case simulator.** Enter a hypothetical person and see the probability of each broad cause from age and sex alone and from the full record. Set any field to "Unknown" to compare the standard model, which reads a blank as "not recorded by the investigator", with the withholding-trained model, which reads it as "not available". Locally, the app also shows 90% shortlists adjusted for the number of blank sections and ranks which missing section to ask about next.
- **Results dashboard.** Every result table and figure in `results/`.

The app runs in two modes:
| Mode | Models used | When |
|---|---|---|
| Local | Gradient boosting and the withholding-trained model from `models/demo_models.joblib` | After running `src/09_train_demo_model.py`; the file is git-ignored because it is derived from restricted data |
| Deployed | Logistic regression rebuilt from the aggregate weights in `results/tables/demo_logreg_weights.csv` | When the model file is absent, as on Streamlit Community Cloud |

**Streamlit Community Cloud settings:** repository `nmdid-cause-of-death`, branch `master`, main file path `app/app.py`, Python 3.11; dependencies are installed from `requirements.txt`. No individual-level data is deployed. After results change, rerun `src/09_train_demo_model.py` so the aggregate weights match, then push.

## Key output tables (`results/tables/`)
| File | Content |
|---|---|
| `cohort_flow.csv`, `table1_by_class.csv`, `paper_table1_cohort.csv` | Sample flow and characteristics |
| `coverage_by_class.csv`, `missingness_tests.csv` | Which fields are recorded, by cause |
| `leakage_check.csv`, `bmi_flag_audit.csv`, `cart_invariance_check.csv`, `intent_parsing_audit.csv` | Data audits |
| `accuracy_metrics.csv`, `accuracy_gains.csv`, `accuracy_per_class.csv`, `accuracy_pairwise.csv`, `confusion_matrix.csv` | Q1 accuracy |
| `accuracy_ceiling.csv`, `accuracy_subgroups.csv` | Q1 ceiling and subgroups |
| `paper_table3_missing_information.csv` | Q2: accuracy by named missing fields and true cause (training data, test data, test accuracy) |
| `missing_natural.csv`, `missing_natural_patterns.csv`, `paper_table3c_records_as_they_are.csv` | Q2, records as they are, by sections the investigator left blank |
| `missing_withheld_scenarios.csv`, `missing_withheld_curve.csv`, `missing_recalibration.csv` | Q2, sections hidden at test time; recalibration temperatures |
| `questions_curve.csv`, `questions_comparison.csv`, `questions_first_choice.csv` | Q2, which question to ask next |
| `reliability_calibration.csv`, `reliability_selective.csv`, `reliability_conformal.csv` | Q3 |
| `explanation_regimes.csv`, `explanation_gain_shares.csv`, `factors_*.csv`, `explanation_shap.csv` | Q4 |
| `controls_level3.csv`, `controls_permutation.csv`, `controls_robustness.csv`, `controls_withholding_design.csv`, `controls_leakage_demo.csv` | Controls |
| `validation_temporal.csv`, `validation_temporal_shortlists.csv`, `validation_regional.csv` | Validation on later years and held-out regions |
| `paper_table1_cohort.csv` to `paper_table4_reliability_explanation_controls.csv` | Formatted manuscript tables |
