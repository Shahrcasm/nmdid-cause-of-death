# TRIPOD+AI Reporting Checklist (mapping to this repository)

TRIPOD+AI (Collins et al., BMJ 2024) applies to studies that develop or evaluate prediction models. This study is an *explanatory* prediction analysis: its aim is to quantify and decompose incremental discrimination, not to deploy a model. Items for clinical deployment are marked not applicable. "Paper" means the manuscript still to be written; draft text for most items is in `results/methods_draft.md`.

| # | Item | Where it is addressed | Status |
|---|------|-----------------------|--------|
| 1 | Title identifies a prediction-model study, population, outcome | Paper title | To write |
| 2 | Structured abstract | Paper | To write |
| 3a | Background, rationale, existing models | Paper introduction | To write |
| 3b | Rationale for AI/ML methods | `methods_draft.md`, Statistical analysis | Drafted |
| 3c | Health inequalities across groups | Sex-stratified importance, age-band models (`rq2_*`) | Done |
| 4 | Objectives (develop / evaluate) | README "Research Questions" | Done |
| 5a | Data sources, design | `methods_draft.md`, Study population; carts and cart-design limitation | Drafted |
| 5b | Dates of data collection | `cohort_flow.csv`; deaths 2010 to 2021 (4 after 2017, sensitivity in `robustness.csv`) | Done |
| 6a | Setting (single medical investigator office) | `methods_draft.md`, Limitation 4 | Drafted |
| 6b | Eligibility criteria | `cohort_flow.csv`, `src/01_build_dataset.py` | Done |
| 6c | Treatments received | Not applicable (decedents) | N/A |
| 7 | Data preparation, pre-processing | `src/01_build_dataset.py`; BMI rules and audit in `bmi_flag_audit.csv` | Done |
| 8a | Outcome definition and how assessed | Outcome hierarchy; parsing rules in `intent_parsing_audit.csv` | Done |
| 8b | Outcome assessed blind to predictors | Outcomes come from the medical investigator record, predictors from the same record. **Not blind**: this is the recording-artifact problem the study quantifies (RQ3) | Discuss |
| 8c | Outcome assessor qualifications | NMDID / OMI investigators and pathologists | To write |
| 9a | Choice of predictors | Lifestyle and medical-history blocks, `methods_draft.md` | Drafted |
| 9b | Predictor definitions and timing | `src/01_build_dataset.py`; all pre-mortem metadata except cadaver BMI (autopsy measurement) | Drafted |
| 9c | Predictor assessment blind to outcome | Not blind (see 8b); investigated directly in RQ3 | Discuss |
| 10 | Sample size | No formal calculation (fixed archive); intervals reported for every estimate; Level 3 power limitation stated | Drafted |
| 11 | Missing data handling | Explicit `_missing` indicators; HistGB native NaN handling; median imputation inside folds for RF/LogReg; complete-case and blank-randomised analyses (RQ3) | Done |
| 12a | Data use (development / evaluation split) | Repeated stratified 5-fold CV (5 x 5), no separate test set | Done |
| 12b | Predictor handling in models | `model_utils.make_model` | Done |
| 12c | Model type, building, hyper-parameter tuning | Fixed settings, no tuning (`model_utils.py`) | Done |
| 12d | Heterogeneity across clusters/centres | Cart = class confound; within-cart check; cart-invariance audit | Done |
| 12e | Performance measures | Macro AUC, macro F1, balanced accuracy, Brier, calibration curves | Done |
| 12f | Model updating / recalibration | Not applicable | N/A |
| 12g | Model comparison methods | Paired bootstrap of AUC differences (same resamples) | Done |
| 13 | Class imbalance methods | Balanced class weights (RF, LogReg); HistGB weighted as robustness | Done |
| 14 | Fairness approaches | Sex and age-band stratified results | Done |
| 15 | Model output and decision thresholds | Probabilities; no decision threshold (not a decision tool) | Done |
| 16 | Differences development vs evaluation data | Not applicable (internal validation only) | N/A |
| 17 | Ethics approval | NMDID data-use agreement; university ethics statement | **To obtain/state** |
| 18a | Funding | Paper | To write |
| 18b | Conflicts of interest | Paper | To write |
| 18c | Protocol / registration | Not registered; analysis plan fixed in code history (git log) | State in paper |
| 18d | Data availability | Restricted (NMDID DUA); aggregate tables in `results/tables` | Done |
| 18e | Code availability | This repository | Done |
| 19 | Patient and public involvement | Not applicable / state none | To write |
| 20a | Participant flow | `cohort_flow.csv` (figure to draw) | Table done |
| 20b | Participant characteristics | `paper_table1_cohort.csv`, `coverage_by_class.csv` | Done |
| 20c | Comparison of development and evaluation data | Not applicable | N/A |
| 21 | Number of participants and outcome events | `cohort_flow.csv`, per-analysis `n` columns | Done |
| 22 | Full model specification | `model_utils.py`; trained demo model is not distributed (DUA) | Done |
| 23a | Model performance with CIs | `paper_table2_rq1_gains.csv`, `paper_table4_rq3_and_controls.csv` | Done |
| 23b | Performance in key subgroups | Age bands, sex, NOK strata, complete cases | Done |
| 24 | Model updating results | Not applicable | N/A |
| 25 | Interpretation | `methods_draft.md`, Interpretation and limitations | Drafted |
| 26 | Limitations | Same section (7 limitations) | Drafted |
| 27a | Handling poor-quality or missing input data in use | Not applicable (not deployed); app labels blanks explicitly | N/A |
| 27b | User interaction / expertise needed | Demo app is a teaching tool, labelled as such | Done |
| 27c | Implications and future research | Paper discussion | To write |

## Items the authors must still supply
1. Ethics statement (item 17): institutional review or exemption, and the NMDID data-use agreement reference.
2. Participant flow diagram (item 20a), drawn from `cohort_flow.csv`.
3. Funding, conflicts, and patient-involvement statements (items 18a, 18b, 19).
4. A STROBE checklist as well if the journal treats the study as observational epidemiology; most items above map to it directly.
