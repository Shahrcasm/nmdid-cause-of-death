# TRIPOD+AI Reporting Checklist (mapping to this repository)

TRIPOD+AI (Collins et al., BMJ 2024) covers studies that develop or evaluate prediction models. This study develops and internally validates models that estimate the broad cause of death from structured decedent records, and evaluates how they behave when parts of the record are missing. It is not a clinical or forensic decision tool; items about deployment are answered with that in mind.

| # | Item | Where it is addressed | Status |
|---|------|-----------------------|--------|
| 1 | Title identifies a prediction-model study, population, outcome | Paper title | To write |
| 2 | Structured abstract | Paper | To write |
| 3a | Background, rationale, existing models | Paper introduction | To write |
| 3b | Rationale for AI/ML methods | `methods_draft.md`, Validation and statistics | Drafted |
| 3c | Health inequalities across groups | Accuracy by sex, age band, race and ethnicity (`accuracy_subgroups.csv`) | Done |
| 4 | Objectives (develop / evaluate) | README "Research statement" and questions | Done |
| 5a | Data sources, design | `methods_draft.md`, Data and cohort; cart design in Limitations | Drafted |
| 5b | Dates of data collection | Deaths 2010 to 2021, 4 after 2017 (`controls_robustness.csv`, deaths after 2017 excluded) | Done |
| 6a | Setting (single medical investigator office) | `methods_draft.md`, Limitations | Drafted |
| 6b | Eligibility criteria | `cohort_flow.csv`, `src/01_build_dataset.py` | Done |
| 6c | Treatments received | Not applicable (decedents) | N/A |
| 7 | Data preparation, pre-processing | `src/01_build_dataset.py`; BMI rules and audit in `bmi_flag_audit.csv` | Done |
| 8a | Outcome definition and how assessed | Outcome levels in `methods_draft.md`; parsing rules in `intent_parsing_audit.csv` | Done |
| 8b | Outcome assessed blind to predictors | **Not blind**: outcome and inputs come from the same investigation record. Quantified in the explanation analysis (`explanation_regimes.csv`, `explanation_gain_shares.csv`) | Discuss |
| 8c | Outcome assessor qualifications | NMDID / Office of the Medical Investigator investigators and pathologists | To write |
| 9a | Choice of predictors | Eight record sections, `methods_draft.md` | Drafted |
| 9b | Predictor definitions and timing | `src/01_build_dataset.py`; all metadata recorded during the death investigation; cadaver BMI is an autopsy measurement | Drafted |
| 9c | Predictor assessment blind to outcome | Not blind (see 8b) | Discuss |
| 10 | Sample size | Fixed archive, no formal calculation; intervals reported for every estimate; small negative control stated | Drafted |
| 11 | Missing data handling | Explicit recorded / not-recorded flags; native missing values in gradient boosting; median imputation inside folds for the other models; four handlings of unavailable sections, including nearest-neighbour multiple imputation, plus a reduced-model reference (`missing_withheld_scenarios.csv`); complete cases (`explanation_regimes.csv`) | Done |
| 12a | Data use (development / evaluation split) | Repeated stratified 5-fold cross-validation (5 x 5), no separate test set | Done |
| 12b | Predictor handling in models | `model_utils.make_model`, `model_utils.Sections` | Done |
| 12c | Model type, building, hyper-parameter tuning | Fixed settings; nested tree-size search only in the accuracy-ceiling analysis (`accuracy_ceiling.csv`) | Done |
| 12d | Heterogeneity across clusters/centres | Cart design; within-cart check (`controls_robustness.csv`); cart-invariance audit (`cart_invariance_check.csv`) | Done |
| 12e | Performance measures | Accuracy, macro AUC, balanced accuracy, macro F1, Brier, top-2 accuracy, calibration error and slope, shortlist coverage | Done |
| 12f | Model updating / recalibration | Not applicable | N/A |
| 12g | Model comparison methods | Paired bootstrap of differences (same resamples) | Done |
| 13 | Class imbalance methods | Balanced class weights (random forest, logistic regression); weighted gradient boosting as robustness | Done |
| 14 | Fairness approaches | Subgroup accuracy by sex, age, race and ethnicity; race and ethnicity are not model inputs | Done |
| 15 | Model output and decision thresholds | Probabilities per cause; 90/95/99% shortlists, including thresholds adjusted for missingness (`reliability_conformal.csv`); answer-only-when-confident thresholds (`reliability_selective.csv`); recalibration (`missing_recalibration.csv`) | Done |
| 16 | Differences development vs evaluation data | Validation on later years (trained to 2014, tested 2015 on) and on held-out county groups (`validation_temporal.csv`, `validation_regional.csv`) | Done |
| 17 | Ethics approval | NMDID data-use agreement; university ethics statement | **To obtain/state** |
| 18a | Funding | Paper | To write |
| 18b | Conflicts of interest | Paper | To write |
| 18c | Protocol / registration | Not registered; analysis plan fixed in code history (git log) | State in paper |
| 18d | Data availability | Restricted (NMDID data-use agreement); aggregate tables in `results/tables` | Done |
| 18e | Code availability | This repository | Done |
| 19 | Patient and public involvement | Not applicable / state none | To write |
| 20a | Participant flow | `cohort_flow.csv` (figure to draw) | Table done |
| 20b | Participant characteristics | `paper_table1_cohort.csv` | Done |
| 20c | Comparison of development and evaluation data | Not applicable | N/A |
| 21 | Number of participants and outcome events | `cohort_flow.csv`, per-analysis `n` columns | Done |
| 22 | Full model specification | `model_utils.py`; trained demo models are not distributed (data-use agreement) | Done |
| 23a | Model performance with CIs | `paper_table2_accuracy.csv`, `paper_table3_missing_information.csv`, `paper_table3b_hidden_sections.csv`, `paper_table4_reliability_explanation_controls.csv` | Done |
| 23b | Performance in key subgroups | `accuracy_subgroups.csv`; by number of blank sections in `missing_natural.csv` | Done |
| 24 | Model updating results | Not applicable | N/A |
| 25 | Interpretation | `methods_draft.md`, Results | Drafted |
| 26 | Limitations | `methods_draft.md`, Limitations (7 items) | Drafted |
| 27a | Handling poor-quality or missing input data in use | Withholding-trained model; "which question to ask next" ranking (`questions_*.csv`); the demo app shows both handlings side by side | Done |
| 27b | User interaction / expertise needed | Demo app labelled a research demonstration, not a forensic tool | Done |
| 27c | Implications and future research | Paper discussion; living-person prediction needs a prospective cohort | To write |

## Items the authors must still supply
1. Ethics statement (item 17): institutional review or exemption, and the NMDID data-use agreement reference.
2. Participant flow diagram (item 20a), drawn from `cohort_flow.csv`.
3. Funding, conflicts, and patient-involvement statements (items 18a, 18b, 19).
4. A STROBE checklist as well if the journal treats the study as observational epidemiology.
