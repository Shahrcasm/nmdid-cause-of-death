"""Step 8 - Paper outputs: manuscript tables and a methods/results draft whose
every number is read from results/tables (nothing is typed by hand).

Outputs: tables/paper_table1_cohort.csv, paper_table2_accuracy.csv,
         paper_table3_missing_information.csv (named missing fields),
         paper_table3b_hidden_sections.csv, paper_table3c_records_as_they_are.csv,
         paper_table4_reliability_explanation_controls.csv, results/methods_draft.md
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import config
import model_utils as mu

# Table 3: fields removed from the same held-out people (withholding-trained model).
TABLE3_SCENARIOS = [
    ("nothing hidden", "None (record as collected)"),
    ("Drinking", "Drinking"),
    ("Drug use", "Drug use"),
    ("Diagnoses", "Diagnoses"),
    ("All lifestyle sections", "All lifestyle fields: drinking, tobacco, drug use, marital status, body size, education, work, diet"),
    ("All medical sections", "All medical-history fields: diagnoses, past injuries and surgery"),
    ("Everything except age and sex", "Everything except age and sex"),
]

T = config.TABLES_DIR


def load(name):
    return pd.read_csv(T / f"{name}.csv")


def pct(x):
    return f"{100 * x:.1f}%"


def ci(r, m, as_pct=False):
    f = (lambda v: f"{100 * v:.1f}") if as_pct else (lambda v: f"{v:.3f}")
    unit = "%" if as_pct else ""
    return f"{f(r[m])}{unit} ({f(r[m + '_ci_low'])} to {f(r[m + '_ci_high'])})"


def signed(v, as_pct=False):
    return f"{100 * v:+.1f} points" if as_pct else f"{v:+.3f}"


def gain(r, g, lo, hi, as_pct=False):
    if as_pct:
        return f"{100 * r[g]:+.1f} points ({100 * r[lo]:+.1f} to {100 * r[hi]:+.1f})"
    return f"{r[g]:+.3f} ({r[lo]:+.3f} to {r[hi]:+.3f})"


def row(df, **match):
    m = df
    for k, v in match.items():
        m = m[m[k] == v]
    assert len(m) == 1, (match, len(m))
    return m.iloc[0]


if __name__ == "__main__":
    flow = load("cohort_flow").set_index("step")["count"]
    t1, cov = load("table1_by_class"), load("coverage_by_class")
    acc, ceil, sg = load("accuracy_metrics"), load("accuracy_ceiling"), load("accuracy_subgroups")
    pcl, cm = load("accuracy_per_class"), load("confusion_matrix")
    nat, scen, curve = load("missing_natural"), load("missing_withheld_scenarios"), load("missing_withheld_curve")
    reg, shares = load("explanation_regimes"), load("explanation_gain_shares")
    u1, a1, u2, a2 = load("factors_univariate_level1"), load("factors_adjusted_level1"), load("factors_univariate_level2"), load("factors_adjusted_level2")
    shap_df = load("explanation_shap")
    cal, sel, conf = load("reliability_calibration"), load("reliability_selective"), load("reliability_conformal")
    l3, perm, rob = load("controls_level3"), load("controls_permutation"), load("controls_robustness")
    design, leak = load("controls_withholding_design"), load("controls_leakage_demo")
    qcur, qcmp, qfirst, temps = load("questions_curve"), load("questions_comparison"), load("questions_first_choice"), load("missing_recalibration")
    vt, vs, vr = load("validation_temporal"), load("validation_temporal_shortlists"), load("validation_regional")
    leakage, bmi, inv = load("leakage_check"), load("bmi_flag_audit"), load("cart_invariance_check")
    C = mu.LEVEL1_CLASSES

    # ---------- Table 1: cohort ----------
    rows = []
    t1i = t1.set_index("level1_class")
    rows.append({"characteristic": "People, n", **{c: int(t1i.loc[c, "n"]) for c in C}, "all": int(t1i["n"].sum())})
    rows.append({"characteristic": "Age, median (IQR)", **{c: t1i.loc[c, "age_median_iqr"] for c in C}, "all": ""})
    rows.append({"characteristic": "Male, %", **{c: t1i.loc[c, "male_pct"] for c in C},
                 "all": round(100 * t1i["male_n"].sum() / t1i["n"].sum(), 1)})
    for _, r in cov.iterrows():
        rows.append({"characteristic": f"Recorded: {r['field_block']}, %", **{c: r[f"{c}_pct"] for c in C}, "all": r["overall_pct"]})
    pd.DataFrame(rows).to_csv(T / "paper_table1_cohort.csv", index=False)

    # ---------- Table 2: accuracy ----------
    names = {"M0_base": "Age + sex", "M1_lifestyle": "+ lifestyle", "M2_med_history": "+ medical history"}
    rows = []
    for _, r in acc.iterrows():
        rows.append({"outcome": {"Level1": "Broad cause (3 classes)", "Level2": "Intent within injury deaths (3 classes)"}[r["level"]],
                     "model": r["model"], "inputs": names[r["feature_set"]], "n": r["n"],
                     "accuracy (95% CI)": ci(r, "accuracy", True), "macro AUC (95% CI)": ci(r, "macro_auc"),
                     "balanced accuracy": r["bal_acc"], "Brier score": r["brier"], "top-2 accuracy": r["top2_accuracy"],
                     "largest-group accuracy": r["majority_class_accuracy"]})
    for _, r in ceil.iterrows():
        rows.append({"outcome": "Broad cause: accuracy ceiling", "model": r["model"], "inputs": r["inputs"], "n": r["n"],
                     "accuracy (95% CI)": ci(r, "accuracy", True), "macro AUC (95% CI)": ci(r, "macro_auc"),
                     "balanced accuracy": r["bal_acc"], "Brier score": r["brier"], "top-2 accuracy": r["top2_accuracy"],
                     "largest-group accuracy": ""})
    pd.DataFrame(rows).to_csv(T / "paper_table2_accuracy.csv", index=False)

    # ---------- Table 3: missing information ----------
    # Table 3 (named missing fields). Each listed field is removed from every
    # held-out person after training; the same people appear in every row group.
    counts = {c: int(t1.set_index("level1_class").loc[c, "n"]) for c in C}
    sc = scen[scen["handling"] == "withheld_model"].set_index("hidden_at_test_time")
    fac_rows = []
    for key, label in TABLE3_SCENARIOS:
        r = sc.loc[key]
        fac_rows.append({"Missing fields": label, "Training data (%)": 80, "Ground truth": f"All causes (n = {sum(counts.values())})",
                         "Test data (%)": 20, "Test accuracy (95% CI)": ci(r, "accuracy", True)})
        for c in C:
            fac_rows.append({"Missing fields": label, "Training data (%)": 80,
                             "Ground truth": f"{mu.CLASS_LABELS[c]} (n = {counts[c]})", "Test data (%)": 20,
                             "Test accuracy (95% CI)": ci(r, f"accuracy_true_{c}", True)})
    pd.DataFrame(fac_rows).to_csv(T / "paper_table3_missing_information.csv", index=False)

    # Records as they are: by number of sections the investigator left blank and true cause.
    band_label = {"none": "None", "1 to 3": "1 to 3", "4 or 5": "4 or 5", "6": "6", "7 or 8": "7 or 8", "any (all people)": "Any"}
    rows = []
    for _, r in nat.iterrows():
        cell = r["ground_truth"] != "all causes" and r["sections_not_recorded"] != "any (all people)"
        total = r["ground_truth"] == "all causes" and r["sections_not_recorded"] == "any (all people)"
        if not (cell or total):
            continue
        rows.append({"Sections left blank by the investigator (of 8)": band_label[r["sections_not_recorded"]],
                     "Training data (%)": f"{r['train_data_pct']:.1f}",
                     "Ground truth": mu.CLASS_LABELS.get(r["ground_truth"], "All causes"),
                     "Test data (%)": f"{r['test_data_pct']:.1f} (n = {int(r['n'])})",
                     "Test accuracy (95% CI)": f"{100 * r['test_accuracy_standard_model']:.1f}% ({100 * r['standard_model_ci_low']:.1f} to {100 * r['standard_model_ci_high']:.1f})"})
    pd.DataFrame(rows).to_csv(T / "paper_table3c_records_as_they_are.csv", index=False)
    labels = {"not_recorded": "standard model, hidden = left blank", "imputed": "standard model, hidden = typical values",
              "multiple_imputation": "standard model, hidden = multiple imputation",
              "withheld_model": "withholding-trained model", "reduced_model": "dedicated reduced model (reference)"}
    rows = []
    for _, r in scen.iterrows():
        rows.append({"hidden at test time": r["hidden_at_test_time"], "handling": labels[r["handling"]],
                     "accuracy (95% CI)": ci(r, "accuracy", True),
                     "change vs full record": gain(r, "accuracy_change_vs_full_record", "acc_change_ci_low", "acc_change_ci_high", True),
                     "macro AUC (95% CI)": ci(r, "macro_auc"), "same answer as full record (%)": r["same_answer_as_full_record_pct"],
                     **{f"accuracy, true {mu.CLASS_LABELS[c].lower()}": f"{100 * r['accuracy_true_' + c]:.1f}%" for c in C}})
    pd.DataFrame(rows).to_csv(T / "paper_table3b_hidden_sections.csv", index=False)

    # ---------- Numbers for the draft ----------
    L1 = acc[acc["level"] == "Level1"]
    hg = {fs: row(L1, model="HistGB", feature_set=fs) for fs in names}
    ens2, rf2, lr2 = (row(L1, model=m, feature_set="M2_med_history") for m in ["Ensemble", "RF", "LogReg"])
    L2 = acc[(acc["level"] == "Level2") & (acc["feature_set"] == "M2_med_history")]
    l2_best = L2.loc[L2["macro_auc"].idxmax()]
    l2_hg0 = row(acc, level="Level2", model="HistGB", feature_set="M0_base")
    l2_hg2 = row(acc, level="Level2", model="HistGB", feature_set="M2_med_history")
    best_ceil = ceil.loc[ceil["accuracy"].idxmax()]
    sg2 = sg[sg["feature_set"] == "M2_med_history"]
    sg_lo, sg_hi = sg2.loc[sg2["accuracy"].idxmin()], sg2.loc[sg2["accuracy"].idxmax()]
    pc = pcl[(pcl["model"] == "HistGB") & (pcl["feature_set"] == "M2_med_history")].set_index("class_name")

    nall = nat[nat["ground_truth"] == "all causes"].set_index("sections_not_recorded")
    s = scen.set_index(["hidden_at_test_time", "handling"])
    every, life = "Everything except age and sex", "All lifestyle sections"
    w_full, s_full = s.loc[("nothing hidden", "withheld_model")], s.loc[("nothing hidden", "not_recorded")]
    single = scen[(scen["scenario_type"] == "single section") & (scen["handling"] == "withheld_model")]
    worst_single = single.loc[single["accuracy_change_vs_full_record"].idxmin()]
    cur = curve.set_index(["handling", "sections_hidden"])

    conf_all = conf[(conf["group_type"] == "all people") & (conf["model"] == "standard") & (conf["method"] == "marginal")].set_index("target_coverage_pct")
    conf_cc = conf[(conf["group_type"] == "true cause") & (conf["model"] == "standard") & (conf["method"] == "marginal") & (conf["target_coverage_pct"] == 90)]
    conf_cls = conf[(conf["group_type"] == "true cause") & (conf["model"] == "standard") & (conf["method"] == "class-conditional") & (conf["target_coverage_pct"] == 90)]
    sel_s = sel[sel["model"] == "standard"].set_index("min_top_probability")
    cal_s = cal[cal["model"] == "standard"]
    cal_w, cal_r = cal[cal["model"] == "withheld"], cal[cal["model"] == "withheld_recal"]
    band90 = conf[(conf["model"] == "standard") & (conf["target_coverage_pct"] == 90) & (conf["group_type"] == "sections not recorded")]
    bm, bc = band90[band90["method"] == "marginal"], band90[band90["method"] == "missingness-conditional"]
    qc = qcur.set_index(["order", "questions_asked"])
    qd = qcmp.set_index(["questions_asked", "comparison"])
    q_first = qfirst.iloc[0]
    vti = vt.set_index("inputs")
    vpool = vr[vr["validation"] == "pooled over held-out county groups"].set_index("inputs")
    vfold = vr[vr["validation"].str.startswith("held-out county group")]
    vs90 = vs[(vs["target_coverage_pct"] == 90) & (vs["method"] == "one threshold")].iloc[0]
    STD, WH = "all inputs (standard model)", "all inputs (withholding-trained model)"
    EV_NR, EV_W = "everything hidden, read as blank (standard model)", "everything hidden (withholding-trained model)"

    sh = shares.set_index(["level", "model", "block"])
    sh_l, sh_lr = sh.loc[("Level1", "HistGB", "lifestyle")], sh.loc[("Level1", "LogReg", "lifestyle")]
    sh_lh = sh.loc[("Level1", "HistGB", "lifestyle + medical history")]
    ccr = reg[(reg["analysis"] == "complete_cases") & (reg["model"] == "HistGB")].set_index("feature_set")
    top_shap = ", ".join(shap_df["feature"].head(8))

    l3lr = l3[l3["model"] == "LogReg"].set_index("feature_set")
    l3gb = l3[l3["model"] == "HistGB"].set_index("feature_set")
    ps = perm.set_index("level")
    robi = rob.set_index(["check", "feature_set"])
    des = design.set_index(["design", "hidden_at_test_time"])
    lk = leak.set_index("inputs")
    n_c66 = int(robi.loc[("within cart 2966 (cardiac vs non-cardiac)", "M0_base"), "n"])

    # ---------- Table 4: reliability, explanation, controls ----------
    t4 = [
        ("Reliability", "Expected calibration error, standard model (range over classes)", f"{cal_s['expected_calibration_error'].min():.3f} to {cal_s['expected_calibration_error'].max():.3f}"),
        ("Reliability", "Calibration slope, standard model (range over classes)", f"{cal_s['calibration_slope'].min():.2f} to {cal_s['calibration_slope'].max():.2f}"),
        ("Reliability", "Answer only when top probability >= 0.7: people answered / accuracy",
         f"{sel_s.loc[0.7, 'answered_pct']:.1f}% / {100 * sel_s.loc[0.7, 'accuracy_answered']:.1f}%"),
    ]
    for target in [90, 95, 99]:
        r = conf_all.loc[target]
        t4.append(("Reliability", f"Shortlist targeting {target}%: coverage reached / average length (of 3)",
                   f"{r['coverage_pct']:.1f}% / {r['mean_shortlist_size']:.2f}"))
    t4 += [
        ("Reliability", "90% shortlist coverage across bands of blank sections: one threshold / threshold adjusted for blank sections",
         f"{bm['coverage_pct'].min():.1f}% to {bm['coverage_pct'].max():.1f}% / {bc['coverage_pct'].min():.1f}% to {bc['coverage_pct'].max():.1f}%"),
        ("Reliability", "Calibration slope of the withholding-trained model, before / after recalibration (range over classes)",
         f"{cal_w['calibration_slope'].min():.2f} to {cal_w['calibration_slope'].max():.2f} / {cal_r['calibration_slope'].min():.2f} to {cal_r['calibration_slope'].max():.2f}"),
        ("Validation", "Later years (trained to 2014, tested 2015 on): accuracy / macro AUC, all inputs",
         f"{pct(vti.loc[STD, 'accuracy'])} / {vti.loc[STD, 'macro_auc']:.3f}"),
        ("Validation", "Held-out county groups: accuracy / macro AUC, all inputs",
         f"{pct(vpool.loc[STD, 'accuracy'])} / {vpool.loc[STD, 'macro_auc']:.3f}"),
        ("Validation", "Later years: 90% shortlist coverage / average length",
         f"{vs90['coverage_pct']:.1f}% / {vs90['mean_shortlist_size']:.2f}"),
        ("Missing information", "Every section hidden: multiple imputation / withholding-trained model, accuracy",
         f"{pct(s.loc[(every, 'multiple_imputation'), 'accuracy'])} / {pct(s.loc[(every, 'withheld_model'), 'accuracy'])}"),
        ("Missing information", "Accuracy after 2 questions from age and sex: model-chosen / fixed order / random order",
         f"{pct(qc.loc[('guided', 2), 'accuracy'])} / {pct(qc.loc[('fixed', 2), 'accuracy'])} / {pct(qc.loc[('random', 2), 'accuracy'])}"),
    ]
    t4 += [
        ("Explanation", "Share of the lifestyle AUC gain from the blank pattern alone (gradient boosting)", f"{sh_l['auc_share_blank_pattern_pct']:.0f}%"),
        ("Explanation", "Share of the lifestyle AUC gain from the content alone, lower bound (gradient boosting)", f"{sh_l['auc_share_content_pct']:.0f}%"),
        ("Explanation", f"Complete cases (n = {int(ccr.loc['M0_base', 'n'])}): AUC gain of all inputs over age + sex",
         gain(ccr.loc["M2_med_history"], "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high")),
        ("Explanation", "Factors associated with the broad cause after age and sex adjustment (FDR 5%)",
         f"{int(a1['sig_fdr_05'].sum())} of {int(a1['p_raw'].notna().sum())}"),
        ("Control", "Negative control, logistic regression: AUC gain of all inputs (stab vs gunshot)",
         gain(l3lr.loc["M2_med_history"], "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high")),
        ("Control", "Permutation test, Level 1: shuffled-label AUC mean (95% range) vs real",
         f"{ps.loc['Level1', 'null_mean']:.3f} ({ps.loc['Level1', 'null_2.5pct']:.3f} to {ps.loc['Level1', 'null_97.5pct']:.3f}) vs {ps.loc['Level1', 'observed_auc_1rep']:.3f}"),
        ("Control", "Leakage demonstration: blocked cart variable alone, accuracy", pct(lk.loc["Blocked cart variable only", "accuracy"])),
    ]
    for check in ["cadaver BMI blanked for decomposition", "living BMI only", "class weights", "deaths after 2017 excluded",
                  "cardiac-conflict ingestion deaths relabelled"]:
        t4.append(("Robustness", f"All inputs, {check}: macro AUC / accuracy",
                   f"{robi.loc[(check, 'M2_med_history'), 'macro_auc']:.3f} / {pct(robi.loc[(check, 'M2_med_history'), 'accuracy'])}"))
    pd.DataFrame(t4, columns=["area", "measure", "value"]).to_csv(T / "paper_table4_reliability_explanation_controls.csv", index=False)

    # ---------- Methods and results draft ----------
    md = f"""# Methods and Results Draft

*Generated by `src/08_paper_outputs.py` from `results/tables/`. Do not edit numbers by hand; rerun the pipeline instead.*

## Research statement
From a person's recorded lifestyle and medical history, how well can we estimate the broad cause of death, and how much does the estimate suffer when parts of the record are missing?

## Data and cohort
We used structured decedent metadata from three data carts of the New Mexico Decedent Image Database (NMDID), eight linked tables per cart: {flow['1. Raw decedent rows across 3 carts']} rows, {flow['2. Unique decedents after deduplication']} unique decedents after de-duplication, and {flow['4. Adult decedents (age >= 18)']} adults after excluding {flow['3. Excluded: age < 18']} people under 18 and {flow['3b. Excluded: missing age']} with missing age. The {inv['duplicate_ids_total'].iloc[0]} decedents present in two carts have identical records in every linked table ({int(inv['identical_across_carts'].sum())} of {int(inv['duplicate_ids_with_rows_in_both_carts'].sum())} table-level comparisons), so cart extraction selects people without altering their records.

## Outcomes
1. **Primary, broad cause (n = {flow['6. Level 1 cohort (cardiac / drug_alcohol / injury)']}):** natural cardiac ({int(t1i.loc['natural_cardiac', 'n'])}), drug or alcohol ({int(t1i.loc['drug_alcohol', 'n'])}), injury ({int(t1i.loc['injury', 'n'])}). Excluded: {flow['5a. Excluded from Level 1: cardiac_conflict']} cardiac deaths with a non-natural manner and {flow['5b. Excluded from Level 1: other causes']} other causes.
2. **Secondary, intent within injury deaths (n = {flow['7. Level 2 cohort (injury: accident / homicide / suicide)']}):** accident, homicide, suicide.
3. **Negative control (n = {flow['8. Level 3 cohort (injury homicide: gunshot vs stab)']}):** stab versus gunshot homicides.

## Inputs and the eight sections of a record
Inputs were built as nested sets: M0, age and sex (2 inputs); M1, M0 plus lifestyle (drinking, tobacco, drug use, marital status, cadaver and living body mass index, education, socioeconomic status, diet, heavy lifting, firearm-access occupation; 60 inputs); M2, M1 plus 18 diagnoses and four lifetime injury or surgery items (87 inputs). Every block has a flag saying whether the investigator recorded it. The blocks form eight sections: drinking, tobacco, drug use, marital status, body size, social background, diagnoses, and past injuries and surgery. Seventeen administrative, geographic and post-mortem columns are blocked from every model and checked at build time; the source cart alone has Cramér's V {leakage.set_index('forbidden_column').loc['source_cart', 'cramers_v_with_level1']:.2f} with the broad cause. Cadaver body mass index uses autopsy weight and length with only an implausibility filter, because blanking it for decomposition copies a blocked post-mortem variable into its flag (Cramér's V {bmi.set_index('flag').loc['cadaver_bmi_decomp_blanked_missing', 'cramers_v_with_decomposition_score']:.2f}). Medications (partly scene evidence) and race and ethnicity are never in M0 to M2; they appear only in the accuracy-ceiling and subgroup analyses.

## Missing information
A section can be in three states: recorded; left blank by the investigator; or not available to the model. We compared five ways of handling a section that is not available: (1) present it as left blank; (2) fill in the typical value of the training fold; (3) multiple imputation by nearest-neighbour hot-deck, in which three donors drawn from the ten training people most similar on the available inputs supply the hidden sections and the predictions are averaged; (4) a single **withholding-trained model**, fitted on the training fold plus five copies in which every section is hidden independently with probability 0.15, 0.3, 0.5, 0.75 or 1.0, where a hidden section has its values and its flag removed and a separate indicator set, so that "not recorded" and "not available" stay distinct; and (5) a dedicated model refitted without the hidden sections, as a reference. Hiding is applied only to held-out people, after the models are fitted. The withholding-trained model's probabilities were recalibrated by temperature scaling learned on one fifth of each training fold, which never changes its answers. To choose the next question, the model ranks the sections a person is missing by the expected reduction in its uncertainty (entropy) over the three causes, averaging over plausible answers taken from training people of each cause and weighted by the current probability of that cause; neither the person's true cause nor their true answers are used. We compared this order with one fixed order for everyone and a random order, starting every held-out person from age and sex only.

## Validation and statistics
All estimates come from stratified 5-fold cross-validation repeated 5 times (seed 42), with each person's held-out probabilities averaged over repeats. Imputation and scaling are fitted inside training folds. Models are gradient boosting (main), random forest and logistic regression with fixed settings, plus their unweighted average. We report accuracy, macro one-vs-rest AUC, balanced accuracy, macro F1 and the multi-class Brier score with 95% intervals from {config.N_BOOT} bootstrap resamples of people; comparisons use the same resamples and are paired. The accuracy ceiling adds a nested search of tree size (inner 3-fold, log-loss) inside every outer training fold. Calibration uses decile reliability curves, expected calibration error and calibration slope. Shortlists use split conformal prediction with the score 1 minus the probability of the true cause, averaged over 500 random calibration and evaluation halves, in marginal, class-conditional and missingness-conditional versions (one threshold per band of blank sections). To check that the results carry over, models were also trained on deaths up to 2014 and tested on deaths from 2015 on, and trained on four groups of counties and tested on the fifth (five groups; the largest county formed its own group). Shortlist thresholds for later years were set on 2014 deaths with a model trained on deaths up to 2013. Year and county only defined these splits and were never model inputs.

## Results

**Accuracy.** With all inputs, gradient boosting named the correct broad cause for {ci(hg['M2_med_history'], 'accuracy', True)} of held-out people (macro AUC {ci(hg['M2_med_history'], 'macro_auc')}), against {ci(hg['M0_base'], 'accuracy', True)} from age and sex alone and {pct(hg['M0_base']['majority_class_accuracy'])} for always naming the largest group. Lifestyle raised accuracy by {gain(hg['M1_lifestyle'], 'acc_gain_vs_ref', 'acc_gain_ci_low', 'acc_gain_ci_high', True)} and all inputs by {gain(hg['M2_med_history'], 'acc_gain_vs_ref', 'acc_gain_ci_low', 'acc_gain_ci_high', True)}. The true cause was among the two most probable for {pct(hg['M2_med_history']['top2_accuracy'])} of people ({pct(hg['M0_base']['top2_accuracy'])} with age and sex alone). Random forest reached {pct(rf2['accuracy'])}, logistic regression {pct(lr2['accuracy'])} and their average with gradient boosting {pct(ens2['accuracy'])}. Recall was {pct(pc.loc['natural_cardiac', 'recall'])} for cardiac, {pct(pc.loc['drug_alcohol', 'recall'])} for drug or alcohol and {pct(pc.loc['injury', 'recall'])} for injury deaths. Intent within injury deaths stayed hard: {ci(l2_hg0, 'accuracy', True)} with age and sex and {ci(l2_hg2, 'accuracy', True)} with all inputs (best macro AUC {l2_best['macro_auc']:.3f}, {l2_best['model']}). The highest accuracy any honest configuration reached was {ci(best_ceil, 'accuracy', True)} ({best_ceil['inputs']}, {best_ceil['model']}), {gain(best_ceil, 'acc_gain_vs_ref', 'acc_gain_ci_low', 'acc_gain_ci_high', True)} over the main model. Across sex, age, race and ethnicity groups accuracy ranged from {pct(sg_lo['accuracy'])} ({sg_lo['subgroup_type']} {sg_lo['subgroup']}) to {pct(sg_hi['accuracy'])} ({sg_hi['subgroup_type']} {sg_hi['subgroup']}).

**Later years and other regions.** Trained on deaths up to 2014 and tested on {int(vti.loc[STD, 'n'])} deaths from 2015 on, the standard model named the correct cause for {ci(vti.loc[STD], 'accuracy', True)} (macro AUC {ci(vti.loc[STD], 'macro_auc')}), close to the cross-validated {pct(hg['M2_med_history']['accuracy'])}. Held-out groups of counties gave {ci(vpool.loc[STD], 'accuracy', True)} (macro AUC {ci(vpool.loc[STD], 'macro_auc')}), ranging from {pct(vfold['accuracy_all_inputs'].min())} to {pct(vfold['accuracy_all_inputs'].max())} across the five groups. With every section hidden, the withholding-trained model again outperformed the naive handling, both in later years ({pct(vti.loc[EV_W, 'accuracy'])} versus {pct(vti.loc[EV_NR, 'accuracy'])}) and in other regions ({pct(vpool.loc[EV_W, 'accuracy'])} versus {pct(vpool.loc[EV_NR, 'accuracy'])}). On complete records in later years it was less accurate than the standard model ({pct(vti.loc[WH, 'accuracy'])} versus {pct(vti.loc[STD, 'accuracy'])}), so the standard model remains the choice when a record is complete. Shortlists targeting 90% contained the true cause for {vs90['coverage_pct']:.1f}% of later-year deaths with thresholds set on 2014 deaths (average length {vs90['mean_shortlist_size']:.2f}).

**Records as they are.** For people with no section left blank the standard model was correct for {pct(nall.loc['none', 'test_accuracy_standard_model'])} (n = {int(nall.loc['none', 'n'])}); for people with seven or eight sections blank, {pct(nall.loc['7 or 8', 'test_accuracy_standard_model'])} (n = {int(nall.loc['7 or 8', 'n'])}), where age and sex alone gave {pct(nall.loc['7 or 8', 'test_accuracy_age_sex_only'])}. Accuracy does not fall steadily with missingness, because which sections are blank itself differs by cause of death.

**Same people, less information.** Hiding one section from the withholding-trained model cost at most {-100 * worst_single['accuracy_change_vs_full_record']:.1f} points ({worst_single['hidden_at_test_time']}). With every section hidden, presenting the gaps as investigator blanks gave {ci(s.loc[(every, 'not_recorded')], 'accuracy', True)}, typical values gave {ci(s.loc[(every, 'imputed')], 'accuracy', True)}, and the withholding-trained model gave {ci(s.loc[(every, 'withheld_model')], 'accuracy', True)}, a difference of {gain(s.loc[(every, 'withheld_model')], 'acc_diff_vs_standard_left_blank', 'acc_diff_vs_standard_left_blank_ci_low', 'acc_diff_vs_standard_left_blank_ci_high', True)} over the naive handling and {gain(s.loc[(every, 'withheld_model')], 'acc_diff_vs_reduced_model', 'acc_diff_vs_reduced_model_ci_low', 'acc_diff_vs_reduced_model_ci_high', True)} against a model trained on age and sex alone. Multiple imputation gave {ci(s.loc[(every, 'multiple_imputation')], 'accuracy', True)}; the withholding-trained model differed from it by {gain(s.loc[(every, 'withheld_model')], 'acc_diff_vs_multiple_imputation', 'acc_diff_vs_multiple_imputation_ci_low', 'acc_diff_vs_multiple_imputation_ci_high', True)}. With all lifestyle sections hidden, the four handlings gave {pct(s.loc[(life, 'not_recorded')]['accuracy'])} (left blank), {pct(s.loc[(life, 'imputed')]['accuracy'])} (typical values), {pct(s.loc[(life, 'multiple_imputation')]['accuracy'])} (multiple imputation) and {pct(s.loc[(life, 'withheld_model')]['accuracy'])} (withholding-trained). With four random sections hidden, accuracy was {pct(cur.loc[('not_recorded', 4), 'accuracy'])} with the naive handling, {pct(cur.loc[('multiple_imputation', 4), 'accuracy'])} with multiple imputation and {pct(cur.loc[('withheld_model', 4), 'accuracy'])} with the withholding-trained model (curve values are means over single cross-validation repeats, because the hidden sections differ between repeats; all other estimates average each person's probabilities over the five repeats). On complete records the withholding-trained model matched the standard model ({pct(w_full['accuracy'])} versus {pct(s_full['accuracy'])}; difference {gain(w_full, 'acc_diff_vs_standard_left_blank', 'acc_diff_vs_standard_left_blank_ci_low', 'acc_diff_vs_standard_left_blank_ci_high', True)}).

**Reliability.** Expected calibration error of the standard model was {cal_s['expected_calibration_error'].min():.3f} to {cal_s['expected_calibration_error'].max():.3f} across classes (calibration slope {cal_s['calibration_slope'].min():.2f} to {cal_s['calibration_slope'].max():.2f}). Answering only when the top probability was at least 0.7 gave an answer for {sel_s.loc[0.7, 'answered_pct']:.1f}% of people with accuracy {100 * sel_s.loc[0.7, 'accuracy_answered']:.1f}% ({100 * sel_s.loc[0.7, 'accuracy_ci_low']:.1f} to {100 * sel_s.loc[0.7, 'accuracy_ci_high']:.1f}). Shortlists targeting 90%, 95% and 99% contained the true cause for {conf_all.loc[90, 'coverage_pct']:.1f}%, {conf_all.loc[95, 'coverage_pct']:.1f}% and {conf_all.loc[99, 'coverage_pct']:.1f}% of people, with average lengths of {conf_all.loc[90, 'mean_shortlist_size']:.2f}, {conf_all.loc[95, 'mean_shortlist_size']:.2f} and {conf_all.loc[99, 'mean_shortlist_size']:.2f} of the three causes; {conf_all.loc[90, 'single_cause_pct']:.1f}% of the 90% shortlists named a single cause. Marginal 90% coverage by true cause ranged from {conf_cc['coverage_pct'].min():.1f}% to {conf_cc['coverage_pct'].max():.1f}%; the class-conditional version gave {conf_cls['coverage_pct'].min():.1f}% to {conf_cls['coverage_pct'].max():.1f}%. With one threshold for everyone, 90% shortlists covered {bm['coverage_pct'].min():.1f}% to {bm['coverage_pct'].max():.1f}% of people across bands of blank sections; adjusting the threshold for the number of blank sections gave {bc['coverage_pct'].min():.1f}% to {bc['coverage_pct'].max():.1f}%.

**Why the estimate survives missing information.** A model that saw only which lifestyle sections were filled in recovered {sh_l['auc_share_blank_pattern_pct']:.0f}% of the lifestyle AUC gain; a model that saw only the recorded content, with the blank pattern destroyed, recovered at least {sh_l['auc_share_content_pct']:.0f}% (lifestyle plus medical history: {sh_lh['auc_share_blank_pattern_pct']:.0f}% and {sh_lh['auc_share_content_pct']:.0f}%; with logistic regression {sh_lr['auc_share_blank_pattern_pct']:.0f}% and {sh_lr['auc_share_content_pct']:.0f}%). Among the {int(ccr.loc['M0_base', 'n'])} people with drinking, tobacco and diagnoses all recorded, all inputs changed AUC by {gain(ccr.loc['M2_med_history'], 'auc_gain_vs_ref', 'gain_ci_low', 'gain_ci_high')}. {int(u1['sig_fdr_05'].sum())} of {len(u1)} factors differed by broad cause after FDR correction and {int(a1['sig_fdr_05'].sum())} of {int(a1['p_raw'].notna().sum())} modelled factors after adjustment for age and sex (intent: {int(u2['sig_fdr_05'].sum())} of {len(u2)} and {int(a2['sig_fdr_05'].sum())} of {int(a2['p_raw'].notna().sum())}). The inputs with the largest held-out SHAP values were {top_shap}.

**Controls.** In the negative control, logistic regression gave AUC {ci(l3lr.loc['M0_base'], 'macro_auc')} with age and sex, and all inputs changed it by {gain(l3lr.loc['M2_med_history'], 'auc_gain_vs_ref', 'gain_ci_low', 'gain_ci_high')} (gradient boosting {gain(l3gb.loc['M2_med_history'], 'auc_gain_vs_ref', 'gain_ci_low', 'gain_ci_high')}). With shuffled labels the AUC was {ps.loc['Level1', 'null_mean']:.3f} (95% range {ps.loc['Level1', 'null_2.5pct']:.3f} to {ps.loc['Level1', 'null_97.5pct']:.3f}; {int(ps.loc['Level1', 'n_permutations'])} shuffles) for the broad cause, against {ps.loc['Level1', 'observed_auc_1rep']:.3f} with real labels on the same single-repeat setup ({ps.loc['Level1', 'observed_minus_null_in_sd']:.0f} standard deviations; p = {ps.loc['Level1', 'empirical_p']:.3g}, the smallest attainable), and {ps.loc['Level3', 'null_mean']:.3f} ({ps.loc['Level3', 'null_2.5pct']:.3f} to {ps.loc['Level3', 'null_97.5pct']:.3f}; {int(ps.loc['Level3', 'n_permutations'])} shuffles) for the negative control, against {ps.loc['Level3', 'observed_auc_1rep']:.3f} (p = {ps.loc['Level3', 'empirical_p']:.3g}). All-input macro AUC was {robi.loc[('cadaver BMI blanked for decomposition', 'M2_med_history'), 'macro_auc']:.3f} with decomposition-blanked body mass index, {robi.loc[('living BMI only', 'M2_med_history'), 'macro_auc']:.3f} with living body mass index only, {robi.loc[('class weights', 'M2_med_history'), 'macro_auc']:.3f} with class weights, {robi.loc[('deaths after 2017 excluded', 'M2_med_history'), 'macro_auc']:.3f} excluding deaths after 2017 and {robi.loc[('cardiac-conflict ingestion deaths relabelled', 'M2_med_history'), 'macro_auc']:.3f} with the cardiac-conflict ingestion deaths relabelled; adding zip code changed it by {gain(robi.loc[('zip code added', 'M2_plus_zip_code')], 'auc_gain_vs_ref', 'gain_ci_low', 'gain_ci_high')}. Within cart 2966 ({n_c66} people) the gain over age and sex was {gain(robi.loc[('within cart 2966 (cardiac vs non-cardiac)', 'M2_med_history')], 'auc_gain_vs_ref', 'gain_ci_low', 'gain_ci_high')}. With every section hidden, the withholding-trained model reached {', '.join(pct(des.loc[(d, 'everything hidden except age and sex'), 'accuracy']) for d in des.index.get_level_values(0).unique())} under the three withholding designs tested ({'; '.join(des.index.get_level_values(0).unique())}), against {pct(s.loc[(every, 'not_recorded')]['accuracy'])} for the standard model with the gaps presented as blanks. A model given the blocked cart variable alone would reach {pct(lk.loc['Blocked cart variable only', 'accuracy'])}, which shows why that variable is blocked: any accuracy near that level on these data would come from how the carts were assembled, not from the person.

## Supplementary results
**Which question to ask next.** Starting from age and sex only, asking the section the model chose first gave {pct(qc.loc[('guided', 1), 'accuracy'])}, against {pct(qc.loc[('fixed', 1), 'accuracy'])} with one fixed order and {pct(qc.loc[('random', 1), 'accuracy'])} with a random order; after two questions the figures were {pct(qc.loc[('guided', 2), 'accuracy'])}, {pct(qc.loc[('fixed', 2), 'accuracy'])} and {pct(qc.loc[('random', 2), 'accuracy'])} (model-chosen minus random: {gain(qd.loc[(2, 'guided_minus_random')], 'accuracy_difference', 'ci_low', 'ci_high', True)}; model-chosen minus fixed: {gain(qd.loc[(2, 'guided_minus_fixed')], 'accuracy_difference', 'ci_low', 'ci_high', True)}). The section most often asked first was {q_first['section'][0].lower() + q_first['section'][1:]} ({q_first['asked_first_by_model_pct']:.1f}% of people). Curve values are means over single repeats.

**Recalibration.** Temperature scaling (mean temperature {temps['temperature'].mean():.2f}) changed the calibration slope of the withholding-trained model from between {cal_w['calibration_slope'].min():.2f} and {cal_w['calibration_slope'].max():.2f} (too confident) to between {cal_r['calibration_slope'].min():.2f} and {cal_r['calibration_slope'].max():.2f} (too cautious), and its expected calibration error from {cal_w['expected_calibration_error'].min():.3f}-{cal_w['expected_calibration_error'].max():.3f} to {cal_r['expected_calibration_error'].min():.3f}-{cal_r['expected_calibration_error'].max():.3f}, without changing any answer; it overcorrects, so we do not use it by default.

## Limitations
1. **Ceiling of the data.** No honest configuration we tested exceeded {pct(best_ceil['accuracy'])} accuracy for three broad causes; accuracy near {pct(lk.loc['Blocked cart variable only', 'accuracy'])} appeared only when the blocked cart variable was used, which reflects how the carts were assembled rather than the person.
2. **Not a living-person tool.** Every person in NMDID died suddenly or violently and was investigated; predicting the eventual cause of death of living people needs a prospective cohort with follow-up.
3. **Investigation-driven records.** Inputs and outcome come from the same investigation, and which sections are blank differs by cause; part of the accuracy reflects this recording pattern, quantified in the explanation analysis.
4. **Cart design.** Each cart was drawn for one cause; identical duplicate records show extraction does not change records, but further selection when the carts were assembled cannot be excluded, and the within-cart check is small.
5. **Simulated missingness.** Hidden sections are hidden completely at random; real gaps in other settings may follow other patterns.
6. **Small groups.** The negative control has {int((l3lr.loc['M0_base', 'n']))} people, intent and subgroup analyses are modest, and conformal coverage is averaged over random halves of one cohort.
7. **No validation outside NMDID.** Later years and other New Mexico regions are still one database and one office; settings were fixed except in the ceiling analysis.
"""
    (config.PROJECT_ROOT / "results" / "methods_draft.md").write_text(md, encoding="utf-8")
    print("Saved paper tables 1-4 (+3b) and results/methods_draft.md")
