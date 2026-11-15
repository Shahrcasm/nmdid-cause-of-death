"""Phase 6: publication-ready summary tables and a methods / limitations draft.

Reads the outputs of scripts 02-07 and writes:
  results/tables/paper_table1_cohort.csv
  results/tables/paper_table2_rq1_gains.csv
  results/tables/paper_table3_rq2_top_factors.csv     (OOF SHAP + univariate + adjusted)
  results/tables/paper_table4_rq3_and_controls.csv
  results/methods_draft.md                            (numbers filled from the tables)
"""
import pandas as pd
import config

T = config.TABLES_DIR


def fmt_ci(val, low, high):
    return f"{val:.3f} ({low:.3f} to {high:.3f})"


def fmt_gain(val, low, high):
    return f"{val:+.3f} ({low:+.3f} to {high:+.3f})"


def build_paper_table1():
    t1 = pd.read_csv(T / "table1_by_class.csv")
    t1.to_csv(T / "paper_table1_cohort.csv", index=False)
    return t1


def build_paper_table2():
    m = pd.read_csv(T / "rq1_metrics.csv")
    g = pd.read_csv(T / "rq1_gains.csv")
    gain_map = {(r["level"], r["model"], r["comparison"]): fmt_gain(r["macro_auc_gain"], r["macro_auc_ci_low"], r["macro_auc_ci_high"])
                for _, r in g.iterrows()}
    rows = []
    for _, r in m.iterrows():
        prev = {"M1_lifestyle": "M1_minus_M0", "M2_med_history": "M2_minus_M1"}.get(r["feature_set"])
        rows.append({
            "level": r["level"], "model": r["model"], "feature_set": r["feature_set"], "n": int(r["n"]),
            "macro_auc_95ci": fmt_ci(r["macro_auc"], r["macro_auc_ci_low"], r["macro_auc_ci_high"]),
            "auc_repeat_sd": r.get("auc_repeat_sd", float("nan")),
            "macro_f1_95ci": fmt_ci(r["macro_f1"], r["macro_f1_ci_low"], r["macro_f1_ci_high"]),
            "bal_acc_95ci": fmt_ci(r["bal_acc"], r["bal_acc_ci_low"], r["bal_acc_ci_high"]),
            "brier_95ci": fmt_ci(r["brier"], r["brier_ci_low"], r["brier_ci_high"]),
            "delta_auc_vs_prev_95ci": gain_map.get((r["level"], r["model"], prev), "-") if prev else "-",
            "delta_auc_vs_M0_95ci": gain_map.get((r["level"], r["model"], "M2_minus_M0"), "-") if r["feature_set"] == "M2_med_history" else "-",
        })
    df = pd.DataFrame(rows)
    df.to_csv(T / "paper_table2_rq1_gains.csv", index=False)
    return df


def build_paper_table3():
    u1 = pd.read_csv(T / "rq2_univariate_level1.csv")
    a1 = pd.read_csv(T / "rq2_adjusted_level1.csv")
    shap_df = pd.read_csv(T / "rq2_shap_importance.csv")
    or_cols = [c for c in a1.columns if c.startswith("OR_")]
    merged = (shap_df.head(15)
              .merge(u1[["feature", "stat", "p_fdr_bh", "sig_fdr_05", "natural_cardiac_rate_or_mean",
                         "drug_alcohol_rate_or_mean", "injury_rate_or_mean"]]
                     .rename(columns={"stat": "univariate_stat", "p_fdr_bh": "univariate_p_fdr", "sig_fdr_05": "univariate_sig_fdr"}),
                     on="feature", how="left")
              .merge(a1[["feature", "lr_chi2", "p_fdr_bh", "sig_fdr_05", "note"] + or_cols]
                     .rename(columns={"lr_chi2": "adjusted_lr_chi2", "p_fdr_bh": "adjusted_p_fdr", "sig_fdr_05": "adjusted_sig_fdr", "note": "adjusted_note"}),
                     on="feature", how="left"))
    merged.to_csv(T / "paper_table3_rq2_top_factors.csv", index=False)
    return merged


def build_paper_table4():
    rq3 = pd.read_csv(T / "rq3_artifact.csv")
    l3 = pd.read_csv(T / "level3_negative_control.csv")
    rob = pd.read_csv(T / "robustness.csv")
    rows = []

    def add(section, subset, r):
        rows.append({"section": section, "level_or_subset": subset, "model": r["model"], "feature_set": r["feature_set"],
                     "n": int(r["n"]), "macro_auc_95ci": fmt_ci(r["macro_auc"], r["macro_auc_ci_low"], r["macro_auc_ci_high"]),
                     "delta_auc_vs_ref_95ci": fmt_gain(r["auc_gain_vs_ref"], r["gain_ci_low"], r["gain_ci_high"]) if pd.notna(r.get("auc_gain_vs_ref")) else "-"})

    for _, r in rq3.iterrows():
        add("RQ3_" + r["analysis"], f"{r['level']} ({r['subset']})", r)
    for _, r in l3.iterrows():
        add("Level3_control", "Homicide gunshot vs stab", r)
    ps = pd.read_csv(T / "permutation_summary.csv")
    for _, r in ps.iterrows():
        rows.append({"section": "Permutation_test", "level_or_subset": f"{r['level']} ({int(r['n_permutations'])} shuffles)",
                     "model": "HistGB", "feature_set": "M2_med_history", "n": int(r["n"]),
                     "macro_auc_95ci": f"null {r['null_mean']:.3f} ({r['null_2.5pct']:.3f} to {r['null_97.5pct']:.3f}); observed {r['observed_auc_1rep']:.3f}, p = {r['empirical_p']:.3g}",
                     "delta_auc_vs_ref_95ci": "-"})
    for _, r in rob.iterrows():
        add("Robustness_" + r["check"], "Level1", r)
    df = pd.DataFrame(rows)
    df.to_csv(T / "paper_table4_rq3_and_controls.csv", index=False)
    return df


def pick(df, **kw):
    sub = df
    for k, v in kw.items():
        sub = sub[sub[k] == v]
    return sub.iloc[0]


def write_methods_draft():
    flow = pd.read_csv(T / "cohort_flow.csv").set_index("step")["count"]
    inv = pd.read_csv(T / "cart_invariance_check.csv")
    m = pd.read_csv(T / "rq1_metrics.csv")
    g = pd.read_csv(T / "rq1_gains.csv")
    rev = pd.read_csv(T / "rq1_reverse_order.csv")
    rq3 = pd.read_csv(T / "rq3_artifact.csv")
    shares = pd.read_csv(T / "rq3_gain_shares.csv")
    l3 = pd.read_csv(T / "level3_negative_control.csv")
    ps = pd.read_csv(T / "permutation_summary.csv").set_index("level")
    rob = pd.read_csv(T / "robustness.csv")
    bmi = pd.read_csv(T / "bmi_flag_audit.csv").set_index("flag")
    a1 = pd.read_csv(T / "rq2_adjusted_level1.csv")
    u1 = pd.read_csv(T / "rq2_univariate_level1.csv")
    intent_audit = pd.read_csv(T / "intent_parsing_audit.csv")

    h = m[(m.level == "Level1") & (m.model == "HistGB")].set_index("feature_set")
    hg = g[(g.level == "Level1") & (g.model == "HistGB")].set_index("comparison")
    h2 = m[(m.level == "Level2") & (m.model == "HistGB")].set_index("feature_set")
    rf2 = m[(m.level == "Level2") & (m.model == "RF")].set_index("feature_set")
    enc = rq3[(rq3.analysis == "A_encoding") & (rq3.level == "Level1") & (rq3.model == "HistGB")].set_index("feature_set")
    cc = rq3[(rq3.analysis == "B_complete_cases") & (rq3.model == "HistGB")].set_index("feature_set")
    fam = rq3[(rq3.analysis == "C_family_vs_shared")].set_index("feature_set")
    nok_y = rq3[(rq3.analysis == "C_nok_subgroup") & (rq3.subset == "nok_interview_yes")].set_index("feature_set")
    nok_n = rq3[(rq3.analysis == "C_nok_subgroup") & (rq3.subset == "nok_interview_no")].set_index("feature_set")
    med = rq3[(rq3.analysis == "D_medications")].set_index("feature_set")
    sens = rq3[(rq3.analysis == "E_cardiac_conflict")].set_index("feature_set")
    sh = shares[(shares.level == "Level1") & (shares.model == "HistGB")].set_index("block")
    l3lr = l3[l3.model == "LogReg"].set_index("feature_set")
    l3gb = l3[l3.model == "HistGB"].set_index("feature_set")
    rob_i = rob.set_index(["check", "feature_set"])
    n_fall = int(intent_audit[(intent_audit.rule.str.startswith("default")) & (intent_audit.n_injury_cohort > 0)].n_injury_cohort.sum())
    n_adj_sig = int(a1.sig_fdr_05.sum()); n_adj_mod = int(a1.p_raw.notna().sum())
    n_uni_sig = int(u1.sig_fdr_05.sum())
    top_adj = a1[a1.sig_fdr_05 == True].head(8).feature.tolist()  # noqa: E712

    def ci(row):
        return fmt_ci(row["macro_auc"], row["macro_auc_ci_low"], row["macro_auc_ci_high"])

    def gain(row):
        return fmt_gain(row["auc_gain_vs_ref"], row["gain_ci_low"], row["gain_ci_high"])

    def ggain(row):
        return fmt_gain(row["macro_auc_gain"], row["macro_auc_ci_low"], row["macro_auc_ci_high"])

    text = f"""# Methods and Results Draft (numbers filled from results/tables)

## Study population and cohort construction
We analysed structured decedent metadata from three data carts (`cart2960`, `cart2966`, `cart2975`) of the New Mexico Decedent Image Database (NMDID): {flow['1. Raw decedent rows across 3 carts']} decedent rows across eight linked tables. The carts were assembled by cause of death, so cart membership is almost collinear with the Level-1 outcome (Cramér's V 0.97, `leakage_check.csv`). Cart is a selection mechanism, not a recording mechanism: the {int(inv.duplicate_ids_total.iloc[0])} decedents present in two carts have identical records in every linked table ({int(inv.identical_across_carts.sum())} of {int(inv.duplicate_ids_with_rows_in_both_carts.sum())} table-level comparisons identical, `cart_invariance_check.csv`). After de-duplication ({flow['2. Unique decedents after deduplication']} unique decedents) and exclusion of {flow['3. Excluded: age < 18']} decedents aged under 18 and {flow['3b. Excluded: missing age']} with missing age, the adult cohort comprised {flow['4. Adult decedents (age >= 18)']} decedents.

## Outcome hierarchy
1. **Level 1 (n = {flow['6. Level 1 cohort (cardiac / drug_alcohol / injury)']}):** `natural_cardiac`, `drug_alcohol`, `injury`; `cardiac_conflict` (n = {flow['5a. Excluded from Level 1: cardiac_conflict']}, cardiac cause with non-natural manner) and sparse `other` causes (n = {flow['5b. Excluded from Level 1: other causes']}) excluded.
2. **Level 2 (n = {flow['7. Level 2 cohort (injury: accident / homicide / suicide)']}):** injury deaths parsed from free-text manner into `accident`, `homicide`, `suicide`; {flow['6b. Excluded from Level 2: injury with unclear manner']} unclear and {flow["6c. Excluded from Level 2: injury coded manner 'Natural'"]} natural-coded gunshot records excluded. Suicide and homicide are keyword rules; `accident` is the residual category ({n_fall} people fell through to it; every string is listed in `intent_parsing_audit.csv`, all describe transport, fall or other unintentional mechanisms).
3. **Level 3 (n = {flow['8. Level 3 cohort (injury homicide: gunshot vs stab)']}):** homicides, stab wound (n = 107) versus gunshot wound (n = 30), used as a weak negative control.

## Feature engineering and leakage prevention
Features were built in `src/01_build_dataset.py` as nested blocks: **M0** (age, sex; 2 features), **M1** (M0 + lifestyle: drinking, tobacco, substance use, marital status, cadaver BMI, living BMI, education, adult and childhood socioeconomic status, diet, strenuous lifting, firearm-access occupation, each with a `_missing` indicator; 60 features), **M2** (M1 + 18 diagnoses and four lifetime injury/surgery blocks with indicators; 87 features). Seventeen administrative, geographic and post-mortem variables are blocked from every model (`FORBIDDEN_INPUTS`, verified at build time).

Cadaver BMI in the main analysis is computed from autopsy weight and length with only an implausibility filter (height outside 120 to 220 cm or BMI outside 12 to 70). Blanking it additionally for moderate or advanced decomposition, as in an earlier version, copies `decomposition_score` into the missing-indicator (Cramér's V {bmi.loc['cadaver_bmi_decomp_blanked_missing', 'cramers_v_with_decomposition_score']:.2f} with decomposition; missing in {bmi.loc['cadaver_bmi_decomp_blanked_missing', 'pct_missing_natural_cardiac']:.0f}% of cardiac versus {bmi.loc['cadaver_bmi_decomp_blanked_missing', 'pct_missing_drug_alcohol']:.0f}% of drug and {bmi.loc['cadaver_bmi_decomp_blanked_missing', 'pct_missing_injury']:.0f}% of injury deaths), whereas the raw-measurement flag is not class-associated (chi-square p = {bmi.loc['cadaver_bmi_missing', 'chi2_p_vs_level1']:.2f}). The decomposition-blanked version is therefore reported only as a sensitivity analysis. Medications (`n_medications`, `any_medication`, `med_none_recorded`) and the family-interview marker `nok_interview` are diagnostic variables for RQ3 and are never in M0 to M2; an empty medications table cannot distinguish "took none" from "not recorded", which the variable name states.

## Statistical analysis
All models (histogram gradient boosting [HistGB], random forest with balanced class weights, and L2 logistic regression with median imputation, standardisation and balanced class weights) use fixed settings with no tuning. Performance was estimated with stratified 5-fold cross-validation repeated 5 times (seed 42); out-of-fold probabilities were averaged per person. Imputation and scaling are fitted inside each training fold. We report macro one-vs-rest AUC, macro F1, balanced accuracy and the multi-class Brier score with 95% percentile intervals from 1,000 bootstrap resamples of people, using the same resamples for every feature set so that differences are paired. These intervals express sampling uncertainty over people given the averaged out-of-fold predictions; the repeat-to-repeat standard deviation of the AUC across the five CV repetitions is reported separately (`auc_repeat_sd`). HistGB early stopping is disabled when a training fold has fewer than 200 rows (Level 3 only).

Factor-level inference (RQ2) used chi-square and Kruskal-Wallis tests with Benjamini-Hochberg correction, and, as the primary adjusted analysis, one multinomial logistic model per factor with age (standardised) and sex as covariates, reporting a 2-df likelihood-ratio test (BH-FDR) and odds ratios versus `natural_cardiac`. Permutation importance (sex-stratified random forest) and TreeSHAP were computed on held-out folds only. RQ3 compared encoding regimes, complete cases, family-only versus investigator-shared fields, strata by family-interview availability, a medications block, and relabelling of cardiac-conflict ingestion deaths. Controls comprised Level 3, a label-permutation test (20 shuffles at Level 1, 200 at Level 3, single-repeat CV for observed and shuffled labels alike), and robustness checks (zip code added, decomposition-blanked cadaver BMI, living BMI only, class weights, exclusion of deaths after 2017, and a within-cart analysis).

## Results summary (HistGB unless stated)
**RQ1.** Level 1 macro AUC rose from {ci(h.loc['M0_base'])} (M0) to {ci(h.loc['M1_lifestyle'])} (M1) and {ci(h.loc['M2_med_history'])} (M2); gains {ggain(hg.loc['M1_minus_M0'])} for lifestyle and {ggain(hg.loc['M2_minus_M1'])} for medical history ({ggain(hg.loc['M2_minus_M0'])} overall). Entered in the reverse order, medical history alone added {fmt_gain(rev.set_index('feature_set').loc['M0_plus_H', 'auc_gain_vs_ref'], rev.set_index('feature_set').loc['M0_plus_H', 'gain_ci_low'], rev.set_index('feature_set').loc['M0_plus_H', 'gain_ci_high'])}, so the attribution between blocks depends on entry order and the two blocks overlap. Level 2 discrimination was weak: {ci(h2.loc['M0_base'])} to {ci(h2.loc['M2_med_history'])} (random forest {rf2.loc['M2_med_history', 'macro_auc']:.3f}).

**RQ2.** {n_uni_sig} of {len(u1)} factors were associated with Level-1 class univariately after FDR; {n_adj_sig} of {n_adj_mod} modelled factors remained associated after adjustment for age and sex (leading factors: {', '.join(top_adj)}). Several of the strongest adjusted factors are `_missing` indicators, i.e. whether a field was recorded.

**RQ3.** Flags alone (which fields were recorded, no values) reached {ci(enc.loc['M0_plus_L_flags_only'])}, {sh.loc['L', 'share_flags_only_pct']:.0f}% of the lifestyle gain; the recorded values with blanks randomised reached {ci(enc.loc['M1_values_blanks_randomised'])}, {sh.loc['L', 'share_values_randomised_pct']:.0f}% of the gain (a lower bound). For lifestyle plus medical history the shares were {sh.loc['L+H', 'share_flags_only_pct']:.0f}% (flags) and {sh.loc['L+H', 'share_values_randomised_pct']:.0f}% (values). Among {int(cc.loc['M0_base', 'n'])} complete cases the lifestyle gain was {gain(cc.loc['M1_lifestyle'])} and the full gain {gain(cc.loc['M2_med_history'])}. Family-interview-only fields added {gain(fam.loc['M0_plus_family_only'])} (any such field recorded in {fam.loc['M0_plus_family_only', 'block_coverage_pct']:.0f}% of people), investigator-shared fields {gain(fam.loc['M0_plus_shared'])}. Gains were {gain(nok_y.loc['M2_med_history'])} with a family interview (n = {int(nok_y.loc['M0_base', 'n'])}) and {gain(nok_n.loc['M2_med_history'])} without (n = {int(nok_n.loc['M0_base', 'n'])}). Adding medications gave {gain(med.loc['M3_plus_medications'])}; relabelling cardiac-conflict ingestion deaths gave M2 = {ci(sens.loc['M2_med_history'])} (n = {int(sens.loc['M0_base', 'n'])}).

**Controls.** Level 3 (logistic regression, primary at this sample size): M0 {ci(l3lr.loc['M0_base'])}, M2 gain {gain(l3lr.loc['M2_med_history'])}; HistGB M2 gain {gain(l3gb.loc['M2_med_history'])}. In the permutation test the null AUC was {ps.loc['Level1', 'null_mean']:.3f} (95% range {ps.loc['Level1', 'null_2.5pct']:.3f} to {ps.loc['Level1', 'null_97.5pct']:.3f}; {int(ps.loc['Level1', 'n_permutations'])} shuffles) at Level 1, versus an observed {ps.loc['Level1', 'observed_auc_1rep']:.3f} (p = {ps.loc['Level1', 'empirical_p']:.3g}, the smallest p attainable with {int(ps.loc['Level1', 'n_permutations'])} shuffles; the observed value lies {(ps.loc['Level1', 'observed_auc_1rep'] - ps.loc['Level1', 'null_mean']) / ps.loc['Level1', 'null_sd']:.0f} null SDs above the null mean), and {ps.loc['Level3', 'null_mean']:.3f} ({ps.loc['Level3', 'null_2.5pct']:.3f} to {ps.loc['Level3', 'null_97.5pct']:.3f}; {int(ps.loc['Level3', 'n_permutations'])} shuffles) at Level 3, versus an observed {ps.loc['Level3', 'observed_auc_1rep']:.3f} (p = {ps.loc['Level3', 'empirical_p']:.3g}). Robustness: M2 = {ci(rob_i.loc[('cadaver_bmi_decomp_blanked', 'M2_med_history')])} with decomposition-blanked cadaver BMI, {ci(rob_i.loc[('living_bmi_only', 'M2_med_history')])} with living BMI only, {ci(rob_i.loc[('class_weight_balanced', 'M2_med_history')])} with class weights, {ci(rob_i.loc[('exclude_deaths_after_2017', 'M2_med_history')])} excluding deaths after 2017, and adding zip code changed M2 by {gain(rob_i.loc[('add_zip_code', 'M2_plus_zip_code')])}. Within cart 2966 (cardiac versus {int(rob_i.loc[('within_cart2966_cardiac_vs_noncardiac', 'M0_base')]['n']) - 373} non-cardiac deaths) the M2 gain was {gain(rob_i.loc[('within_cart2966_cardiac_vs_noncardiac', 'M2_med_history')])}.

## Interpretation and limitations (to carry into the Discussion)
1. **Recording pattern is a large part of the signal.** Indicators of which fields were recorded, with no values, deliver most of the lifestyle gain; in complete cases the gain is close to zero with a wide interval. The randomised-blank regime shows the recorded values also carry information, but that estimate is a lower bound and cannot be separated cleanly from selection into being recorded.
2. **Design: carts are cause-specific.** The three classes come from three cart extractions. Identical duplicate records across carts show that extraction does not alter metadata, but any additional selection applied when the carts were assembled cannot be excluded. The within-cart analysis has only 50 non-cardiac deaths and an interval that includes both no effect and the main-analysis gain; it is inconclusive, not confirmatory.
3. **Level 3 is a weak control.** With 30 gunshot deaths the gain interval spans about 0.2 AUC; the control can expose a large leak, not establish the absence of signal, and weapon choice is not fully independent of lifestyle (firearm access, sex, substance use).
4. **Single site and selected sample.** NMDID contains CT-scanned decedents from one medical investigator office; recording practices and case mix differ elsewhere. No external validation was possible.
5. **No tuning, modest samples.** Fixed model settings avoid optimism from model selection but may understate achievable discrimination; Level 2 and the subgroup analyses are small.
6. **Intervals.** Bootstrap intervals condition on the averaged out-of-fold predictions and do not include model-training variability beyond the five CV repetitions (reported as `auc_repeat_sd`).
7. **Small-cell reporting.** Published crosstabs suppress cells below 5 (`intent_by_cause.csv`); person-level data and trained models are never distributed.
"""
    (config.PROJECT_ROOT / "results" / "methods_draft.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build_paper_table1()
    build_paper_table2()
    build_paper_table3()
    build_paper_table4()
    write_methods_draft()
    print("Saved paper_table1_cohort.csv, paper_table2_rq1_gains.csv, paper_table3_rq2_top_factors.csv, "
          "paper_table4_rq3_and_controls.csv and results/methods_draft.md")
