"""Phase 1: cohort flow, Table 1, coverage, leakage and design audits.

Outputs (results/tables unless noted):
  cohort_flow.csv            counts computed from the raw files (nothing hard-coded)
  table1_by_class.csv        age / sex by Level-1 class
  intent_by_cause.csv        cause x parsed intent, cells < 5 suppressed
  intent_parsing_audit.csv   every manner-of-death string, how it was parsed, and
                             whether it matched a rule or fell through to 'accident'
  coverage_by_class.csv, missingness_tests.csv, figures/coverage_heatmap.png
  leakage_check.csv          Cramer's V of forbidden columns with Level-1 class
  bmi_flag_audit.csv         why the decomposition-blanked BMI flag is a leak
  cart_invariance_check.csv  the 43 cross-cart duplicate records are identical
                             in every linked table (cart selects people; it does
                             not change their metadata)
"""
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import config

build = importlib.import_module("01_build_dataset")
SMALL_CELL = 5


def read_raw(name, dedup=False):
    parts = []
    for cart in config.CARTS:
        df = pd.read_csv(config.RAW_DIR / cart / f"{name}.csv", dtype=str,
                         keep_default_na=False, na_values=[""])
        df["source_cart"] = cart
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def save_cohort_flow(cohort):
    dec = read_raw("decedents")
    unique = dec.drop_duplicates(subset=config.KEY, keep="first")
    age = pd.to_numeric(unique["age_years"], errors="coerce")
    l1_mask = cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])
    l2_mask = (cohort["level1"] == "injury") & cohort["intent"].isin(["accident", "homicide", "suicide"])
    l3_mask = (cohort["level1"] == "injury") & (cohort["intent"] == "homicide") & cohort[
        "primary_cause_of_death"].isin(["Gunshot wound of", "Stab wound (slash, penetrating cut)"])
    rows = [
        {"step": "1. Raw decedent rows across 3 carts", "count": len(dec)},
        {"step": "2. Unique decedents after deduplication", "count": len(unique)},
        {"step": "3. Excluded: age < 18", "count": int((age < 18).sum())},
        {"step": "3b. Excluded: missing age", "count": int(age.isna().sum())},
        {"step": "4. Adult decedents (age >= 18)", "count": len(cohort)},
        {"step": "5a. Excluded from Level 1: cardiac_conflict", "count": int((cohort["level1"] == "cardiac_conflict").sum())},
        {"step": "5b. Excluded from Level 1: other causes", "count": int((cohort["level1"] == "other").sum())},
        {"step": "6. Level 1 cohort (cardiac / drug_alcohol / injury)", "count": int(l1_mask.sum())},
        {"step": "6b. Excluded from Level 2: injury with unclear manner", "count": int(((cohort["level1"] == "injury") & (cohort["intent"] == "unclear")).sum())},
        {"step": "6c. Excluded from Level 2: injury coded manner 'Natural'", "count": int(((cohort["level1"] == "injury") & (cohort["intent"] == "natural")).sum())},
        {"step": "7. Level 2 cohort (injury: accident / homicide / suicide)", "count": int(l2_mask.sum())},
        {"step": "8. Level 3 cohort (injury homicide: gunshot vs stab)", "count": int(l3_mask.sum())},
    ]
    flow_df = pd.DataFrame(rows)
    assert int(flow_df.loc[1, "count"]) - int(flow_df.loc[2, "count"]) - int(flow_df.loc[3, "count"]) == len(cohort)
    flow_df.to_csv(config.TABLES_DIR / "cohort_flow.csv", index=False)
    print("--- Cohort Flow ---")
    print(flow_df.to_string(index=False))
    return dec


def save_table1(cohort):
    classes = ["natural_cardiac", "drug_alcohol", "injury"]
    rows = []
    for cls in classes:
        sub = cohort[cohort["level1"] == cls]
        ages = sub["age_years"]
        q25, q75, med = ages.quantile(0.25), ages.quantile(0.75), ages.median()
        rows.append({
            "level1_class": cls, "n": len(sub),
            "age_median": round(med, 1), "age_q25": round(q25, 1), "age_q75": round(q75, 1),
            "age_median_iqr": f"{int(round(med))} ({int(round(q25))}-{int(round(q75))})",
            "male_n": int((sub["sex_code"] == "Male").sum()),
            "male_pct": round(100.0 * (sub["sex_code"] == "Male").mean(), 1),
        })
    table1_df = pd.DataFrame(rows)
    table1_df.to_csv(config.TABLES_DIR / "table1_by_class.csv", index=False)
    print("\n--- Table 1 by Class ---")
    print(table1_df.to_string(index=False))


def save_intent_tables(cohort):
    """Cause x intent crosstab with small-cell suppression, plus a parsing audit."""
    ct = pd.crosstab(cohort["primary_cause_of_death"].fillna("Missing"), cohort["intent"])
    suppressed = ct.astype(object)
    for c in ct.columns:
        suppressed[c] = ct[c].map(lambda v: f"<{SMALL_CELL}" if 0 < v < SMALL_CELL else str(int(v)))
    suppressed.to_csv(config.TABLES_DIR / "intent_by_cause.csv")

    rows = []
    manners = cohort["manner_of_death"].fillna("NA")
    for text, n in manners.value_counts().items():
        parsed = build.intent(np.nan if text == "NA" else text)
        low = text.strip().lower()
        if text == "NA":
            rule = "missing -> unclear"
        elif low == "natural":
            rule = "exact 'natural'"
        elif build.contains_any(low, build.UNCLEAR_WORDS) or low in ["gunshot wound", "nan", "other"]:
            rule = "unclear keyword"
        elif build.contains_any(low, build.SUICIDE_WORDS):
            rule = "suicide keyword"
        elif build.contains_any(low, build.HOMICIDE_WORDS):
            rule = "homicide keyword"
        else:
            rule = "default (no keyword) -> accident"
        inj_n = int(((cohort["level1"] == "injury") & (manners == text)).sum())
        rows.append({"manner_of_death": text, "parsed_intent": parsed, "rule": rule,
                     "n_all_adults": int(n), "n_injury_cohort": inj_n})
    audit = pd.DataFrame(rows).sort_values(["rule", "n_all_adults"], ascending=[True, False])
    audit.to_csv(config.TABLES_DIR / "intent_parsing_audit.csv", index=False)
    fall = audit[(audit["rule"].str.startswith("default")) & (audit["n_injury_cohort"] > 0)]
    print(f"\n--- Intent parsing: {len(fall)} distinct manner strings fell through to 'accident' "
          f"in the injury cohort ({int(fall['n_injury_cohort'].sum())} people); review intent_parsing_audit.csv ---")


def save_coverage_and_tests(cohort, features):
    med = read_raw("medical-history")
    sub = read_raw("substance-usages")
    l1 = cohort[cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])].copy()
    merged = l1[[config.KEY, "level1", "cadaver_weight", "cadaver_length"]].merge(features, on=config.KEY)

    raw_cad_present = (pd.to_numeric(merged["cadaver_weight"], errors="coerce").notna()
                       & pd.to_numeric(merged["cadaver_length"], errors="coerce").notna()).astype(int)
    sub_clean = sub["substance_usage_type"].astype(str).str.strip()
    drug_ids = set(sub.loc[sub["substance_usage_type"].notna() & (sub_clean != "None") & (sub_clean != ""), config.KEY])
    dx_rows = med[(med["medical_history_type"].astype(str).str.strip() == "Diagnosis") & med["medical_history_detail"].notna()]
    dx_ids = set(dx_rows[config.KEY])
    drink_cols = [c for c in merged.columns if c.startswith("drink_") and c != "drink_missing"]
    tob_cols = [c for c in merged.columns if c.startswith("tobtype_") or c.startswith("tobuse_")]

    present = pd.DataFrame({
        "level1": merged["level1"],
        "drinking": (merged[drink_cols].sum(axis=1) > 0).astype(int),
        "tobacco": (merged[tob_cols].sum(axis=1) > 0).astype(int),
        "drug_use": merged[config.KEY].isin(drug_ids).astype(int),
        "diagnosis": merged[config.KEY].isin(dx_ids).astype(int),
        "medications": merged["any_medication"],
        "marital": 1 - merged["marital_missing"],
        "nok_interview": merged["nok_interview"],
        "cadaver_bmi_raw_measurements": raw_cad_present,
        "cadaver_bmi_main (implausible removed)": 1 - merged["cadaver_bmi_missing"],
        "cadaver_bmi_decomp_blanked (sensitivity)": 1 - merged["cadaver_bmi_decomp_blanked_missing"],
        "living_bmi": 1 - merged["living_bmi_missing"],
    })
    classes = ["natural_cardiac", "drug_alcohol", "injury"]
    cov_rows, test_rows = [], []
    for block in [c for c in present.columns if c != "level1"]:
        row = {"field_block": block, "overall_pct": round(100.0 * present[block].mean(), 1)}
        for cls in classes:
            row[cls + "_pct"] = round(100.0 * present.loc[present["level1"] == cls, block].mean(), 1)
        cov_rows.append(row)
        chi2, p_val, dof, _ = chi2_contingency(pd.crosstab(present["level1"], present[block]))
        test_rows.append({"field_block": block, "chi2": round(chi2, 2), "dof": dof, "p_value": p_val})
    cov_df, test_df = pd.DataFrame(cov_rows), pd.DataFrame(test_rows)
    cov_df.to_csv(config.TABLES_DIR / "coverage_by_class.csv", index=False)
    test_df.to_csv(config.TABLES_DIR / "missingness_tests.csv", index=False)
    print("\n--- Coverage by Class (% present) ---")
    print(cov_df.to_string(index=False))
    print("\n--- Missingness Chi-Square Tests ---")
    print(test_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    matrix = cov_df[["natural_cardiac_pct", "drug_alcohol_pct", "injury_pct"]].values
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Natural Cardiac", "Drug / Alcohol", "Injury"])
    ax.set_yticks(range(len(cov_df))); ax.set_yticklabels(cov_df["field_block"], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label="% Field Present")
    ax.set_title("Metadata Field Coverage by Cause-of-Death Class (%)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "coverage_heatmap.png", dpi=150)
    plt.close(fig)


def cramers_v(x_series, y_series):
    table = pd.crosstab(x_series, y_series)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))


def save_leakage_check(cohort, features):
    l1 = cohort[cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])].copy()
    rows = []
    for col in ["time_delay_after_death", "death_county", "identification_method_code",
                "decomposition_score", "settings", "death_year", "zip_code", "source_cart"]:
        values = l1[col].fillna("Missing").astype(str)
        rows.append({"forbidden_column": col, "n_categories": values.nunique(),
                     "cramers_v_with_level1": round(cramers_v(values, l1["level1"]), 3)})
    leak_df = pd.DataFrame(rows).sort_values("cramers_v_with_level1", ascending=False)
    leak_df.to_csv(config.TABLES_DIR / "leakage_check.csv", index=False)
    print("\n--- Leakage Check (Cramer's V with Level 1) ---")
    print(leak_df.to_string(index=False))

    # Why the decomposition-blanked BMI flag is excluded from the main analysis.
    m = l1[[config.KEY, "level1", "decomposition_score"]].merge(features, on=config.KEY)
    decomp = m["decomposition_score"].fillna("Missing")
    rows = []
    for flag in ["cadaver_bmi_missing", "cadaver_bmi_decomp_blanked_missing", "living_bmi_missing"]:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(m["level1"], m[flag]))
        rows.append({"flag": flag,
                     "pct_missing_overall": round(100 * m[flag].mean(), 1),
                     "pct_missing_natural_cardiac": round(100 * m.loc[m.level1 == "natural_cardiac", flag].mean(), 1),
                     "pct_missing_drug_alcohol": round(100 * m.loc[m.level1 == "drug_alcohol", flag].mean(), 1),
                     "pct_missing_injury": round(100 * m.loc[m.level1 == "injury", flag].mean(), 1),
                     "cramers_v_with_decomposition_score": round(cramers_v(m[flag], decomp), 3),
                     "cramers_v_with_level1": round(cramers_v(m[flag], m["level1"]), 3),
                     "chi2_p_vs_level1": p})
    pct_mod_adv = (100 * decomp.isin(["moderate decomposition", "advanced decomposition"])
                   .groupby(m["level1"]).mean()).round(1)
    bmi_df = pd.DataFrame(rows)
    bmi_df.to_csv(config.TABLES_DIR / "bmi_flag_audit.csv", index=False)
    print("\n--- Cadaver BMI flag audit ---")
    print(bmi_df.to_string(index=False))
    print("% moderate/advanced decomposition by class:", pct_mod_adv.to_dict())


def save_cart_invariance(dec):
    """Are the records of people who appear in two carts identical in every table?"""
    dup_ids = dec[dec.duplicated(config.KEY, keep=False)][config.KEY].unique()
    rows = []
    for name in build.TABLES:
        tab = dec if name == "decedents" else read_raw(name)
        tab = tab[tab[config.KEY].isin(dup_ids)].astype(str)   # filter once
        value_cols = [c for c in tab.columns if c != "source_cart"]
        n_dup_present, n_identical = 0, 0
        for pid, sub in tab.groupby(config.KEY):
            if sub["source_cart"].nunique() < 2:
                continue
            n_dup_present += 1
            sigs = {tuple(sorted(map(tuple, g[value_cols].values.tolist())))
                    for _, g in sub.groupby("source_cart")}
            n_identical += int(len(sigs) == 1)
        rows.append({"table": name, "duplicate_ids_total": len(dup_ids),
                     "duplicate_ids_with_rows_in_both_carts": n_dup_present,
                     "identical_across_carts": n_identical,
                     "pct_identical": round(100 * n_identical / n_dup_present, 1) if n_dup_present else np.nan})
    inv = pd.DataFrame(rows)
    inv.to_csv(config.TABLES_DIR / "cart_invariance_check.csv", index=False)
    print("\n--- Cart invariance check (duplicate records across carts) ---")
    print(inv.to_string(index=False))


if __name__ == "__main__":
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    cohort["age_years"] = pd.to_numeric(cohort["age_years"], errors="coerce")
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)

    dec = save_cohort_flow(cohort)
    save_table1(cohort)
    save_intent_tables(cohort)
    save_coverage_and_tests(cohort, features)
    save_leakage_check(cohort, features)
    save_cart_invariance(dec)
