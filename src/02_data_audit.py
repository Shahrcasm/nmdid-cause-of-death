import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import config


def save_cohort_flow(cohort):
    """Build and save results/tables/cohort_flow.csv."""
    l1_mask = cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])
    l2_mask = (cohort["level1"] == "injury") & (
        cohort["intent"].isin(["accident", "homicide", "suicide"])
    )
    l3_mask = (
        (cohort["level1"] == "injury")
        & (cohort["intent"] == "homicide")
        & (cohort["primary_cause_of_death"].isin([
            "Gunshot wound of", "Stab wound (slash, penetrating cut)"
        ]))
    )

    rows = [
        {"step": "1. Raw decedent rows across 3 carts", "count": 1500},
        {"step": "2. Unique decedents after deduplication", "count": 1457},
        {"step": "3. Excluded: age < 18 or missing age", "count": 7},
        {"step": "4. Adult decedents (age >= 18)", "count": len(cohort)},
        {"step": "5a. Excluded from Level 1: cardiac_conflict", "count": int((cohort["level1"] == "cardiac_conflict").sum())},
        {"step": "5b. Excluded from Level 1: other causes", "count": int((cohort["level1"] == "other").sum())},
        {"step": "6. Level 1 cohort (cardiac / drug_alcohol / injury)", "count": int(l1_mask.sum())},
        {"step": "7. Level 2 cohort (injury: accident / homicide / suicide)", "count": int(l2_mask.sum())},
        {"step": "8. Level 3 cohort (injury homicide: gunshot vs stab)", "count": int(l3_mask.sum())},
    ]
    flow_df = pd.DataFrame(rows)
    flow_df.to_csv(config.TABLES_DIR / "cohort_flow.csv", index=False)
    print("--- Cohort Flow ---")
    print(flow_df.to_string(index=False))


def save_table1(cohort):
    """Build and save results/tables/table1_by_class.csv."""
    classes = ["natural_cardiac", "drug_alcohol", "injury"]
    rows = []

    for cls in classes:
        sub = cohort[cohort["level1"] == cls]
        ages = sub["age_years"]
        median_age = ages.median()
        q25 = ages.quantile(0.25)
        q75 = ages.quantile(0.75)
        pct_male = 100.0 * (sub["sex_code"] == "Male").mean()

        rows.append({
            "level1_class": cls,
            "n": len(sub),
            "age_median": round(median_age, 1),
            "age_q25": round(q25, 1),
            "age_q75": round(q75, 1),
            "age_median_iqr": f"{int(round(median_age))} ({int(round(q25))}-{int(round(q75))})",
            "male_n": int((sub["sex_code"] == "Male").sum()),
            "male_pct": round(pct_male, 1),
        })

    table1_df = pd.DataFrame(rows)
    table1_df.to_csv(config.TABLES_DIR / "table1_by_class.csv", index=False)
    print("\n--- Table 1 by Class ---")
    print(table1_df.to_string(index=False))


def save_intent_by_cause(cohort):
    """Build and save results/tables/intent_by_cause.csv."""
    ct = pd.crosstab(
        cohort["primary_cause_of_death"].fillna("Missing"),
        cohort["intent"],
        margins=True,
    )
    ct.to_csv(config.TABLES_DIR / "intent_by_cause.csv")


def save_coverage_and_tests(cohort, features):
    """Coverage by class, chi-square tests, and heatmap."""
    tables = {
        "med": pd.concat([
            pd.read_csv(config.RAW_DIR / c / "medical-history.csv", dtype=str, keep_default_na=False, na_values=[""])
            for c in config.CARTS
        ], ignore_index=True),
        "sub": pd.concat([
            pd.read_csv(config.RAW_DIR / c / "substance-usages.csv", dtype=str, keep_default_na=False, na_values=[""])
            for c in config.CARTS
        ], ignore_index=True),
    }
    l1 = cohort[cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])].copy()
    merged = l1[[config.KEY, "level1", "cadaver_weight", "cadaver_length"]].merge(
        features, on=config.KEY
    )

    raw_cad_bmi_present = (
        pd.to_numeric(merged["cadaver_weight"], errors="coerce").notna()
        & pd.to_numeric(merged["cadaver_length"], errors="coerce").notna()
    ).astype(int)

    # Drug use row present in substance-usages (excluding blank and 'None').
    sub_df = tables["sub"]
    sub_clean = sub_df["substance_usage_type"].astype(str).str.strip()
    drug_ids = set(sub_df.loc[sub_df["substance_usage_type"].notna() & (sub_clean != "None") & (sub_clean != ""), config.KEY].astype(str).str.strip())
    drug_user_present = merged[config.KEY].isin(drug_ids).astype(int)

    # Diagnosis row present in medical-history.
    med_df = tables["med"]
    dx_rows = med_df[
        (med_df["medical_history_type"].astype(str).str.strip() == "Diagnosis")
        & med_df["medical_history_detail"].notna()
        & (med_df["medical_history_detail"].astype(str).str.strip() != "")
    ]
    dx_ids = set(dx_rows[config.KEY].astype(str).str.strip())
    dx_row_present = merged[config.KEY].isin(dx_ids).astype(int)

    # Drinking & tobacco row present.
    drink_cols = [c for c in merged.columns if c.startswith("drink_") and c != "drink_missing"]
    drink_row_present = (merged[drink_cols].sum(axis=1) > 0).astype(int)

    tob_cols = [c for c in merged.columns if c.startswith("tobtype_") or c.startswith("tobuse_")]
    tob_row_present = (merged[tob_cols].sum(axis=1) > 0).astype(int)

    present_flags = pd.DataFrame({
        "level1": merged["level1"],
        "drinking": drink_row_present,
        "tobacco": tob_row_present,
        "drug_use": drug_user_present,
        "diagnosis": dx_row_present,
        "medications": merged["any_medication"],
        "marital": 1 - merged["marital_missing"],
        "nok_interview": merged["nok_interview"],
        "cadaver_bmi_raw": raw_cad_bmi_present,
        "cadaver_bmi_clean": 1 - merged["cadaver_bmi_missing"],
        "living_bmi": 1 - merged["living_bmi_missing"],
    })

    classes = ["natural_cardiac", "drug_alcohol", "injury"]
    cov_rows = []
    test_rows = []

    block_names = [c for c in present_flags.columns if c != "level1"]
    for block in block_names:
        overall_pct = 100.0 * present_flags[block].mean()
        row = {"field_block": block, "overall_pct": round(overall_pct, 1)}
        for cls in classes:
            sub = present_flags[present_flags["level1"] == cls]
            row[cls + "_pct"] = round(100.0 * sub[block].mean(), 1)
        cov_rows.append(row)

        # Chi-square test of independence across the 3 Level-1 classes.
        table = pd.crosstab(present_flags["level1"], present_flags[block])
        chi2, p_val, dof, _ = chi2_contingency(table)
        test_rows.append({
            "field_block": block,
            "chi2": round(chi2, 2),
            "dof": dof,
            "p_value": p_val,
        })

    cov_df = pd.DataFrame(cov_rows)
    test_df = pd.DataFrame(test_rows)
    cov_df.to_csv(config.TABLES_DIR / "coverage_by_class.csv", index=False)
    test_df.to_csv(config.TABLES_DIR / "missingness_tests.csv", index=False)

    print("\n--- Coverage by Class (% present) ---")
    print(cov_df.to_string(index=False))
    print("\n--- Missingness Chi-Square Tests ---")
    print(test_df.to_string(index=False))

    # Plot coverage heatmap.
    fig, ax = plt.subplots(figsize=(7, 5))
    matrix = cov_df[["natural_cardiac_pct", "drug_alcohol_pct", "injury_pct"]].values
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Natural Cardiac", "Drug / Alcohol", "Injury"])
    ax.set_yticks(range(len(cov_df)))
    ax.set_yticklabels(cov_df["field_block"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax, label="% Field Present")
    ax.set_title("Metadata Field Coverage by Cause-of-Death Class (%)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "coverage_heatmap.png", dpi=150)
    plt.close(fig)


def cramers_v(x_series, y_series):
    """Calculate standard Cramér's V between two categorical columns."""
    table = pd.crosstab(x_series, y_series)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / (n * min_dim)))


def save_leakage_check(cohort):
    """Compute Cramér's V of forbidden columns with Level-1 class."""
    l1 = cohort[cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])].copy()

    check_cols = [
        ("time_delay_after_death", "time_delay_after_death"),
        ("death_county", "death_county"),
        ("identification_method_code", "identification_method_code"),
        ("decomposition_score", "decomposition_score"),
        ("settings", "settings"),
        ("death_year", "death_year"),
        ("zip_code", "zip_code"),
        ("source_cart", "source_cart"),
    ]

    rows = []
    for label_name, col in check_cols:
        values = l1[col].fillna("Missing").astype(str)
        v = cramers_v(values, l1["level1"])
        rows.append({
            "forbidden_column": label_name,
            "n_categories": values.nunique(),
            "cramers_v_with_level1": round(v, 3),
        })

    leak_df = pd.DataFrame(rows).sort_values("cramers_v_with_level1", ascending=False)
    leak_df.to_csv(config.TABLES_DIR / "leakage_check.csv", index=False)
    print("\n--- Leakage Check (Cramér's V with Level 1) ---")
    print(leak_df.to_string(index=False))


if __name__ == "__main__":
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    cohort["age_years"] = pd.to_numeric(cohort["age_years"], errors="coerce")
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)

    save_cohort_flow(cohort)
    save_table1(cohort)
    save_intent_by_cause(cohort)
    save_coverage_and_tests(cohort, features)
    save_leakage_check(cohort)