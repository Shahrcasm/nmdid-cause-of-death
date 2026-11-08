import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import shap
import config


def get_feature_columns(features):
    """Return lists of base, lifestyle, and medical history columns."""
    rq3_only = ["n_medications", "any_medication", "med_missing", "nok_interview"]
    med_hist_cols = [
        c for c in features.columns
        if c.startswith("dx_") or c.startswith("medhist_")
    ]
    base_cols = ["age_years", "male"]
    lifestyle_cols = [
        c for c in features.columns
        if c not in ([config.KEY] + base_cols + med_hist_cols + rq3_only)
    ]
    return base_cols, lifestyle_cols, med_hist_cols


def run_univariate_tests(df, label_col, classes, factor_cols, level_label):
    """Chi-square across classes + Kruskal-Wallis age check + BH-FDR correction."""
    rows = []
    continuous_cols = ["cadaver_bmi", "living_bmi", "edu_level", "ses_adult", "ses_child"]

    for col in factor_cols:
        if col in continuous_cols:
            # Kruskal-Wallis test across the classes on non-missing values.
            groups = [
                df.loc[df[label_col] == c, col].dropna().values
                for c in classes
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                stat, p_val = kruskal(*groups)
            else:
                stat, p_val = 0.0, 1.0
            row = {
                "level": level_label,
                "feature": col,
                "test_type": "kruskal_continuous",
                "stat": round(float(stat), 2),
                "p_raw": float(p_val),
                "age_kruskal_p": np.nan,
            }
            for c in classes:
                vals = df.loc[df[label_col] == c, col].dropna()
                row[c + "_rate_or_mean"] = round(float(vals.mean()), 2) if len(vals) > 0 else np.nan
            rows.append(row)
            continue

        # Binary 0/1 indicator column: chi-square across classes.
        table = pd.crosstab(df[label_col], df[col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_val, _, _ = chi2_contingency(table)

        # Kruskal-Wallis test of age between factor=1 and factor=0.
        age_pos = df.loc[df[col] == 1, "age_years"].dropna().values
        age_neg = df.loc[df[col] == 0, "age_years"].dropna().values
        if len(age_pos) >= 3 and len(age_neg) >= 3:
            _, age_p = kruskal(age_pos, age_neg)
        else:
            age_p = np.nan

        row = {
            "level": level_label,
            "feature": col,
            "test_type": "chi2_binary",
            "stat": round(float(chi2), 2),
            "p_raw": float(p_val),
            "age_kruskal_p": float(age_p) if not np.isnan(age_p) else np.nan,
        }
        for c in classes:
            sub = df[df[label_col] == c]
            row[c + "_rate_or_mean"] = round(100.0 * float(sub[col].mean()), 1)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(out_df["p_raw"].values, alpha=0.05, method="fdr_bh")
    out_df["p_fdr_bh"] = p_fdr
    out_df["sig_fdr_05"] = reject
    out_df = out_df.sort_values("p_fdr_bh")
    return out_df


def run_sex_stratified_rf_importance(df, label_col, classes, feature_cols):
    """Random-forest permutation importance separately for Males and Females."""
    class_to_int = {c: i for i, c in enumerate(classes)}
    imputer = SimpleImputer(strategy="median")

    sex_groups = [("Male", 1), ("Female", 0)]
    imp_df = pd.DataFrame({"feature": feature_cols})

    for sex_label, male_val in sex_groups:
        sub = df[df["male"] == male_val].copy()
        X = pd.DataFrame(imputer.fit_transform(sub[feature_cols]), columns=feature_cols)
        y = sub[label_col].map(class_to_int).values

        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=config.SEED, n_jobs=1
        )
        rf.fit(X, y)
        perm = permutation_importance(
            rf, X, y, scoring="roc_auc_ovr", n_repeats=5, random_state=config.SEED, n_jobs=1
        )
        imp_df[sex_label + "_imp_mean"] = perm.importances_mean
        imp_df[sex_label + "_imp_std"] = perm.importances_std

    imp_df["overall_rank_score"] = (imp_df["Male_imp_mean"] + imp_df["Female_imp_mean"]) / 2.0
    imp_df = imp_df.sort_values("overall_rank_score", ascending=False)
    imp_df.to_csv(config.TABLES_DIR / "rq2_rf_importance_by_sex.csv", index=False)

    # Plot top 15 features by sex.
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(top))
    height = 0.38
    ax.barh(y_pos - height / 2, top["Male_imp_mean"], height, label="Male (n=975)")
    ax.barh(y_pos + height / 2, top["Female_imp_mean"], height, label="Female (n=405)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Permutation Importance (Macro AUC drop)")
    ax.set_title("Top 15 Features by Sex-Stratified Random Forest (Level 1)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "rq2_rf_importance_by_sex.png", dpi=150)
    plt.close(fig)
    return imp_df


def run_shap_analysis(df, label_col, classes, feature_cols):
    """Compute TreeExplainer SHAP values on Level-1 M2 model and save table + plot."""
    class_to_int = {c: i for i, c in enumerate(classes)}
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df[feature_cols]), columns=feature_cols)
    y = df[label_col].map(class_to_int).values

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=config.SEED, n_jobs=1
    )
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)

    # shap_vals can be a 3D array (n_samples, n_features, n_classes) or list of arrays.
    if isinstance(shap_vals, list):
        mean_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_vals]
    else:
        mean_per_class = [np.mean(np.abs(shap_vals[:, :, i]), axis=0) for i in range(len(classes))]

    shap_df = pd.DataFrame({"feature": feature_cols})
    for i, c_name in enumerate(classes):
        shap_df["mean_abs_shap_" + c_name] = mean_per_class[i]
    shap_df["mean_abs_shap_overall"] = np.mean(mean_per_class, axis=0)
    shap_df = shap_df.sort_values("mean_abs_shap_overall", ascending=False)
    shap_df.to_csv(config.TABLES_DIR / "rq2_shap_importance.csv", index=False)

    # Save clean stacked bar chart of top 15 SHAP features across the 3 classes.
    top = shap_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(top))
    for c_name in classes:
        vals = top["mean_abs_shap_" + c_name].values
        ax.barh(top["feature"], vals, left=left, label=c_name)
        left += vals
    ax.set_xlabel("Mean |SHAP value| (impact on model output probability)")
    ax.set_title("SHAP Feature Attribution by Cause-of-Death Class (Top 15, M2)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "rq2_shap_summary.png", dpi=150)
    plt.close(fig)
    return shap_df


def run_age_band_analysis(df, label_col, classes, base_cols, life_cols, med_cols):
    """Evaluate M0, M1, M2 within age bands: 18-39, 40-59, 60+."""
    class_to_int = {c: i for i, c in enumerate(classes)}
    bands = [
        ("18-39", (df["age_years"] >= 18) & (df["age_years"] <= 39)),
        ("40-59", (df["age_years"] >= 40) & (df["age_years"] <= 59)),
        ("60+", df["age_years"] >= 60),
    ]
    sets = {
        "M0_base": base_cols,
        "M1_lifestyle": base_cols + life_cols,
        "M2_med_history": base_cols + life_cols + med_cols,
    }

    rows = []
    for band_name, mask in bands:
        sub = df[mask].reset_index(drop=True)
        y = sub[label_col].map(class_to_int).values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

        for set_name, cols in sets.items():
            X = sub[cols]
            oof = np.zeros((len(sub), len(classes)))
            for tr, te in skf.split(X, y):
                clf = HistGradientBoostingClassifier(
                    max_iter=50, early_stopping=True, random_state=config.SEED
                )
                clf.fit(X.iloc[tr], y[tr])
                oof[te] = clf.predict_proba(X.iloc[te])
            auc = roc_auc_score(y, oof, multi_class="ovr", average="macro")
            rows.append({
                "age_band": band_name,
                "n": len(sub),
                "feature_set": set_name,
                "macro_auc": round(float(auc), 3),
            })

    band_df = pd.DataFrame(rows)
    band_df.to_csv(config.TABLES_DIR / "rq2_age_bands.csv", index=False)
    return band_df


if __name__ == "__main__":
    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)

    base_cols, life_cols, med_cols = get_feature_columns(features)
    all_factor_cols = life_cols + med_cols

    l1_classes = ["natural_cardiac", "drug_alcohol", "injury"]
    l1_df = cohort[cohort["level1"].isin(l1_classes)][[config.KEY, "level1"]].merge(
        features, on=config.KEY
    )

    l2_classes = ["accident", "homicide", "suicide"]
    l2_df = cohort[
        (cohort["level1"] == "injury") & (cohort["intent"].isin(l2_classes))
    ][[config.KEY, "intent"]].merge(features, on=config.KEY)

    print("1. Running univariate tests with BH-FDR correction...")
    u1 = run_univariate_tests(l1_df, "level1", l1_classes, all_factor_cols, "Level1")
    u2 = run_univariate_tests(l2_df, "intent", l2_classes, all_factor_cols, "Level2")
    u1.to_csv(config.TABLES_DIR / "rq2_univariate_level1.csv", index=False)
    u2.to_csv(config.TABLES_DIR / "rq2_univariate_level2.csv", index=False)
    print("  Level 1 FDR-significant factors:", int(u1["sig_fdr_05"].sum()), "of", len(u1))
    print("  Level 2 FDR-significant factors:", int(u2["sig_fdr_05"].sum()), "of", len(u2))

    print("2. Running sex-stratified Random Forest permutation importance...")
    m2_cols_no_sex = ["age_years"] + all_factor_cols
    imp_df = run_sex_stratified_rf_importance(l1_df, "level1", l1_classes, m2_cols_no_sex)
    print(imp_df.head(10).to_string(index=False))

    print("3. Running SHAP TreeExplainer analysis...")
    m2_cols = base_cols + all_factor_cols
    shap_df = run_shap_analysis(l1_df, "level1", l1_classes, m2_cols)
    print(shap_df.head(10).to_string(index=False))

    print("4. Running age-band stratified analysis...")
    band_df = run_age_band_analysis(l1_df, "level1", l1_classes, base_cols, life_cols, med_cols)
    print(band_df.to_string(index=False))

