"""Phase 3 / RQ2: which factors carry the signal, and is it stable?

  1. Univariate chi-square / Kruskal-Wallis with BH-FDR (unchanged).
  2. Age- and sex-adjusted multinomial logistic regression, one factor at a
     time (likelihood-ratio test, BH-FDR, odds ratios vs the reference class).
  3. Sex-stratified Random Forest permutation importance computed OUT OF FOLD
     (fit on 4 folds, permute on the held-out fold, average over folds).
  4. TreeSHAP on out-of-fold predictions of the Random Forest M2 model.
  5. Age-band models with repeated CV and bootstrap CIs.

Outputs: rq2_univariate_level{1,2}.csv, rq2_adjusted_level{1,2}.csv,
         rq2_rf_importance_by_sex.csv, rq2_shap_importance.csv, rq2_age_bands.csv,
         figures/rq2_rf_importance_by_sex.png, figures/rq2_shap_summary.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal, chi2 as chi2_dist
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import shap
import config
import model_utils as mu

MIN_EVENTS = 10   # factors with fewer positives (or fewer non-positives) are not modelled


def run_univariate_tests(df, label_col, classes, factor_cols, level_label):
    rows = []
    for col in factor_cols:
        if col in mu.CONTINUOUS_COLS:
            groups = [df.loc[df[label_col] == c, col].dropna().values for c in classes]
            groups = [g for g in groups if len(g) > 0]
            stat, p_val = kruskal(*groups) if len(groups) >= 2 else (0.0, 1.0)
            row = {"level": level_label, "feature": col, "test_type": "kruskal_continuous",
                   "stat": round(float(stat), 2), "p_raw": float(p_val), "age_kruskal_p": np.nan}
            for c in classes:
                vals = df.loc[df[label_col] == c, col].dropna()
                row[c + "_rate_or_mean"] = round(float(vals.mean()), 2) if len(vals) else np.nan
            rows.append(row)
            continue
        table = pd.crosstab(df[label_col], df[col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_val, _, _ = chi2_contingency(table)
        age_pos = df.loc[df[col] == 1, "age_years"].dropna().values
        age_neg = df.loc[df[col] == 0, "age_years"].dropna().values
        age_p = kruskal(age_pos, age_neg)[1] if len(age_pos) >= 3 and len(age_neg) >= 3 else np.nan
        row = {"level": level_label, "feature": col, "test_type": "chi2_binary",
               "stat": round(float(chi2), 2), "p_raw": float(p_val), "age_kruskal_p": age_p}
        for c in classes:
            row[c + "_rate_or_mean"] = round(100.0 * float(df.loc[df[label_col] == c, col].mean()), 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(out["p_raw"].values, alpha=0.05, method="fdr_bh")
    out["p_fdr_bh"], out["sig_fdr_05"] = p_fdr, reject
    return out.sort_values("p_fdr_bh")


def _fit_mnlogit(y, X):
    for method in ("newton", "bfgs"):
        try:
            res = sm.MNLogit(y, X).fit(method=method, maxiter=300, disp=0)
            if res.mle_retvals.get("converged", False) and np.isfinite(res.llf):
                return res
        except Exception:
            continue
    return None


def run_adjusted_tests(df, label_col, classes, factor_cols, level_label):
    """Per factor: multinomial logit of class on age + sex + factor versus age + sex.
    Reports the 2-df likelihood-ratio p (BH-FDR) and odds ratios per unit
    (binary: 0/1; continuous: per SD among people with the value recorded)."""
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_all = df[label_col].map(class_to_int).values
    age_z = ((df["age_years"] - df["age_years"].mean()) / df["age_years"].std()).values
    male = df["male"].values.astype(float)
    rows = []
    for col in factor_cols:
        x = df[col].astype(float).values
        keep = ~np.isnan(x)
        note = ""
        if col in mu.CONTINUOUS_COLS:
            xs = x[keep]
            x_in = (xs - xs.mean()) / xs.std() if xs.std() > 0 else xs
        else:
            x_in = x[keep]
            n_pos = int(x_in.sum())
            if n_pos < MIN_EVENTS or (len(x_in) - n_pos) < MIN_EVENTS:
                note = f"not modelled: {n_pos} positives"
        row = {"level": level_label, "feature": col, "n_used": int(keep.sum()), "note": note,
               "lr_chi2": np.nan, "p_raw": np.nan}
        for c in classes[1:]:
            row[f"OR_{c}_vs_{classes[0]}"] = np.nan
        if not note:
            y = y_all[keep]
            X0 = sm.add_constant(np.column_stack([age_z[keep], male[keep]]))
            X1 = sm.add_constant(np.column_stack([age_z[keep], male[keep], x_in]))
            r0, r1 = _fit_mnlogit(y, X0), _fit_mnlogit(y, X1)
            if r0 is None or r1 is None or len(np.unique(y)) < len(classes):
                row["note"] = "not estimable (non-convergence / separation)"
            else:
                lr = 2 * (r1.llf - r0.llf)
                row["lr_chi2"] = round(float(lr), 2)
                row["p_raw"] = float(chi2_dist.sf(max(lr, 0), df=len(classes) - 1))
                params = np.asarray(r1.params)          # (k, n_classes-1)
                for j, c in enumerate(classes[1:]):
                    row[f"OR_{c}_vs_{classes[0]}"] = round(float(np.exp(params[3, j])), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    ok = out["p_raw"].notna()
    out["p_fdr_bh"], out["sig_fdr_05"] = np.nan, False
    if ok.any():
        reject, p_fdr, _, _ = multipletests(out.loc[ok, "p_raw"].values, alpha=0.05, method="fdr_bh")
        out.loc[ok, "p_fdr_bh"], out.loc[ok, "sig_fdr_05"] = p_fdr, reject
    return out.sort_values("p_fdr_bh", na_position="last")


def oof_folds(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    return list(skf.split(X, y))


def run_sex_stratified_rf_importance(df, label_col, classes, feature_cols):
    class_to_int = {c: i for i, c in enumerate(classes)}
    imp_df = pd.DataFrame({"feature": feature_cols})
    ns = {}
    for sex_label, male_val in [("Male", 1), ("Female", 0)]:
        sub = df[df["male"] == male_val].reset_index(drop=True)
        X, y = sub[feature_cols], sub[label_col].map(class_to_int).values
        ns[sex_label] = len(sub)
        fold_imps = []
        for tr, te in oof_folds(X, y):
            model = mu.make_model("RF").fit(X.iloc[tr], y[tr])
            perm = permutation_importance(model, X.iloc[te], y[te], scoring="roc_auc_ovr",
                                          n_repeats=5, random_state=config.SEED, n_jobs=1)
            fold_imps.append(perm.importances_mean)
        fold_imps = np.array(fold_imps)
        imp_df[sex_label + "_imp_mean"] = fold_imps.mean(axis=0)
        imp_df[sex_label + "_imp_fold_sd"] = fold_imps.std(axis=0, ddof=1)
    imp_df["overall_rank_score"] = (imp_df["Male_imp_mean"] + imp_df["Female_imp_mean"]) / 2.0
    imp_df = imp_df.sort_values("overall_rank_score", ascending=False)
    imp_df.to_csv(config.TABLES_DIR / "rq2_rf_importance_by_sex.csv", index=False)

    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos, height = np.arange(len(top)), 0.38
    ax.barh(y_pos - height / 2, top["Male_imp_mean"], height, xerr=top["Male_imp_fold_sd"], label=f"Male (n={ns['Male']})")
    ax.barh(y_pos + height / 2, top["Female_imp_mean"], height, xerr=top["Female_imp_fold_sd"], label=f"Female (n={ns['Female']})")
    ax.set_yticks(y_pos); ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Out-of-fold permutation importance (macro OvR AUC drop), mean +/- SD over folds")
    ax.set_title("Top 15 Features, Sex-Stratified Random Forest (Level 1, M2)")
    ax.legend(); ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "rq2_rf_importance_by_sex.png", dpi=150)
    plt.close(fig)
    return imp_df


def run_shap_analysis(df, label_col, classes, feature_cols):
    """Mean |SHAP| per class from TreeSHAP applied to each person's held-out fold model."""
    class_to_int = {c: i for i, c in enumerate(classes)}
    X, y = df[feature_cols], df[label_col].map(class_to_int).values
    abs_sum = np.zeros((len(feature_cols), len(classes)))
    for tr, te in oof_folds(X, y):
        model = mu.make_model("RF").fit(X.iloc[tr], y[tr])
        X_te = pd.DataFrame(model.named_steps["imputer"].transform(X.iloc[te]), columns=feature_cols)
        sv = shap.TreeExplainer(model.named_steps["clf"]).shap_values(X_te)
        arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (n, f, c)
        abs_sum += np.abs(arr).sum(axis=0)
    mean_abs = abs_sum / len(y)
    shap_df = pd.DataFrame({"feature": feature_cols})
    for i, c in enumerate(classes):
        shap_df["mean_abs_shap_" + c] = mean_abs[:, i]
    shap_df["mean_abs_shap_overall"] = mean_abs.mean(axis=1)
    shap_df = shap_df.sort_values("mean_abs_shap_overall", ascending=False)
    shap_df.to_csv(config.TABLES_DIR / "rq2_shap_importance.csv", index=False)

    top = shap_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(top))
    for c in classes:
        vals = top["mean_abs_shap_" + c].values
        ax.barh(top["feature"], vals, left=left, label=c)
        left += vals
    ax.set_xlabel("Mean |SHAP value| on out-of-fold predictions (Random Forest, M2)")
    ax.set_title("SHAP Feature Attribution by Cause-of-Death Class (Top 15)")
    ax.legend(); ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "rq2_shap_summary.png", dpi=150)
    plt.close(fig)
    return shap_df


def run_age_band_analysis(df, label_col, classes, nested):
    bands = [("18-39", (df["age_years"] >= 18) & (df["age_years"] <= 39)),
             ("40-59", (df["age_years"] >= 40) & (df["age_years"] <= 59)),
             ("60+", df["age_years"] >= 60)]
    rows = []
    for band_name, mask in bands:
        sub = df[mask].reset_index(drop=True)
        res = mu.run_sets(sub, label_col, classes, nested, "HistGB")
        for r in res["metrics"]:
            r["age_band"] = band_name
            r["class_counts"] = str(sub[label_col].value_counts().to_dict())
        rows += res["metrics"]
    band_df = pd.DataFrame(rows)
    front = ["age_band", "feature_set", "n", "class_counts", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high",
             "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high"]
    band_df = band_df[front]
    band_df.to_csv(config.TABLES_DIR / "rq2_age_bands.csv", index=False)
    return band_df


if __name__ == "__main__":
    cohort, features = mu.load_data()
    base_cols, life_cols, med_cols = mu.get_column_groups(features)
    nested = mu.get_nested_sets(features)
    all_factor_cols = life_cols + med_cols

    l1_df = cohort[cohort["level1"].isin(mu.LEVEL1_CLASSES)][[config.KEY, "level1"]].merge(features, on=config.KEY)
    l2_df = cohort[(cohort["level1"] == "injury") & cohort["intent"].isin(mu.LEVEL2_CLASSES)][
        [config.KEY, "intent"]].merge(features, on=config.KEY)

    print("1. Univariate tests with BH-FDR...", flush=True)
    u1 = run_univariate_tests(l1_df, "level1", mu.LEVEL1_CLASSES, all_factor_cols, "Level1")
    u2 = run_univariate_tests(l2_df, "intent", mu.LEVEL2_CLASSES, all_factor_cols, "Level2")
    u1.to_csv(config.TABLES_DIR / "rq2_univariate_level1.csv", index=False)
    u2.to_csv(config.TABLES_DIR / "rq2_univariate_level2.csv", index=False)
    print("  Level 1 FDR-significant:", int(u1["sig_fdr_05"].sum()), "of", len(u1),
          "| Level 2:", int(u2["sig_fdr_05"].sum()), "of", len(u2))

    print("2. Age- and sex-adjusted multinomial models...", flush=True)
    a1 = run_adjusted_tests(l1_df, "level1", mu.LEVEL1_CLASSES, all_factor_cols, "Level1")
    a2 = run_adjusted_tests(l2_df, "intent", mu.LEVEL2_CLASSES, all_factor_cols, "Level2")
    a1.to_csv(config.TABLES_DIR / "rq2_adjusted_level1.csv", index=False)
    a2.to_csv(config.TABLES_DIR / "rq2_adjusted_level2.csv", index=False)
    print("  Level 1 adjusted FDR-significant:", int(a1["sig_fdr_05"].sum()), "of", int(a1["p_raw"].notna().sum()), "modelled",
          "| Level 2:", int(a2["sig_fdr_05"].sum()), "of", int(a2["p_raw"].notna().sum()))
    print(a1.head(12)[["feature", "lr_chi2", "p_fdr_bh"] + [c for c in a1.columns if c.startswith("OR_")]].to_string(index=False))

    print("3. Sex-stratified out-of-fold RF permutation importance...", flush=True)
    imp_df = run_sex_stratified_rf_importance(l1_df, "level1", mu.LEVEL1_CLASSES, ["age_years"] + all_factor_cols)
    print(imp_df.head(10).to_string(index=False))

    print("4. Out-of-fold TreeSHAP (Random Forest, M2)...", flush=True)
    shap_df = run_shap_analysis(l1_df, "level1", mu.LEVEL1_CLASSES, nested["M2_med_history"])
    print(shap_df.head(10).to_string(index=False))

    print("5. Age-band models (HistGB, repeated CV + bootstrap)...", flush=True)
    print(run_age_band_analysis(l1_df, "level1", mu.LEVEL1_CLASSES, nested).to_string(index=False))
