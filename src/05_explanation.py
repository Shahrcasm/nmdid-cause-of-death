"""Step 5 - Explanation: why does the estimate survive missing information?

  A. Content versus blank pattern. Models that see only WHICH sections were
     filled in (blank pattern), only WHAT the filled-in sections say (content;
     the blank pattern is destroyed by giving blank people the values of a
     random recorded donor), both, and neither (age + sex).
  B. Complete cases: people with drinking, tobacco and diagnoses all recorded.
  C. Factor associations: univariate tests and age- and sex-adjusted
     multinomial models (one factor at a time), Benjamini-Hochberg FDR.
  D. TreeSHAP on held-out folds (random forest, M2).

Outputs: tables/explanation_regimes.csv, explanation_gain_shares.csv,
         factors_univariate_level{1,2}.csv, factors_adjusted_level{1,2}.csv,
         explanation_shap.csv, figures/explanation_content_vs_blanks.png,
         figures/explanation_shap.png
"""
import time
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
import shap
import config
import model_utils as mu

REGIME_LABELS = {
    "M0_base": "Age + sex only",
    "M0_plus_L_flags_only": "Blank pattern only (lifestyle)",
    "M1_values_blanks_randomised": "Content only (lifestyle)",
    "M1_values_implicit_blanks": "Content with blanks left visible (lifestyle)",
    "M1_full": "Content + blank pattern (lifestyle)",
    "M0_plus_LH_flags_only": "Blank pattern only (all sections)",
    "M2_values_blanks_randomised": "Content only (all sections)",
    "M2_values_implicit_blanks": "Content with blanks left visible (all sections)",
    "M2_full": "Content + blank pattern (all sections)",
    "M1_lifestyle": "Age + sex + lifestyle",
    "M2_med_history": "Age + sex + lifestyle + medical history",
}


MIN_EVENTS = 10


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
    shap_df.to_csv(config.TABLES_DIR / "explanation_shap.csv", index=False)

    top = shap_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(top))
    for c in classes:
        vals = top["mean_abs_shap_" + c].values
        ax.barh(top["feature"], vals, left=left, label=mu.CLASS_LABELS.get(c, c))
        left += vals
    ax.set_xlabel("Mean |SHAP value| on out-of-fold predictions (Random Forest, M2)")
    ax.set_title("Which inputs move the estimate most (top 15, held-out people)")
    ax.legend(); ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "explanation_shap.png", dpi=150)
    plt.close(fig)
    return shap_df


def encoding_sets(base, lifestyle, med_hist):
    life_flags = [c for c in lifestyle if c.endswith("_missing")]
    life_values = [c for c in lifestyle if not c.endswith("_missing")]
    med_flags = [c for c in med_hist if c.endswith("_missing")]
    med_values = [c for c in med_hist if not c.endswith("_missing")]
    return {
        "M0_base": base,
        "M0_plus_L_flags_only": base + life_flags,
        "M1_values_blanks_randomised": base + life_values,
        "M1_values_implicit_blanks": base + life_values,
        "M1_full": base + lifestyle,
        "M0_plus_LH_flags_only": base + life_flags + med_flags,
        "M2_values_blanks_randomised": base + life_values + med_values,
        "M2_values_implicit_blanks": base + life_values + med_values,
        "M2_full": base + lifestyle + med_hist,
    }


RANDOMISED = {"M1_values_blanks_randomised", "M2_values_blanks_randomised"}


def run_encoding(df, label_col, classes, enc, model_name, all_cols):
    """run_sets, but the randomised regimes are fitted on the blank-randomised copy."""
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = df[label_col].map(class_to_int).values
    df_rand = mu.randomise_blanks(df, all_cols)
    probs, per_rep = {}, {}
    for name, cols in enc.items():
        src = df_rand if name in RANDOMISED else df
        probs[name], per_rep[name] = mu.repeated_oof_probs(src[cols], y, model_name)
    rows, _ = mu.bootstrap_compare(y, probs, reference="M0_base", per_repeat=per_rep)
    for r in rows:
        r["model"] = model_name
    return rows


def gain_shares(rows, level, model):
    """Share of the full gain over age + sex that each partial regime recovers.
    The two shares come from separate models, so they need not add up to 100."""
    d = {r["feature_set"]: r for r in rows}
    out = []
    for block, full, flags, content in [("lifestyle", "M1_full", "M0_plus_L_flags_only", "M1_values_blanks_randomised"),
                                        ("lifestyle + medical history", "M2_full", "M0_plus_LH_flags_only", "M2_values_blanks_randomised")]:
        row = {"level": level, "model": model, "block": block}
        for metric, key in [("auc", "auc_gain_vs_ref"), ("accuracy", "acc_gain_vs_ref")]:
            g_full, g_flag, g_cont = d[full][key], d[flags][key], d[content][key]
            row.update({f"{metric}_gain_full": g_full, f"{metric}_gain_blank_pattern_only": g_flag,
                        f"{metric}_gain_content_only": g_cont,
                        f"{metric}_share_blank_pattern_pct": round(100 * g_flag / g_full) if g_full > 0 else np.nan,
                        f"{metric}_share_content_pct": round(100 * g_cont / g_full) if g_full > 0 else np.nan})
        out.append(row)
    return out


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    base, lifestyle, med_hist = mu.get_column_groups(features)
    nested = mu.get_nested_sets(features)
    enc = encoding_sets(base, lifestyle, med_hist)
    all_cols = nested["M2_med_history"]
    l1, y1 = mu.level1_frame(cohort, features)
    l2, y2 = mu.level2_frame(cohort, features)

    # ---------- A. Content versus blank pattern ----------
    rows, share_rows = [], []
    for level, df, label, classes, models in [("Level1", l1, "level1", mu.LEVEL1_CLASSES, ["HistGB", "LogReg"]),
                                              ("Level2", l2, "intent", mu.LEVEL2_CLASSES, ["HistGB"])]:
        for model_name in models:
            r = run_encoding(df, label, classes, enc, model_name, all_cols)
            rows += mu.tag(r, level=level, analysis="content_vs_blank_pattern", subset="all people")
            share_rows += gain_shares(r, level, model_name)
            mu.log(f"A {level} {model_name} done", t0)

    # ---------- B. Complete cases ----------
    cc = (l1["drink_missing"] == 0) & (l1["tob_missing"] == 0) & (l1["dx_missing"] == 0)
    l1_cc = l1[cc].reset_index(drop=True)
    print("Complete cases (drinking, tobacco and diagnoses recorded):", len(l1_cc), dict(l1_cc["level1"].value_counts()))
    for model_name in ["HistGB", "LogReg"]:
        res = mu.run_sets(l1_cc, "level1", mu.LEVEL1_CLASSES, nested, model_name)
        rows += mu.tag(res["metrics"], level="Level1", analysis="complete_cases",
                       subset="drinking, tobacco and diagnoses recorded")
    mu.log("B complete cases done", t0)

    regimes = pd.DataFrame(rows)
    regimes["description"] = regimes["feature_set"].map(REGIME_LABELS)
    front = ["analysis", "level", "subset", "model", "feature_set", "description", "n"]
    regimes = regimes[front + [c for c in regimes.columns if c not in front]]
    regimes.to_csv(config.TABLES_DIR / "explanation_regimes.csv", index=False)
    shares = pd.DataFrame(share_rows)
    shares.to_csv(config.TABLES_DIR / "explanation_gain_shares.csv", index=False)
    print("\n--- Content versus blank pattern ---")
    print(regimes[front[:5] + ["n", "accuracy", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref",
                               "gain_ci_low", "gain_ci_high", "acc_gain_vs_ref"]].to_string(index=False))
    print(shares.to_string(index=False))

    # ---------- C. Factor associations ----------
    factor_cols = lifestyle + med_hist
    u1 = run_univariate_tests(l1, "level1", mu.LEVEL1_CLASSES, factor_cols, "Level1")
    u2 = run_univariate_tests(l2, "intent", mu.LEVEL2_CLASSES, factor_cols, "Level2")
    a1 = run_adjusted_tests(l1, "level1", mu.LEVEL1_CLASSES, factor_cols, "Level1")
    a2 = run_adjusted_tests(l2, "intent", mu.LEVEL2_CLASSES, factor_cols, "Level2")
    for name, d in [("factors_univariate_level1", u1), ("factors_univariate_level2", u2),
                    ("factors_adjusted_level1", a1), ("factors_adjusted_level2", a2)]:
        d.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    print("\nUnivariate FDR-significant: Level 1", int(u1["sig_fdr_05"].sum()), "of", len(u1),
          "| Level 2", int(u2["sig_fdr_05"].sum()), "of", len(u2))
    print("Adjusted FDR-significant: Level 1", int(a1["sig_fdr_05"].sum()), "of", int(a1["p_raw"].notna().sum()),
          "modelled | Level 2", int(a2["sig_fdr_05"].sum()), "of", int(a2["p_raw"].notna().sum()))
    print(a1.head(10)[["feature", "lr_chi2", "p_fdr_bh"] + [c for c in a1.columns if c.startswith("OR_")]].to_string(index=False))
    mu.log("C factor tests done", t0)

    # ---------- D. SHAP ----------
    shap_df = run_shap_analysis(l1, "level1", mu.LEVEL1_CLASSES, all_cols)
    print(shap_df.head(10).to_string(index=False))
    mu.log("D SHAP done", t0)

    # ---------- Figure: content versus blank pattern ----------
    d = regimes[(regimes["analysis"] == "content_vs_blank_pattern") & (regimes["level"] == "Level1") &
                (regimes["model"] == "HistGB")].set_index("feature_set")
    order = ["M0_base", "M0_plus_L_flags_only", "M1_values_blanks_randomised", "M1_full"]
    names = ["Age + sex\nonly", "Which lifestyle\nsections are filled in", "What the filled-in\nsections say", "Both"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric, title in [(axes[0], "macro_auc", "Macro AUC"), (axes[1], "accuracy", "Accuracy")]:
        sub = d.loc[order]
        err = [sub[metric] - sub[metric + "_ci_low"], sub[metric + "_ci_high"] - sub[metric]]
        ax.bar(range(4), sub[metric], yerr=err, capsize=3, color=["#95a5a6", "#c0392b", "#1f77b4", "#2c3e50"])
        for i, v in enumerate(sub[metric]):
            ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0.4 if metric == "accuracy" else 0.6, 0.8 if metric == "macro_auc" else 0.7)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("Where the lifestyle information comes from (gradient boosting, held-out people, 95% intervals)", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "explanation_content_vs_blanks.png", dpi=150)
    plt.close(fig)
    mu.log("05_explanation finished", t0)
