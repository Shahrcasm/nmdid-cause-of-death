"""Step 7b - Validation on later years and on other regions (internal-external validation).

  A. Later years. Models are trained on deaths up to 2014 and tested on deaths
     from 2015 on, which none of them saw. Shortlist thresholds are calibrated on
     2014 deaths with a model trained on deaths up to 2013, then checked on 2015+.
  B. Other regions. Counties are split into five groups (the largest county is a
     group by itself); each group is predicted by models trained on the other
     counties. County only defines the groups and is never a model input.

The year and the county are blocked as inputs; here they only decide who is in
the training and who is in the test data.

Outputs: tables/validation_temporal.csv, validation_temporal_shortlists.csv,
         validation_regional.csv, figures/validation.png
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import config
import model_utils as mu

C = mu.LEVEL1_CLASSES
BANDS = [("none", 0, 0), ("1 to 3", 1, 3), ("4 or 5", 4, 5), ("6", 6, 6), ("7 or 8", 7, 8)]


def calibration_slopes(y, p):
    out = []
    for i in range(p.shape[1]):
        z = np.log(np.clip(p[:, i], 1e-6, 1 - 1e-6) / np.clip(1 - p[:, i], 1e-6, 1))
        out.append(float(sm.Logit((y == i).astype(float), sm.add_constant(z)).fit(disp=0).params[1]))
    return out


def threshold(scores, alpha):
    sc = np.sort(scores)
    k = int(np.ceil((len(sc) + 1) * (1 - alpha)))
    return float(sc[k - 1]) if k <= len(sc) else 1.0


def band_of(n_blank):
    return np.array([next(b for b, lo, hi in BANDS if lo <= v <= hi) for v in n_blank], dtype=object)


def fit_set(X, X0, y, sec, seed):
    return {"std": mu.make_model("HistGB", n_train=len(y)).fit(X, y),
            "base": mu.make_model("HistGB", n_train=len(y)).fit(X0, y),
            "withheld": sec.fit_withholding_model(X, y, seed=seed)}


def predict_set(models, X, X0, sec):
    every = sec.mask(len(X), sec.names)
    return {"age and sex only": models["base"].predict_proba(X0),
            "all inputs (standard model)": models["std"].predict_proba(X),
            "all inputs (withholding-trained model)": models["withheld"].predict_proba(sec.as_withheld(X, sec.mask(len(X), []))),
            "everything hidden, read as blank (standard model)": models["std"].predict_proba(sec.as_not_recorded(X, every)),
            "everything hidden (withholding-trained model)": models["withheld"].predict_proba(sec.as_withheld(X, every))}


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)
    m0, m2 = nested["M0_base"], nested["M2_med_history"]
    l1, y = mu.level1_frame(cohort, features, extra=["death_year", "death_county"])
    sec = mu.Sections(m2)
    year = pd.to_numeric(l1["death_year"], errors="coerce").values
    n_blank = sec.not_recorded(l1).sum(axis=1)

    # ---------- A. Later years ----------
    tr, te = year <= 2014, year >= 2015
    print(f"Temporal split: trained on {tr.sum()} deaths up to 2014, tested on {te.sum()} deaths from 2015 on")
    models = fit_set(l1.loc[tr, m2], l1.loc[tr, m0], y[tr], sec, config.SEED)
    P = predict_set(models, l1.loc[te, m2].reset_index(drop=True), l1.loc[te, m0], sec)
    yt = y[te]
    rows, _ = mu.bootstrap_compare(yt, P, reference="age and sex only")
    for r in rows:
        r["calibration_slope_range"] = "{:.2f} to {:.2f}".format(*[f(calibration_slopes(yt, P[r["feature_set"]])) for f in (min, max)])
    temporal = pd.DataFrame(rows).rename(columns={"feature_set": "inputs"})
    temporal.insert(0, "validation", "trained to 2014, tested 2015 on")
    temporal.to_csv(config.TABLES_DIR / "validation_temporal.csv", index=False)
    print(temporal[["inputs", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_auc", "macro_auc_ci_low",
                    "macro_auc_ci_high", "calibration_slope_range"]].to_string(index=False))

    # Shortlists: model trained to 2013, thresholds from 2014, coverage on 2015 on.
    fit, cal = year <= 2013, year == 2014
    std_c = mu.make_model("HistGB", n_train=int(fit.sum())).fit(l1.loc[fit, m2], y[fit])
    p_cal, p_te = std_c.predict_proba(l1.loc[cal, m2]), std_c.predict_proba(l1.loc[te, m2])
    s_cal = 1 - p_cal[np.arange(cal.sum()), y[cal]]
    b_cal, b_te = band_of(n_blank[cal]), band_of(n_blank[te])
    srows = []
    for alpha in [0.10, 0.05]:
        for method in ["one threshold", "threshold adjusted for blank sections"]:
            sets = np.zeros_like(p_te, dtype=bool)
            if method == "one threshold":
                sets = p_te >= 1 - threshold(s_cal, alpha)
            else:
                for b in np.unique(b_te):
                    m = b_te == b
                    sets[m] = p_te[m] >= 1 - threshold(s_cal[b_cal == b], alpha)
            hit = sets[np.arange(len(yt)), yt]
            row = {"target_coverage_pct": round(100 * (1 - alpha)), "method": method, "n_test": int(len(yt)),
                   "n_calibration": int(cal.sum()), "coverage_pct": round(100 * float(hit.mean()), 1),
                   "mean_shortlist_size": round(float(sets.sum(axis=1).mean()), 2)}
            for b in [b_ for b_, _, _ in BANDS]:
                m = b_te == b
                row[f"coverage_pct_{b.replace(' ', '_')}_blank"] = round(100 * float(hit[m].mean()), 1) if m.sum() >= 10 else np.nan
            srows.append(row)
    short = pd.DataFrame(srows)
    short.to_csv(config.TABLES_DIR / "validation_temporal_shortlists.csv", index=False)
    print(short.to_string(index=False))
    mu.log("A later years done", t0)

    # ---------- B. Other regions ----------
    county = l1["death_county"].fillna("unknown").values
    folds = list(GroupKFold(n_splits=5).split(l1, y, groups=county))
    pooled = {}
    fold_rows = []
    for g, (tr_i, te_i) in enumerate(folds):
        mdl = fit_set(l1.iloc[tr_i][m2], l1.iloc[tr_i][m0], y[tr_i], sec, config.SEED + g)
        Pg = predict_set(mdl, l1.iloc[te_i][m2].reset_index(drop=True), l1.iloc[te_i][m0], sec)
        for k, v in Pg.items():
            pooled.setdefault(k, np.zeros((len(y), 3)))[te_i] = v
        counts = np.bincount(y[te_i], minlength=3)
        fold_rows.append({"validation": f"held-out county group {g + 1}", "n": len(te_i),
                          "class_counts": "; ".join(f"{c}: {n if n >= 5 else '<5'}" for c, n in zip(C, counts)),
                          "accuracy_all_inputs": round(mu.accuracy(y[te_i], Pg["all inputs (standard model)"]), 3),
                          "macro_auc_all_inputs": round(mu.fast_ovr_auc(y[te_i], Pg["all inputs (standard model)"]), 3),
                          "accuracy_age_sex_only": round(mu.accuracy(y[te_i], Pg["age and sex only"]), 3)})
    rows, _ = mu.bootstrap_compare(y, pooled, reference="age and sex only")
    regional = pd.DataFrame(rows).rename(columns={"feature_set": "inputs"})
    regional.insert(0, "validation", "pooled over held-out county groups")
    regional = pd.concat([regional, pd.DataFrame(fold_rows)], ignore_index=True)
    regional.to_csv(config.TABLES_DIR / "validation_regional.csv", index=False)
    print(regional[["validation", "inputs", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_auc",
                    "accuracy_all_inputs", "macro_auc_all_inputs"]].to_string(index=False))

    # ---------- Figure ----------
    acc_tab = pd.read_csv(config.TABLES_DIR / "accuracy_metrics.csv")
    internal = acc_tab[(acc_tab["level"] == "Level1") & (acc_tab["model"] == "HistGB") & (acc_tab["feature_set"] == "M2_med_history")].iloc[0]
    tt = temporal.set_index("inputs").loc["all inputs (standard model)"]
    rr = regional[regional["validation"] == "pooled over held-out county groups"].set_index("inputs").loc["all inputs (standard model)"]
    labels = ["Internal\n(5 x 5 cross-validation)", "Later years\n(2015 on)", "Other regions\n(held-out counties)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy, all inputs"), (axes[1], "macro_auc", "Macro AUC, all inputs")]:
        vals = [internal[metric], tt[metric], rr[metric]]
        lo = [internal[metric + "_ci_low"], tt[metric + "_ci_low"], rr[metric + "_ci_low"]]
        hi = [internal[metric + "_ci_high"], tt[metric + "_ci_high"], rr[metric + "_ci_high"]]
        ax.bar(range(3), vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], capsize=4, color=["#7f8c8d", "#1f77b4", "#2c3e50"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0.4 if metric == "accuracy" else 0.6, 0.75 if metric == "accuracy" else 0.85)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("Does the estimate hold for later years and other regions? (95% intervals)", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "validation.png", dpi=150)
    plt.close(fig)
    mu.log("07b_validation finished", t0)
