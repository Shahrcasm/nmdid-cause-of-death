"""Step 7 - Controls and robustness: is the estimate honest?

  1. Negative control (Level 3): stab versus gunshot within homicides. Lifestyle
     records should say little about the weapon an attacker used, so adding
     them should not help. Logistic regression is primary at this sample size.
  2. Label-permutation test: shuffle the true causes, refit, and check that the
     score falls to guessing level (Level 1: 20 shuffles; Level 3: 200).
  3. Robustness of the main estimate (Level 1, gradient boosting).
  4. Design of the withholding-trained model: alternative withholding rates.
  5. Leakage demonstration: what a model given the blocked cart variable
     would score. This shows why that variable is blocked; it is not a result.

Outputs: tables/controls_level3.csv, controls_permutation.csv,
         controls_permutation_draws.csv, controls_robustness.csv,
         controls_withholding_design.csv, controls_leakage_demo.csv,
         figures/controls.png
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import model_utils as mu

DESIGNS = {"default: rates 0.15, 0.3, 0.5, 0.75, 1.0": mu.WITHHOLD_RATES,
           "single rate 0.5": (0.5,),
           "no fully hidden copy: rates 0.15 to 0.75": (0.15, 0.3, 0.5, 0.75)}
DESIGN_SCENARIOS = {"nothing hidden": [], "all lifestyle sections hidden": mu.LIFESTYLE_SECTIONS,
                    "all medical sections hidden": mu.MEDICAL_SECTIONS, "everything hidden except age and sex": list(mu.SECTIONS)}


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)
    m0, m1, m2 = nested["M0_base"], nested["M1_lifestyle"], nested["M2_med_history"]

    # ---------- 1. Negative control ----------
    l3, y3 = mu.level3_frame(cohort, features)
    print("Level 3:", dict(l3["weapon"].value_counts()))
    rows = []
    for model_name in ["LogReg", "RF", "HistGB"]:
        res = mu.run_sets(l3, "weapon", mu.LEVEL3_CLASSES, nested, model_name)
        for r in res["metrics"]:
            r["n_predicted_gunshot"] = int(res["probs"][r["feature_set"]].argmax(axis=1).sum())
        rows += res["metrics"]
    l3_df = pd.DataFrame(rows)
    front = ["model", "feature_set", "n"]
    l3_df = l3_df[front + [c for c in l3_df.columns if c not in front]]
    l3_df.to_csv(config.TABLES_DIR / "controls_level3.csv", index=False)
    print(l3_df[front + ["macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref", "gain_ci_low",
                         "gain_ci_high", "accuracy", "n_predicted_gunshot"]].to_string(index=False))
    mu.log("1 negative control done", t0)

    # ---------- 2. Permutation test ----------
    l1, y1 = mu.level1_frame(cohort, features, extra=["zip_code", "death_year", "source_cart"])
    n_perm = {"Level1": 5 if config.FAST else 20, "Level3": 20 if config.FAST else 200}
    draws, summary = [], []
    for level, df, y in [("Level1", l1, y1), ("Level3", l3, y3)]:
        X = df[m2]
        # observed and shuffled runs use the same single-repeat CV, so they are comparable
        observed = mu.fast_ovr_auc(y, mu.repeated_oof_probs(X, y, "HistGB", n_repeats=1)[0])
        null = []
        for k in range(n_perm[level]):
            y_perm = np.random.default_rng(config.SEED + k).permutation(y)
            null.append(mu.fast_ovr_auc(y_perm, mu.repeated_oof_probs(X, y_perm, "HistGB", n_repeats=1)[0]))
            draws.append({"level": level, "permutation": k + 1, "macro_auc": round(null[-1], 3)})
        null = np.array(null)
        summary.append({"level": level, "model": "HistGB", "feature_set": "M2_med_history", "n": len(y),
                        "n_permutations": len(null), "observed_auc_1rep": round(observed, 3),
                        "null_mean": round(float(null.mean()), 3), "null_sd": round(float(null.std(ddof=1)), 3),
                        "null_2.5pct": round(float(np.percentile(null, 2.5)), 3),
                        "null_97.5pct": round(float(np.percentile(null, 97.5)), 3),
                        "observed_minus_null_in_sd": round(float((observed - null.mean()) / null.std(ddof=1)), 1),
                        "empirical_p": round(float((1 + np.sum(null >= observed)) / (len(null) + 1)), 4)})
        mu.log(f"2 permutations {level} done", t0)
    pd.DataFrame(draws).to_csv(config.TABLES_DIR / "controls_permutation_draws.csv", index=False)
    perm = pd.DataFrame(summary)
    perm.to_csv(config.TABLES_DIR / "controls_permutation.csv", index=False)
    print(perm.to_string(index=False))

    # ---------- 3. Robustness (Level 1, HistGB) ----------
    rob = []
    l1["zip_num"] = pd.to_numeric(l1["zip_code"], errors="coerce")
    rob += mu.tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, {"M2_med_history": m2, "M2_plus_zip_code": m2 + ["zip_num"]},
                              reference="M2_med_history")["metrics"], check="zip code added")
    swap = {"cadaver_bmi": "cadaver_bmi_decomp_blanked", "cadaver_bmi_missing": "cadaver_bmi_decomp_blanked_missing"}
    rob += mu.tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, {"M0_base": m0, "M1_lifestyle": [swap.get(c, c) for c in m1],
                                                               "M2_med_history": [swap.get(c, c) for c in m2]})["metrics"],
                  check="cadaver BMI blanked for decomposition")
    drop = {"cadaver_bmi", "cadaver_bmi_missing"}
    rob += mu.tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, {"M0_base": m0, "M1_lifestyle": [c for c in m1 if c not in drop],
                                                               "M2_med_history": [c for c in m2 if c not in drop]})["metrics"],
                  check="living BMI only")
    rob += mu.tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, nested, balanced=True)["metrics"], check="class weights")
    year = pd.to_numeric(l1["death_year"], errors="coerce")
    print("Level-1 deaths after 2017:", int((year > 2017).sum()))
    rob += mu.tag(mu.run_sets(l1[~(year > 2017)].reset_index(drop=True), "level1", mu.LEVEL1_CLASSES, nested)["metrics"],
                  check="deaths after 2017 excluded")
    sens = cohort.copy()
    ingest = (sens["level1"] == "cardiac_conflict") & sens["manner_of_death"].str.contains("Ingested", na=False)
    print("Cardiac-conflict ingestion deaths relabelled as drug/alcohol:", int(ingest.sum()))
    sens.loc[ingest, "level1"] = "drug_alcohol"
    l1s, _ = mu.level1_frame(sens, features)
    rob += mu.tag(mu.run_sets(l1s, "level1", mu.LEVEL1_CLASSES, nested)["metrics"],
                  check="cardiac-conflict ingestion deaths relabelled")
    c66 = cohort[cohort["source_cart"] == "cart2966"].copy()
    ing66 = (c66["level1"] == "cardiac_conflict") & c66["manner_of_death"].str.contains("Ingested", na=False)
    c66.loc[ing66, "level1"] = "drug_alcohol"
    c66 = c66[c66["level1"].isin(mu.LEVEL1_CLASSES)].copy()
    c66["cardiac"] = np.where(c66["level1"] == "natural_cardiac", "cardiac", "non_cardiac")
    print("Within cart 2966:", dict(c66["cardiac"].value_counts()))
    c66 = c66[[config.KEY, "cardiac"]].merge(features, on=config.KEY).reset_index(drop=True)
    rob += mu.tag(mu.run_sets(c66, "cardiac", ["non_cardiac", "cardiac"], nested)["metrics"],
                  check="within cart 2966 (cardiac vs non-cardiac)")
    rob_df = pd.DataFrame(rob)
    front = ["check", "model", "feature_set", "n"]
    rob_df = rob_df[front + [c for c in rob_df.columns if c not in front]]
    rob_df.to_csv(config.TABLES_DIR / "controls_robustness.csv", index=False)
    print(rob_df[front + ["accuracy", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref",
                          "gain_ci_low", "gain_ci_high"]].to_string(index=False))
    mu.log("3 robustness done", t0)

    # ---------- 4. Design of the withholding-trained model ----------
    sec = mu.Sections(m2)
    X = l1[m2]
    splits, R = mu.cv_splits(X, y1)
    P = {(d, s): np.zeros((R, len(y1), 3)) for d in DESIGNS for s in DESIGN_SCENARIOS}
    for f, (rep, tr, te) in enumerate(splits):
        for d, rates in DESIGNS.items():
            Xa, ya = sec.augment(X.iloc[tr], y1[tr], seed=config.SEED + f, rates=rates)  # same seeds as step 4
            model = mu.make_model("HistGB", n_train=len(Xa)).fit(Xa, ya)
            for s, hidden in DESIGN_SCENARIOS.items():
                P[(d, s)][rep, te] = model.predict_proba(sec.as_withheld(X.iloc[te], sec.mask(len(te), hidden)))
    sets = {f"{d} | {s}": v.mean(axis=0) for (d, s), v in P.items()}
    rows, _ = mu.bootstrap_compare(y1, sets)
    for r in rows:
        r["design"], r["hidden_at_test_time"] = r.pop("feature_set").split(" | ")
    design = pd.DataFrame(rows)
    front = ["design", "hidden_at_test_time", "n"]
    design = design[front + [c for c in design.columns if c not in front]]
    design.to_csv(config.TABLES_DIR / "controls_withholding_design.csv", index=False)
    print(design[front + ["accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_auc"]].to_string(index=False))
    mu.log("4 withholding design done", t0)

    # ---------- 5. Leakage demonstration ----------
    carts = pd.get_dummies(l1["source_cart"], prefix="cart").astype(int)
    lk = pd.concat([l1[m2], carts], axis=1)
    sets = {"M2 main inputs": m2, "Blocked cart variable only": list(carts.columns), "M2 + blocked cart variable": m2 + list(carts.columns)}
    lprobs, lrep = {}, {}
    for name, cols in sets.items():
        lprobs[name], lrep[name] = mu.repeated_oof_probs(lk[cols], y1, "HistGB")
    rows, _ = mu.bootstrap_compare(y1, lprobs, reference="M2 main inputs", per_repeat=lrep)
    leak = pd.DataFrame(rows).rename(columns={"feature_set": "inputs"})
    leak["note"] = "demonstration only: the cart variable is blocked from every model"
    leak.to_csv(config.TABLES_DIR / "controls_leakage_demo.csv", index=False)
    print(leak[["inputs", "n", "accuracy", "macro_auc"]].to_string(index=False))

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = {"M0_base": "Age + sex", "M1_lifestyle": "+ lifestyle", "M2_med_history": "+ medical"}
    for k, model in enumerate(["LogReg", "RF", "HistGB"]):
        sub = l3_df[l3_df["model"] == model].set_index("feature_set").loc[list(labels)]
        x = np.arange(3) + (k - 1) * 0.27
        axes[0].bar(x, sub["macro_auc"], 0.27, yerr=[sub["macro_auc"] - sub["macro_auc_ci_low"], sub["macro_auc_ci_high"] - sub["macro_auc"]],
                    capsize=2, label={"LogReg": "Logistic regression", "RF": "Random forest", "HistGB": "Gradient boosting"}[model])
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(list(labels.values())); axes[0].set_ylim(0.3, 0.9)
    axes[0].set_title(f"Negative control: stab vs gunshot (n = {len(y3)})", fontsize=10); axes[0].legend(fontsize=8)
    axes[0].set_ylabel("AUC"); axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    dr = pd.DataFrame(draws)
    for level, color in [("Level1", "#1f77b4"), ("Level3", "#e67e22")]:
        vals = dr[dr["level"] == level]["macro_auc"]
        obs = perm.set_index("level").loc[level, "observed_auc_1rep"]
        axes[1].hist(vals, bins=20, alpha=0.6, color=color, label=f"{level}: shuffled labels")
        axes[1].axvline(obs, color=color, linewidth=2, label=f"{level}: real labels ({obs:.3f})")
    axes[1].set_xlabel("AUC"); axes[1].set_title("Permutation test", fontsize=10); axes[1].legend(fontsize=8)
    main = rob_df[rob_df["feature_set"].isin(["M2_med_history", "M2_plus_zip_code"]) &
                  ~((rob_df["check"] == "zip code added") & (rob_df["feature_set"] == "M2_med_history"))]
    ypos = np.arange(len(main))
    axes[2].errorbar(main["macro_auc"], ypos, xerr=[main["macro_auc"] - main["macro_auc_ci_low"], main["macro_auc_ci_high"] - main["macro_auc"]],
                     fmt="o", capsize=3, color="#2c3e50")
    axes[2].set_yticks(ypos); axes[2].set_yticklabels(main["check"], fontsize=8); axes[2].invert_yaxis()
    axes[2].set_xlabel("Macro AUC, all inputs (95% interval)"); axes[2].set_title("Robustness checks", fontsize=10)
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "controls.png", dpi=150)
    plt.close(fig)
    mu.log("07_controls finished", t0)
