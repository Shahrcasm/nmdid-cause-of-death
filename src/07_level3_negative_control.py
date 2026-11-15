"""Phase 5: Level 3 control, permuted-label leakage check, and robustness.

  1. Level 3 (homicides: gunshot vs stab, n = 137, 30 gunshot). Lifestyle
     cannot see a wound, so a large gain would point to leakage. With 30
     events the 95% CI of any gain is about +/- 0.11 wide, so this is a
     weak control: it can expose a large leak, not prove absence of signal.
     HistGB early stopping is disabled below 200 training rows (model_utils),
     because its 10% internal validation split made it stop at iteration 1.
  2. Permuted-label check (Levels 1 and 3): labels shuffled, pipeline re-run.
     A leak-free pipeline must give chance-level AUC.
  3. Robustness on Level 1 (HistGB, M0/M1/M2):
     - add zip_code (location) as an input;
     - cadaver BMI blanked for moderate/advanced decomposition (the previous
       main definition; its _missing flag copies decomposition_score);
     - living BMI only (no cadaver BMI at all);
     - class-weighted HistGB;
     - exclude deaths after 2017 (1 Level-1 death; 4 adults overall);
     - within cart 2966 only: natural cardiac vs non-cardiac. Only 50
       non-cardiac people are in that cart, so this is a low-power check.

Outputs: level3_negative_control.csv, permutation_check.csv, robustness.csv,
         figures/level3_and_robustness.png
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import model_utils as mu


def tag(rows, **info):
    for r in rows:
        r.update(info)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)

    # ---------- 1. Level 3 control ----------
    stab, gun = "Stab wound (slash, penetrating cut)", "Gunshot wound of"
    l3 = cohort[(cohort["level1"] == "injury") & (cohort["intent"] == "homicide") &
                cohort["primary_cause_of_death"].isin([stab, gun])].copy()
    l3["weapon"] = np.where(l3["primary_cause_of_death"] == gun, "gunshot", "stab")
    l3 = l3[[config.KEY, "weapon"]].merge(features, on=config.KEY)
    print("Level 3 cohort:", dict(l3["weapon"].value_counts()))
    l3_rows = []
    for model_name in ["LogReg", "RF", "HistGB"]:
        res = mu.run_sets(l3, "weapon", mu.LEVEL3_CLASSES, nested, model_name)
        for name, p in res["probs"].items():
            pred = p.argmax(1)
            for r in res["metrics"]:
                if r["feature_set"] == name:
                    r["n_predicted_gunshot"] = int(pred.sum())
        l3_rows += tag(res["metrics"], level="Level3")
    l3_df = pd.DataFrame(l3_rows)
    l3_df.to_csv(config.TABLES_DIR / "level3_negative_control.csv", index=False)
    print("\n--- Level 3 control (AUC for gunshot vs stab) ---")
    print(l3_df[["model", "feature_set", "n", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high",
                 "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high", "n_predicted_gunshot"]].to_string(index=False))
    print(f"[{time.time() - t0:.0f}s] Level 3 done", flush=True)

    # ---------- 2. Permuted-label check ----------
    l1 = cohort[cohort["level1"].isin(mu.LEVEL1_CLASSES)][
        [config.KEY, "level1", "zip_code", "death_year", "source_cart", "manner_of_death"]].merge(features, on=config.KEY)
    y1 = l1["level1"].map({c: i for i, c in enumerate(mu.LEVEL1_CLASSES)}).values
    y3 = (l3["weapon"] == "gunshot").astype(int).values
    # Permutation test: the observed AUC uses the same single-repeat CV as the
    # shuffles, so observed and null values are directly comparable. With 30
    # gunshot deaths one shuffled AUC varies by about +/- 0.06, so many shuffles
    # are needed to locate the null distribution.
    n_perm = {"Level1": 5 if config.FAST else 20, "Level3": 20 if config.FAST else 200}
    perm_rows, summary_rows = [], []
    for level, df, y in [("Level1", l1, y1), ("Level3", l3, y3)]:
        X = df[nested["M2_med_history"]]
        obs_probs, _ = mu.repeated_oof_probs(X, y, "HistGB", n_repeats=1)
        observed = mu.fast_ovr_auc(y, obs_probs)
        null = []
        for k in range(n_perm[level]):
            rng = np.random.default_rng(config.SEED + k)
            y_perm = rng.permutation(y)
            probs, _ = mu.repeated_oof_probs(X, y_perm, "HistGB", n_repeats=1)
            null.append(mu.fast_ovr_auc(y_perm, probs))
            perm_rows.append({"level": level, "feature_set": "M2_med_history", "model": "HistGB",
                              "permutation": k + 1, "n": len(y), "macro_auc": round(null[-1], 3)})
        null = np.array(null)
        summary_rows.append({"level": level, "feature_set": "M2_med_history", "model": "HistGB", "n": len(y),
                             "n_permutations": len(null), "observed_auc_1rep": round(observed, 3),
                             "null_mean": round(float(null.mean()), 3), "null_sd": round(float(null.std(ddof=1)), 3),
                             "null_2.5pct": round(float(np.percentile(null, 2.5)), 3),
                             "null_97.5pct": round(float(np.percentile(null, 97.5)), 3),
                             "empirical_p": round(float((1 + np.sum(null >= observed)) / (len(null) + 1)), 4)})
        print(f"[{time.time() - t0:.0f}s] {level} permutations done", flush=True)
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(config.TABLES_DIR / "permutation_check.csv", index=False)
    perm_sum = pd.DataFrame(summary_rows)
    perm_sum.to_csv(config.TABLES_DIR / "permutation_summary.csv", index=False)
    print("\n--- Permutation test (null should centre on 0.50) ---")
    print(perm_sum.to_string(index=False))
    print(f"[{time.time() - t0:.0f}s] permutation check done", flush=True)

    # ---------- 3. Robustness (Level 1) ----------
    rob_rows = []
    m0, m1, m2 = nested["M0_base"], nested["M1_lifestyle"], nested["M2_med_history"]

    l1["zip_num"] = pd.to_numeric(l1["zip_code"], errors="coerce")
    sets = {"M2_med_history": m2, "M2_plus_zip_code": m2 + ["zip_num"]}
    rob_rows += tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, sets, reference="M2_med_history")["metrics"], check="add_zip_code")

    swap = {"cadaver_bmi": "cadaver_bmi_decomp_blanked", "cadaver_bmi_missing": "cadaver_bmi_decomp_blanked_missing"}
    sets = {"M0_base": m0, "M1_lifestyle": [swap.get(c, c) for c in m1], "M2_med_history": [swap.get(c, c) for c in m2]}
    rob_rows += tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, sets)["metrics"], check="cadaver_bmi_decomp_blanked")

    drop = ["cadaver_bmi", "cadaver_bmi_missing"]
    sets = {"M0_base": m0, "M1_lifestyle": [c for c in m1 if c not in drop], "M2_med_history": [c for c in m2 if c not in drop]}
    rob_rows += tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, sets)["metrics"], check="living_bmi_only")

    rob_rows += tag(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, nested, balanced=True)["metrics"], check="class_weight_balanced")

    year = pd.to_numeric(l1["death_year"], errors="coerce")
    print("Level-1 deaths after 2017:", int((year > 2017).sum()))
    sub = l1[~(year > 2017)].reset_index(drop=True)
    rob_rows += tag(mu.run_sets(sub, "level1", mu.LEVEL1_CLASSES, nested)["metrics"], check="exclude_deaths_after_2017")
    print(f"[{time.time() - t0:.0f}s] robustness 3a-3e done", flush=True)

    c66 = cohort[cohort["source_cart"] == "cart2966"].copy()
    ingest = (c66["level1"] == "cardiac_conflict") & c66["manner_of_death"].str.contains("Ingested", na=False)
    c66.loc[ingest, "level1"] = "drug_alcohol"
    c66 = c66[c66["level1"].isin(mu.LEVEL1_CLASSES)].copy()
    c66["cardiac"] = np.where(c66["level1"] == "natural_cardiac", "cardiac", "non_cardiac")
    print("Cart 2966 within-cart check:", dict(c66["cardiac"].value_counts()))
    c66 = c66[[config.KEY, "cardiac"]].merge(features, on=config.KEY)
    rob_rows += tag(mu.run_sets(c66, "cardiac", ["non_cardiac", "cardiac"], nested)["metrics"], check="within_cart2966_cardiac_vs_noncardiac")

    rob_df = pd.DataFrame(rob_rows)
    front = ["check", "model", "feature_set", "n"]
    rob_df = rob_df[front + [c for c in rob_df.columns if c not in front]]
    rob_df.to_csv(config.TABLES_DIR / "robustness.csv", index=False)
    print("\n--- Robustness (Level 1, HistGB) ---")
    print(rob_df[front + ["macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high"]].to_string(index=False))

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    width = 0.26
    for j, fs in enumerate(["M0_base", "M1_lifestyle", "M2_med_history"]):
        d = l3_df[l3_df["feature_set"] == fs].set_index("model").loc[["LogReg", "RF", "HistGB"]]
        x = np.arange(3) + (j - 1) * width
        ax.bar(x, d["macro_auc"], width, yerr=[d["macro_auc"] - d["macro_auc_ci_low"], d["macro_auc_ci_high"] - d["macro_auc"]], capsize=3, label=fs)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["LogReg", "RF", "HistGB"])
    ax.set_ylim(0.2, 0.95); ax.set_ylabel("AUC (95% CI)")
    ax.set_title(f"Level 3 control: gunshot vs stab (n={len(l3)}, 30 gunshot)")
    ax.legend(fontsize=8)

    ax = axes[1]
    main = pd.read_csv(config.TABLES_DIR / "rq1_metrics.csv")
    main = main[(main["level"] == "Level1") & (main["model"] == "HistGB")].set_index("feature_set")
    labels, m0v, m2v, lo, hi = ["Main\n(rq1)"], [main.loc["M0_base", "macro_auc"]], [main.loc["M2_med_history", "macro_auc"]], \
        [main.loc["M2_med_history", "macro_auc_ci_low"]], [main.loc["M2_med_history", "macro_auc_ci_high"]]
    for chk, lab in [("cadaver_bmi_decomp_blanked", "Decomp-blanked\ncadaver BMI"), ("living_bmi_only", "Living BMI\nonly"),
                     ("class_weight_balanced", "Class\nweights"), ("exclude_deaths_after_2017", "Deaths\n<=2017"),
                     ("within_cart2966_cardiac_vs_noncardiac", "Cart 2966\ncardiac vs not")]:
        d = rob_df[rob_df["check"] == chk].set_index("feature_set")
        labels.append(lab); m0v.append(d.loc["M0_base", "macro_auc"]); m2v.append(d.loc["M2_med_history", "macro_auc"])
        lo.append(d.loc["M2_med_history", "macro_auc_ci_low"]); hi.append(d.loc["M2_med_history", "macro_auc_ci_high"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, m0v, 0.4, label="M0_base", color="#888888")
    ax.bar(x + 0.2, m2v, 0.4, yerr=[np.array(m2v) - lo, np.array(hi) - m2v], capsize=3, label="M2_med_history", color="#2a9d8f")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.4, 0.95); ax.set_ylabel("Macro AUC")
    ax.set_title("Level 1 robustness checks (HistGB; M2 bars show 95% CI)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "level3_and_robustness.png", dpi=150)
    plt.close(fig)
    print(f"[{time.time() - t0:.0f}s] Saved Phase 5 tables and figure", flush=True)
