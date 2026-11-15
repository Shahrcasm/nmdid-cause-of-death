"""Phase 4 / RQ3: is the lifestyle signal information or a recording artifact?

Analyses (5x5 repeated stratified CV + 1,000 paired bootstrap CIs):
  A. Encoding regimes, Levels 1 and 2:
       M0                      age + sex
       +flags only             M0 + the _missing indicators (what was recorded)
       +values, blanks randomised
                               M0 + value columns, after every blank block has
                               been replaced by the values of a random recorded
                               person. No all-zero pattern or NaN marks a blank,
                               so this is the clean 'what the values say' regime
                               (a lower bound, because blanks receive noise).
       +values, implicit blanks
                               M0 + value columns as stored. Kept for
                               transparency only: an all-zero dummy pattern or a
                               NaN reproduces the _missing flag in 93-100% of
                               people, so this regime still sees the blanks.
       full                    M0 + values + flags
  B. Complete cases: drinking, tobacco AND diagnosis all recorded.
  C. Family-interview-only fields vs fields investigators also record, and
     models inside people with / without a family (NOK) interview.
  D. Medications block added as M3 (scene-evidence check).
  E. Cardiac-conflict sensitivity: relabel the ingestion cases as drug_alcohol.

Outputs: rq3_artifact.csv, rq3_gain_shares.csv,
         figures/rq3_encoding_regimes.png, figures/rq3_subgroups.png
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


def add_info(rows, analysis, subset, level):
    for r in rows:
        r.update({"analysis": analysis, "subset": subset, "level": level})
    return rows


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    base, lifestyle, med_hist = mu.get_column_groups(features)
    nested = mu.get_nested_sets(features)
    enc = encoding_sets(base, lifestyle, med_hist)
    all_cols = nested["M2_med_history"]
    all_rows = []

    l1 = cohort[cohort["level1"].isin(mu.LEVEL1_CLASSES)][[config.KEY, "level1"]].merge(features, on=config.KEY)
    l2 = cohort[(cohort["level1"] == "injury") & cohort["intent"].isin(mu.LEVEL2_CLASSES)][
        [config.KEY, "intent"]].merge(features, on=config.KEY)

    # ---------- A. Encoding regimes ----------
    for model_name in ["HistGB", "LogReg"]:
        all_rows += add_info(run_encoding(l1, "level1", mu.LEVEL1_CLASSES, enc, model_name, all_cols), "A_encoding", "all", "Level1")
        log(f"A Level1 {model_name} done", t0)
    all_rows += add_info(run_encoding(l2, "intent", mu.LEVEL2_CLASSES, enc, "HistGB", all_cols), "A_encoding", "all", "Level2")
    log("A Level2 HistGB done", t0)

    # ---------- B. Complete cases ----------
    cc_mask = (l1["drink_missing"] == 0) & (l1["tob_missing"] == 0) & (l1["dx_missing"] == 0)
    l1_cc = l1[cc_mask].reset_index(drop=True)
    print("Complete cases (drinking + tobacco + diagnosis known):", len(l1_cc), dict(l1_cc["level1"].value_counts()))
    for model_name in ["HistGB", "LogReg"]:
        all_rows += add_info(mu.run_sets(l1_cc, "level1", mu.LEVEL1_CLASSES, nested, model_name)["metrics"],
                             "B_complete_cases", "complete_cases", "Level1")
    log("B complete cases done", t0)

    # ---------- C. Family vs investigator fields, and NOK subgroups ----------
    family = [c for c in lifestyle if mu.starts_with_any(c, mu.FAMILY_ONLY_PREFIXES)]
    shared = [c for c in lifestyle if mu.starts_with_any(c, mu.SHARED_PREFIXES)]
    assert set(family + shared) == set(lifestyle), "every lifestyle column must be family-only or shared"
    fam_flags = [c for c in family if c.endswith("_missing")]
    fam_cov = float(((l1[fam_flags] == 0).any(axis=1)).mean())
    sh_flags = [c for c in shared if c.endswith("_missing")]
    sh_cov = float(((l1[sh_flags] == 0).any(axis=1)).mean())
    print(f"Family-only columns: {len(family)} (any field recorded in {100 * fam_cov:.1f}% of people) | "
          f"shared columns: {len(shared)} (any field recorded in {100 * sh_cov:.1f}%)")
    fam_sets = {"M0_base": base, "M0_plus_family_only": base + family, "M0_plus_shared": base + shared, "M1_full": base + lifestyle}
    rows = mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, fam_sets, "HistGB")["metrics"]
    for r in rows:
        r["block_coverage_pct"] = {"M0_plus_family_only": round(100 * fam_cov, 1), "M0_plus_shared": round(100 * sh_cov, 1)}.get(r["feature_set"], np.nan)
    all_rows += add_info(rows, "C_family_vs_shared", "all", "Level1")
    for nok_value, label in [(1, "nok_interview_yes"), (0, "nok_interview_no")]:
        sub = l1[l1["nok_interview"] == nok_value].reset_index(drop=True)
        print(label, "n =", len(sub), dict(sub["level1"].value_counts()))
        all_rows += add_info(mu.run_sets(sub, "level1", mu.LEVEL1_CLASSES, nested, "HistGB")["metrics"], "C_nok_subgroup", label, "Level1")
    log("C family / NOK done", t0)

    # ---------- D. Medications block as M3 ----------
    med_sets = {"M2_med_history": nested["M2_med_history"],
                "M3_plus_medications": nested["M2_med_history"] + ["n_medications", "any_medication", "med_none_recorded"]}
    all_rows += add_info(mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, med_sets, "HistGB", reference="M2_med_history")["metrics"],
                         "D_medications", "all", "Level1")
    log("D medications done", t0)

    # ---------- E. Cardiac-conflict sensitivity ----------
    sens = cohort.copy()
    ingest = (sens["level1"] == "cardiac_conflict") & sens["manner_of_death"].str.contains("Ingested", na=False)
    print("Cardiac-conflict ingestion cases relabelled as drug_alcohol:", int(ingest.sum()))
    sens.loc[ingest, "level1"] = "drug_alcohol"
    l1_sens = sens[sens["level1"].isin(mu.LEVEL1_CLASSES)][[config.KEY, "level1"]].merge(features, on=config.KEY)
    all_rows += add_info(mu.run_sets(l1_sens, "level1", mu.LEVEL1_CLASSES, nested, "HistGB")["metrics"],
                         "E_cardiac_conflict", "ingestion_relabelled", "Level1")
    log("E cardiac conflict done", t0)

    # ---------- Save table ----------
    out = pd.DataFrame(all_rows)
    front = ["analysis", "level", "subset", "model", "feature_set", "n"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(config.TABLES_DIR / "rq3_artifact.csv", index=False)
    print("\n--- RQ3 results ---")
    print(out[front + ["macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high"]].to_string(index=False))

    # Share of the full gain delivered by each clean regime (point estimates).
    share_rows = []
    for level in ["Level1", "Level2"]:
        for model_name in ["HistGB", "LogReg"]:
            a = out[(out["analysis"] == "A_encoding") & (out["level"] == level) & (out["model"] == model_name)]
            if a.empty:
                continue
            auc = dict(zip(a["feature_set"], a["macro_auc"]))
            for blk, flags, rand, full in [("L", "M0_plus_L_flags_only", "M1_values_blanks_randomised", "M1_full"),
                                           ("L+H", "M0_plus_LH_flags_only", "M2_values_blanks_randomised", "M2_full")]:
                g_full = auc[full] - auc["M0_base"]
                share_rows.append({"level": level, "model": model_name, "block": blk,
                                   "gain_full": round(g_full, 3),
                                   "gain_flags_only": round(auc[flags] - auc["M0_base"], 3),
                                   "gain_values_blanks_randomised": round(auc[rand] - auc["M0_base"], 3),
                                   "share_flags_only_pct": round(100 * (auc[flags] - auc["M0_base"]) / g_full, 0) if g_full > 0 else np.nan,
                                   "share_values_randomised_pct": round(100 * (auc[rand] - auc["M0_base"]) / g_full, 0) if g_full > 0 else np.nan})
    shares = pd.DataFrame(share_rows)
    shares.to_csv(config.TABLES_DIR / "rq3_gain_shares.csv", index=False)
    print("\n--- Share of the full gain (point estimates) ---")
    print(shares.to_string(index=False))

    # ---------- Figure 1: encoding regimes ----------
    order = ["M0_base", "M0_plus_L_flags_only", "M1_values_blanks_randomised", "M1_full",
             "M0_plus_LH_flags_only", "M2_values_blanks_randomised", "M2_full"]
    labels = ["Age+sex", "+L flags\nonly", "+L values\n(blanks\nrandomised)", "+L full",
              "+L+H flags\nonly", "+L+H values\n(blanks\nrandomised)", "+L+H full"]
    colors = ["#888888", "#e07b39", "#3b75af", "#2a9d8f", "#e07b39", "#3b75af", "#2a9d8f"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, level in zip(axes, ["Level1", "Level2"]):
        d = out[(out["analysis"] == "A_encoding") & (out["level"] == level) & (out["model"] == "HistGB")].set_index("feature_set").loc[order]
        err = [d["macro_auc"] - d["macro_auc_ci_low"], d["macro_auc_ci_high"] - d["macro_auc"]]
        ax.bar(range(len(order)), d["macro_auc"], yerr=err, capsize=4, color=colors)
        for i, v in enumerate(d["macro_auc"]):
            ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.4, 0.9); ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylabel("Macro AUC (95% CI)")
        ax.set_title(f"{level}: recorded-or-not flags vs recorded values (HistGB, 5x5 CV)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "rq3_encoding_regimes.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 2: subgroups ----------
    groups = [("B_complete_cases", "complete_cases", "Complete cases"),
              ("C_nok_subgroup", "nok_interview_yes", "NOK interview: yes"),
              ("C_nok_subgroup", "nok_interview_no", "NOK interview: no"),
              ("E_cardiac_conflict", "ingestion_relabelled", "Cardiac conflict relabelled")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.26
    for j, fs in enumerate(["M0_base", "M1_lifestyle", "M2_med_history"]):
        vals, lows, highs = [], [], []
        for analysis, subset, _ in groups:
            r = out[(out["analysis"] == analysis) & (out["subset"] == subset) & (out["model"] == "HistGB") & (out["feature_set"] == fs)].iloc[0]
            vals.append(r["macro_auc"]); lows.append(r["macro_auc_ci_low"]); highs.append(r["macro_auc_ci_high"])
        x = np.arange(len(groups)) + (j - 1) * width
        ax.bar(x, vals, width, yerr=[np.array(vals) - lows, np.array(highs) - vals], capsize=3, label=fs)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[2]}\n(n={int(out[(out['analysis'] == g[0]) & (out['subset'] == g[1])]['n'].iloc[0])})" for g in groups], fontsize=8)
    ax.set_ylim(0.4, 0.95); ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylabel("Macro AUC (95% CI)")
    ax.set_title("Level 1 subgroups and sensitivity (HistGB, 5x5 CV)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "rq3_subgroups.png", dpi=150)
    plt.close(fig)
    log("Saved rq3_artifact.csv, rq3_gain_shares.csv and figures", t0)
