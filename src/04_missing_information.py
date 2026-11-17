"""Step 4 - Missing information: how much does the estimate suffer when parts
of the record are missing, and can a model be trained to cope?

  A. Records as they are. Held-out accuracy by how many of the eight record
     sections the investigator left blank, and by true cause. This is the table
     with the columns: missing fields, share of training data, ground truth,
     share of test data, test accuracy.
  B. Same people, less information. Sections are hidden at test time: one at a
     time, in groups, and k at random. Five ways of handling a hidden section:
       not_recorded         naive: it looks like a section the investigator left blank
       imputed              typical values of the training fold are filled in
       multiple_imputation  3 donors from the 10 most similar training people supply the hidden sections
       withheld_model       ONE model trained with randomly withheld sections
       reduced_model        a dedicated model refitted without the section (reference)
     The withholding-trained model is also recalibrated by temperature scaling,
     learned on a split of each training fold (accuracy is unchanged by design).
  C. Out-of-fold predictions are cached in data/processed (git-ignored) for
     06_reliability.py.

Outputs: tables/missing_natural.csv, missing_natural_patterns.csv,
         missing_withheld_scenarios.csv, missing_withheld_curve.csv, missing_recalibration.csv,
         figures/missing_information_natural.png, missing_information_withheld.png
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import config
import model_utils as mu

STRATEGIES = ["not_recorded", "imputed", "multiple_imputation", "withheld_model", "reduced_model"]
CURVE_STRATEGIES = ["not_recorded", "imputed", "multiple_imputation", "withheld_model"]
STRATEGY_LABELS = {"not_recorded": "Standard model, hidden = left blank",
                   "imputed": "Standard model, hidden = typical values",
                   "multiple_imputation": "Standard model, hidden = multiple imputation (similar donors)",
                   "withheld_model": "Withholding-trained model",
                   "reduced_model": "Dedicated reduced models (reference)"}
N_DRAWS = 3            # random section subsets per person, per k, per fold
BANDS = [("none", 0, 0), ("1 to 3", 1, 3), ("4 or 5", 4, 5), ("6", 6, 6), ("7 or 8", 7, 8)]
MIN_CELL = 10          # rows with fewer people are not reported


def acc_ci(correct, n_boot):
    """Accuracy with a percentile bootstrap interval for one group of people.
    Uses the same random sequence as mu.bootstrap_compare (fresh generator with
    the study seed, one draw of people per resample), so for the whole cohort
    the interval is identical to the one in accuracy_metrics.csv."""
    correct = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(config.SEED)
    draws = np.array([correct[rng.integers(0, len(correct), size=len(correct))].mean() for _ in range(n_boot)])
    return round(float(correct.mean()), 3), round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)
    m0, m2 = nested["M0_base"], nested["M2_med_history"]
    l1, y = mu.level1_frame(cohort, features)
    C, n = mu.LEVEL1_CLASSES, len(y)
    sec = mu.Sections(m2)
    S = len(sec.names)
    X = l1[m2]

    scenarios = {"nothing hidden": []}
    scenarios.update({mu.SECTION_LABELS[s]: [s] for s in sec.names})
    scenarios.update({"All lifestyle sections": mu.LIFESTYLE_SECTIONS,
                      "All medical sections": mu.MEDICAL_SECTIONS,
                      "Everything except age and sex": sec.names})
    kind = {"nothing hidden": "none", "All lifestyle sections": "group", "All medical sections": "group",
            "Everything except age and sex": "group"}

    splits, R = mu.cv_splits(X, y)
    probs = {st: {sc: np.zeros((R, n, 3)) for sc in scenarios} for st in STRATEGIES}
    p_base = np.zeros((R, n, 3))
    curve = {st: np.zeros((S - 1, N_DRAWS, R, n, 3)) for st in CURVE_STRATEGIES}
    p_rob_recal = np.zeros((R, n, 3))
    temps = []
    train_share = {}            # (band, ground truth) -> share of each training fold
    blank = sec.not_recorded(l1)                 # natural state, people x sections
    n_blank = blank.sum(axis=1)

    for f, (rep, tr, te) in enumerate(splits):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        std = mu.make_model("HistGB", n_train=len(tr)).fit(X_tr, y[tr])
        typical = sec.typical_values(X_tr)
        robust = sec.fit_withholding_model(X_tr, y[tr], seed=config.SEED + f)
        imputers = sec.fit_imputers(X_tr, seed=config.SEED + 100 * f)
        # Temperature for the withholding-trained model, learned inside the training fold only.
        in_tr, in_cal = next(StratifiedKFold(5, shuffle=True, random_state=config.SEED + f).split(X_tr, y[tr]))
        inner = sec.fit_withholding_model(X_tr.iloc[in_tr], y[tr][in_tr], seed=config.SEED + 500 + f)
        T_f = mu.fit_temperature(inner.predict_proba(sec.as_withheld(X_tr.iloc[in_cal], sec.mask(len(in_cal), []))), y[tr][in_cal])
        temps.append({"repeat": rep + 1, "fold": f % 5 + 1, "temperature": round(T_f, 3)})
        p_rob_recal[rep, te] = mu.apply_temperature(robust.predict_proba(sec.as_withheld(X_te, sec.mask(len(te), []))), T_f)
        base = mu.make_model("HistGB", n_train=len(tr)).fit(l1.iloc[tr][m0], y[tr])
        p_base[rep, te] = base.predict_proba(l1.iloc[te][m0])
        for sc, hidden in scenarios.items():
            mask = sec.mask(len(te), hidden)
            probs["not_recorded"][sc][rep, te] = std.predict_proba(sec.as_not_recorded(X_te, mask))
            probs["imputed"][sc][rep, te] = std.predict_proba(sec.as_imputed(X_te, mask, typical))
            probs["multiple_imputation"][sc][rep, te] = sec.predict_multiply_imputed(std, X_te, mask, imputers)
            probs["withheld_model"][sc][rep, te] = robust.predict_proba(sec.as_withheld(X_te, mask))
            if not hidden:
                probs["reduced_model"][sc][rep, te] = probs["not_recorded"][sc][rep, te]
            elif len(hidden) == S:
                probs["reduced_model"][sc][rep, te] = p_base[rep, te]
            else:
                drop = set(c for s in hidden for c in sec.section_columns(s))
                keep = [c for c in m2 if c not in drop]
                red = mu.make_model("HistGB", n_train=len(tr)).fit(X_tr[keep], y[tr])
                probs["reduced_model"][sc][rep, te] = red.predict_proba(X_te[keep])
        rng = np.random.default_rng(config.SEED + 1000 + f)
        for k in range(1, S):
            for d in range(N_DRAWS):
                mask = sec.random_mask(len(te), k, rng)
                curve["not_recorded"][k - 1, d, rep, te] = std.predict_proba(sec.as_not_recorded(X_te, mask))
                curve["imputed"][k - 1, d, rep, te] = std.predict_proba(sec.as_imputed(X_te, mask, typical))
                curve["multiple_imputation"][k - 1, d, rep, te] = sec.predict_multiply_imputed(std, X_te, mask, imputers)
                curve["withheld_model"][k - 1, d, rep, te] = robust.predict_proba(sec.as_withheld(X_te, mask))
        for name, lo, hi in BANDS + [("any (all people)", 0, S)]:
            in_band_tr = (n_blank[tr] >= lo) & (n_blank[tr] <= hi)
            train_share.setdefault((name, "all causes"), []).append(float(in_band_tr.mean()))
            for i, c in enumerate(C):
                train_share.setdefault((name, c), []).append(float((in_band_tr & (y[tr] == i)).mean()))
        if (f + 1) % 5 == 0:
            mu.log(f"repeat {rep + 1} of {R} done", t0)

    P = {st: {sc: probs[st][sc].mean(axis=0) for sc in scenarios} for st in STRATEGIES}
    P0 = p_base.mean(axis=0)
    p_std, p_rob = P["not_recorded"]["nothing hidden"], P["withheld_model"]["nothing hidden"]
    print(f"Check: standard model on records as they are: macro AUC {mu.fast_ovr_auc(y, p_std):.3f}, accuracy {mu.accuracy(y, p_std):.3f} "
          f"(must equal HistGB M2 in accuracy_metrics.csv)")

    # ---------- A. Records as they are ----------
    correct = {"age_sex_only": P0.argmax(1) == y, "standard_model": p_std.argmax(1) == y, "withheld_model": p_rob.argmax(1) == y}
    rows = []
    for name, lo, hi in BANDS + [("any (all people)", 0, S)]:
        in_band = (n_blank >= lo) & (n_blank <= hi)
        for truth in ["all causes"] + C:
            idx = np.flatnonzero(in_band & ((y == C.index(truth)) if truth in C else True))
            if len(idx) < MIN_CELL:
                continue
            # shares of ALL people in this row (band and ground truth): in the training folds
            # (mean over the CV folds) and in the test data (each person is tested once per repeat)
            row = {"sections_not_recorded": name, "ground_truth": truth, "n": len(idx),
                   "train_data_pct": round(100 * float(np.mean(train_share[(name, truth)])), 1),
                   "test_data_pct": round(100 * len(idx) / n, 1)}
            for model, ok in correct.items():
                row[f"test_accuracy_{model}"], row[f"{model}_ci_low"], row[f"{model}_ci_high"] = acc_ci(ok[idx], config.N_BOOT)
            rows.append(row)
    natural = pd.DataFrame(rows)
    natural.to_csv(config.TABLES_DIR / "missing_natural.csv", index=False)
    print("\n--- A. Records as they are: accuracy by number of sections not recorded ---")
    print(natural[["sections_not_recorded", "ground_truth", "n", "train_data_pct", "test_data_pct", "test_accuracy_age_sex_only",
                   "test_accuracy_standard_model", "test_accuracy_withheld_model"]].to_string(index=False))

    codes = pd.Series(["".join("1" if b else "0" for b in r) for r in blank])
    rows = []
    for code, cnt in codes.value_counts().items():
        if cnt < 30:
            continue
        idx = np.flatnonzero((codes == code).values)
        missing = [mu.SECTION_LABELS[s] for s, b in zip(sec.names, code) if b == "1"]
        mix = np.bincount(y[idx], minlength=3)
        rows.append({"sections_not_recorded": "; ".join(missing) if missing else "none", "n_sections_not_recorded": len(missing),
                     "n": int(cnt), "data_pct": round(100 * cnt / n, 1),
                     "ground_truth_pct_cardiac_drug_injury": " / ".join(str(int(round(100 * v / cnt))) for v in mix),
                     "test_accuracy_age_sex_only": round(float(correct["age_sex_only"][idx].mean()), 3),
                     "test_accuracy_standard_model": round(float(correct["standard_model"][idx].mean()), 3),
                     "test_accuracy_withheld_model": round(float(correct["withheld_model"][idx].mean()), 3)})
    patterns = pd.DataFrame(rows)
    patterns.to_csv(config.TABLES_DIR / "missing_natural_patterns.csv", index=False)
    print("\n--- Most common exact patterns (30 or more people) ---")
    print(patterns.to_string(index=False))

    # ---------- B. Sections hidden at test time ----------
    sets = {f"{sc} | {st}": P[st][sc] for sc in scenarios for st in STRATEGIES}
    pairs = [(f"{sc} | {st} | vs_full", f"{sc} | {st}", f"nothing hidden | {st}") for sc in scenarios for st in STRATEGIES]
    pairs += [(f"{sc} | withheld_model | vs_not_recorded", f"{sc} | withheld_model", f"{sc} | not_recorded") for sc in scenarios]
    pairs += [(f"{sc} | withheld_model | vs_reduced", f"{sc} | withheld_model", f"{sc} | reduced_model") for sc in scenarios]
    pairs += [(f"{sc} | withheld_model | vs_mi", f"{sc} | withheld_model", f"{sc} | multiple_imputation") for sc in scenarios]
    mrows, grows = mu.bootstrap_compare(y, sets, pairs=pairs)
    gains = {g["comparison"]: g for g in grows}
    rows = []
    for r in mrows:
        sc, st = r["feature_set"].split(" | ")
        p = P[st][sc]
        pred = p.argmax(1)
        agree = np.mean([(probs[st][sc][k].argmax(1) == probs[st]["nothing hidden"][k].argmax(1)).mean() for k in range(R)])
        g = gains[f"{sc} | {st} | vs_full"]
        row = {"hidden_at_test_time": sc, "scenario_type": kind.get(sc, "single section"), "handling": st, "n": n,
               "accuracy": r["accuracy"], "accuracy_ci_low": r["accuracy_ci_low"], "accuracy_ci_high": r["accuracy_ci_high"],
               "accuracy_change_vs_full_record": g["accuracy_gain"], "acc_change_ci_low": g["accuracy_ci_low"], "acc_change_ci_high": g["accuracy_ci_high"],
               "macro_auc": r["macro_auc"], "macro_auc_ci_low": r["macro_auc_ci_low"], "macro_auc_ci_high": r["macro_auc_ci_high"],
               "auc_change_vs_full_record": g["macro_auc_gain"], "auc_change_ci_low": g["macro_auc_ci_low"], "auc_change_ci_high": g["macro_auc_ci_high"],
               "same_answer_as_full_record_pct": round(100 * float(agree), 1),
               "mean_probability_of_true_cause": round(float(p[np.arange(n), y].mean()), 3)}
        for i, c in enumerate(C):
            row[f"accuracy_true_{c}"], row[f"accuracy_true_{c}_ci_low"], row[f"accuracy_true_{c}_ci_high"] =                 acc_ci(pred[y == i] == i, config.N_BOOT)
        if st == "withheld_model":
            for other, tag in [("vs_not_recorded", "vs_standard_left_blank"), ("vs_reduced", "vs_reduced_model"),
                               ("vs_mi", "vs_multiple_imputation")]:
                h = gains[f"{sc} | withheld_model | {other}"]
                row[f"acc_diff_{tag}"], row[f"acc_diff_{tag}_ci_low"], row[f"acc_diff_{tag}_ci_high"] = h["accuracy_gain"], h["accuracy_ci_low"], h["accuracy_ci_high"]
                row[f"auc_diff_{tag}"], row[f"auc_diff_{tag}_ci_low"], row[f"auc_diff_{tag}_ci_high"] = h["macro_auc_gain"], h["macro_auc_ci_low"], h["macro_auc_ci_high"]
        rows.append(row)
    scen = pd.DataFrame(rows)
    scen.to_csv(config.TABLES_DIR / "missing_withheld_scenarios.csv", index=False)
    print("\n--- B. Sections hidden at test time ---")
    print(scen[["hidden_at_test_time", "handling", "accuracy", "accuracy_change_vs_full_record", "macro_auc",
                "same_answer_as_full_record_pct"] + [f"accuracy_true_{c}" for c in C]].to_string(index=False))

    # Curve: every point is the mean over single CV repeats (and random draws), never
    # over repeat-averaged probabilities, so that all points are computed the same way.
    def per_repeat(prob_list):
        accs = [mu.accuracy(y, p) for p in prob_list]
        aucs = [mu.fast_ovr_auc(y, p) for p in prob_list]
        sd = (lambda v: round(float(np.std(v, ddof=1)), 4)) if len(prob_list) > 1 else (lambda v: 0.0)
        return round(float(np.mean(accs)), 3), sd(accs), round(float(np.mean(aucs)), 3), sd(aucs)

    base_acc, _, base_auc, _ = per_repeat([p_base[r] for r in range(R)])
    rows = []
    for st in CURVE_STRATEGIES:
        for k in range(S + 1):
            if k == 0:
                plist = [probs[st]["nothing hidden"][r] for r in range(R)]
            elif k == S:
                plist = [probs[st]["Everything except age and sex"][r] for r in range(R)]
            else:
                plist = [curve[st][k - 1, d, r] for d in range(N_DRAWS) for r in range(R)]
            a, a_sd, u, u_sd = per_repeat(plist)
            rows.append({"handling": st, "sections_hidden": k, "accuracy": a, "accuracy_sd": a_sd,
                         "macro_auc": u, "macro_auc_sd": u_sd})
    curve_df = pd.DataFrame(rows)
    curve_df["age_sex_only_accuracy"], curve_df["age_sex_only_macro_auc"] = round(base_acc, 3), round(base_auc, 3)
    curve_df.to_csv(config.TABLES_DIR / "missing_withheld_curve.csv", index=False)
    print("\n--- k random sections hidden (accuracy) ---")
    print(curve_df.pivot(index="sections_hidden", columns="handling", values="accuracy").to_string())

    # ---------- C. Cache for 06_reliability.py (person-level, git-ignored) ----------
    cache = pd.DataFrame({config.KEY: l1[config.KEY], "y": y, "n_sections_not_recorded": n_blank})
    for tag, p in [("age_sex", P0), ("standard", p_std), ("withheld", p_rob), ("withheld_recal", p_rob_recal.mean(axis=0))]:
        for i, c in enumerate(C):
            cache[f"p_{tag}_{c}"] = p[:, i]
    cache.to_csv(config.PROCESSED_DIR / "oof_level1.csv", index=False)
    tdf = pd.DataFrame(temps)
    tdf.to_csv(config.TABLES_DIR / "missing_recalibration.csv", index=False)
    print(f"Temperature of the withholding-trained model: mean {tdf['temperature'].mean():.2f} "
          f"(range {tdf['temperature'].min():.2f} to {tdf['temperature'].max():.2f}; above 1 = probabilities softened)")

    # ---------- Figures ----------
    colors = {"not_recorded": "#c0392b", "imputed": "#e67e22", "multiple_imputation": "#8e44ad",
              "withheld_model": "#1f77b4", "reduced_model": "#7f8c8d"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    d = natural[natural["sections_not_recorded"] != "any (all people)"]
    order = [b[0] for b in BANDS]
    for truth, style in [("all causes", dict(color="black", linewidth=2.2, marker="o"))] + [
            (c, dict(linewidth=1.2, marker="s", alpha=0.8)) for c in C]:
        sub = d[d["ground_truth"] == truth].set_index("sections_not_recorded").reindex(order)
        axes[0].plot(range(len(order)), sub["test_accuracy_standard_model"], label=mu.CLASS_LABELS.get(truth, "All causes"), **style)
    sub = d[d["ground_truth"] == "all causes"].set_index("sections_not_recorded").reindex(order)
    axes[0].plot(range(len(order)), sub["test_accuracy_age_sex_only"], color="grey", linestyle="--", label="All causes, age + sex only")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([f"{b}\n(n={int(sub.loc[b, 'n'])})" for b in order])
    axes[0].set_xlabel("Sections the investigator left blank (of 8)"); axes[0].set_ylabel("Held-out accuracy")
    axes[0].set_title("Records as they are", fontsize=10); axes[0].legend(fontsize=8); axes[0].grid(linestyle="--", alpha=0.5)
    single = scen[(scen["scenario_type"] == "single section") & scen["handling"].isin(["withheld_model", "reduced_model"])]
    names = [mu.SECTION_LABELS[s] for s in sec.names]
    ypos = np.arange(len(names))
    for k, st in enumerate(["withheld_model", "reduced_model"]):
        sub = single[single["handling"] == st].set_index("hidden_at_test_time").loc[names]
        val = sub["accuracy_change_vs_full_record"]
        err = [val - sub["acc_change_ci_low"], sub["acc_change_ci_high"] - val]
        axes[1].barh(ypos + (k - 0.5) * 0.38, val, 0.38, xerr=err, capsize=2, color=colors[st], label=STRATEGY_LABELS[st])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(ypos); axes[1].set_yticklabels(names); axes[1].invert_yaxis()
    axes[1].set_xlabel("Change in accuracy when the section is hidden (95% interval)")
    axes[1].set_title("One section hidden from the same people", fontsize=10); axes[1].legend(fontsize=8); axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "missing_information_natural.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, base, title in [(axes[0], "accuracy", base_acc, "Accuracy"), (axes[1], "macro_auc", base_auc, "Macro AUC")]:
        for st in CURVE_STRATEGIES:
            sub = curve_df[curve_df["handling"] == st]
            ax.errorbar(sub["sections_hidden"], sub[metric], yerr=sub[metric + "_sd"], marker="o", capsize=2,
                        color=colors[st], label=STRATEGY_LABELS[st])
        ax.axhline(base, color="grey", linestyle="--", linewidth=1, label="Model that only ever saw age + sex")
        ax.set_xlabel("Number of sections hidden at random (of 8)"); ax.set_title(title, fontsize=10)
        ax.grid(linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same people, less information: what happens as sections of the record are hidden", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "missing_information_withheld.png", dpi=150)
    plt.close(fig)
    mu.log("04_missing_information finished", t0)
