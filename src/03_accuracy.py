"""Step 3 - Accuracy: how well can the broad cause of death be estimated?

  1. Primary outcome (Level 1) and secondary outcome (Level 2, intent within
     injury deaths): nested input sets M0 -> M1 -> M2 for gradient boosting,
     random forest, logistic regression and their soft-vote ensemble.
  2. Per-class and pairwise discrimination, confusion matrix (Level 1, M2).
  3. Accuracy ceiling: extended inputs (medications; race and ethnicity) and a
     small nested search, to show the most these records can support.
  4. Performance within subgroups (sex, age band, race, ethnicity).

Every number is computed on people the model did not learn from.

Outputs: tables/accuracy_metrics.csv, accuracy_gains.csv, accuracy_per_class.csv,
         accuracy_pairwise.csv, confusion_matrix.csv, accuracy_ceiling.csv,
         accuracy_subgroups.csv, figures/accuracy_by_inputs.png
"""
import time
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import model_utils as mu

MODELS = ["HistGB", "RF", "LogReg"]
PAIRS = [("M1_minus_M0", "M1_lifestyle", "M0_base"),
         ("M2_minus_M1", "M2_med_history", "M1_lifestyle"),
         ("M2_minus_M0", "M2_med_history", "M0_base")]
FRONT = ["level", "model", "feature_set", "n"]


def run_level(df, label_col, classes, nested, level, t0):
    """All models + their soft-vote ensemble for one outcome level."""
    y = mu.encode(df[label_col], classes)
    majority = round(float(np.bincount(y).max() / len(y)), 3)
    metric_rows, gain_rows, probs, per_rep = [], [], {}, {}
    for model_name in MODELS:
        res = mu.run_sets(df, label_col, classes, nested, model_name, pairs=PAIRS)
        probs[model_name], per_rep[model_name] = res["probs"], res["per_repeat"]
        metric_rows += res["metrics"]
        gain_rows += res["gains"]
        mu.log(f"{level} {model_name} done", t0)
    # Soft vote: the plain average of the three models' probabilities (no weights, no tuning).
    probs["Ensemble"] = {s: np.mean([probs[m][s] for m in MODELS], axis=0) for s in nested}
    per_rep["Ensemble"] = {s: np.mean([per_rep[m][s] for m in MODELS], axis=0) for s in nested}
    rows, gains = mu.bootstrap_compare(y, probs["Ensemble"], reference="M0_base", pairs=PAIRS,
                                       per_repeat=per_rep["Ensemble"])
    metric_rows += mu.tag(rows, model="Ensemble")
    gain_rows += mu.tag(gains, model="Ensemble")
    for r in metric_rows:
        r["top2_accuracy"] = round(mu.top2_accuracy(y, probs[r["model"]][r["feature_set"]]), 3)
        r["majority_class_accuracy"] = majority
    return mu.tag(metric_rows, level=level), mu.tag(gain_rows, level=level), probs, y


def per_class_rows(y, p, classes, model, feature_set):
    pred = p.argmax(axis=1)
    rows = []
    for i, c in enumerate(classes):
        tp = int(((pred == i) & (y == i)).sum())
        rows.append({"model": model, "feature_set": feature_set, "class_name": c, "n_true": int((y == i).sum()),
                     "ovr_auc": round(mu.fast_ovr_auc((y == i).astype(int), np.column_stack([1 - p[:, i], p[:, i]])), 3),
                     "recall": round(tp / max(1, int((y == i).sum())), 3),
                     "precision": round(tp / max(1, int((pred == i).sum())), 3)})
    return rows


def counts_text(y, classes):
    """Class counts for a table; counts below 5 are suppressed."""
    return "; ".join(f"{c}: {n if n >= 5 else '<5'}" for c, n in zip(classes, np.bincount(y, minlength=len(classes))))


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)
    l1, y1 = mu.level1_frame(cohort, features)
    l2, y2 = mu.level2_frame(cohort, features)
    C1 = mu.LEVEL1_CLASSES
    print("Level 1:", len(l1), dict(l1["level1"].value_counts()), "| Level 2:", len(l2), dict(l2["intent"].value_counts()))
    print("Inputs: M0", len(nested["M0_base"]), "| M1", len(nested["M1_lifestyle"]), "| M2", len(nested["M2_med_history"]))

    # ---------- 1. Nested input sets, both outcome levels ----------
    m1_rows, g1_rows, probs1, _ = run_level(l1, "level1", C1, nested, "Level1", t0)
    m2_rows, g2_rows, _, _ = run_level(l2, "intent", mu.LEVEL2_CLASSES, nested, "Level2", t0)
    metrics = pd.DataFrame(m1_rows + m2_rows)
    metrics = metrics[FRONT + [c for c in metrics.columns if c not in FRONT]]
    metrics.to_csv(config.TABLES_DIR / "accuracy_metrics.csv", index=False)
    gains = pd.DataFrame(g1_rows + g2_rows)
    front = ["level", "model", "comparison", "n"]
    gains[front + [c for c in gains.columns if c not in front]].to_csv(config.TABLES_DIR / "accuracy_gains.csv", index=False)
    show = ["level", "model", "feature_set", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_auc",
            "macro_auc_ci_low", "macro_auc_ci_high", "bal_acc", "brier", "top2_accuracy", "acc_gain_vs_ref"]
    print("\n--- Accuracy and discrimination (held-out) ---")
    print(metrics[show].to_string(index=False))

    # ---------- 2. Per class, pairwise, confusion matrix (Level 1) ----------
    pc_rows = []
    for model in ["HistGB", "Ensemble"]:
        for fs in ["M0_base", "M2_med_history"]:
            pc_rows += per_class_rows(y1, probs1[model][fs], C1, model, fs)
    pd.DataFrame(pc_rows).to_csv(config.TABLES_DIR / "accuracy_per_class.csv", index=False)
    pw_rows = []
    for model in ["HistGB", "Ensemble"]:
        for fs in ["M0_base", "M2_med_history"]:
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                pw_rows.append({"model": model, "feature_set": fs, "pair": f"{C1[a]} vs {C1[b]}",
                                "auc": round(mu.pairwise_auc(y1, probs1[model][fs], a, b), 3)})
    pd.DataFrame(pw_rows).to_csv(config.TABLES_DIR / "accuracy_pairwise.csv", index=False)
    p_main = probs1["HistGB"]["M2_med_history"]
    cm = pd.crosstab(pd.Series(y1, name="true"), pd.Series(p_main.argmax(axis=1), name="predicted"))
    cm_rows = [{"true_cause": C1[i], "predicted_cause": C1[j], "n": int(cm.loc[i, j]),
                "pct_of_true": round(100 * cm.loc[i, j] / cm.loc[i].sum(), 1)} for i in cm.index for j in cm.columns]
    pd.DataFrame(cm_rows).to_csv(config.TABLES_DIR / "confusion_matrix.csv", index=False)
    print("\n--- Per class (HistGB, M2) ---")
    print(pd.DataFrame(pc_rows).query("model == 'HistGB' and feature_set == 'M2_med_history'").to_string(index=False))

    # ---------- 3. Accuracy ceiling (Level 1) ----------
    l1x = l1
    m2 = nested["M2_med_history"]
    ext_sets = {"M2 main inputs": m2,
                "M2 + medications": m2 + config.EXTENDED_INPUTS,
                "M2 + medications + race and ethnicity": m2 + config.EXTENDED_INPUTS + config.DEMOGRAPHIC_EXTRA}
    REF = "M2 main inputs | fixed settings"
    cprobs, crep, note = {}, {}, {}
    for name, cols in ext_sets.items():
        cprobs[f"{name} | fixed settings"], crep[f"{name} | fixed settings"] = mu.repeated_oof_probs(l1x[cols], y1, "HistGB")
    mu.log("ceiling: fixed-setting runs done", t0)
    for name in ["M2 main inputs", "M2 + medications + race and ethnicity"]:
        key = f"{name} | nested search"
        cprobs[key], crep[key], chosen = mu.nested_tuned_oof_probs(l1x[ext_sets[name]], y1)
        note[key] = "leaves per tree chosen (count of outer folds): " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(chosen).items()))
        mu.log(f"ceiling: nested search for {name} done", t0)
    cprobs["M2 main inputs | ensemble of three models"] = probs1["Ensemble"]["M2_med_history"]
    crows, _ = mu.bootstrap_compare(y1, cprobs, reference=REF, per_repeat=crep)
    for r in crows:
        inputs, model = r.pop("feature_set").split(" | ")
        r.update({"inputs": inputs, "model": "HistGB, " + model if "ensemble" not in model else model,
                  "top2_accuracy": round(mu.top2_accuracy(y1, cprobs[f"{inputs} | {model}"]), 3),
                  "note": note.get(f"{inputs} | {model}", "")})
    ceiling = pd.DataFrame(crows)
    front = ["inputs", "model", "n"]
    ceiling = ceiling[front + [c for c in ceiling.columns if c not in front]]
    ceiling.to_csv(config.TABLES_DIR / "accuracy_ceiling.csv", index=False)
    print("\n--- Accuracy ceiling (Level 1; gains are against M2 with fixed settings) ---")
    print(ceiling[front + ["accuracy", "accuracy_ci_low", "accuracy_ci_high", "acc_gain_vs_ref", "acc_gain_ci_low",
                           "acc_gain_ci_high", "macro_auc", "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high", "note"]].to_string(index=False))

    # ---------- 4. Subgroups (one model for everyone, evaluated inside each group) ----------
    p0 = probs1["HistGB"]["M0_base"]
    white = (l1x[["race_native_american", "race_black", "race_other"]].sum(axis=1) == 0).values
    groups = [("sex", "Male", (l1x["male"] == 1).values), ("sex", "Female", (l1x["male"] == 0).values),
              ("age", "18-39", ((l1x["age_years"] >= 18) & (l1x["age_years"] < 40)).values),
              ("age", "40-59", ((l1x["age_years"] >= 40) & (l1x["age_years"] < 60)).values),
              ("age", "60+", (l1x["age_years"] >= 60).values),
              ("race", "White", white), ("race", "Native American", (l1x["race_native_american"] == 1).values),
              ("race", "Black or other", (~white & (l1x["race_native_american"] == 0)).values),
              ("ethnicity", "Hispanic or Latino", (l1x["ethnicity_hispanic"] == 1).values),
              ("ethnicity", "Not Hispanic or Latino", (l1x["ethnicity_hispanic"] == 0).values)]
    sg_rows = []
    for kind, label, mask in groups:
        idx = np.flatnonzero(mask)
        if len(idx) < 50 or len(np.unique(y1[idx])) < 3:
            continue
        rows, _ = mu.bootstrap_compare(y1[idx], {"M0_base": p0[idx], "M2_med_history": p_main[idx]}, reference="M0_base")
        sg_rows += mu.tag(rows, subgroup_type=kind, subgroup=label, class_counts=counts_text(y1[idx], C1), model="HistGB")
    sg = pd.DataFrame(sg_rows)
    front = ["subgroup_type", "subgroup", "model", "feature_set", "n", "class_counts"]
    sg = sg[front + [c for c in sg.columns if c not in front]]
    sg.to_csv(config.TABLES_DIR / "accuracy_subgroups.csv", index=False)
    print("\n--- Subgroups (HistGB; model trained on everyone) ---")
    print(sg[["subgroup_type", "subgroup", "feature_set", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
              "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "acc_gain_vs_ref"]].to_string(index=False))

    # ---------- Figure ----------
    d = metrics[metrics["level"] == "Level1"]
    labels = {"M0_base": "Age + sex", "M1_lifestyle": "+ lifestyle", "M2_med_history": "+ medical history"}
    names = {"HistGB": "Gradient boosting", "RF": "Random forest", "LogReg": "Logistic regression", "Ensemble": "Ensemble of the three"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy (share of people with the correct cause)"),
                              (axes[1], "macro_auc", "Macro AUC (0.5 = guessing, 1.0 = perfect)")]:
        width = 0.2
        for k, model in enumerate(names):
            sub = d[d["model"] == model].set_index("feature_set").loc[list(labels)]
            x = np.arange(3) + (k - 1.5) * width
            err = [sub[metric] - sub[metric + "_ci_low"], sub[metric + "_ci_high"] - sub[metric]]
            ax.bar(x, sub[metric], width, yerr=err, capsize=2, label=names[model])
        if metric == "accuracy":
            ax.axhline(d["majority_class_accuracy"].iloc[0], color="grey", linestyle="--", linewidth=1)
            ax.text(2.45, d["majority_class_accuracy"].iloc[0] + 0.005, "always guess the largest group", ha="right", fontsize=8, color="grey")
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(list(labels.values()))
        ax.set_ylim(0.3 if metric == "accuracy" else 0.5, 0.7 if metric == "accuracy" else 0.85)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Broad cause of death from the record (n = {len(y1)}; held-out people, 95% intervals)", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "accuracy_by_inputs.png", dpi=150)
    plt.close(fig)
    mu.log("03_accuracy finished", t0)
