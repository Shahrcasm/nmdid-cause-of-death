"""Phase 2 / RQ1: incremental value of lifestyle (L) and medical history (H).

Nested sets M0 (age, sex) -> M1 (+L) -> M2 (+H), three model families, 5x5
repeated stratified CV, paired bootstrap gains, calibration curves, per-class
AUC, and the reverse entry order (M0 -> +H -> +L) so the attribution of the
gain to L versus H does not depend on which block is entered first.

Outputs: rq1_metrics.csv, rq1_gains.csv, rq1_per_class_auc.csv,
         rq1_reverse_order.csv, figures/rq1_calibration_level{1,2}.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import config
import model_utils as mu

PAIRS = [("M1_minus_M0", "M1_lifestyle", "M0_base"),
         ("M2_minus_M1", "M2_med_history", "M1_lifestyle"),
         ("M2_minus_M0", "M2_med_history", "M0_base")]


def plot_calibration(y_true, probs_by_set, class_names, out_path, title):
    fig, axes = plt.subplots(1, len(class_names), figsize=(12, 3.8), sharey=True)
    for c_idx, cls_name in enumerate(class_names):
        ax = axes[c_idx]
        y_bin = (y_true == c_idx).astype(int)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        for set_name, probs in probs_by_set.items():
            frac_pos, mean_pred = calibration_curve(y_bin, probs[:, c_idx], n_bins=8)
            ax.plot(mean_pred, frac_pos, marker="o", label=set_name)
        ax.set_title(cls_name)
        ax.set_xlabel("Mean Predicted Probability")
        if c_idx == 0:
            ax.set_ylabel("Observed Fraction Positive")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_level(level_name, df, label_col, class_names, nested_sets):
    metrics, gains, per_class = [], [], []
    for model_name in ["HistGB", "RF", "LogReg"]:
        print(f"  [{level_name}] model: {model_name}...", flush=True)
        res = mu.run_sets(df, label_col, class_names, nested_sets, model_name, pairs=PAIRS)
        for set_name, probs in res["probs"].items():
            for c_idx, cls_name in enumerate(class_names):
                per_class.append({"level": level_name, "model": model_name, "feature_set": set_name,
                                  "class_name": cls_name,
                                  "ovr_auc": round(float(roc_auc_score((res["y"] == c_idx).astype(int), probs[:, c_idx])), 3)})
        for r in res["metrics"] + res["gains"]:
            r["level"] = level_name
        metrics += res["metrics"]
        gains += res["gains"]
        if model_name == "HistGB":
            plot_calibration(res["y"], res["probs"], class_names,
                             config.FIGURES_DIR / f"rq1_calibration_{level_name.lower()}.png",
                             f"{level_name} Calibration Curves (HistGB, 5x5 Repeated CV)")
    return metrics, gains, per_class


if __name__ == "__main__":
    cohort, features = mu.load_data()
    nested_sets = mu.get_nested_sets(features)
    base, lifestyle, med_hist = mu.get_column_groups(features)

    l1_df = cohort[cohort["level1"].isin(mu.LEVEL1_CLASSES)][[config.KEY, "level1"]].merge(features, on=config.KEY)
    l2_df = cohort[(cohort["level1"] == "injury") & cohort["intent"].isin(mu.LEVEL2_CLASSES)][
        [config.KEY, "intent"]].merge(features, on=config.KEY)

    print(f"Running RQ1 Level 1 (n={len(l1_df)})...")
    m1, g1, pc1 = run_level("Level1", l1_df, "level1", mu.LEVEL1_CLASSES, nested_sets)
    print(f"Running RQ1 Level 2 (n={len(l2_df)})...")
    m2, g2, pc2 = run_level("Level2", l2_df, "intent", mu.LEVEL2_CLASSES, nested_sets)

    metrics_df = pd.DataFrame(m1 + m2)
    gains_df = pd.DataFrame(g1 + g2)
    front = ["level", "model", "feature_set", "n"]
    metrics_df = metrics_df[front + [c for c in metrics_df.columns if c not in front]]
    gains_df = gains_df[["level", "model", "comparison", "n"] + [c for c in gains_df.columns if c not in ["level", "model", "comparison", "n"]]]
    metrics_df.to_csv(config.TABLES_DIR / "rq1_metrics.csv", index=False)
    gains_df.to_csv(config.TABLES_DIR / "rq1_gains.csv", index=False)
    pd.DataFrame(pc1 + pc2).to_csv(config.TABLES_DIR / "rq1_per_class_auc.csv", index=False)

    # Reverse entry order (HistGB): does H add less because L was entered first?
    print("Running reverse entry order (Level 1, HistGB)...")
    rev_sets = {"M0_base": base, "M0_plus_H": base + med_hist, "M2_med_history": base + med_hist + lifestyle}
    rev_pairs = [("H_minus_M0", "M0_plus_H", "M0_base"), ("L_after_H", "M2_med_history", "M0_plus_H")]
    res = mu.run_sets(l1_df, "level1", mu.LEVEL1_CLASSES, rev_sets, "HistGB", pairs=rev_pairs)
    rev = pd.DataFrame(res["metrics"])[["feature_set", "n", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref", "gain_ci_low", "gain_ci_high"]]
    rev_g = pd.DataFrame(res["gains"])[["comparison", "macro_auc_gain", "macro_auc_ci_low", "macro_auc_ci_high"]]
    rev["level"], rev["model"] = "Level1", "HistGB"
    rev.to_csv(config.TABLES_DIR / "rq1_reverse_order.csv", index=False)

    print("\n--- RQ1 Metrics ---")
    print(metrics_df[["level", "model", "feature_set", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high",
                      "auc_repeat_sd", "macro_f1", "brier"]].to_string(index=False))
    print("\n--- RQ1 Gains ---")
    print(gains_df[["level", "model", "comparison", "macro_auc_gain", "macro_auc_ci_low", "macro_auc_ci_high"]].to_string(index=False))
    print("\n--- Reverse entry order (Level 1, HistGB) ---")
    print(rev_g.to_string(index=False))
