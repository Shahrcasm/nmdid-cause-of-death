"""Phase 2 baseline (Level 1): age + sex, missing flags only, M1, M2.

Uses the same models, CV and bootstrap as every later script (model_utils), so
the numbers are directly comparable with rq1_metrics.csv and rq3_artifact.csv.
Outputs: results/tables/level1_baseline.csv, results/figures/level1_baseline.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import model_utils as mu


def get_feature_sets(features):
    base, lifestyle, med_hist = mu.get_column_groups(features)
    flags = [c for c in lifestyle + med_hist if c.endswith("_missing")]
    return {
        "M0_age_sex": base,
        "M_missing_flags_only": base + flags,
        "M1_plus_lifestyle": base + lifestyle,
        "M2_plus_med_history": base + lifestyle + med_hist,
    }


def plot_baseline(results_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ["HistGB", "LogReg"]
    feature_sets = list(results_df["feature_set"].unique())
    x = np.arange(len(feature_sets))
    width = 0.35
    for i, m_name in enumerate(models):
        sub = results_df[results_df["model"] == m_name].set_index("feature_set").loc[feature_sets]
        means = sub["macro_auc"].values
        err = [means - sub["macro_auc_ci_low"].values, sub["macro_auc_ci_high"].values - means]
        ax.bar(x + (i - 0.5) * width, means, width, label=m_name, yerr=err, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["M0: Age + Sex", "Missing Flags Only", "M1: + Lifestyle", "M2: + Med History"], rotation=12)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Macro One-vs-Rest AUC (95% CI)")
    ax.set_title("Level 1 Cause-of-Death Classification (n = 1,380, 5x5 repeated CV)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "level1_baseline.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cohort, features = mu.load_data()
    l1 = cohort[cohort["level1"].isin(mu.LEVEL1_CLASSES)][[config.KEY, "level1"]].merge(features, on=config.KEY)
    f_sets = get_feature_sets(features)
    rows = []
    for model_name in ["HistGB", "LogReg"]:
        res = mu.run_sets(l1, "level1", mu.LEVEL1_CLASSES, f_sets, model_name, reference="M0_age_sex")
        for r in res["metrics"]:
            r["n_features"] = len(f_sets[r["feature_set"]])
        rows += res["metrics"]
    results_df = pd.DataFrame(rows)
    front = ["model", "feature_set", "n_features", "n"]
    results_df = results_df[front + [c for c in results_df.columns if c not in front]]
    results_df.to_csv(config.TABLES_DIR / "level1_baseline.csv", index=False)
    plot_baseline(results_df)
    print("\n--- Level 1 Baseline Models (n = 1,380) ---")
    print(results_df[front + ["macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "auc_gain_vs_ref",
                              "gain_ci_low", "gain_ci_high"]].to_string(index=False))
