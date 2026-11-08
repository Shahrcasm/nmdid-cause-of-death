import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.calibration import calibration_curve
import config


def get_nested_sets(features):
    """Return M0 (base), M1 (+ lifestyle), and M2 (+ medical history) column lists."""
    rq3_only = ["n_medications", "any_medication", "med_missing", "nok_interview"]
    med_hist_cols = [
        c for c in features.columns
        if c.startswith("dx_") or c.startswith("medhist_")
    ]
    base_cols = ["age_years", "male"]
    lifestyle_cols = [
        c for c in features.columns
        if c not in ([config.KEY] + base_cols + med_hist_cols + rq3_only)
    ]

    return {
        "M0_base": base_cols,
        "M1_lifestyle": base_cols + lifestyle_cols,
        "M2_med_history": base_cols + lifestyle_cols + med_hist_cols,
    }


def make_model(model_name):
    """Build LogisticRegression, RandomForest, or HistGradientBoosting."""
    if model_name == "HistGB":
        return HistGradientBoostingClassifier(
            max_iter=50, early_stopping=True, random_state=config.SEED
        )

    if model_name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=config.SEED,
                n_jobs=1,
            )),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.SEED
        )),
    ])


def get_repeated_oof_probs(X, y, model_name):
    """Run 5-fold Stratified CV repeated 5 times and average OOF probabilities per person."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=config.SEED)
    n_classes = len(np.unique(y))
    prob_sum = np.zeros((len(y), n_classes))
    counts = np.zeros((len(y), 1))

    for train_idx, test_idx in rskf.split(X, y):
        model = make_model(model_name)
        model.fit(X.iloc[train_idx], y[train_idx])
        prob_sum[test_idx] += model.predict_proba(X.iloc[test_idx])
        counts[test_idx] += 1.0

    return prob_sum / counts


def fast_ovr_auc(y_true, probs):
    """Fast tie-aware rank-based 1-vs-rest macro AUC (identical to sklearn roc_auc_score)."""
    n_classes = probs.shape[1]
    aucs = []
    for c in range(n_classes):
        pos = (y_true == c)
        n_pos = int(pos.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(probs[:, c], method="average")
        auc_c = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs.append(auc_c)
    return float(np.mean(aucs))


def compute_point_metrics(y_true, probs):
    """Compute macro AUC, macro F1, balanced accuracy, and Brier score."""
    n_classes = probs.shape[1]
    y_pred = np.argmax(probs, axis=1)

    f1s = []
    recalls = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        f1_c = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(f1_c)
        recalls.append(rec_c)

    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    return {
        "macro_auc": fast_ovr_auc(y_true, probs),
        "macro_f1": float(np.mean(f1s)),
        "bal_acc": float(np.mean(recalls)),
        "brier": brier,
    }


def bootstrap_metrics_and_gains(y_true, probs_by_set, n_boot=1000):
    """Bootstrap 95% CIs for each feature set and for gains (M1 - M0, M2 - M1)."""
    rng = np.random.default_rng(config.SEED)
    n = len(y_true)
    n_classes = probs_by_set["M0_base"].shape[1]

    metric_names = ["macro_auc", "macro_f1", "bal_acc", "brier"]
    boot_draws = {s: {m: [] for m in metric_names} for s in probs_by_set}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < n_classes:
            continue
        for set_name, probs in probs_by_set.items():
            vals = compute_point_metrics(y_true[idx], probs[idx])
            for m in metric_names:
                boot_draws[set_name][m].append(vals[m])

    metric_rows = []
    for set_name, probs in probs_by_set.items():
        pt = compute_point_metrics(y_true, probs)
        row = {"feature_set": set_name}
        for m in metric_names:
            draws = boot_draws[set_name][m]
            row[m] = round(pt[m], 3)
            row[m + "_ci_low"] = round(float(np.percentile(draws, 2.5)), 3)
            row[m + "_ci_high"] = round(float(np.percentile(draws, 97.5)), 3)
        metric_rows.append(row)

    gain_pairs = [
        ("M1_minus_M0", "M1_lifestyle", "M0_base"),
        ("M2_minus_M1", "M2_med_history", "M1_lifestyle"),
        ("M2_minus_M0", "M2_med_history", "M0_base"),
    ]
    gain_rows = []
    for label, set_b, set_a in gain_pairs:
        pt_b = compute_point_metrics(y_true, probs_by_set[set_b])
        pt_a = compute_point_metrics(y_true, probs_by_set[set_a])
        row = {"comparison": label}
        for m in metric_names:
            diff_pt = pt_b[m] - pt_a[m]
            diff_draws = np.array(boot_draws[set_b][m]) - np.array(boot_draws[set_a][m])
            row[m + "_gain"] = round(float(diff_pt), 3)
            row[m + "_ci_low"] = round(float(np.percentile(diff_draws, 2.5)), 3)
            row[m + "_ci_high"] = round(float(np.percentile(diff_draws, 97.5)), 3)
        gain_rows.append(row)

    return metric_rows, gain_rows


def plot_calibration(y_true, probs_by_set, class_names, out_path, title):
    """Plot reliability (calibration) curves for M0, M1, M2 across classes (HistGB)."""
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
    """Run all 3 models on M0, M1, M2 for one level and return summary tables."""
    class_to_int = {c: i for i, c in enumerate(class_names)}
    y = df[label_col].map(class_to_int).values

    all_metrics = []
    all_gains = []
    all_per_class = []

    for model_name in ["HistGB", "RF", "LogReg"]:
        print(f"  [{level_name}] model: {model_name}...", flush=True)
        probs_by_set = {}
        for set_name, cols in nested_sets.items():
            X = df[cols]
            probs = get_repeated_oof_probs(X, y, model_name)
            probs_by_set[set_name] = probs

            for c_idx, cls_name in enumerate(class_names):
                y_bin = (y == c_idx).astype(int)
                c_auc = roc_auc_score(y_bin, probs[:, c_idx])
                all_per_class.append({
                    "level": level_name,
                    "model": model_name,
                    "feature_set": set_name,
                    "class_name": cls_name,
                    "ovr_auc": round(float(c_auc), 3),
                })

        m_rows, g_rows = bootstrap_metrics_and_gains(y, probs_by_set, n_boot=1000)
        for r in m_rows:
            r["level"] = level_name
            r["model"] = model_name
            all_metrics.append(r)
        for r in g_rows:
            r["level"] = level_name
            r["model"] = model_name
            all_gains.append(r)

        if model_name == "HistGB":
            plot_calibration(
                y, probs_by_set, class_names,
                config.FIGURES_DIR / f"rq1_calibration_{level_name.lower()}.png",
                f"{level_name} Calibration Curves (HistGB, 5x5 Repeated CV)",
            )

    return all_metrics, all_gains, all_per_class


if __name__ == "__main__":
    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)
    nested_sets = get_nested_sets(features)

    l1_classes = ["natural_cardiac", "drug_alcohol", "injury"]
    l1_df = cohort[cohort["level1"].isin(l1_classes)][[config.KEY, "level1"]].merge(
        features, on=config.KEY
    )

    l2_classes = ["accident", "homicide", "suicide"]
    l2_df = cohort[
        (cohort["level1"] == "injury") & (cohort["intent"].isin(l2_classes))
    ][[config.KEY, "intent"]].merge(features, on=config.KEY)

    print("Running RQ1 Level 1 (n=1380)...")
    m1, g1, pc1 = run_level("Level1", l1_df, "level1", l1_classes, nested_sets)
    print("Running RQ1 Level 2 (n=487)...")
    m2, g2, pc2 = run_level("Level2", l2_df, "intent", l2_classes, nested_sets)

    metrics_df = pd.DataFrame(m1 + m2)
    gains_df = pd.DataFrame(g1 + g2)
    per_class_df = pd.DataFrame(pc1 + pc2)

    metrics_df.to_csv(config.TABLES_DIR / "rq1_metrics.csv", index=False)
    gains_df.to_csv(config.TABLES_DIR / "rq1_gains.csv", index=False)
    per_class_df.to_csv(config.TABLES_DIR / "rq1_per_class_auc.csv", index=False)

    print("\n--- RQ1 Metrics ---")
    print(metrics_df[["level", "model", "feature_set", "macro_auc", "macro_auc_ci_low", "macro_auc_ci_high", "macro_f1", "brier"]].to_string(index=False))
    print("\n--- RQ1 Gains ---")
    print(gains_df[["level", "model", "comparison", "macro_auc_gain", "macro_auc_ci_low", "macro_auc_ci_high"]].to_string(index=False))
