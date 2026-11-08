import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
import config


def get_feature_sets(features):
    """Return a dictionary of column lists for M0, Missing_Only, M1, and M2."""
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
    missing_flag_cols = [
        c for c in (lifestyle_cols + med_hist_cols)
        if c.endswith("_missing")
    ]

    return {
        "M0_age_sex": base_cols,
        "M_missing_flags_only": base_cols + missing_flag_cols,
        "M1_plus_lifestyle": base_cols + lifestyle_cols,
        "M2_plus_med_history": base_cols + lifestyle_cols + med_hist_cols,
    }


def make_model(model_name):
    """Build either HistGradientBoosting or LogisticRegression pipeline."""
    if model_name == "HistGB":
        return HistGradientBoostingClassifier(random_state=config.SEED)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.SEED
        )),
    ])


def get_oof_predictions(X, y, model_name):
    """Run 5-fold Stratified CV and collect out-of-fold probabilities."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    n_classes = len(np.unique(y))
    oof_probs = np.zeros((len(y), n_classes))

    for train_idx, test_idx in skf.split(X, y):
        model = make_model(model_name)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_probs[test_idx] = model.predict_proba(X.iloc[test_idx])

    return oof_probs


def fast_auc_and_f1(y_true, probs, y_pred):
    """Fast calculation of macro 1-vs-rest AUC and macro F1 for bootstrap resamples."""
    n_classes = probs.shape[1]
    aucs = []
    f1s = []
    for c in range(n_classes):
        pos = (y_true == c)
        n_pos = int(pos.sum())
        n_neg = len(y_true) - n_pos
        if n_pos > 0 and n_neg > 0:
            order = np.argsort(probs[:, c], kind="mergesort")
            ranks = np.empty(len(y_true), dtype=float)
            ranks[order] = np.arange(1, len(y_true) + 1)
            aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

        tp = int(np.sum(pos & (y_pred == c)))
        fp = int(np.sum((~pos) & (y_pred == c)))
        fn = n_pos - tp
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)

    return float(np.mean(aucs)), float(np.mean(f1s))


def bootstrap_metrics(y_true, oof_probs, n_boot=1000):
    """Compute macro-AUC and macro-F1 with 95% bootstrap confidence intervals."""
    rng = np.random.default_rng(config.SEED)
    y_pred = np.argmax(oof_probs, axis=1)

    auc_point = roc_auc_score(y_true, oof_probs, multi_class="ovr", average="macro")
    f1_point = f1_score(y_true, y_pred, average="macro")

    aucs = []
    f1s = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, f = fast_auc_and_f1(y_true[idx], oof_probs[idx], y_pred[idx])
        aucs.append(a)
        f1s.append(f)

    return {
        "macro_auc": round(float(auc_point), 3),
        "auc_ci_low": round(float(np.percentile(aucs, 2.5)), 3),
        "auc_ci_high": round(float(np.percentile(aucs, 97.5)), 3),
        "macro_f1": round(float(f1_point), 3),
        "f1_ci_low": round(float(np.percentile(f1s, 2.5)), 3),
        "f1_ci_high": round(float(np.percentile(f1s, 97.5)), 3),
    }


def plot_baseline(results_df):
    """Save bar chart of Level-1 Macro AUC with 95% CIs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ["HistGB", "LogReg"]
    feature_sets = list(results_df["feature_set"].unique())
    x = np.arange(len(feature_sets))
    width = 0.35

    for i, m_name in enumerate(models):
        sub = results_df[results_df["model"] == m_name].set_index("feature_set").loc[feature_sets]
        means = sub["macro_auc"].values
        err_low = means - sub["auc_ci_low"].values
        err_high = sub["auc_ci_high"].values - means
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=m_name, yerr=[err_low, err_high], capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([
        "M0: Age + Sex",
        "Missing Flags Only",
        "M1: + Lifestyle",
        "M2: + Med History",
    ], rotation=12)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Macro One-vs-Rest AUC (95% CI)")
    ax.set_title("Level 1 Cause-of-Death Classification (n = 1,380)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "level1_baseline.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)

    classes = ["natural_cardiac", "drug_alcohol", "injury"]
    l1 = cohort[cohort["level1"].isin(classes)][[config.KEY, "level1"]].merge(
        features, on=config.KEY
    )

    class_to_int = {cls: i for i, cls in enumerate(classes)}
    y = l1["level1"].map(class_to_int).values
    f_sets = get_feature_sets(features)

    rows = []
    for model_name in ["HistGB", "LogReg"]:
        for set_name, cols in f_sets.items():
            X = l1[cols]
            oof = get_oof_predictions(X, y, model_name)
            stats = bootstrap_metrics(y, oof, n_boot=1000)
            row = {"model": model_name, "feature_set": set_name, "n_features": len(cols)}
            row.update(stats)
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(config.TABLES_DIR / "level1_baseline.csv", index=False)
    plot_baseline(results_df)

    print("\n--- Level 1 Baseline Models (n = 1,380) ---")
    print(results_df.to_string(index=False))
