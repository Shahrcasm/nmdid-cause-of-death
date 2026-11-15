"""Shared modelling helpers used by scripts 03-09 and the demo app.

One implementation of: feature-set definitions, model constructors,
5-fold stratified CV repeated 5 times (random_state=42) with out-of-fold
probabilities averaged per person, tie-aware macro one-vs-rest AUC, and
1,000-resample paired bootstrap confidence intervals.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import config

LEVEL1_CLASSES = ["natural_cardiac", "drug_alcohol", "injury"]
LEVEL2_CLASSES = ["accident", "homicide", "suicide"]
LEVEL3_CLASSES = ["stab", "gunshot"]

# Fields that only come from the family (next-of-kin) interview.
FAMILY_ONLY_PREFIXES = ["edu_level", "ses_adult", "ses_child", "diet_", "lift_",
                        "occ_", "living_bmi"]
# Fields that investigators also record (scene, records, autopsy).
SHARED_PREFIXES = ["drink_", "tobtype_", "tobuse_", "tob_missing", "drug_",
                   "marital_", "cadaver_bmi"]

# Below this training-fold size HistGB's internal 10% validation split is too
# small for early stopping to be reliable (it stopped at iteration 1 on Level 3).
SMALL_N_NO_EARLY_STOP = 200

CONTINUOUS_COLS = ["cadaver_bmi", "living_bmi", "edu_level", "ses_adult", "ses_child"]
MEDHIST_BLOCKS = ["broken_bone", "facial_trauma", "surgery", "implant"]


def starts_with_any(name, prefixes):
    return any(name.startswith(p) for p in prefixes)


def load_data():
    """Load cohort.csv and features.csv with the join key as text."""
    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    cohort["age_years"] = pd.to_numeric(cohort["age_years"], errors="coerce")
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)
    return cohort, features


def get_column_groups(features):
    """Return base, lifestyle and medical-history column lists."""
    med_hist = [c for c in features.columns if c.startswith("dx_") or c.startswith("medhist_")]
    base = ["age_years", "male"]
    excluded = set([config.KEY] + base + med_hist + config.RQ3_ONLY + config.SENSITIVITY_ONLY)
    lifestyle = [c for c in features.columns if c not in excluded]
    return base, lifestyle, med_hist


def get_nested_sets(features):
    base, lifestyle, med_hist = get_column_groups(features)
    return {
        "M0_base": base,
        "M1_lifestyle": base + lifestyle,
        "M2_med_history": base + lifestyle + med_hist,
    }


def missingness_blocks(columns):
    """(flag column, value columns) for every recorded block in `columns`."""
    columns = list(columns)
    blocks = []

    def add(flag, vals):
        vals = [v for v in vals if v in columns]
        if flag in columns and vals:
            blocks.append((flag, vals))

    add("drink_missing", [c for c in columns if c.startswith("drink_") and c != "drink_missing"])
    add("tob_missing", [c for c in columns if c.startswith("tobtype_") or c.startswith("tobuse_")])
    add("drug_missing", [c for c in columns if c.startswith("drug_") and c != "drug_missing"])
    add("marital_missing", [c for c in columns if c.startswith("marital_") and c != "marital_missing"])
    add("diet_missing", [c for c in columns if c.startswith("diet_") and c != "diet_missing"])
    add("lift_missing", [c for c in columns if c.startswith("lift_") and c != "lift_missing"])
    add("occ_missing", ["occ_firearm_access"])
    for c in CONTINUOUS_COLS + ["cadaver_bmi_decomp_blanked"]:
        add(c + "_missing", [c])
    add("dx_missing", [c for c in columns if c.startswith("dx_") and c != "dx_missing"])
    for hb in MEDHIST_BLOCKS:
        add(f"medhist_{hb}_missing", [f"medhist_{hb}_any"])
    return blocks


def randomise_blanks(df, columns, seed=config.SEED):
    """Destroy the missingness pattern while keeping the recorded values.

    For every block, each person whose block is blank receives the block values
    of a randomly chosen person who has it recorded (sampling with replacement,
    seeded). Afterwards no all-zero dummy pattern or NaN marks a blank, so a
    model fitted on the value columns cannot read 'which fields were recorded'.
    The step uses no labels. Because blank people receive unrelated values, the
    gain of this regime is a lower bound on the signal carried by the values.
    """
    out = df.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    for flag, vals in missingness_blocks(columns):
        blank = np.flatnonzero(out[flag].values == 1)
        known = np.flatnonzero(out[flag].values == 0)
        if len(blank) == 0 or len(known) == 0:
            continue
        donors = rng.choice(known, size=len(blank), replace=True)
        for v in vals:
            col = out[v].values.copy()
            col[blank] = col[donors]
            out[v] = col
    return out


def make_model(model_name, balanced=False, n_train=None):
    """HistGB / RF / LogReg with fixed settings (no tuning anywhere in the study)."""
    if model_name == "HistGB":
        early = n_train is None or n_train >= SMALL_N_NO_EARLY_STOP
        return HistGradientBoostingClassifier(
            max_iter=50, early_stopping=early, random_state=config.SEED,
            class_weight="balanced" if balanced else None,
        )
    if model_name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                           random_state=config.SEED, n_jobs=1)),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=config.SEED)),
    ])


def repeated_oof_probs(X, y, model_name, balanced=False, n_repeats=None):
    """Stratified 5-fold CV repeated n_repeats times.

    Returns (mean OOF probabilities, per-repeat OOF probabilities with shape
    (n_repeats, n, n_classes)). Imputation/scaling live inside the model
    pipelines, so they are fitted on training folds only."""
    n_repeats = config.N_REPEATS if n_repeats is None else n_repeats
    n_classes = len(np.unique(y))
    min_class = int(np.bincount(y).min())
    n_splits = 5 if min_class >= 5 else max(2, min_class)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=config.SEED)
    per_rep = np.zeros((n_repeats, len(y), n_classes))
    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        rep = i // n_splits
        X_tr = X.iloc[train_idx]
        valid_cols = [c for c in X_tr.columns if X_tr[c].notna().any()]
        model = make_model(model_name, balanced=balanced, n_train=len(train_idx))
        model.fit(X_tr[valid_cols], y[train_idx])
        per_rep[rep, test_idx] = model.predict_proba(X.iloc[test_idx][valid_cols])
    return per_rep.mean(axis=0), per_rep


def fast_ovr_auc(y_true, probs):
    """Tie-aware rank-based macro one-vs-rest AUC (identical to sklearn's).
    For 2 classes this equals the ordinary binary AUC of class 1."""
    n_classes = probs.shape[1]
    class_list = [1] if n_classes == 2 else range(n_classes)
    aucs = []
    for c in class_list:
        pos = (y_true == c)
        n_pos = int(pos.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(probs[:, c], method="average")
        aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(aucs))


def point_metrics(y_true, probs):
    """Macro AUC, macro F1, balanced accuracy and multi-class Brier score
    (sum of squared errors over classes, averaged over people; range 0-2)."""
    n_classes = probs.shape[1]
    y_pred = np.argmax(probs, axis=1)
    f1s, recalls = [], []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        f1s.append((2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    return {"macro_auc": fast_ovr_auc(y_true, probs), "macro_f1": float(np.mean(f1s)),
            "bal_acc": float(np.mean(recalls)), "brier": brier}


METRICS = ["macro_auc", "macro_f1", "bal_acc", "brier"]


def bootstrap_compare(y_true, probs_by_set, reference=None, pairs=None, n_boot=None,
                      per_repeat=None):
    """Bootstrap 95% CIs (people resampled, same draws for every set).

    Returns (metric_rows, gain_rows). metric_rows: one per set with every
    metric + CI, the repeat-to-repeat SD of macro AUC when `per_repeat` is
    given, and macro-AUC gain vs `reference`. gain_rows: one per (label, b, a)
    pair with the paired gain and CI for every metric. Note: the CIs describe
    sampling uncertainty over people given the averaged OOF predictions; the
    repeat SD describes CV-split variability."""
    n_boot = config.N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(config.SEED)
    n = len(y_true)
    n_classes = next(iter(probs_by_set.values())).shape[1]
    draws = {s: {m: [] for m in METRICS} for s in probs_by_set}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < n_classes:
            continue
        for s, p in probs_by_set.items():
            vals = point_metrics(y_true[idx], p[idx])
            for m in METRICS:
                draws[s][m].append(vals[m])
    draws = {s: {m: np.array(v) for m, v in d.items()} for s, d in draws.items()}

    metric_rows = []
    for s, p in probs_by_set.items():
        pt = point_metrics(y_true, p)
        row = {"feature_set": s, "n": n}
        for m in METRICS:
            row[m] = round(pt[m], 3)
            row[m + "_ci_low"] = round(float(np.percentile(draws[s][m], 2.5)), 3)
            row[m + "_ci_high"] = round(float(np.percentile(draws[s][m], 97.5)), 3)
        if per_repeat is not None and s in per_repeat:
            rep_aucs = [fast_ovr_auc(y_true, per_repeat[s][r]) for r in range(per_repeat[s].shape[0])]
            row["auc_repeat_sd"] = round(float(np.std(rep_aucs, ddof=1)), 4) if len(rep_aucs) > 1 else np.nan
        if reference is not None:
            ref_pt = point_metrics(y_true, probs_by_set[reference])
            diff = draws[s]["macro_auc"] - draws[reference]["macro_auc"]
            row["auc_gain_vs_ref"] = round(pt["macro_auc"] - ref_pt["macro_auc"], 3)
            row["gain_ci_low"] = round(float(np.percentile(diff, 2.5)), 3)
            row["gain_ci_high"] = round(float(np.percentile(diff, 97.5)), 3)
        metric_rows.append(row)

    if pairs is None and reference is not None:
        pairs = [(f"{s}_minus_{reference}", s, reference) for s in probs_by_set if s != reference]
    gain_rows = []
    for label, set_b, set_a in (pairs or []):
        pt_b = point_metrics(y_true, probs_by_set[set_b])
        pt_a = point_metrics(y_true, probs_by_set[set_a])
        row = {"comparison": label, "n": n}
        for m in METRICS:
            diff = draws[set_b][m] - draws[set_a][m]
            row[m + "_gain"] = round(pt_b[m] - pt_a[m], 3)
            row[m + "_ci_low"] = round(float(np.percentile(diff, 2.5)), 3)
            row[m + "_ci_high"] = round(float(np.percentile(diff, 97.5)), 3)
        gain_rows.append(row)
    return metric_rows, gain_rows


def run_sets(df, label_col, class_names, sets, model_name="HistGB",
             reference="M0_base", balanced=False, pairs=None, n_boot=None):
    """Fit every feature set with repeated CV; return a dict with
    metrics rows, gain rows, OOF probabilities, per-repeat probabilities, y."""
    class_to_int = {c: i for i, c in enumerate(class_names)}
    y = df[label_col].map(class_to_int).values
    probs, per_rep = {}, {}
    for name, cols in sets.items():
        probs[name], per_rep[name] = repeated_oof_probs(df[cols], y, model_name, balanced=balanced)
    metric_rows, gain_rows = bootstrap_compare(y, probs, reference=reference, pairs=pairs,
                                               n_boot=n_boot, per_repeat=per_rep)
    for r in metric_rows + gain_rows:
        r["model"] = model_name
    return {"metrics": metric_rows, "gains": gain_rows, "probs": probs,
            "per_repeat": per_rep, "y": y}
