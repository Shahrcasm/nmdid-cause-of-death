"""Shared helpers for scripts 03-09 and the demo app.

One implementation of:
  - cohort frames for the three outcome levels and the nested input sets M0-M2;
  - the eight sections of a record and the ways of handling a section that is
    not available to the model (class `Sections`);
  - model constructors with fixed settings, plus a small nested search;
  - 5-fold stratified CV repeated 5 times (seed 42) with out-of-fold
    probabilities averaged per person;
  - metrics (tie-aware macro one-vs-rest AUC, accuracy, balanced accuracy,
    macro F1, Brier) and paired bootstrap confidence intervals.
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize_scalar
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import config

LEVEL1_CLASSES = ["natural_cardiac", "drug_alcohol", "injury"]
LEVEL2_CLASSES = ["accident", "homicide", "suicide"]
LEVEL3_CLASSES = ["stab", "gunshot"]
CLASS_LABELS = {"natural_cardiac": "Natural cardiac", "drug_alcohol": "Drug / alcohol", "injury": "Injury"}

# Below this training-fold size HistGB's internal 10% validation split is too
# small for early stopping to be reliable (it stopped at iteration 1 on Level 3).
SMALL_N_NO_EARLY_STOP = 200

CONTINUOUS_COLS = ["cadaver_bmi", "living_bmi", "edu_level", "ses_adult", "ses_child"]
MEDHIST_BLOCKS = ["broken_bone", "facial_trauma", "surgery", "implant"]

# A record has eight sections. A section is one or more blocks; a block is a
# `_missing` flag (1 = the investigator recorded nothing) plus its value columns.
SECTIONS = {
    "drinking": ["drink_missing"],
    "tobacco": ["tob_missing"],
    "drug_use": ["drug_missing"],
    "marital_status": ["marital_missing"],
    "body_size": ["cadaver_bmi_missing", "living_bmi_missing"],
    "social_background": ["edu_level_missing", "ses_adult_missing", "ses_child_missing",
                          "diet_missing", "lift_missing", "occ_missing"],
    "diagnoses": ["dx_missing"],
    "injury_surgery_history": ["medhist_broken_bone_missing", "medhist_facial_trauma_missing",
                               "medhist_surgery_missing", "medhist_implant_missing"],
}
SECTION_LABELS = {
    "drinking": "Drinking", "tobacco": "Tobacco", "drug_use": "Drug use",
    "marital_status": "Marital status", "body_size": "Body size (BMI)",
    "social_background": "Education, work, diet", "diagnoses": "Diagnoses",
    "injury_surgery_history": "Past injuries and surgery",
}
LIFESTYLE_SECTIONS = ["drinking", "tobacco", "drug_use", "marital_status", "body_size", "social_background"]
MEDICAL_SECTIONS = ["diagnoses", "injury_surgery_history"]

# Withholding-trained model: besides the records as they are, the training fold
# contains one copy per rate in which every section is hidden with that probability.
WITHHOLD_RATES = (0.15, 0.3, 0.5, 0.75, 1.0)

# Multiple hot-deck imputation (comparison handling for hidden sections): each
# person receives the hidden sections of donors drawn from the most similar
# training people on the available inputs.
N_IMPUTATIONS = 3
N_NEIGHBOURS = 10

# Small nested search used only in the accuracy-ceiling analysis.
TUNING_GRID = [{"max_leaf_nodes": k} for k in (4, 7, 15, 31)]


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data():
    """Load cohort.csv and features.csv with the join key as text."""
    cohort = pd.read_csv(config.PROCESSED_DIR / "cohort.csv", dtype=str)
    cohort["age_years"] = pd.to_numeric(cohort["age_years"], errors="coerce")
    features = pd.read_csv(config.PROCESSED_DIR / "features.csv")
    features[config.KEY] = features[config.KEY].astype(str)
    return cohort, features


def encode(labels, classes):
    return pd.Series(labels).map({c: i for i, c in enumerate(classes)}).values


def level1_frame(cohort, features, extra=()):
    """Level 1 (primary outcome): natural cardiac / drug-alcohol / injury."""
    cols = [config.KEY, "level1"] + list(extra)
    df = cohort[cohort["level1"].isin(LEVEL1_CLASSES)][cols].merge(features, on=config.KEY).reset_index(drop=True)
    return df, encode(df["level1"], LEVEL1_CLASSES)


def level2_frame(cohort, features):
    """Level 2 (secondary outcome): intent within injury deaths."""
    keep = (cohort["level1"] == "injury") & cohort["intent"].isin(LEVEL2_CLASSES)
    df = cohort[keep][[config.KEY, "intent"]].merge(features, on=config.KEY).reset_index(drop=True)
    return df, encode(df["intent"], LEVEL2_CLASSES)


def level3_frame(cohort, features):
    """Level 3 (negative control): stab versus gunshot within homicides."""
    stab, gun = "Stab wound (slash, penetrating cut)", "Gunshot wound of"
    keep = ((cohort["level1"] == "injury") & (cohort["intent"] == "homicide") &
            cohort["primary_cause_of_death"].isin([stab, gun]))
    df = cohort[keep].copy()
    df["weapon"] = np.where(df["primary_cause_of_death"] == gun, "gunshot", "stab")
    df = df[[config.KEY, "weapon"]].merge(features, on=config.KEY).reset_index(drop=True)
    return df, encode(df["weapon"], LEVEL3_CLASSES)


def get_column_groups(features):
    """Return base, lifestyle and medical-history column lists."""
    med_hist = [c for c in features.columns if c.startswith("dx_") or c.startswith("medhist_")]
    base = ["age_years", "male"]
    excluded = set([config.KEY] + base + med_hist + config.NON_MODEL_COLUMNS)
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
    """Destroy the pattern of blanks while keeping the recorded values.

    For every block, each person whose block is blank receives the block values
    of a randomly chosen person who has it recorded (sampling with replacement,
    seeded). Afterwards no all-zero dummy pattern or NaN marks a blank, so a
    model fitted on the value columns cannot read which fields were recorded.
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


# ----------------------------------------------------------------------------
# Sections of the record and unavailable information
# ----------------------------------------------------------------------------
class Sections:
    """Positions of the eight record sections inside one column list, and the
    three ways of presenting a section that is not available to a model.

    `mask` is always a boolean array (people x sections): True = hide it."""

    def __init__(self, columns):
        self.columns = list(columns)
        pos = {c: i for i, c in enumerate(self.columns)}
        by_flag = dict(missingness_blocks(self.columns))
        listed = [f for flags in SECTIONS.values() for f in flags]
        assert sorted(listed) == sorted(by_flag), "every block must belong to exactly one section"
        self.names = list(SECTIONS)
        self.blocks, self.flag_idx, self.cat_idx, self.num_idx = [], [], [], []
        for s in self.names:
            vals = [v for f in SECTIONS[s] for v in by_flag[f]]
            self.flag_idx.append(np.array([pos[f] for f in SECTIONS[s]]))
            self.cat_idx.append(np.array([pos[v] for v in vals if v not in CONTINUOUS_COLS], dtype=int))
            self.num_idx.append(np.array([pos[v] for v in vals if v in CONTINUOUS_COLS], dtype=int))
            self.blocks += [(pos[f], [pos[v] for v in by_flag[f]]) for f in SECTIONS[s]]
        self.withheld_cols = ["withheld_" + s for s in self.names]
        self.all_columns = self.columns + self.withheld_cols

    def section_columns(self, name):
        """Flag and value column names of one section (used for reduced models)."""
        j = self.names.index(name)
        idx = np.concatenate([self.flag_idx[j], self.cat_idx[j], self.num_idx[j]])
        return [self.columns[i] for i in idx]

    def mask(self, n, hidden):
        """Same sections hidden for all n people."""
        row = np.array([s in hidden for s in self.names])
        return np.repeat(row[None, :], n, axis=0)

    def random_mask(self, n, k, rng):
        """Exactly k randomly chosen sections hidden per person."""
        order = np.argsort(rng.random((n, len(self.names))), axis=1)
        m = np.zeros((n, len(self.names)), dtype=bool)
        np.put_along_axis(m, order[:, :k], True, axis=1)
        return m

    def not_recorded(self, X):
        """Natural state: True where the investigator recorded nothing in the section."""
        A = X[self.columns].values
        return np.column_stack([(A[:, fi] == 1).all(axis=1) for fi in self.flag_idx])

    def _blank(self, X, mask, flag_value):
        A = X[self.columns].values.astype(float)
        for j in range(len(self.names)):
            rows = np.flatnonzero(mask[:, j])
            if len(rows) == 0:
                continue
            A[np.ix_(rows, self.flag_idx[j])] = flag_value
            if len(self.cat_idx[j]):
                A[np.ix_(rows, self.cat_idx[j])] = 0.0
            if len(self.num_idx[j]):
                A[np.ix_(rows, self.num_idx[j])] = np.nan
        return A

    def as_not_recorded(self, X, mask):
        """Handling 1 (naive): a hidden section looks exactly like one the
        investigator left blank (flag = 1, values empty)."""
        return pd.DataFrame(self._blank(X, mask, 1.0), columns=self.columns)

    def as_imputed(self, X, mask, typical):
        """Handling 2 (imputation): hidden values are replaced by the typical
        value of the training fold and the block is presented as recorded."""
        A = X[self.columns].values.astype(float)
        for j in range(len(self.names)):
            rows = np.flatnonzero(mask[:, j])
            if len(rows) == 0:
                continue
            A[np.ix_(rows, self.flag_idx[j])] = 0.0
            idx = np.concatenate([self.cat_idx[j], self.num_idx[j]])
            A[np.ix_(rows, idx)] = typical[idx]
        return pd.DataFrame(A, columns=self.columns)

    def typical_values(self, X_train):
        """Per value column: prevalence (median for continuous columns) among
        training people whose block was recorded."""
        A = X_train[self.columns].values.astype(float)
        typical = np.zeros(A.shape[1])
        for flag, vals in self.blocks:
            rec = A[:, flag] == 0
            for v in vals:
                col = A[rec, v]
                col = col[~np.isnan(col)]
                if len(col):
                    typical[v] = np.median(col) if self.columns[v] in CONTINUOUS_COLS else col.mean()
        return typical

    def as_withheld(self, X, mask):
        """Handling 3 (two channels): the values AND the recorded / not-recorded
        flag of a hidden section are removed, and a separate `withheld_<section>`
        indicator says the section was not available. 'Not recorded by the
        investigator' and 'not available to the model' stay different states."""
        out = pd.DataFrame(self._blank(X, mask, 0.0), columns=self.columns)
        for j, w in enumerate(self.withheld_cols):
            out[w] = mask[:, j].astype(float)
        return out

    def augment(self, X, y, seed, rates=WITHHOLD_RATES):
        """Training data of the withholding-trained model: the records as they
        are, plus one copy per rate with sections hidden at random."""
        rng = np.random.default_rng(seed)
        n, k = len(X), len(self.names)
        parts = [self.as_withheld(X, np.zeros((n, k), dtype=bool))]
        for r in rates:
            parts.append(self.as_withheld(X, rng.random((n, k)) < r))
        return pd.concat(parts, ignore_index=True), np.tile(y, len(rates) + 1)

    def fit_withholding_model(self, X_train, y_train, seed, model_name="HistGB"):
        Xa, ya = self.augment(X_train, y_train, seed)
        return make_model(model_name, n_train=len(Xa)).fit(Xa, ya)

    def fit_imputers(self, X_train, seed, m=N_IMPUTATIONS):
        """Donor pool for multiple hot-deck imputation: the training fold's records
        plus the column means and standard deviations used for distances."""
        A = X_train[self.columns].values.astype(float)
        mean = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return {"A": A, "Z": np.nan_to_num((A - mean) / sd), "mean": mean, "sd": sd, "seed": seed, "m": m}

    def predict_multiply_imputed(self, model, X, mask, pool, k=N_NEIGHBOURS):
        """Handling 4 (multiple imputation): for every person, m donors are drawn at
        random from the k training people closest on the inputs that are still
        available (standardised Euclidean distance). The donors' values for the
        hidden sections (flags and values) are copied in, so every imputed record
        is a combination that occurs in real data. The standard model's
        probabilities are averaged over the m imputations."""
        A = X[self.columns].values.astype(float)
        H = np.zeros(A.shape, dtype=bool)
        for j in range(len(self.names)):
            rows = np.flatnonzero(mask[:, j])
            if len(rows):
                cols = np.concatenate([self.flag_idx[j], self.cat_idx[j], self.num_idx[j]]).astype(int)
                H[np.ix_(rows, cols)] = True
        if not H.any():
            return model.predict_proba(pd.DataFrame(A, columns=self.columns))
        Zt = np.nan_to_num((A - pool["mean"]) / pool["sd"])
        rng = np.random.default_rng(pool["seed"])
        donors = np.zeros((len(A), pool["m"]), dtype=int)
        for i in range(len(A)):
            d = (((pool["Z"] - Zt[i]) ** 2) * (~H[i])).sum(axis=1)
            nearest = np.argpartition(d, k)[:k]
            donors[i] = rng.choice(nearest, pool["m"], replace=False)
        out = []
        for r in range(pool["m"]):
            Zr = A.copy()
            Zr[H] = pool["A"][donors[:, r]][H]
            out.append(model.predict_proba(pd.DataFrame(Zr, columns=self.columns)))
        return np.mean(out, axis=0)

    def section_positions(self, j):
        """Positions of section j's flag and value columns (the same in the plain and
        the withheld column lists) and of its withheld indicator."""
        idx = np.concatenate([self.flag_idx[j], self.cat_idx[j], self.num_idx[j]]).astype(int)
        return idx, len(self.columns) + j


def entropy(p):
    """Uncertainty of a probability vector (natural log); 0 = certain."""
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=-1)


def apply_temperature(p, T):
    """Temperature scaling: soften (T > 1) or sharpen (T < 1) probabilities.
    The most probable cause never changes, so accuracy is unchanged."""
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(p, y):
    """Temperature that minimises the log-loss of held-out probabilities."""
    def nll(T):
        q = apply_temperature(p, T)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0)).mean()
    return float(minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded").x)


# ----------------------------------------------------------------------------
# Models and cross-validation
# ----------------------------------------------------------------------------
def make_model(model_name, balanced=False, n_train=None, params=None):
    """HistGB / RF / LogReg with fixed settings. `params` overrides HistGB
    settings and is used only by the nested search of the ceiling analysis."""
    if model_name == "HistGB":
        early = n_train is None or n_train >= SMALL_N_NO_EARLY_STOP
        return HistGradientBoostingClassifier(
            max_iter=50, early_stopping=early, random_state=config.SEED,
            class_weight="balanced" if balanced else None, **(params or {}),
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


def cv_splits(X, y, n_repeats=None):
    """The one set of folds used everywhere: (repeat index, train idx, test idx)."""
    n_repeats = config.N_REPEATS if n_repeats is None else n_repeats
    min_class = int(np.bincount(y).min())
    n_splits = 5 if min_class >= 5 else max(2, min_class)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=config.SEED)
    return [(i // n_splits, tr, te) for i, (tr, te) in enumerate(rskf.split(X, y))], n_repeats


def repeated_oof_probs(X, y, model_name, balanced=False, n_repeats=None):
    """Stratified 5-fold CV repeated n_repeats times.

    Returns (mean OOF probabilities, per-repeat OOF probabilities with shape
    (n_repeats, n, n_classes)). Imputation/scaling live inside the model
    pipelines, so they are fitted on training folds only."""
    splits, n_repeats = cv_splits(X, y, n_repeats)
    per_rep = np.zeros((n_repeats, len(y), len(np.unique(y))))
    for rep, train_idx, test_idx in splits:
        X_tr = X.iloc[train_idx]
        valid_cols = [c for c in X_tr.columns if X_tr[c].notna().any()]
        model = make_model(model_name, balanced=balanced, n_train=len(train_idx))
        model.fit(X_tr[valid_cols], y[train_idx])
        per_rep[rep, test_idx] = model.predict_proba(X.iloc[test_idx][valid_cols])
    return per_rep.mean(axis=0), per_rep


def nested_tuned_oof_probs(X, y, n_repeats=None):
    """HistGB whose tree size is chosen inside every outer training fold by an
    inner 3-fold search (log-loss). The outer test fold is never used for the
    choice, so the estimate carries no selection optimism.
    Returns (mean OOF probabilities, per-repeat probabilities, chosen settings)."""
    splits, n_repeats = cv_splits(X, y, n_repeats)
    per_rep = np.zeros((n_repeats, len(y), len(np.unique(y))))
    chosen = []
    for rep, train_idx, test_idx in splits:
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        valid_cols = [c for c in X_tr.columns if X_tr[c].notna().any()]
        losses = np.zeros(len(TUNING_GRID))
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
        for i_tr, i_te in inner.split(X_tr, y_tr):
            for g, params in enumerate(TUNING_GRID):
                m = make_model("HistGB", n_train=len(i_tr), params=params)
                m.fit(X_tr.iloc[i_tr][valid_cols], y_tr[i_tr])
                p = np.clip(m.predict_proba(X_tr.iloc[i_te][valid_cols]), 1e-9, 1.0)
                losses[g] += -np.log(p[np.arange(len(i_te)), y_tr[i_te]]).sum()
        best = TUNING_GRID[int(np.argmin(losses))]
        chosen.append(best["max_leaf_nodes"])
        model = make_model("HistGB", n_train=len(train_idx), params=best).fit(X_tr[valid_cols], y_tr)
        per_rep[rep, test_idx] = model.predict_proba(X.iloc[test_idx][valid_cols])
    return per_rep.mean(axis=0), per_rep, chosen


# ----------------------------------------------------------------------------
# Metrics and bootstrap
# ----------------------------------------------------------------------------
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


def pairwise_auc(y_true, probs, a, b):
    """AUC for telling class a from class b among people in one of the two."""
    keep = (y_true == a) | (y_true == b)
    score = probs[keep, a] / np.clip(probs[keep, a] + probs[keep, b], 1e-12, None)
    pos = y_true[keep] == a
    ranks = rankdata(score, method="average")
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy(y_true, probs):
    return float(np.mean(np.argmax(probs, axis=1) == y_true))


def top2_accuracy(y_true, probs):
    """Share of people whose true class is one of the two most probable."""
    order = np.argsort(-probs, axis=1)[:, :2]
    return float((order == y_true[:, None]).any(axis=1).mean())


def point_metrics(y_true, probs):
    """Macro AUC, accuracy, balanced accuracy, macro F1 and multi-class Brier
    score (sum of squared errors over classes, averaged over people; range 0-2)."""
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
    return {"macro_auc": fast_ovr_auc(y_true, probs), "accuracy": float(np.mean(y_pred == y_true)),
            "bal_acc": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)), "brier": brier}


METRICS = ["macro_auc", "accuracy", "bal_acc", "macro_f1", "brier"]


def bootstrap_compare(y_true, probs_by_set, reference=None, pairs=None, n_boot=None,
                      per_repeat=None):
    """Bootstrap 95% CIs (people resampled, same draws for every set).

    Returns (metric_rows, gain_rows). metric_rows: one per set with every
    metric + CI, the repeat-to-repeat SD of macro AUC when `per_repeat` is
    given, and the macro-AUC and accuracy gains vs `reference`. gain_rows: one
    per (label, b, a) pair with the paired gain and CI for every metric.
    The CIs describe sampling uncertainty over people given the averaged OOF
    predictions; the repeat SD describes CV-split variability."""
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
    points = {s: point_metrics(y_true, p) for s, p in probs_by_set.items()}

    def ci(values):
        return round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)

    metric_rows = []
    for s in probs_by_set:
        row = {"feature_set": s, "n": n}
        for m in METRICS:
            row[m] = round(points[s][m], 3)
            row[m + "_ci_low"], row[m + "_ci_high"] = ci(draws[s][m])
        if per_repeat is not None and s in per_repeat:
            rep_aucs = [fast_ovr_auc(y_true, per_repeat[s][r]) for r in range(per_repeat[s].shape[0])]
            row["auc_repeat_sd"] = round(float(np.std(rep_aucs, ddof=1)), 4) if len(rep_aucs) > 1 else np.nan
        if reference is not None:
            row["auc_gain_vs_ref"] = round(points[s]["macro_auc"] - points[reference]["macro_auc"], 3)
            row["gain_ci_low"], row["gain_ci_high"] = ci(draws[s]["macro_auc"] - draws[reference]["macro_auc"])
            row["acc_gain_vs_ref"] = round(points[s]["accuracy"] - points[reference]["accuracy"], 3)
            row["acc_gain_ci_low"], row["acc_gain_ci_high"] = ci(draws[s]["accuracy"] - draws[reference]["accuracy"])
        metric_rows.append(row)

    if pairs is None and reference is not None:
        pairs = [(f"{s}_minus_{reference}", s, reference) for s in probs_by_set if s != reference]
    gain_rows = []
    for label, set_b, set_a in (pairs or []):
        row = {"comparison": label, "n": n}
        for m in METRICS:
            row[m + "_gain"] = round(points[set_b][m] - points[set_a][m], 3)
            row[m + "_ci_low"], row[m + "_ci_high"] = ci(draws[set_b][m] - draws[set_a][m])
        gain_rows.append(row)
    return metric_rows, gain_rows


def run_sets(df, label_col, class_names, sets, model_name="HistGB",
             reference="M0_base", balanced=False, pairs=None, n_boot=None):
    """Fit every input set with repeated CV; return a dict with metric rows,
    gain rows, OOF probabilities, per-repeat probabilities and y."""
    y = encode(df[label_col], class_names)
    probs, per_rep = {}, {}
    for name, cols in sets.items():
        probs[name], per_rep[name] = repeated_oof_probs(df[cols], y, model_name, balanced=balanced)
    metric_rows, gain_rows = bootstrap_compare(y, probs, reference=reference, pairs=pairs,
                                               n_boot=n_boot, per_repeat=per_rep)
    for r in metric_rows + gain_rows:
        r["model"] = model_name
    return {"metrics": metric_rows, "gains": gain_rows, "probs": probs,
            "per_repeat": per_rep, "y": y}


def tag(rows, **info):
    for r in rows:
        r.update(info)
    return rows
