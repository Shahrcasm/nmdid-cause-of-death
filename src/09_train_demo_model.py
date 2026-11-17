"""Step 9 - Train the demo models used by the Streamlit app (app/app.py).

Fits on ALL eligible people (no CV; the CV estimates of performance are in
results/tables). Saves to models/ (git-ignored because the models are derived
from restricted NMDID data).

  models/demo_models.joblib -> dict with:
      level1_M0        gradient boosting, age + sex
      level1_M2        gradient boosting, all inputs (standard model)
      level1_withheld  gradient boosting trained with randomly withheld sections
      level2_M2        intent within injury deaths
      column lists, sections, class names, training sizes, shortlist thresholds

Also exports, for the deployed app (no restricted model file online), aggregate
weights of three logistic regressions (age + sex, all inputs, and the
withholding-trained version) plus the intent model, and their 90/95/99% shortlist
thresholds computed from held-out predictions. These files hold summary numbers only.
Run after 06_reliability.py (it reads the shortlist thresholds of the local models).
"""
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import config
import model_utils as mu

if __name__ == "__main__":
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)
    cols_m0, cols_m2 = nested["M0_base"], nested["M2_med_history"]
    l1, y1 = mu.level1_frame(cohort, features)
    l2, y2 = mu.level2_frame(cohort, features)
    sec = mu.Sections(cols_m2)

    thresholds, band_thresholds = {}, {}
    conf_path = config.TABLES_DIR / "reliability_conformal.csv"
    if conf_path.exists():
        conf = pd.read_csv(conf_path)
        marg = conf[(conf["group_type"] == "all people") & (conf["method"] == "marginal")]
        for _, r in marg.iterrows():
            thresholds.setdefault(r["model"], {})[int(r["target_coverage_pct"])] = float(r["threshold_all_people"])
        band = conf[(conf["group_type"] == "sections not recorded") & (conf["method"] == "missingness-conditional")]
        for _, r in band.iterrows():
            band_thresholds.setdefault(r["model"], {}).setdefault(int(r["target_coverage_pct"]), {})[r["group"]] = float(r["threshold_all_people"])
    # Possible answers for the "which question next" ranking: 30 records per cause
    # (person-level, so they stay in the git-ignored models/ folder).
    rng = np.random.default_rng(config.SEED)
    donors = {c: l1[cols_m2].iloc[rng.choice(np.flatnonzero(y1 == i), 30, replace=False)].reset_index(drop=True)
              for i, c in enumerate(mu.LEVEL1_CLASSES)}

    bundle = {
        "level1_M0": mu.make_model("HistGB", n_train=len(l1)).fit(l1[cols_m0], y1),
        "level1_M2": mu.make_model("HistGB", n_train=len(l1)).fit(l1[cols_m2], y1),
        "level1_withheld": sec.fit_withholding_model(l1[cols_m2], y1, seed=config.SEED),
        "level2_M2": mu.make_model("HistGB", n_train=len(l2)).fit(l2[cols_m2], y2),
        "columns_M0": cols_m0,
        "columns_M2": cols_m2,
        "columns_withheld": sec.all_columns,
        "sections": mu.SECTIONS,
        "section_labels": mu.SECTION_LABELS,
        "level1_classes": mu.LEVEL1_CLASSES,
        "level2_classes": mu.LEVEL2_CLASSES,
        "n_level1": len(l1),
        "n_level2": len(l2),
        "shortlist_thresholds": thresholds,
        "shortlist_thresholds_band": band_thresholds,
        "bands": [("none", 0, 0), ("1 to 3", 1, 3), ("4 or 5", 4, 5), ("6", 6, 6), ("7 or 8", 7, 8)],
        "section_columns": {name: sec.section_columns(name) for name in sec.names},
        "donors": donors,
        "median_cadaver_bmi": float(l1["cadaver_bmi"].median()),
        "median_living_bmi": float(l1["living_bmi"].median()),
    }
    out_dir = config.PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, out_dir / "demo_models.joblib")

    # Aggregate logistic-regression weights for the GitHub-only fallback (standard models only).
    def pack_logreg(pipe, cols, name, classes):
        imp, scl, clf = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["clf"]
        rows = []
        for idx, col in enumerate(cols):
            r = {"model_id": name, "feature": col, "impute_median": float(imp.statistics_[idx]),
                 "scale_mean": float(scl.mean_[idx]), "scale_std": float(scl.scale_[idx])}
            for c_i, c_name in enumerate(classes):
                r[f"coef_{c_name}"] = float(clf.coef_[c_i, idx])
                r[f"intercept_{c_name}"] = float(clf.intercept_[c_i])
            rows.append(r)
        return rows

    w_rows = (pack_logreg(mu.make_model("LogReg").fit(l1[cols_m0], y1), cols_m0, "level1_M0", mu.LEVEL1_CLASSES)
              + pack_logreg(mu.make_model("LogReg").fit(l1[cols_m2], y1), cols_m2, "level1_M2", mu.LEVEL1_CLASSES)
              + pack_logreg(mu.make_model("LogReg").fit(l2[cols_m2], y2), cols_m2, "level2_M2", mu.LEVEL2_CLASSES))
    # Withholding-trained logistic regression for the deployed app.
    Xa, ya = sec.augment(l1[cols_m2], y1, seed=config.SEED)
    w_rows += pack_logreg(mu.make_model("LogReg").fit(Xa, ya), sec.all_columns, "level1_withheld", mu.LEVEL1_CLASSES)
    pd.DataFrame(w_rows).to_csv(config.TABLES_DIR / "demo_logreg_weights.csv", index=False)

    # Shortlist thresholds of the deployed logistic regressions, from held-out predictions (5 x 5 CV).
    def oof_logreg(withheld):
        splits, R = mu.cv_splits(l1[cols_m2], y1)
        P = np.zeros((R, len(y1), 3))
        for f, (rep, tr, te) in enumerate(splits):
            X_tr, X_te = l1.iloc[tr][cols_m2], l1.iloc[te][cols_m2]
            if withheld:
                Xa_f, ya_f = sec.augment(X_tr, y1[tr], seed=config.SEED + f)
                model = mu.make_model("LogReg").fit(Xa_f, ya_f)
                P[rep, te] = model.predict_proba(sec.as_withheld(X_te, sec.mask(len(te), [])))
            else:
                P[rep, te] = mu.make_model("LogReg").fit(X_tr, y1[tr]).predict_proba(X_te)
        return P.mean(axis=0)

    def quantile_threshold(scores, alpha):
        sc = np.sort(scores)
        k = int(np.ceil((len(sc) + 1) * (1 - alpha)))
        return float(sc[k - 1]) if k <= len(sc) else 1.0

    bands = [("none", 0, 0), ("1 to 3", 1, 3), ("4 or 5", 4, 5), ("6", 6, 6), ("7 or 8", 7, 8)]
    n_blank = sec.not_recorded(l1).sum(axis=1)
    band = np.array([next(b for b, lo, hi in bands if lo <= v <= hi) for v in n_blank], dtype=object)
    t_rows = []
    for model_id, withheld in [("standard", False), ("withheld", True)]:
        p = oof_logreg(withheld)
        scores = 1 - p[np.arange(len(y1)), y1]
        for target in (90, 95, 99):
            alpha = 1 - target / 100
            t_rows.append({"model": model_id, "method": "marginal", "band": "all", "target_coverage_pct": target,
                           "threshold": round(quantile_threshold(scores, alpha), 4), "n_calibration": len(scores),
                           "heldout_accuracy": round(mu.accuracy(y1, p), 3)})
            for b, _, _ in bands:
                m = band == b
                t_rows.append({"model": model_id, "method": "missingness-conditional", "band": b, "target_coverage_pct": target,
                               "threshold": round(quantile_threshold(scores[m], alpha), 4), "n_calibration": int(m.sum()),
                               "heldout_accuracy": round(mu.accuracy(y1, p), 3)})
    pd.DataFrame(t_rows).to_csv(config.TABLES_DIR / "demo_shortlist_thresholds.csv", index=False)

    print("Saved models/demo_models.joblib, results/tables/demo_logreg_weights.csv and demo_shortlist_thresholds.csv")
    print("Level 1 training n:", len(l1), "| Level 2 training n:", len(l2), "| M2 inputs:", len(cols_m2))
    print("Shortlist thresholds:", thresholds if thresholds else "not found (run 06_reliability.py first)")
