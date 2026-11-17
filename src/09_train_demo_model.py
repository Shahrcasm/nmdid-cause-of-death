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

Also exports aggregate logistic-regression weights to results/tables so that a
GitHub-only deployment of the app works without the restricted model file.
Run after 06_reliability.py (it reads the shortlist thresholds).
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
    pd.DataFrame(w_rows).to_csv(config.TABLES_DIR / "demo_logreg_weights.csv", index=False)

    print("Saved models/demo_models.joblib and results/tables/demo_logreg_weights.csv")
    print("Level 1 training n:", len(l1), "| Level 2 training n:", len(l2), "| M2 inputs:", len(cols_m2))
    print("Shortlist thresholds:", thresholds if thresholds else "not found (run 06_reliability.py first)")
